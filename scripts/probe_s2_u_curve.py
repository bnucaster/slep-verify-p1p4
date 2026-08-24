"""S2 效用曲线探测（任务八，选项 4）：逐检查点导航成功率的形状。

目的：判断 S2 是否存在效用平台（S1 实测为峰后缓降、无平台），供 P1
平台前提的协议层决策（选项 2/3）参考。探测口径，非冻结判定：

- 每种子固定 80 个评估任务（迷宫 + 起点 + BFS 2–6 目标），跨检查点
  配对复用——差异只来自模型，配对设计压方差；
- 检查点取均匀段全部（每 500 步共 12 个）加早期 {1, 32, 256}；
- 定稿规划器（穷举视界 1 + ε）；成功率的种子池化曲线附二项标准误。

细粒度缓存：逐（种子 × 检查点）落 cache/。产物
results/description/s2_u_curve/<campaign>/：curves.csv、summary.json、
u_curve.png。

用法：.venv/Scripts/python.exe scripts/probe_s2_u_curve.py [--seed N]
（--seed 只算一个种子，供分块执行）
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.systems import s2_gridworld as gw
from slep.systems.s2_planner import ExhaustiveMPCPlanner, mpc_episode
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CFG_TRAIN = yaml.safe_load((REPO_ROOT / "configs" / "s2_train.yaml").read_text(encoding="utf-8"))
N_TASKS = 80
MAX_STEPS = 24
EARLY_STEPS = [1, 32, 256]
BFS_BAND = (2, 6)


def eval_steps() -> list[int]:
    uniform = list(range(CFG_TRAIN["checkpoint_every"], CFG_TRAIN["train_steps"] + 1,
                         CFG_TRAIN["checkpoint_every"]))
    return EARLY_STEPS + uniform


def build_tasks(seed: int):
    rng = np.random.default_rng(seed + 313_000)
    tasks = []
    while len(tasks) < N_TASKS:
        maze = gw.generate_maze(CFG_TRAIN["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or not BFS_BAND[0] <= d <= BFS_BAND[1]:
            continue
        tasks.append((maze, start, goal))
    return tasks


def load_model(seed: int, step: int) -> S2WorldModel:
    ck = torch.load(
        REPO_ROOT / "results" / "description" / "s2_train" / CFG_TRAIN["campaign"]
        / f"s{seed}" / "checkpoints" / f"ckpt_{step:06d}.pt", weights_only=True)
    model = S2WorldModel(CFG_TRAIN["obs_dim"], CFG_TRAIN["action_dim"], CFG_TRAIN["embed_dim"],
                         CFG_TRAIN["hidden_dim"], CFG_TRAIN["sigma_dec"],
                         CFG_TRAIN.get("goal_sigma_dec"))
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main() -> None:
    torch.set_num_threads(10)
    only_seed = None
    if "--seed" in sys.argv:
        only_seed = int(sys.argv[sys.argv.index("--seed") + 1])
    campaign = create_campaign_dir("description", "s2_u_curve", CFG_TRAIN["campaign"],
                                   {"n_tasks": N_TASKS, "max_steps": MAX_STEPS,
                                    "bfs_band": list(BFS_BAND), "steps": eval_steps()})
    cache_dir = campaign / "cache"
    cache_dir.mkdir(exist_ok=True)

    seeds = guard.family_seeds("development", purpose="s2-u-curve")
    if only_seed is not None:
        seeds = [only_seed]
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="s2-u-curve")
        tasks = build_tasks(seed)
        for step in eval_steps():
            cache = cache_dir / f"s{seed}_c{step:06d}.json"
            if cache.exists():
                continue
            t0 = time.time()
            model = load_model(seed, step)
            planner = ExhaustiveMPCPlanner(model, view=CFG_TRAIN["view"])
            gen = torch.Generator()
            gen.manual_seed(seed * 131 + step)
            succ = 0
            for maze, start, goal in tasks:
                env = gw.GridWorld(maze.copy(), start, CFG_TRAIN["view"], goal)
                succ += int(mpc_episode(model, env, planner, MAX_STEPS, gen)["success"])
            cache.write_text(json.dumps({"success": succ, "n": N_TASKS}), encoding="utf-8")
            print(f"s{seed} ckpt {step}: {succ}/{N_TASKS} ({time.time() - t0:.0f}s)", flush=True)

    # 汇总（存在全部缓存时）
    all_done = all((cache_dir / f"s{s}_c{c:06d}.json").exists()
                   for s in guard.family_seeds("development", purpose="s2-u-curve")
                   for c in eval_steps())
    if not all_done:
        print("尚有缓存缺口，汇总跳过")
        return
    dev_seeds = guard.family_seeds("development", purpose="s2-u-curve")
    rows = []
    for step in eval_steps():
        per = [json.loads((cache_dir / f"s{s}_c{step:06d}.json").read_text())["success"] / N_TASKS
               for s in dev_seeds]
        pooled = float(np.mean(per))
        n_pool = N_TASKS * len(dev_seeds)
        rows.append({"step": step, "pooled": pooled,
                     "se_binomial": float(np.sqrt(pooled * (1 - pooled) / n_pool)),
                     **{f"s{s}": p for s, p in zip(dev_seeds, per)}})
    with open(campaign / "curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    uni = [r for r in rows if r["step"] % CFG_TRAIN["checkpoint_every"] == 0]
    peak = max(uni, key=lambda r: r["pooled"])
    tail = uni[-3:]
    summary = {
        "pooled_curve": [(r["step"], round(r["pooled"], 4)) for r in rows],
        "uniform_peak": {"step": peak["step"], "pooled": peak["pooled"]},
        "tail_mean_last3": float(np.mean([r["pooled"] for r in tail])),
        "peak_minus_tail": peak["pooled"] - float(np.mean([r["pooled"] for r in tail])),
        "se_binomial_pooled": rows[-1]["se_binomial"],
        "note": "探测口径（80 任务/种子，配对清单）；平台判定的正式数据条件见 CAL-P1",
    }
    (campaign / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    steps = [r["step"] for r in rows]
    for s in dev_seeds:
        ax.plot(steps, [r[f"s{s}"] for r in rows], color="#2a78d6", alpha=0.25, linewidth=1.1)
    pooled = [r["pooled"] for r in rows]
    se = [r["se_binomial"] for r in rows]
    ax.plot(steps, pooled, color="#2a78d6", linewidth=2.6, marker="o", markersize=5,
            label="种子池化")
    ax.fill_between(steps, [p - 2 * e for p, e in zip(pooled, se)],
                    [p + 2 * e for p, e in zip(pooled, se)], color="#2a78d6", alpha=0.15,
                    linewidth=0, label="±2 二项标准误")
    ax.set_xscale("log")
    ax.set_xlabel("训练步（对数轴）", color=ink2)
    ax.set_ylabel("近距导航成功率（BFS 2–6）", color=ink2)
    ax.set_title("S2 效用曲线探测（配对任务清单，探索性）", color=ink, fontsize=11)
    ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(campaign / "u_curve.png", facecolor=surface)
    print(f"汇总完成：峰值 {peak['pooled']:.3f}@{peak['step']}, "
          f"末三均值 {summary['tail_mean_last3']:.3f}, "
          f"峰-尾差 {summary['peak_minus_tail']:.3f} (±SE {summary['se_binomial_pooled']:.3f})")


if __name__ == "__main__":
    main()
