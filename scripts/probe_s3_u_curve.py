"""S3 效用曲线探测（计划 11.3；口径与 probe_s2_u_curve 完全同构）。

逐种子固定 160 任务（BFS 2–6）、逐任务固定随机流、检查点均匀段全部
加早期 {1, 32, 256}；S3 穷举规划器（视界 1 + ε）。细粒度缓存逐
（种子 × 检查点）。产物 results/description/s3_u_curve/<campaign>/。

用法：.venv/Scripts/python.exe scripts/probe_s3_u_curve.py
  [--config configs/s3_train_v2.yaml] [--seed N]
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
from slep.systems.s3_transformer import S3ExhaustivePlanner, S3TransformerWM, s3_mpc_episode
from slep.utils.runs import REPO_ROOT, create_campaign_dir

_cfg_file = "configs/s3_train_v2.yaml"
if "--config" in sys.argv:
    _cfg_file = sys.argv[sys.argv.index("--config") + 1]
CFG = yaml.safe_load((REPO_ROOT / _cfg_file).read_text(encoding="utf-8"))
N_TASKS, MAX_STEPS = 160, 24
EARLY_STEPS = [1, 32, 256]
BFS_BAND = (2, 6)


def eval_steps() -> list[int]:
    uniform = list(range(CFG["checkpoint_every"], CFG["train_steps"] + 1,
                         CFG["checkpoint_every"]))
    return EARLY_STEPS + uniform


def build_tasks(seed: int):
    rng = np.random.default_rng(seed + 313_000)
    tasks = []
    while len(tasks) < N_TASKS:
        maze = gw.generate_maze(CFG["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or not BFS_BAND[0] <= d <= BFS_BAND[1]:
            continue
        tasks.append((maze, start, goal))
    return tasks


def load_model(seed: int, step: int) -> S3TransformerWM:
    ck = torch.load(REPO_ROOT / "results" / CFG.get("stage", "description") / "s3_train"
                    / CFG["campaign"] / f"s{seed}" / "checkpoints"
                    / f"ckpt_{step:06d}.pt", weights_only=True)
    model = S3TransformerWM(CFG["obs_dim"], CFG["action_dim"], CFG["d_model"],
                            CFG["n_layers"], CFG["n_heads"], CFG["ff_dim"],
                            max_len=CFG["episode_len"] + 4, sigma_dec=CFG["sigma_dec"],
                            goal_sigma_dec=CFG.get("goal_sigma_dec"),
                            multi_step_k=CFG.get("multi_step_k", 1))
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main() -> None:
    torch.set_num_threads(10)
    only_seed = None
    if "--seed" in sys.argv:
        only_seed = int(sys.argv[sys.argv.index("--seed") + 1])
    campaign = create_campaign_dir("description", "s3_u_curve", CFG["campaign"],
                                   {"n_tasks": N_TASKS, "max_steps": MAX_STEPS,
                                    "bfs_band": list(BFS_BAND), "steps": eval_steps()})
    cache_dir = campaign / "cache"
    cache_dir.mkdir(exist_ok=True)

    seeds = guard.family_seeds(CFG["seed_family"], purpose="s3-u-curve")
    if only_seed is not None:
        seeds = [only_seed]
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="s3-u-curve")
        tasks = build_tasks(seed)
        for step in eval_steps():
            cache = cache_dir / f"s{seed}_c{step:06d}.json"
            if cache.exists():
                continue
            t0 = time.time()
            model = load_model(seed, step)
            planner = S3ExhaustivePlanner(model, view=CFG["view"])
            succ = 0
            for ti, (maze, start, goal) in enumerate(tasks):
                gen = torch.Generator()
                gen.manual_seed(seed * 100_003 + ti)
                env = gw.GridWorld(maze.copy(), start, CFG["view"], goal)
                succ += int(s3_mpc_episode(model, env, planner, MAX_STEPS, gen)["success"])
            cache.write_text(json.dumps({"success": succ, "n": N_TASKS}), encoding="utf-8")
            print(f"s{seed} ckpt {step}: {succ}/{N_TASKS} ({time.time() - t0:.0f}s)",
                  flush=True)

    all_seeds = guard.family_seeds(CFG["seed_family"], purpose="s3-u-curve")
    if not all((cache_dir / f"s{s}_c{c:06d}.json").exists()
               for s in all_seeds for c in eval_steps()):
        print("尚有缓存缺口，汇总跳过")
        return
    rows = []
    for step in eval_steps():
        per = [json.loads((cache_dir / f"s{s}_c{step:06d}.json").read_text())["success"]
               / N_TASKS for s in all_seeds]
        rows.append({"step": step, "pooled": float(np.mean(per)),
                     **{f"s{s}": p for s, p in zip(all_seeds, per)}})
    with open(campaign / "curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    uni = [r for r in rows if r["step"] % CFG["checkpoint_every"] == 0]
    peak = max(uni, key=lambda r: r["pooled"])
    summary = {"pooled_curve": [(r["step"], round(r["pooled"], 4)) for r in rows],
               "uniform_peak": {"step": peak["step"], "pooled": peak["pooled"]},
               "tail_mean_last3": float(np.mean([r["pooled"] for r in uni[-3:]]))}
    (campaign / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"汇总完成：峰值 {peak['pooled']:.3f}@{peak['step']}, "
          f"末三均值 {summary['tail_mean_last3']:.3f}")


if __name__ == "__main__":
    main()
