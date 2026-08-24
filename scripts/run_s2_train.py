"""S2 正式开发族训练（M2 描述，任务五 5d）。

开发族 5 种子各一 run，可断点续跑（DONE 标记跳过）。每 run：随机策略
数据采集（带目标通道）→ 教师强制训练 + 密集检查点 → 末检查点 MPC
近距目标导航 sanity（BFS 距离 2–6 的留出迷宫成功率，只记录不判定；
正式 Û 曲线在任务六按 CAL-P1 数据条件计算）。

产物 results/description/s2_train/<campaign>/：config.json、meta_*.json、
s<种子>/（losses.csv、nav_sanity.json、DONE、checkpoints/ 为 git 忽略）。

用法：.venv/Scripts/python.exe scripts/run_s2_train.py [--smoke]
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

CONFIG_FILE = REPO_ROOT / "configs" / "s2_train.yaml"


def checkpoint_steps(cfg: dict) -> list[int]:
    steps = set()
    k = 1
    while k <= cfg["checkpoint_log2_until"]:
        steps.add(k)
        k *= 2
    steps.update(range(cfg["checkpoint_every"], cfg["train_steps"] + 1, cfg["checkpoint_every"]))
    steps.add(cfg["train_steps"])
    return sorted(steps)


def nav_sanity(model: S2WorldModel, cfg: dict, seed: int) -> dict:
    """末检查点近距目标导航成功率（留出迷宫，BFS 距离带内的目标）。"""
    sn = cfg["sanity_nav"]
    rng = np.random.default_rng(seed + 10_000)  # 留出迷宫流，与训练数据流分离
    gen = torch.Generator()
    gen.manual_seed(seed + 10_000)
    planner = ExhaustiveMPCPlanner(model, view=cfg["view"])  # 定稿口径（视界 1 + ε）
    successes, steps_used = 0, []
    n_done = 0
    while n_done < sn["n_mazes"]:
        maze = gw.generate_maze(cfg["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or not sn["bfs_min"] <= d <= sn["bfs_max"]:
            continue
        env = gw.GridWorld(maze, start, cfg["view"], goal)
        out = mpc_episode(model, env, planner, sn["max_steps"], gen)
        successes += int(out["success"])
        steps_used.append(out["steps"])
        n_done += 1
    return {
        "success_rate": successes / sn["n_mazes"],
        "n_mazes": sn["n_mazes"],
        "mean_steps": float(np.mean(steps_used)),
        "bfs_band": [sn["bfs_min"], sn["bfs_max"]],
    }


def train_one(cfg: dict, seed: int, campaign, log) -> None:
    run_dir = campaign / f"s{seed}"
    if (run_dir / "DONE").exists():
        log(f"跳过已完成 run s{seed}")
        return
    run_dir.mkdir(exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    obs_np, act_np = gw.collect_rollouts(
        cfg["episodes"], cfg["episode_len"], cfg["maze_cells"], cfg["view"], rng
    )
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    n_hold = cfg["holdout_episodes"]
    obs_train, act_train = obs[:-n_hold], act[:-n_hold]
    obs_hold, act_hold = obs[-n_hold:], act[-n_hold:]
    log(f"  s{seed} 数据采集完成 {tuple(obs.shape)}（{time.time() - t0:.0f}s）")

    model = S2WorldModel(
        cfg["obs_dim"], cfg["action_dim"], cfg["embed_dim"], cfg["hidden_dim"],
        cfg["sigma_dec"], cfg.get("goal_sigma_dec"),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    ckpts = set(checkpoint_steps(cfg))

    with open(run_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "mse_per_pixel", "holdout_total"])
        for step in range(1, cfg["train_steps"] + 1):
            idx = rng.integers(0, obs_train.shape[0], size=cfg["batch_episodes"])
            out = model.rollout_loss(obs_train[idx], act_train[idx])
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            if step % cfg["log_every"] == 0 or step == 1:
                with torch.no_grad():
                    hold = model.rollout_loss(obs_hold, act_hold)
                writer.writerow(
                    [step, f"{out['total'].item():.4f}", f"{out['mse_per_pixel'].item():.6f}",
                     f"{hold['total'].item():.4f}"]
                )
            if step in ckpts:
                torch.save(
                    {"model": model.state_dict(), "step": step, "seed": seed},
                    ckpt_dir / f"ckpt_{step:06d}.pt",
                )
            if step % 2000 == 0:
                log(
                    f"  s{seed} {step}/{cfg['train_steps']} nll={out['total'].item():.2f} "
                    f"({time.time() - t0:.0f}s)"
                )

    sanity = nav_sanity(model, cfg, seed)
    (run_dir / "nav_sanity.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    wall = time.time() - t0
    (run_dir / "DONE").write_text(f"wall_seconds={wall:.0f}\n", encoding="utf-8")
    log(
        f"完成 s{seed}：{wall:.0f}s，导航 sanity 成功率 {sanity['success_rate']:.2f}"
        f"（近距带 {sanity['bfs_band']}）"
    )


def main() -> None:
    config_file = CONFIG_FILE
    if "--config" in sys.argv:
        config_file = REPO_ROOT / sys.argv[sys.argv.index("--config") + 1]
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    smoke = "--smoke" in sys.argv
    if smoke:
        cfg = {**cfg, "campaign": "smoke_v1", "episodes": 60, "train_steps": 120,
               "checkpoint_log2_until": 16, "checkpoint_every": 60, "holdout_episodes": 12,
               "log_every": 40,
               "sanity_nav": {**cfg["sanity_nav"], "n_mazes": 4, "max_steps": 10,
                              "planner": {"horizon": 6, "n_samples": 24, "n_elites": 4, "n_iters": 2}}}
    torch.set_num_threads(int(cfg["torch_threads"]))
    campaign = create_campaign_dir("description", "s2_train", cfg["campaign"], cfg)
    log_path = campaign / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds = guard.family_seeds(cfg["seed_family"], purpose="s2-train-dev")
    if smoke:
        seeds = seeds[:1]
    log(f"campaign: {campaign}，seeds={seeds}")
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="s2-train")
        train_one(cfg, seed, campaign, log)
    log("S2 战役完成")


if __name__ == "__main__":
    main()
