"""S2 导航能力评估（任务五 5d 收尾；正式 Û 曲线的口径基础）。

对战役内全部种子的末检查点，用定稿规划器（ExhaustiveMPCPlanner
视界 1 + ε=0.2）在固定迷宫清单上评估分距离段成功率，并配对随机基线。
落盘各 run 目录 nav_eval.json 与战役级 nav_eval_summary.json。

用法：.venv/Scripts/python.exe scripts/eval_s2_nav.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.systems import s2_gridworld as gw
from slep.systems.s2_planner import ExhaustiveMPCPlanner, mpc_episode
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT

CONFIG_FILE = REPO_ROOT / "configs" / "s2_train.yaml"
N_PER_BIN = 60
MAX_STEPS = 24
BINS = {"visible": None, "bfs2-4": (2, 4), "bfs5-6": (5, 6)}
EVAL_STREAM_OFFSET = 424242  # 评估迷宫流，与训练/往期诊断流分离


def build_cases(cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed + EVAL_STREAM_OFFSET)
    half = cfg["view"] // 2

    def visible(start, goal):
        return abs(goal[0] - start[0]) <= half and abs(goal[1] - start[1]) <= half

    cases = {k: [] for k in BINS}
    while any(len(v) < N_PER_BIN for v in cases.values()):
        maze = gw.generate_maze(cfg["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or d == 0:
            continue
        if visible(start, goal) and d <= 4 and len(cases["visible"]) < N_PER_BIN:
            cases["visible"].append((maze, start, goal))
        elif not visible(start, goal):
            for name, band in BINS.items():
                if band and band[0] <= d <= band[1] and len(cases[name]) < N_PER_BIN:
                    cases[name].append((maze, start, goal))
                    break
    return cases


def eval_run(cfg: dict, seed: int, run_dir) -> dict:
    ckpt = torch.load(
        run_dir / "checkpoints" / f"ckpt_{cfg['train_steps']:06d}.pt", weights_only=True
    )
    model = S2WorldModel(
        cfg["obs_dim"], cfg["action_dim"], cfg["embed_dim"], cfg["hidden_dim"],
        cfg["sigma_dec"], cfg.get("goal_sigma_dec"),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    planner = ExhaustiveMPCPlanner(model, view=cfg["view"])

    cases = build_cases(cfg, seed)
    out = {"planner": "exhaustive_h1_eps0.2", "n_per_bin": N_PER_BIN, "max_steps": MAX_STEPS}
    for name, lst in cases.items():
        gen = torch.Generator()
        gen.manual_seed(seed)
        rng_pol = np.random.default_rng(seed + 99)
        mpc_s = rnd_s = 0
        for maze, start, goal in lst:
            env = gw.GridWorld(maze.copy(), start, cfg["view"], goal)
            mpc_s += int(mpc_episode(model, env, planner, MAX_STEPS, gen)["success"])
            env2 = gw.GridWorld(maze.copy(), start, cfg["view"], goal)
            for a in rng_pol.integers(0, 4, size=MAX_STEPS):
                env2.step(int(a))
                if env2.at_goal:
                    break
            rnd_s += int(env2.at_goal)
        out[name] = {"mpc": mpc_s / N_PER_BIN, "random": rnd_s / N_PER_BIN}
    return out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    campaign = REPO_ROOT / "results" / "description" / "s2_train" / cfg["campaign"]
    seeds = guard.family_seeds(cfg["seed_family"], purpose="s2-nav-eval")
    summary = {"campaign": cfg["campaign"], "runs": {}, "date": time.strftime("%Y-%m-%d")}
    for seed in seeds:
        run_dir = campaign / f"s{seed}"
        if not (run_dir / "DONE").exists():
            print(f"跳过未完成 run s{seed}")
            continue
        res = eval_run(cfg, seed, run_dir)
        (run_dir / "nav_eval.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["runs"][f"s{seed}"] = res
        print(
            f"s{seed}: visible {res['visible']['mpc']:.2f}/{res['visible']['random']:.2f}, "
            f"bfs2-4 {res['bfs2-4']['mpc']:.2f}/{res['bfs2-4']['random']:.2f}, "
            f"bfs5-6 {res['bfs5-6']['mpc']:.2f}/{res['bfs5-6']['random']:.2f} (MPC/随机)"
        )
    (campaign / "nav_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"汇总已写入 {campaign / 'nav_eval_summary.json'}")


if __name__ == "__main__":
    main()
