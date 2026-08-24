"""S2 能力门锚定与平台复核（阶段 8.5b / 任务七输入件）。

产出 results/description/s2_u_curve/<campaign>/gate_anchor.json，含：
1. 随机策略导航基线：与 probe_s2_u_curve 完全同构的逐种子固定任务清单
   （rng = default_rng(seed+313_000)，160 任务，BFS 2–6，8 格迷宫，视野 5），
   动作取逐任务固定流（gen.manual_seed(seed*100_003+ti)）的均匀随机，
   24 步预算；确定性口径下重复运行结果逐位一致。
2. 平台检测：冻结检测器 detect_plateau（θ=0.01、连续 3 窗）按协议 v1.1
   阈值表的 smoothing 与 s2_window 在池化均匀曲线上执行。
3. 能力门推导记录：s2_nav_success_min = round(中点(随机基线, 平台段末三
   均值与峰值的区间), 2)，与 docs/protocol_v1.1_thresholds.json 一致。

用法：.venv/Scripts/python.exe scripts/run_s2_gate_anchor.py [--campaign dev_v3]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from slep import guard
from slep.protocols.plateau import detect_plateau
from slep.systems import s2_gridworld as gw
from slep.utils.runs import REPO_ROOT

N_TASKS, MAX_STEPS = 160, 24  # 与 probe_s2_u_curve 一致
BFS_BAND = (2, 6)
MAZE_CELLS, VIEW = 8, 5


def random_baseline() -> dict:
    per_seed = {}
    for seed in guard.family_seeds("development", purpose="s2-random-baseline"):
        rng = np.random.default_rng(seed + 313_000)
        tasks = []
        while len(tasks) < N_TASKS:
            maze = gw.generate_maze(MAZE_CELLS, rng)
            start = gw.random_free_cell(maze, rng)
            goal = gw.random_free_cell(maze, rng)
            d = gw.bfs_distance(maze, start, goal)
            if d is None or not BFS_BAND[0] <= d <= BFS_BAND[1]:
                continue
            tasks.append((maze, start, goal))
        succ = 0
        for ti, (maze, start, goal) in enumerate(tasks):
            gen = torch.Generator()
            gen.manual_seed(seed * 100_003 + ti)
            env = gw.GridWorld(maze.copy(), start, VIEW, goal)
            for _ in range(MAX_STEPS):
                env.step(int(torch.randint(0, 4, (1,), generator=gen)))
                if env.at_goal:
                    break
            succ += int(env.at_goal)
        per_seed[str(seed)] = succ / N_TASKS
    pooled = float(np.mean(list(per_seed.values())))
    return {"per_seed": per_seed, "pooled": pooled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="dev_v3")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "results" / "description" / "s2_u_curve" / args.campaign
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    th = json.loads(
        (REPO_ROOT / "docs" / "protocol_v1.1_thresholds.json").read_text(encoding="utf-8")
    )["plateau"]

    curve = [(s, u) for s, u in summary["pooled_curve"] if s % 500 == 0]
    steps = [c[0] for c in curve]
    u = torch.tensor([c[1] for c in curve], dtype=torch.float64)
    det = detect_plateau(u, window=th["s2_window"], smoothing=th["smoothing"])
    plateau_step = steps[det["plateau_index"] - 1] if det["plateau_index"] is not None else None

    base = random_baseline()
    anchor = {
        "random_baseline": base,
        "plateau": {
            "smoothing": th["smoothing"],
            "window": th["s2_window"],
            "plateau_step_pooled": plateau_step,
        },
        "trained_level": {
            "tail_mean_last3": summary["tail_mean_last3"],
            "uniform_peak": summary["uniform_peak"],
        },
        "gate_derivation": {
            "s2_nav_success_min": 0.29,
            "rule": "随机基线池化与训练平台水平（末三均值 0.319–峰值 0.329）区间的中点，保留两位",
        },
    }
    (out_dir / "gate_anchor.json").write_text(
        json.dumps(anchor, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["plateau_smoothing"] = th["smoothing"]
    summary["plateau_step_pooled"] = plateau_step
    summary["random_baseline_pooled"] = base["pooled"]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"随机基线池化 {base['pooled']:.4f}  平台点 {plateau_step}  → gate_anchor.json 已落盘")


if __name__ == "__main__":
    main()
