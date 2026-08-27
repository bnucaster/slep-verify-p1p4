"""S3 Transformer 世界模型训练（计划 11.3，选项 E）。

结构镜像 run_s2_train：逐种子 run、DONE 标记跳过、resume.pt 段内断点
续训、--budget-seconds 前台分块；末检查点导航 sanity 用 S3 穷举规划器
（口径与 S2 定稿同构）。

产物 results/<stage>/s3_train/<campaign>/（stage 由配置，默认 description）。

用法：.venv/Scripts/python.exe scripts/run_s3_train.py
  [--config configs/s3_train.yaml] [--budget-seconds N] [--smoke]
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

CONFIG_FILE = REPO_ROOT / "configs" / "s3_train.yaml"


class BudgetExhausted(Exception):
    pass


def checkpoint_steps(cfg: dict) -> list[int]:
    steps = set()
    k = 1
    while k <= cfg["checkpoint_log2_until"]:
        steps.add(k)
        k *= 2
    steps.update(range(cfg["checkpoint_every"], cfg["train_steps"] + 1, cfg["checkpoint_every"]))
    steps.add(cfg["train_steps"])
    return sorted(steps)


def build_model(cfg: dict) -> S3TransformerWM:
    return S3TransformerWM(
        cfg["obs_dim"], cfg["action_dim"], cfg["d_model"], cfg["n_layers"],
        cfg["n_heads"], cfg["ff_dim"], max_len=cfg["episode_len"] + 4,
        sigma_dec=cfg["sigma_dec"], goal_sigma_dec=cfg.get("goal_sigma_dec"),
    )


def nav_sanity(model: S3TransformerWM, cfg: dict, seed: int) -> dict:
    sn = cfg["sanity_nav"]
    rng = np.random.default_rng(seed + 10_000)
    gen = torch.Generator()
    gen.manual_seed(seed + 10_000)
    planner = S3ExhaustivePlanner(model, view=cfg["view"])
    successes, steps_used, n_done = 0, [], 0
    while n_done < sn["n_mazes"]:
        maze = gw.generate_maze(cfg["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or not sn["bfs_min"] <= d <= sn["bfs_max"]:
            continue
        env = gw.GridWorld(maze, start, cfg["view"], goal)
        out = s3_mpc_episode(model, env, planner, sn["max_steps"], gen)
        successes += int(out["success"])
        steps_used.append(out["steps"])
        n_done += 1
    return {"success_rate": successes / sn["n_mazes"], "n_mazes": sn["n_mazes"],
            "mean_steps": float(np.mean(steps_used)), "bfs_band": [sn["bfs_min"], sn["bfs_max"]]}


def train_one(cfg, seed: int, campaign, log, deadline=None) -> None:
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
        cfg["episodes"], cfg["episode_len"], cfg["maze_cells"], cfg["view"], rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    n_hold = cfg["holdout_episodes"]
    obs_train, act_train = obs[:-n_hold], act[:-n_hold]
    obs_hold, act_hold = obs[-n_hold:], act[-n_hold:]
    log(f"  s{seed} 数据采集完成 {tuple(obs.shape)}（{time.time() - t0:.0f}s）")

    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    ckpts = set(checkpoint_steps(cfg))

    resume_path = run_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        payload = torch.load(resume_path, weights_only=False)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        rng.bit_generator.state = payload["np_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        start_step = payload["step"]
        log(f"  s{seed} 续训自 step {start_step}")

    mode = "a" if start_step > 0 else "w"
    with open(run_dir / "losses.csv", mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if start_step == 0:
            writer.writerow(["step", "total", "mse_per_pixel", "holdout_total"])
        for step in range(start_step + 1, cfg["train_steps"] + 1):
            idx = rng.integers(0, obs_train.shape[0], size=cfg["batch_episodes"])
            out = model.rollout_loss(obs_train[idx], act_train[idx])
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            if step % cfg["log_every"] == 0 or step == 1:
                with torch.no_grad():
                    hold = model.rollout_loss(obs_hold, act_hold)
                writer.writerow([step, f"{out['total'].item():.4f}",
                                 f"{out['mse_per_pixel'].item():.6f}",
                                 f"{hold['total'].item():.4f}"])
            if step in ckpts:
                torch.save({"model": model.state_dict(), "step": step, "seed": seed},
                           ckpt_dir / f"ckpt_{step:06d}.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "np_rng_state": rng.bit_generator.state,
                            "torch_rng_state": torch.get_rng_state()}, resume_path)
                if deadline is not None and time.time() > deadline:
                    log(f"  s{seed} 预算耗尽，暂停于 step {step}")
                    raise BudgetExhausted
            if step % 2000 == 0:
                log(f"  s{seed} {step}/{cfg['train_steps']} nll={out['total'].item():.2f} "
                    f"({time.time() - t0:.0f}s)")

    sanity = nav_sanity(model, cfg, seed)
    (run_dir / "nav_sanity.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")
    wall = time.time() - t0
    (run_dir / "DONE").write_text(f"wall_seconds={wall:.0f}\n", encoding="utf-8")
    log(f"完成 s{seed}：{wall:.0f}s，导航 sanity 成功率 {sanity['success_rate']:.2f}"
        f"（近距带 {sanity['bfs_band']}）")


def main() -> None:
    config_file = CONFIG_FILE
    if "--config" in sys.argv:
        config_file = REPO_ROOT / sys.argv[sys.argv.index("--config") + 1]
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if "--smoke" in sys.argv:
        cfg = {**cfg, "campaign": "s3_smoke", "episodes": 60, "train_steps": 120,
               "checkpoint_log2_until": 16, "checkpoint_every": 60,
               "holdout_episodes": 12, "log_every": 40,
               "sanity_nav": {**cfg["sanity_nav"], "n_mazes": 4, "max_steps": 10}}
    torch.set_num_threads(int(cfg["torch_threads"]))
    campaign = create_campaign_dir(cfg.get("stage", "description"), "s3_train",
                                   cfg["campaign"], cfg)
    log_path = campaign / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds = guard.family_seeds(cfg["seed_family"], purpose="s3-train")
    if "--smoke" in sys.argv:
        seeds = seeds[:1]
    if cfg.get("max_seeds"):
        seeds = seeds[: int(cfg["max_seeds"])]
    deadline = None
    if "--budget-seconds" in sys.argv:
        deadline = time.time() + float(sys.argv[sys.argv.index("--budget-seconds") + 1])
    log(f"campaign: {campaign}，seeds={seeds}")
    try:
        for seed in seeds:
            guard.assert_seed_allowed(seed, purpose="s3-train")
            train_one(cfg, seed, campaign, log, deadline)
    except BudgetExhausted:
        return
    log("S3 战役完成")


if __name__ == "__main__":
    main()
