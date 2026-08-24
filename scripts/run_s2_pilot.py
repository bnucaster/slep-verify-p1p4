"""S2（GRU 世界模型 / 网格世界）短程试点：训练 + 隐状态 ĝ 谱提取（任务二 2c）。

目的同 S1 试点（docs/plan_v2.md 第 5 节第 1 条）：几何测绘，供构造几何
匹配的合成校准系统。训练长度与容量不代表任务五的正式 S2。

产物落盘 results/calibration/s2_pilot/<run_id>/：
config.json、meta.json、log.txt、每 run 的 losses.csv 与
spectra_seed<seed>.npz、summary.json、spectra.png、checkpoints/（git 忽略）。

用法：.venv/Scripts/python.exe scripts/run_s2_pilot.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from slep import guard
from slep.systems.s2_gridworld import collect_rollouts
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_run_dir
from slep.utils.spectra import plot_rank_spectra, spectra_at_points, summarize_spectra

CONFIG_FILE = REPO_ROOT / "configs" / "s2_pilot.yaml"


def train_one(cfg: dict, run_cfg: dict, run_dir: Path, log) -> dict:
    seed = guard.family_seeds(run_cfg["seed_family"], purpose="s2-pilot-train")[
        run_cfg["seed_index"]
    ]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    log(f"  seed={seed} 采集 {cfg['episodes']} 回合 × {cfg['episode_len']} 步…")
    obs_np, act_np = collect_rollouts(
        cfg["episodes"], cfg["episode_len"], cfg["maze_cells"], cfg["view"], rng
    )
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    n_hold = cfg["holdout_episodes"]
    obs_train, act_train = obs[:-n_hold], act[:-n_hold]
    obs_hold, act_hold = obs[-n_hold:], act[-n_hold:]

    model = S2WorldModel(
        cfg["obs_dim"], cfg["action_dim"], cfg["embed_dim"], cfg["hidden_dim"], cfg["sigma_dec"]
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    t0 = time.time()
    with open(run_dir / f"losses_seed{seed}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "mse_per_pixel"])
        for step in range(1, cfg["train_steps"] + 1):
            idx = rng.integers(0, obs_train.shape[0], size=cfg["batch_episodes"])
            out = model.rollout_loss(obs_train[idx], act_train[idx])
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            if step % cfg["log_every"] == 0 or step == 1:
                writer.writerow(
                    [step, f"{out['total'].item():.4f}", f"{out['mse_per_pixel'].item():.6f}"]
                )
                if step % (cfg["log_every"] * 5) == 0 or step == 1:
                    log(
                        f"  seed={seed} step {step}/{cfg['train_steps']} "
                        f"nll={out['total'].item():.2f} mse={out['mse_per_pixel'].item():.4f} "
                        f"({time.time() - t0:.0f}s)"
                    )

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"s2_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "seed": seed}, ckpt_path)
    log(f"  seed={seed} 检查点: {ckpt_path.name}")

    # 留出回合的隐状态轨迹（教师强制，参数已冻结在 no_grad 下），剔除
    # burn_in 步后子采样谱提取点。
    with torch.no_grad():
        hs = model.hidden_trajectory(obs_hold, act_hold)  # (E, T, H)
    hs = hs[:, cfg["burn_in"] :].reshape(-1, cfg["hidden_dim"])
    sub = rng.choice(hs.shape[0], size=cfg["n_spectra_points"], replace=False)
    h_points = hs[sub].to(torch.float64)

    model.double()
    spec = spectra_at_points(model.decoder_mean_flat, cfg["sigma_dec"] ** 2, h_points)
    model.float()
    np.savez(
        run_dir / f"spectra_seed{seed}.npz",
        z=h_points.numpy(),
        eigvals=spec["eigvals"].numpy(),
        logdet=spec["logdet"].numpy(),
        seed=seed,
    )
    stats = {"seed": seed, "checkpoint": str(ckpt_path.relative_to(REPO_ROOT))}
    stats.update(summarize_spectra(spec))
    stats["final_nll"] = out["total"].item()
    stats["final_mse_per_pixel"] = out["mse_per_pixel"].item()
    log(
        f"  seed={seed} 谱: log10 条件数中位 {stats['log10_cond_quantiles'][1]:.2f}, "
        f"logdet {stats['logdet_mean']:.1f}±{stats['logdet_std']:.1f}, "
        f"参与维数中位 {stats['eff_dim_participation_median']:.2f}"
    )
    return stats


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    run_dir = create_run_dir("calibration", "s2_pilot", cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    all_stats = []
    for run_cfg in cfg["runs"]:
        log(f"run: {run_cfg}")
        all_stats.append(train_one(cfg, run_cfg, run_dir, log))

    (run_dir / "summary.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    entries = []
    for stats, color in zip(all_stats, ["#2a78d6", "#eb6834"]):
        data = np.load(run_dir / f"spectra_seed{stats['seed']}.npz")
        entries.append(
            {"label": f"种子 {stats['seed']}", "eigvals": data["eigvals"], "color": color}
        )
    plot_rank_spectra(entries, run_dir / "spectra.png", "S2 试点：访问隐状态处 Fisher 度量谱")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
