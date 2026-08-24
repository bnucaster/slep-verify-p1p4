"""S1（β-VAE / dSprites）短程试点：训练 + ĝ 谱与各向异性提取（任务二 2b）。

目的（docs/plan_v2.md 第 5 节第 1 条）：在校准种子上短训，提取访问潜点处
Fisher 度量 ĝ 的特征谱、条件数与各向异性，供构造几何匹配的合成校准系统。
试点定位为几何测绘，训练长度与容量不代表任务五的正式 S1。

产物落盘 results/calibration/s1_pilot/<run_id>/：
config.json、meta.json、log.txt、每个 run 的 losses.csv、
spectra_beta<β>.npz（潜点、ĝ 特征值、log det）、summary.json、spectra.png、
checkpoints/（git 忽略）。

用法：.venv/Scripts/python.exe scripts/run_s1_pilot.py
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
from slep.estimators import metric
from slep.systems.s1_beta_vae import S1BetaVAE
from slep.utils.runs import REPO_ROOT, create_run_dir

CONFIG_FILE = REPO_ROOT / "configs" / "s1_pilot.yaml"


def load_packed_dataset(cfg: dict) -> np.ndarray:
    """dSprites 图像打包成位数组缓存（737280×512 uint8，节省内存）。"""
    cache = REPO_ROOT / cfg["packed_cache"]
    if cache.exists():
        return np.load(cache)
    raw = np.load(REPO_ROOT / cfg["data_file"])["imgs"]  # (N, 64, 64) uint8 0/1
    packed = np.packbits(raw.reshape(raw.shape[0], -1), axis=1)
    np.save(cache, packed)
    return packed


def batch_from_packed(packed: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    imgs = np.unpackbits(packed[idx], axis=1).astype(np.float32)
    return torch.from_numpy(imgs).reshape(-1, 1, 64, 64)


def train_one(cfg: dict, run_cfg: dict, packed: np.ndarray, run_dir: Path, log) -> dict:
    seed = guard.family_seeds(run_cfg["seed_family"], purpose="s1-pilot-train")[
        run_cfg["seed_index"]
    ]
    beta = float(run_cfg["beta"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = S1BetaVAE(cfg["latent_dim"], cfg["channels"], cfg["fc_dim"], cfg["sigma_dec"], beta)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    losses_path = run_dir / f"losses_beta{beta:g}.csv"
    t0 = time.time()
    with open(losses_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "recon", "kl"])
        for step in range(1, cfg["train_steps"] + 1):
            idx = rng.integers(0, packed.shape[0], size=cfg["batch_size"])
            x = batch_from_packed(packed, idx)
            out = model.loss(x)
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            if step % cfg["log_every"] == 0 or step == 1:
                writer.writerow(
                    [step, f"{out['total'].item():.4f}", f"{out['recon'].item():.4f}", f"{out['kl'].item():.4f}"]
                )
                if step % (cfg["log_every"] * 5) == 0 or step == 1:
                    log(
                        f"  beta={beta:g} step {step}/{cfg['train_steps']} "
                        f"total={out['total'].item():.1f} recon={out['recon'].item():.1f} "
                        f"kl={out['kl'].item():.2f} ({time.time() - t0:.0f}s)"
                    )

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"s1_beta{beta:g}_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "run": run_cfg, "seed": seed}, ckpt_path)
    log(f"  beta={beta:g} 检查点: {ckpt_path.name}")

    stats = extract_spectra(cfg, model, packed, rng, beta, seed, run_dir, log)
    stats["kl_per_dim_final"] = [round(v, 4) for v in model_kl_per_dim(model, packed, rng, cfg)]
    stats["checkpoint"] = str(ckpt_path.relative_to(REPO_ROOT))
    return stats


def model_kl_per_dim(model: S1BetaVAE, packed: np.ndarray, rng, cfg: dict) -> list[float]:
    """留出批上的逐维 KL，维度坍缩诊断（接近 0 的维已坍缩）。"""
    with torch.no_grad():
        idx = rng.integers(0, packed.shape[0], size=256)
        out = model.loss(batch_from_packed(packed, idx))
    return out["kl_per_dim"].tolist()


def extract_spectra(cfg, model, packed, rng, beta, seed, run_dir: Path, log) -> dict:
    """访问潜点（编码器后验均值）处 ĝ 的特征谱统计。"""
    with torch.no_grad():
        idx = rng.integers(0, packed.shape[0], size=cfg["n_encode_for_spectra"])
        mu_z, _ = model.encoder(batch_from_packed(packed, idx))
    sub = rng.choice(mu_z.shape[0], size=cfg["n_spectra_points"], replace=False)
    z_points = mu_z[sub].to(torch.float64)

    obs_var = cfg["sigma_dec"] ** 2
    eigvals = []
    model.decoder.double()
    for z in z_points:
        g = metric.fisher_pullback_gaussian(model.decoder_mean_flat, z, obs_var)
        eigvals.append(torch.linalg.eigvalsh(g.detach()))
    model.decoder.float()
    eig = torch.stack(eigvals)  # (n, d)，升序
    logdet = torch.log(eig).sum(dim=-1)
    cond = eig[:, -1] / eig[:, 0]

    np.savez(
        run_dir / f"spectra_beta{beta:g}.npz",
        z=z_points.numpy(), eigvals=eig.numpy(), logdet=logdet.numpy(), seed=seed,
    )
    q = torch.tensor([0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    stats = {
        "beta": beta,
        "seed": seed,
        "eig_median_by_rank": torch.quantile(eig, 0.5, dim=0).tolist(),
        "log10_cond_quantiles": torch.quantile(torch.log10(cond), q).tolist(),
        "logdet_mean": logdet.mean().item(),
        "logdet_std": logdet.std().item(),
        "eff_dim_participation_median": torch.quantile(
            eig.sum(-1) ** 2 / (eig**2).sum(-1), 0.5
        ).item(),
    }
    log(
        f"  beta={beta:g} 谱: log10 条件数中位 {stats['log10_cond_quantiles'][1]:.2f}, "
        f"logdet {stats['logdet_mean']:.1f}±{stats['logdet_std']:.1f}, "
        f"参与维数中位 {stats['eff_dim_participation_median']:.2f}"
    )
    return stats


def plot_spectra(all_stats: list[dict], run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    colors = ["#2a78d6", "#eb6834"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for stats, color in zip(all_stats, colors):
        data = np.load(run_dir / f"spectra_beta{stats['beta']:g}.npz")
        eig = data["eigvals"]  # (n, d) 升序
        ranks = np.arange(1, eig.shape[1] + 1)
        med = np.median(eig, axis=0)[::-1]
        lo = np.quantile(eig, 0.25, axis=0)[::-1]
        hi = np.quantile(eig, 0.75, axis=0)[::-1]
        ax.plot(ranks, med, color=color, linewidth=2, marker="o", markersize=6,
                label=f"β={stats['beta']:g}（种子 {stats['seed']}）")
        ax.fill_between(ranks, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("特征值序（大 → 小）", color=ink2)
    ax.set_ylabel("ĝ 特征值（中位数与四分位带）", color=ink2)
    ax.set_title("S1 试点：访问潜点处 Fisher 度量谱", color=ink, fontsize=11)
    ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(run_dir / "spectra.png", facecolor=surface)
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    run_dir = create_run_dir("calibration", "s1_pilot", cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    packed = load_packed_dataset(cfg)
    log(f"数据集: {packed.shape[0]} 张图（位打包缓存）")

    all_stats = []
    for run_cfg in cfg["runs"]:
        log(f"run: {run_cfg}")
        all_stats.append(train_one(cfg, run_cfg, packed, run_dir, log))

    (run_dir / "summary.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_spectra(all_stats, run_dir)
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
