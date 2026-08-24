"""V̂ 主操作化的 k 敏感性曲线（docs/plan_v2.md 第 4 节要求随 V̂ 报告）。

在已知势合成系统（systems/selfcheck.py，参数与 tests/test_potential.py
一致）上扫描近邻数 k，对确定性查询网格计算两条误差曲线：

- 总误差：mean_q |V̂(z_q) − V_true(z_q)|，含离散化偏差与观测噪声；
- 噪声项误差：mean_q |V̂(z_q) − V_oracle(z_q; k)|，扣除确定性偏差后
  纯观测噪声的贡献，随 k 增大按 1/sqrt(k) 收缩。

两条曲线之差随 k 的抬升即离散化偏差的增长，供后续校准阶段为 k 的选取
提供实测依据。本脚本属估计器自检配套产物，不产生任何判定阈值；阈值
待校准（来源缺口：docs/plan_v2.md 第 5 节校准阶段尚未运行）。

产物落盘 results/calibration/potential_k_sensitivity/<run_id>/：
config.json、meta.json（git 哈希）、log.txt、curve.csv、curve.json、
curve.png。

用法：.venv/Scripts/python.exe scripts/run_potential_k_sensitivity.py
"""
from __future__ import annotations

import csv
import json

import torch

from slep import guard
from slep.estimators import potential
from slep.systems.selfcheck import LinearGaussianSelfCheck
from slep.utils.runs import create_run_dir

CONFIG = {
    "seed_family": "calibration",  # 实际种子经 guard.family_seeds 从 configs/seeds.yaml 解析
    "seed_index": 0,
    "latent_dim": 2,
    "obs_dim": 8,
    "alpha": 1.0,
    "sigma_x": 0.1,
    "sigma_dec": 0.3,
    "n_ref": 40_000,
    "k_grid": [4, 8, 16, 32, 64, 128, 256],
    "query_grid_side": 7,
    "query_grid_halfwidth": 1.4,
}


def main() -> None:
    seed = guard.family_seeds(CONFIG["seed_family"], purpose="cal-potential-k-sensitivity")[
        CONFIG["seed_index"]
    ]
    run_dir = create_run_dir("calibration", "potential_k_sensitivity", {**CONFIG, "seed_resolved": seed})
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")

    gen = torch.Generator()
    gen.manual_seed(seed)
    system = LinearGaussianSelfCheck.build(
        CONFIG["latent_dim"], CONFIG["obs_dim"], CONFIG["alpha"],
        CONFIG["sigma_x"], CONFIG["sigma_dec"], generator=gen,
    )
    z_ref, x_ref = system.sample(CONFIG["n_ref"], generator=gen, purpose="cal-k-sens-sample")

    side, half = CONFIG["query_grid_side"], CONFIG["query_grid_halfwidth"]
    axis = torch.linspace(-half, half, side, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    z_query = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    v_true = system.v_true(z_query)
    log(f"查询点 {z_query.shape[0]} 个，V_true 范围 [{v_true.min():.3f}, {v_true.max():.3f}] nats")

    rows = []
    for k in CONFIG["k_grid"]:
        v_hat, info = potential.potential_knn(
            system.decoder_mean, CONFIG["sigma_dec"] ** 2,
            z_query, z_ref, x_ref, k=k, return_info=True,
        )
        z_nb = z_ref[info["indices"]]
        v_oracle = system.v_oracle_knn(z_query, z_nb)
        row = {
            "k": k,
            "err_vs_true": (v_hat - v_true).abs().mean().item(),
            "err_vs_oracle": (v_hat - v_oracle).abs().mean().item(),
            "mean_noise_se": system.noise_se(z_query, z_nb).mean().item(),
            "mean_knn_radius": info["radius"].mean().item(),
            "max_knn_radius": info["radius"].max().item(),
        }
        rows.append(row)
        log(
            f"k={k:4d}  err_true={row['err_vs_true']:.4f}  "
            f"err_oracle={row['err_vs_oracle']:.4f}  "
            f"radius(mean/max)={row['mean_knn_radius']:.3f}/{row['max_knn_radius']:.3f}"
        )

    with open(run_dir / "curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "curve.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _plot(rows, run_dir, seed)
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


def _plot(rows: list[dict], run_dir, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Windows 中文字形；缺失时退回 DejaVu（仅影响标签显示）。
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    blue, orange = "#2a78d6", "#eb6834"

    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.plot(ks, [r["err_vs_true"] for r in rows], color=blue, linewidth=2,
            marker="o", markersize=6, label="总误差 |V̂ − V_true|")
    ax.plot(ks, [r["err_vs_oracle"] for r in rows], color=orange, linewidth=2,
            marker="o", markersize=6, label="噪声项 |V̂ − V_oracle(k)|")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("近邻数 k", color=ink2)
    ax.set_ylabel("查询点平均绝对误差（nats）", color=ink2)
    ax.set_title(f"V̂ 主操作化的 k 敏感性（已知势合成系统，种子 {seed}）",
                 color=ink, fontsize=11)
    ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2)

    fig.tight_layout()
    fig.savefig(run_dir / "curve.png", facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
