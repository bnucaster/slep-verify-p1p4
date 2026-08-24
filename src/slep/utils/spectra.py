"""ĝ 谱提取与汇总（几何测绘，plan_v2 第 5 节第 1 条）。

对一组潜点逐点求 Fisher 度量拉回，给出特征值、log det、条件数与
汇总统计；供 S1/S2 试点脚本与几何匹配拟合共用。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from slep.estimators import metric


def spectra_at_points(
    decoder_mean_flat: Callable[[torch.Tensor], torch.Tensor],
    obs_var: float | torch.Tensor,
    z_points: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """逐点特征谱。z_points 形状 (n, d)；返回 eigvals (n, d) 升序、
    logdet (n,)、cond (n,)。"""
    eigvals = []
    for z in z_points:
        g = metric.fisher_pullback_gaussian(decoder_mean_flat, z, obs_var)
        eigvals.append(torch.linalg.eigvalsh(g.detach()))
    eig = torch.stack(eigvals)
    return {
        "eigvals": eig,
        "logdet": torch.log(eig).sum(dim=-1),
        "cond": eig[:, -1] / eig[:, 0],
    }


def summarize_spectra(spec: dict[str, torch.Tensor]) -> dict:
    """匹配拟合与报告用的汇总统计。"""
    eig, logdet, cond = spec["eigvals"], spec["logdet"], spec["cond"]
    q = torch.tensor([0.25, 0.5, 0.75, 0.95], dtype=eig.dtype)
    return {
        "eig_median_by_rank": torch.quantile(eig, 0.5, dim=0).tolist(),
        "log10_cond_quantiles": torch.quantile(torch.log10(cond), q).tolist(),
        "logdet_mean": logdet.mean().item(),
        "logdet_std": logdet.std().item(),
        "eff_dim_participation_median": torch.quantile(
            eig.sum(-1) ** 2 / (eig**2).sum(-1), 0.5
        ).item(),
    }


def plot_rank_spectra(entries: list[dict], path, title: str) -> None:
    """特征值序谱图。entries 元素含 label、eigvals (n,d) 升序 ndarray、color。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for entry in entries:
        eig = np.asarray(entry["eigvals"])
        ranks = np.arange(1, eig.shape[1] + 1)
        ax.plot(ranks, np.median(eig, axis=0)[::-1], color=entry["color"], linewidth=2,
                marker="o", markersize=6, label=entry["label"])
        ax.fill_between(ranks, np.quantile(eig, 0.25, axis=0)[::-1],
                        np.quantile(eig, 0.75, axis=0)[::-1],
                        color=entry["color"], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("特征值序（大 → 小）", color=ink2)
    ax.set_ylabel("ĝ 特征值（中位数与四分位带）", color=ink2)
    ax.set_title(title, color=ink, fontsize=11)
    ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
