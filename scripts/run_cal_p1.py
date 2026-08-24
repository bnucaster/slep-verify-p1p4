"""CAL-P1：平台检测器定标（任务四 4a，plan_v2 第 5 节第 4 条）。

构造已知真值的两相 / 无两相效用曲线族，扫描噪声水平 σ 与窗长 W，量出：

- 检出率与定位误差 |t̂ − t*_W|（t*_W 为同一 W 下无噪声曲线的检出点，
  把窗长自身的固有延迟从噪声误差中分离）；
- 无平台曲线上的误检率（任何位置检出即计误检）；
- 缓增边界例（幂律）的检出时间单列报告。

真实 Û 的噪声水平待描述阶段实测（来源缺口）；本定标交付 σ→(W 选择)
的工作特性表与选择规则，冻结时按实测 σ 查表定 W。

产物 results/calibration/cal_p1/<run_id>/：config.json、meta.json、
metrics.json、operating.png、log.txt。

用法：.venv/Scripts/python.exe scripts/run_cal_p1.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import yaml

from slep import guard
from slep.protocols.plateau import detect_plateau
from slep.utils.runs import REPO_ROOT, create_run_dir

CONFIG_FILE = REPO_ROOT / "configs" / "cal_p1.yaml"


def make_curve(spec: dict, horizon: int, u_max: float) -> torch.Tensor:
    t = torch.arange(horizon, dtype=torch.float64)
    kind = spec["kind"]
    if kind == "exp":
        return u_max * (1 - torch.exp(-t / spec["tau"]))
    if kind == "logistic":
        return u_max / (1 + torch.exp(-(t - spec["center"]) / spec["scale"]))
    if kind == "piecewise":
        return torch.minimum(u_max * t / spec["knee"], torch.tensor(u_max, dtype=torch.float64))
    if kind == "linear":
        return 0.1 + (u_max - 0.1) * t / horizon
    if kind == "powerlaw":
        return u_max * (t / horizon).clamp_min(1e-12) ** spec["exponent"]
    raise ValueError(f"未知曲线类型 {kind!r}")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    run_dir = create_run_dir("calibration", "cal_p1", cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    seeds = guard.family_seeds(cfg["seed_family"], purpose="cal-p1")
    horizon, u_max = cfg["horizon"], cfg["u_max"]

    metrics: dict = {"curves": {}, "theta_source": "plan_v2 第 8 节 EXP-P1 第 1 条冻结值 1%/窗、连续 3 窗"}
    for name, spec in cfg["curves"].items():
        clean = make_curve(spec, horizon, u_max)
        curve_out: dict = {"group": spec["group"], "by_window": {}}
        for window in cfg["windows"]:
            t_star = detect_plateau(clean, window)["plateau_index"]
            row: dict = {"t_star_noiseless": t_star, "by_sigma": {}}
            for sigma in cfg["noise_sigmas"]:
                detects, errs = 0, []
                for rep in range(cfg["replicas"]):
                    gen = torch.Generator()
                    gen.manual_seed(seeds[rep % 2] * 1_000_003 + rep)
                    noisy = clean + sigma * torch.randn(horizon, generator=gen, dtype=torch.float64)
                    idx = detect_plateau(noisy, window)["plateau_index"]
                    if idx is not None:
                        detects += 1
                        if t_star is not None:
                            errs.append(abs(idx - t_star))
                stats = {
                    "detect_rate": detects / cfg["replicas"],
                }
                if errs:
                    errs_t = torch.tensor(errs, dtype=torch.float64)
                    stats["err_median"] = float(errs_t.median())
                    stats["err_q90"] = float(torch.quantile(errs_t, 0.9))
                row["by_sigma"][str(sigma)] = stats
            curve_out["by_window"][str(window)] = row
        metrics["curves"][name] = curve_out
        log(f"曲线 {name}（{spec['group']}）完成")

    # 选择规则（提案）：对每个 σ，取两相曲线全检出、无平台曲线零误检、
    # 且 q90 定位误差最小的最小 W。
    selection = {}
    for sigma in cfg["noise_sigmas"]:
        best = None
        for window in cfg["windows"]:
            ok, q90s = True, []
            for name, spec in cfg["curves"].items():
                row = metrics["curves"][name]["by_window"][str(window)]["by_sigma"][str(sigma)]
                if spec["group"] == "two_phase":
                    if row["detect_rate"] < 1.0:
                        ok = False
                    q90s.append(row.get("err_q90", float("inf")))
                elif spec["group"] == "no_plateau" and row["detect_rate"] > 0.0:
                    ok = False
            if ok:
                cand = (max(q90s), window)
                if best is None or cand < best:
                    best = cand
        selection[str(sigma)] = (
            {"window": best[1], "worst_two_phase_err_q90": best[0]} if best else None
        )
    metrics["window_selection_proposal"] = {
        "rule": "两相全检出且无平台零误检的窗中，取最差 q90 定位误差最小者；按描述阶段实测 σ 查表",
        "by_sigma": selection,
        "status": "提案：待任务七冻结",
    }
    log(f"选择表: {json.dumps(selection, ensure_ascii=False)}")

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(cfg, metrics, run_dir)
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


def _plot(cfg: dict, metrics: dict, run_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    sigmas = [float(s) for s in cfg["noise_sigmas"]]

    ax = axes[0]
    for window, color in zip(cfg["windows"], colors):
        ys = []
        for sigma in cfg["noise_sigmas"]:
            worst = 0.0
            for name, spec in cfg["curves"].items():
                if spec["group"] != "two_phase":
                    continue
                row = metrics["curves"][name]["by_window"][str(window)]["by_sigma"][str(sigma)]
                worst = max(worst, row.get("err_q90", float("nan")))
            ys.append(worst)
        ax.plot(sigmas, ys, color=color, linewidth=2, marker="o", markersize=6, label=f"W={window}")
    ax.set_xlabel("效用噪声 σ", color=ink2)
    ax.set_ylabel("两相曲线最差 q90 定位误差（检查点）", color=ink2)
    ax.set_title("定位误差", color=ink, fontsize=11)

    ax = axes[1]
    for window, color in zip(cfg["windows"], colors):
        ys = []
        for sigma in cfg["noise_sigmas"]:
            fp = 0.0
            for name, spec in cfg["curves"].items():
                if spec["group"] != "no_plateau":
                    continue
                row = metrics["curves"][name]["by_window"][str(window)]["by_sigma"][str(sigma)]
                fp = max(fp, row["detect_rate"])
            ys.append(fp)
        ax.plot(sigmas, ys, color=color, linewidth=2, marker="o", markersize=6, label=f"W={window}")
    ax.set_xlabel("效用噪声 σ", color=ink2)
    ax.set_ylabel("无平台曲线误检率", color=ink2)
    ax.set_title("误检", color=ink, fontsize=11)

    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelcolor=ink2)
        ax.legend(frameon=False, labelcolor=ink2)
    fig.suptitle("CAL-P1：平台检测器工作特性", color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "operating.png", facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
