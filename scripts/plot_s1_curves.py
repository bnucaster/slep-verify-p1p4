"""S1 三曲线跨种子绘图（任务六 6a 收尾）。

读 s1_curves 战役的 curves.csv，逐 β 画三面板（Û、⟨V̂⟩、Ŝ 双路），
细线为单种子、粗线为逃逸种子均值；坍缩种子（末检查点 Û 低于
collapse_u_cut）以浅色标注并不入均值（均值口径注明图题）。

用法：.venv/Scripts/python.exe scripts/plot_s1_curves.py
"""
from __future__ import annotations

import csv

import numpy as np
import yaml

from slep.utils.runs import REPO_ROOT

COLLAPSE_U_CUT = 0.45  # 坍缩标注线：逃逸种子末 Û 实测 ≥0.6，坍缩 ≈0.33（探针 sanity），
                       # 取中点偏下；仅作图面分组，非判定阈值


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "s1_curves.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / "configs" / "s1_train.yaml").read_text(encoding="utf-8"))
    campaign = REPO_ROOT / "results" / "description" / "s1_curves" / cfg["out_campaign"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"

    for beta in train_cfg["betas"]:
        runs = []
        for run_dir in sorted(campaign.glob(f"b{beta:g}_s*")):
            with open(run_dir / "curves.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            runs.append((run_dir.name, rows))
        if not runs:
            continue
        steps = [int(r["step"]) for r in runs[0][1]]
        escaped = [name for name, rows in runs if float(rows[-1]["u_composite"]) >= COLLAPSE_U_CUT]

        fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.6), dpi=150, sharex=True)
        fig.patch.set_facecolor(surface)
        panels = [
            ("u_composite", None, "Û（因子探针复合）", blue),
            ("v_mean", None, "⟨V̂⟩（nats）", orange),
            ("s_flow", "s_knn", "Ŝ（nats；实线 flow，虚线 kNN）", aqua),
        ]
        for ax, (key, key2, label, color) in zip(axes, panels):
            ax.set_facecolor(surface)
            esc_stack = []
            for name, rows in runs:
                y = [float(r[key]) for r in rows]
                is_esc = name in escaped
                ax.plot(steps, y, color=color, alpha=0.45 if is_esc else 0.15, linewidth=1.1)
                if key2:
                    y2 = [float(r[key2]) for r in rows]
                    ax.plot(steps, y2, color=color, alpha=0.35 if is_esc else 0.1,
                            linewidth=1.0, linestyle="--")
                if is_esc:
                    esc_stack.append(y)
            if esc_stack:
                ax.plot(steps, np.mean(esc_stack, axis=0), color=color, linewidth=2.6)
            ax.set_ylabel(label, color=ink2)
            ax.grid(True, axis="y", color=muted, alpha=0.22, linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(muted)
            ax.tick_params(colors=muted, labelcolor=ink2)
        axes[0].set_ylim(0, 1)
        axes[1].set_yscale("symlog", linthresh=100)  # V̂ 早期 ~1e4、后期为负，对称对数轴
        axes[-1].set_xscale("log")
        axes[-1].set_xlabel("训练步（对数轴）", color=ink2)
        fig.suptitle(
            f"S1 三曲线 β={beta:g}（开发族；粗线 = 逃逸种子均值 {len(escaped)}/5，"
            f"浅线 = 坍缩种子）", color=ink, fontsize=12)
        fig.tight_layout()
        fig.savefig(campaign / f"curves_b{beta:g}.png", facecolor=surface)
        plt.close(fig)
        print(f"β={beta:g}: 逃逸 {len(escaped)}/5 → curves_b{beta:g}.png")


if __name__ == "__main__":
    main()
