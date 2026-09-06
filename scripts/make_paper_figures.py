"""论文七图生成（任务十二 / M6，plan_v2 第 9 节清单）——英文投稿版。

输入全部为已入库产物（exp_p1_*/exp_p2/exp_p3/exp_p4 缓存与汇总、
scoreboard_v12、ladder_pilot 数字），输出 docs/paper/figures/figN.{pdf,png}。

学术规范：Times New Roman 正文 + STIX 数学体（与 Times 匹配）、白底、
300 dpi、矢量 PDF（供 LaTeX）+ PNG（预览）；图内不放编号大标题，编号与
说明由 LaTeX \\caption 承担；两系列面板一律用标记形状作二次编码
（圆/方/三角），配色蓝橙为 CVD 安全对，状态色仅记分牌通过格且必配文字。

用法：.venv/Scripts/python.exe scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

from slep.utils.runs import REPO_ROOT

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # 嵌入 TrueType，投稿兼容
    "ps.fonttype": 42,
})

SURFACE, INK, INK2, MUTED = "#ffffff", "#000000", "#333333", "#8a8a8a"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2e7d4f"
OUT = REPO_ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CONF = REPO_ROOT / "results" / "confirmation"
SEEDS = [5, 6, 7, 8, 9]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelcolor=INK, labelsize=8)


def new_fig(w, h, ncols, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------- fig1 pipeline ----------
def fig1():
    fig, ax = new_fig(9.0, 3.2, 1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face="#eef3fa"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=face, edgecolor=INK2, linewidth=0.8))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", color=INK,
                fontsize=10, weight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.74 - 0.36 * i, ln, ha="center",
                    color=INK, fontsize=8)

    def arrow(x0, y, x1):
        ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.006, head_width=0.13,
                                head_length=0.12, color=INK2, length_includes_head=True))

    box(0.1, 0.55, 2.2, 3.0, "Test systems",
        [r"S1 $\beta$-VAE / dSprites", "S2 / S2P GRU world model",
         "S3 small Transformer", "(seed-family guard)"])
    arrow(2.35, 2.05, 2.85)
    box(2.9, 0.55, 2.4, 3.0, "Estimator layer",
        [r"$\hat{g}$ Fisher pullback", r"$\hat{V}$ kNN pot., $\hat{p}/\hat{S}$ dual-path",
         r"$\hat{I}$ vol.-corr., $\hat{A}_{\mathrm{OM}}$", "geodesic, drift split"])
    arrow(5.35, 2.05, 5.85)
    box(5.9, 0.55, 2.0, 3.0, "Calibration",
        ["geometry-matched synth.", "Langevin recovery", "feasibility boundary",
         "all thresholds traced"], face="#fdf1e8")
    arrow(7.95, 2.05, 8.45)
    box(8.5, 0.55, 1.4, 3.0, "Frozen judging",
        ["admission gates", "judge v1.1/v1.2", "three-valued", "external timestamp"],
        face="#eef7f0")
    fig.tight_layout()
    save(fig, "fig1_pipeline")


# ---------- fig2 P1 ----------
def load_p1_rows(base: Path, run: str):
    return sorted((json.loads(p.read_text(encoding="utf-8"))
                   for p in (base / run / "cache").glob("c*.json")),
                  key=lambda r: r["step"])


def fig2():
    fig, axes = new_fig(11.0, 5.4, 3, 2, sharex="col")
    systems = [
        (r"S1 ($\beta{=}1$ escape runs)", CONF / "exp_p1_s1" / "eval_v1",
         ["b1_s5", "b1_s6", "b1_s9"]),
        ("S2P", CONF / "exp_p1_s2p" / "s2p_eval_v1", [f"s{s}" for s in SEEDS]),
        ("S3", CONF / "exp_p1_s3" / "s3_eval_v1", [f"s{s}" for s in SEEDS]),
    ]
    win_min = 500
    for ci, (name, base, runs) in enumerate(systems):
        ax_u, ax_s = axes[0][ci], axes[1][ci]
        all_s = []
        for run in runs:
            rows = load_p1_rows(base, run)
            steps = [r["step"] for r in rows]
            u = [r["u_composite"] for r in rows]
            ax_u.plot(steps, u, color=BLUE, alpha=0.5, linewidth=1.1)
            pk = int(np.argmax(u))
            ax_u.plot(steps[pk], u[pk], marker="v", color=ORANGE, markersize=6,
                      linestyle="none")
            wr = [r for r in rows if r["step"] >= win_min]
            ax_s.plot([r["step"] for r in wr], [r["s_knn"] for r in wr],
                      color=BLUE, alpha=0.5, linewidth=1.1)
            all_s += [r["s_knn"] for r in wr]
        lo, hi = np.percentile(all_s, [1, 99])
        pad = 0.1 * (hi - lo + 1e-9)
        ax_s.set_ylim(lo - pad, hi + pad)
        ax_s.set_xlim(win_min * 0.8, None)
        ax_u.set_title(name, color=INK, fontsize=11)
        ax_u.set_xscale("log")
        ax_s.set_xscale("log")
        ax_s.set_xlabel(r"Training step (log; entropy shows uniform seg. $\geq$500)",
                        color=INK, fontsize=9)
        if ci == 0:
            ax_u.set_ylabel(r"Utility $\hat{U}$", color=INK, fontsize=10)
            ax_s.set_ylabel(r"Entropy $\hat{S}$ (kNN std., nats)", color=INK, fontsize=10)
        style_ax(ax_u)
        style_ax(ax_s)
    axes[0][0].plot([], [], color=BLUE, alpha=0.6, linewidth=1.1, label="per-run curve")
    axes[0][0].plot([], [], marker="v", color=ORANGE, linestyle="none",
                    label="utility peak")
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_p1")


# ---------- fig3 P2 ----------
def fig3():
    fig, axes = new_fig(11.0, 3.5, 3)
    ax = axes[0]
    gaps = []
    for s in SEEDS:
        rows = []
        for p in sorted((CONF / "exp_p2/eval_v1" / f"s{s}" / "main_cache").glob("b*.json")):
            rows += json.loads(p.read_text(encoding="utf-8"))
        gaps.append([r["gap"] for r in rows])
    parts = ax.violinplot(gaps, positions=range(len(SEEDS)), widths=0.7, showmedians=True)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE)
        b.set_alpha(0.45)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[k].set_color(INK2)
        parts[k].set_linewidth(1.0)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_xlabel("Evaluation seed", color=INK, fontsize=9)
    ax.set_ylabel("Standardized gap (surr. median $-$ obs) / IQR", color=INK, fontsize=9)
    ax.set_title("(a) Concentration: fraction below Q1 all 1.000", color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[1]
    diffs = []
    for s in SEEDS:
        main, abl = {}, {}
        for p in sorted((CONF / "exp_p2/eval_v1" / f"s{s}" / "main_cache").glob("b*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                main[r["traj"]] = r["gap"]
        for p in sorted((CONF / "exp_p2/eval_v1" / f"s{s}" / "abl_cache").glob("b*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                abl[r["traj"]] = r["gap"]
        diffs += [main[t] - abl[t] for t in main if t in abl]
    ax.hist(diffs, bins=80, color=BLUE, alpha=0.75)
    med = float(np.median(diffs))
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(med, color=ORANGE, linewidth=1.6)
    ax.text(med, ax.get_ylim()[1] * 0.92, f"  median $+{med:.4f}$", color=ORANGE, fontsize=8)
    ax.set_xlim(np.quantile(diffs, 0.005), np.quantile(diffs, 0.995))
    ax.set_xlabel("Per-traj. paired diff: with-pot. $-$ const-pot. gap", color=INK, fontsize=9)
    ax.set_ylabel("Trajectory count (five seeds pooled)", color=INK, fontsize=9)
    ax.set_title(r"(b) Potential-term ablation: Wilcoxon $p \leq 10^{-12}$",
                 color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[2]
    nov, pl = [], []
    for s in SEEDS:
        d = json.loads((CONF / "exp_p2/eval_v1" / f"s{s}" / "novelty.json")
                       .read_text(encoding="utf-8"))
        nov.append(d["logistic_p"])
        pl.append(d["placebo_p"])
    x = np.arange(len(SEEDS))
    ax.scatter(x - 0.12, nov, color=BLUE, s=42, label="novelty injection")
    ax.scatter(x + 0.12, pl, color=ORANGE, s=42, marker="s", label="placebo (time-shift)")
    ax.axhline(0.01, color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(SEEDS) - 1.0, 0.013, "decision line 0.01", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_xlabel("Evaluation seed", color=INK, fontsize=9)
    ax.set_ylabel("Logistic regression $p$ (log)", color=INK, fontsize=9)
    ax.set_title("(c) Novelty association vs. placebo", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig3_p2")


# ---------- fig4 P3 ----------
def fig4():
    fig, axes = new_fig(10.5, 3.4, 3)
    ax = axes[0]
    res = []
    for s in SEEDS:
        recs = [json.loads(p.read_text(encoding="utf-8"))
                for p in (CONF / "exp_p3/eval_v1" / f"s{s}" / "geo_cache").glob("seg_*.json")]
        res.append([r["residual"] for r in recs])
    ax.boxplot(res, positions=range(len(SEEDS)), widths=0.55,
               medianprops={"color": ORANGE}, boxprops={"color": INK2},
               whiskerprops={"color": INK2}, capprops={"color": INK2},
               flierprops={"markeredgecolor": MUTED, "markersize": 3})
    ax.axhline(3e-3, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(-0.3, 4.4e-3, r"QC threshold $3{\times}10^{-3}$ (analytic-metric)",
            color=ORANGE, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("Geodesic solve residual (log)", color=INK, fontsize=9)
    ax.set_title("(a) Residual vs. frozen threshold", color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[1]
    cert, qcf = [], []
    for s in SEEDS:
        d = json.loads((CONF / "exp_p3/eval_v1" / f"s{s}" / "p3_summary.json")
                       .read_text(encoding="utf-8"))
        cert.append(d["n_certified_pairs"])
        qcf.append(d["attrition"]["qc_fail"])
    x = np.arange(len(SEEDS))
    ax.bar(x, qcf, width=0.55, color=MUTED, alpha=0.7, label="residual QC rejected")
    ax.bar(x, cert, width=0.55, bottom=qcf, color=BLUE, label="certified")
    ax.axhline(8, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(len(SEEDS) - 2.4, 10, "certification floor 8", color=ORANGE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("Endpoint-pair count", color=INK, fontsize=9)
    ax.set_title("(b) Attrition: 1–3 certified", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style_ax(ax)

    ax = axes[2]
    lo, hi, mid = [], [], []
    for s in SEEDS:
        f = json.loads((CONF / "exp_p3/eval_v1" / f"s{s}" / "field.json")
                       .read_text(encoding="utf-8"))
        lo.append(f["ratio_ci90"][0])
        hi.append(f["ratio_ci90"][1])
        mid.append(f["hi_lo_median_ratio"])
    ax.errorbar(x, mid, yerr=[np.array(mid) - lo, np.array(hi) - mid], fmt="o",
                color=BLUE, markersize=6, capsize=4, linewidth=1.4)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("High/low gradient tertile ratio (CI90)", color=INK, fontsize=9)
    ax.set_title("(c) Terrain-gate health", color=INK, fontsize=10)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig4_p3")


# ---------- fig5 P4 ----------
def fig5():
    fig, axes = new_fig(11.0, 3.4, 3)
    ax = axes[0]
    sc = np.load(CONF / "exp_p4/s2p_eval_v1/s5/scatter.npz")
    m = json.loads((CONF / "exp_p4/s2p_eval_v1/s5/main.json").read_text(encoding="utf-8"))
    v, i_c = sc["v"], sc["i_corr"]
    ax.scatter(v, i_c, s=3, color=BLUE, alpha=0.25, linewidths=0)
    vv = np.linspace(v.min(), v.max(), 50)
    fit = m["affine_main"]
    ax.plot(vv, fit["intercept"] + fit["slope"] * vv, color=ORANGE, linewidth=1.8)
    ax.set_xlabel(r"$\hat{V}$ (nats)", color=INK, fontsize=10)
    ax.set_ylabel(r"$\hat{I}$ (volume-corrected, nats)", color=INK, fontsize=10)
    ax.set_title(rf"(a) $\hat{{I}}$--$\hat{{V}}$ (s5): $R^2={fit['r_squared']:.2f}$, "
                 rf"$\rho={fit['curvature_effect_ratio']:.2f}$", color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[1]
    uni, swp = [], []
    for s in SEEDS:
        c = json.loads((CONF / "exp_p4/s2p_eval_v1" / f"s{s}" / "controls.json")
                       .read_text(encoding="utf-8"))
        uni.append(c["uniform_decoder"]["shift_rel"])
        swp.append(c["policy_swap"]["shift_rel"])
    x = np.arange(len(SEEDS))
    ax.scatter(x - 0.12, uni, color=BLUE, s=42, label="uniform-decoder control")
    ax.scatter(x + 0.12, swp, color=ORANGE, s=42, marker="s", label="policy-swap control")
    ax.axhline(0.12, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(-0.3, 0.128, "tolerance 0.12", color=INK2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("Slope relative drift", color=INK, fontsize=9)
    ax.set_title("(b) Decoupling: uniform training exceeds", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    style_ax(ax)

    ax = axes[2]
    rho, split = [], []
    for s in SEEDS:
        mm = json.loads((CONF / "exp_p4/s2p_eval_v1" / f"s{s}" / "main.json")
                        .read_text(encoding="utf-8"))
        rho.append(mm["affine_main"]["curvature_effect_ratio"])
        split.append(mm["t_split_rel"])
    ax.scatter(x - 0.12, rho, color=BLUE, s=42, label=r"curvature effect size $\rho$")
    ax.scatter(x + 0.12, split, color=ORANGE, s=42, marker="s",
               label="split-half temp. rel. diff")
    ax.axhline(0.17, color=BLUE, linestyle="--", linewidth=1.1)
    ax.axhline(0.20, color=ORANGE, linestyle=":", linewidth=1.1)
    ax.text(len(SEEDS) - 2.0, 0.18, r"$\rho$ bound 0.17", color=BLUE, fontsize=8)
    ax.text(-0.3, 0.205, "split-half tol. 0.20", color=ORANGE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("Statistic value", color=INK, fontsize=9)
    ax.set_title("(c) Curvature over bound; split-half OK", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig5_p4")


# ---------- fig6 scoreboard ----------
def fig6():
    board = json.loads((CONF / "scoreboard_v12.json").read_text(encoding="utf-8"))
    verdict_en = {"通过": "Pass", "失败": "Fail", "未构成检验": "TNC"}
    rows = [
        ("P2 $\\times$ S2P", "P2×S2P"),
        ("P4 $\\times$ S2P", "P4×S2P"),
        ("P4 $\\times$ S2 (base)", "P4×S2(基线)"),
        ("P3 $\\times$ S2P", "P3×S2P"),
        ("P1 $\\times$ S1", "P1×S1"),
        ("P1 $\\times$ S2P", "P1×S2P"),
        ("P1 $\\times$ S3", "P1×S3"),
    ]
    reasons = {
        "P2×S2P": "five-seed sweep: concentration + ablation + novelty",
        "P4×S2P": "decoupling fails (drift 0.16–0.30 > 0.12); $\\rho$ > 0.17",
        "P4×S2(基线)": "geometry gate (cond. median 9.2–9.7 > 7.0)",
        "P3×S2P": "certified pairs 1–3 < 8 (residual-scale block)",
        "P1×S1": "plateau premise / dual-path instr. / data condition",
        "P1×S2P": "plateau not detected (post-peak slow decline)",
        "P1×S3": "plateau not detected (window-scale wobble)",
    }
    colors = {"通过": ("#e9f5ee", GREEN), "失败": ("#fbecec", "#b3392f"),
              "未构成检验": ("#f0f0f0", MUTED)}
    fig, ax = new_fig(9.6, 3.4, 1)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 1)
    ax.text(0.2, len(rows) + 0.4, "Prediction $\\times$ system", color=INK,
            fontsize=9.5, weight="bold")
    ax.text(2.6, len(rows) + 0.4, "Verdict (v1.2)", color=INK, fontsize=9.5, weight="bold")
    ax.text(4.7, len(rows) + 0.4, "One-line reason", color=INK, fontsize=9.5, weight="bold")
    for i, (disp, key) in enumerate(rows):
        y = len(rows) - 1 - i
        v = board["cells"][key]["verdict"]
        face, edge = colors[v]
        ax.text(0.2, y + 0.32, disp, color=INK, fontsize=9)
        ax.add_patch(FancyBboxPatch((2.5, y + 0.1), 1.7, 0.58,
                                    boxstyle="round,pad=0.04", facecolor=face,
                                    edgecolor=edge, linewidth=1.0))
        ax.text(3.35, y + 0.32, verdict_en[v], ha="center", color=edge,
                fontsize=9, weight="bold")
        ax.text(4.7, y + 0.32, reasons[key], color=INK, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig6_scoreboard")


# ---------- fig7 ladder (development family, exploratory) ----------
def fig7():
    fig, ax = new_fig(6.4, 3.3, 1)
    bins = ["Visible", "Blind near", "Far"]
    base_mpc = [0.58, 0.33, 0.12]
    low_mpc = [0.50, 0.31, 0.11]
    rand = [0.51, 0.29, 0.08]
    x = np.arange(3)
    ax.scatter(x - 0.15, base_mpc, color=BLUE, s=50, label="base tier MPC")
    ax.scatter(x, low_mpc, color=ORANGE, s=50, marker="s",
               label=r"low tier (hidden 16$\to$8) MPC")
    ax.scatter(x + 0.15, rand, color=MUTED, s=44, marker="^", label="random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("Navigation success (n=60/bin, seed mean)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig7_ladder")


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        fn()
        print(f"{fn.__name__} done", flush=True)
    print(f"figures written to {OUT}")
