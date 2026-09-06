"""论文七图生成（任务十二 / M6，plan_v2 第 9 节清单）。

输入全部为已入库产物（exp_p1_*/exp_p2/exp_p3/exp_p4 缓存与汇总、
scoreboard_v12、ladder_pilot 数字），输出 docs/paper/figures/fig1–fig7.png。
样式沿用仓库图例惯例：单色调 + 灰阶文字，种子内个体用同色淡线、
汇总用重线；状态色只用于记分牌且必配文字。

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

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3d8f5f"
OUT = REPO_ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CONF = REPO_ROOT / "results" / "confirmation"
SEEDS = [5, 6, 7, 8, 9]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=MUTED, alpha=0.22, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)


def new_fig(w, h, ncols, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=180, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


# ---------- fig1 管线总图 ----------
def fig1():
    fig, ax = new_fig(9.0, 3.4, 1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face="#eef3fa"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=face, edgecolor=MUTED, linewidth=0.8))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", color=INK,
                fontsize=9.5, weight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.72 - 0.34 * i, ln, ha="center",
                    color=INK2, fontsize=8)

    def arrow(x0, y, x1):
        ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.008, head_width=0.12,
                                head_length=0.12, color=INK2))

    box(0.1, 0.6, 2.2, 2.9, "测试系统",
        ["S1 β-VAE / dSprites", "S2 / S2P GRU 世界模型", "S3 小 Transformer",
         "（种子分族守卫）"])
    arrow(2.4, 2.05, 2.9)
    box(2.9, 0.6, 2.4, 2.9, "估计器层",
        ["ĝ Fisher 拉回", "V̂ kNN 势 · p̂/Ŝ 双路", "Î 体积校正 · Â_OM",
         "测地 · 漂移分解"])
    arrow(5.4, 2.05, 5.9)
    box(5.9, 0.6, 2.0, 2.9, "校准定标",
        ["几何匹配合成", "Langevin 恢复", "可行域边界", "阈值全溯源"],
        face="#fdf1e8")
    arrow(8.0, 2.05, 8.5)
    box(8.5, 0.6, 1.4, 2.9, "冻结判定",
        ["入域门", "judge v1.1/v1.2", "三值记账", "外部时间戳"], face="#eef7f0")
    ax.set_title("图 1 · 系统—估计器—校准—判定管线", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pipeline.png", facecolor=SURFACE)
    plt.close(fig)


# ---------- fig2 P1 主图 ----------
def load_p1_rows(base: Path, run: str):
    rows = sorted((json.loads(p.read_text(encoding="utf-8"))
                   for p in (base / run / "cache").glob("c*.json")),
                  key=lambda r: r["step"])
    return rows


def fig2():
    fig, axes = new_fig(11.0, 5.6, 3, 2, sharex="col")
    systems = [
        ("S1（β=1 逃逸 run）", CONF / "exp_p1_s1" / "eval_v1",
         ["b1_s5", "b1_s6", "b1_s9"]),
        ("S2P", CONF / "exp_p1_s2p" / "s2p_eval_v1", [f"s{s}" for s in SEEDS]),
        ("S3", CONF / "exp_p1_s3" / "s3_eval_v1", [f"s{s}" for s in SEEDS]),
    ]
    for ci, (name, base, runs) in enumerate(systems):
        ax_u, ax_s = axes[0][ci], axes[1][ci]
        # 熵曲线只画均匀段（≥500 步，判定窗口所在）：早期未训练检查点的
        # 流密度数值崩溃产生大离群，与 P1 的峰后形状无关
        win_min = 500
        all_s = []
        for run in runs:
            rows = load_p1_rows(base, run)
            steps = [r["step"] for r in rows]
            u = [r["u_composite"] for r in rows]
            ax_u.plot(steps, u, color=BLUE, alpha=0.45, linewidth=1.2)
            peak_i = int(np.argmax(u))
            ax_u.plot(steps[peak_i], u[peak_i], marker="v", color=ORANGE,
                      markersize=6, linestyle="none")
            # 熵取 kNN 标准化口径（双路互证的数值稳健路；S2P 高维隐态上
            # flow 熵多点崩溃，判定的双路形状检查用整条曲线秩相关不受单点
            # 影响，但作图取稳健路更清楚）
            wr = [r for r in rows if r["step"] >= win_min]
            ax_s.plot([r["step"] for r in wr], [r["s_knn"] for r in wr],
                      color=BLUE, alpha=0.45, linewidth=1.2)
            all_s += [r["s_knn"] for r in wr]
        lo_s, hi_s = np.percentile(all_s, [1, 99])
        pad = 0.1 * (hi_s - lo_s + 1e-9)
        ax_s.set_ylim(lo_s - pad, hi_s + pad)
        ax_s.set_xlim(win_min * 0.8, None)
        ax_u.set_title(f"{name}", color=INK, fontsize=10)
        ax_u.set_xscale("log")
        ax_s.set_xscale("log")
        ax_s.set_xlabel("训练步（对数轴；熵只画均匀段 ≥500）", color=INK2, fontsize=9)
        if ci == 0:
            ax_u.set_ylabel("效用 Û", color=INK2, fontsize=9)
            ax_s.set_ylabel("熵 Ŝ（kNN 标准化，nats）", color=INK2, fontsize=9)
        style_ax(ax_u)
        style_ax(ax_s)
    axes[0][0].plot([], [], color=BLUE, alpha=0.6, label="逐 run 曲线")
    axes[0][0].plot([], [], marker="v", color=ORANGE, linestyle="none",
                    label="效用峰值")
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.suptitle("图 2 · P1：效用与熵沿训练（同列共 x 轴；峰后熵普遍上升，平台均未检出）",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig2_p1.png", facecolor=SURFACE)
    plt.close(fig)


# ---------- fig3 P2 ----------
def fig3():
    fig, axes = new_fig(11.0, 3.6, 3)
    ax = axes[0]
    gaps_by_seed = []
    for s in SEEDS:
        rows = []
        for p in sorted((CONF / "exp_p2" / "eval_v1" / f"s{s}" / "main_cache").glob("b*.json")):
            rows += json.loads(p.read_text(encoding="utf-8"))
        gaps_by_seed.append([r["gap"] for r in rows])
    parts = ax.violinplot(gaps_by_seed, positions=range(len(SEEDS)), widths=0.7,
                          showmedians=True)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE)
        b.set_alpha(0.45)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[k].set_color(INK2)
        parts[k].set_linewidth(1.0)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_xlabel("评估种子", color=INK2, fontsize=9)
    ax.set_ylabel("标准化间隙（代理中位 − 观测）/ IQR", color=INK2, fontsize=9)
    ax.set_title("(a) 集中性：低于 Q1 比例全部 1.000", color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[1]
    diffs_all = []
    for s in SEEDS:
        main, abl = {}, {}
        for p in sorted((CONF / "exp_p2" / "eval_v1" / f"s{s}" / "main_cache").glob("b*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                main[r["traj"]] = r["gap"]
        for p in sorted((CONF / "exp_p2" / "eval_v1" / f"s{s}" / "abl_cache").glob("b*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                abl[r["traj"]] = r["gap"]
        diffs_all += [main[t] - abl[t] for t in main if t in abl]
    ax.hist(diffs_all, bins=80, color=BLUE, alpha=0.75)
    med = float(np.median(diffs_all))
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(med, color=ORANGE, linewidth=1.6)
    ax.text(med, ax.get_ylim()[1] * 0.92, f" 中位 +{med:.4f}", color=ORANGE, fontsize=8)
    ax.set_xlim(np.quantile(diffs_all, 0.005), np.quantile(diffs_all, 0.995))
    ax.set_xlabel("逐轨迹配对差：含势间隙 − 常数势间隙", color=INK2, fontsize=9)
    ax.set_ylabel("轨迹数（五种子合并）", color=INK2, fontsize=9)
    ax.set_title("(b) 势项消融：Wilcoxon p ≤ 1e-12", color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[2]
    nov_p, pl_p = [], []
    for s in SEEDS:
        d = json.loads((CONF / "exp_p2" / "eval_v1" / f"s{s}" / "novelty.json")
                       .read_text(encoding="utf-8"))
        nov_p.append(d["logistic_p"])
        pl_p.append(d["placebo_p"])
    x = np.arange(len(SEEDS))
    ax.scatter(x - 0.12, nov_p, color=BLUE, s=42, label="新奇注入")
    ax.scatter(x + 0.12, pl_p, color=ORANGE, s=42, marker="s", label="安慰剂（时移）")
    ax.axhline(0.01, color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(SEEDS) - 0.4, 0.013, "判定线 0.01", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_xlabel("评估种子", color=INK2, fontsize=9)
    ax.set_ylabel("逻辑回归 p（对数轴）", color=INK2, fontsize=9)
    ax.set_title("(c) 新奇关联与安慰剂", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
    style_ax(ax)
    fig.suptitle("图 3 · P2 × S2P（通过）：三层判据", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig3_p2.png", facecolor=SURFACE)
    plt.close(fig)


# ---------- fig4 P3 ----------
def fig4():
    fig, axes = new_fig(10.0, 3.4, 3)
    ax = axes[0]
    res_by_seed = []
    for s in SEEDS:
        recs = [json.loads(p.read_text(encoding="utf-8"))
                for p in (CONF / "exp_p3" / "eval_v1" / f"s{s}" / "geo_cache").glob("seg_*.json")]
        res_by_seed.append([r["residual"] for r in recs])
    ax.boxplot(res_by_seed, positions=range(len(SEEDS)), widths=0.55,
               medianprops={"color": ORANGE}, boxprops={"color": INK2},
               whiskerprops={"color": INK2}, capprops={"color": INK2},
               flierprops={"markeredgecolor": MUTED, "markersize": 3})
    ax.axhline(3e-3, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(0.0, 4.2e-3, "质检阈值 3e-3（定标于解析度量）", color=ORANGE, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("测地求解残差（对数轴）", color=INK2, fontsize=9)
    ax.set_title("(a) 残差分布对冻结阈值", color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[1]
    cert, qcf, nonu = [], [], []
    for s in SEEDS:
        d = json.loads((CONF / "exp_p3" / "eval_v1" / f"s{s}" / "p3_summary.json")
                       .read_text(encoding="utf-8"))
        cert.append(d["n_certified_pairs"])
        qcf.append(d["attrition"]["qc_fail"])
        nonu.append(d["attrition"]["non_unique"])
    x = np.arange(len(SEEDS))
    ax.bar(x, qcf, width=0.55, color=MUTED, alpha=0.7, label="残差质检剔除")
    ax.bar(x, cert, width=0.55, bottom=qcf, color=BLUE, label="认证")
    ax.axhline(8, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(len(SEEDS) - 1.9, 9, "认证下限 8", color=ORANGE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("端点对数", color=INK2, fontsize=9)
    ax.set_title("(b) 衰减：认证对 1–3", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    style_ax(ax)

    ax = axes[2]
    lo, hi, mid = [], [], []
    for s in SEEDS:
        f = json.loads((CONF / "exp_p3" / "eval_v1" / f"s{s}" / "field.json")
                       .read_text(encoding="utf-8"))
        lo.append(f["ratio_ci90"][0])
        hi.append(f["ratio_ci90"][1])
        mid.append(f["hi_lo_median_ratio"])
    ax.errorbar(x, mid, yerr=[np.array(mid) - lo, np.array(hi) - mid], fmt="o",
                color=BLUE, markersize=6, capsize=4, linewidth=1.4)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("高/低梯度三分位中位比（CI90）", color=INK2, fontsize=9)
    ax.set_title("(c) 地形门健康", color=INK, fontsize=10)
    style_ax(ax)
    fig.suptitle("图 4 · P3 × S2P（检验未运行）：地形可检，认证被残差量表阻断",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig4_p3.png", facecolor=SURFACE)
    plt.close(fig)


# ---------- fig5 P4 ----------
def fig5():
    fig, axes = new_fig(11.0, 3.4, 3)
    ax = axes[0]
    sc = np.load(CONF / "exp_p4" / "s2p_eval_v1" / "s5" / "scatter.npz")
    m = json.loads((CONF / "exp_p4" / "s2p_eval_v1" / "s5" / "main.json")
                   .read_text(encoding="utf-8"))
    v, i_c = sc["v"], sc["i_corr"]
    ax.scatter(v, i_c, s=3, color=BLUE, alpha=0.25, linewidths=0)
    vv = np.linspace(v.min(), v.max(), 50)
    fit = m["affine_main"]
    ax.plot(vv, fit["intercept"] + fit["slope"] * vv, color=ORANGE, linewidth=1.8)
    ax.set_xlabel("V̂（nats）", color=INK2, fontsize=9)
    ax.set_ylabel("Î（体积校正，nats）", color=INK2, fontsize=9)
    ax.set_title(f"(a) Î–V̂（s5）：R²={fit['r_squared']:.2f}，ρ={fit['curvature_effect_ratio']:.2f}",
                 color=INK, fontsize=10)
    style_ax(ax)

    ax = axes[1]
    uni, swp = [], []
    for s in SEEDS:
        c = json.loads((CONF / "exp_p4" / "s2p_eval_v1" / f"s{s}" / "controls.json")
                       .read_text(encoding="utf-8"))
        uni.append(c["uniform_decoder"]["shift_rel"])
        swp.append(c["policy_swap"]["shift_rel"])
    x = np.arange(len(SEEDS))
    ax.scatter(x - 0.12, uni, color=BLUE, s=42, label="均匀解码器对照")
    ax.scatter(x + 0.12, swp, color=ORANGE, s=42, marker="s", label="换策略对照")
    ax.axhline(0.12, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.0, 0.128, "容差 0.12", color=INK2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("斜率相对漂移", color=INK2, fontsize=9)
    ax.set_title("(b) 解耦对照：均匀化训练全部越界", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
    style_ax(ax)

    ax = axes[2]
    rho, split = [], []
    for s in SEEDS:
        mm = json.loads((CONF / "exp_p4" / "s2p_eval_v1" / f"s{s}" / "main.json")
                        .read_text(encoding="utf-8"))
        rho.append(mm["affine_main"]["curvature_effect_ratio"])
        split.append(mm["t_split_rel"])
    ax.scatter(x - 0.12, rho, color=BLUE, s=42, label="曲率效应量 ρ")
    ax.scatter(x + 0.12, split, color=ORANGE, s=42, marker="s", label="分半温度相对差")
    ax.axhline(0.17, color=BLUE, linestyle="--", linewidth=1.1)
    ax.axhline(0.20, color=ORANGE, linestyle=":", linewidth=1.1)
    ax.text(len(SEEDS) - 1.6, 0.175, "ρ 上限 0.17", color=BLUE, fontsize=8)
    ax.text(0.0, 0.205, "分半容差 0.20", color=ORANGE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in SEEDS])
    ax.set_ylabel("统计量值", color=INK2, fontsize=9)
    ax.set_title("(c) 曲率超限、分半合格", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    style_ax(ax)
    fig.suptitle("图 5 · P4 × S2P（未构成检验）：仪器有分辨力，表观仿射不构成支持",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig5_p4.png", facecolor=SURFACE)
    plt.close(fig)


# ---------- fig6 记分牌 ----------
def fig6():
    board = json.loads((CONF / "scoreboard_v12.json").read_text(encoding="utf-8"))
    rows = [
        ("P2 × S2P", "低作用量集中"),
        ("P4 × S2P", "仿射律与温度"),
        ("P4 × S2(基线)", "仿射律与温度"),
        ("P3 × S2P", "低梯度区近测地"),
        ("P1 × S1", "能量下降与压缩"),
        ("P1 × S2P", "能量下降与压缩"),
        ("P1 × S3", "能量下降与压缩"),
    ]
    colors = {"通过": ("#eef7f0", GREEN), "失败": ("#fbecec", "#b3392f"),
              "未构成检验": ("#f2f2f0", MUTED)}
    fig, ax = new_fig(9.2, 3.6, 1)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 1)
    ax.text(0.2, len(rows) + 0.45, "预测 × 系统", color=INK2, fontsize=9, weight="bold")
    ax.text(2.6, len(rows) + 0.45, "判定（v1.2）", color=INK2, fontsize=9, weight="bold")
    ax.text(4.6, len(rows) + 0.45, "一句话依据", color=INK2, fontsize=9, weight="bold")
    for i, (cell, label) in enumerate(rows):
        y = len(rows) - 1 - i
        v = board["cells"][cell.replace(" ", "")]
        face, edge = colors[v["verdict"]]
        ax.text(0.2, y + 0.35, f"{cell}", color=INK, fontsize=9)
        ax.add_patch(FancyBboxPatch((2.5, y + 0.12), 1.7, 0.6,
                                    boxstyle="round,pad=0.04", facecolor=face,
                                    edgecolor=edge, linewidth=1.0))
        ax.text(3.35, y + 0.35, v["verdict"], ha="center", color=edge,
                fontsize=9, weight="bold")
        reason = {
            "P2 × S2P": "五种子全票：集中 + 消融归因 + 新奇关联",
            "P4 × S2P": "解耦对照失败（漂移 0.16–0.30 > 0.12）；ρ 全超 0.17",
            "P4 × S2(基线)": "几何门（条件数中位 9.2–9.7 > 7.0）",
            "P3 × S2P": "认证端点对 1–3 < 8（残差量表阻断）",
            "P1 × S1": "平台前提 / 双路仪器 / 数据条件各一",
            "P1 × S2P": "平台未检出（峰后缓降型）",
            "P1 × S3": "平台未检出（窗间波动型）",
        }[cell]
        ax.text(4.6, y + 0.35, reason, color=INK2, fontsize=8.5)
    ax.set_title("图 6 · 三值记分牌（judge v1.2；每格附门与对照状态见 metrics.json）",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_scoreboard.png", facecolor=SURFACE)
    plt.close(fig)


# ---------- fig7 阶梯（开发族，探索性） ----------
def fig7():
    fig, ax = new_fig(6.4, 3.4, 1)
    bins = ["可见段", "盲区近段", "远段"]
    base_mpc = [0.58, 0.33, 0.12]
    low_mpc = [0.50, 0.31, 0.11]
    rand = [0.51, 0.29, 0.08]
    x = np.arange(3)
    ax.scatter(x - 0.15, base_mpc, color=BLUE, s=48, label="基准档 MPC")
    ax.scatter(x, low_mpc, color=ORANGE, s=48, marker="s", label="低档（隐维 16→8）MPC")
    ax.scatter(x + 0.15, rand, color=MUTED, s=42, marker="^", label="随机基线")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("导航成功率（n=60/段，种子均值）", color=INK2, fontsize=9)
    ax.set_title("图 7 · 能力阶梯试点（S2，开发族，探索性）：隐维减半使可见段优势消失",
                 color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_ladder.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        fn()
        print(f"{fn.__name__} 完成", flush=True)
    print(f"图件写入 {OUT}")
