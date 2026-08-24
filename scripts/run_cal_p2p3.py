"""CAL-P2/P3：Â_OM 分位分离与测地偏差的噪声地板（任务四 4b，plan_v2
第 5 节第 3 条）。

P2 部分（几何匹配 d=16 系统）：
- 观测轨迹 = 真 Langevin 连续段；每条配 n_surrogates 条主代理（速度
  分布匹配的平滑随机路径，protocols/surrogates.py）；
- 真值臂：Â_OM 用闭式 V/g/T；全链臂：V̂ 用 k 近邻估计（含 autograd
  梯度）在子样本上重复；
- 统计：逐轨迹观测作用量在代理分布中的分位、"低于代理 Q1"比例；
  空白地板：以代理充当伪观测得到的同一统计（应落在 25% 基线的二项
  波动带内）；势项消融（常数势打分）同表并报。

P3 部分：
- 中段弦长端点对上多起点测地求解：残差与唯一性散布分布（认证基准）；
- 求解路径对闭式真测地的偏差 = 求解器地板；注入横向正弦偏差
  （幅度 α × 弦长，y 空间构造后拉回）后的实测偏差–注入曲线 →
  最小可检测差 = 实测中位数超出零注入地板 q95 的最小 α。

产物 results/calibration/cal_p2p3/<run_id>/：metrics.json、
p2_separation.png、p3_deviation.png、log.txt。阈值以提案状态输出。

用法：.venv/Scripts/python.exe scripts/run_cal_p2p3.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators import potential
from slep.estimators.geodesic import geodesic_deviation, solve_geodesic
from slep.estimators.om_action import om_action
from slep.protocols.surrogates import smooth_random_surrogate
from slep.systems.cal_langevin import system_from_params
from slep.utils.runs import REPO_ROOT, create_run_dir

CONFIG_FILE = REPO_ROOT / "configs" / "cal_p2p3.yaml"


def simulate_continuous(system, cfg_p2, gen) -> torch.Tensor:
    """thin=1 连续链，切成 (n_traj, seg_len+1, d) 观测轨迹。"""
    z = system.simulate_chains(
        cfg_p2["n_chains"], cfg_p2["sim_steps"], cfg_p2["dt"], cfg_p2["burn_in"], 1, gen
    )  # (C, m, d)
    seg = cfg_p2["seg_len"] + 1
    per_chain = z.shape[1] // seg
    trajs = z[:, : per_chain * seg].reshape(z.shape[0] * per_chain, seg, z.shape[2])
    idx = torch.randperm(trajs.shape[0], generator=gen)[: cfg_p2["n_traj"]]
    return trajs[idx]


def actions_for(
    trajs: torch.Tensor,
    system,
    cfg_p2: dict,
    potential_fn,
    gen: torch.Generator,
    label: str,
    log,
) -> dict:
    """逐轨迹：观测与代理的 Â_OM；返回分位统计。potential_fn=None 表示
    常数势（消融）。"""
    dt = cfg_p2["dt"]
    t_true = system.temperature

    def pot(z):
        if potential_fn is None:
            return (z * 0.0).sum(dim=-1)  # 常数势；保持与 z 的图连通，梯度为零
        return potential_fn(z)

    ranks, below_q1, null_below_q1, length_ratios = [], [], [], []
    for i, traj in enumerate(trajs):
        a_obs = float(
            om_action(traj, dt, pot, system.metric_true, t_true, cfg_p2["grad_point"])
        )
        a_surr = []
        for _ in range(cfg_p2["n_surrogates"]):
            sur, info = smooth_random_surrogate(traj, gen)
            length_ratios.append(info["length_ratio"])
            a_surr.append(
                float(om_action(sur, dt, pot, system.metric_true, t_true, cfg_p2["grad_point"]))
            )
        a_surr_t = torch.tensor(a_surr, dtype=torch.float64)
        ranks.append(float((a_surr_t < a_obs).double().mean()))
        below_q1.append(a_obs < float(torch.quantile(a_surr_t, 0.25)))
        # 空白地板：首条代理充当伪观测，对其余代理取同一统计
        null_below_q1.append(
            a_surr[0] < float(torch.quantile(a_surr_t[1:], 0.25))
        )
        if (i + 1) % 50 == 0:
            log(f"  [{label}] {i + 1}/{trajs.shape[0]}")

    n = len(below_q1)
    out = {
        "n_traj": n,
        "rank_median": float(torch.tensor(ranks).median()),
        "frac_below_q1": sum(below_q1) / n,
        "null_frac_below_q1": sum(null_below_q1) / n,
        "null_binomial_q95": 0.25 + 1.645 * math.sqrt(0.25 * 0.75 / n),
        "surrogate_length_ratio_mean": float(torch.tensor(length_ratios).mean()),
    }
    log(
        f"[{label}] 低于Q1比例={out['frac_below_q1']:.2f} "
        f"(空白 {out['null_frac_below_q1']:.2f}, 二项 q95={out['null_binomial_q95']:.2f}), "
        f"观测分位中位={out['rank_median']:.3f}"
    )
    return out


def p2_block(system, cfg: dict, seeds, log) -> dict:
    cfg_p2 = cfg["p2"]
    gen = torch.Generator()
    gen.manual_seed(seeds[0])
    trajs = simulate_continuous(system, cfg_p2, gen)
    log(f"P2 观测轨迹 {tuple(trajs.shape)}")

    results = {
        "true_chain_with_potential": actions_for(
            trajs, system, cfg_p2, system.potential_dyn, gen, "真值臂·含势", log
        ),
        "true_chain_const_potential": actions_for(
            trajs, system, cfg_p2, None, gen, "真值臂·常数势", log
        ),
    }

    # 全链臂：V̂ 用 k 近邻（对参照集），autograd 经固定近邻可微
    ea = cfg_p2["est_arm"]
    gen2 = torch.Generator()
    gen2.manual_seed(seeds[1])
    z_ref = system.simulate_chains(
        cfg_p2["n_chains"], ea["ref_steps"], cfg_p2["dt"], cfg_p2["burn_in"], ea["ref_thin"], gen2
    ).reshape(-1, system.latent_dim)
    x_ref = system.sample_observations(z_ref, gen2)
    log(f"P2 全链臂参照集 {tuple(z_ref.shape)}")

    def v_hat(z):
        return potential.potential_knn(
            system.decoder_mean, system.sigma_dec**2, z, z_ref, x_ref, k=ea["k_potential"]
        )

    sub = trajs[: ea["n_traj"]]
    results["estimated_chain_with_potential"] = actions_for(
        sub, system, cfg_p2, v_hat, gen, "全链臂·含势", log
    )
    results["estimated_chain_const_potential"] = actions_for(
        sub, system, cfg_p2, None, gen, "全链臂·常数势", log
    )
    return results


def p3_block(system, cfg: dict, seeds, log) -> dict:
    cfg_p3 = cfg["p3"]
    gen = torch.Generator()
    gen.manual_seed(seeds[0])
    pool = system.exact_sample(256, gen)
    # 弦长（度量意义用 y 空间欧氏）取中段的端点对
    y_pool = system.phi(pool)
    pairs = []
    chords = []
    for _ in range(512):
        ij = torch.randint(0, pool.shape[0], (2,), generator=gen)
        if int(ij[0]) == int(ij[1]):
            continue
        chords.append(float((y_pool[ij[0]] - y_pool[ij[1]]).norm()))
        pairs.append((int(ij[0]), int(ij[1])))
    chords_t = torch.tensor(chords)
    lo, hi = (
        float(torch.quantile(chords_t, cfg_p3["chord_quantile_band"][0])),
        float(torch.quantile(chords_t, cfg_p3["chord_quantile_band"][1])),
    )
    selected = [p for p, c in zip(pairs, chords) if lo <= c <= hi][: cfg_p3["n_pairs"]]

    residuals, spreads, floor_devs = [], [], []
    inject_devs: dict[str, list[float]] = {str(a): [] for a in cfg_p3["inject_alphas"]}
    tau = torch.linspace(0, 1, cfg_p3["n_segments"] + 1, dtype=torch.float64).unsqueeze(-1)
    for n_done, (i, j) in enumerate(selected):
        z_a, z_b = pool[i], pool[j]
        out = solve_geodesic(
            z_a, z_b, system.metric_true,
            n_segments=cfg_p3["n_segments"], n_starts=cfg_p3["n_starts"], generator=gen,
        )
        residuals.append(out["residual"])
        spreads.append(out["uniqueness_spread"])

        y_a, y_b = system.phi(z_a), system.phi(z_b)
        chord = y_b - y_a
        geo_true = system.phi_inverse(y_a + tau * chord)
        floor_devs.append(
            geodesic_deviation(geo_true, out["path"], system.metric_true)["normalized_area"]
        )
        # 注入横向正弦偏差（y 空间），拉回后对求解测地量偏差
        for alpha in cfg_p3["inject_alphas"]:
            for _ in range(cfg_p3["n_inject_reps"]):
                raw = torch.randn(chord.shape, generator=gen, dtype=torch.float64)
                perp = raw - (raw @ chord) / (chord @ chord) * chord
                perp = perp / perp.norm()
                y_pert = y_a + tau * chord + alpha * float(chord.norm()) * torch.sin(
                    math.pi * tau
                ) * perp
                traj = system.phi_inverse(y_pert)
                inject_devs[str(alpha)].append(
                    geodesic_deviation(traj, out["path"], system.metric_true)["normalized_area"]
                )
        log(f"  P3 端点对 {n_done + 1}/{len(selected)}")

    res_t = torch.tensor(residuals, dtype=torch.float64)
    spread_t = torch.tensor(spreads, dtype=torch.float64)
    floor_t = torch.tensor(floor_devs, dtype=torch.float64)
    floor_q95 = float(torch.quantile(floor_t, 0.95))
    inject_summary = {
        a: {
            "median": float(torch.tensor(v).median()),
            "q05": float(torch.quantile(torch.tensor(v), 0.05)),
        }
        for a, v in inject_devs.items()
    }
    mde = None
    for a in cfg_p3["inject_alphas"]:
        if a > 0 and inject_summary[str(a)]["q05"] > floor_q95:
            mde = a
            break
    out = {
        "n_pairs": len(selected),
        "residual_q95": float(torch.quantile(res_t, 0.95)),
        "residual_max": float(res_t.max()),
        "uniqueness_spread_q99": float(torch.quantile(spread_t, 0.99)),
        "uniqueness_spread_max": float(spread_t.max()),
        "deviation_floor_median": float(floor_t.median()),
        "deviation_floor_q95": floor_q95,
        "injected_deviation": inject_summary,
        "minimal_detectable_alpha": mde,
        "mde_rule": "实测偏差 q05 超过零注入地板 q95 的最小注入幅度（q05 口径使检出以 ≥95% 概率超地板）",
    }
    log(
        f"[P3] 求解残差 q95={out['residual_q95']:.2e}, 唯一性散布 q99={out['uniqueness_spread_q99']:.2e}, "
        f"偏差地板 q95={floor_q95:.4f}, 最小可检测 α={mde}"
    )
    return out


def plot_p2(metrics_p2: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    labels = {
        "true_chain_with_potential": "真值·含势",
        "true_chain_const_potential": "真值·常数势",
        "estimated_chain_with_potential": "全链·含势",
        "estimated_chain_const_potential": "全链·常数势",
    }
    names = list(labels.values())
    fracs = [metrics_p2[k]["frac_below_q1"] for k in labels]
    nulls = [metrics_p2[k]["null_frac_below_q1"] for k in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, fracs, width=0.36, color="#2a78d6", label="观测轨迹")
    ax.bar(xpos + 0.18, nulls, width=0.36, color="#eb6834", label="空白（代理充观测）")
    ax.axhline(0.25, color=muted, linewidth=1, linestyle="--")
    ax.axhline(0.75, color="#52514e", linewidth=1, linestyle=":")
    ax.text(len(names) - 0.5, 0.76, "判定线 75%", color=ink2, fontsize=9)
    ax.set_xticks(xpos, names)
    ax.set_ylabel("低于代理 Q1 的轨迹比例", color=ink2)
    ax.set_title("CAL-P2：Â_OM 分位分离与消融基线", color=ink, fontsize=11)
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


def plot_p3(metrics_p3: dict, alphas: list, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    med = [metrics_p3["injected_deviation"][str(a)]["median"] for a in alphas]
    q05 = [metrics_p3["injected_deviation"][str(a)]["q05"] for a in alphas]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.plot(alphas, med, color="#2a78d6", linewidth=2, marker="o", markersize=6, label="实测偏差中位数")
    ax.plot(alphas, q05, color="#2a78d6", linewidth=1.2, linestyle="--", label="实测偏差 q05")
    ax.axhline(metrics_p3["deviation_floor_q95"], color="#eb6834", linewidth=2,
               linestyle=":", label="零注入地板 q95")
    ax.set_xlabel("注入偏差幅度 α（弦长倍数）", color=ink2)
    ax.set_ylabel("归一化偏差面积", color=ink2)
    ax.set_title("CAL-P3：测地偏差地板与最小可检测差", color=ink, fontsize=11)
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


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    run_dir = create_run_dir("calibration", "cal_p2p3", cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    seeds = guard.family_seeds(cfg["seed_family"], purpose="cal-p2p3")
    params = json.loads((REPO_ROOT / cfg["matched_system"]).read_text(encoding="utf-8"))
    system = system_from_params(params)

    metrics = {"matched_system": cfg["matched_system"]}
    metrics["p2"] = p2_block(system, cfg, seeds, log)
    metrics["p3"] = p3_block(system, cfg, seeds, log)

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_p2(metrics["p2"], run_dir / "p2_separation.png")
    plot_p3(metrics["p3"], cfg["p3"]["inject_alphas"], run_dir / "p3_deviation.png")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
