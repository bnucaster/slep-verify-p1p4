"""CAL-P4：合成校准系统上全链恢复仿射律与真温度（任务二 2e，plan_v2 第 5 节第 2 条）。

臂位：
- main：几何匹配 d=16 系统（2d 产物），多链 Langevin 采样 → 观测生成 →
  V̂（k 近邻）、p̂（kNN 密度）、ĝ（拉回）、Î（体积校正）→ 仿射拟合 →
  T̂ = 1/斜率。按两个校准种子各跑一遍（链与观测随机性换种子；系统参数
  固定为 2d 拟合结果）。
- exact：同系统、不变测度精确采样（区分离散化偏差与估计器误差）。
- low_dim：同构造 d=6 系统（维数灾对照，链路正确性的低维见证）。
- null_control：同源噪声对照——采样分布与势失配（u 空间刚度反序 +
  中心偏移），V̂ 与 ĝ 共用一套带噪解码器副本；管线不得在此拟出仿射。
- manipulation：观测噪声 σ_x 扫描（状态与解码器冻结，仅重加噪观测、
  重算 V̂），对 docs/manipulation_signature.md 的预期签名做数值验证。

产物 results/calibration/cal_p4/<run_id>/：metrics.json（各臂 T̂、
lack-of-fit、R²、分半一致性）、tolerance_proposal.json（误差带→P4 容差
公式化提案，冻结前为提案）、manipulation.json、affine_main.png、
manipulation.png、log.txt。判定字段不在此产生（judge 属任务七）。

用法：.venv/Scripts/python.exe scripts/run_cal_p4.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators import potential
from slep.estimators.density import log_density_knn
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.affine import affine_fit_report, split_half_temperature
from slep.systems.cal_langevin import (
    CalLangevinSystem,
    build_matched_system,
    system_from_params,
    system_to_params,
)
from slep.utils.runs import REPO_ROOT, create_run_dir

CONFIG_FILE = REPO_ROOT / "configs" / "cal_p4.yaml"


def estimate_chain(
    system: CalLangevinSystem,
    z_ref: torch.Tensor,
    x_ref: torch.Tensor,
    z_query: torch.Tensor,
    k_potential: int,
    k_density: int,
    chunk: int = 256,
) -> dict[str, torch.Tensor]:
    """全链估计：V̂、log p̂、log det ĝ、Î。查询点须不在参照集内。"""
    v_parts, logp_parts = [], []
    for i in range(0, z_query.shape[0], chunk):
        zq = z_query[i : i + chunk]
        v_parts.append(
            potential.potential_knn(
                system.decoder_mean, system.sigma_dec**2, zq, z_ref, x_ref, k=k_potential
            )
        )
        logp_parts.append(log_density_knn(zq, z_ref, k=k_density))
    v_hat = torch.cat(v_parts)
    log_p = torch.cat(logp_parts)
    g = fisher_pullback_gaussian_batch(
        system.decoder_mean, z_query, system.sigma_dec**2, chunk_size=chunk
    )
    logdet = torch.linalg.slogdet(g).logabsdet
    return {"v_hat": v_hat, "log_p": log_p, "logdet_g": logdet, "i_hat": -log_p + 0.5 * logdet}


def run_arm(
    system: CalLangevinSystem,
    seed: int,
    cfg: dict,
    log,
    label: str,
    sampling: str = "chain",
) -> dict:
    """一个臂位的完整流程：采样 → 观测 → 估计 → 拟合 → 分半。"""
    gen = torch.Generator()
    gen.manual_seed(seed)
    if sampling == "chain":
        z = system.simulate_chains(
            cfg["chains"], cfg["steps"], cfg["dt"], cfg["burn_in"], cfg["thin"], gen
        )  # (C, m, d)
        n_chains, m, d = z.shape
        chain_idx = torch.arange(n_chains).unsqueeze(1).expand(n_chains, m).reshape(-1)
        time_idx = torch.arange(m).unsqueeze(0).expand(n_chains, m).reshape(-1)
        z = z.reshape(-1, d)
    else:
        n_total = cfg["chains"] * ((cfg["steps"] - cfg["burn_in"]) // cfg["thin"] + 1)
        z = system.exact_sample(n_total, gen)
        chain_idx = torch.zeros(z.shape[0], dtype=torch.long)
        time_idx = torch.arange(z.shape[0])
    x = system.sample_observations(z, gen)

    perm = torch.randperm(z.shape[0], generator=gen)
    q_idx, r_idx = perm[: cfg["n_queries"]], perm[cfg["n_queries"] :]
    est = estimate_chain(
        system, z[r_idx], x[r_idx], z[q_idx], cfg["k_potential"], cfg["k_density"]
    )

    rep = affine_fit_report(est["v_hat"], est["i_hat"])
    t_true = system.temperature
    rep["t_true"] = t_true
    rep["t_rel_err"] = abs(rep["temperature_hat"] - t_true) / t_true
    rep["split_time"] = split_half_temperature(
        est["v_hat"], est["i_hat"], time_idx[q_idx] < time_idx[q_idx].median()
    )
    if sampling == "chain":
        rep["split_chain"] = split_half_temperature(
            est["v_hat"], est["i_hat"], chain_idx[q_idx] < cfg["chains"] // 2
        )
    log(
        f"[{label}] T̂={rep['temperature_hat']:.4f} (真值 {t_true}) "
        f"相对误差 {rep['t_rel_err']:.1%}, R²={rep['r_squared']:.4f}, "
        f"lack-of-fit p={rep['p_lack_of_fit']:.3g}, ΔBIC={rep['delta_bic_lin_minus_quad']:.2f}, "
        f"分半(时间)={rep['split_time']['relative_gap']:.1%}"
    )
    return {"report": rep, "est": est, "z": z, "x": x, "q_idx": q_idx, "r_idx": r_idx, "gen": gen}


def perturbed_decoder_system(system: CalLangevinSystem, scale: float, gen: torch.Generator) -> CalLangevinSystem:
    """带噪解码器副本：s 与 B 同一实现噪声，V̂ 与 ĝ 共用（同源噪声注入）。"""
    params = system_to_params(system)
    s = torch.tensor(params["s"], dtype=torch.float64)
    s_noisy = s * (1 + scale * torch.randn(s.shape, generator=gen, dtype=torch.float64))
    basis = torch.tensor(params["basis"], dtype=torch.float64)
    noise = scale * torch.randn(basis.shape, generator=gen, dtype=torch.float64)
    basis_noisy, _ = torch.linalg.qr(basis + noise)
    params["s"] = s_noisy.abs().tolist()
    params["basis"] = basis_noisy.tolist()
    return system_from_params(params)


def null_control(system: CalLangevinSystem, seed: int, cfg: dict, log) -> dict:
    """采样与势失配 + 共享解码器噪声：仿射不得被拟出。"""
    gen = torch.Generator()
    gen.manual_seed(seed)
    nc = cfg["null_control"]
    d = system.latent_dim
    # u 空间刚度反序 + 中心偏移的高斯（与势的 Boltzmann 测度失配）
    w_scr = torch.flip(system.well_w, dims=[0])
    std = torch.sqrt(system.temperature / w_scr)
    offset = 0.8 * torch.ones(d, dtype=torch.float64)
    n_total = cfg["chains"] * ((cfg["steps"] - cfg["burn_in"]) // cfg["thin"] + 1)
    u = system.well_u0 + offset + std * torch.randn((n_total, d), generator=gen, dtype=torch.float64)
    z = system.z_of_u(u)
    x = system.sample_observations(z, gen)

    est_system = perturbed_decoder_system(system, nc["decoder_noise_scale"], gen)
    perm = torch.randperm(z.shape[0], generator=gen)
    q_idx, r_idx = perm[: cfg["n_queries"]], perm[cfg["n_queries"] :]
    est = estimate_chain(
        est_system, z[r_idx], x[r_idx], z[q_idx], cfg["k_potential"], cfg["k_density"]
    )
    rep = affine_fit_report(est["v_hat"], est["i_hat"])
    log(
        f"[null] R²={rep['r_squared']:.4f}, lack-of-fit p={rep['p_lack_of_fit']:.3g}, "
        f"ΔBIC={rep['delta_bic_lin_minus_quad']:.2f}（期望：拒绝仿射或低 R²）"
    )
    return rep


def manipulation_sweep(system: CalLangevinSystem, arm: dict, cfg: dict, log) -> dict:
    """σ_x 扫描：状态冻结、观测重加噪、只重算 V̂；对签名推导做数值验证。"""
    z, q_idx, r_idx, gen = arm["z"], arm["q_idx"], arm["r_idx"], arm["gen"]
    i_hat = arm["est"]["i_hat"]
    base = arm["report"]
    d_obs = system.obs_dim
    rows = []
    v_true_q = system.potential_estimand(z[q_idx])
    var_v_true = float(v_true_q.var())
    for sigma_x in cfg["sigma_x_sweep"]:
        sys_x = system_from_params({**system_to_params(system), "sigma_x": sigma_x})
        x_new = sys_x.sample_observations(z, gen)
        v_parts = []
        for i in range(0, q_idx.shape[0], 256):
            zq = z[q_idx][i : i + 256]
            v_parts.append(
                potential.potential_knn(
                    system.decoder_mean, system.sigma_dec**2, zq, z[r_idx],
                    x_new[r_idx], k=cfg["k_potential"],
                )
            )
        v_hat = torch.cat(v_parts)
        rep = affine_fit_report(v_hat, i_hat)
        # 预期截距/均值移动 Δ 与斜率衰减 λ（docs/manipulation_signature.md）
        delta_pred = d_obs * (sigma_x**2 - system.sigma_x**2) / (2 * system.sigma_dec**2)
        e_msq = 2 * system.sigma_dec**2 * system.potential_dyn(z[q_idx])
        var_noise = float(
            ((sigma_x**2 * e_msq + d_obs * sigma_x**4 / 2) / (system.sigma_dec**4 * cfg["k_potential"])).mean()
        )
        lam = var_v_true / (var_v_true + var_noise)
        rows.append(
            {
                "sigma_x": sigma_x,
                "t_hat": rep["temperature_hat"],
                "t_pred_attenuation": system.temperature / lam,
                "v_mean_shift_measured": float(v_hat.mean() - arm["est"]["v_hat"].mean()),
                "v_mean_shift_pred": delta_pred,
                "slope": rep["slope"],
                "r_squared": rep["r_squared"],
                "p_lack_of_fit": rep["p_lack_of_fit"],
            }
        )
        log(
            f"[manip σ_x={sigma_x}] T̂={rep['temperature_hat']:.4f} "
            f"(衰减预测 {system.temperature / lam:.4f}), "
            f"V̂ 移动 {rows[-1]['v_mean_shift_measured']:.3f} (预测 {delta_pred:.3f})"
        )
    return {"base_t_hat": base["temperature_hat"], "sweep": rows}


def plot_affine(arm: dict, path, t_true: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    v = arm["est"]["v_hat"].numpy()
    i = arm["est"]["i_hat"].numpy()
    rep = arm["report"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.scatter(v, i, s=4, color="#2a78d6", alpha=0.25, linewidths=0)
    vx = np.linspace(v.min(), v.max(), 50)
    ax.plot(vx, rep["intercept"] + rep["slope"] * vx, color="#eb6834", linewidth=2,
            label=f"拟合斜率 {rep['slope']:.3f} → T̂={rep['temperature_hat']:.3f}（真值 {t_true}）")
    ax.set_xlabel("V̂（nats）", color=ink2)
    ax.set_ylabel("Î（体积校正后，nats）", color=ink2)
    ax.set_title("CAL-P4 主臂：Î–V̂ 散点与仿射拟合", color=ink, fontsize=11)
    ax.grid(True, color=muted, alpha=0.2, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def plot_manipulation(manip: dict, base_sigma_x: float, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    xs = [base_sigma_x] + [r["sigma_x"] for r in manip["sweep"]]
    measured = [manip["base_t_hat"]] + [r["t_hat"] for r in manip["sweep"]]
    predicted = [manip["base_t_hat"]] + [r["t_pred_attenuation"] for r in manip["sweep"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.plot(xs, measured, color="#2a78d6", linewidth=2, marker="o", markersize=6, label="实测 T̂")
    ax.plot(xs, predicted, color="#eb6834", linewidth=2, marker="s", markersize=5,
            linestyle="--", label="衰减预测 T/λ")
    ax.set_xlabel("观测噪声 σ_x（状态冻结，仅重加噪）", color=ink2)
    ax.set_ylabel("T̂", color=ink2)
    ax.set_title("操纵签名验证：噪声调大后 T̂ 的判向", color=ink, fontsize=11)
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
    run_dir = create_run_dir("calibration", "cal_p4", cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    seeds = guard.family_seeds(cfg["seed_family"], purpose="cal-p4")
    params = json.loads((REPO_ROOT / cfg["matched_system"]).read_text(encoding="utf-8"))
    system = system_from_params(params)
    metrics: dict = {"matched_system": cfg["matched_system"], "arms": {}}

    main_arms = {}
    for seed in seeds:
        arm = run_arm(system, seed, cfg, log, label=f"main-seed{seed}")
        main_arms[seed] = arm
        metrics["arms"][f"main_seed{seed}"] = {
            k: v for k, v in arm["report"].items() if not isinstance(v, torch.Tensor)
        }

    arm_exact = run_arm(system, seeds[0], cfg, log, label="exact", sampling="exact")
    metrics["arms"]["exact"] = arm_exact["report"]

    if cfg["low_dim_arm"]["enabled"]:
        gen = torch.Generator()
        gen.manual_seed(seeds[0])
        la = cfg["low_dim_arm"]
        d_low = len(la["target_eig"])
        well_w = np.exp(np.linspace(math.log(0.5), math.log(2.0), d_low)).tolist()
        low_system, _ = build_matched_system(
            la["target_eig"], la["target_logdet_std"], la["obs_dim"],
            system.sigma_dec, system.sigma_x, well_w, system.temperature, gen,
        )
        arm_low = run_arm(low_system, seeds[0], cfg, log, label="low-dim(d=6)")
        metrics["arms"]["low_dim"] = arm_low["report"]

    metrics["null_control"] = null_control(system, seeds[0], cfg, log)
    manip = manipulation_sweep(system, main_arms[seeds[0]], cfg, log)
    (run_dir / "manipulation.json").write_text(
        json.dumps(manip, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rel_errs = {
        name: arm["t_rel_err"]
        for name, arm in metrics["arms"].items()
        if "t_rel_err" in arm
    }
    tolerance = {
        "observed_t_rel_err": rel_errs,
        "observed_split_time_gap": {
            name: arm["split_time"]["relative_gap"]
            for name, arm in metrics["arms"].items()
            if "split_time" in arm
        },
        "observed_r_squared": {name: arm["r_squared"] for name, arm in metrics["arms"].items()},
        "proposal": {
            "t_rel_err_tolerance": 2.0 * max(rel_errs.values()),
            "derivation": (
                "容差 = 2 × 校准各臂位最大相对误差（安全系数 2 为提案值，"
                "冻结前与描述阶段种子间方差合并复核；plan_v2 第 5 节第 2 条）"
            ),
            "status": "提案：待任务四仪器门总验收与任务七冻结",
        },
    }
    (run_dir / "tolerance_proposal.json").write_text(
        json.dumps(tolerance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    plot_affine(main_arms[seeds[0]], run_dir / "affine_main.png", system.temperature)
    plot_manipulation(manip, system.sigma_x, run_dir / "manipulation.png")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
