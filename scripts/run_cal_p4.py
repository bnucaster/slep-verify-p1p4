"""CAL-P4：合成校准系统上全链恢复仿射律与真温度（任务二 2e 建立，任务三 3b
扩展密度三口径，plan_v2 第 5 节第 2 条与第 5 条仪器门分支闭环）。

臂位（采样按臂位模拟一次，三种密度口径共用）：
- main：几何匹配 d=16 系统（2d 产物），两个校准种子各一遍；
- exact：同系统不变测度精确采样（区分离散化偏差与估计器误差）；
- low_dim：同构造 d=6 系统（维数灾对照）。

密度口径（Î = −log p̂ + (1/2)log det ĝ 中 p̂ 的路子）：
- knn：原坐标 kNN（任务二失败口径，保留作对照）；
- knn_std：标准化坐标 kNN（潜坐标尺度跨约 3 个量级的对策）；
- flow：RealNVP 流密度（estimators/flow.py），每臂位在参照集上训练。

另有（用 flow 主口径）：
- null_control：同源噪声对照——采样与势失配 + 共享带噪解码器，管线
  不得拟出仿射；
- manipulation：σ_x 扫描验证 docs/manipulation_signature.md 的签名。

产物 results/calibration/cal_p4/<run_id>/：metrics.json（臂位×口径的
T̂、lack-of-fit、R²、分半）、tolerance_proposal.json、manipulation.json、
affine_main.png、manipulation.png、log.txt。判定字段不在此产生。

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
from slep.estimators.flow import fit_flow_density
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


def sample_arm(system: CalLangevinSystem, seed: int, cfg: dict, sampling: str) -> dict:
    """臂位采样：状态、观测与切分索引；密度口径共用。"""
    gen = torch.Generator()
    gen.manual_seed(seed)
    if sampling == "chain":
        z = system.simulate_chains(
            cfg["chains"], cfg["steps"], cfg["dt"], cfg["burn_in"], cfg["thin"], gen
        )
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
    return {
        "z": z, "x": x, "chain_idx": chain_idx, "time_idx": time_idx,
        "q_idx": perm[: cfg["n_queries"]], "r_idx": perm[cfg["n_queries"] :],
        "gen": gen, "seed": seed,
    }


def log_density_by_method(
    z_ref: torch.Tensor, z_query: torch.Tensor, cfg: dict, method: str, seed: int
) -> torch.Tensor:
    if method == "knn":
        parts = [
            log_density_knn(z_query[i : i + 256], z_ref, k=cfg["k_density"])
            for i in range(0, z_query.shape[0], 256)
        ]
        return torch.cat(parts)
    if method == "knn_std":
        parts = [
            log_density_knn(z_query[i : i + 256], z_ref, k=cfg["k_density"], standardize=True)
            for i in range(0, z_query.shape[0], 256)
        ]
        return torch.cat(parts)
    if method == "flow":
        fcfg = cfg["flow"]
        n_val = max(z_ref.shape[0] // 10, 1000)
        fd, _ = fit_flow_density(
            z_ref[n_val:], z_ref[:n_val], seed=seed,
            n_couplings=fcfg["couplings"], hidden=fcfg["hidden"],
            epochs=fcfg["epochs"], batch_size=fcfg["batch"],
        )
        return fd.log_prob(z_query)
    raise ValueError(f"未知密度口径 {method!r}")


def estimate_arm(
    system: CalLangevinSystem, samples: dict, cfg: dict, method: str
) -> dict[str, torch.Tensor]:
    """全链估计：V̂、log p̂（按口径）、log det ĝ、Î。"""
    z, x = samples["z"], samples["x"]
    q_idx, r_idx = samples["q_idx"], samples["r_idx"]
    z_ref, x_ref, z_query = z[r_idx], x[r_idx], z[q_idx]

    v_parts = [
        potential.potential_knn(
            system.decoder_mean, system.sigma_dec**2,
            z_query[i : i + 256], z_ref, x_ref, k=cfg["k_potential"],
        )
        for i in range(0, z_query.shape[0], 256)
    ]
    v_hat = torch.cat(v_parts)
    log_p = log_density_by_method(z_ref, z_query, cfg, method, samples["seed"])
    g = fisher_pullback_gaussian_batch(system.decoder_mean, z_query, system.sigma_dec**2)
    logdet = torch.linalg.slogdet(g).logabsdet
    return {"v_hat": v_hat, "log_p": log_p, "logdet_g": logdet, "i_hat": -log_p + 0.5 * logdet}


def fit_and_report(system, samples: dict, est: dict, cfg: dict, log, label: str) -> dict:
    rep = affine_fit_report(est["v_hat"], est["i_hat"])
    t_true = system.temperature
    rep["t_true"] = t_true
    rep["t_rel_err"] = abs(rep["temperature_hat"] - t_true) / t_true
    q_idx = samples["q_idx"]
    time_q = samples["time_idx"][q_idx]
    rep["split_time"] = split_half_temperature(est["v_hat"], est["i_hat"], time_q < time_q.median())
    if int(samples["chain_idx"].max()) > 0:
        rep["split_chain"] = split_half_temperature(
            est["v_hat"], est["i_hat"], samples["chain_idx"][q_idx] < cfg["chains"] // 2
        )
    log(
        f"[{label}] T̂={rep['temperature_hat']:.4f} (真值 {t_true}) "
        f"误差 {rep['t_rel_err']:.1%}, R²={rep['r_squared']:.4f}, "
        f"p={rep['p_lack_of_fit']:.3g}, ΔBIC={rep['delta_bic_lin_minus_quad']:.2f}, "
        f"分半 {rep['split_time']['relative_gap']:.1%}"
    )
    return rep


def perturbed_decoder_system(system, scale: float, gen: torch.Generator) -> CalLangevinSystem:
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


def null_control(system, seed: int, cfg: dict, method: str, log) -> dict:
    """采样与势失配 + 共享解码器噪声：仿射不得被拟出（主口径）。"""
    gen = torch.Generator()
    gen.manual_seed(seed)
    d = system.latent_dim
    w_scr = torch.flip(system.well_w, dims=[0])
    std = torch.sqrt(system.temperature / w_scr)
    offset = 0.8 * torch.ones(d, dtype=torch.float64)
    n_total = cfg["chains"] * ((cfg["steps"] - cfg["burn_in"]) // cfg["thin"] + 1)
    u = system.well_u0 + offset + std * torch.randn((n_total, d), generator=gen, dtype=torch.float64)
    z = system.z_of_u(u)
    x = system.sample_observations(z, gen)

    est_system = perturbed_decoder_system(system, cfg["null_control"]["decoder_noise_scale"], gen)
    perm = torch.randperm(z.shape[0], generator=gen)
    samples = {
        "z": z, "x": x, "chain_idx": torch.zeros(z.shape[0], dtype=torch.long),
        "time_idx": torch.arange(z.shape[0]),
        "q_idx": perm[: cfg["n_queries"]], "r_idx": perm[cfg["n_queries"] :],
        "gen": gen, "seed": seed,
    }
    est = estimate_arm(est_system, samples, cfg, method)
    rep = affine_fit_report(est["v_hat"], est["i_hat"])
    log(
        f"[null|{method}] R²={rep['r_squared']:.4f}, p={rep['p_lack_of_fit']:.3g}, "
        f"ΔBIC={rep['delta_bic_lin_minus_quad']:.2f}（期望：拒绝仿射或低 R²）"
    )
    return rep


def manipulation_sweep(system, samples: dict, est_base: dict, base_rep: dict, cfg: dict, log) -> dict:
    """σ_x 扫描：状态冻结、观测重加噪、只重算 V̂（签名数值验证）。"""
    z, q_idx, r_idx, gen = samples["z"], samples["q_idx"], samples["r_idx"], samples["gen"]
    i_hat = est_base["i_hat"]
    d_obs = system.obs_dim
    rows = []
    v_true_q = system.potential_estimand(z[q_idx])
    var_v_true = float(v_true_q.var())
    for sigma_x in cfg["sigma_x_sweep"]:
        sys_x = system_from_params({**system_to_params(system), "sigma_x": sigma_x})
        x_new = sys_x.sample_observations(z, gen)
        v_parts = [
            potential.potential_knn(
                system.decoder_mean, system.sigma_dec**2,
                z[q_idx][i : i + 256], z[r_idx], x_new[r_idx], k=cfg["k_potential"],
            )
            for i in range(0, q_idx.shape[0], 256)
        ]
        v_hat = torch.cat(v_parts)
        rep = affine_fit_report(v_hat, i_hat)
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
                "v_mean_shift_measured": float(v_hat.mean() - est_base["v_hat"].mean()),
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
    return {"base_t_hat": base_rep["temperature_hat"], "sweep": rows}


def plot_affine(est: dict, rep: dict, path, t_true: float, subtitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    v = est["v_hat"].numpy()
    i = est["i_hat"].numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.scatter(v, i, s=4, color="#2a78d6", alpha=0.25, linewidths=0)
    vx = np.linspace(v.min(), v.max(), 50)
    ax.plot(vx, rep["intercept"] + rep["slope"] * vx, color="#eb6834", linewidth=2,
            label=f"拟合斜率 {rep['slope']:.3f} → T̂={rep['temperature_hat']:.3f}（真值 {t_true}）")
    ax.set_xlabel("V̂（nats）", color=ink2)
    ax.set_ylabel("Î（体积校正后，nats）", color=ink2)
    ax.set_title(f"CAL-P4 主臂：Î–V̂ 散点与仿射拟合（{subtitle}）", color=ink, fontsize=11)
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
    methods = cfg["density_methods"]
    primary = cfg["primary_density_method"]
    metrics: dict = {
        "matched_system": cfg["matched_system"],
        "density_methods": methods,
        "primary_density_method": primary,
        "arms": {},
    }

    # 低维对照系统
    arms: list[tuple[str, CalLangevinSystem, dict]] = []
    for seed in seeds:
        arms.append((f"main-seed{seed}", system, sample_arm(system, seed, cfg, "chain")))
    arms.append(("exact", system, sample_arm(system, seeds[0], cfg, "exact")))
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
        arms.append(("low-dim(d=6)", low_system, sample_arm(low_system, seeds[0], cfg, "chain")))

    primary_main_est, primary_main_rep, primary_main_samples = None, None, None
    for label, arm_system, samples in arms:
        for method in methods:
            est = estimate_arm(arm_system, samples, cfg, method)
            rep = fit_and_report(arm_system, samples, est, cfg, log, f"{label}|{method}")
            metrics["arms"][f"{label}|{method}"] = rep
            if label == f"main-seed{seeds[0]}" and method == primary:
                primary_main_est, primary_main_rep, primary_main_samples = est, rep, samples

    metrics["null_control"] = null_control(system, seeds[0], cfg, primary, log)
    manip = manipulation_sweep(system, primary_main_samples, primary_main_est, primary_main_rep, cfg, log)
    (run_dir / "manipulation.json").write_text(
        json.dumps(manip, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rel_errs = {
        name: arm["t_rel_err"] for name, arm in metrics["arms"].items() if "t_rel_err" in arm
    }
    primary_errs = {k: v for k, v in rel_errs.items() if k.endswith(f"|{primary}")}
    tolerance = {
        "observed_t_rel_err": rel_errs,
        "observed_split_time_gap": {
            name: arm["split_time"]["relative_gap"]
            for name, arm in metrics["arms"].items()
            if "split_time" in arm
        },
        "observed_r_squared": {name: arm["r_squared"] for name, arm in metrics["arms"].items()},
        "proposal": {
            "based_on_method": primary,
            "t_rel_err_tolerance": 2.0 * max(primary_errs.values()),
            "derivation": (
                "容差 = 2 × 主口径各臂位最大相对误差（安全系数 2 为提案值，"
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
    plot_affine(
        primary_main_est, primary_main_rep, run_dir / "affine_main.png",
        system.temperature, f"密度口径 {primary}",
    )
    plot_manipulation(manip, system.sigma_x, run_dir / "manipulation.png")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
