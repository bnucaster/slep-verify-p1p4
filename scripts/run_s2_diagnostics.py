"""S2 四诊断图（任务六 6b，plan_v2 第 6 节）：末检查点、开发族 5 种子。

图一 V̂ 地形梯度分布：查询点处 ‖∇_ĝV̂‖_ĝ = sqrt(∂V̂ᵀ ĝ⁻¹ ∂V̂) 的分布、
  高/低三分位中位数之比与自助置信区间 → P3 有无统计功效；
图二 Î–V̂ 散点：flow 密度 + 体积校正，含 / 不含校正双版本拟合，
  R² 对打乱标签零假设 q95 → 校正是否翻转形状、结构是否存在；
图三 Â_OM 重叠度：观测隐轨迹段对度量口径主代理的标准化间隙与
  低于 Q1 比例（T̂ 取 1，排序对公共正因子不变）→ P2 可分辨性；
图四 经验漂移梯度占比：kNN 漂移回归 + ĝ 度量下标量势拟合（留出
  口径）→ P2 触及变分结构还是漂移结构。

产物 results/description/s2_diagnostics/<out_campaign>/：
s<种子>/diagnostics.json、campaign 汇总 summary.json 与 diagnostics.png。

用法：.venv/Scripts/python.exe scripts/run_s2_diagnostics.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators import potential
from slep.estimators.drift import estimate_drift_knn, gradient_fraction
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.estimators.om_action import om_action
from slep.protocols.affine import affine_fit_report
from slep.protocols.surrogates import smooth_random_surrogate_metric
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "s2_diagnostics.yaml"


def load_model(train_cfg: dict, seed: int) -> S2WorldModel:
    ckpt = torch.load(
        REPO_ROOT / "results" / "description" / "s2_train" / train_cfg["campaign"]
        / f"s{seed}" / "checkpoints" / f"ckpt_{train_cfg['train_steps']:06d}.pt",
        weights_only=True,
    )
    model = S2WorldModel(
        train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["embed_dim"],
        train_cfg["hidden_dim"], train_cfg["sigma_dec"], train_cfg.get("goal_sigma_dec"),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def diagnose_seed(cfg, train_cfg, seed: int, out_dir, log) -> dict:
    seed_dir = out_dir / f"s{seed}"
    seed_dir.mkdir(exist_ok=True)
    if (seed_dir / "diagnostics.json").exists():
        log(f"跳过已完成 s{seed}")
        return json.loads((seed_dir / "diagnostics.json").read_text(encoding="utf-8"))

    t0 = time.time()
    model = load_model(train_cfg, seed)
    obs_var = model.obs_var.double()
    rng = np.random.default_rng(seed + 888_000)
    obs_np, act_np = gw.collect_rollouts(
        cfg["n_episodes"], cfg["episode_len"], train_cfg["maze_cells"], train_cfg["view"], rng
    )
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)  # (E, T, H)
    bi = cfg["burn_in"]
    h_pool = hs[:, bi:-1].reshape(-1, model.hidden_dim).double()
    o_next_pool = obs[:, bi + 1 :].reshape(-1, train_cfg["obs_dim"]).double()
    h_next_pool = hs[:, bi + 1 :].reshape(-1, model.hidden_dim).double()

    perm = torch.from_numpy(rng.permutation(h_pool.shape[0]))
    idx_ref = perm[: cfg["n_ref"]]
    idx_q = perm[cfg["n_ref"] : cfg["n_ref"] + cfg["n_query"]]
    idx_flow = perm[cfg["n_ref"] + cfg["n_query"] :
                    cfg["n_ref"] + cfg["n_query"] + cfg["n_flow_train"] + cfg["n_flow_val"]]
    h_ref, o_ref = h_pool[idx_ref], o_next_pool[idx_ref]
    h_q = h_pool[idx_q]

    def dec(z):
        return model.decoder_mean(z.float()).double()

    # 图一：V̂ 与其 ĝ 度量下梯度范数
    g_q = fisher_pullback_gaussian_batch(dec, h_q, obs_var).detach()
    logdet_g = torch.linalg.slogdet(g_q).logabsdet
    h_req = h_q.detach().requires_grad_(True)
    v_parts = []
    for i in range(0, h_req.shape[0], 256):
        v_parts.append(potential.potential_knn(
            dec, obs_var, h_req[i : i + 256], h_ref, o_ref, k=cfg["k_potential"]))
    v_q = torch.cat(v_parts)
    (grad_v,) = torch.autograd.grad(v_q.sum(), h_req)
    nat = torch.linalg.solve(g_q, grad_v.unsqueeze(-1)).squeeze(-1)
    grad_norm = torch.einsum("ti,tij,tj->t", nat, g_q, nat).clamp_min(0).sqrt().detach()
    v_q = v_q.detach()
    q_lo, q_hi = torch.quantile(grad_norm, 1 / 3), torch.quantile(grad_norm, 2 / 3)
    lo_med = float(grad_norm[grad_norm <= q_lo].median())
    hi_med = float(grad_norm[grad_norm >= q_hi].median())
    boots = []
    gen_b = torch.Generator()
    gen_b.manual_seed(seed)
    for _ in range(cfg["bootstrap"]):
        bidx = torch.randint(0, grad_norm.shape[0], (grad_norm.shape[0],), generator=gen_b)
        gb = grad_norm[bidx]
        ql, qh = torch.quantile(gb, 1 / 3), torch.quantile(gb, 2 / 3)
        boots.append(float(gb[gb >= qh].median() / gb[gb <= ql].median().clamp_min(1e-30)))
    fig1 = {
        "grad_norm_quantiles": [float(torch.quantile(grad_norm, q)) for q in (0.1, 1 / 3, 0.5, 2 / 3, 0.9)],
        "hi_lo_median_ratio": hi_med / max(lo_med, 1e-30),
        "ratio_ci90": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))],
    }

    # 图二：Î–V̂ 散点（flow 密度），含/不含体积校正 + 打乱标签零假设
    n_ft = cfg["n_flow_train"]
    fd, _ = fit_flow_density(
        h_pool[idx_flow[:n_ft]], h_pool[idx_flow[n_ft:]], seed=seed,
        n_couplings=cfg["flow"]["couplings"], hidden=cfg["flow"]["hidden"],
        epochs=cfg["flow"]["epochs"], batch_size=cfg["flow"]["batch"],
    )
    log_p = fd.log_prob(h_q)
    i_corr = -log_p + 0.5 * logdet_g
    i_raw = -log_p
    fit_corr = affine_fit_report(v_q, i_corr)
    fit_raw = affine_fit_report(v_q, i_raw)
    gen_n = torch.Generator()
    gen_n.manual_seed(seed + 1)
    null_r2 = []
    for _ in range(cfg["n_shuffle_null"]):
        p = torch.randperm(i_corr.shape[0], generator=gen_n)
        null_r2.append(affine_fit_report(v_q, i_corr[p])["r_squared"])
    fig2 = {
        "corrected": {k: fit_corr[k] for k in ("slope", "r_squared", "p_lack_of_fit",
                                               "curvature_effect_ratio", "temperature_hat")},
        "uncorrected": {k: fit_raw[k] for k in ("slope", "r_squared")},
        "null_r2_q95": float(np.quantile(null_r2, 0.95)),
    }

    # 图三：Â_OM 重叠度（观测段对度量口径代理，T̂=1 排序不变）
    def v_fn(z):
        return potential.potential_knn(dec, obs_var, z, h_ref, o_ref, k=cfg["k_potential"])

    # 诊断层正则：代理可游走到解码器饱和的度量奇异区，g⁻¹∇V 与白化
    # 分解在该区数值爆炸；加相对地板 floor·I（floor = 1e-3 × 查询点度量
    # 特征值中位数）。确证阶段由支撑度剔除承接此情形，此处只保证诊断
    # 可算并记录口径。
    g_floor = 1e-3 * float(torch.linalg.eigvalsh(g_q).median())
    eye_h = torch.eye(model.hidden_dim, dtype=torch.float64)

    def g_fn(z):
        return fisher_pullback_gaussian_batch(dec, z, obs_var) + g_floor * eye_h

    ep_idx = rng.permutation(cfg["n_episodes"])[: cfg["n_traj"]]
    gen_s = torch.Generator()
    gen_s.manual_seed(seed + 2)
    gaps, below_q1 = [], []
    for k, e in enumerate(ep_idx):
        traj = hs[e, bi : bi + cfg["traj_len"] + 1].double()
        a_obs = float(om_action(traj, 1.0, v_fn, g_fn, 1.0, "mid"))
        a_surr = []
        for _ in range(cfg["n_surrogates"]):
            sur, _ = smooth_random_surrogate_metric(traj, g_fn, gen_s)
            a_surr.append(float(om_action(sur, 1.0, v_fn, g_fn, 1.0, "mid")))
        a_t = torch.tensor(a_surr, dtype=torch.float64)
        iqr = float(torch.quantile(a_t, 0.75) - torch.quantile(a_t, 0.25))
        gaps.append((float(a_t.median()) - a_obs) / max(iqr, 1e-30))
        below_q1.append(a_obs < float(torch.quantile(a_t, 0.25)))
        if (k + 1) % 25 == 0:
            log(f"  s{seed} 图三 {k + 1}/{cfg['n_traj']} ({time.time() - t0:.0f}s)")
    fig3 = {
        "gap_standardized_median": float(torch.tensor(gaps).median()),
        "frac_below_q1": sum(below_q1) / len(below_q1),
        "null_binomial_q95": 0.25 + 1.645 * float(np.sqrt(0.1875 / len(below_q1))),
    }

    # 图四：经验漂移梯度占比（转移 dt=1）
    idx_drift = perm[-cfg["n_drift_eval"]:]
    drift = estimate_drift_knn(h_pool, h_next_pool, 1.0, h_pool[idx_drift], k=cfg["k_drift"])
    frac = gradient_fraction(h_pool[idx_drift], drift, g_fn, seed=seed)
    fig4 = {"gradient_fraction": frac["fraction"],
            "residual_ratio_train": frac["residual_ratio_train"]}

    out = {"seed": seed, "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
           "n_states": int(h_pool.shape[0]), "wall_seconds": round(time.time() - t0)}
    (seed_dir / "diagnostics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez(seed_dir / "fig2_scatter.npz", v=v_q.numpy(), i_corr=i_corr.numpy(),
             i_raw=i_raw.numpy(), grad_norm=grad_norm.numpy(), gaps=np.array(gaps))
    log(
        f"完成 s{seed}（{out['wall_seconds']}s）：梯度比 {fig1['hi_lo_median_ratio']:.2f} "
        f"CI90 {fig1['ratio_ci90']}, Î–V̂ 斜率 {fig2['corrected']['slope']:.3f} "
        f"R² {fig2['corrected']['r_squared']:.3f} (null q95 {fig2['null_r2_q95']:.3f}), "
        f"P2 间隙 {fig3['gap_standardized_median']:.2f} 低于Q1 {fig3['frac_below_q1']:.2f}, "
        f"梯度占比 {fig4['gradient_fraction']:.2f}"
    )
    return out


def plot_campaign(results: list[dict], out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    blue, orange = "#2a78d6", "#eb6834"
    seeds = [r["seed"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), dpi=150)
    fig.patch.set_facecolor(surface)

    ax = axes[0, 0]
    ratios = [r["fig1"]["hi_lo_median_ratio"] for r in results]
    ci_lo = [r["fig1"]["ratio_ci90"][0] for r in results]
    ci_hi = [r["fig1"]["ratio_ci90"][1] for r in results]
    ax.errorbar(seeds, ratios,
                yerr=[np.array(ratios) - ci_lo, np.array(ci_hi) - ratios],
                fmt="o", color=blue, markersize=7, capsize=4, linewidth=1.6)
    ax.axhline(1.0, color=muted, linestyle="--", linewidth=1)
    ax.set_xlabel("种子", color=ink2)
    ax.set_ylabel("高/低梯度三分位中位数比", color=ink2)
    ax.set_title("图一 V̂ 地形梯度分位差（P3 功效）", color=ink, fontsize=11)

    ax = axes[0, 1]
    data = np.load(out_dir / f"s{seeds[0]}" / "fig2_scatter.npz")
    ax.scatter(data["v"], data["i_corr"], s=3, color=blue, alpha=0.3, linewidths=0,
               label="含体积校正")
    ax.scatter(data["v"], data["i_raw"], s=3, color=orange, alpha=0.2, linewidths=0,
               label="不含校正")
    ax.set_xlabel("V̂（nats）", color=ink2)
    ax.set_ylabel("Î（nats）", color=ink2)
    r0 = results[0]["fig2"]
    ax.set_title(
        f"图二 Î–V̂ 散点（种子 {seeds[0]}）：校正版斜率 {r0['corrected']['slope']:.2f}, "
        f"R² {r0['corrected']['r_squared']:.2f}", color=ink, fontsize=11)
    ax.legend(frameon=False, labelcolor=ink2, markerscale=3)

    ax = axes[1, 0]
    gaps = [r["fig3"]["gap_standardized_median"] for r in results]
    fracs = [r["fig3"]["frac_below_q1"] for r in results]
    null_q95 = results[0]["fig3"]["null_binomial_q95"]
    ax.bar(seeds, fracs, width=0.5, color=blue)
    ax.axhline(0.75, color=ink2, linestyle=":", linewidth=1.2)
    ax.axhline(null_q95, color=orange, linestyle="--", linewidth=1.2)
    ax.text(seeds[-1] + 0.3, 0.76, "判定线 75%", color=ink2, fontsize=8)
    ax.text(seeds[-1] + 0.3, null_q95 + 0.01, "空白 q95", color=orange, fontsize=8)
    for s, f, g in zip(seeds, fracs, gaps):
        ax.text(s, f + 0.03, f"间隙 {g:.1f}", ha="center", color=ink2, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("种子", color=ink2)
    ax.set_ylabel("低于代理 Q1 的轨迹比例", color=ink2)
    ax.set_title("图三 Â_OM 重叠度（P2 可分辨性；柱上注标准化间隙中位）",
                 color=ink, fontsize=11)

    ax = axes[1, 1]
    fracs4 = [r["fig4"]["gradient_fraction"] for r in results]
    ax.bar(seeds, fracs4, width=0.5, color=blue)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("种子", color=ink2)
    ax.set_ylabel("梯度占比（留出口径）", color=ink2)
    ax.set_title("图四 经验漂移梯度占比（P2 归因）", color=ink, fontsize=11)

    for ax in axes.flat:
        ax.set_facecolor(surface)
        ax.grid(True, axis="y", color=muted, alpha=0.22, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelcolor=ink2)
    fig.suptitle("S2 描述阶段四诊断图（开发族，探索性）", color=ink, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", facecolor=surface)
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / "configs" / "s2_train.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    out_dir = create_campaign_dir("description", "s2_diagnostics", cfg["out_campaign"], cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds = guard.family_seeds("development", purpose="s2-diagnostics")
    results = []
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="s2-diagnostics")
        results.append(diagnose_seed(cfg, train_cfg, seed, out_dir, log))
    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_campaign(results, out_dir)
    log("四诊断图批次完成")


if __name__ == "__main__":
    main()
