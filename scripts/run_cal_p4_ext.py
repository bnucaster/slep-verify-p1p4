"""CAL-P4-EXT（选项 B）：极端各向异性几何下的全链恢复检验。

动机：评估族 20k 步 S2 模型 log10 条件数中位 9.2–9.7，超出校准已证
可行域（5.74，定标于 6k 步谱）。本实验把几何匹配目标换成 20k 步实测
谱重走校准链，回答"仪器在该几何下是否仍恢复已知真值"。两种结局都
有效：恢复成立 → 有资格提出几何门扩展（协议 v1.2 披露修订）；恢复
失败 → 挂门被反证坐实。

阶段（--stage 分块可断点续跑，产物 results/calibration/cal_p4_ext/<run>/）：

1. target——开发族 dev_v3 五种子 20k 步检查点的谱目标：逐秩特征值
   中位数（跨种子中位）与 logdet 标准差（跨种子中位）。不触评估族。
2. match——build_matched_system 拟合合成系统（校准种子），落盘参数
   与匹配报告（logdet_std 缺口如实记录：径向层 η ≤ 3τ₀ 构造上限）。
3. simulate——8 链 × 200k 步白化 Langevin，拉回 z 样本缓存。
4. estimate——两臂（主链样本 / 不变测度精确采样）× 两密度口径
   （flow 冻结主口径 / kNN 标准化消融）：ĝ（Fisher 拉回估计）→ V̂
   （kNN）→ Î → 仿射拟合 → T̂ 对真值 0.5；主链臂另报分半双口径。

用法：.venv/Scripts/python.exe scripts/run_cal_p4_ext.py [--stage target|match|simulate|estimate]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators import potential
from slep.estimators.density import log_density_knn
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.affine import affine_fit_report, split_half_temperature
from slep.systems.cal_langevin import build_matched_system, system_from_params, system_to_params
from slep.systems.s2_world_model import S2WorldModel
from slep.systems import s2_gridworld as gw
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "cal_p4_ext.yaml"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "protocol_v1.1_thresholds.json"
OUT_NAME = "ext_20k_v1"


def stage_target(cfg, out_dir, log) -> dict:
    out_file = out_dir / "target_spectra.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    tc = yaml.safe_load((REPO_ROOT / cfg["target_train_config"]).read_text(encoding="utf-8"))
    per_seed_med, per_seed_logdet_std = [], []
    for seed in guard.family_seeds("development", purpose="cal-p4-ext-target"):
        rng = np.random.default_rng(seed + 970_000)
        obs_np, act_np = gw.collect_rollouts(cfg["target_n_episodes"], tc["episode_len"],
                                             tc["maze_cells"], tc["view"], rng)
        obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
        ck = torch.load(REPO_ROOT / "results/description/s2_train" / cfg["target_campaign"]
                        / f"s{seed}" / "checkpoints" / f"ckpt_{tc['train_steps']:06d}.pt",
                        weights_only=True)
        model = S2WorldModel(tc["obs_dim"], tc["action_dim"], tc["embed_dim"],
                             tc["hidden_dim"], tc["sigma_dec"], tc.get("goal_sigma_dec"))
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            hs = model.hidden_trajectory(obs, act)
        h_pool = hs[:, 8:].reshape(-1, tc["hidden_dim"]).double()
        idx = torch.from_numpy(rng.permutation(h_pool.shape[0])[: cfg["target_n_query"]])

        def dec(z, m=model):
            return m.decoder_mean(z.float()).double()

        g = fisher_pullback_gaussian_batch(dec, h_pool[idx], model.obs_var.double()).detach()
        eig = torch.linalg.eigvalsh(g).clamp_min(1e-30)
        per_seed_med.append(torch.quantile(eig, 0.5, dim=0))
        per_seed_logdet_std.append(float(torch.linalg.slogdet(g).logabsdet.std()))
        log(f"  target s{seed}: log10cond 中位 "
            f"{float(torch.log10(eig[:, -1] / eig[:, 0]).median()):.2f}, "
            f"logdet std {per_seed_logdet_std[-1]:.1f}")
    med = torch.quantile(torch.stack(per_seed_med), 0.5, dim=0)
    out = {"eig_median_by_rank": med.tolist(),
           "logdet_std_median": float(np.median(per_seed_logdet_std)),
           "logdet_std_per_seed": per_seed_logdet_std,
           "log10_cond_of_target": float(torch.log10(med[-1] / med[0]))}
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"谱目标：log10(秩谱最大/最小) = {out['log10_cond_of_target']:.2f}, "
        f"logdet std 目标 {out['logdet_std_median']:.1f}")
    return out


def stage_match(cfg, target, out_dir, log) -> dict:
    out_file = out_dir / "matched_ext.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    seed = guard.family_seeds(cfg["seed_family"], purpose="cal-p4-ext-match")[cfg["seed_index"]]
    gen = torch.Generator()
    gen.manual_seed(seed)
    d = len(target["eig_median_by_rank"])
    log_w = torch.rand(d, generator=gen, dtype=torch.float64)
    well_w = (cfg["well_w_min"] * (cfg["well_w_max"] / cfg["well_w_min"]) ** log_w).tolist()
    system, report = build_matched_system(
        target_eig_median_by_rank=target["eig_median_by_rank"],
        target_logdet_std=target["logdet_std_median"],
        obs_dim=cfg["obs_dim"], sigma_dec=cfg["sigma_dec"], sigma_x=cfg["sigma_x"],
        well_w=well_w, temperature=cfg["temperature"], generator=gen,
        warp_tau=cfg["warp_tau"],
    )
    payload = {"params": system_to_params(system), "match_report": report}
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"匹配完成：达成 log10cond 中位 {report['achieved_log10_cond_median']:.2f} "
        f"(目标秩谱 {target['log10_cond_of_target']:.2f}), logdet std 达成 "
        f"{report['achieved_logdet_std']:.2f} / 目标 {report['target_logdet_std']:.1f}"
        f"（缺口={report['achieved_logdet_std'] < 0.8 * report['target_logdet_std']}）")
    return payload


def stage_simulate(cfg, matched, out_dir, log) -> None:
    out_file = out_dir / "samples.npz"
    if out_file.exists():
        return
    t0 = time.time()
    system = system_from_params(matched["params"])
    seed = guard.family_seeds(cfg["seed_family"], purpose="cal-p4-ext-sim")[cfg["seed_index"]]
    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    z = system.simulate_chains(cfg["chains"], cfg["steps"], cfg["dt"], cfg["burn_in"],
                               cfg["thin"], gen)  # (C, m, d)
    z_exact = system.exact_sample(z.shape[0] * z.shape[1], gen)
    np.savez_compressed(out_file, z=z.numpy(), z_exact=z_exact.numpy())
    log(f"模拟完成：链样本 {tuple(z.shape)}，精确样本 {tuple(z_exact.shape)}"
        f"（{time.time() - t0:.0f}s）")


def estimate_arm(cfg, system, z_flat, tags, arm, gen, log) -> dict:
    """单臂估计链：ĝ → V̂ → Î（flow 主 / kNN 消融）→ 仿射与分半。"""
    t0 = time.time()
    th_p4 = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["p4"]
    obs_var = torch.full((cfg["obs_dim"],), float(cfg["sigma_dec"]) ** 2,
                         dtype=torch.float64)
    n = z_flat.shape[0]
    perm = torch.randperm(n, generator=gen)
    idx_ref = perm[: cfg["n_ref"]]
    idx_q = perm[cfg["n_ref"]: cfg["n_ref"] + cfg["n_queries"]]
    idx_flow = perm[cfg["n_ref"] + cfg["n_queries"]:
                    cfg["n_ref"] + cfg["n_queries"] + cfg["n_flow_train"] + cfg["n_flow_val"]]
    z_ref = z_flat[idx_ref]
    o_ref = system.sample_observations(z_ref, gen)
    z_q = z_flat[idx_q]

    g_q = fisher_pullback_gaussian_batch(system.decoder_mean, z_q, obs_var).detach()
    logdet_est = torch.linalg.slogdet(g_q).logabsdet
    eig = torch.linalg.eigvalsh(g_q).clamp_min(1e-30)
    v_parts = []
    for i in range(0, z_q.shape[0], 256):
        with torch.no_grad():
            v_parts.append(potential.potential_knn(
                system.decoder_mean, obs_var, z_q[i: i + 256], z_ref, o_ref,
                k=cfg["k_potential"]))
    v_q = torch.cat(v_parts)

    fc = th_p4["flow_config"]
    nft = cfg["n_flow_train"]
    fd, _ = fit_flow_density(z_flat[idx_flow[:nft]], z_flat[idx_flow[nft:]],
                             seed=99, n_couplings=fc["couplings"], hidden=fc["hidden"],
                             epochs=fc["epochs"], batch_size=fc["batch"])
    i_flow = -fd.log_prob(z_q) + 0.5 * logdet_est
    i_knn = -log_density_knn(z_q, z_flat[idx_flow[:nft]], k=cfg["k_density"],
                             standardize=True) + 0.5 * logdet_est

    t_true = float(cfg["temperature"])
    out = {"arm": arm, "n_queries": int(z_q.shape[0]),
           "log10_cond_median_est": float(torch.log10(eig[:, -1] / eig[:, 0]).median()),
           "wall_seconds": round(time.time() - t0)}
    for name, i_hat in (("flow", i_flow), ("knn_std", i_knn)):
        rep = affine_fit_report(v_q, i_hat)
        rep["t_true"] = t_true
        rep["t_rel_err"] = abs(rep["temperature_hat"] - t_true) / t_true
        keep = {k: rep[k] for k in ("slope", "temperature_hat", "t_rel_err", "r_squared",
                                    "p_lack_of_fit", "delta_bic_lin_minus_quad",
                                    "curvature_effect_ratio", "n")}
        if tags is not None:
            time_q, chain_q = tags[0][idx_q], tags[1][idx_q]
            keep["split_time"] = split_half_temperature(v_q, i_hat, time_q < time_q.median())
            keep["split_chain"] = split_half_temperature(v_q, i_hat,
                                                        chain_q < cfg["chains"] // 2)
        out[name] = keep
        log(f"  [{arm}/{name}] T̂={keep['temperature_hat']:.4f}（真 {t_true}）"
            f"误差 {keep['t_rel_err']:.1%} R²={keep['r_squared']:.3f} "
            f"ρ={keep['curvature_effect_ratio']:.3f}")
    return out


def stage_estimate(cfg, matched, out_dir, log) -> None:
    out_file = out_dir / "recovery_report.json"
    if out_file.exists():
        log("跳过已完成 estimate")
        return
    system = system_from_params(matched["params"])
    data = np.load(out_dir / "samples.npz")
    z = torch.from_numpy(data["z"])  # (C, m, d)
    c, m, d = z.shape
    z_flat = z.reshape(-1, d)
    time_idx = torch.arange(m).repeat(c)
    chain_idx = torch.arange(c).repeat_interleave(m)
    seed = guard.family_seeds(cfg["seed_family"], purpose="cal-p4-ext-est")[cfg["seed_index"]]
    gen = torch.Generator()
    gen.manual_seed(seed + 2)
    arms = [estimate_arm(cfg, system, z_flat, (time_idx, chain_idx), "chains", gen, log),
            estimate_arm(cfg, system, torch.from_numpy(data["z_exact"]), None,
                         "exact", gen, log)]
    report = {
        "true_temperature": cfg["temperature"],
        "matched_log10_cond": matched["match_report"]["achieved_log10_cond_median"],
        "matched_logdet_std": matched["match_report"]["achieved_logdet_std"],
        "target_logdet_std": matched["match_report"]["target_logdet_std"],
        "arms": arms,
        "reading": ("对照原可行域（5.74 下 flow 误差 1.2–6.0%）与温度容差语境 12% 读数；"
                    "logdet_std 匹配缺口沿用 v1 记录的构造族上限，位置变率引起的误差"
                    "在本检验中可能被低估，扩门论证须计入该限定"),
    }
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_isolation(cfg, matched, out_dir, log) -> None:
    """失效环节归因：对精确采样臂逐环节对照闭式真值。

    链条 Î_est = −log p̂ + ½ logdet ĝ 与 V̂ 各自对照真值：V̂ 对
    potential_estimand、密度对 −log p_z 真值（由 Î 真值 − ½ logdet g
    真值得出）、ĝ logdet 估计对闭式；真值对真值的仿射斜率应精确为
    1/T（构造保证），作管线基准。
    """
    out_file = out_dir / "isolation.json"
    if out_file.exists():
        return
    system = system_from_params(matched["params"])
    data = np.load(out_dir / "samples.npz")
    z_flat = torch.from_numpy(data["z_exact"])
    seed = guard.family_seeds(cfg["seed_family"], purpose="cal-p4-ext-iso")[cfg["seed_index"]]
    gen = torch.Generator()
    gen.manual_seed(seed + 3)
    obs_var = torch.full((cfg["obs_dim"],), float(cfg["sigma_dec"]) ** 2, dtype=torch.float64)
    perm = torch.randperm(z_flat.shape[0], generator=gen)
    z_ref = z_flat[perm[: cfg["n_ref"]]]
    o_ref = system.sample_observations(z_ref, gen)
    z_q = z_flat[perm[cfg["n_ref"]: cfg["n_ref"] + 2000]]
    nft = cfg["n_flow_train"]
    idx_flow = perm[cfg["n_ref"] + 2000: cfg["n_ref"] + 2000 + nft + cfg["n_flow_val"]]

    v_true = system.potential_estimand(z_q)
    i_true = system.self_information_true(z_q)
    logdet_true = system.logdet_metric_true(z_q)
    logp_true = -(i_true - 0.5 * logdet_true)

    g_q = fisher_pullback_gaussian_batch(system.decoder_mean, z_q, obs_var).detach()
    logdet_est = torch.linalg.slogdet(g_q).logabsdet
    v_parts = []
    for i in range(0, z_q.shape[0], 256):
        with torch.no_grad():
            v_parts.append(potential.potential_knn(
                system.decoder_mean, obs_var, z_q[i: i + 256], z_ref, o_ref,
                k=cfg["k_potential"]))
    v_hat = torch.cat(v_parts)
    th_p4 = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["p4"]
    fc = th_p4["flow_config"]
    fd, _ = fit_flow_density(z_flat[idx_flow[:nft]], z_flat[idx_flow[nft:]], seed=99,
                             n_couplings=fc["couplings"], hidden=fc["hidden"],
                             epochs=fc["epochs"], batch_size=fc["batch"])
    logp_est = fd.log_prob(z_q)

    def link(est: torch.Tensor, true: torch.Tensor) -> dict:
        r = affine_fit_report(true, est)
        return {"corr": float(torch.corrcoef(torch.stack([est, true]))[0, 1]),
                "slope_est_on_true": r["slope"],
                "bias_mean": float((est - true).mean()),
                "rmse": float(((est - true) ** 2).mean().sqrt())}

    out = {
        "baseline_true_on_true": {
            "slope": affine_fit_report(v_true, i_true)["slope"],
            "expected": 1.0 / cfg["temperature"],
        },
        "v_hat_vs_estimand": link(v_hat, v_true),
        "logp_flow_vs_true": link(logp_est, logp_true),
        "logdet_ghat_vs_true": link(logdet_est, logdet_true),
        "spread_nats": {"v_true_sd": float(v_true.std()),
                        "logdet_true_sd": float(logdet_true.std()),
                        "logp_true_sd": float(logp_true.std())},
    }
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"隔离：真值基准斜率 {out['baseline_true_on_true']['slope']:.3f}（应 2.0）；"
        f"V̂ 相关 {out['v_hat_vs_estimand']['corr']:.3f}，"
        f"flow logp 相关 {out['logp_flow_vs_true']['corr']:.3f}，"
        f"logdet ĝ 相关 {out['logdet_ghat_vs_true']['corr']:.3f}")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    stage = None
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    out_dir = create_campaign_dir("calibration", "cal_p4_ext", OUT_NAME, cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    target = stage_target(cfg, out_dir, log) if stage in (None, "target") else (
        json.loads((out_dir / "target_spectra.json").read_text(encoding="utf-8")))
    if stage in (None, "match", "simulate", "estimate", "isolation"):
        matched = stage_match(cfg, target, out_dir, log)
    if stage in (None, "simulate", "estimate"):
        stage_simulate(cfg, matched, out_dir, log)
    if stage in (None, "estimate"):
        stage_estimate(cfg, matched, out_dir, log)
    if stage in (None, "isolation"):
        stage_isolation(cfg, matched, out_dir, log)
    log("CAL-P4-EXT 批次完成")


if __name__ == "__main__":
    main()
