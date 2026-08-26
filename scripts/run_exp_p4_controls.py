"""EXP-P4 解耦对照与另列报告（确证阶段，任务八/10d）。

四件（plan_v2 第 8 节 EXP-P4 第 5、6、7、8 条；判定字段前两件）：

1. 均匀解码器对照：新采随机策略 rollout（记录位置），冻结 GRU/编码器，
   仅重训解码器头——(h, o_next) 对按目标位置格的逆访问频次加权，等效
   把解码器训练分布在位置格上均匀化；访问分布（Î 与查询点）不动，
   V̂ 换新解码器重算，斜率相对漂移 = |b_ctrl − b_main|/|b_main|。
   操作化注记："均匀采样状态"取位置格占据的均匀化（迷宫逐回合随机，
   位置是跨回合可比的占据变量）。
2. 换策略对照：粘滞策略（重复上一动作概率 p_sticky，否则均匀）重采
   两条链，冻结解码器全链重算（新 ref/query/flow），斜率相对漂移。
3. 盆地分段（另列报告）：查询点 GMM（BIC 选 K∈2–8）分簇，簇内
   （n≥150）仿射拟合 + ANCOVA 共同斜率 F 检验。
4. 噪声操纵判向（估计层口径，docs/manipulation_signature.md）：冻结
   样本与解码器，仅 V̂ 读到的邻居观测 o_ref 加噪 σ'，判向三条：均值
   上移量对 Δ = σ'²·Σ_j(1/σ_j²)/2、斜率衰减（T̂ 上升）、lack-of-fit
   不新增结构。另报：异种子度量副本（体积校正项换他种子解码器的 ĝ）。

依赖 run_exp_p4.py 产物（refsets.npz、scatter.npz、main.json）。
产物 results/confirmation/exp_p4/<out_campaign>/s<seed>/controls.json。

用法：.venv/Scripts/python.exe scripts/run_exp_p4_controls.py [--seed N]
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
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.affine import affine_fit_report
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT

CONFIG_FILE = REPO_ROOT / "configs" / "exp_p4.yaml"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "protocol_v1.1_thresholds.json"
CTRL_EPISODES = 2000        # 均匀解码器对照的数据量（burn-in 后约 11 万对）
SWAP_EPISODES = 4096        # 换策略对照每链 episode 数
SWAP_CHAINS = 2
SWAP_RESERVOIR = 20000      # 每链子采样（两链 4 万 ≥ ref+query+flow 预算 3.6 万）
P_STICKY = 0.65             # 粘滞策略重复概率
DEC_EPOCHS, DEC_BATCH, DEC_LR = 8, 1024, 1.0e-3
NOISE_LEVELS = [0.02, 0.05]  # 操纵档 σ'
GMM_K_RANGE = range(2, 9)
BASIN_MIN_N = 150


def collect_rollouts_pos(n_episodes, episode_len, n_cells, view, rng, p_sticky=None):
    """collect_rollouts 的扩展：附带逐步位置；p_sticky 非空时用粘滞策略。"""
    obs_dim = 2 * view * view
    obs = np.empty((n_episodes, episode_len + 1, obs_dim), dtype=np.float32)
    act = np.zeros((n_episodes, episode_len, 4), dtype=np.float32)
    pos = np.empty((n_episodes, episode_len + 1, 2), dtype=np.int16)
    for e in range(n_episodes):
        env = gw.make_episode_env(n_cells, view, rng)
        obs[e, 0] = env.observe()
        pos[e, 0] = env.position
        prev = None
        for t in range(episode_len):
            if p_sticky is not None and prev is not None and rng.random() < p_sticky:
                a = prev
            else:
                a = int(rng.integers(0, 4))
            obs[e, t + 1] = env.step(a)
            pos[e, t + 1] = env.position
            act[e, t, a] = 1.0
            prev = a
    return obs, act, pos


def load_frozen_model(train_cfg: dict, seed: int) -> S2WorldModel:
    ck = torch.load(
        REPO_ROOT / "results" / "confirmation" / "s2_train" / train_cfg["campaign"]
        / f"s{seed}" / "checkpoints" / f"ckpt_{train_cfg['train_steps']:06d}.pt",
        weights_only=True)
    model = S2WorldModel(
        train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["embed_dim"],
        train_cfg["hidden_dim"], train_cfg["sigma_dec"], train_cfg.get("goal_sigma_dec"))
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def vhat(dec, obs_var, h_q, h_ref, o_ref, k):
    parts = []
    for i in range(0, h_q.shape[0], 256):
        with torch.no_grad():
            parts.append(potential.potential_knn(
                dec, obs_var, h_q[i: i + 256], h_ref, o_ref, k=k))
    return torch.cat(parts)


def uniform_decoder_control(cfg, train_cfg, model, seed, seed_dir, log) -> dict:
    """对照一：逆占据频次加权重训解码器头，V̂ 重算，同轴重拟合。"""
    t0 = time.time()
    rng = np.random.default_rng(seed * 7919 + 910_000)
    obs_np, act_np, pos_np = collect_rollouts_pos(
        CTRL_EPISODES, train_cfg["episode_len"], train_cfg["maze_cells"],
        train_cfg["view"], rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)
    bi = cfg["burn_in"]
    # 配对约定同 run_exp_p4：hs[:, i] ↔ obs[:, i+1] ↔ pos[:, i+1]，
    # 三切片每回合等长，展平后逐行对齐。
    h_all = hs[:, bi:].reshape(-1, model.hidden_dim)
    o_all = obs[:, bi + 1:].reshape(-1, train_cfg["obs_dim"])
    p_all = pos_np[:, bi + 1:].reshape(-1, 2)

    # 逆占据频次权重（位置格）
    size = 2 * train_cfg["maze_cells"] + 1
    cell_id = p_all[:, 0].astype(np.int64) * size + p_all[:, 1].astype(np.int64)
    counts = np.bincount(cell_id, minlength=size * size).astype(np.float64)
    w = 1.0 / counts[cell_id]
    w = torch.from_numpy(w / w.mean())

    # 仅重训解码器头（GRU/编码器冻结在 h 里）
    import copy

    dec_ctrl = copy.deepcopy(model.decoder)
    opt = torch.optim.Adam(dec_ctrl.parameters(), lr=DEC_LR)
    obs_var = model.obs_var
    n = h_all.shape[0]
    n_hold = n // 10
    gen = torch.Generator()
    gen.manual_seed(seed)
    order = torch.randperm(n, generator=gen)
    tr, ho = order[n_hold:], order[:n_hold]
    for epoch in range(DEC_EPOCHS):
        perm = tr[torch.randperm(tr.shape[0], generator=gen)]
        for i in range(0, perm.shape[0], DEC_BATCH):
            idx = perm[i: i + DEC_BATCH]
            mu = torch.sigmoid(dec_ctrl(h_all[idx]))
            nll = (((o_all[idx] - mu) ** 2) / obs_var).sum(-1) / 2
            loss = (nll * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mu_h = torch.sigmoid(dec_ctrl(h_all[ho]))
        hold_nll = float(((((o_all[ho] - mu_h) ** 2) / obs_var).sum(-1) / 2).mean())

    refs = np.load(seed_dir / "refsets.npz")
    sc = np.load(seed_dir / "scatter.npz")
    h_ref = torch.from_numpy(refs["h_ref"]).double()
    o_ref = torch.from_numpy(refs["o_ref"]).double()
    h_q = torch.from_numpy(refs["h_q"]).double()

    def dec_fn(z):
        return torch.sigmoid(dec_ctrl(z.float())).double()

    v_ctrl = vhat(dec_fn, obs_var.double(), h_q, h_ref, o_ref, cfg["k_potential"])
    fit_ctrl = affine_fit_report(v_ctrl, torch.from_numpy(sc["i_corr"]))
    slope_main = json.loads((seed_dir / "main.json").read_text(encoding="utf-8"))[
        "affine_main"]["slope"]
    out = {
        "slope_ctrl": fit_ctrl["slope"],
        "slope_main": slope_main,
        "shift_rel": abs(fit_ctrl["slope"] - slope_main) / abs(slope_main),
        "r_squared_ctrl": fit_ctrl["r_squared"],
        "decoder_holdout_nll": hold_nll,
        "n_pairs": int(n),
        "occupancy_cells": int((counts > 0).sum()),
        "wall_seconds": round(time.time() - t0),
    }
    log(f"  s{seed} 均匀解码器对照：斜率 {fit_ctrl['slope']:.3f} 对主 {slope_main:.3f}，"
        f"漂移 {out['shift_rel']:.3f}（{out['wall_seconds']}s）")
    return out


def policy_swap_control(cfg, train_cfg, model, seed, seed_dir, log) -> dict:
    """对照二：粘滞策略重采链，冻结解码器全链重算。逐链缓存。"""
    t0 = time.time()
    th_p4 = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["p4"]
    obs_var = model.obs_var.double()
    pools_h, pools_o = [], []
    for c in range(SWAP_CHAINS):
        cache = seed_dir / f"swap_chain_{c}.npz"
        if not cache.exists():
            rng = np.random.default_rng(seed * 7919 + c + 920_000)
            obs_np, act_np, _ = collect_rollouts_pos(
                SWAP_EPISODES, train_cfg["episode_len"], train_cfg["maze_cells"],
                train_cfg["view"], rng, p_sticky=P_STICKY)
            obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
            with torch.no_grad():
                hs = model.hidden_trajectory(obs, act)
            bi = cfg["burn_in"]
            h_pool = hs[:, bi:].reshape(-1, model.hidden_dim)  # 配对约定同上
            o_next = obs[:, bi + 1:].reshape(-1, train_cfg["obs_dim"])
            idx = np.sort(rng.choice(h_pool.shape[0], size=SWAP_RESERVOIR, replace=False))
            np.savez_compressed(cache, h=h_pool[idx].numpy().astype(np.float32),
                                o_next=o_next[idx].numpy().astype(np.float32))
            log(f"  s{seed} 换策略链 {c} 完成（{time.time() - t0:.0f}s）")
        z = np.load(cache)
        pools_h.append(torch.from_numpy(z["h"]).double())
        pools_o.append(torch.from_numpy(z["o_next"]).double())
    h_pool = torch.cat(pools_h)
    o_pool = torch.cat(pools_o)

    rng = np.random.default_rng(seed * 7919 + 930_000)
    perm = torch.from_numpy(rng.permutation(h_pool.shape[0]))
    n_ref, n_q, n_ft, n_fv = cfg["n_ref"], cfg["n_query"], 16000, 2000
    h_ref, o_ref = h_pool[perm[:n_ref]], o_pool[perm[:n_ref]]
    h_q = h_pool[perm[n_ref: n_ref + n_q]]
    idx_flow = perm[n_ref + n_q: n_ref + n_q + n_ft + n_fv]

    def dec(z):
        return model.decoder_mean(z.float()).double()

    g_q = fisher_pullback_gaussian_batch(dec, h_q, obs_var).detach()
    logdet_g = torch.linalg.slogdet(g_q).logabsdet
    v_q = vhat(dec, obs_var, h_q, h_ref, o_ref, cfg["k_potential"])
    fc = th_p4["flow_config"]
    fd, _ = fit_flow_density(
        h_pool[idx_flow[:n_ft]], h_pool[idx_flow[n_ft:]], seed=seed,
        n_couplings=fc["couplings"], hidden=fc["hidden"],
        epochs=fc["epochs"], batch_size=fc["batch"])
    i_corr = -fd.log_prob(h_q) + 0.5 * logdet_g
    fit_swap = affine_fit_report(v_q, i_corr)
    slope_main = json.loads((seed_dir / "main.json").read_text(encoding="utf-8"))[
        "affine_main"]["slope"]
    out = {
        "p_sticky": P_STICKY,
        "slope_swap": fit_swap["slope"],
        "slope_main": slope_main,
        "shift_rel": abs(fit_swap["slope"] - slope_main) / abs(slope_main),
        "r_squared_swap": fit_swap["r_squared"],
        "temperature_swap": fit_swap["temperature_hat"],
        "wall_seconds": round(time.time() - t0),
    }
    log(f"  s{seed} 换策略对照：斜率 {fit_swap['slope']:.3f} 对主 {slope_main:.3f}，"
        f"漂移 {out['shift_rel']:.3f}（{out['wall_seconds']}s）")
    return out


def basin_segmentation(seed, seed_dir, log) -> dict:
    """另列报告：GMM 分簇 + 簇内仿射 + ANCOVA 共同斜率检验。"""
    from scipy import stats
    from sklearn.mixture import GaussianMixture

    refs = np.load(seed_dir / "refsets.npz")
    sc = np.load(seed_dir / "scatter.npz")
    h_q = refs["h_q"].astype(np.float64)
    v, i_c = sc["v"], sc["i_corr"]

    bics = {}
    for k in GMM_K_RANGE:
        gm = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=2)
        gm.fit(h_q)
        bics[k] = float(gm.bic(h_q))
    k_best = min(bics, key=bics.get)
    gm = GaussianMixture(k_best, covariance_type="full", random_state=seed, n_init=2)
    labels = gm.fit_predict(h_q)

    per_basin, used = [], []
    for b in range(k_best):
        m = labels == b
        if m.sum() < BASIN_MIN_N:
            continue
        fit = affine_fit_report(torch.from_numpy(v[m]), torch.from_numpy(i_c[m]))
        per_basin.append({"basin": int(b), "n": int(m.sum()), "slope": fit["slope"],
                          "se_slope": fit["se_slope"], "r_squared": fit["r_squared"]})
        used.append(m)
    # ANCOVA 共同斜率：完全模型（各簇own斜率）对约束模型（共同斜率+各簇截距）
    if len(per_basin) >= 2:
        mask = np.logical_or.reduce(used)
        vv, ii = v[mask], i_c[mask]
        lb = labels[mask]
        groups = sorted({int(g) for g in np.unique(lb)})
        gidx = np.array([groups.index(int(g)) for g in lb])
        n, k = vv.shape[0], len(groups)
        d = np.zeros((n, k))
        d[np.arange(n), gidx] = 1.0
        x_common = np.column_stack([d, vv])                       # 截距×k + 共同斜率
        x_full = np.column_stack([d, d * vv[:, None]])            # 截距×k + 斜率×k
        rss_c = float(np.sum((ii - x_common @ np.linalg.lstsq(x_common, ii, rcond=None)[0]) ** 2))
        rss_f = float(np.sum((ii - x_full @ np.linalg.lstsq(x_full, ii, rcond=None)[0]) ** 2))
        df1, df2 = k - 1, n - 2 * k
        f_stat = ((rss_c - rss_f) / df1) / (rss_f / df2)
        p_common = float(stats.f.sf(f_stat, df1, df2))
    else:
        f_stat, p_common = None, None
    out = {"k_best": int(k_best), "bic": bics, "per_basin": per_basin,
           "common_slope_f": f_stat, "common_slope_p": p_common}
    log(f"  s{seed} 盆地分段：K={k_best}，可拟合簇 {len(per_basin)}，"
        f"共同斜率 p={p_common if p_common is None else round(p_common, 4)}")
    return out


def manipulation_check(cfg, train_cfg, model, seed, seed_dir, log) -> dict:
    """判向：估计层加噪 σ'，三条签名（docs/manipulation_signature.md）。"""
    refs = np.load(seed_dir / "refsets.npz")
    sc = np.load(seed_dir / "scatter.npz")
    h_ref = torch.from_numpy(refs["h_ref"]).double()
    o_ref = torch.from_numpy(refs["o_ref"]).double()
    h_q = torch.from_numpy(refs["h_q"]).double()
    i_corr = torch.from_numpy(sc["i_corr"])
    v_base = torch.from_numpy(sc["v"])
    obs_var = model.obs_var.double()
    fit_base = affine_fit_report(v_base, i_corr)

    def dec(z):
        return model.decoder_mean(z.float()).double()

    gen = torch.Generator()
    gen.manual_seed(seed + 940_000)
    levels = []
    for s_prime in NOISE_LEVELS:
        o_noisy = o_ref + s_prime * torch.randn(o_ref.shape, generator=gen,
                                                dtype=torch.float64)
        v_noisy = vhat(dec, obs_var, h_q, h_ref, o_noisy, cfg["k_potential"])
        fit_n = affine_fit_report(v_noisy, i_corr)
        delta_pred = s_prime**2 * float((1.0 / obs_var).sum()) / 2
        delta_meas = float(v_noisy.mean() - v_base.mean())
        levels.append({
            "sigma_prime": s_prime,
            "delta_pred": delta_pred,
            "delta_measured": delta_meas,
            "delta_ratio": delta_meas / delta_pred,
            "slope": fit_n["slope"],
            "temperature_hat": fit_n["temperature_hat"],
            "p_lack_of_fit": fit_n["p_lack_of_fit"],
        })
    t_up = all(lv["temperature_hat"] > fit_base["temperature_hat"] for lv in levels)
    t_monotone = all(levels[i]["temperature_hat"] < levels[i + 1]["temperature_hat"]
                     for i in range(len(levels) - 1))
    out = {"base_temperature": fit_base["temperature_hat"], "levels": levels,
           "t_up": t_up, "t_monotone": t_monotone}
    log(f"  s{seed} 操纵判向：T̂ 基 {fit_base['temperature_hat']:.3f} → "
        f"{[round(lv['temperature_hat'], 3) for lv in levels]}，上升 {t_up}，"
        f"Δ 比 {[round(lv['delta_ratio'], 2) for lv in levels]}")
    return out


def cross_seed_metric(cfg, train_cfg, seed, seeds_all, seed_dir, log) -> dict:
    """另列报告：体积校正项换异种子解码器的 ĝ。"""
    other = seeds_all[(seeds_all.index(seed) + 1) % len(seeds_all)]
    model_o = load_frozen_model(train_cfg, other)
    refs = np.load(seed_dir / "refsets.npz")
    sc = np.load(seed_dir / "scatter.npz")
    h_q = torch.from_numpy(refs["h_q"]).double()
    obs_var_o = model_o.obs_var.double()

    def dec_o(z):
        return model_o.decoder_mean(z.float()).double()

    g_o = fisher_pullback_gaussian_batch(dec_o, h_q, obs_var_o).detach()
    logdet_o = torch.linalg.slogdet(g_o).logabsdet
    i_cross = torch.from_numpy(sc["i_raw"]) + 0.5 * logdet_o
    fit = affine_fit_report(torch.from_numpy(sc["v"]), i_cross)
    slope_main = json.loads((seed_dir / "main.json").read_text(encoding="utf-8"))[
        "affine_main"]["slope"]
    out = {"other_seed": other, "slope_cross": fit["slope"],
           "shift_rel": abs(fit["slope"] - slope_main) / abs(slope_main),
           "r_squared_cross": fit["r_squared"]}
    log(f"  s{seed} 异种子度量（s{other}）：斜率 {fit['slope']:.3f}，"
        f"漂移 {out['shift_rel']:.3f}")
    return out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / cfg["train_config"]).read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    only_seed = None
    if "--seed" in sys.argv:
        only_seed = int(sys.argv[sys.argv.index("--seed") + 1])
    out_dir = REPO_ROOT / "results" / "confirmation" / "exp_p4" / cfg["out_campaign"]
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds_all = guard.family_seeds("evaluation", purpose="exp-p4-controls")
    seeds = [only_seed] if only_seed is not None else seeds_all
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="exp-p4-controls")
        seed_dir = out_dir / f"s{seed}"
        out_file = seed_dir / "controls.json"
        if out_file.exists():
            log(f"跳过已完成 controls s{seed}")
            continue
        model = load_frozen_model(train_cfg, seed)
        out = {
            "seed": seed,
            "uniform_decoder": uniform_decoder_control(cfg, train_cfg, model, seed,
                                                       seed_dir, log),
            "policy_swap": policy_swap_control(cfg, train_cfg, model, seed, seed_dir, log),
            "basins": basin_segmentation(seed, seed_dir, log),
            "manipulation": manipulation_check(cfg, train_cfg, model, seed, seed_dir, log),
            "cross_seed_metric": cross_seed_metric(cfg, train_cfg, seed, seeds_all,
                                                   seed_dir, log),
        }
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log("EXP-P4 对照批次完成")


if __name__ == "__main__":
    main()
