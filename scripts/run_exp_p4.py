"""EXP-P4 主链（确证阶段，任务八/10b+10c）：S2 评估族仿射律与温度。

协议 v1.1 第 5 节 P4 原判据 + 增补判据 a/b 的统计量生产；解耦对照与
盆地分段在 run_exp_p4_controls.py（10d）。judge 判定不在本脚本——本
脚本只落盘统计量，汇总进 judge_input.json 由 10e 完成。

双时间尺度（CLAUDE.md 硬规则 4）：模型参数冻结在训练末检查点，
逐种子解析的检查点路径写入产物 main.json 的 frozen_checkpoint 字段。

流程（逐评估种子）：
1. 长程自由运行：n_chains 条独立链（随机策略环境流，RNG 与训练/
   诊断流分离），每链 episodes_per_chain × 64 步，GRU 编码取隐态，
   burn-in 后每链均匀子采样 reservoir_per_chain 态存档（时间标签保
   留），另存每链时间序等距 rhat_series_n 态供平稳门；逐链缓存
   chain_<c>.npz，断点续跑粒度为链。
2. 估计：汇合态池不相交抽 ref/query/flow 集；ĝ（Fisher 拉回）于
   query 点 → logdet 与谱（几何门统计同源）；V̂（kNN 解码 NLL）；
   Î = −log p̂ + ½ log det ĝ，密度主口径 flow（参数读冻结阈值表
   p4.flow_config），kNN 标准化消融并报；仿射拟合全套统计量。
3. 分半温度（CAL-P4 同构，容差 0.12 的推导口径）：同一密度拟合下
   query 点按时间中位与链半分两口径掩码，t_split_rel 取两口径较大者。
4. 平稳门：每链 V̂ 序列（共享 ref 集）的 Gelman–Rubin R̂。
5. 能力门：末检查点确定性导航评估（probe 同构：逐任务固定随机流，
   任务清单 rng = default_rng(seed + 313000)），逐种子落盘，池化在
   10e 汇总时计算。

产物 results/confirmation/exp_p4/<out_campaign>/s<seed>/：
chain_<c>.npz、main.json、scatter.npz、capability.json。

用法：.venv/Scripts/python.exe scripts/run_exp_p4.py [--seed N] [--stage chains|main|capability]
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
from slep.systems import s2_gridworld as gw
from slep.systems.s2_planner import ExhaustiveMPCPlanner, mpc_episode
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "exp_p4.yaml"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "protocol_v1.1_thresholds.json"


def load_frozen_model(train_cfg: dict, seed: int) -> tuple[S2WorldModel, str]:
    ckpt_path = (
        REPO_ROOT / "results" / "confirmation" / "s2_train" / train_cfg["campaign"]
        / f"s{seed}" / "checkpoints" / f"ckpt_{train_cfg['train_steps']:06d}.pt"
    )
    ck = torch.load(ckpt_path, weights_only=True)
    model = S2WorldModel(
        train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["embed_dim"],
        train_cfg["hidden_dim"], train_cfg["sigma_dec"], train_cfg.get("goal_sigma_dec"),
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model, str(ckpt_path.relative_to(REPO_ROOT))


def run_chain(cfg, train_cfg, model, seed: int, chain: int, seed_dir, log) -> None:
    """单链自由运行 → 子采样存档。缓存存在即跳过。"""
    out = seed_dir / f"chain_{chain}.npz"
    if out.exists():
        return
    t0 = time.time()
    rng = np.random.default_rng(seed * 7919 + chain + cfg["chain_stream_offset"])
    obs_np, act_np = gw.collect_rollouts(
        cfg["episodes_per_chain"], train_cfg["episode_len"],
        train_cfg["maze_cells"], train_cfg["view"], rng,
    )
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)  # (E, T, H)
    bi = cfg["burn_in"]
    # 配对约定：hs[:, i] 解码预测 obs[:, i+1]（rollout_loss 同构）。
    # 两切片每回合均 T−bi 个元素，展平后逐行对齐。
    h_pool = hs[:, bi:].reshape(-1, model.hidden_dim)
    o_next = obs[:, bi + 1:].reshape(-1, train_cfg["obs_dim"])
    n = h_pool.shape[0]
    # 全局时间标签：episode 内步序展平（episode 序 × 步序），链内单调
    t_glob = np.arange(n, dtype=np.int64)

    res_idx = np.sort(rng.choice(n, size=cfg["reservoir_per_chain"], replace=False))
    ser_idx = np.linspace(0, n - 1, cfg["rhat_series_n"]).astype(np.int64)
    np.savez_compressed(
        out,
        h=h_pool[res_idx].numpy().astype(np.float32),
        o_next=o_next[res_idx].numpy().astype(np.float32),
        t=t_glob[res_idx],
        h_series=h_pool[ser_idx].numpy().astype(np.float32),
        n_states=n,
    )
    log(f"  s{seed} 链 {chain} 完成：{n} 态（环境步 "
        f"{cfg['episodes_per_chain'] * train_cfg['episode_len']}），{time.time() - t0:.0f}s")


def gelman_rubin(series: torch.Tensor) -> float:
    """R̂（Gelman–Rubin）。series 形状 (m 链, n 点)。"""
    m, n = series.shape
    chain_means = series.mean(dim=1)
    w = series.var(dim=1, unbiased=True).mean()
    b = n * chain_means.var(unbiased=True)
    var_plus = (n - 1) / n * w + b / n
    return float(torch.sqrt(var_plus / w))


def run_main(cfg, train_cfg, model, ckpt_rel: str, seed: int, seed_dir, log) -> dict:
    out_file = seed_dir / "main.json"
    if out_file.exists():
        log(f"跳过已完成 main s{seed}")
        return json.loads(out_file.read_text(encoding="utf-8"))
    t0 = time.time()
    th_p4 = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["p4"]
    obs_var = model.obs_var.double()

    chains = [np.load(seed_dir / f"chain_{c}.npz") for c in range(cfg["n_chains"])]
    h_pool = torch.from_numpy(np.concatenate([c["h"] for c in chains])).double()
    o_pool = torch.from_numpy(np.concatenate([c["o_next"] for c in chains])).double()
    t_pool = torch.from_numpy(np.concatenate([c["t"] for c in chains]))
    chain_tag = torch.from_numpy(np.concatenate(
        [np.full(c["h"].shape[0], ci, dtype=np.int64) for ci, c in enumerate(chains)]
    ))

    rng = np.random.default_rng(seed * 7919 + 777_000)
    perm = torch.from_numpy(rng.permutation(h_pool.shape[0]))
    n_ref, n_q = cfg["n_ref"], cfg["n_query"]
    n_ft, n_fv = 16000, 2000  # flow 训练/验证集规模（与描述阶段加强档一致）
    idx_ref = perm[:n_ref]
    idx_q = perm[n_ref: n_ref + n_q]
    idx_flow = perm[n_ref + n_q: n_ref + n_q + n_ft + n_fv]
    h_ref, o_ref = h_pool[idx_ref], o_pool[idx_ref]
    h_q = h_pool[idx_q]
    t_q, chain_q = t_pool[idx_q], chain_tag[idx_q]

    def dec(z):
        return model.decoder_mean(z.float()).double()

    # ĝ 于 query 点：logdet 供体积校正；谱供几何门（10e 取用）
    g_q = fisher_pullback_gaussian_batch(dec, h_q, obs_var).detach()
    eig = torch.linalg.eigvalsh(g_q)
    logdet_g = torch.log(eig).sum(dim=-1)
    log10_cond = torch.log10(eig[:, -1] / eig[:, 0])
    participation = eig.sum(-1) ** 2 / (eig**2).sum(-1)
    log(f"  s{seed} ĝ 完成（{time.time() - t0:.0f}s）")

    # V̂ 于 query 点
    v_parts = []
    for i in range(0, h_q.shape[0], 256):
        with torch.no_grad():
            v_parts.append(potential.potential_knn(
                dec, obs_var, h_q[i: i + 256], h_ref, o_ref, k=cfg["k_potential"]))
    v_q = torch.cat(v_parts)
    log(f"  s{seed} V̂ 完成（{time.time() - t0:.0f}s）")

    # 密度主口径 flow（冻结参数）；消融 kNN 标准化
    fc = th_p4["flow_config"]
    fd, flow_diag = fit_flow_density(
        h_pool[idx_flow[:n_ft]], h_pool[idx_flow[n_ft:]], seed=seed,
        n_couplings=fc["couplings"], hidden=fc["hidden"],
        epochs=fc["epochs"], batch_size=fc["batch"],
    )
    log_p = fd.log_prob(h_q)
    i_corr = -log_p + 0.5 * logdet_g
    i_raw = -log_p
    log_p_knn = log_density_knn(h_q, h_pool[idx_flow[:n_ft]],
                                k=cfg["k_knn_density"], standardize=True)
    i_knn = -log_p_knn + 0.5 * logdet_g
    log(f"  s{seed} 密度完成（{time.time() - t0:.0f}s）")

    fit = affine_fit_report(v_q, i_corr)
    fit_raw = affine_fit_report(v_q, i_raw)
    fit_knn = affine_fit_report(v_q, i_knn)

    # 分半温度（CAL-P4 同构：同一密度拟合下按掩码分半）
    split_time = split_half_temperature(v_q, i_corr, t_q < t_q.median())
    split_chain = split_half_temperature(v_q, i_corr, chain_q < cfg["n_chains"] // 2)
    t_split_rel = max(split_time["relative_gap"], split_chain["relative_gap"])

    # 平稳门：每链 V̂ 序列（共享 ref 集）R̂
    v_series = []
    for c in chains:
        hc = torch.from_numpy(c["h_series"]).double()
        parts = []
        for i in range(0, hc.shape[0], 256):
            with torch.no_grad():
                parts.append(potential.potential_knn(
                    dec, obs_var, hc[i: i + 256], h_ref, o_ref, k=cfg["k_potential"]))
        v_series.append(torch.cat(parts))
    rhat = gelman_rubin(torch.stack(v_series))
    log(f"  s{seed} 平稳门 R̂={rhat:.4f}（{time.time() - t0:.0f}s）")

    out = {
        "seed": seed,
        "frozen_checkpoint": ckpt_rel,
        "n_env_steps_total": int(cfg["n_chains"] * cfg["episodes_per_chain"]
                                 * train_cfg["episode_len"]),
        "affine_main": fit,
        "affine_uncorrected": {k: fit_raw[k] for k in ("slope", "r_squared", "temperature_hat")},
        "affine_knn_ablation": {k: fit_knn[k] for k in ("slope", "r_squared", "temperature_hat",
                                                        "p_lack_of_fit")},
        "split_time": split_time,
        "split_chain": split_chain,
        "t_split_rel": t_split_rel,
        "geometry": {
            "log10_cond_median": float(log10_cond.median()),
            "participation_median": float(participation.median()),
        },
        "stationarity": {"rhat": rhat, "n_chains": cfg["n_chains"],
                         "series_n": cfg["rhat_series_n"]},
        "flow_val_nll": flow_diag["final_val_nll"],
        "wall_seconds": round(time.time() - t0),
    }
    np.savez_compressed(seed_dir / "refsets.npz",
                        h_ref=h_ref.numpy().astype(np.float32),
                        o_ref=o_ref.numpy().astype(np.float32),
                        h_q=h_q.numpy().astype(np.float32))
    np.savez_compressed(seed_dir / "scatter.npz", v=v_q.numpy(), i_corr=i_corr.numpy(),
                        i_raw=i_raw.numpy(), i_knn=i_knn.numpy(),
                        logdet_g=logdet_g.numpy(), t=t_q.numpy(), chain=chain_q.numpy(),
                        log10_cond=log10_cond.numpy(), participation=participation.numpy())
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"完成 main s{seed}（{out['wall_seconds']}s）：斜率 {fit['slope']:.3f} "
        f"T̂={fit['temperature_hat']:.3f} R²={fit['r_squared']:.3f} "
        f"p={fit['p_lack_of_fit']:.3g} ΔBIC={fit['delta_bic_lin_minus_quad']:.2f} "
        f"ρ={fit['curvature_effect_ratio']:.3f} 分半 {t_split_rel:.3f} R̂={rhat:.4f}"
    )
    return out


def run_capability(cfg, train_cfg, model, seed: int, seed_dir, log) -> None:
    """能力门：末检查点确定性导航评估（probe 同构口径）。"""
    out_file = seed_dir / "capability.json"
    if out_file.exists():
        return
    cap = cfg["capability"]
    t0 = time.time()
    rng = np.random.default_rng(seed + cap["task_stream_offset"])
    tasks = []
    while len(tasks) < cap["n_tasks"]:
        maze = gw.generate_maze(train_cfg["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or not cap["bfs_band"][0] <= d <= cap["bfs_band"][1]:
            continue
        tasks.append((maze, start, goal))
    planner = ExhaustiveMPCPlanner(model, view=train_cfg["view"])
    succ = 0
    for ti, (maze, start, goal) in enumerate(tasks):
        gen = torch.Generator()
        gen.manual_seed(seed * 100_003 + ti)
        env = gw.GridWorld(maze.copy(), start, train_cfg["view"], goal)
        succ += int(mpc_episode(model, env, planner, cap["max_steps"], gen)["success"])
    out_file.write_text(json.dumps(
        {"seed": seed, "success": succ, "n": cap["n_tasks"],
         "rate": succ / cap["n_tasks"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"能力门 s{seed}: {succ}/{cap['n_tasks']} = {succ / cap['n_tasks']:.3f} "
        f"({time.time() - t0:.0f}s)")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (REPO_ROOT / cfg["train_config"]).read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    only_seed = None
    if "--seed" in sys.argv:
        only_seed = int(sys.argv[sys.argv.index("--seed") + 1])
    stage = None
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    out_dir = create_campaign_dir("confirmation", "exp_p4", cfg["out_campaign"], cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds = guard.family_seeds("evaluation", purpose="exp-p4")
    if only_seed is not None:
        seeds = [only_seed]
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="exp-p4")
        seed_dir = out_dir / f"s{seed}"
        seed_dir.mkdir(exist_ok=True)
        model, ckpt_rel = load_frozen_model(train_cfg, seed)
        if stage in (None, "chains"):
            for c in range(cfg["n_chains"]):
                run_chain(cfg, train_cfg, model, seed, c, seed_dir, log)
        if stage in (None, "main"):
            run_main(cfg, train_cfg, model, ckpt_rel, seed, seed_dir, log)
        if stage in (None, "capability"):
            run_capability(cfg, train_cfg, model, seed, seed_dir, log)
    log("EXP-P4 主链批次完成")


if __name__ == "__main__":
    main()
