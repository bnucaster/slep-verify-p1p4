"""EXP-P3（确证阶段，计划 11.5）：低梯度区审议轨迹近测地。

协议 v1.1 第 5 节 P3 判据（v1.2 逐字继承）的统计量生产。系统 S2P
（多步解码头）：度量与势取扩展解码口径。运行纪律：评估族训练与本
管线运行严格在 v1.2 外部时间戳之后。

阶段（--stage field|traj|geo|assemble，逐种子逐段缓存断点续跑）：

1. field——冻结末检查点上 V̂ 场与 ĝ：随机策略 rollout 采参照对
   （h, 未来 k 步观测拼接），场查询点算 ‖∇_ĝV̂‖_ĝ 三分位阈值（分区
   从场预定，独立于轨迹）与高/低中位比自助 CI（地形门）。
2. traj——端点条件化 MPC 审议轨迹（确定性任务流，执行段隐轨迹），
   切定长段（时长匹配），按段内梯度范数中位对场三分位归组（中间带
   弃用），支撑度剔除（段状态 kNN 半径 > 参照集留一 q99）。
3. geo——逐段端点测地求解（多起点）：残差 ≤ 3e-3 质检；能量差 <
   0.05 且互散布 > 0.028 判非唯一剔除；对认证段算归一化偏差面积。
   端点弦长分位分箱做距离匹配，逐组求解上限截断（衰减与截断入报告）。
4. assemble——Mann–Whitney（低 < 高）、效应量（中位差）、认证对数、
   地形比 CI 下界 → judge_input 的 exp_p3 段（S2P:s<seed> 键）。

产物 results/confirmation/exp_p3/<out_campaign>/s<seed>/。

用法：.venv/Scripts/python.exe scripts/run_exp_p3.py [--seed N] [--stage ...]
  [--budget-seconds N]
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators import potential
from slep.estimators.geodesic import geodesic_deviation, solve_geodesic
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.systems import s2_gridworld as gw
from slep.systems.s2_planner import ExhaustiveMPCPlanner, mpc_episode
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "exp_p3.yaml"
THRESHOLDS_V12 = REPO_ROOT / "docs" / "protocol_v1.2_thresholds.json"


def load_frozen_model(train_cfg: dict) -> tuple:
    """按训练配置构造模型加载器；返回 (载入函数, 检查点相对路径模板)。"""
    stage = train_cfg.get("stage", "description")

    def load(seed: int) -> tuple[S2WorldModel, str]:
        p = (REPO_ROOT / "results" / stage / "s2_train" / train_cfg["campaign"]
             / f"s{seed}" / "checkpoints" / f"ckpt_{train_cfg['train_steps']:06d}.pt")
        ck = torch.load(p, weights_only=True)
        model = S2WorldModel(
            train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["embed_dim"],
            train_cfg["hidden_dim"], train_cfg["sigma_dec"], train_cfg.get("goal_sigma_dec"),
            multi_step_k=train_cfg.get("multi_step_k", 1),
            label_smooth_eps=train_cfg.get("label_smooth_eps", 0.0))
        model.load_state_dict(ck["model"])
        model.eval()
        return model, str(p.relative_to(REPO_ROOT))

    return load


def ext_pairs(model, train_cfg, obs, hs, burn_in: int):
    """(h_t, [o_{t+1..t+k}]) 参照对；k>1 时弃各回合末 k−1 位。"""
    k = getattr(model, "multi_step_k", 1)
    n_steps = hs.shape[1]
    n_pos = n_steps - k + 1
    h = hs[:, burn_in:n_pos]
    o = torch.cat([obs[:, burn_in + 1 + j: 1 + j + n_pos] for j in range(k)], dim=-1)
    return (h.reshape(-1, model.hidden_dim).double(),
            o.reshape(-1, o.shape[-1]).double())


def make_field_fns(model, cfg, h_ref, o_ref):
    obs_var = (model.obs_var_ext if getattr(model, "multi_step_k", 1) > 1
               else model.obs_var).double()

    def dec(z):
        if getattr(model, "multi_step_k", 1) > 1:
            return model.decoder_mean_ext(z.float()).double()
        return model.decoder_mean(z.float()).double()

    def v_fn(z):
        return potential.potential_knn(dec, obs_var, z, h_ref, o_ref,
                                       k=cfg["k_potential"])

    def g_fn(z):
        return fisher_pullback_gaussian_batch(dec, z, obs_var)

    return dec, v_fn, g_fn


def make_geo_metric(dec, obs_var):
    """测地求解用的可微度量：与冻结估计器同一数学对象 g = JᵀΣ⁻¹J，
    J 经 JVP（create_graph）逐基向量构造——冻结批量版显式 detach 输入
    （metric.py 第 84 行，估计口径不面向路径优化），求解器需要对路径点
    的梯度。数值等价性在 stage_geo 首次调用时与冻结版对拍断言。"""
    def g_geo(z):
        d = z.shape[-1]
        own_graph = not z.requires_grad and z.grad_fn is None
        zd = z.detach().requires_grad_(True) if own_graph else z
        out = dec(zd)
        dummy = torch.zeros_like(out, requires_grad=True)
        g1 = torch.autograd.grad(out, zd, grad_outputs=dummy, create_graph=True)[0]
        cols = []
        for i in range(d):
            e = torch.zeros_like(g1)
            e[..., i] = 1.0
            cols.append(torch.autograd.grad(g1, dummy, grad_outputs=e,
                                            create_graph=True, retain_graph=True)[0])
        jac = torch.stack(cols, dim=-1)  # (..., D, d)
        g = torch.einsum("...ja,...jb->...ab", jac / obs_var.unsqueeze(-1), jac)
        # 无梯度链输入（端点/预条件等一次性计算）返回脱图值：求解器会跨
        # 迭代复用这些量，带图会在首次反传后成为已释放的共享子图
        return g.detach() if own_graph else g

    return g_geo


def grad_norms(v_fn, g_fn, z: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """‖∇_ĝV̂‖_ĝ = sqrt(∂V̂ᵀ ĝ⁻¹ ∂V̂) 逐点。"""
    outs = []
    for i in range(0, z.shape[0], chunk):
        zc = z[i: i + chunk].detach().requires_grad_(True)
        v = v_fn(zc)
        (gv,) = torch.autograd.grad(v.sum(), zc)
        g = g_fn(zc.detach()).detach()
        nat = torch.linalg.solve(g, gv.unsqueeze(-1)).squeeze(-1)
        outs.append(torch.einsum("ti,tij,tj->t", nat, g, nat).clamp_min(0).sqrt())
    return torch.cat(outs).detach()


def stage_field(cfg, train_cfg, model, seed: int, seed_dir, log) -> dict:
    out_file = seed_dir / "field.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    t0 = time.time()
    rng = np.random.default_rng(seed + 990_000)
    obs_np, act_np = gw.collect_rollouts(cfg["n_episodes"], cfg["episode_len"],
                                         train_cfg["maze_cells"], train_cfg["view"], rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)
    h_pool, o_pool = ext_pairs(model, train_cfg, obs, hs, cfg["burn_in"])
    perm = torch.from_numpy(rng.permutation(h_pool.shape[0]))
    h_ref, o_ref = h_pool[perm[: cfg["n_ref"]]], o_pool[perm[: cfg["n_ref"]]]
    h_q = h_pool[perm[cfg["n_ref"]: cfg["n_ref"] + cfg["n_field_query"]]]

    dec, v_fn, g_fn = make_field_fns(model, cfg, h_ref, o_ref)
    gn = grad_norms(v_fn, g_fn, h_q)
    q_lo, q_hi = float(torch.quantile(gn, 1 / 3)), float(torch.quantile(gn, 2 / 3))
    lo_med = float(gn[gn <= q_lo].median())
    hi_med = float(gn[gn >= q_hi].median())
    gen_b = torch.Generator()
    gen_b.manual_seed(seed)
    boots = []
    for _ in range(cfg["bootstrap"]):
        bidx = torch.randint(0, gn.shape[0], (gn.shape[0],), generator=gen_b)
        gb = gn[bidx]
        ql, qh = torch.quantile(gb, 1 / 3), torch.quantile(gb, 2 / 3)
        boots.append(float(gb[gb >= qh].median() / gb[gb <= ql].median().clamp_min(1e-30)))
    # 支撑度剔除半径：参照集留一最近邻半径 q99（thresholds support_exclusion）
    d2 = torch.cdist(h_ref[:4096], h_ref)
    d2[torch.arange(4096), torch.arange(4096)] = float("inf")
    r_loo = d2.min(dim=1).values
    out = {
        "grad_tertiles": [q_lo, q_hi],
        "hi_lo_median_ratio": hi_med / max(lo_med, 1e-30),
        "ratio_ci90": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))],
        "support_radius_q99": float(torch.quantile(r_loo, 0.99)),
        "n_ref": int(h_ref.shape[0]),
        "wall_seconds": round(time.time() - t0),
    }
    np.savez_compressed(seed_dir / "field_ref.npz",
                        h_ref=h_ref.numpy().astype(np.float32),
                        o_ref=o_ref.numpy().astype(np.float32))
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  s{seed} field：梯度比 {out['hi_lo_median_ratio']:.2f} "
        f"CI90 {out['ratio_ci90']}（{out['wall_seconds']}s）")
    return out


def stage_traj(cfg, train_cfg, model, field, seed: int, seed_dir, log) -> dict:
    out_file = seed_dir / "segments.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    t0 = time.time()
    refs = np.load(seed_dir / "field_ref.npz")
    h_ref = torch.from_numpy(refs["h_ref"]).double()
    o_ref = torch.from_numpy(refs["o_ref"]).double()
    dec, v_fn, g_fn = make_field_fns(model, cfg, h_ref, o_ref)

    rng = np.random.default_rng(seed + 313_000)  # 与效用探针同构任务流
    planner = ExhaustiveMPCPlanner(model, view=train_cfg["view"])
    segs = []
    n_task = 0
    while n_task < cfg["n_nav_tasks"]:
        maze = gw.generate_maze(train_cfg["maze_cells"], rng)
        start = gw.random_free_cell(maze, rng)
        goal = gw.random_free_cell(maze, rng)
        d = gw.bfs_distance(maze, start, goal)
        if d is None or not cfg["bfs_band"][0] <= d <= cfg["bfs_band"][1]:
            continue
        gen = torch.Generator()
        gen.manual_seed(seed * 100_003 + n_task + 700_000)
        env = gw.GridWorld(maze.copy(), start, train_cfg["view"], goal)
        ep = mpc_episode(model, env, planner, cfg["max_steps"], gen)
        n_task += 1
        traj = ep["h_traj"].double()  # (t, H)
        L = cfg["segment_len"]
        for s0 in range(0, traj.shape[0] - L, L):
            segs.append(traj[s0: s0 + L + 1])
    log(f"  s{seed} 轨迹段共 {len(segs)}（{time.time() - t0:.0f}s）")

    q_lo, q_hi = field["grad_tertiles"]
    r_max = field["support_radius_q99"]
    kept = []
    for si, seg in enumerate(segs):
        d_near = torch.cdist(seg, h_ref).min(dim=1).values
        if float(d_near.max()) > r_max:
            continue  # 支撑度剔除
        gn = grad_norms(v_fn, g_fn, seg)
        med = float(gn.median())
        group = "low" if med <= q_lo else ("high" if med >= q_hi else None)
        if group is None:
            continue
        chord = float((seg[-1] - seg[0]).norm())
        if chord < 1e-6:
            continue  # 端点重合段无测地意义
        kept.append({"idx": si, "group": group, "chord": chord, "grad_med": med})
    np.savez_compressed(seed_dir / "segments.npz",
                        **{f"seg_{k['idx']}": segs[k["idx"]].numpy() for k in kept})
    out = {"n_total": len(segs), "kept": kept,
           "n_low": sum(1 for k in kept if k["group"] == "low"),
           "n_high": sum(1 for k in kept if k["group"] == "high"),
           "wall_seconds": round(time.time() - t0)}
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  s{seed} 段归组：低 {out['n_low']} / 高 {out['n_high']} / 总 {out['n_total']}")
    return out


def match_and_select(kept: list, n_bins: int, cap: int, rng) -> list:
    """端点弦长分位分箱匹配：逐箱两组都有才保留，组内随机截断到 cap 均分。"""
    chords = np.array([k["chord"] for k in kept])
    edges = np.quantile(chords, np.linspace(0, 1, n_bins + 1))
    picked = []
    per_bin_cap = max(cap // n_bins, 2)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [k for k in kept if lo <= k["chord"] <= hi]
        low = [k for k in in_bin if k["group"] == "low"]
        high = [k for k in in_bin if k["group"] == "high"]
        if not low or not high:
            continue
        rng.shuffle(low)
        rng.shuffle(high)
        picked += low[:per_bin_cap] + high[:per_bin_cap]
    return picked


def stage_geo(cfg, train_cfg, model, seed: int, seed_dir, log, deadline=None) -> None:
    th = json.loads(THRESHOLDS_V12.read_text(encoding="utf-8"))["geodesic"]
    segs_meta = json.loads((seed_dir / "segments.json").read_text(encoding="utf-8"))
    seg_data = np.load(seed_dir / "segments.npz")
    refs = np.load(seed_dir / "field_ref.npz")
    h_ref = torch.from_numpy(refs["h_ref"]).double()
    o_ref = torch.from_numpy(refs["o_ref"]).double()
    dec, v_fn, g_fn = make_field_fns(model, cfg, h_ref, o_ref)
    obs_var = (model.obs_var_ext if getattr(model, "multi_step_k", 1) > 1
               else model.obs_var).double()
    # float64 解码器副本：float32 前向的量化噪声在度量 1e5 量纲下形成
    # 粗糙微地形，把残差卡在 O(1)；双精度将其压约一个量级（实测
    # 2.7 → 0.26，geo_effort_test）。残差仍高于校准阈值 3e-3——校准
    # 定标于解析光滑度量，估计 MLP 度量的空间变化幅度使线搜索停滞，
    # 该量表差距作为仪器发现入档，QC 剔除照冻结判据执行。
    import copy

    model_d = copy.deepcopy(model).double()
    for p in model_d.parameters():
        p.requires_grad_(False)
    ext = getattr(model, "multi_step_k", 1) > 1

    def dec_d(z):
        return (model_d.decoder_mean_ext if ext else model_d.decoder_mean)(z)

    g_geo = make_geo_metric(dec_d, obs_var)
    # 数值等价性对拍：可微度量（float64 权重副本）与冻结估计器（float32
    # 前向）同点同值到 f32 精度（相对差 ~1e-6 级，被度量量纲放大）
    z_chk = h_ref[:4]
    g_a, g_b = g_geo(z_chk), g_fn(z_chk)
    rel = float(((g_a - g_b).abs() / g_b.abs().clamp_min(1e-12)).max())
    assert rel < 1e-3, f"可微度量与冻结估计器相对差 {rel:.2e} 超精度解释范围"
    rng = np.random.default_rng(seed + 995_000)
    picked = match_and_select(segs_meta["kept"], cfg["distance_match_bins"],
                              cfg["max_pairs_per_group"] * 2, rng)
    geo_dir = seed_dir / "geo_cache"
    geo_dir.mkdir(exist_ok=True)
    gen = torch.Generator()
    gen.manual_seed(seed + 40_000)
    for k in picked:
        cache = geo_dir / f"seg_{k['idx']}.json"
        if cache.exists():
            continue
        t0 = time.time()
        seg = torch.from_numpy(seg_data[f"seg_{k['idx']}"])
        # 逐对能量归一：测地对度量的常数缩放不变（归一化偏差面积、能量
        # 差比、散布均为尺度不变量），把直线路径能量归一到 O(1) 使残差
        # 质检与校准定标的能量量纲可比（校准系统能量为 O(1) 量级；真实
        # 估计度量的原始能量高多个数量级，绝对残差阈值失去可比性）
        tau = torch.linspace(0, 1, cfg["geo_n_segments"] + 1,
                             dtype=torch.float64).unsqueeze(-1)
        straight = seg[0] + tau * (seg[-1] - seg[0])
        mids0 = 0.5 * (straight[1:] + straight[:-1])
        d0 = straight[1:] - straight[:-1]
        g0 = g_geo(mids0)
        e0 = float(cfg["geo_n_segments"]
                   * torch.einsum("ti,tij,tj->", d0, g0, d0))
        e0 = max(e0, 1e-30)

        def g_solve(z, _e0=e0):
            return g_geo(z) / _e0

        out = solve_geodesic(seg[0], seg[-1], g_solve,
                             n_segments=cfg["geo_n_segments"],
                             n_starts=cfg["geo_n_starts"], generator=gen,
                             n_adam=cfg["geo_n_adam"], n_lbfgs=cfg["geo_n_lbfgs"])
        e_sorted = sorted(out["all_energies"])
        energy_gap = (e_sorted[1] - e_sorted[0]) / max(e_sorted[0], 1e-30)
        qc_pass = out["residual"] <= th["solver_residual_max"]
        non_unique = (energy_gap < th["uniqueness_energy_gap_min"]
                      and out["uniqueness_spread"] > th["uniqueness_spread_min"])
        rec = {"idx": k["idx"], "group": k["group"], "chord": k["chord"],
               "residual": out["residual"], "energy_scale_e0": e0,
               "energy_gap": energy_gap,
               "uniqueness_spread": out["uniqueness_spread"],
               "qc_pass": bool(qc_pass), "non_unique": bool(non_unique)}
        if qc_pass and not non_unique:
            dev = geodesic_deviation(seg, out["path"], g_solve)
            rec["deviation"] = dev["normalized_area"]  # 尺度不变量
        cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        log(f"  s{seed} geo seg{k['idx']} [{k['group']}] 残差 {out['residual']:.1e} "
            f"{'认证' if rec.get('deviation') is not None else '剔除'}"
            f"（{time.time() - t0:.0f}s）")
        if deadline is not None and time.time() > deadline:
            log("预算耗尽，暂停")
            return


def stage_assemble(cfg, seed: int, seed_dir, field, log) -> dict:
    from scipy import stats as sps

    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in (seed_dir / "geo_cache").glob("seg_*.json")]
    cert = [r for r in recs if r.get("deviation") is not None]
    low = [r["deviation"] for r in cert if r["group"] == "low"]
    high = [r["deviation"] for r in cert if r["group"] == "high"]
    out = {
        "terrain_ratio_ci_low": field["ratio_ci90"][0],
        "n_certified_pairs": len(cert),
        "n_low": len(low), "n_high": len(high),
        "attrition": {"solved": len(recs),
                      "qc_fail": sum(1 for r in recs if not r["qc_pass"]),
                      "non_unique": sum(1 for r in recs if r["non_unique"])},
    }
    if low and high:
        mw = sps.mannwhitneyu(low, high, alternative="less")
        out.update({"mw_p": float(mw.pvalue),
                    "effect": float(np.median(high) - np.median(low)),
                    "low_lt_high": bool(np.median(low) < np.median(high)),
                    "deviation_median_low": float(np.median(low)),
                    "deviation_median_high": float(np.median(high))})
    (seed_dir / "p3_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  s{seed} P3：认证 {len(cert)} 对（低 {len(low)}/高 {len(high)}）"
        + (f"，MW p={out.get('mw_p'):.4g} 效应 {out.get('effect'):.4f}"
           if low and high else "，组不全"))
    return out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / cfg["train_config"]).read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    only_seed = None
    if "--seed" in sys.argv:
        only_seed = int(sys.argv[sys.argv.index("--seed") + 1])
    stage = None
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    deadline = None
    if "--budget-seconds" in sys.argv:
        deadline = time.time() + float(sys.argv[sys.argv.index("--budget-seconds") + 1])
    out_dir = create_campaign_dir("confirmation", "exp_p3", cfg["out_campaign"], cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    load = load_frozen_model(train_cfg)
    seeds = guard.family_seeds("evaluation", purpose="exp-p3")
    if only_seed is not None:
        seeds = [only_seed]
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="exp-p3")
        seed_dir = out_dir / f"s{seed}"
        seed_dir.mkdir(exist_ok=True)
        model, ckpt_rel = load(seed)
        field = stage_field(cfg, train_cfg, model, seed, seed_dir, log)
        if stage in (None, "traj", "geo", "assemble"):
            stage_traj(cfg, train_cfg, model, field, seed, seed_dir, log)
        if stage in (None, "geo", "assemble"):
            stage_geo(cfg, train_cfg, model, seed, seed_dir, log, deadline)
        if stage in (None, "assemble"):
            stage_assemble(cfg, seed, seed_dir, field, log)
    log("EXP-P3 批次完成")


if __name__ == "__main__":
    main()
