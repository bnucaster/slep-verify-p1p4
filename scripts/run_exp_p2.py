"""EXP-P2（确证阶段，计划 11.5）：轨迹集中于低作用量路径。

协议 v1.1 第 5 节 P2 判据（v1.2 继承）的统计量生产，系统 S2P。运行
纪律：评估族运行严格在 v1.2 外部时间戳之后。排序统计对温度公共正
因子不变，Â_OM 取 T̂=1（与描述阶段诊断一致；T̂ 实值由 EXP-P4' 另报）。

阶段（--stage field|main|ablation|novelty|assemble；main 逐块缓存）：

1. field——平稳 rollout：参照对（h, 未来 k 步观测拼接）、轨迹池、
   度量地板（描述阶段口径：查询点特征值中位 × metric_floor_rel）、
   漂移梯度占比（分层变量）。
2. main——逐观测轨迹配 n_surrogates 条端点匹配度量口径代理
   （smooth_random_surrogate_metric），Â_OM 标准化间隙与低于 Q1 指示；
   逐块缓存。
3. ablation——同批路径按常数势 Â_OM 重打分（动能项不变、漂移项置
   零），逐轨迹配对差 gap_full − gap_const，Wilcoxon 符号秩。
4. novelty——注入回合（t_inject 处切换未见迷宫拓扑）：逐步 OM 增量
   对新奇指示的逻辑回归（似然比检验 p），安慰剂 = 注入时刻随机平移。
5. assemble——judge_input 的 exp_p2 段（S2P:s<seed> 键）。

产物 results/confirmation/exp_p2/<out_campaign>/s<seed>/。

用法：.venv/Scripts/python.exe scripts/run_exp_p2.py [--seed N] [--stage ...]
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
from slep.estimators.drift import estimate_drift_knn, gradient_fraction
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.estimators.om_action import om_action
from slep.protocols.surrogates import smooth_random_surrogate_metric
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "exp_p2.yaml"


def load_frozen_model(train_cfg: dict, seed: int) -> S2WorldModel:
    stage = train_cfg.get("stage", "description")
    ck = torch.load(REPO_ROOT / "results" / stage / "s2_train" / train_cfg["campaign"]
                    / f"s{seed}" / "checkpoints" / f"ckpt_{train_cfg['train_steps']:06d}.pt",
                    weights_only=True)
    model = S2WorldModel(
        train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["embed_dim"],
        train_cfg["hidden_dim"], train_cfg["sigma_dec"], train_cfg.get("goal_sigma_dec"),
        multi_step_k=train_cfg.get("multi_step_k", 1),
        label_smooth_eps=train_cfg.get("label_smooth_eps", 0.0))
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def make_fns(model, cfg, h_ref, o_ref, g_floor: float):
    """估计器闭包（与 run_exp_p3 同构；g 加相对地板供代理与 OM 用）。"""
    obs_var = (model.obs_var_ext if getattr(model, "multi_step_k", 1) > 1
               else model.obs_var).double()
    eye = torch.eye(model.hidden_dim, dtype=torch.float64)

    def dec(z):
        if getattr(model, "multi_step_k", 1) > 1:
            return model.decoder_mean_ext(z.float()).double()
        return model.decoder_mean(z.float()).double()

    def v_fn(z):
        return potential.potential_knn(dec, obs_var, z, h_ref, o_ref,
                                       k=cfg["k_potential"])

    def v_const(z):
        return (z * 0.0).sum(dim=-1) if z.dim() > 1 else (z * 0.0).sum()

    def g_fn(z):
        return fisher_pullback_gaussian_batch(dec, z, obs_var) + g_floor * eye

    return dec, v_fn, v_const, g_fn


def stage_field(cfg, train_cfg, model, seed: int, seed_dir, log) -> dict:
    out_file = seed_dir / "field.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    t0 = time.time()
    rng = np.random.default_rng(seed + 996_000)
    obs_np, act_np = gw.collect_rollouts(cfg["n_episodes"], cfg["episode_len"],
                                         train_cfg["maze_cells"], train_cfg["view"], rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)
    k = getattr(model, "multi_step_k", 1)
    n_pos = hs.shape[1] - k + 1
    bi = cfg["burn_in"]
    h_pool = hs[:, bi:n_pos].reshape(-1, model.hidden_dim).double()
    o_ext = torch.cat([obs[:, bi + 1 + j: 1 + j + n_pos] for j in range(k)], dim=-1)
    o_pool = o_ext.reshape(-1, o_ext.shape[-1]).double()
    h_next = hs[:, bi + 1: n_pos + 1].reshape(-1, model.hidden_dim).double()

    perm = torch.from_numpy(rng.permutation(h_pool.shape[0]))
    idx_ref = perm[: cfg["n_ref"]]
    np.savez_compressed(seed_dir / "field_ref.npz",
                        h_ref=h_pool[idx_ref].numpy().astype(np.float32),
                        o_ref=o_pool[idx_ref].numpy().astype(np.float32))
    # 轨迹池：整回合隐轨迹存档（traj 阶段抽取）
    np.savez_compressed(seed_dir / "traj_pool.npz",
                        hs=hs[:, bi:].numpy().astype(np.float32))

    h_ref = h_pool[idx_ref]
    o_ref = o_pool[idx_ref]
    obs_var = (model.obs_var_ext if k > 1 else model.obs_var).double()

    def dec(z):
        return (model.decoder_mean_ext(z.float()).double() if k > 1
                else model.decoder_mean(z.float()).double())

    idx_q = perm[cfg["n_ref"]: cfg["n_ref"] + 1024]
    g_q = fisher_pullback_gaussian_batch(dec, h_pool[idx_q], obs_var).detach()
    g_floor = cfg["metric_floor_rel"] * float(torch.linalg.eigvalsh(g_q).median())

    idx_drift = perm[-cfg["n_drift_eval"]:]
    eye = torch.eye(model.hidden_dim, dtype=torch.float64)

    def g_fn(z):
        return fisher_pullback_gaussian_batch(dec, z, obs_var) + g_floor * eye

    drift = estimate_drift_knn(h_pool, h_next, 1.0, h_pool[idx_drift], k=cfg["k_drift"])
    frac = gradient_fraction(h_pool[idx_drift], drift, g_fn, seed=seed)
    out = {"g_floor": g_floor, "drift_fraction": frac["fraction"],
           "n_pool": int(h_pool.shape[0]), "wall_seconds": round(time.time() - t0)}
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  s{seed} field：漂移梯度占比 {frac['fraction']:.3f}，地板 {g_floor:.3e}"
        f"（{out['wall_seconds']}s）")
    return out


def score_block(cfg, model, field, seed: int, seed_dir, block_idx: int,
                ablation: bool, log) -> None:
    tag = "abl" if ablation else "main"
    cache = seed_dir / f"{tag}_cache" / f"b{block_idx:03d}.json"
    if cache.exists():
        return
    cache.parent.mkdir(exist_ok=True)
    t0 = time.time()
    refs = np.load(seed_dir / "field_ref.npz")
    h_ref = torch.from_numpy(refs["h_ref"]).double()
    o_ref = torch.from_numpy(refs["o_ref"]).double()
    dec, v_fn, v_const, g_fn = make_fns(model, cfg, h_ref, o_ref, field["g_floor"])
    v_use = v_const if ablation else v_fn
    pool = np.load(seed_dir / "traj_pool.npz")["hs"]
    n_ep = pool.shape[0]
    rng = np.random.default_rng(seed + 997_000)
    order = rng.permutation(n_ep)[: cfg["n_traj"]]
    lo = block_idx * cfg["block"]
    hi = min(lo + cfg["block"], cfg["n_traj"])
    rows = []
    for ti in range(lo, hi):
        e = int(order[ti])
        traj = torch.from_numpy(pool[e, : cfg["traj_len"] + 1]).double()
        gen_s = torch.Generator()
        gen_s.manual_seed(seed * 1_000_003 + ti)  # 主/消融同代理流（配对）
        a_obs = float(om_action(traj, 1.0, v_use, g_fn, 1.0, "mid"))
        a_surr = []
        for _ in range(cfg["n_surrogates"]):
            sur, _ = smooth_random_surrogate_metric(traj, g_fn, gen_s)
            a_surr.append(float(om_action(sur, 1.0, v_use, g_fn, 1.0, "mid")))
        a_t = torch.tensor(a_surr, dtype=torch.float64)
        iqr = float(torch.quantile(a_t, 0.75) - torch.quantile(a_t, 0.25))
        rows.append({"traj": ti,
                     "gap": (float(a_t.median()) - a_obs) / max(iqr, 1e-30),
                     "below_q1": bool(a_obs < float(torch.quantile(a_t, 0.25)))})
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    log(f"  s{seed} {tag} 块 {block_idx}（轨迹 {lo}-{hi - 1}）"
        f"（{time.time() - t0:.0f}s）")


def stage_novelty(cfg, train_cfg, model, field, seed: int, seed_dir, log) -> dict:
    out_file = seed_dir / "novelty.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    t0 = time.time()
    refs = np.load(seed_dir / "field_ref.npz")
    h_ref = torch.from_numpy(refs["h_ref"]).double()
    o_ref = torch.from_numpy(refs["o_ref"]).double()
    dec, v_fn, _, g_fn = make_fns(model, cfg, h_ref, o_ref, field["g_floor"])
    rng = np.random.default_rng(seed + 998_000)
    t_inj, win, bi = cfg["t_inject"], cfg["novelty_window"], cfg["burn_in"]
    incs, labels = [], []
    for _ in range(cfg["n_novelty_episodes"]):
        env = gw.make_episode_env(train_cfg["maze_cells"], train_cfg["view"], rng)
        obs_seq = [env.observe()]
        for t in range(cfg["episode_len"]):
            if t == t_inj:
                gw.inject_novelty(env, rng)
            obs_seq.append(env.step(int(rng.integers(0, 4))))
        obs = torch.from_numpy(np.stack(obs_seq)[None].astype(np.float32))
        act = torch.zeros(1, cfg["episode_len"], 4)
        act[0, torch.arange(cfg["episode_len"]),
            torch.from_numpy(rng.integers(0, 4, cfg["episode_len"]))] = 1.0
        with torch.no_grad():
            hs = model.hidden_trajectory(obs, act)[0].double()
        traj = hs[bi:]
        # 逐步 OM 增量：om_action 的单段值（中点式）
        for t in range(traj.shape[0] - 1):
            seg = traj[t: t + 2]
            incs.append(float(om_action(seg, 1.0, v_fn, g_fn, 1.0, "mid")))
            glob_t = t + bi
            labels.append(1 if t_inj <= glob_t < t_inj + win else 0)
    x = np.log1p(np.array(incs))
    y = np.array(labels)

    def logit_lr_p(x, y):
        from sklearn.linear_model import LogisticRegression

        xs = (x - x.mean()) / max(x.std(), 1e-12)
        clf = LogisticRegression(C=1e6, max_iter=1000).fit(xs[:, None], y)
        p1 = clf.predict_proba(xs[:, None])[:, 1].clip(1e-12, 1 - 1e-12)
        ll_full = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
        pbar = y.mean()
        ll_null = float(len(y) * (pbar * math.log(pbar) + (1 - pbar) * math.log(1 - pbar)))
        from scipy import stats as sps

        lr = 2 * (ll_full - ll_null)
        return float(clf.coef_[0][0]), float(sps.chi2.sf(max(lr, 0.0), 1))

    coef, p = logit_lr_p(x, y)
    rng_pl = np.random.default_rng(seed + 999_000)
    y_pl = np.array(labels)
    # 安慰剂：逐回合把注入时刻随机平移（标签循环移位）
    steps_per_ep = (cfg["episode_len"] - bi) - 1
    y_pl = y_pl.reshape(cfg["n_novelty_episodes"], steps_per_ep)
    for r in range(y_pl.shape[0]):
        y_pl[r] = np.roll(y_pl[r], int(rng_pl.integers(1, steps_per_ep)))
    coef_pl, p_pl = logit_lr_p(x, y_pl.reshape(-1))
    out = {"coef": coef, "logistic_p": p, "positive": bool(coef > 0),
           "placebo_coef": coef_pl, "placebo_p": p_pl,
           "n_steps": len(incs), "wall_seconds": round(time.time() - t0)}
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  s{seed} novelty：系数 {coef:.3f} p={p:.3g}（安慰剂 p={p_pl:.3g}）"
        f"（{out['wall_seconds']}s）")
    return out


def stage_assemble(cfg, seed: int, seed_dir, field, log) -> dict:
    from scipy import stats as sps

    def load_rows(tag):
        rows = []
        for p in sorted((seed_dir / f"{tag}_cache").glob("b*.json")):
            rows += json.loads(p.read_text(encoding="utf-8"))
        return {r["traj"]: r for r in rows}

    main = load_rows("main")
    abl = load_rows("abl")
    novelty = json.loads((seed_dir / "novelty.json").read_text(encoding="utf-8"))
    frac_below = float(np.mean([r["below_q1"] for r in main.values()]))
    common = sorted(set(main) & set(abl))
    diffs = [main[t]["gap"] - abl[t]["gap"] for t in common]
    w = sps.wilcoxon(diffs, alternative="greater") if diffs else None
    out = {
        "frac_below_q1": frac_below,
        "n_traj": len(main),
        "ablation": {"wilcoxon_p": float(w.pvalue) if w else None,
                     "median_diff": float(np.median(diffs)) if diffs else None,
                     "n_pairs": len(diffs)},
        "novelty": {"logistic_p": novelty["logistic_p"], "positive": novelty["positive"],
                    "placebo_p": novelty["placebo_p"]},
        "drift_fraction": field["drift_fraction"],
        "gap_median": float(np.median([r["gap"] for r in main.values()])),
    }
    (seed_dir / "p2_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  s{seed} P2：低于Q1 {frac_below:.3f}（n={len(main)}），消融中位差 "
        f"{out['ablation']['median_diff']} p={out['ablation']['wilcoxon_p']}，"
        f"新奇 p={novelty['logistic_p']:.3g}")
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
    out_dir = create_campaign_dir("confirmation", "exp_p2", cfg["out_campaign"], cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds = guard.family_seeds("evaluation", purpose="exp-p2")
    if only_seed is not None:
        seeds = [only_seed]
    n_blocks = math.ceil(cfg["n_traj"] / cfg["block"])
    for seed in seeds:
        guard.assert_seed_allowed(seed, purpose="exp-p2")
        seed_dir = out_dir / f"s{seed}"
        seed_dir.mkdir(exist_ok=True)
        model = load_frozen_model(train_cfg, seed)
        field = stage_field(cfg, train_cfg, model, seed, seed_dir, log)
        if stage in (None, "main", "ablation", "assemble"):
            for b in range(n_blocks):
                score_block(cfg, model, field, seed, seed_dir, b, ablation=False, log=log)
                if deadline is not None and time.time() > deadline:
                    log("预算耗尽，暂停")
                    return
        if stage in (None, "ablation", "assemble"):
            for b in range(n_blocks):
                score_block(cfg, model, field, seed, seed_dir, b, ablation=True, log=log)
                if deadline is not None and time.time() > deadline:
                    log("预算耗尽，暂停")
                    return
        if stage in (None, "novelty", "assemble"):
            stage_novelty(cfg, train_cfg, model, field, seed, seed_dir, log)
        if stage in (None, "assemble"):
            stage_assemble(cfg, seed, seed_dir, field, log)
    log("EXP-P2 批次完成")


if __name__ == "__main__":
    main()
