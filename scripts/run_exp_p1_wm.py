"""EXP-P1 × 世界模型系统（S2P / S3；确证 11.5）：能量随学习下降。

与 run_exp_p1_s1 的关系：判定结构完全同构（门检 → 逐检查点曲线 →
冻结平台口径 + 峰值窗口增补 + 双路形状 + judge 字段），差异在系统层
——Û 取确定性导航探针缓存（σ_Û 重复噪声为零），V̂/Ŝ 在世界模型隐态
上逐检查点现算（S2P/S3 均走扩展解码口径）。

v1.2 判据（docs/protocol_v1.2_thresholds.json）：能力门按系统表、
几何门 7.0、双路符号判据带幅度地板（max|Δ| < 0.25×s_band 时符号
分歧不记仪器未过——在本汇总层实现，judge 函数不改）。

阶段：gates（末检查点能力池化 + 几何谱 + 逐种子门布尔）→ curves
（过门种子逐检查点 V̂/Ŝ 双路，缓存粒度 种子 × 检查点）→ assemble
（分析 + judge_input 并入 "<system_class>" 键；判定统一由末段
judge_v12 全量复判执行，本脚本不调 judge）。

用法：.venv/Scripts/python.exe scripts/run_exp_p1_wm.py --config configs/exp_p1_s2p.yaml
  [--seed N] [--stage gates|curves|assemble] [--budget-seconds N]
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
from slep.estimators.entropy import entropy_flow, entropy_knn
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.plateau import REL_SLOPE_THRESHOLD, detect_plateau, moving_average
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel
from slep.systems.s3_transformer import S3TransformerWM
from slep.utils.runs import REPO_ROOT, create_campaign_dir

THRESHOLDS_V12 = REPO_ROOT / "docs" / "protocol_v1.2_thresholds.json"
JUDGE_INPUT = REPO_ROOT / "results" / "confirmation" / "judge_input.json"


def build_model(system_class: str, train_cfg: dict):
    if system_class == "S2P":
        return S2WorldModel(
            train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["embed_dim"],
            train_cfg["hidden_dim"], train_cfg["sigma_dec"], train_cfg.get("goal_sigma_dec"),
            multi_step_k=train_cfg.get("multi_step_k", 1),
            label_smooth_eps=train_cfg.get("label_smooth_eps", 0.0))
    if system_class == "S3":
        return S3TransformerWM(
            train_cfg["obs_dim"], train_cfg["action_dim"], train_cfg["d_model"],
            train_cfg["n_layers"], train_cfg["n_heads"], train_cfg["ff_dim"],
            max_len=train_cfg["episode_len"] + 4, sigma_dec=train_cfg["sigma_dec"],
            goal_sigma_dec=train_cfg.get("goal_sigma_dec"),
            multi_step_k=train_cfg.get("multi_step_k", 1))
    raise ValueError(system_class)


def train_dir_name(system_class: str) -> str:
    return "s2_train" if system_class == "S2P" else "s3_train"


def load_ckpt(cfg, train_cfg, seed: int, step: int):
    p = (REPO_ROOT / "results" / train_cfg.get("stage", "confirmation")
         / train_dir_name(cfg["system_class"]) / train_cfg["campaign"]
         / f"s{seed}" / "checkpoints" / f"ckpt_{step:06d}.pt")
    model = build_model(cfg["system_class"], train_cfg)
    model.load_state_dict(torch.load(p, weights_only=True)["model"])
    model.eval()
    return model


def model_io(model):
    ext = getattr(model, "multi_step_k", 1) > 1
    obs_var = (model.obs_var_ext if ext else model.obs_var).double()

    def dec(z):
        return ((model.decoder_mean_ext if ext else model.decoder_mean)(
            z.float()).double())

    return dec, obs_var, ext


def pool_pairs(model, obs, hs, bi: int):
    k = getattr(model, "multi_step_k", 1)
    n_pos = hs.shape[1] - k + 1
    h = hs[:, bi:n_pos].reshape(-1, model.hidden_dim)
    o_ext = torch.cat([obs[:, bi + 1 + j: 1 + j + n_pos] for j in range(k)], dim=-1)
    return h.double(), o_ext.reshape(-1, o_ext.shape[-1]).double()


def ckpt_steps(train_cfg) -> list[int]:
    early = [1, 32, 256]
    uniform = list(range(train_cfg["checkpoint_every"], train_cfg["train_steps"] + 1,
                         train_cfg["checkpoint_every"]))
    return early + uniform


def u_from_probe(cfg, seed: int, step: int) -> float:
    cache = (REPO_ROOT / cfg["u_curve_campaign"] / "cache" / f"s{seed}_c{step:06d}.json")
    d = json.loads(cache.read_text(encoding="utf-8"))
    return d["success"] / d["n"]


def collect_stream(cfg, train_cfg, seed: int):
    rng = np.random.default_rng(seed + 991_000)
    obs_np, act_np = gw.collect_rollouts(cfg["n_episodes"], train_cfg["episode_len"],
                                         train_cfg["maze_cells"], train_cfg["view"], rng)
    return torch.from_numpy(obs_np), torch.from_numpy(act_np), rng


def stage_gates(cfg, train_cfg, out_dir, log) -> dict:
    out_file = out_dir / "gates.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    th = json.loads(THRESHOLDS_V12.read_text(encoding="utf-8"))
    seeds = guard.family_seeds("evaluation", purpose="exp-p1-wm")
    final = train_cfg["train_steps"]
    cap_pooled = float(np.mean([u_from_probe(cfg, s, final) for s in seeds]))
    gates = {}
    for seed in seeds:
        t0 = time.time()
        model = load_ckpt(cfg, train_cfg, seed, final)
        dec, obs_var, _ = model_io(model)
        obs, act, rng = collect_stream(cfg, train_cfg, seed)
        with torch.no_grad():
            hs = model.hidden_trajectory(obs, act)
        h_pool, _ = pool_pairs(model, obs, hs, cfg["burn_in"])
        idx = torch.from_numpy(rng.permutation(h_pool.shape[0])[: cfg["n_geometry_query"]])
        eig_parts = []
        for i in range(0, len(idx), 128):
            g = fisher_pullback_gaussian_batch(dec, h_pool[idx[i: i + 128]],
                                               obs_var).detach()
            eig_parts.append(torch.linalg.eigvalsh(g).clamp_min(1e-30))
        eig = torch.cat(eig_parts)
        cond = float(torch.log10(eig[:, -1] / eig[:, 0]).median())
        part = float((eig.sum(-1) ** 2 / (eig ** 2).sum(-1)).median())
        cap_ok = cap_pooled >= th["capability_gate"]["by_system"][cfg["system_class"]]
        geo_ok = (cond <= th["geometry_gate"]["log10_cond_median_max"]
                  and part >= th["geometry_gate"]["participation_dim_median_min"])
        gates[f"s{seed}"] = {
            "u_final": u_from_probe(cfg, seed, final), "capability_pooled": cap_pooled,
            "log10_cond_median": cond, "participation_median": part,
            "gate_ok": bool(cap_ok and geo_ok),
        }
        log(f"gates s{seed}: 池化Û={cap_pooled:.3f} cond={cond:.2f} 参与维={part:.2f}"
            f" → {'过门' if gates[f's{seed}']['gate_ok'] else '未过门'}"
            f"（{time.time() - t0:.0f}s）")
    out_file.write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    return gates


def curve_row(cfg, train_cfg, seed: int, step: int, obs, act, rng_ids) -> dict:
    model = load_ckpt(cfg, train_cfg, seed, step)
    dec, obs_var, _ = model_io(model)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)
    h_pool, o_pool = pool_pairs(model, obs, hs, cfg["burn_in"])
    perm = rng_ids
    n_ref, n_q = cfg["n_ref"], cfg["n_query"]
    n_ft, n_fv, n_ent = cfg["n_flow_train"], cfg["n_flow_val"], cfg["n_entropy_eval"]
    h_ref, o_ref = h_pool[perm[:n_ref]], o_pool[perm[:n_ref]]
    h_q = h_pool[perm[n_ref: n_ref + n_q]]
    idx_flow = perm[n_ref + n_q: n_ref + n_q + n_ft + n_fv]
    h_ent = h_pool[perm[n_ref + n_q + n_ft + n_fv: n_ref + n_q + n_ft + n_fv + n_ent]]

    v_parts = []
    for i in range(0, h_q.shape[0], 256):
        with torch.no_grad():
            v_parts.append(potential.potential_knn(
                dec, obs_var, h_q[i: i + 256], h_ref, o_ref, k=cfg["k_potential"]))
    v_hat = torch.cat(v_parts)
    th_p4 = json.loads(THRESHOLDS_V12.read_text(encoding="utf-8"))["p4"]
    fc = th_p4["flow_config"]
    fd, _ = fit_flow_density(h_pool[idx_flow[:n_ft]], h_pool[idx_flow[n_ft:]], seed=seed,
                             n_couplings=fc["couplings"], hidden=fc["hidden"],
                             epochs=fc["epochs"], batch_size=fc["batch"])
    s_flow = entropy_flow(fd, h_ent)
    s_knn = entropy_knn(h_ent, k=cfg["k_entropy_knn"], standardize=True)
    return {"step": step, "u_composite": u_from_probe(cfg, seed, step),
            "v_mean": float(v_hat.mean()), "s_flow": s_flow, "s_knn": s_knn}


def stage_curves(cfg, train_cfg, gates, out_dir, log, only_seed=None, deadline=None):
    seeds = guard.family_seeds("evaluation", purpose="exp-p1-wm")
    for seed in seeds:
        if only_seed is not None and seed != only_seed:
            continue
        if not gates[f"s{seed}"]["gate_ok"]:
            continue
        cache_dir = out_dir / f"s{seed}" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        obs, act, rng = collect_stream(cfg, train_cfg, seed)
        n_pool_est = cfg["n_episodes"] * (train_cfg["episode_len"]
                                          - cfg["burn_in"]
                                          - train_cfg.get("multi_step_k", 1) + 1)
        rng_ids = torch.from_numpy(rng.permutation(n_pool_est))
        for step in ckpt_steps(train_cfg):
            cache = cache_dir / f"c{step:06d}.json"
            if cache.exists():
                continue
            t0 = time.time()
            row = curve_row(cfg, train_cfg, seed, step, obs, act, rng_ids)
            cache.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            log(f"  s{seed} ckpt {step}: U={row['u_composite']:.3f} "
                f"V={row['v_mean']:.1f} S={row['s_flow']:.2f}/{row['s_knn']:.2f} "
                f"({time.time() - t0:.0f}s)")
            if deadline is not None and time.time() > deadline:
                log("预算耗尽，暂停")
                return


def w_run_rule(u_uniform: torch.Tensor, w_min: int) -> tuple[int, float]:
    sm = moving_average(u_uniform, 5)
    tail = sm[-16:]
    sigma_rel = float(tail.diff().std() / math.sqrt(2)
                      / tail.mean().abs().clamp_min(1e-12))
    w = max(w_min, math.ceil(12 * sigma_rel ** 2 / (5 * REL_SLOPE_THRESHOLD ** 2)))
    return w, sigma_rel


def analyze_seed(cfg, train_cfg, seed: int, out_dir, w_min: int) -> dict:
    cache_dir = out_dir / f"s{seed}" / "cache"
    rows = sorted((json.loads(p.read_text(encoding="utf-8"))
                   for p in cache_dir.glob("c*.json")), key=lambda r: r["step"])
    uni = [r for r in rows if r["step"] % train_cfg["checkpoint_every"] == 0]
    steps_u = [r["step"] for r in uni]
    u = torch.tensor([r["u_composite"] for r in uni], dtype=torch.float64)
    w_run, sigma_rel = w_run_rule(u, w_min)
    n_smoothed = len(uni) - 4
    plateau_step = None
    if n_smoothed // w_run >= 3:
        det = detect_plateau(u, window=w_run, smoothing=5)
        if det["plateau_index"] is not None:
            plateau_step = steps_u[det["plateau_index"] - 1]
    w_peak = w_min
    win = torch.tensor([float(u[i - w_peak + 1: i + 1].mean())
                        for i in range(w_peak - 1, len(uni))])
    peak_step = steps_u[int(win.argmax()) + w_peak - 1]

    def window_delta(anchor: int, key: str) -> float:
        end = min(2 * anchor, steps_u[-1])
        by = {r["step"]: r[key] for r in rows}
        near = lambda t: min(by, key=lambda s: abs(s - t))  # noqa: E731
        return by[near(end)] - by[near(anchor)]

    from scipy import stats as sps

    out = {"seed": seed, "w_run": w_run, "sigma_rel": sigma_rel,
           "plateau_step": plateau_step, "peak_step": peak_step,
           "s_drop_peak_window": window_delta(peak_step, "s_flow"),
           "v_drop_peak_window": window_delta(peak_step, "v_mean"),
           "rho": float(sps.spearmanr([r["s_flow"] for r in rows],
                                      [r["s_knn"] for r in rows]).statistic)}
    anchor = plateau_step if plateau_step is not None else peak_step
    out["delta_flow"] = window_delta(anchor, "s_flow")
    out["delta_knn"] = window_delta(anchor, "s_knn")
    if plateau_step is not None:
        out["s_drop"] = window_delta(plateau_step, "s_flow")
        out["v_drop"] = window_delta(plateau_step, "v_mean")
    return out


def stage_assemble(cfg, train_cfg, gates, out_dir, log) -> None:
    th = json.loads(THRESHOLDS_V12.read_text(encoding="utf-8"))
    w_min = th["plateau"]["s2p_window" if cfg["system_class"] == "S2P" else "s3_window"]
    seeds = guard.family_seeds("evaluation", purpose="exp-p1-wm")
    gated = [s for s in seeds if gates[f"s{s}"]["gate_ok"]]
    analyses = {s: analyze_seed(cfg, train_cfg, s, out_dir, w_min) for s in gated}
    mult = th["p1"]["s_drop_band_multiplier"]
    floor_frac = th["p1"]["dual_delta_floor_frac"]
    s_deltas = [analyses[s].get("s_drop", analyses[s]["s_drop_peak_window"])
                for s in gated]
    v_deltas = [analyses[s].get("v_drop", analyses[s]["v_drop_peak_window"])
                for s in gated]
    s_band = mult * float(np.std(s_deltas)) if len(gated) >= 2 else None
    v_band = mult * float(np.std(v_deltas)) if len(gated) >= 2 else None

    seeds_inp = {}
    for s in seeds:
        if s not in analyses:
            seeds_inp[f"s{s}"] = {}
            continue
        a = analyses[s]
        same_sign = bool(a["delta_flow"] * a["delta_knn"] > 0)
        if (s_band is not None
                and max(abs(a["delta_flow"]), abs(a["delta_knn"])) < floor_frac * s_band):
            same_sign = True  # v1.2 幅度地板：近零变化量的符号分歧不记仪器未过
        v_delta = a.get("v_drop")
        v_trend = None
        if v_delta is not None and v_band is not None:
            v_trend = ("down_beyond_band" if v_delta < -v_band else
                       ("up_beyond_band" if v_delta > v_band else "flat_within_band"))
        seeds_inp[f"s{s}"] = {
            "dual_shape": {"rho": a["rho"], "delta_same_sign": same_sign},
            "sigma_u": 0.0,  # 确定性口径
            "sigma_u_max": 0.01 * math.sqrt(a["w_run"]) / (3 * math.sqrt(12)),
            "plateau_step": a["plateau_step"], "peak_step": a["peak_step"],
            "s_drop": a.get("s_drop"), "s_band": s_band, "v_post_trend": v_trend,
            "s_drop_peak_window": a["s_drop_peak_window"],
        }
    inp = {"systems": {}}
    if JUDGE_INPUT.exists():
        inp = json.loads(JUDGE_INPUT.read_text(encoding="utf-8"))
    best = max(gated, key=lambda s: gates[f"s{s}"]["u_final"]) if gated else seeds[0]
    block = inp["systems"].setdefault(cfg["system_class"], {})
    block["gates"] = {
        "capability": {"value": gates[f"s{best}"]["capability_pooled"],
                       "system": cfg["system_class"]},
        "geometry": {"log10_cond_median": gates[f"s{best}"]["log10_cond_median"],
                     "participation_median": gates[f"s{best}"]["participation_median"]},
    }
    block["gates_by_seed"] = {f"s{s}": gates[f"s{s}"]["gate_ok"] for s in seeds}
    block["gates_detail"] = gates
    block["exp_p1"] = {"seeds": seeds_inp}
    block["bands"] = {"s_band": s_band, "v_band": v_band, "multiplier": mult,
                      "dual_delta_floor_frac": floor_frac}
    JUDGE_INPUT.write_text(json.dumps(inp, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "p1_analyses.json").write_text(
        json.dumps({str(k): v for k, v in analyses.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log(f"{cfg['system_class']} P1 汇总：过门 {len(gated)}/5，"
        f"s_band={s_band}，逐种子平台 "
        f"{[analyses[s]['plateau_step'] for s in gated]}")


def main() -> None:
    cfg_file = sys.argv[sys.argv.index("--config") + 1]
    cfg = yaml.safe_load((REPO_ROOT / cfg_file).read_text(encoding="utf-8"))
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
    out_dir = create_campaign_dir("confirmation", f"exp_p1_{cfg['system_class'].lower()}",
                                  cfg["out_campaign"], cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    gates = stage_gates(cfg, train_cfg, out_dir, log)
    if stage in (None, "curves", "assemble"):
        stage_curves(cfg, train_cfg, gates, out_dir, log, only_seed, deadline)
    if stage in (None, "assemble"):
        stage_assemble(cfg, train_cfg, gates, out_dir, log)


if __name__ == "__main__":
    main()
