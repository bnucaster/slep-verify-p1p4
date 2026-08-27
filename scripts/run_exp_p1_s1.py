"""EXP-P1 × S1（确证阶段，计划 11.1）：能量随学习下降、拟合后压缩。

协议 v1.1 第 5 节 P1 的 S1 支线全量测量 + 冻结 judge（v1.1）判定。
复用 run_s1_curves 的三曲线机制，差异：评估族路径、冻结平台口径
（平滑 5 + 运行期窗长 W_run）、Ŝ 互证换协议口径（kNN 标准化 k=1）、
flow 主口径取加强档、几何门谱实测、峰值窗口增补层、判定链。

阶段（--stage 分块，逐检查点缓存断点续跑）：

1. gates——全部 10 run 末检查点：因子探针复合 Û（能力门）、σ_Û
   分折实测、ĝ 谱（几何门：log10 条件数中位、参与维中位）。门布尔
   按 docs/protocol_v1.1_thresholds.json 机械比对，产 gates.json。
2. curves——仅过门 run：逐检查点 Û、⟨V̂⟩、Ŝ 双路；缓存粒度
   （run × 检查点）。judge 对未过门 run 短路记未构成检验，无须全曲线。
3. assemble——平台检测（冻结检测器：θ=1%、连续 3 窗、平滑 5、
   W_run = max(3, ceil(12σ̂_rel²/(5θ²)))，σ̂_rel 取平滑均匀段末 16 点
   一阶差分 std/√2 除以段均值）；分析窗 [平台点, min(2×平台点, 末)]，
   未检出时峰值窗口增补层另列（峰值点 = 滑窗 W 均值最大的窗末检查
   点）；ΔŜ/Δ⟨V̂⟩ 与跨 run 噪声带（std × 2，thresholds p1）；双路形状
   （整条曲线 Spearman ρ + 分析窗 ΔŜ 同号）；末检查点静态 Î–V̂ 仿射
   T̂（只供 F̂_S，不进 P4 判定）；并入 judge_input.json 的 "S1" 键
   （gates_by_seed 逐 run 布尔），运行冻结 judge 写 metrics.json，
   票型聚合另存 aggregate.json。

产物 results/confirmation/exp_p1_s1/<out_campaign>/。

用法：.venv/Scripts/python.exe scripts/run_exp_p1_s1.py
  [--stage gates|curves|assemble] [--run b1_s5] [--budget-seconds N]
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
from slep.protocols.affine import affine_fit_report
from slep.protocols.judge import run_judge
from slep.protocols.plateau import REL_SLOPE_THRESHOLD, detect_plateau, moving_average
from slep.systems.s1_beta_vae import S1BetaVAE
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "exp_p1_s1.yaml"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "protocol_v1.1_thresholds.json"
JUDGE_INPUT = REPO_ROOT / "results" / "confirmation" / "judge_input.json"
METRICS = REPO_ROOT / "results" / "confirmation" / "metrics.json"


def load_data(train_cfg: dict):
    packed = np.load(REPO_ROOT / train_cfg["packed_cache"])
    lat = np.load(REPO_ROOT / train_cfg["data_file"])
    return packed, lat["latents_classes"], lat["latents_values"]


def batch_imgs(packed, idx) -> torch.Tensor:
    imgs = np.unpackbits(packed[idx], axis=1).astype(np.float32)
    return torch.from_numpy(imgs).reshape(-1, 1, 64, 64)


def load_model(train_cfg, beta: float, ckpt_path) -> S1BetaVAE:
    payload = torch.load(ckpt_path, weights_only=True)
    model = S1BetaVAE(train_cfg["latent_dim"], train_cfg["channels"],
                      train_cfg["fc_dim"], train_cfg["sigma_dec"], beta)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def encode(model, packed, idx, gen, sample: bool) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            mu, logvar = model.encoder(batch_imgs(packed, idx[i: i + 512]))
            if sample:
                eps = torch.randn(mu.shape, generator=gen, dtype=mu.dtype)
                outs.append(mu + torch.exp(0.5 * logvar) * eps)
            else:
                outs.append(mu)
    return torch.cat(outs)


def probes(z_train, z_test, cls_train, cls_test, val_train, val_test) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score

    zt, zv = z_train.numpy(), z_test.numpy()
    out = {}
    clf = LogisticRegression(max_iter=300).fit(zt, cls_train[:, 1])
    out["shape_acc"] = float(clf.score(zv, cls_test[:, 1]))
    out["_shape_pred"] = clf.predict(zv)
    for name, col in (("scale_r2", 2), ("posx_r2", 4), ("posy_r2", 5)):
        reg = Ridge().fit(zt, val_train[:, col])
        out[name] = float(r2_score(val_test[:, col], reg.predict(zv)))
        out[f"_{name}_pred"] = reg.predict(zv)
    out["u_composite"] = float(np.mean(
        [out["shape_acc"]] + [max(out[k], 0.0) for k in ("scale_r2", "posx_r2", "posy_r2")]))
    return out


def sigma_u_from_folds(probe_out, cls_test, val_test, n_folds: int) -> float:
    n = len(cls_test)
    fold = n // n_folds
    comps = []
    for f in range(n_folds):
        sl = slice(f * fold, (f + 1) * fold)
        shape_acc = float((probe_out["_shape_pred"][sl] == cls_test[sl, 1]).mean())
        r2s = []
        for name, col in (("scale_r2", 2), ("posx_r2", 4), ("posy_r2", 5)):
            pred = probe_out[f"_{name}_pred"][sl]
            y = val_test[sl, col]
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2s.append(max(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 0.0))
        comps.append(float(np.mean([shape_acc] + r2s)))
    return float(np.std(comps)) / math.sqrt(n_folds)


def sample_sets(cfg, packed, seed: int):
    rng = np.random.default_rng(seed + 777_000)
    total = (cfg["n_ref"] + cfg["n_query"] + cfg["n_probe_train"] + cfg["n_probe_test"]
             + cfg["n_flow_train"] + cfg["n_flow_val"] + cfg["n_entropy_eval"])
    pool = rng.choice(packed.shape[0], size=total, replace=False)
    ofs = [0]

    def take(n):
        out = pool[ofs[0]: ofs[0] + n]
        ofs[0] += n
        return out

    return {k: take(cfg[f"n_{k}"]) for k in
            ("ref", "query", "probe_train", "probe_test", "flow_train", "flow_val",
             "entropy_eval")}


def run_names(train_cfg) -> list[str]:
    seeds = guard.family_seeds("evaluation", purpose="exp-p1-s1")
    return [f"b{float(b):g}_s{s}" for b in train_cfg["betas"] for s in seeds]


def parse_run(name: str) -> tuple[float, int]:
    b, s = name.split("_")
    return float(b[1:]), int(s[1:])


def stage_gates(cfg, train_cfg, packed, classes, values, out_dir, log) -> dict:
    out_file = out_dir / "gates.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    th = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    final_step = train_cfg["train_steps"]
    cache_dir = out_dir / "gates_cache"
    cache_dir.mkdir(exist_ok=True)
    gates = {}
    for name in run_names(train_cfg):
        cache = cache_dir / f"{name}.json"
        if cache.exists():
            gates[name] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        beta, seed = parse_run(name)
        ckpt = (REPO_ROOT / "results/confirmation/s1_train" / cfg["train_campaign"]
                / name / "checkpoints" / f"ckpt_{final_step:06d}.pt")
        model = load_model(train_cfg, beta, ckpt)
        sets = sample_sets(cfg, packed, seed)
        gen = torch.Generator()
        gen.manual_seed(seed * 1000 + final_step)
        t0 = time.time()
        z_pt = encode(model, packed, sets["probe_train"], gen, sample=False)
        z_pv = encode(model, packed, sets["probe_test"], gen, sample=False)
        pr = probes(z_pt, z_pv, classes[sets["probe_train"]], classes[sets["probe_test"]],
                    values[sets["probe_train"]], values[sets["probe_test"]])
        sigma_u = sigma_u_from_folds(pr, classes[sets["probe_test"]],
                                     values[sets["probe_test"]], cfg["sigma_u_folds"])
        z_g = encode(model, packed, sets["query"][: cfg["n_geometry_query"]], gen,
                     sample=True).double()
        d_lat = train_cfg["latent_dim"]

        def dec(z, m=model, d=d_lat):
            # 形状鲁棒：单点 (d,) 与批量 (…, d) 皆可（jacfwd 逐点求导需要）
            flat = m.decoder(z.reshape(-1, d).float())
            return flat.reshape(*z.shape[:-1], -1).double()

        eig_parts = []
        for i in range(0, z_g.shape[0], 128):
            g = fisher_pullback_gaussian_batch(dec, z_g[i: i + 128],
                                               train_cfg["sigma_dec"] ** 2).detach()
            eig_parts.append(torch.linalg.eigvalsh(g).clamp_min(1e-30))
        eig = torch.cat(eig_parts)
        log10_cond = float(torch.log10(eig[:, -1] / eig[:, 0]).median())
        part = float((eig.sum(-1) ** 2 / (eig ** 2).sum(-1)).median())
        cap_ok = pr["u_composite"] >= th["capability_gate"]["s1_u_composite_min"]
        geo_ok = (log10_cond <= th["geometry_gate"]["log10_cond_median_max"]
                  and part >= th["geometry_gate"]["participation_dim_median_min"])
        gates[name] = {
            "u_composite": pr["u_composite"], "sigma_u": sigma_u,
            "shape_acc": pr["shape_acc"], "posx_r2": pr["posx_r2"],
            "log10_cond_median": log10_cond, "participation_median": part,
            "capability_ok": bool(cap_ok), "geometry_ok": bool(geo_ok),
            "gate_ok": bool(cap_ok and geo_ok),
        }
        cache.write_text(json.dumps(gates[name], ensure_ascii=False), encoding="utf-8")
        log(f"gates {name}: Û={pr['u_composite']:.3f} σ_Û={sigma_u:.4f} "
            f"cond={log10_cond:.2f} 参与维={part:.2f} → "
            f"{'过门' if gates[name]['gate_ok'] else '未过门'}（{time.time() - t0:.0f}s）")
    out_file.write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    return gates


def curve_row(cfg, train_cfg, packed, classes, values, beta, seed, ckpt_path, sets,
              th_p4) -> dict:
    model = load_model(train_cfg, beta, ckpt_path)
    step = int(torch.load(ckpt_path, weights_only=True)["step"])
    gen = torch.Generator()
    gen.manual_seed(seed * 1000 + step)
    z_ref = encode(model, packed, sets["ref"], gen, sample=True).double()
    z_q = encode(model, packed, sets["query"], gen, sample=True).double()
    z_pt = encode(model, packed, sets["probe_train"], gen, sample=False)
    z_pv = encode(model, packed, sets["probe_test"], gen, sample=False)
    z_ft = encode(model, packed, sets["flow_train"], gen, sample=True).double()
    z_fv = encode(model, packed, sets["flow_val"], gen, sample=True).double()
    z_ent = encode(model, packed, sets["entropy_eval"], gen, sample=True).double()

    pr = probes(z_pt, z_pv, classes[sets["probe_train"]], classes[sets["probe_test"]],
                values[sets["probe_train"]], values[sets["probe_test"]])
    x_ref = batch_imgs(packed, sets["ref"]).reshape(len(sets["ref"]), -1).double()

    def dec(z, m=model):
        return m.decoder(z.float()).reshape(z.shape[0], -1).double()

    v_parts = []
    for i in range(0, z_q.shape[0], 256):
        with torch.no_grad():
            v_parts.append(potential.potential_knn(
                dec, train_cfg["sigma_dec"] ** 2, z_q[i: i + 256], z_ref, x_ref,
                k=cfg["k_potential"]))
    v_hat = torch.cat(v_parts)
    fc = th_p4["flow_config"]
    fd, _ = fit_flow_density(z_ft, z_fv, seed=seed, n_couplings=fc["couplings"],
                             hidden=fc["hidden"], epochs=fc["epochs"],
                             batch_size=fc["batch"])
    s_flow = entropy_flow(fd, z_ent)
    s_knn = entropy_knn(z_ent, k=cfg["k_entropy_knn"], standardize=True)
    return {"step": step, "u_composite": pr["u_composite"],
            "v_mean": float(v_hat.mean()), "s_flow": s_flow, "s_knn": s_knn}


def stage_curves(cfg, train_cfg, packed, classes, values, gates, out_dir, log,
                 only_run=None, deadline=None) -> None:
    th_p4 = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["p4"]
    for name in run_names(train_cfg):
        if only_run is not None and name != only_run:
            continue
        if not gates[name]["gate_ok"]:
            continue  # judge 对未过门 run 短路，无须曲线
        beta, seed = parse_run(name)
        sets = sample_sets(cfg, packed, seed)
        cache_dir = out_dir / name / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpts = sorted((REPO_ROOT / "results/confirmation/s1_train" / cfg["train_campaign"]
                        / name / "checkpoints").glob("ckpt_*.pt"))
        for ck in ckpts:
            step = int(ck.stem.split("_")[1])
            cache = cache_dir / f"c{step:06d}.json"
            if cache.exists():
                continue
            t0 = time.time()
            row = curve_row(cfg, train_cfg, packed, classes, values, beta, seed, ck,
                            sets, th_p4)
            cache.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            log(f"  {name} ckpt {step}: U={row['u_composite']:.3f} "
                f"V={row['v_mean']:.1f} S={row['s_flow']:.2f}/{row['s_knn']:.2f} "
                f"({time.time() - t0:.0f}s)")
            if deadline is not None and time.time() > deadline:
                log("预算耗尽，暂停")
                return


def w_run_rule(u_uniform: torch.Tensor) -> tuple[int, float]:
    """S1 运行期窗长（协议 v1.1 阈值表 plateau.window_runtime_rule）。"""
    sm = moving_average(u_uniform, 5)
    tail = sm[-16:]
    sigma_rel = float(tail.diff().std() / math.sqrt(2) / tail.mean().abs().clamp_min(1e-12))
    w = max(3, math.ceil(12 * sigma_rel ** 2 / (5 * REL_SLOPE_THRESHOLD ** 2)))
    return w, sigma_rel


def analyze_run(cfg, train_cfg, name: str, out_dir) -> dict:
    cache_dir = out_dir / name / "cache"
    rows = sorted((json.loads(p.read_text(encoding="utf-8"))
                   for p in cache_dir.glob("c*.json")), key=lambda r: r["step"])
    uni = [r for r in rows if r["step"] % train_cfg["checkpoint_every"] == 0]
    steps_u = [r["step"] for r in uni]
    u = torch.tensor([r["u_composite"] for r in uni], dtype=torch.float64)
    w_run, sigma_rel = w_run_rule(u)
    # W_run 超视界（平滑序列容不下连续 3 窗）即数据条件不足：平台记
    # 未检出（协议降级预案 2），照走峰值窗口增补层
    n_smoothed = len(uni) - 4
    w_feasible = n_smoothed // w_run >= 3
    plateau_step = None
    if w_feasible:
        det = detect_plateau(u, window=w_run, smoothing=5)
        if det["plateau_index"] is not None:
            plateau_step = steps_u[det["plateau_index"] - 1]

    # 峰值点：滑窗均值最大的窗末检查点（thresholds p1.peak_window_rule）；
    # 定位窗取 s1_window_min=3（argmax 对窗长稳健，非斜率检验）
    w_peak = 3
    win = torch.tensor([float(u[i - w_peak + 1: i + 1].mean())
                        for i in range(w_peak - 1, len(uni))])
    peak_step = steps_u[int(win.argmax()) + w_peak - 1]

    def window_delta(anchor_step: int, key: str) -> float:
        end_step = min(2 * anchor_step, steps_u[-1])
        by = {r["step"]: r[key] for r in rows}
        grid = sorted(by)
        near = lambda t: min(grid, key=lambda s: abs(s - t))  # noqa: E731
        return by[near(end_step)] - by[near(anchor_step)]

    out = {
        "name": name, "w_run": w_run, "sigma_rel": sigma_rel,
        "w_feasible": w_feasible,
        "plateau_step": plateau_step, "peak_step": peak_step,
        "curve": {"steps": [r["step"] for r in rows],
                  "u": [r["u_composite"] for r in rows],
                  "v": [r["v_mean"] for r in rows],
                  "s_flow": [r["s_flow"] for r in rows],
                  "s_knn": [r["s_knn"] for r in rows]},
        "s_drop_peak_window": window_delta(peak_step, "s_flow"),
        "v_drop_peak_window": window_delta(peak_step, "v_mean"),
    }
    if plateau_step is not None:
        out["s_drop"] = window_delta(plateau_step, "s_flow")
        out["v_drop"] = window_delta(plateau_step, "v_mean")
    from scipy import stats as sps

    rho = float(sps.spearmanr(out["curve"]["s_flow"], out["curve"]["s_knn"]).statistic)
    anchor = plateau_step if plateau_step is not None else peak_step
    d_flow = window_delta(anchor, "s_flow")
    d_knn = window_delta(anchor, "s_knn")
    out["dual_shape"] = {"rho": rho,
                        "delta_same_sign": bool(d_flow * d_knn > 0),
                        "delta_flow": d_flow, "delta_knn": d_knn}
    return out


def static_temperature(cfg, train_cfg, packed, name: str, log) -> dict:
    """S1 静态支线：末检查点 Î–V̂ 仿射 T̂（只供 F̂_S，不进 P4 判定）。"""
    beta, seed = parse_run(name)
    th_p4 = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["p4"]
    final = train_cfg["train_steps"]
    ckpt = (REPO_ROOT / "results/confirmation/s1_train" / cfg["train_campaign"]
            / name / "checkpoints" / f"ckpt_{final:06d}.pt")
    model = load_model(train_cfg, beta, ckpt)
    sets = sample_sets(cfg, packed, seed)
    gen = torch.Generator()
    gen.manual_seed(seed * 1000 + final + 1)
    z_ref = encode(model, packed, sets["ref"], gen, sample=True).double()
    z_q = encode(model, packed, sets["query"], gen, sample=True).double()
    z_ft = encode(model, packed, sets["flow_train"], gen, sample=True).double()
    z_fv = encode(model, packed, sets["flow_val"], gen, sample=True).double()
    x_ref = batch_imgs(packed, sets["ref"]).reshape(len(sets["ref"]), -1).double()
    d_lat = train_cfg["latent_dim"]

    def dec(z, m=model, d=d_lat):
        flat = m.decoder(z.reshape(-1, d).float())
        return flat.reshape(*z.shape[:-1], -1).double()

    v_parts = []
    for i in range(0, z_q.shape[0], 256):
        with torch.no_grad():
            v_parts.append(potential.potential_knn(
                dec, train_cfg["sigma_dec"] ** 2, z_q[i: i + 256], z_ref, x_ref,
                k=cfg["k_potential"]))
    v_q = torch.cat(v_parts)
    logdet_parts = []
    for i in range(0, z_q.shape[0], 128):
        g = fisher_pullback_gaussian_batch(dec, z_q[i: i + 128],
                                           train_cfg["sigma_dec"] ** 2).detach()
        logdet_parts.append(torch.linalg.slogdet(g).logabsdet)
    logdet = torch.cat(logdet_parts)
    fc = th_p4["flow_config"]
    fd, _ = fit_flow_density(z_ft, z_fv, seed=seed, n_couplings=fc["couplings"],
                             hidden=fc["hidden"], epochs=fc["epochs"],
                             batch_size=fc["batch"])
    i_corr = -fd.log_prob(z_q) + 0.5 * logdet
    fit = affine_fit_report(v_q, i_corr)
    log(f"  静态 T̂ {name}: 斜率 {fit['slope']:.3f} T̂={fit['temperature_hat']:.3f} "
        f"R²={fit['r_squared']:.3f}")
    return {k: fit[k] for k in ("slope", "temperature_hat", "r_squared",
                                "p_lack_of_fit", "curvature_effect_ratio")}


def stage_assemble(cfg, train_cfg, packed, gates, out_dir, log) -> None:
    th = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    gated = [n for n in run_names(train_cfg) if gates[n]["gate_ok"]]
    analyses = {n: analyze_run(cfg, train_cfg, n, out_dir) for n in gated}
    # 跨 run 噪声带：过门 run 分析窗变化量 std × 乘数（thresholds p1）
    mult = th["p1"]["s_drop_band_multiplier"]
    anchors = {n: ("s_drop" if analyses[n].get("s_drop") is not None else
                   "s_drop_peak_window") for n in gated}
    s_deltas = [analyses[n][anchors[n]] for n in gated]
    v_deltas = [analyses[n].get("v_drop", analyses[n]["v_drop_peak_window"]) for n in gated]
    s_band = mult * float(np.std(s_deltas)) if len(gated) >= 2 else None
    v_band = mult * float(np.std(v_deltas)) if len(gated) >= 2 else None

    statics = {n: static_temperature(cfg, train_cfg, packed, n, log) for n in gated}

    w_max_rule = {n: 0.01 * math.sqrt(analyses[n]["w_run"]) / (3 * math.sqrt(12))
                  for n in gated}
    seeds_inp = {}
    for n in run_names(train_cfg):
        if n not in analyses:
            seeds_inp[n] = {}  # 未过门：judge 短路，无需字段
            continue
        a = analyses[n]
        v_delta = a.get("v_drop")
        v_trend = None
        if v_delta is not None and v_band is not None:
            v_trend = ("down_beyond_band" if v_delta < -v_band else
                       ("up_beyond_band" if v_delta > v_band else "flat_within_band"))
        seeds_inp[n] = {
            "dual_shape": {"rho": a["dual_shape"]["rho"],
                           "delta_same_sign": a["dual_shape"]["delta_same_sign"]},
            "sigma_u": gates[n]["sigma_u"],
            "sigma_u_max": w_max_rule[n],
            "plateau_step": a["plateau_step"],
            "peak_step": a["peak_step"],
            "s_drop": a.get("s_drop"),
            "s_band": s_band,
            "v_post_trend": v_trend,
            "s_drop_peak_window": a["s_drop_peak_window"],
        }

    best = max(run_names(train_cfg), key=lambda n: gates[n]["u_composite"])
    inp = {"systems": {}}
    if JUDGE_INPUT.exists():
        inp = json.loads(JUDGE_INPUT.read_text(encoding="utf-8"))
    inp["systems"]["S1"] = {
        "gates": {
            "capability": {"value": gates[best]["u_composite"], "system": "S1",
                           "note": "代表值取最高 run；逐 run 门以 gates_by_seed 为准"},
            "geometry": {"log10_cond_median": gates[best]["log10_cond_median"],
                         "participation_median": gates[best]["participation_median"]},
        },
        "gates_by_seed": {n: gates[n]["gate_ok"] for n in run_names(train_cfg)},
        "gates_detail": gates,
        "exp_p1": {"seeds": seeds_inp},
        "static_p4_branch": statics,
        "bands": {"s_band": s_band, "v_band": v_band, "multiplier": mult},
    }
    JUDGE_INPUT.write_text(json.dumps(inp, ensure_ascii=False, indent=2), encoding="utf-8")
    out = run_judge(JUDGE_INPUT, METRICS)
    p1 = out["systems"]["S1"]["P1"]
    (out_dir / "aggregate.json").write_text(json.dumps(
        {"prediction": "P1", "system": "S1", "judge": p1,
         "analyses": {n: {k: v for k, v in analyses[n].items() if k != "curve"}
                      for n in gated}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"P1 × S1 判定：{p1['verdict']}（{p1.get('reason', '')}）")
    for n, r in p1.get("per_seed", {}).items():
        log(f"  {n}: {r.get('verdict')} {r.get('reason', '')} "
            f"增补峰值层={r.get('supplement_peak_window')}")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / cfg["train_config"]).read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    stage = None
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    only_run = None
    if "--run" in sys.argv:
        only_run = sys.argv[sys.argv.index("--run") + 1]
    deadline = None
    if "--budget-seconds" in sys.argv:
        deadline = time.time() + float(sys.argv[sys.argv.index("--budget-seconds") + 1])
    out_dir = create_campaign_dir("confirmation", "exp_p1_s1", cfg["out_campaign"], cfg)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    packed, classes, values = load_data(train_cfg)
    gates = stage_gates(cfg, train_cfg, packed, classes, values, out_dir, log)
    if stage in (None, "curves"):
        stage_curves(cfg, train_cfg, packed, classes, values, gates, out_dir, log,
                     only_run, deadline)
    if stage in (None, "assemble"):
        stage_assemble(cfg, train_cfg, packed, gates, out_dir, log)


if __name__ == "__main__":
    main()
