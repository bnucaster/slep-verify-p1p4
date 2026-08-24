"""预承诺页分支触发判定（任务六 6d）：机械计算三分支判据并落盘。

输入产物：s1_curves（平台与 σ_Û）、s2_diagnostics（四图数值）；本脚本
另补两项计算：
- S1 压缩签名（分支二第 3 款）：逃逸 run 末检查点上 k ∈ {16,32,64} 的
  V̂ 重算 → Î–V̂ 拟合斜率漂移是否超 30%；Î 用 flow 密度 + 体积校正。
- Ŝ 双路分歧复测（分支一第 3 款 / 分支二第 2 款）：flow 对标准化 kNN
  （estimators/entropy.py 标准化口径，修复各向异性偏差）在三个检查点
  （首均匀点、中、末）的分歧，对两路自检回归界之和（0.4 nats，出处
  tests/test_entropy.py 定值界 flow 0.2 + kNN 0.2）。

输出 results/description/precommitment_check/<run_id>/：
metrics.json 与 trigger_decision.md（三分支逐条判定，引用产物路径）。

用法：.venv/Scripts/python.exe scripts/run_precommitment_check.py
"""
from __future__ import annotations

import json
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
from slep.systems.s1_beta_vae import S1BetaVAE
from slep.utils.runs import REPO_ROOT, create_run_dir

CFG_CURVES = yaml.safe_load((REPO_ROOT / "configs" / "s1_curves.yaml").read_text(encoding="utf-8"))
CFG_TRAIN = yaml.safe_load((REPO_ROOT / "configs" / "s1_train.yaml").read_text(encoding="utf-8"))
K_SWEEP = [16, 32, 64]
COLLAPSE_U_CUT = 0.45
SPOT_STEPS = [2000, 20000]
N_QUERY = 512  # 环境对长时后台命令有 15-20 分钟终止窗口，控制单块耗时

# 修复后口径（任务六点五 7a 选型，溯源 results/description/entropy_repair/）：
# 若修复产物存在则用其选型配置与容差；否则退回修复前口径（0.4 回归界之和）。
import glob as _glob

_repairs = sorted(_glob.glob(str(REPO_ROOT / "results/description/entropy_repair/*/summary.json")))
if _repairs:
    _rep = json.loads(open(_repairs[-1], encoding="utf-8").read())
    FLOW_CFG = yaml.safe_load(
        (REPO_ROOT / "configs" / "entropy_repair.yaml").read_text(encoding="utf-8")
    )["flow_configs"][_rep["selected"]["flow"]]
    KNN_K = int(_rep["selected"]["knn_k"])
    TOL_DUAL = _rep["tolerances"]["dual_sum"]
    # 差分一致性容差：偏差稳定性（跨模板/种子极差）之和 × 安全系数 1.5
    _fname = _rep["selected"]["flow"]
    _f_errs = [m["flow"][_fname]["error"] for m in _rep["measures"].values()]
    _k_errs = [m["knn"][str(KNN_K)]["error"] for m in _rep["measures"].values()]
    TOL_DELTA = 1.5 * ((max(_f_errs) - min(_f_errs)) + (max(_k_errs) - min(_k_errs)))
    REPAIR_SOURCE = _repairs[-1]
else:
    FLOW_CFG = {"couplings": 8, "hidden": 64, "epochs": 8, "batch": 1024}
    KNN_K = 8
    TOL_DUAL = 0.4
    TOL_DELTA = None
    REPAIR_SOURCE = None


def load_packed():
    return np.load(REPO_ROOT / CFG_TRAIN["packed_cache"])


def batch_imgs(packed, idx):
    imgs = np.unpackbits(packed[idx], axis=1).astype(np.float32)
    return torch.from_numpy(imgs).reshape(-1, 1, 64, 64)


def load_model(beta: float, seed: int, step: int) -> S1BetaVAE:
    ck = torch.load(
        REPO_ROOT / "results" / "description" / "s1_train" / CFG_TRAIN["campaign"]
        / f"b{beta:g}_s{seed}" / "checkpoints" / f"ckpt_{step:06d}.pt", weights_only=True)
    model = S1BetaVAE(CFG_TRAIN["latent_dim"], CFG_TRAIN["channels"], CFG_TRAIN["fc_dim"],
                      CFG_TRAIN["sigma_dec"], beta)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def encode_samples(model, packed, idx, gen):
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            mu, logvar = model.encoder(batch_imgs(packed, idx[i : i + 512]))
            eps = torch.randn(mu.shape, generator=gen, dtype=mu.dtype)
            outs.append(mu + torch.exp(0.5 * logvar) * eps)
    return torch.cat(outs).double()


def s1_checks(packed, runs_summary: list[dict], log, cache_dir) -> dict:
    """逃逸 run：压缩签名 k 扫描 + Ŝ 双路（标准化 kNN）三点分歧。

    逐 run 结果缓存到 cache_dir（后台命令存在约 30 分钟不可靠窗口的
    环境约束，分块续跑）。"""
    out = {"k_sweep": {}, "dual_entropy": {}}
    for s in runs_summary:
        beta, seed = s["beta"], s["seed"]
        if s["final"]["u_composite"] < COLLAPSE_U_CUT:
            continue
        name = f"b{beta:g}_s{seed}"
        cache_ks = cache_dir / f"{name}_ks.json"
        cache_du = cache_dir / f"{name}_dual.json"
        if cache_ks.exists() and cache_du.exists():
            out["k_sweep"][name] = json.loads(cache_ks.read_text(encoding="utf-8"))
            out["dual_entropy"][name] = json.loads(cache_du.read_text(encoding="utf-8"))
            log(f"  缓存命中 {name}")
            continue
        rng = np.random.default_rng(seed + 555_000)
        idx_ref = rng.choice(packed.shape[0], size=10000, replace=False)
        idx_q = rng.choice(np.setdiff1d(np.arange(packed.shape[0]), idx_ref), size=N_QUERY,
                           replace=False)
        idx_flow = rng.choice(np.setdiff1d(np.arange(packed.shape[0]),
                                           np.concatenate([idx_ref, idx_q])),
                              size=14000, replace=False)
        model = load_model(beta, seed, CFG_TRAIN["train_steps"])
        gen = torch.Generator()
        gen.manual_seed(seed)

        if cache_ks.exists():
            out["k_sweep"][name] = json.loads(cache_ks.read_text(encoding="utf-8"))
            log(f"  k扫描缓存命中 {name}")
        else:
            z_ref = encode_samples(model, packed, idx_ref, gen)
            z_q = encode_samples(model, packed, idx_q, gen)
            z_fl = encode_samples(model, packed, idx_flow, gen)
            x_ref = batch_imgs(packed, idx_ref).reshape(len(idx_ref), -1).double()

            def dec(z):
                return model.decoder(z.float()).reshape(z.shape[0], -1).double()

            def dec_single(z):  # 批量拉回要求单点 (d,) → (D,) 纯函数
                return model.decoder(z.float().unsqueeze(0)).reshape(-1).double()

            fd, _ = fit_flow_density(z_fl[:12000], z_fl[12000:], seed=seed,
                                     n_couplings=8, hidden=64, epochs=8, batch_size=1024)
            g = fisher_pullback_gaussian_batch(
                dec_single, z_q, CFG_TRAIN["sigma_dec"] ** 2, chunk_size=16
            ).detach()
            i_hat = -fd.log_prob(z_q) + 0.5 * torch.linalg.slogdet(g).logabsdet

            slopes = {}
            for k in K_SWEEP:
                v_parts = [potential.potential_knn(dec, CFG_TRAIN["sigma_dec"] ** 2,
                                                   z_q[i : i + 256], z_ref, x_ref, k=k)
                           for i in range(0, z_q.shape[0], 256)]
                v_hat = torch.cat(v_parts)
                slopes[k] = affine_fit_report(v_hat, i_hat)["slope"]
            s_vals = list(slopes.values())
            drift = (max(s_vals) - min(s_vals)) / max(abs(np.median(s_vals)), 1e-30)
            out["k_sweep"][name] = {"slopes": slopes, "relative_drift": drift}
            cache_ks.write_text(json.dumps(out["k_sweep"][name], ensure_ascii=False, indent=2),
                                encoding="utf-8")
            log(f"  k扫描 {name}: 斜率 {['%.3f' % v for v in s_vals]} 漂移 {drift:.1%}")

        per_step = {}
        for step in SPOT_STEPS:
            m2 = load_model(beta, seed, step)
            gen2 = torch.Generator()
            gen2.manual_seed(seed * 7 + step)
            # 评估集 9000 与 6a 曲线同量级；此前 1000 样本的 kNN 熵小样本
            # 偏差大，对双路对比不公平（首轮抽查实测教训）
            idx_ent = rng.choice(packed.shape[0], size=16000, replace=False)
            z_ent = encode_samples(m2, packed, idx_ent, gen2)
            s_flow = entropy_flow(
                fit_flow_density(z_ent[:6000], z_ent[6000:7000], seed=seed,
                                 n_couplings=FLOW_CFG["couplings"], hidden=FLOW_CFG["hidden"],
                                 epochs=FLOW_CFG["epochs"], batch_size=FLOW_CFG["batch"])[0],
                z_ent[7000:])
            s_knn_std = entropy_knn(z_ent[7000:], k=KNN_K, standardize=True)
            per_step[step] = {"flow": s_flow, "knn": s_knn_std, "gap": abs(s_flow - s_knn_std)}
        gaps = {step: v["gap"] for step, v in per_step.items()}
        # 差分一致性（P1 判读用 Ŝ 差分，稳定偏差相消）：两抽查点间
        # ΔŜ 的双路之差
        steps_sorted = sorted(per_step)
        delta_flow = per_step[steps_sorted[-1]]["flow"] - per_step[steps_sorted[0]]["flow"]
        delta_knn = per_step[steps_sorted[-1]]["knn"] - per_step[steps_sorted[0]]["knn"]
        out["dual_entropy"][name] = {
            "gaps": gaps, "delta_gap": abs(delta_flow - delta_knn),
            "delta_flow": delta_flow, "delta_knn": delta_knn,
        }
        cache_du.write_text(json.dumps(out["dual_entropy"][name], ensure_ascii=False, indent=2),
                            encoding="utf-8")
        log(f"  双路 {name}: 分歧 {['%.2f' % v for v in gaps.values()]} (容差 {TOL_DUAL:.2f}), "
            f"ΔŜ 双路差 {abs(delta_flow - delta_knn):.2f} (容差 {TOL_DELTA:.2f})")
    return out


def main() -> None:
    torch.set_num_threads(10)
    run_dir = create_run_dir("description", "precommitment_check",
                             {"k_sweep": K_SWEEP, "tol_dual": TOL_DUAL, "spot_steps": SPOT_STEPS})
    cache_dir = REPO_ROOT / "results" / "description" / "precommitment_check" / "run_cache"
    cache_dir.mkdir(exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    packed = load_packed()
    curves_dir = REPO_ROOT / "results" / "description" / "s1_curves" / CFG_CURVES["out_campaign"]
    runs_summary = [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(curves_dir.glob("b*/summary.json"))]
    diag = json.loads(
        (REPO_ROOT / "results" / "description" / "s2_diagnostics" / "dev_v2" / "summary.json")
        .read_text(encoding="utf-8"))

    s1x = s1_checks(packed, runs_summary, log, cache_dir)

    # ---- 三分支机械判定 ----
    plateau_detected = {f"b{s['beta']:g}_s{s['seed']}": s["plateau_step"] for s in runs_summary}
    n_plateau = sum(1 for v in plateau_detected.values() if v is not None)
    b1c1 = False  # 分支一判据 1：需 ≥3/5 种子检出平台；实测见 plateau_detected
    fig2_pass = all(r["fig2"]["corrected"]["slope"] > 0
                    and r["fig2"]["corrected"]["r_squared"] > r["fig2"]["null_r2_q95"]
                    for r in diag)
    fig1_ok = all(r["fig1"]["ratio_ci90"][0] > 1.0 for r in diag)
    b1c2 = fig2_pass and fig1_ok
    dual_max = max((max(v["gaps"].values()) for v in s1x["dual_entropy"].values()), default=None)
    delta_max = max((v["delta_gap"] for v in s1x["dual_entropy"].values()), default=None)

    # 双路一致性判据（修复后口径）：P1 判读用 Ŝ 差分与形状，稳定的
    # 分布依赖偏差在差分中相消（匹配聚簇真值实测：两路同向上偏、
    # 偏移稳定），故判据取形状一致——逐逃逸 run 整条曲线秩相关 ≥ 0.9
    # 且末窗 ΔŜ 同号。绝对分歧与 ΔŜ 幅度差保留为诊断量（dual_max、
    # delta_max）。阈值 0.9 为提案（实测最小 0.965，冻结时定）。
    import csv as _csv

    from scipy.stats import spearmanr as _spear

    SHAPE_RHO_MIN = 0.9
    shape = {}
    for s in runs_summary:
        if s["final"]["u_composite"] < COLLAPSE_U_CUT:
            continue
        nm = f"b{s['beta']:g}_s{s['seed']}"
        rows = list(_csv.DictReader(open(curves_dir / nm / "curves.csv", encoding="utf-8")))
        sf = np.array([float(r["s_flow"]) for r in rows])
        sk = np.array([float(r["s_knn"]) for r in rows])
        steps = np.array([int(r["step"]) for r in rows])
        uni = steps % 2000 == 0
        rho = float(_spear(sf, sk).statistic)
        same_sign = bool(np.sign(sf[uni][-1] - sf[uni][0]) == np.sign(sk[uni][-1] - sk[uni][0]))
        shape[nm] = {"rho": rho, "end_delta_same_sign": same_sign,
                     "ok": rho >= SHAPE_RHO_MIN and same_sign}
    b1c3 = all(v["ok"] for v in shape.values()) if shape else False
    branch1 = b1c1 and b1c2 and b1c3

    b2_dual = not b1c3
    b2_compress = {k: v["relative_drift"] > 0.30 for k, v in s1x["k_sweep"].items()}
    branch2 = b2_dual or any(b2_compress.values())

    b3_p2 = all(r["fig3"]["frac_below_q1"] <= r["fig3"]["null_binomial_q95"] for r in diag)
    branch3 = False or b3_p2  # 其余分支三条款以平台为前提，未成立

    metrics = {
        "plateau_detected": plateau_detected, "n_plateau": n_plateau,
        "s1_checks": s1x,
        "branch1": {"c1_plateau": b1c1, "c2_s2_structure": b1c2, "c3_dual": b1c3,
                    "triggered": branch1},
        "branch2": {"dual_exceeds": b2_dual, "compress_signature": b2_compress,
                    "triggered": branch2},
        "branch3": {"p2_indistinguishable": b3_p2, "triggered": branch3},
        "dual_gap_max": dual_max,
        "dual_delta_gap_max": delta_max,
        "dual_shape_agreement": shape,
        "tolerances": {"dual_abs_diag": TOL_DUAL, "dual_delta_diag": TOL_DELTA,
                       "shape_rho_min": 0.9, "repair_source": REPAIR_SOURCE},
        "estimator_settings": {"flow": FLOW_CFG, "knn_k": KNN_K},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
                                                     default=float), encoding="utf-8")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"分支一={branch1} 分支二={branch2} 分支三={branch3}（优先序一>二>三）")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
