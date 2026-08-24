"""熵双路修复选型（任务六点五 7a）：聚簇合成真值上的误差测量。

流程：
1. 对模板 run 的末检查点潜态（后验采样）拟合 GMM（BIC 选阶）——密度
   已知的聚簇真值；真值熵 H* 取高精度 MC（−E log p，SE 随报告）。
2. 从 GMM 精确采样，按 6a/6d 同口径评估 flow 三档配置与 kNN(标准化)
   多 k 的熵误差，校准种子 {99,100} 重复。
3. 选型：逐路取匹配真值上最大绝对误差最小的配置；新容差 =
   tolerance_safety × 该配置的最大绝对误差（逐路，加和为双路分歧容差），
   全部溯源本产物。

细粒度缓存（环境后台命令 15–20 分钟终止窗口）：逐（模板 × 配置 × 种子）
缓存到 cache/。产物 results/description/entropy_repair/<run_id>/。

用法：.venv/Scripts/python.exe scripts/run_entropy_repair.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators.entropy import entropy_flow, entropy_knn
from slep.estimators.flow import fit_flow_density
from slep.systems.s1_beta_vae import S1BetaVAE
from slep.utils.runs import REPO_ROOT, create_run_dir

CONFIG_FILE = REPO_ROOT / "configs" / "entropy_repair.yaml"
CFG_TRAIN = yaml.safe_load((REPO_ROOT / "configs" / "s1_train.yaml").read_text(encoding="utf-8"))
CACHE = REPO_ROOT / "results" / "description" / "entropy_repair" / "cache"


def batch_imgs(packed, idx):
    imgs = np.unpackbits(packed[idx], axis=1).astype(np.float32)
    return torch.from_numpy(imgs).reshape(-1, 1, 64, 64)


def fit_truth(cfg: dict, packed, beta: float, seed: int, log) -> dict:
    """GMM 真值：拟合、BIC 选阶、MC 熵。缓存于 cache/truth_*.json（参数）。"""
    from sklearn.mixture import GaussianMixture

    cache = CACHE / f"truth_b{beta:g}_s{seed}.json"
    if cache.exists():
        log(f"  真值缓存命中 b{beta:g}_s{seed}")
        return json.loads(cache.read_text(encoding="utf-8"))

    ck = torch.load(
        REPO_ROOT / "results" / "description" / "s1_train" / CFG_TRAIN["campaign"]
        / f"b{beta:g}_s{seed}" / "checkpoints" / f"ckpt_{CFG_TRAIN['train_steps']:06d}.pt",
        weights_only=True)
    model = S1BetaVAE(CFG_TRAIN["latent_dim"], CFG_TRAIN["channels"], CFG_TRAIN["fc_dim"],
                      CFG_TRAIN["sigma_dec"], beta)
    model.load_state_dict(ck["model"])
    model.eval()
    rng = np.random.default_rng(seed + 999_000)
    idx = rng.choice(packed.shape[0], size=cfg["n_fit"], replace=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    zs = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            mu, logvar = model.encoder(batch_imgs(packed, idx[i : i + 512]))
            eps = torch.randn(mu.shape, generator=gen, dtype=mu.dtype)
            zs.append(mu + torch.exp(0.5 * logvar) * eps)
    z = torch.cat(zs).double().numpy()

    best = None
    for k in cfg["gmm_components"]:
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=99,
                             max_iter=300).fit(z)
        bic = gm.bic(z)
        log(f"  GMM b{beta:g}_s{seed} K={k}: BIC={bic:.0f}")
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    _, k_best, gm = best

    samp, _ = gm.sample(cfg["truth_mc_n"])
    logp = gm.score_samples(samp)
    h_star = float(-logp.mean())
    h_se = float(logp.std() / np.sqrt(len(logp)))
    out = {
        "beta": beta, "seed": seed, "k_best": k_best, "h_star": h_star, "h_se": h_se,
        "weights": gm.weights_.tolist(), "means": gm.means_.tolist(),
        "covariances": gm.covariances_.tolist(),
    }
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"  真值 b{beta:g}_s{seed}: K={k_best}, H*={h_star:.4f}±{h_se:.4f}")
    return out


def gmm_sampler(truth: dict, rng: np.random.Generator):
    weights = np.array(truth["weights"])
    means = np.array(truth["means"])
    covs = np.array(truth["covariances"])
    chols = np.linalg.cholesky(covs)

    def sample(n: int) -> torch.Tensor:
        comps = rng.choice(len(weights), size=n, p=weights)
        eps = rng.standard_normal((n, means.shape[1]))
        out = means[comps] + np.einsum("nij,nj->ni", chols[comps], eps)
        return torch.from_numpy(out)

    return sample


def measure(cfg: dict, truth: dict, rep_seed: int, log) -> dict:
    """一个（模板 × 种子）的全配置误差测量。缓存逐条。"""
    name = f"b{truth['beta']:g}_s{truth['seed']}_r{rep_seed}"
    cache = CACHE / f"measure_{name}.json"
    if cache.exists():
        log(f"  测量缓存命中 {name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    rng = np.random.default_rng(rep_seed)
    sample = gmm_sampler(truth, rng)
    z_train = sample(cfg["n_eval_flow_train"])
    z_val = sample(cfg["n_eval_flow_val"])
    z_eval = sample(cfg["n_eval"])
    h_star = truth["h_star"]

    out = {"flow": {}, "knn": {}}
    for fname, fc in cfg["flow_configs"].items():
        fd, rep = fit_flow_density(z_train, z_val, seed=rep_seed,
                                   n_couplings=fc["couplings"], hidden=fc["hidden"],
                                   epochs=fc["epochs"], batch_size=fc["batch"])
        err = entropy_flow(fd, z_eval) - h_star
        out["flow"][fname] = {"error": err, "val_nll": rep["final_val_nll"]}
        log(f"    {name} flow[{fname}]: 误差 {err:+.3f}")
    for k in cfg["knn_ks"]:
        err = entropy_knn(z_eval, k=k, standardize=True) - h_star
        out["knn"][str(k)] = {"error": err}
        log(f"    {name} knn[k={k}]: 误差 {err:+.3f}")
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir("description", "entropy_repair", cfg)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    packed = np.load(REPO_ROOT / CFG_TRAIN["packed_cache"])
    rep_seeds = guard.family_seeds("calibration", purpose="entropy-repair")

    truths, measures = [], {}
    for src in cfg["truth_sources"]:
        truth = fit_truth(cfg, packed, float(src["beta"]), src["seed"], log)
        truths.append(truth)
        for rs in rep_seeds[: cfg["n_repeats"]]:
            measures[f"b{truth['beta']:g}_s{truth['seed']}_r{rs}"] = measure(cfg, truth, rs, log)

    # 选型：逐路最大绝对误差最小者
    flow_worst = {f: max(abs(m["flow"][f]["error"]) for m in measures.values())
                  for f in cfg["flow_configs"]}
    knn_worst = {k: max(abs(m["knn"][str(k)]["error"]) for m in measures.values())
                 for k in cfg["knn_ks"]}
    flow_pick = min(flow_worst, key=flow_worst.get)
    knn_pick = min(knn_worst, key=knn_worst.get)
    tol_flow = cfg["tolerance_safety"] * flow_worst[flow_pick]
    tol_knn = cfg["tolerance_safety"] * knn_worst[knn_pick]

    summary = {
        "truths": [{k: t[k] for k in ("beta", "seed", "k_best", "h_star", "h_se")}
                   for t in truths],
        "flow_worst_abs_error": flow_worst,
        "knn_worst_abs_error": {str(k): v for k, v in knn_worst.items()},
        "selected": {"flow": flow_pick, "knn_k": knn_pick},
        "tolerances": {"flow": tol_flow, "knn": tol_knn, "dual_sum": tol_flow + tol_knn,
                       "derivation": f"{cfg['tolerance_safety']} × 匹配聚簇真值上最大绝对误差"},
        "measures": measures,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"选型: flow={flow_pick}（最坏误差 {flow_worst[flow_pick]:.3f}）, "
        f"knn k={knn_pick}（{knn_worst[knn_pick]:.3f}）, 新双路容差 {tol_flow + tol_knn:.3f}")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
