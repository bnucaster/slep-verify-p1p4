"""解码器 Jacobian 谱与有效秩（补充实验 A1，回应审稿 critical 3 + 因果性质疑）。

对 S2（基线）、S2P、S3-v1（无多步头）、S3-v2（多步头）在末检查点，逐查询
点算解码器均值 μ(z) 对隐态 z 的 Jacobian J 的奇异值谱，汇总：奇异值中位
谱、有效秩（(Σσ)²/Σσ²）、逐潜维解码敏感度（‖∂μ/∂z_i‖ = 列范数）。检验
"未用记忆维经多步头对观测可见"假说：S2/S3-v1 应有近零奇异值与低有效秩，
S2P/S3-v2 抬升。

口径：Jacobian 取扩展解码口径（多步头用 decoder_mean_ext，与几何门/度量
一致）；奇异值 σ = sqrt(eig(JᵀJ))，与 Fisher 拉回 g = JᵀΣ⁻¹J 同一 J。
数据全部为开发族已训模型（探索性诊断，不进裁决）。

产物 results/description/jac_spectra/summary.json 与 spectra.png。

用法：.venv/Scripts/python.exe scripts/run_jac_spectra.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import yaml

from slep import guard
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel
from slep.systems.s3_transformer import S3TransformerWM
from slep.utils.runs import REPO_ROOT, create_campaign_dir

N_EP, N_QUERY, BURN_IN = 150, 400, 8


def jac_batch(dec, z: torch.Tensor) -> torch.Tensor:
    """逐点 Jacobian (q, D, d)，jacfwd + vmap（与 metric 同机制）。"""
    return torch.func.vmap(torch.func.jacfwd(dec))(z)


def summarize(dec, z: torch.Tensor, ext_dim: int) -> dict:
    sv_all, eff_ranks, col_norm = [], [], None
    for i in range(0, z.shape[0], 64):
        J = jac_batch(dec, z[i: i + 64]).detach()  # (b, D, d)
        sv = torch.linalg.svdvals(J.double())      # (b, d) 降序
        sv_all.append(sv)
        cn = J.double().pow(2).sum(dim=1).sqrt()   # (b, d) 逐潜维列范数
        col_norm = cn if col_norm is None else torch.cat([col_norm, cn])
    sv = torch.cat(sv_all)                          # (q, d)
    eff = (sv.sum(-1) ** 2 / (sv ** 2).sum(-1))     # 有效秩逐点
    return {
        "sv_median_by_rank": torch.quantile(sv, 0.5, dim=0).tolist(),
        "sv_min_median": float(sv[:, -1].median()),
        "sv_max_median": float(sv[:, 0].median()),
        "log10_cond_median": float(torch.log10(sv[:, 0] / sv[:, -1].clamp_min(1e-30)).median()),
        "effective_rank_median": float(eff.median()),
        "latent_sensitivity_median": torch.quantile(col_norm, 0.5, dim=0).tolist(),
        "n_near_zero_sv_median": float((sv < 1e-4 * sv[:, :1]).sum(-1).double().median()),
        "d": int(sv.shape[1]),
    }


def s2_models():
    cfg = yaml.safe_load((REPO_ROOT / "configs/s2_train_long.yaml").read_text(encoding="utf-8"))
    cfgp = yaml.safe_load((REPO_ROOT / "configs/s2p_v1_train.yaml").read_text(encoding="utf-8"))
    for label, c, camp, k in (("S2 (base)", cfg, "dev_v3", 1),
                              ("S2P (multi-step)", cfgp, "s2p_v1", 4)):
        yield label, c, camp, k


def load_s2(c, camp, seed, k):
    ck = torch.load(REPO_ROOT / "results/description/s2_train" / camp / f"s{seed}"
                    / "checkpoints" / "ckpt_020000.pt", weights_only=True)
    m = S2WorldModel(c["obs_dim"], c["action_dim"], c["embed_dim"], c["hidden_dim"],
                     c["sigma_dec"], c.get("goal_sigma_dec"), multi_step_k=k)
    # strict=False:旧检查点(加 obs_var_ext 缓冲前存)缺该派生键,__init__ 已正确
    # 初始化(k=1 时 obs_var_ext=obs_var),不影响 decoder 权重加载
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m


def load_s3(camp, seed, k):
    c = yaml.safe_load((REPO_ROOT / ("configs/s3_train.yaml" if k == 1
                        else "configs/s3_train_v2.yaml")).read_text(encoding="utf-8"))
    ck = torch.load(REPO_ROOT / "results/description/s3_train" / camp / f"s{seed}"
                    / "checkpoints" / "ckpt_020000.pt", weights_only=True)
    m = S3TransformerWM(c["obs_dim"], c["action_dim"], c["d_model"], c["n_layers"],
                        c["n_heads"], c["ff_dim"], max_len=c["episode_len"] + 4,
                        sigma_dec=c["sigma_dec"], goal_sigma_dec=c.get("goal_sigma_dec"),
                        multi_step_k=k)
    m.load_state_dict(ck["model"], strict=False)  # 同 load_s2:旧检查点缺派生缓冲
    m.eval()
    return m, c


def hidden_pool(model, c, seed, is_s3):
    rng = np.random.default_rng(seed + 970_000)
    obs_np, act_np = gw.collect_rollouts(N_EP, c["episode_len"], c["maze_cells"],
                                         c["view"], rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
    with torch.no_grad():
        hs = model.hidden_trajectory(obs, act)
    hd = c["d_model"] if is_s3 else c["hidden_dim"]
    pool = hs[:, BURN_IN:].reshape(-1, hd).double()
    idx = torch.from_numpy(rng.permutation(pool.shape[0])[:N_QUERY])
    return pool[idx]


def main() -> None:
    torch.set_num_threads(6)
    out = create_campaign_dir("description", "jac_spectra", "v1", {"n_query": N_QUERY})
    seeds = guard.family_seeds("development", purpose="jac-spectra")
    results = {}
    # S2 / S2P
    for label, c, camp, k in s2_models():
        per = []
        for seed in seeds:
            m = load_s2(c, camp, seed, k)
            z = hidden_pool(m, c, seed, is_s3=False)
            dec = (lambda zz, mm=m, kk=k: (mm.decoder_mean_ext if kk > 1
                   else mm.decoder_mean)(zz.float()).double())
            per.append(summarize(dec, z, k))
        results[label] = agg(per)
        print(f"{label}: 有效秩中位 {results[label]['effective_rank_median']:.2f}，"
              f"近零奇异值数中位 {results[label]['n_near_zero_sv_median']:.1f}，"
              f"log10 cond {results[label]['log10_cond_median']:.2f}", flush=True)
    # S3 v1 / v2
    for label, camp, k in (("S3-v1 (no head)", "s3_dev_v1", 1),
                           ("S3-v2 (multi-step)", "s3_dev_v2", 4)):
        per = []
        for seed in seeds:
            m, c = load_s3(camp, seed, k)
            z = hidden_pool(m, c, seed, is_s3=True)
            dec = (lambda zz, mm=m, kk=k: (mm.decoder_mean_ext if kk > 1
                   else mm.decoder_mean)(zz.float()).double())
            per.append(summarize(dec, z, k))
        results[label] = agg(per)
        print(f"{label}: 有效秩中位 {results[label]['effective_rank_median']:.2f}，"
              f"近零奇异值数中位 {results[label]['n_near_zero_sv_median']:.1f}，"
              f"log10 cond {results[label]['log10_cond_median']:.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    plot(results, out)
    print(f"产物写入 {out}")


def agg(per: list) -> dict:
    keys_scalar = ("sv_min_median", "sv_max_median", "log10_cond_median",
                   "effective_rank_median", "n_near_zero_sv_median")
    o = {k: float(np.median([p[k] for p in per])) for k in keys_scalar}
    o["d"] = per[0]["d"]
    o["sv_median_by_rank"] = np.median([p["sv_median_by_rank"] for p in per], axis=0).tolist()
    o["latent_sensitivity_sorted"] = sorted(
        np.median([p["latent_sensitivity_median"] for p in per], axis=0).tolist(), reverse=True)
    return o


def plot(results, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.unicode_minus": False})
    ink, blue, orange = "#000", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=300)
    fig.patch.set_facecolor("#fff")
    styles = {"S2 (base)": (blue, "-", "o"), "S2P (multi-step)": (orange, "-", "s"),
              "S3-v1 (no head)": (blue, "--", "^"), "S3-v2 (multi-step)": (orange, "--", "D")}
    ax = axes[0]
    for label, r in results.items():
        col, ls, mk = styles[label]
        sv = np.array(r["sv_median_by_rank"])
        ax.plot(np.arange(1, len(sv) + 1), sv[::-1], color=col, linestyle=ls, marker=mk,
                markersize=4, linewidth=1.3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value rank (large $\\to$ small)", color=ink, fontsize=9)
    ax.set_ylabel("Decoder-Jacobian singular value (median)", color=ink, fontsize=9)
    ax.set_title("(a) Jacobian singular-value spectra", color=ink, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax = axes[1]
    labels = list(results.keys())
    eff = [results[k]["effective_rank_median"] for k in labels]
    x = np.arange(len(labels))
    cols = [styles[k][0] for k in labels]
    ax.bar(x, eff, color=cols, alpha=0.8, width=0.6)
    for xi, e in zip(x, eff):
        ax.text(xi, e + 0.05, f"{e:.2f}", ha="center", color=ink, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["S2", "S2P", "S3-v1", "S3-v2"], fontsize=8)
    ax.set_ylabel("Effective rank (median)", color=ink, fontsize=9)
    ax.set_title("(b) Effective rank of decoder Jacobian", color=ink, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "jac_spectra.pdf", facecolor="#fff", bbox_inches="tight")
    fig.savefig(out / "jac_spectra.png", facecolor="#fff", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
