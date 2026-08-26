"""几何门诊断（选项 C，探索性，开发族）：条件数的来源分解与训练轨迹。

问题：20k 步 S2 模型 log10 条件数中位 ≈ 9.4，远超几何门 6.0（定标于
6k 步模型谱）。本诊断回答两问：

1. 分解——条件数里多少来自逐通道 σ_dec 权重（墙体 0.2 / 目标 0.04，
   Σ⁻¹ 注入 625 倍通道权重差），多少来自解码器本身（sigmoid 饱和使
   Jacobian 行退化）？对比 cond(JᵀΣ⁻¹J)（判定口径）与 cond(JᵀJ)
   （等方差口径，Σ = I 等价于任意各向同性 σ）。
2. 轨迹——条件数中位沿训练步何时越过 6？

口径：同一观测流（与模型无关）逐检查点编码取隐态，每检查点抽
n_query 点算两种度量的谱。产物 results/description/geom_decomp/
<campaign>/summary.json 与 decomp.png。

用法：.venv/Scripts/python.exe scripts/run_geom_decomposition.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel
from slep.utils.runs import REPO_ROOT, create_campaign_dir

TRAIN_CONFIG = REPO_ROOT / "configs" / "s2_train_long.yaml"  # dev_v3
CAMPAIGN = "dev_v3"
CKPT_STEPS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000]
N_EPISODES, N_QUERY, BURN_IN = 200, 512, 8


def spectra(model: S2WorldModel, h_q: torch.Tensor, obs_var) -> dict:
    def dec(z):
        return model.decoder_mean(z.float()).double()

    g = fisher_pullback_gaussian_batch(dec, h_q, obs_var).detach()
    eig = torch.linalg.eigvalsh(g).clamp_min(1e-30)
    cond = torch.log10(eig[:, -1] / eig[:, 0])
    part = eig.sum(-1) ** 2 / (eig**2).sum(-1)
    return {"log10_cond_median": float(cond.median()),
            "participation_median": float(part.median())}


def main() -> None:
    cfg = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    torch.set_num_threads(10)
    out_dir = create_campaign_dir(
        "description", "geom_decomp", CAMPAIGN,
        {"ckpt_steps": CKPT_STEPS, "n_episodes": N_EPISODES, "n_query": N_QUERY,
         "burn_in": BURN_IN, "train_campaign": CAMPAIGN})
    results = {}
    for seed in guard.family_seeds("development", purpose="geom-decomp"):
        t0 = time.time()
        rng = np.random.default_rng(seed + 960_000)
        obs_np, act_np = gw.collect_rollouts(N_EPISODES, cfg["episode_len"],
                                             cfg["maze_cells"], cfg["view"], rng)
        obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
        per_ckpt = {}
        for step in CKPT_STEPS:
            ck = torch.load(REPO_ROOT / "results/description/s2_train/dev_v3"
                            / f"s{seed}" / "checkpoints" / f"ckpt_{step:06d}.pt",
                            weights_only=True)
            model = S2WorldModel(cfg["obs_dim"], cfg["action_dim"], cfg["embed_dim"],
                                 cfg["hidden_dim"], cfg["sigma_dec"],
                                 cfg.get("goal_sigma_dec"))
            model.load_state_dict(ck["model"])
            model.eval()
            with torch.no_grad():
                hs = model.hidden_trajectory(obs, act)
            h_pool = hs[:, BURN_IN:].reshape(-1, cfg["hidden_dim"]).double()
            idx = torch.from_numpy(rng.permutation(h_pool.shape[0])[:N_QUERY])
            h_q = h_pool[idx]
            per_ckpt[step] = {
                "per_channel": spectra(model, h_q, model.obs_var.double()),
                "isotropic": spectra(model, h_q, 1.0),
            }
        results[f"s{seed}"] = per_ckpt
        last = per_ckpt[CKPT_STEPS[-1]]
        print(f"s{seed}（{time.time() - t0:.0f}s）末检查点：判定口径 "
              f"{last['per_channel']['log10_cond_median']:.2f}，等方差 "
              f"{last['isotropic']['log10_cond_median']:.2f}", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for si, (sname, per_ckpt) in enumerate(results.items()):
        steps = sorted(int(s) for s in per_ckpt)
        pc = [per_ckpt[s]["per_channel"]["log10_cond_median"] for s in steps]
        iso = [per_ckpt[s]["isotropic"]["log10_cond_median"] for s in steps]
        ax.plot(steps, pc, color=blue, alpha=0.5, linewidth=1.4,
                label="判定口径（逐通道 Σ）" if si == 0 else None)
        ax.plot(steps, iso, color=orange, alpha=0.5, linewidth=1.4,
                label="等方差口径（Σ = I）" if si == 0 else None)
    ax.axhline(6.0, color=ink2, linestyle="--", linewidth=1.2)
    ax.text(CKPT_STEPS[0], 6.15, "几何门 6.0", color=ink2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("训练步（对数轴）", color=ink2)
    ax.set_ylabel("log10 条件数中位", color=ink2)
    ax.set_title("S2 条件数分解与训练轨迹（dev_v3，探索性）", color=ink, fontsize=11)
    ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(out_dir / "decomp.png", facecolor=surface)
    print(f"产物已写入 {out_dir}")


if __name__ == "__main__":
    main()
