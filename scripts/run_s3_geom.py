"""S3 几何试点（计划 11.3）：残差流隐态上的 ĝ 谱与训练轨迹。

口径与 run_geom_decomposition 一致（同一观测流逐检查点编码、特征值
1e-30 下限、查询点中位数），模型换 S3TransformerWM。产物
results/description/geom_decomp/<campaign>/summary.json。

用法：.venv/Scripts/python.exe scripts/run_s3_geom.py [--config configs/s3_train_v2.yaml]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.systems import s2_gridworld as gw
from slep.systems.s3_transformer import S3TransformerWM
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CKPT_STEPS = [500, 2000, 6000, 12000, 20000]
N_EPISODES, N_QUERY, BURN_IN = 200, 512, 8


def main() -> None:
    cfg_file = "configs/s3_train.yaml"
    if "--config" in sys.argv:
        cfg_file = sys.argv[sys.argv.index("--config") + 1]
    cfg = yaml.safe_load((REPO_ROOT / cfg_file).read_text(encoding="utf-8"))
    torch.set_num_threads(10)
    out_dir = create_campaign_dir(
        "description", "geom_decomp", cfg["campaign"],
        {"ckpt_steps": CKPT_STEPS, "n_episodes": N_EPISODES, "n_query": N_QUERY,
         "burn_in": BURN_IN, "system": "S3"})
    results = {}
    for seed in guard.family_seeds(cfg["seed_family"], purpose="s3-geom"):
        t0 = time.time()
        rng = np.random.default_rng(seed + 980_000)
        obs_np, act_np = gw.collect_rollouts(N_EPISODES, cfg["episode_len"],
                                             cfg["maze_cells"], cfg["view"], rng)
        obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
        per_ckpt = {}
        for step in CKPT_STEPS:
            ck = torch.load(REPO_ROOT / "results" / cfg.get("stage", "description")
                            / "s3_train" / cfg["campaign"] / f"s{seed}" / "checkpoints"
                            / f"ckpt_{step:06d}.pt", weights_only=True)
            model = S3TransformerWM(cfg["obs_dim"], cfg["action_dim"], cfg["d_model"],
                                    cfg["n_layers"], cfg["n_heads"], cfg["ff_dim"],
                                    max_len=cfg["episode_len"] + 4,
                                    sigma_dec=cfg["sigma_dec"],
                                    goal_sigma_dec=cfg.get("goal_sigma_dec"),
                                    multi_step_k=cfg.get("multi_step_k", 1))
            model.load_state_dict(ck["model"])
            model.eval()
            with torch.no_grad():
                hs = model.hidden_trajectory(obs, act)
            h_pool = hs[:, BURN_IN:].reshape(-1, cfg["d_model"]).double()
            idx = torch.from_numpy(rng.permutation(h_pool.shape[0])[:N_QUERY])
            h_q = h_pool[idx]
            ext = cfg.get("multi_step_k", 1) > 1

            def dec(z, m=model, e=ext):
                return (m.decoder_mean_ext if e else m.decoder_mean)(z.float()).double()

            var = (model.obs_var_ext if ext else model.obs_var).double()
            eig_parts = []
            for i in range(0, h_q.shape[0], 128):
                g = fisher_pullback_gaussian_batch(dec, h_q[i: i + 128], var).detach()
                eig_parts.append(torch.linalg.eigvalsh(g).clamp_min(1e-30))
            eig = torch.cat(eig_parts)
            per_ckpt[step] = {
                "log10_cond_median": float(torch.log10(eig[:, -1] / eig[:, 0]).median()),
                "participation_median": float(
                    (eig.sum(-1) ** 2 / (eig ** 2).sum(-1)).median()),
            }
        results[f"s{seed}"] = per_ckpt
        last = per_ckpt[CKPT_STEPS[-1]]
        print(f"s{seed}（{time.time() - t0:.0f}s）末检查点：cond {last['log10_cond_median']:.2f} "
              f"参与维 {last['participation_median']:.2f}", flush=True)
    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物已写入 {out_dir}")


if __name__ == "__main__":
    main()
