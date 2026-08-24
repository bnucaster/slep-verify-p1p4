"""图四补丁：以零初始化头的梯度占比估计器重算 S2 诊断的图四（任务六 6b）。

背景：主运行期间发现 ψ 网络头非零初始化在真实 S2 几何（近奇异度量
方向）上使 g⁻¹∇ψ 初始爆炸，占比失去 ψ=0 基线下界（实测 −60）。
estimators/drift.py 已修（零初始化 + 基线先入榜）；本脚本重放各种子的
状态流（同 rng 流 → 同状态池），重算图四，写入 s<种子>/fig4_v2.json；
campaign summary.json 中 fig4 替换为 v2 并注明；diagnostics.png 由
run_s2_diagnostics.plot_campaign 重生成。

用法：.venv/Scripts/python.exe scripts/patch_s2_fig4.py
"""
from __future__ import annotations

import importlib.util
import json
import time

import torch
import yaml

from slep import guard
from slep.estimators.drift import estimate_drift_knn, gradient_fraction
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.systems import s2_gridworld as gw
from slep.utils.runs import REPO_ROOT

CONFIG_FILE = REPO_ROOT / "configs" / "s2_diagnostics.yaml"

spec = importlib.util.spec_from_file_location(
    "run_s2_diagnostics", REPO_ROOT / "scripts" / "run_s2_diagnostics.py"
)
rsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rsd)


def main() -> None:
    import numpy as np

    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / "configs" / "s2_train.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    out_dir = REPO_ROOT / "results" / "description" / "s2_diagnostics" / cfg["out_campaign"]
    seeds = guard.family_seeds("development", purpose="s2-fig4-patch")

    results = []
    for seed in seeds:
        seed_dir = out_dir / f"s{seed}"
        base = json.loads((seed_dir / "diagnostics.json").read_text(encoding="utf-8"))
        t0 = time.time()
        model = rsd.load_model(train_cfg, seed)
        obs_var = model.obs_var.double()
        rng = np.random.default_rng(seed + 888_000)
        obs_np, act_np = gw.collect_rollouts(
            cfg["n_episodes"], cfg["episode_len"], train_cfg["maze_cells"], train_cfg["view"], rng
        )
        obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)
        with torch.no_grad():
            hs = model.hidden_trajectory(obs, act)
        bi = cfg["burn_in"]
        h_pool = hs[:, bi:-1].reshape(-1, model.hidden_dim).double()
        h_next_pool = hs[:, bi + 1 :].reshape(-1, model.hidden_dim).double()
        perm = torch.from_numpy(rng.permutation(h_pool.shape[0]))
        idx_drift = perm[-cfg["n_drift_eval"]:]

        def dec(z):
            return model.decoder_mean(z.float()).double()

        g_probe = fisher_pullback_gaussian_batch(dec, h_pool[idx_drift][:512], obs_var).detach()
        g_floor = 1e-3 * float(torch.linalg.eigvalsh(g_probe).median())
        eye_h = torch.eye(model.hidden_dim, dtype=torch.float64)

        def g_fn(z):
            return fisher_pullback_gaussian_batch(dec, z, obs_var) + g_floor * eye_h

        drift = estimate_drift_knn(h_pool, h_next_pool, 1.0, h_pool[idx_drift], k=cfg["k_drift"])
        frac = gradient_fraction(h_pool[idx_drift], drift, g_fn, seed=seed)
        fig4 = {
            "gradient_fraction": frac["fraction"],
            "residual_ratio_train": frac["residual_ratio_train"],
            "estimator_note": "v2：零初始化头 + ψ=0 基线（estimators/drift.py 修复后）",
        }
        (seed_dir / "fig4_v2.json").write_text(
            json.dumps(fig4, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        base["fig4"] = fig4
        results.append(base)
        print(f"s{seed}: 占比 v2 = {fig4['gradient_fraction']:.3f}（{time.time() - t0:.0f}s）")

    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rsd.plot_campaign(results, out_dir)
    print("summary.json 与 diagnostics.png 已按 v2 图四重生成")


if __name__ == "__main__":
    main()
