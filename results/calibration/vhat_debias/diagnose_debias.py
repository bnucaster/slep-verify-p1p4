"""V̂ 去偏最终候选：度量球邻域 + 精确解码几何项扣除 + 局部线性截距。

目标恒等式：‖x_i − μ(z_q)‖² = ‖x_i − μ(z_i)‖² + 2⟨x_i − μ(z_i),
μ(z_i) − μ(z_q)⟩ + ‖μ(z_i) − μ(z_q)‖²。扣掉末项（精确可算）后，
目标在 δ=0 处期望恰为 V(z_q)；剩余交叉项一阶于 δ，由局部线性吸收。
"""
import math

import numpy as np
import torch

from slep import guard
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.affine import affine_fit_report
from slep.systems.cal_langevin import build_matched_system, system_from_params
from slep.utils.runs import REPO_ROOT
import json

seeds = guard.family_seeds("calibration", purpose="cal-p4-debias")


def potential_debiased(decoder_mean, metric_fn, obs_var, z_query, z_ref, x_ref, k, local_linear=True):
    g_q = metric_fn(z_query)
    chol = torch.linalg.cholesky(g_q)
    const = 0.5 * x_ref.shape[1] * math.log(2 * math.pi * obs_var)
    out = []
    for i in range(0, z_query.shape[0], 64):
        zq = z_query[i : i + 64]
        diff = z_ref.unsqueeze(0) - zq.unsqueeze(1)
        w_all = torch.einsum("cnd,cde->cne", diff, chol[i : i + 64])
        dist = (w_all**2).sum(-1)
        idx = dist.topk(k, dim=-1, largest=False).indices
        mu_q = decoder_mean(zq)  # (c, D)
        mu_nb = decoder_mean(z_ref[idx].reshape(-1, zq.shape[1])).reshape(
            idx.shape[0], idx.shape[1], -1
        )  # (c, k, D)
        x_nb = x_ref[idx]
        nll = ((x_nb - mu_q.unsqueeze(1)) ** 2).sum(-1) / (2 * obs_var) + const
        geom = ((mu_nb - mu_q.unsqueeze(1)) ** 2).sum(-1) / (2 * obs_var)
        target = nll - geom
        if not local_linear:
            out.append(target.mean(dim=-1))
        else:
            w_nb = torch.gather(w_all, 1, idx.unsqueeze(-1).expand(-1, -1, w_all.shape[-1]))
            ones = torch.ones(w_nb.shape[:2] + (1,), dtype=w_nb.dtype)
            design = torch.cat([ones, w_nb], dim=-1)
            beta = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution
            out.append(beta[:, 0, 0])
    return torch.cat(out)


def evaluate(tag, system, z, x, n_q=2000):
    gen = torch.Generator()
    gen.manual_seed(seeds[0])
    perm = torch.randperm(z.shape[0], generator=gen)
    qi, ri = perm[:n_q], perm[n_q:]
    zq, zr, xr = z[qi], z[ri], x[ri]
    v_true = system.potential_estimand(zq)
    i_true = system.self_information_true(zq)

    for label, kk, ll in [("debias_lin_k32", 32, True), ("debias_lin_k64", 64, True), ("debias_mean_k32", 32, False)]:
        vv = potential_debiased(
            system.decoder_mean, system.metric_true, system.sigma_dec**2, zq, zr, xr, kk, ll
        )
        r = affine_fit_report(v_true, vv)
        r2 = affine_fit_report(vv, i_true)
        print(
            f"[{tag}] V̂[{label}]~V_true: slope={r['slope']:.3f} R²={r['r_squared']:.3f} "
            f"偏差均值={float((vv - v_true).mean()):.3f} | 隔离V̂ T̂={r2['temperature_hat']:.4f}"
        )


# 低维臂系统（任务三诊断同款）
gen = torch.Generator()
gen.manual_seed(seeds[0])
well_w = np.exp(np.linspace(math.log(0.5), math.log(2.0), 6)).tolist()
low_sys, _ = build_matched_system(
    [0.05, 0.4, 3.0, 20.0, 120.0, 600.0], 2.0, 16, 0.2, 0.05, well_w, 0.5, gen
)
g2 = torch.Generator()
g2.manual_seed(seeds[0])
z6 = low_sys.simulate_chains(8, 50000, 0.05, 5000, 25, g2).reshape(-1, 6)
x6 = low_sys.sample_observations(z6, g2)
evaluate("d=6", low_sys, z6, x6)

# d=16 主配置核对（去偏口径不应破坏已达标的情形）
params = json.loads(
    (REPO_ROOT / "results/calibration/geometry_match/20260824-075955/matched_s2.json").read_text(
        encoding="utf-8"
    )
)
s16 = system_from_params(params)
g3 = torch.Generator()
g3.manual_seed(seeds[0])
z16 = s16.simulate_chains(8, 50000, 0.05, 5000, 25, g3).reshape(-1, 16)
x16 = s16.sample_observations(z16, g3)
evaluate("d=16", s16, z16, x16)
