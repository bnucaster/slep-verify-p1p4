"""低维臂偏差定位：用真值隔离 V̂ 侧与 Î 侧贡献（诊断用，不落 results）。"""
import math

import numpy as np
import torch

from slep import guard
from slep.estimators import potential
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.affine import affine_fit_report
from slep.systems.cal_langevin import build_matched_system

seeds = guard.family_seeds("calibration", purpose="cal-p4-diagnose")
gen = torch.Generator()
gen.manual_seed(seeds[0])
well_w = np.exp(np.linspace(math.log(0.5), math.log(2.0), 6)).tolist()
system, _ = build_matched_system(
    [0.05, 0.4, 3.0, 20.0, 120.0, 600.0], 2.0, 16, 0.2, 0.05, well_w, 0.5, gen
)

g2 = torch.Generator()
g2.manual_seed(seeds[0])
z = system.simulate_chains(8, 50000, 0.05, 5000, 25, g2).reshape(-1, 6)
x = system.sample_observations(z, g2)
perm = torch.randperm(z.shape[0], generator=g2)
qi, ri = perm[:2000], perm[2000:]
zq, zr, xr = z[qi], z[ri], x[ri]

def potential_local_linear(decoder_mean, obs_var, z_query, z_ref, x_ref, k):
    """局部线性回归口径：NLL_i 对位移 δ_i 拟线性，取截距。"""
    import math as _math

    std = z_ref.std(dim=0).clamp_min(1e-12)
    dists = torch.cdist(z_query / std, z_ref / std)
    idx = dists.topk(k, dim=-1, largest=False).indices  # (q, k)
    mu_q = decoder_mean(z_query)
    nll = ((x_ref[idx] - mu_q.unsqueeze(1)) ** 2).sum(-1) / (2 * obs_var) + 0.5 * x_ref.shape[
        1
    ] * _math.log(2 * _math.pi * obs_var)
    delta = (z_ref[idx] - z_query.unsqueeze(1)) / std  # (q, k, d) 条件数友好
    ones = torch.ones(delta.shape[:2] + (1,), dtype=delta.dtype)
    design = torch.cat([ones, delta], dim=-1)  # (q, k, d+1)
    beta = torch.linalg.lstsq(design, nll.unsqueeze(-1)).solution  # (q, d+1, 1)
    return beta[:, 0, 0]


def potential_pointwise(decoder_mean, obs_var, z_query, z_ref, x_ref, k, local_linear):
    """逐点重构 NLL 的邻域均值 / 局部线性截距。

    v_i = ‖x_i − μ(z_i)‖²/(2σ²) + 配分常数，在查询点邻域上取均值
    （local_linear=False）或对位移拟线性取截距（True，去一阶下坡偏斜）。
    """
    import math as _math

    std = z_ref.std(dim=0).clamp_min(1e-12)
    v_point = ((x_ref - decoder_mean(z_ref)) ** 2).sum(-1) / (2 * obs_var) + 0.5 * x_ref.shape[
        1
    ] * _math.log(2 * _math.pi * obs_var)  # (N,)
    dists = torch.cdist(z_query / std, z_ref / std)
    idx = dists.topk(k, dim=-1, largest=False).indices  # (q, k)
    v_nb = v_point[idx]  # (q, k)
    if not local_linear:
        return v_nb.mean(dim=-1)
    delta = (z_ref[idx] - z_query.unsqueeze(1)) / std
    ones = torch.ones(delta.shape[:2] + (1,), dtype=delta.dtype)
    design = torch.cat([ones, delta], dim=-1)
    beta = torch.linalg.lstsq(design, v_nb.unsqueeze(-1)).solution
    return beta[:, 0, 0]


def potential_metric_ball(decoder_mean, metric_fn, obs_var, z_query, z_ref, x_ref, k, pointwise):
    """局部度量球邻域：dist²(z_i, z_q) = δᵀ g(z_q) δ；NLL 取查询点解码
    （pointwise=False）或逐点重构（True）。"""
    import math as _math

    g_q = metric_fn(z_query)  # (q, d, d)
    chol = torch.linalg.cholesky(g_q)  # (q, d, d)
    out = []
    if pointwise:
        v_point = ((x_ref - decoder_mean(z_ref)) ** 2).sum(-1) / (2 * obs_var) + 0.5 * x_ref.shape[
            1
        ] * _math.log(2 * _math.pi * obs_var)
    for i in range(0, z_query.shape[0], 64):
        zq = z_query[i : i + 64]
        diff = z_ref.unsqueeze(0) - zq.unsqueeze(1)  # (c, N, d)
        w = torch.einsum("cnd,cde->cne", diff, chol[i : i + 64])  # 白化位移
        dist = (w**2).sum(-1)  # (c, N)
        idx = dist.topk(k, dim=-1, largest=False).indices
        if pointwise:
            out.append(v_point[idx].mean(dim=-1))
        else:
            mu_q = decoder_mean(zq)
            nll = ((x_ref[idx] - mu_q.unsqueeze(1)) ** 2).sum(-1) / (2 * obs_var) + 0.5 * x_ref.shape[
                1
            ] * _math.log(2 * _math.pi * obs_var)
            out.append(nll.mean(dim=-1))
    return torch.cat(out)


def potential_metric_ball_corrected(
    decoder_mean, metric_fn, obs_var, z_query, z_ref, x_ref, k, local_linear
):
    """度量球邻域 + 解码几何项解析扣除（NLL_i − ‖w_i‖²/2）+ 可选局部线性。"""
    import math as _math

    g_q = metric_fn(z_query)
    chol = torch.linalg.cholesky(g_q)
    out = []
    for i in range(0, z_query.shape[0], 64):
        zq = z_query[i : i + 64]
        diff = z_ref.unsqueeze(0) - zq.unsqueeze(1)
        w_all = torch.einsum("cnd,cde->cne", diff, chol[i : i + 64])
        dist = (w_all**2).sum(-1)
        knn = dist.topk(k, dim=-1, largest=False)
        idx = knn.indices
        w_sq = knn.values  # (c, k) = ‖w_i‖²
        mu_q = decoder_mean(zq)
        nll = ((x_ref[idx] - mu_q.unsqueeze(1)) ** 2).sum(-1) / (2 * obs_var) + 0.5 * x_ref.shape[
            1
        ] * _math.log(2 * _math.pi * obs_var)
        target = nll - 0.5 * w_sq
        if not local_linear:
            out.append(target.mean(dim=-1))
        else:
            w_nb = torch.gather(
                w_all, 1, idx.unsqueeze(-1).expand(-1, -1, w_all.shape[-1])
            )  # (c, k, d)
            ones = torch.ones(w_nb.shape[:2] + (1,), dtype=w_nb.dtype)
            design = torch.cat([ones, w_nb], dim=-1)
            beta = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution
            out.append(beta[:, 0, 0])
    return torch.cat(out)


variants = {}
for label, kk, ll in [("gcorr_k32", 32, False), ("gcorrlin_k32", 32, True), ("gcorrlin_k64", 64, True)]:
    variants[label] = torch.cat(
        [
            potential_metric_ball_corrected(
                system.decoder_mean, system.metric_true, 0.04, zq[i : i + 256], zr, xr, kk, ll
            )
            for i in range(0, 2000, 256)
        ]
    )
for label, kk, pw in [("gball_k32", 32, False)]:
    variants[label] = torch.cat(
        [
            potential_metric_ball(
                system.decoder_mean, system.metric_true, 0.04, zq[i : i + 256], zr, xr, kk, pw
            )
            for i in range(0, 2000, 256)
        ]
    )
for label, kk, ll in [("pw_k32", 32, False), ("pwlin_k32", 32, True), ("pwlin_k64", 64, True)]:
    variants[label] = torch.cat(
        [
            potential_pointwise(system.decoder_mean, 0.04, zq[i : i + 256], zr, xr, kk, ll)
            for i in range(0, 2000, 256)
        ]
    )
variants["loclin_k32"] = torch.cat(
    [
        potential_local_linear(system.decoder_mean, 0.04, zq[i : i + 256], zr, xr, k=32)
        for i in range(0, 2000, 256)
    ]
)
variants["loclin_k64"] = torch.cat(
    [
        potential_local_linear(system.decoder_mean, 0.04, zq[i : i + 256], zr, xr, k=64)
        for i in range(0, 2000, 256)
    ]
)
for label, kk, stdz in [("k32", 32, False), ("k8std", 8, True)]:
    variants[label] = torch.cat(
        [
            potential.potential_knn(
                system.decoder_mean, 0.04, zq[i : i + 256], zr, xr, k=kk, standardize=stdz
            )
            for i in range(0, 2000, 256)
        ]
    )
v_hat = variants["k32"]
n_val = zr.shape[0] // 10
fd, _ = fit_flow_density(
    zr[n_val:], zr[:n_val], seed=seeds[0], n_couplings=10, hidden=128, epochs=15, batch_size=1024
)
g = fisher_pullback_gaussian_batch(system.decoder_mean, zq, 0.04)
i_flow = -fd.log_prob(zq) + 0.5 * torch.linalg.slogdet(g).logabsdet

v_true = system.potential_estimand(zq)
i_true = system.self_information_true(zq)

for name, xv, yv in [
    ("全估计 Î_flow~V̂     ", v_hat, i_flow),
    ("隔离V̂: Î_true~V̂    ", v_hat, i_true),
    ("隔离Î: Î_flow~V_true", v_true, i_flow),
]:
    r = affine_fit_report(xv, yv)
    print(
        f"{name}: slope={r['slope']:.3f} T̂={r['temperature_hat']:.4f} "
        f"R²={r['r_squared']:.3f} p={r['p_lack_of_fit']:.2g}"
    )
for label, vv in variants.items():
    r = affine_fit_report(v_true, vv)
    r2 = affine_fit_report(vv, i_true)
    print(
        f"V̂[{label}]~V_true: slope={r['slope']:.3f} R²={r['r_squared']:.3f} "
        f"偏差均值={float((vv - v_true).mean()):.3f} | 隔离V̂ T̂={r2['temperature_hat']:.4f}"
    )
