"""d=16 主臂隔离诊断：核实 flow 口径的成功不是两侧偏差抵消。"""
import json

import torch

from slep import guard
from slep.estimators import potential
from slep.estimators.flow import fit_flow_density
from slep.estimators.metric import fisher_pullback_gaussian_batch
from slep.protocols.affine import affine_fit_report
from slep.systems.cal_langevin import system_from_params
from slep.utils.runs import REPO_ROOT

seeds = guard.family_seeds("calibration", purpose="cal-p4-diagnose-d16")
params = json.loads(
    (REPO_ROOT / "results/calibration/geometry_match/20260824-075955/matched_s2.json").read_text(
        encoding="utf-8"
    )
)
system = system_from_params(params)

g2 = torch.Generator()
g2.manual_seed(seeds[0])
z = system.simulate_chains(8, 50000, 0.05, 5000, 25, g2).reshape(-1, system.latent_dim)
x = system.sample_observations(z, g2)
perm = torch.randperm(z.shape[0], generator=g2)
qi, ri = perm[:2000], perm[2000:]
zq, zr, xr = z[qi], z[ri], x[ri]

v_hat = torch.cat(
    [
        potential.potential_knn(system.decoder_mean, 0.04, zq[i : i + 256], zr, xr, k=32)
        for i in range(0, 2000, 256)
    ]
)
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
r = affine_fit_report(v_true, v_hat)
print(
    f"V̂~V_true: slope={r['slope']:.3f} R²={r['r_squared']:.3f} "
    f"偏差均值={float((v_hat - v_true).mean()):.3f}"
)
