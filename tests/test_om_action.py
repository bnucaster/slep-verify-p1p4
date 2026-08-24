"""Â_OM 的自检：合成 Langevin 恢复（docs/plan_v2.md 第 4 节）。

容差依据：

1. 白化坐标恒等式（左点口径）：Euler–Maruyama 模拟 du = −∇W dt +
   sqrt(2T·dt)·ε 的轨迹上，逐步贡献恰为 ‖ε_t‖²/2（ε_t 为该步标准正态
   增量），故总作用量服从 (1/2)·χ²(n·d)：E = n·d/2，Var = n·d/2。
   多链平均后取 6·sqrt(总方差) 容差，χ² 尾部略重于高斯，6 倍系数留余量。
2. 打乱代理方向性：把增量时间打乱后作用量的期望多出漂移错配项，断言
   均值严格大于真轨迹（固定种子确定性，方向性检验）。
3. 弯曲坐标一致性：同一 u 轨迹映到 z 坐标（z = φ⁻¹(σ_dec·u)）后用
   z 侧势与度量计算，与 u 侧精确值之差来自 φ 的二阶项，随 dt 缩小。
   断言 dt/4 的相对差小于 dt 的相对差（收敛方向），且 dt 档相对差
   < 5%（固定种子回归定值界，非推导容差）。

种子纪律：guard.family_seeds 取校准族。
"""
import math

import torch

from slep import guard
from slep.estimators.om_action import om_action, om_action_batch
from slep.systems.cal_langevin import build_matched_system

D_LOW = 6
TARGET_EIG = [0.05, 0.4, 3.0, 20.0, 120.0, 600.0]


def _gen(idx: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-om")[idx]
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _simulate_ou(well_w, temperature, n_steps, dt, n_chains, gen):
    """直接模拟白化坐标 OU，返回轨迹 (C, n+1, d) 与逐步噪声 (C, n, d)。"""
    d = well_w.shape[0]
    std0 = torch.sqrt(temperature / well_w)
    u = std0 * torch.randn((n_chains, d), generator=gen, dtype=torch.float64)
    trajs, noises = [u.clone()], []
    scale = math.sqrt(2 * temperature * dt)
    for _ in range(n_steps):
        eps = torch.randn((n_chains, d), generator=gen, dtype=torch.float64)
        u = u - well_w * u * dt + scale * eps
        trajs.append(u.clone())
        noises.append(eps)
    return torch.stack(trajs, dim=1), torch.stack(noises, dim=1)


def test_flat_ou_recovers_chi2_statistics():
    gen = _gen(0)
    well_w = torch.tensor([0.5, 0.8, 1.2, 2.0], dtype=torch.float64)
    temperature, dt, n_steps, n_chains = 0.5, 0.05, 1500, 8
    trajs, noises = _simulate_ou(well_w, temperature, n_steps, dt, n_chains, gen)

    def potential(u):
        return 0.5 * (well_w * u**2).sum(dim=-1)

    def metric(u):
        return torch.eye(4, dtype=torch.float64).expand(u.shape[0], 4, 4)

    actions = om_action_batch(trajs, dt, potential, metric, temperature, grad_point="left")
    # 恒等式核对：Â 应逐条等于 Σ‖ε‖²/2（浮点舍入余量 1e-8 相对）
    exact = 0.5 * (noises**2).sum(dim=(1, 2))
    assert torch.allclose(actions, exact, rtol=1e-8), "左点口径未复现白化恒等式"

    d = 4
    mean_expected = n_steps * d / 2
    se_total = math.sqrt(n_steps * d / 2 / n_chains)
    assert abs(float(actions.mean()) - mean_expected) <= 6.0 * se_total


def test_shuffled_surrogate_has_higher_action():
    gen = _gen(0)
    well_w = torch.tensor([0.5, 0.8, 1.2, 2.0], dtype=torch.float64)
    temperature, dt, n_steps = 0.5, 0.05, 1500
    trajs, _ = _simulate_ou(well_w, temperature, n_steps, dt, 8, gen)

    def potential(u):
        return 0.5 * (well_w * u**2).sum(dim=-1)

    def metric(u):
        return torch.eye(4, dtype=torch.float64).expand(u.shape[0], 4, 4)

    true_mean = om_action_batch(trajs, dt, potential, metric, temperature, "left").mean()
    # 增量时间打乱：破坏漂移对齐，作用量期望上升
    increments = trajs[:, 1:] - trajs[:, :-1]
    perm = torch.randperm(n_steps, generator=gen)
    shuffled = torch.cat(
        [trajs[:, :1], trajs[:, :1] + increments[:, perm].cumsum(dim=1)], dim=1
    )
    shuf_mean = om_action_batch(shuffled, dt, potential, metric, temperature, "left").mean()
    assert float(shuf_mean) > float(true_mean)


def test_curved_coordinates_consistency_and_dt_convergence():
    gen = _gen(1)
    import numpy as np

    well_w = np.exp(np.linspace(math.log(0.5), math.log(2.0), D_LOW)).tolist()
    system, _ = build_matched_system(
        TARGET_EIG, 2.0, 16, 0.2, 0.05, well_w, 0.5, gen
    )

    rel_errs = []
    for dt, n_steps in ((0.04, 500), (0.01, 2000)):
        trajs_u, _ = _simulate_ou(system.well_w, system.temperature, n_steps, dt, 4, gen)

        def potential_u(u):
            return system.well(u)

        def metric_u(u):
            return torch.eye(D_LOW, dtype=torch.float64).expand(u.shape[0], D_LOW, D_LOW)

        a_u = om_action_batch(trajs_u, dt, potential_u, metric_u, system.temperature, "left")

        flat = trajs_u.reshape(-1, D_LOW)
        z_flat = system.z_of_u(flat)
        trajs_z = z_flat.reshape(trajs_u.shape)
        a_z = om_action_batch(
            trajs_z, dt, system.potential_dyn, system.metric_true, system.temperature, "left"
        )
        rel_errs.append(float(((a_z - a_u).abs() / a_u).mean()))

    # 收敛方向：更小 dt 的坐标间差异更小；dt=0.04 档 < 5%（回归定值界）
    assert rel_errs[1] < rel_errs[0], rel_errs
    assert rel_errs[0] < 0.05, rel_errs
