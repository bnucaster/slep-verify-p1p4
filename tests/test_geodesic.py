"""测地求解与偏差估计的自检（docs/plan_v2.md 第 4 节）。

容差依据：

- 平坦度量：能量泛函为凸二次，LBFGS（强 Wolfe 线搜索）收敛到机器精度
  量级；直线偏差与多起点散布取 1e-6 / 1e-5 回归界（优化器精度，固定
  种子确定性）。
- 合成校准系统闭式对拍：g = ∇φᵀ∇φ/σ² 与平直坐标 y = φ(z) 等距（常数
  共形因子不改测地线），真测地 = φ⁻¹(y 直线)。求解路径与真测地的
  归一化偏差面积取 5e-3 回归界（含离散能量与连续测地的 O(1/m²) 差、
  优化器残差；实测约束回归，非推导容差）。
- 双路几何（中央高斯凸包）：两条对称近最短路径并存，多起点散布应
  显著大于平坦情形（>0.2，机制方向性检验）。
- 偏差面积解析对拍：平坦度量下正弦偏移轨迹的归一化面积 = 2A/(πL)
  （A 为幅度，L 为线长），梯形离散取 5% 相对容差（离散化余量）。

种子纪律：guard.family_seeds 取校准族。
"""
import math

import torch

from slep import guard
from slep.estimators.geodesic import geodesic_deviation, solve_geodesic
from slep.systems.cal_langevin import build_matched_system

TARGET_EIG = [0.05, 0.4, 3.0, 20.0, 120.0, 600.0]


def _gen(idx: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-geodesic")[idx]
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _flat_metric(d: int):
    eye = torch.eye(d, dtype=torch.float64)

    def metric(z: torch.Tensor) -> torch.Tensor:
        return eye.expand(z.shape[0], d, d)

    return metric


def test_flat_metric_recovers_straight_line():
    gen = _gen(0)
    z_a = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    z_b = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    out = solve_geodesic(z_a, z_b, _flat_metric(3), generator=gen)
    tau = torch.linspace(0, 1, out["path"].shape[0], dtype=torch.float64).unsqueeze(-1)
    straight = z_a * (1 - tau) + z_b * tau
    assert float((out["path"] - straight).norm(dim=-1).max()) < 1e-6
    assert out["uniqueness_spread"] < 1e-5
    assert out["residual"] < 1e-8


def test_matched_system_geodesic_closed_form():
    gen = _gen(0)
    import numpy as np

    well_w = np.exp(np.linspace(math.log(0.5), math.log(2.0), 6)).tolist()
    system, _ = build_matched_system(TARGET_EIG, 2.0, 16, 0.2, 0.05, well_w, 0.5, gen)
    z_pts = system.exact_sample(2, gen)
    z_a, z_b = z_pts[0], z_pts[1]

    out = solve_geodesic(z_a, z_b, system.metric_true, n_segments=48, generator=gen)
    # 真测地：y = φ(z) 平直坐标里的直线拉回
    y_a, y_b = system.phi(z_a), system.phi(z_b)
    tau = torch.linspace(0, 1, 49, dtype=torch.float64).unsqueeze(-1)
    geo_true = system.phi_inverse(y_a * (1 - tau) + y_b * tau)

    dev = geodesic_deviation(out["path"], geo_true, system.metric_true)
    assert dev["normalized_area"] < 5e-3, dev
    # 能量不应低于真测地的离散能量（真测地是极小元；容 1e-6 相对余量）
    from slep.estimators.geodesic import _energy

    e_true = float(_energy(geo_true, system.metric_true))
    assert out["energy"] <= e_true * (1 + 1e-6) + 1e-12


def test_two_route_geometry_detected_by_multistart():
    gen = _gen(1)

    def bump_metric(z: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + 25.0 * torch.exp(-(z**2).sum(-1) / 0.3**2)
        eye = torch.eye(2, dtype=torch.float64)
        return factor.unsqueeze(-1).unsqueeze(-1) * eye

    z_a = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    z_b = torch.tensor([1.0, 0.0], dtype=torch.float64)
    out = solve_geodesic(z_a, z_b, bump_metric, n_starts=6, generator=gen)
    assert out["uniqueness_spread"] > 0.2, out["uniqueness_spread"]


def test_deviation_area_matches_analytic_flat():
    z_a = torch.tensor([0.0, 0.0], dtype=torch.float64)
    z_b = torch.tensor([2.0, 0.0], dtype=torch.float64)
    # 测地离散足够密（257 点），避免最近顶点距离高估进入 5% 容差
    tau = torch.linspace(0, 1, 257, dtype=torch.float64).unsqueeze(-1)
    geo = z_a * (1 - tau) + z_b * tau

    n = 200
    t = torch.linspace(0, 1, n, dtype=torch.float64)
    amp = 0.1
    traj = torch.stack([2.0 * t, amp * torch.sin(math.pi * t)], dim=-1)
    dev = geodesic_deviation(traj, geo, _flat_metric(2))
    # 解析值：面积 ∫A·sin(πt)·L dt = 2AL/π，归一化除 L² → 2A/(πL)
    analytic = 2 * amp / (math.pi * 2.0)
    assert abs(dev["normalized_area"] - analytic) / analytic < 0.05
    # 测地对自身偏差为零
    self_dev = geodesic_deviation(geo, geo, _flat_metric(2))
    assert self_dev["normalized_area"] < 1e-12
