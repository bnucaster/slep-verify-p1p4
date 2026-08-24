"""仿射拟合统计量的自检（protocols/affine.py）。

容差依据：
- 斜率恢复：OLS 斜率的标准误有闭式（affine_fit_report 自报 se_slope），
  取 6·SE。
- lack-of-fit 方向性：纯线性 + 高斯噪声数据（固定校准种子，确定性）
  应不触发拒绝（p > 0.05、ΔBIC < 2）；带明显二次项的数据应强拒绝
  （p < 1e-6、ΔBIC > 10）。二次项幅度选为噪声的 30 倍以上，拒绝断言
  不依赖种子运气。
"""
import torch

from slep import guard
from slep.protocols.affine import affine_fit_report, split_half_temperature


def _gen(idx: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-affine")[idx]
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _linear_data(gen, n=2000, slope=2.0, intercept=1.0, noise=0.3):
    v = 3.0 * torch.rand((n,), generator=gen, dtype=torch.float64)
    y = intercept + slope * v + noise * torch.randn((n,), generator=gen, dtype=torch.float64)
    return v, y


def test_linear_recovery_and_no_false_rejection():
    v, y = _linear_data(_gen(0))
    rep = affine_fit_report(v, y)
    assert abs(rep["slope"] - 2.0) <= 6 * rep["se_slope"]
    assert rep["p_lack_of_fit"] > 0.05
    assert rep["delta_bic_lin_minus_quad"] < 2.0
    assert rep["r_squared"] > 0.9
    assert abs(rep["temperature_hat"] - 0.5) < 0.01  # 1/slope，斜率误差 <0.4% 时成立
    # 纯线性数据的曲率效应量应接近零（二次系数仅噪声，量级 ~SE）
    assert rep["curvature_effect_ratio"] < 0.02


def test_quadratic_rejected():
    gen = _gen(1)
    v = 3.0 * torch.rand((2000,), generator=gen, dtype=torch.float64)
    y = 1.0 + 2.0 * v + 1.5 * (v - 1.5) ** 2 + 0.3 * torch.randn(
        (2000,), generator=gen, dtype=torch.float64
    )
    rep = affine_fit_report(v, y)
    assert rep["p_lack_of_fit"] < 1e-6
    assert rep["delta_bic_lin_minus_quad"] > 10.0


def test_split_half_consistency_on_homogeneous_data():
    v, y = _linear_data(_gen(0), n=4000)
    mask = torch.zeros(4000, dtype=torch.bool)
    mask[:2000] = True
    rep = split_half_temperature(v, y, mask)
    # 同分布两半的 T̂ 相对差应远小于 plan 容差 20%；此处只作方向性
    # 断言（<10%），实际容差由 CAL-P4 校准产出。
    assert rep["relative_gap"] < 0.10
