"""漂移分解与支撑度的自检（docs/plan_v2.md 第 4 节）。

容差依据：

- 漂移恢复：单条转移增量的噪声方差为 2T/Δt（Euler–Maruyama 构造），
  k 近邻平均后每维 SE = sqrt(2T/(Δt·k))；平滑偏差按中值定理
  ‖∇b‖₂·r = max(w)·r（OU 漂移 b = −w⊙u，实测半径 r）。合计
  6·SE + 偏差界为逐点逐维容差。
- 纯梯度流占比：真 ψ 为二次型，tanh-MLP 在有界区域内可良好逼近，
  占比 > 0.95 为回归定值界（固定种子确定性）。
- 混合场占比：b = −(diag(w) − c·S)·z（S 反对称）、采样 z ~ N(0, I) 时
  任意梯度场与 S·z 在 L²(N(0,I)) 下正交（分部积分 + tr S = 0 +
  zᵀSz = 0），最优 ψ 为二次型，闭式占比

      frac = ‖diag(w)‖_F² / (‖diag(w)‖_F² + c²·‖S‖_F²)

  （Σ = I 时 Lyapunov 方程解 Q* = sym(A) = diag(w)，残差恰为 c·S·z）。
  MLP 结果与闭式预期之差 < 0.05（拟合与优化余量，回归界）。

种子纪律：guard.family_seeds 取校准族。
"""
import math

import torch

from slep import guard
from slep.estimators.drift import estimate_drift_knn, gradient_fraction, support_radius


def _gen(idx: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-drift")[idx]
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_drift_recovery_on_ou_transitions():
    gen = _gen(0)
    d, temperature, dt = 4, 0.5, 0.05
    well_w = torch.tensor([0.5, 0.8, 1.2, 2.0], dtype=torch.float64)
    n = 60_000
    std_stat = torch.sqrt(temperature / well_w)
    z_from = std_stat * torch.randn((n, d), generator=gen, dtype=torch.float64)
    noise = torch.randn((n, d), generator=gen, dtype=torch.float64)
    z_next = z_from - well_w * z_from * dt + math.sqrt(2 * temperature * dt) * noise

    z_eval = 0.5 * std_stat * torch.randn((64, d), generator=gen, dtype=torch.float64)
    k = 256
    drift_hat, info = estimate_drift_knn(z_from, z_next, dt, z_eval, k=k, return_info=True)
    drift_true = -well_w * z_eval

    se = math.sqrt(2 * temperature / (dt * k))
    # 标准化坐标半径换算回原坐标的上界：乘以最大逐维标准差
    r_orig = info["radius"].unsqueeze(-1) * z_from.std(dim=0).max()
    tol = 6.0 * se + float(well_w.max()) * r_orig
    assert torch.all((drift_hat - drift_true).abs() <= tol)


def test_gradient_fraction_pure_gradient_flow():
    seeds = guard.family_seeds("calibration", purpose="test-drift")
    gen = _gen(0)
    d = 4
    well_w = torch.tensor([0.5, 0.8, 1.2, 2.0], dtype=torch.float64)
    z = torch.randn((2000, d), generator=gen, dtype=torch.float64)
    drift = -well_w * z  # −∇(½Σw z²)，平坦度量下即负自然梯度

    def metric(zz):
        return torch.eye(d, dtype=torch.float64).expand(zz.shape[0], d, d)

    out = gradient_fraction(z, drift, metric, seed=seeds[0])
    assert out["fraction"] > 0.95, out


def test_gradient_fraction_mixed_field_matches_closed_form():
    seeds = guard.family_seeds("calibration", purpose="test-drift")
    gen = _gen(1)
    d, c = 4, 0.8
    well_w = torch.tensor([0.5, 0.8, 1.2, 2.0], dtype=torch.float64)
    s_raw = torch.randn((d, d), generator=gen, dtype=torch.float64)
    s_antisym = 0.5 * (s_raw - s_raw.T)
    a_mat = torch.diag(well_w) - c * s_antisym

    z = torch.randn((8000, d), generator=gen, dtype=torch.float64)
    drift = -(z @ a_mat.T)

    def metric(zz):
        return torch.eye(d, dtype=torch.float64).expand(zz.shape[0], d, d)

    frac_expected = float(
        (well_w**2).sum() / ((well_w**2).sum() + c**2 * (s_antisym**2).sum())
    )
    out = gradient_fraction(z, drift, metric, seed=seeds[1], hidden=48)
    assert abs(out["fraction"] - frac_expected) < 0.05, (out, frac_expected)


def test_support_radius_monotone_and_far_point():
    gen = _gen(0)
    z_ref = torch.randn((5000, 3), generator=gen, dtype=torch.float64)
    z_query = torch.tensor([[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]], dtype=torch.float64)
    r8 = support_radius(z_query, z_ref, k=8)
    r32 = support_radius(z_query, z_ref, k=32)
    assert torch.all(r32 >= r8)
    assert r8[1] > r8[0]  # 远离支撑的点半径更大
