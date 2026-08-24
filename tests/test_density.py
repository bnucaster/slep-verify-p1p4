"""kNN 密度与体积校正自信息 Î 的自检（docs/plan_v2.md 第 4 节）。

容差依据（全部为推导值，无手拍数）：

- 波动项：Kozachenko–Leonenko 型估计的单点波动方差趋于 trigamma ψ₁(k)
  （k 近邻距离对数的渐近方差），取 6·sqrt(ψ₁(k))，高斯尾界下单点误检
  概率 ~2e-9。Q 个查询点求均值时，网格间距远大于 kNN 半径，各点近邻集
  互不重叠，近似独立，均值波动取 6·sqrt(ψ₁(k)/Q)（近似依据注明于此，
  额外以 6 倍系数留余量）。
- 偏差项：ψ 校正后剩余偏差来自密度在 kNN 球内的变化，
  |E log p̂ − log p(z)| ≤ r·sup_ball‖∇log p‖（中值定理）。高斯 N(0, C)
  下 ∇log p = −C^{-1}z，sup_ball‖∇log p‖ ≤ ‖C^{-1}‖₂·(‖z‖ + r)，r 取
  实测第 k 近邻距离。
- Î 的度量项：线性解码器下 ĝ 为常数闭式，float64 舍入 ~1e-10，并入
  容差的静态余量。

坐标不变性测试用纯扩张对角变换 A = diag(4, 3, 2)（log|det A| = log 24 ≈
3.18）：校正后 Î 的均值差须落在推导容差内，未校正 −log p̂ 的均值差须
等于 log|det A|（同容差），且容差 < log|det A|，两个断言合起来才证明
体积校正在起作用。选纯扩张的原因：压缩轴会放大变换系的密度梯度，
使偏差容差超过 log|det A|，测试失去分辨力；扩张使变换系梯度变小，
偏差界受控。梯度界取逐点值加算子范数乘半径：
sup_ball‖∇log p‖ ≤ ‖C⁻¹z‖ + ‖C⁻¹‖₂·r（三角不等式）。

种子纪律：数据种子取 guard.family_seeds("calibration")（configs/seeds.yaml），
查询点为确定性网格。
"""
import math

import torch

from slep import guard
from slep.estimators import density, metric

K_2D = 32
K_INV = 64


def _gen(seed: int) -> torch.Generator:
    guard.assert_seed_allowed(seed, purpose="test-density")
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _trigamma(k: int) -> float:
    return float(torch.polygamma(1, torch.tensor(float(k))))


def _grid(halfwidth: float, side: int, d: int) -> torch.Tensor:
    axis = torch.linspace(-halfwidth, halfwidth, side, dtype=torch.float64)
    grids = torch.meshgrid(*([axis] * d), indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def test_gaussian_log_density_recovery_2d():
    seed = guard.family_seeds("calibration", purpose="test-density")[0]
    gen = _gen(seed)
    z_ref = torch.randn((40_000, 2), generator=gen, dtype=torch.float64)
    z_query = _grid(1.4, 5, 2)  # 间距 0.7，远大于实测 kNN 半径（~0.06）
    log_p_hat, info = density.log_density_knn(z_query, z_ref, k=K_2D, return_info=True)
    log_p_true = -0.5 * (z_query**2).sum(-1) - math.log(2 * math.pi)

    r = info["radius"]
    bias = r * (z_query.norm(dim=-1) + r)  # C = I，‖C⁻¹‖₂ = 1
    tol = 6.0 * _trigamma(K_2D) ** 0.5 + bias
    err = log_p_hat - log_p_true
    assert torch.all(err.abs() <= tol), f"逐点超差:\n{err.abs() - tol}"

    q = z_query.shape[0]
    mean_tol = 6.0 * (_trigamma(K_2D) / q) ** 0.5 + bias.mean()
    assert err.mean().abs() <= mean_tol


def test_gaussian_log_density_recovery_4d():
    seed = guard.family_seeds("calibration", purpose="test-density")[0]
    gen = _gen(seed)
    z_ref = torch.randn((60_000, 4), generator=gen, dtype=torch.float64)
    z_query = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0], [0.7, -0.7, 0.7, -0.7]],
        dtype=torch.float64,
    )
    log_p_hat, info = density.log_density_knn(z_query, z_ref, k=K_2D, return_info=True)
    log_p_true = -0.5 * (z_query**2).sum(-1) - 2 * math.log(2 * math.pi)
    r = info["radius"]
    tol = 6.0 * _trigamma(K_2D) ** 0.5 + r * (z_query.norm(dim=-1) + r)
    assert torch.all((log_p_hat - log_p_true).abs() <= tol)


def _linear_decoder(gen: torch.Generator, latent_dim: int = 3, obs_dim: int = 6):
    weight = torch.randn((obs_dim, latent_dim), generator=gen, dtype=torch.float64)

    def decoder_mean(z: torch.Tensor) -> torch.Tensor:
        return z @ weight.T

    return weight, decoder_mean


def test_self_information_recovery_linear_decoder():
    seeds = guard.family_seeds("calibration", purpose="test-density")
    gen = _gen(seeds[0])
    weight, decoder_mean = _linear_decoder(gen)
    sigma_sq = 0.09
    z_ref = torch.randn((40_000, 3), generator=gen, dtype=torch.float64)
    z_query = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, -0.8, 0.8], [0.6, 0.6, -0.6]],
        dtype=torch.float64,
    )
    i_hat, info = density.self_information_knn(
        decoder_mean, sigma_sq, z_query, z_ref, k=K_2D, return_info=True
    )
    g_true = weight.T @ weight / sigma_sq
    log_p_true = -0.5 * (z_query**2).sum(-1) - 1.5 * math.log(2 * math.pi)
    i_true = -log_p_true + 0.5 * torch.linalg.slogdet(g_true).logabsdet
    r = info["radius"]
    tol = 6.0 * _trigamma(K_2D) ** 0.5 + r * (z_query.norm(dim=-1) + r) + 1e-8
    assert torch.all((i_hat - i_true).abs() <= tol)


def test_self_information_coordinate_invariance():
    seeds = guard.family_seeds("calibration", purpose="test-density")
    gen = _gen(seeds[1])
    weight, decoder_mean = _linear_decoder(gen)
    sigma_sq = 0.09
    n_ref = 40_000
    z_ref = torch.randn((n_ref, 3), generator=gen, dtype=torch.float64)
    z_query = _grid(0.6, 3, 3)  # 27 点，间距 0.6，近原点使梯度界小

    a_diag = torch.tensor([4.0, 3.0, 2.0], dtype=torch.float64)
    log_det_a = float(torch.log(a_diag).sum())  # = log 24 ≈ 3.178
    a_inv = torch.diag(1.0 / a_diag)

    def decoder_mean_t(z_t: torch.Tensor) -> torch.Tensor:
        return decoder_mean(z_t @ a_inv.T)

    zt_ref = z_ref * a_diag
    zt_query = z_query * a_diag

    i_orig, info_o = density.self_information_knn(
        decoder_mean, sigma_sq, z_query, z_ref, k=K_INV, return_info=True
    )
    i_trans, info_t = density.self_information_knn(
        decoder_mean_t, sigma_sq, zt_query, zt_ref, k=K_INV, return_info=True
    )

    # 两坐标系各自的偏差界；变换系密度 N(0, diag(a²))，∇log p' = −diag(a⁻²)z'，
    # sup_ball‖∇log p'‖ ≤ ‖diag(a⁻²)z'‖ + ‖diag(a⁻²)‖₂·r（三角不等式）。
    r_o, r_t = info_o["radius"], info_t["radius"]
    bias_o = r_o * (z_query.norm(dim=-1) + r_o)
    c_inv_norm = float((1.0 / a_diag**2).max())
    grad_t = (zt_query / a_diag**2).norm(dim=-1)
    bias_t = r_t * (grad_t + c_inv_norm * r_t)
    q = z_query.shape[0]
    mean_tol = 6.0 * (2 * _trigamma(K_INV) / q) ** 0.5 + bias_o.mean() + bias_t.mean() + 1e-8

    # 断言 1：校正后 Î 坐标不变（均值差在推导容差内）。
    corrected_gap = (i_trans - i_orig).mean().abs()
    assert corrected_gap <= mean_tol, f"{corrected_gap} > {mean_tol}"

    # 断言 2：未校正 −log p̂ 的均值差等于 log|det A|，且容差 < log|det A|，
    # 证明不变性由体积校正带来。
    assert mean_tol < log_det_a, "容差大于 log|det A|，本测试无分辨力"
    uncorrected_gap = (-info_t["log_density"] + info_o["log_density"]).mean()
    assert (uncorrected_gap - log_det_a).abs() <= mean_tol
