"""熵双路（kNN / flow）与标准化 kNN 密度的自检（docs/plan_v2.md 第 4 节）。

容差依据，分两层：

- 推导层（硬界）：解析熵 H 的 MC 估计波动 SE₁ = sqrt(Var(log p)/N)
  （高斯下 Var(log p) = d/2）；kNN 半径波动 SE₂ = sqrt(ψ₁(k)/N)
  （逐样本 trigamma 波动，样本间近邻集部分重叠，按独立近似并以 6 倍
  系数容纳）；偏差项用中值定理界 mean_i[r_i·sup_ball‖∇log p‖]，
  高斯下 sup ≤ ‖C⁻¹z‖ + ‖C⁻¹‖₂·r，逐样本用实测半径。合计
  tol = 6·(SE₁+SE₂) + bias_mean。该界对偏差项很保守（一阶界不含
  球对称抵消），是正确性硬界。
- 回归层（定值界）：固定校准种子下全流程确定性，加一条实测水平的
  定值界（如 |Ĥ − H| < 0.1 nats）防住硬界容不住的粗错；此为实现
  回归约束，非推导容差，也不进入任何协议判定。

flow 熵的方向性恒等式：Ĥ_flow = −E_p[log q̂] = H + KL(p‖q̂) ≥ H，
故断言 Ĥ_flow − H ∈ [−6·SE_eval, KL 定值界 + 6·SE_eval]，下界是
恒等式的 MC 松弛，上界的 KL 定值界（0.1 nats）为回归层。

种子纪律：guard.family_seeds 取校准族。
"""
import math

import torch

from slep import guard
from slep.estimators import density
from slep.estimators.entropy import entropy_flow, entropy_knn
from slep.estimators.flow import fit_flow_density

K = 8


def _gen(idx: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-entropy")[idx]
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _trigamma(k: int) -> float:
    return float(torch.polygamma(1, torch.tensor(float(k))))


def _knn_tolerances(z: torch.Tensor, radii: torch.Tensor, c_inv: torch.Tensor, d: int, n: int):
    se = math.sqrt(d / 2 / n) + math.sqrt(_trigamma(K) / n)
    grad_norm = (z @ c_inv).norm(dim=-1)
    c_inv_op = float(torch.linalg.matrix_norm(c_inv, ord=2))
    bias = (radii * (grad_norm + c_inv_op * radii)).mean()
    return 6.0 * se + float(bias)


def test_entropy_knn_standard_gaussian_d4():
    gen = _gen(0)
    d, n = 4, 20_000
    z = torch.randn((n, d), generator=gen, dtype=torch.float64)
    h_true = 0.5 * d * math.log(2 * math.pi * math.e)
    h_hat, info = entropy_knn(z, k=K, return_info=True)
    tol = _knn_tolerances(z, info["radius"], torch.eye(d, dtype=torch.float64), d, n)
    assert abs(h_hat - h_true) <= tol
    # 回归层定值界（固定种子确定性，防粗错；非推导容差）
    assert abs(h_hat - h_true) < 0.1


def test_entropy_knn_correlated_gaussian_d8():
    gen = _gen(0)
    d, n = 8, 40_000
    a = torch.randn((d, d), generator=gen, dtype=torch.float64)
    cov = a @ a.T / d + 0.2 * torch.eye(d, dtype=torch.float64)
    chol = torch.linalg.cholesky(cov)
    z = torch.randn((n, d), generator=gen, dtype=torch.float64) @ chol.T
    h_true = 0.5 * float(torch.linalg.slogdet(2 * math.pi * math.e * cov).logabsdet)
    h_hat, info = entropy_knn(z, k=K, return_info=True)
    tol = _knn_tolerances(z, info["radius"], torch.linalg.inv(cov), d, n)
    assert abs(h_hat - h_true) <= tol
    assert abs(h_hat - h_true) < 0.2  # 回归层定值界，d=8 偏差更大留余量


def test_flow_density_and_entropy_standard_gaussian_d4():
    seeds = guard.family_seeds("calibration", purpose="test-entropy-flow")
    gen = _gen(0)
    d = 4
    z_train = torch.randn((16_000, d), generator=gen, dtype=torch.float64)
    z_val = torch.randn((4_000, d), generator=gen, dtype=torch.float64)
    z_eval = torch.randn((10_000, d), generator=gen, dtype=torch.float64)
    fd, report = fit_flow_density(z_train, z_val, seed=seeds[0])

    log_p_true = -0.5 * (z_eval**2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
    kl_hat = float((log_p_true - fd.log_prob(z_eval)).mean())
    se_eval = math.sqrt(d / 2 / z_eval.shape[0])
    # KL(p‖q̂) ≥ 0：下界为 MC 松弛（恒等式检验）；上界 0.1 nats 为回归层
    # 定值界（训练质量约束，固定种子确定性）。
    assert kl_hat >= -6.0 * se_eval, f"KL 估计为负超 MC 松弛: {kl_hat}"
    assert kl_hat < 0.1, f"流拟合质量退化: KL≈{kl_hat:.4f}, val_nll={report['final_val_nll']:.4f}"

    h_true = 0.5 * d * math.log(2 * math.pi * math.e)
    h_flow = entropy_flow(fd, z_eval)
    assert -6.0 * se_eval <= h_flow - h_true <= 0.1 + 6.0 * se_eval


def test_flow_density_correlated_gaussian_d8_and_dual_agreement():
    seeds = guard.family_seeds("calibration", purpose="test-entropy-flow")
    gen = _gen(1)
    d = 8
    a = torch.randn((d, d), generator=gen, dtype=torch.float64)
    cov = a @ a.T / d + 0.2 * torch.eye(d, dtype=torch.float64)
    chol = torch.linalg.cholesky(cov)

    def sample(n: int) -> torch.Tensor:
        return torch.randn((n, d), generator=gen, dtype=torch.float64) @ chol.T

    z_train, z_val, z_eval = sample(32_000), sample(4_000), sample(10_000)
    fd, report = fit_flow_density(z_train, z_val, seed=seeds[1])

    h_true = 0.5 * float(torch.linalg.slogdet(2 * math.pi * math.e * cov).logabsdet)
    se_eval = math.sqrt(d / 2 / z_eval.shape[0])
    h_flow = entropy_flow(fd, z_eval)
    assert -6.0 * se_eval <= h_flow - h_true <= 0.2 + 6.0 * se_eval, (
        f"h_flow−h_true={h_flow - h_true:.4f}, val_nll={report['final_val_nll']:.4f}"
    )

    h_knn = entropy_knn(z_eval, k=K)
    # 双路互证：各自与真值的定值界之和作互差界（回归层）
    assert abs(h_flow - h_knn) < 0.4


def test_knn_density_standardized_recovers_anisotropic_gaussian():
    gen = _gen(0)
    scales = torch.tensor([100.0, 1.0, 0.01], dtype=torch.float64)  # 跨 4 个量级
    n = 40_000
    z_ref = torch.randn((n, 3), generator=gen, dtype=torch.float64) * scales
    z_query = torch.randn((25, 3), generator=gen, dtype=torch.float64) * scales * 0.8

    log_p_true = (
        -0.5 * ((z_query / scales) ** 2).sum(-1)
        - 1.5 * math.log(2 * math.pi)
        - torch.log(scales).sum()
    )
    log_p_hat, info = density.log_density_knn(
        z_query, z_ref, k=32, return_info=True, standardize=True
    )
    # 标准化坐标下参照分布近似 N(0, I)（σ̂ 由样本估计，偏差并入回归余量），
    # 偏差界与 tests/test_density.py 同构：r·(‖x'‖ + r)，r 为标准化坐标半径。
    x_query = z_query / scales
    r = info["radius"]
    tol = 6.0 * math.sqrt(_trigamma(32)) + r * (x_query.norm(dim=-1) + r) + 0.05
    assert torch.all((log_p_hat - log_p_true).abs() <= tol)
