"""ĝ 双路估计的自检：线性高斯解码器闭式对拍（docs/plan_v2.md 第 4 节）。

对拍系统：解码器 μ(z) = W z + b。W 为 D×d 常数矩阵（本文件 D=5, d=3），
b 为 D 维常数偏置，观测协方差 Σ = diag(σ_1², …, σ_D²)。解析 Fisher 矩阵

    g = W^T Σ^{-1} W

与 z 无关，是对拍的真值。

容差依据（CLAUDE.md 硬规则 2 要求测试容差注明来源；以下均为推导值）：

- 闭式路径：autograd 对线性映射求 Jacobian 精确，float64 下仅剩矩阵乘的
  浮点舍入，量级 ~1e-16 × 条目规模（本例条目 O(10)）。取绝对容差 1e-10，
  相对机器精度留约 5 个数量级余量。
- MC 路径：得分 s = W^T Σ^{-1} ε（ε ~ N(0, Σ) 为观测噪声），故
  Cov(s) = g，单样本外积条目 (a,b) 的方差 = g_aa·g_bb + g_ab²（高斯四阶
  矩恒等式）。n 样本平均的标准误 SE_ab = sqrt((g_aa·g_bb + g_ab²)/n)。
  逐条目取 6·SE_ab 为容差：高斯尾界下单条目误检概率约 2e-9，d²=9 个条目
  联合误检概率 < 2e-8，可忽略。
- 非线性一致性检验：解析 g 不可得，改用估计器自报的经验标准误
  SE_emp（见 fisher_score_covariance_mc 的 return_se），容差 6·SE_emp 加
  闭式路径的舍入余量 1e-10，依据同上。

种子纪律：随机数只用校准族种子 {99, 100}（configs/seeds.yaml），使用前经
guard.assert_seed_allowed 校验（CLAUDE.md 硬规则 1）。
"""
import torch

from slep import guard
from slep.estimators import metric

torch.manual_seed(0)  # 兜底；实际随机数全部走显式 generator


def _calibration_generator(seed: int) -> torch.Generator:
    family = guard.assert_seed_allowed(seed, purpose="test-metric")
    assert family == "calibration"
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _linear_system(gen: torch.Generator):
    obs_dim, latent_dim = 5, 3
    weight = torch.randn((obs_dim, latent_dim), generator=gen, dtype=torch.float64)
    bias = torch.randn((obs_dim,), generator=gen, dtype=torch.float64)
    z = torch.randn((latent_dim,), generator=gen, dtype=torch.float64)

    def decoder_mean(z_in: torch.Tensor) -> torch.Tensor:
        return z_in @ weight.T + bias

    return weight, decoder_mean, z


def _analytic_fisher(weight: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return weight.T @ (weight / var.unsqueeze(-1))


def test_pullback_matches_analytic_isotropic():
    gen = _calibration_generator(99)
    weight, decoder_mean, z = _linear_system(gen)
    sigma_sq = 0.3**2
    g_analytic = _analytic_fisher(weight, torch.full((5,), sigma_sq, dtype=torch.float64))
    g_hat = metric.fisher_pullback_gaussian(decoder_mean, z, sigma_sq)
    assert torch.allclose(g_hat, g_analytic, atol=1e-10, rtol=0.0)


def test_pullback_matches_analytic_diagonal():
    gen = _calibration_generator(99)
    weight, decoder_mean, z = _linear_system(gen)
    var = torch.tensor([0.04, 0.09, 0.25, 0.09, 0.01], dtype=torch.float64)
    g_analytic = _analytic_fisher(weight, var)
    g_hat = metric.fisher_pullback_gaussian(decoder_mean, z, var)
    assert torch.allclose(g_hat, g_analytic, atol=1e-10, rtol=0.0)


def test_mc_matches_analytic_within_6se():
    gen = _calibration_generator(99)
    weight, decoder_mean, z = _linear_system(gen)
    var = torch.tensor([0.04, 0.09, 0.25, 0.09, 0.01], dtype=torch.float64)
    g_analytic = _analytic_fisher(weight, var)

    n = 200_000
    g_mc = metric.fisher_score_covariance_mc(decoder_mean, z, var, n_samples=n, generator=gen)

    diag = torch.diag(g_analytic)
    se = ((diag.unsqueeze(0) * diag.unsqueeze(1) + g_analytic**2) / n).sqrt()
    exceed = (g_mc - g_analytic).abs() - 6.0 * se
    assert torch.all(exceed <= 0), f"逐条目超出 6·SE 的量:\n{exceed}"


def test_two_paths_agree_on_nonlinear_decoder():
    gen = _calibration_generator(100)
    latent_dim, hidden_dim, obs_dim = 3, 16, 5
    w1 = torch.randn((hidden_dim, latent_dim), generator=gen, dtype=torch.float64)
    w2 = torch.randn((obs_dim, hidden_dim), generator=gen, dtype=torch.float64) / hidden_dim**0.5
    z = torch.randn((latent_dim,), generator=gen, dtype=torch.float64)
    var = torch.tensor([0.09, 0.04, 0.09, 0.16, 0.09], dtype=torch.float64)

    def decoder_mean(z_in: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z_in @ w1.T) @ w2.T

    g_pull = metric.fisher_pullback_gaussian(decoder_mean, z, var)
    g_mc, se_emp = metric.fisher_score_covariance_mc(
        decoder_mean, z, var, n_samples=400_000, generator=gen, return_se=True
    )
    exceed = (g_mc - g_pull).abs() - (6.0 * se_emp + 1e-10)
    assert torch.all(exceed <= 0), f"逐条目超出 6·SE_emp 的量:\n{exceed}"


def test_pullback_symmetric_psd():
    gen = _calibration_generator(100)
    weight, decoder_mean, z = _linear_system(gen)
    g_hat = metric.fisher_pullback_gaussian(decoder_mean, z, 0.25)
    assert torch.allclose(g_hat, g_hat.T, atol=1e-12, rtol=0.0)
    eigvals = torch.linalg.eigvalsh(g_hat)
    # 半正定：最小特征值允许 −1e-12 量级浮点负偏差（float64 舍入）。
    assert torch.all(eigvals > -1e-12)
