"""V̂ 双操作化的自检：已知势合成系统上的恢复（docs/plan_v2.md 第 4 节）。

合成系统见 src/slep/systems/selfcheck.py：观测 x = W z + α‖z‖²u + ε，
解码器 μ(z) = W z（方差 σ_dec²），真势 V_true 与 finite-k oracle 参照
V_oracle 均有闭式。W 列正交，‖W δ‖ = ‖δ‖，误差界因此不含 W 的谱。

误差分解与容差依据（CLAUDE.md 硬规则 2 要求容差注明来源；以下均为
推导值，无手拍数）：

1. V̂ − V_oracle：只含各邻居观测噪声 ε_i 的贡献，逐点标准误 SE 有闭式
   （selfcheck.noise_se，推导见其 docstring）。取 6·SE：k 个独立邻居
   平均后按中心极限定理近似高斯，单点误检概率 ~2e-9，25 个查询点联合
   误检概率 < 1e-7，可忽略。ε_i 含卡方分量，尾部略重于高斯，6 倍系数
   相对 CLT 近似留有余量。

2. V_oracle − V_true：确定性离散化偏差，来源是邻居 z_i 偏离查询点 z。
   记 r 为实测第 k 近邻距离，δ_i = z_i − z（‖δ_i‖ ≤ r）。逐邻居
   ‖m_i‖² = ‖δ_i‖² + 2α‖z_i‖²⟨Wδ_i, u⟩ + α²‖z_i‖⁴ 与 α²‖z‖⁴ 之差按
   Cauchy–Schwarz 与三角不等式界住：

       |‖m_i‖² − α²‖z‖⁴| ≤ r² + 2α(‖z‖+r)²·r + α²·Δ4(z, r)
       Δ4(z, r) = max( (‖z‖+r)⁴ − ‖z‖⁴, ‖z‖⁴ − max(‖z‖−r, 0)⁴ )

   除以 2σ_dec² 得偏差界 B(z, r)。该界只用实测半径与已知系统参数。

3. 恢复判据（主张本身）：|V̂ − V_true| ≤ B + 6·SE，由 1、2 三角不等式
   合成。

加权操作化同理：权重 w_i 由 oracle 编码器后验 N(z_i, τ²I) 生成，与 ε
无关；SE_w = sqrt(Σ w_i² Var_i)（selfcheck.noise_se 加权分支），偏差界
B_w = Σ_i w_i·b_i，b_i 为上式逐邻居界把 r 换成实际 ‖δ_i‖、‖z‖+r 换成
实际 ‖z_i‖ 后的量（对全体参照点求和，权重远端衰减）。

种子纪律：数据种子 99（校准族，configs/seeds.yaml），采样入口经
guard.assert_seed_allowed 校验；查询点为确定性网格，不耗种子。
"""
import pytest
import torch

from slep import guard
from slep.estimators import potential
from slep.systems.selfcheck import LinearGaussianSelfCheck

LATENT_DIM, OBS_DIM = 2, 8
ALPHA, SIGMA_X, SIGMA_DEC = 1.0, 0.1, 0.3
N_REF, K = 40_000, 32
TAU = 0.05  # oracle 编码器后验标准差


@pytest.fixture(scope="module")
def system_and_data():
    seed = 99
    guard.assert_seed_allowed(seed, purpose="test-potential-build")
    gen = torch.Generator()
    gen.manual_seed(seed)
    system = LinearGaussianSelfCheck.build(
        LATENT_DIM, OBS_DIM, ALPHA, SIGMA_X, SIGMA_DEC, generator=gen
    )
    z_ref, x_ref = system.sample(N_REF, generator=gen, purpose="test-potential-sample")
    return system, z_ref, x_ref


@pytest.fixture(scope="module")
def z_query():
    axis = torch.linspace(-1.4, 1.4, 5, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (25, 2)


def _bias_bound(z_query: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """偏差界 B(z, r)，推导见文件头第 2 条。"""
    norm = z_query.norm(dim=-1)
    delta4 = torch.maximum(
        (norm + radius) ** 4 - norm**4,
        norm**4 - torch.clamp(norm - radius, min=0.0) ** 4,
    )
    return (radius**2 + 2 * ALPHA * (norm + radius) ** 2 * radius + ALPHA**2 * delta4) / (
        2 * SIGMA_DEC**2
    )


def test_knn_matches_oracle_within_noise(system_and_data, z_query):
    system, z_ref, x_ref = system_and_data
    v_hat, info = potential.potential_knn(
        system.decoder_mean, SIGMA_DEC**2, z_query, z_ref, x_ref, k=K, return_info=True
    )
    z_neighbors = z_ref[info["indices"]]  # (q, k, d)
    v_oracle = system.v_oracle_knn(z_query, z_neighbors)
    se = system.noise_se(z_query, z_neighbors)
    exceed = (v_hat - v_oracle).abs() - 6.0 * se
    assert torch.all(exceed <= 0), f"超出 6·SE 的量:\n{exceed}"


def test_oracle_bias_within_derived_bound(system_and_data, z_query):
    system, z_ref, x_ref = system_and_data
    _, info = potential.potential_knn(
        system.decoder_mean, SIGMA_DEC**2, z_query, z_ref, x_ref, k=K, return_info=True
    )
    z_neighbors = z_ref[info["indices"]]
    gap = (system.v_oracle_knn(z_query, z_neighbors) - system.v_true(z_query)).abs()
    assert torch.all(gap <= _bias_bound(z_query, info["radius"]))


def test_knn_recovers_true_potential(system_and_data, z_query):
    system, z_ref, x_ref = system_and_data
    v_hat, info = potential.potential_knn(
        system.decoder_mean, SIGMA_DEC**2, z_query, z_ref, x_ref, k=K, return_info=True
    )
    z_neighbors = z_ref[info["indices"]]
    tol = _bias_bound(z_query, info["radius"]) + 6.0 * system.noise_se(z_query, z_neighbors)
    exceed = (v_hat - system.v_true(z_query)).abs() - tol
    assert torch.all(exceed <= 0), f"超出恢复容差的量:\n{exceed}"


def test_knn_standardized_neighborhood_equals_manual_whitening(system_and_data, z_query):
    """标准化邻域口径的机械等价性：与手工白化输入后的普通口径同邻居同值。

    （标准化只改邻居选取，NLL 计算不变；其对极端各向异性几何的效果在
    CAL-P4 产物中量化，见 results/calibration/cal_p4/。）
    """
    system, z_ref, x_ref = system_and_data
    scales = torch.tensor([50.0, 1.0], dtype=torch.float64)
    zq_s, zr_s = z_query * scales, z_ref * scales

    def decoder_mean_scaled(z):
        return system.decoder_mean(z / scales)

    v_std, info_std = potential.potential_knn(
        decoder_mean_scaled, SIGMA_DEC**2, zq_s, zr_s, x_ref, k=K,
        return_info=True, standardize=True,
    )
    mean, std = zr_s.mean(dim=0), zr_s.std(dim=0).clamp_min(1e-12)

    def decoder_mean_white(zw):
        return decoder_mean_scaled(zw * std + mean)

    v_man, info_man = potential.potential_knn(
        decoder_mean_white, SIGMA_DEC**2, (zq_s - mean) / std, (zr_s - mean) / std, x_ref,
        k=K, return_info=True,
    )
    assert torch.equal(info_std["indices"], info_man["indices"])
    assert torch.allclose(v_std, v_man, rtol=1e-12)


def test_weighted_matches_oracle_within_noise(system_and_data, z_query):
    system, z_ref, x_ref = system_and_data
    v_hat, info = potential.potential_posterior_weighted(
        system.decoder_mean,
        SIGMA_DEC**2,
        z_query,
        x_ref,
        posterior_mean=z_ref,
        posterior_var=TAU**2,
        return_info=True,
    )
    weights = info["weights"]  # (q, N)，只依赖 z_ref 与查询点，与观测噪声独立
    v_oracle = system.v_oracle_weighted(z_query, z_ref, weights)
    z_ref_exp = z_ref.unsqueeze(0).expand(z_query.shape[0], -1, -1)
    se = system.noise_se(z_query, z_ref_exp, weights=weights)
    exceed = (v_hat - v_oracle).abs() - 6.0 * se
    assert torch.all(exceed <= 0), f"超出 6·SE 的量:\n{exceed}"


def test_weighted_recovers_true_potential(system_and_data, z_query):
    system, z_ref, x_ref = system_and_data
    v_hat, info = potential.potential_posterior_weighted(
        system.decoder_mean,
        SIGMA_DEC**2,
        z_query,
        x_ref,
        posterior_mean=z_ref,
        posterior_var=TAU**2,
        return_info=True,
    )
    weights = info["weights"]
    # 加权偏差界 B_w = Σ_i w_i·b_i，b_i 用逐参照点的实际 ‖δ_i‖ 与 ‖z_i‖。
    delta = z_ref.unsqueeze(0) - z_query.unsqueeze(1)  # (q, N, d)
    dist = delta.norm(dim=-1)
    ref_norm = z_ref.norm(dim=-1).unsqueeze(0)
    query_norm4 = (z_query.norm(dim=-1) ** 4).unsqueeze(-1)
    b_i = (
        dist**2
        + 2 * ALPHA * ref_norm**2 * dist
        + ALPHA**2 * (ref_norm**4 - query_norm4).abs()
    ) / (2 * SIGMA_DEC**2)
    bound = (weights * b_i).sum(dim=-1)
    z_ref_exp = z_ref.unsqueeze(0).expand(z_query.shape[0], -1, -1)
    tol = bound + 6.0 * system.noise_se(z_query, z_ref_exp, weights=weights)
    exceed = (v_hat - system.v_true(z_query)).abs() - tol
    assert torch.all(exceed <= 0), f"超出恢复容差的量:\n{exceed}"
