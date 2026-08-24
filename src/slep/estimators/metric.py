"""潜点处 Fisher 度量估计 ĝ（docs/plan_v2.md 第 4 节估计器层）。

观测模型：高斯解码器 p(x|z) = N(x; μ(z), Σ)。z 为潜变量（d 维实向量），
x 为观测（D 维实向量），μ: R^d → R^D 为解码器均值函数，Σ 为对角观测协方差
（D 个正方差，与 z 无关）。Σ 依赖 z 的情形闭式含附加项，本模块不支持，
调用方须保证方差与 z 无关。

该模型下潜点 z 处的 Fisher 信息矩阵有闭式

    g(z) = J(z)^T Σ^{-1} J(z)

J(z) = ∂μ/∂z 为解码器均值对潜变量的 Jacobian（D×d 矩阵），Σ^{-1} 为观测
协方差之逆。g(z) 是 d×d 对称半正定矩阵，含义：在潜空间沿单位方向移动时
观测分布的可分辨变化率，即观测空间几何经解码器拉回到潜空间得到的度量。

两条估计路径互为对拍：

1. fisher_pullback_gaussian —— 按上式直接计算，J 由 autograd 反向模式求出；
2. fisher_score_covariance_mc —— 按定义 g(z) = E_{x~p(·|z)}[s(x,z) s(x,z)^T]
   做蒙特卡洛。s(x,z) = ∇_z log p(x|z) 为得分（d 维向量，对数似然对潜变量
   的梯度），逐样本由 autograd 求出后取外积平均。两条路径只共享解码器
   前向函数，Jacobian 的构造方式相互独立，可交叉验证实现错误。

自检见 tests/test_metric.py：线性高斯解码器闭式对拍（CLAUDE.md 工程约定）。
几何门用的条件数阈值待校准（来源缺口：docs/plan_v2.md 第 5 节校准阶段
尚未运行），本模块只产出矩阵，不做任何判定。
"""
from __future__ import annotations

from typing import Callable

import torch


def _as_var_vector(
    obs_var: float | torch.Tensor, obs_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """把标量或 D 维方差参数规整成 D 维正向量。"""
    var = torch.as_tensor(obs_var, dtype=dtype, device=device)
    if var.ndim == 0:
        var = var.expand(obs_dim)
    if var.shape != (obs_dim,):
        raise ValueError(f"obs_var 形状 {tuple(var.shape)} 与观测维度 {obs_dim} 不符")
    if not torch.all(var > 0):
        raise ValueError("观测方差必须全部为正")
    return var


def decoder_mean_jacobian(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor
) -> torch.Tensor:
    """解码器均值在单个潜点 z（d 维向量）处的 Jacobian，返回 D×d 矩阵。

    取前向模式（jacfwd）：代价按输入维 d 计，观测维 D 远大于 d 的解码器
    （图像解码器 D 为像素数）下远快于反向模式。
    """
    if z.ndim != 1:
        raise ValueError("z 须为单个潜点（一维张量）；批量调用见 *_batch")
    return torch.func.jacfwd(decoder_mean)(z.detach())


def fisher_pullback_gaussian(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    obs_var: float | torch.Tensor,
) -> torch.Tensor:
    """路径一：Jacobian 拉回闭式 g(z) = J^T Σ^{-1} J，返回 d×d 矩阵。"""
    jac = decoder_mean_jacobian(decoder_mean, z)
    var = _as_var_vector(obs_var, jac.shape[0], jac.dtype, jac.device)
    return jac.T @ (jac / var.unsqueeze(-1))


def fisher_pullback_gaussian_batch(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    z_query: torch.Tensor,
    obs_var: float | torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """路径一的批量版：z_query 形状 (q, d)，返回 (q, d, d)。

    vmap + jacfwd 逐块求 Jacobian，chunk_size 限制 (chunk, D, d) 中间张量
    内存。decoder_mean 须为纯函数（单点 (d,) → (D,)，无批内状态）。
    """
    z_query = z_query.detach()
    jac_fn = torch.func.vmap(torch.func.jacfwd(decoder_mean))
    out = []
    for start in range(0, z_query.shape[0], chunk_size):
        jac = jac_fn(z_query[start : start + chunk_size])  # (c, D, d)
        var = _as_var_vector(obs_var, jac.shape[1], jac.dtype, jac.device)
        out.append(torch.einsum("cja,cjb->cab", jac / var.unsqueeze(0).unsqueeze(-1), jac))
    return torch.cat(out)


def fisher_score_covariance_mc(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    obs_var: float | torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
    chunk_size: int = 65536,
    return_se: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """路径二：MC 得分协方差 g(z) ≈ (1/n) Σ_i s_i s_i^T，返回 d×d 矩阵。

    采样 x_i = μ(z) + Σ^{1/2} ε_i（ε_i 为标准正态），得分 s_i 由 autograd 对
    log p(x_i|z) 中依赖 z 的项求梯度得到；对数配分项与 z 无关，梯度为零，
    省略后结果精确不变。

    return_se 为真时同时返回逐条目经验标准误矩阵 SE（d×d）：
    SE_ab = sqrt((M2_ab − ĝ_ab²)/n)，M2_ab 为外积条目 (s_a s_b) 平方的样本
    均值。SE 供对拍测试构造容差，判定阈值不由此产生。

    generator 的种子由调用方管理，必须先过 guard.assert_seed_allowed
    （CLAUDE.md 硬规则 1）。
    """
    if z.ndim != 1:
        raise ValueError("z 须为单个潜点（一维张量）")
    if n_samples <= 0:
        raise ValueError("n_samples 须为正")
    z = z.detach()
    mu = decoder_mean(z).detach()
    var = _as_var_vector(obs_var, mu.shape[-1], mu.dtype, mu.device)
    std = var.sqrt()

    def _log_lik_z_part(z_single: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
        m = decoder_mean(z_single)
        return -0.5 * (((x_single - m) ** 2) / var).sum()

    score_fn = torch.func.vmap(torch.func.grad(_log_lik_z_part, argnums=0), in_dims=(None, 0))

    d = z.shape[-1]
    acc = torch.zeros(d, d, dtype=mu.dtype, device=mu.device)
    acc_sq = torch.zeros(d, d, dtype=mu.dtype, device=mu.device)
    remaining = n_samples
    while remaining > 0:
        m_chunk = min(chunk_size, remaining)
        eps = torch.randn(
            (m_chunk, mu.shape[-1]), generator=generator, dtype=mu.dtype, device=mu.device
        )
        x = mu + eps * std
        scores = score_fn(z, x)
        acc += scores.T @ scores
        if return_se:
            sq = scores**2
            acc_sq += sq.T @ sq
        remaining -= m_chunk

    g_hat = acc / n_samples
    if not return_se:
        return g_hat
    second_moment = acc_sq / n_samples
    se = torch.clamp(second_moment - g_hat**2, min=0.0).sqrt() / (n_samples**0.5)
    return g_hat, se
