"""语义势估计 V̂（docs/plan_v2.md 第 4 节估计器层）。

被估计量：潜点 z 邻域局部观测律上解码负对数似然的均值，即母论文 D6
（语义势定义）的操作化。观测模型沿用 metric 模块的高斯解码器
p(x|z) = N(x; μ(z), Σ)：z 为潜变量（d 维），x 为观测（D 维），μ 为解码器
均值函数，Σ = diag(σ_1², …, σ_D²) 为与 z 无关的对角观测协方差。单个观测
的负对数似然

    NLL(x, z) = (1/2)·Σ_j [ (x_j − μ_j(z))² / σ_j² + log(2π σ_j²) ]

j 遍历观测维度；平方项是解码误差按解码器方差加权的代价，对数项是高斯
配分常数。

两种操作化，参照集为 N 对 (z_i, x_i)（z_i 为观测 x_i 的潜表示）：

1. potential_knn（主操作化）：取 z 在参照潜点集中的 k 个欧氏最近邻，

       V̂(z) = (1/k)·Σ_{i ∈ N_k(z)} NLL(x_i, z)

   N_k(z) 为最近邻下标集。近邻按潜坐标欧氏距离取，度量感知的邻域是
   可能的改进方向，留给校准阶段评估。k 敏感性须随结果报告
   （docs/plan_v2.md 第 4 节）。

2. potential_posterior_weighted（第二操作化）：以编码器后验密度为权重，

       V̂₂(z) = Σ_i w_i(z)·NLL(x_i, z)，
       w_i(z) ∝ q(z | x_i)，Σ_i w_i(z) = 1

   q(z|x_i) = N(z; m_i, diag(τ_i²)) 为编码器对观测 x_i 给出的高斯后验，
   m_i 为后验均值（d 维），τ_i² 为后验方差（d 维）。权重在对数域计算后
   softmax 归一化。

自检见 tests/test_potential.py：已知势合成系统上的恢复
（systems/selfcheck.py），容差由观测噪声方差闭式推导。支撑度剔除等
判定规则的阈值待校准（来源缺口：docs/plan_v2.md 第 5 节校准阶段尚未
运行），本模块只产出数值与诊断信息，不做判定。
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from slep.estimators.metric import _as_var_vector


def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, obs_var: float | torch.Tensor) -> torch.Tensor:
    """逐观测负对数似然。x 与 mu 形状 (…, D)，广播后返回 (…,)。"""
    var = _as_var_vector(obs_var, x.shape[-1], x.dtype, x.device)
    const = 0.5 * torch.log(2 * math.pi * var).sum()
    return 0.5 * (((x - mu) ** 2) / var).sum(dim=-1) + const


def potential_knn(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    obs_var: float | torch.Tensor,
    z_query: torch.Tensor,
    z_ref: torch.Tensor,
    x_ref: torch.Tensor,
    k: int,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """主操作化：潜空间 k 近邻球上解码 NLL 的均值。

    z_query 形状 (q, d)；z_ref 形状 (N, d)；x_ref 形状 (N, D)，与 z_ref 逐行
    对应。decoder_mean 须支持批量输入（(q, d) → (q, D)）。返回 (q,) 的 V̂。

    return_info 为真时附带诊断字典：indices（(q, k) 近邻下标）、
    radius（(q,) 第 k 近邻距离，供支撑度诊断与自检误差界使用）。
    """
    if not 1 <= k <= z_ref.shape[0]:
        raise ValueError(f"k={k} 超出参照集大小 {z_ref.shape[0]}")
    dists = torch.cdist(z_query, z_ref)  # (q, N)
    knn = dists.topk(k, dim=-1, largest=False)
    idx = knn.indices  # (q, k)
    mu_q = decoder_mean(z_query)  # (q, D)
    neighbor_x = x_ref[idx]  # (q, k, D)
    nll = gaussian_nll(neighbor_x, mu_q.unsqueeze(1), obs_var)  # (q, k)
    v_hat = nll.mean(dim=-1)
    if not return_info:
        return v_hat
    return v_hat, {"indices": idx, "radius": knn.values[:, -1]}


def potential_posterior_weighted(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    obs_var: float | torch.Tensor,
    z_query: torch.Tensor,
    x_ref: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_var: float | torch.Tensor,
    chunk_size: int = 32,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """第二操作化：编码器后验加权的解码 NLL。

    posterior_mean 形状 (N, d)：各参照观测的编码器后验均值。posterior_var
    为标量或 (N, d)：对应的后验方差。x_ref 形状 (N, D)。返回 (q,) 的 V̂₂。

    权重在对数域计算（含 −(1/2)Σ_d log(2πτ_id²) 项，后验方差逐点不同时
    影响相对权重）后按查询点 softmax 归一化。chunk_size 控制查询分块，
    限制 (chunk, N, d) 中间张量的内存。

    return_info 为真时附带诊断字典：weights（(q, N) 归一化权重）、
    ess（(q,) 有效样本量 1/Σ_i w_i²，供支撑度诊断使用）。
    """
    n_ref, d = posterior_mean.shape
    tau_sq = torch.as_tensor(posterior_var, dtype=posterior_mean.dtype, device=posterior_mean.device)
    if tau_sq.ndim == 0:
        tau_sq = tau_sq.expand(n_ref, d)
    if tau_sq.shape != (n_ref, d):
        raise ValueError(f"posterior_var 形状 {tuple(tau_sq.shape)} 与 (N={n_ref}, d={d}) 不符")
    if not torch.all(tau_sq > 0):
        raise ValueError("后验方差必须全部为正")

    log_norm = -0.5 * torch.log(2 * math.pi * tau_sq).sum(dim=-1)  # (N,)
    mu_q = decoder_mean(z_query)  # (q, D)

    v_parts, w_parts = [], []
    for start in range(0, z_query.shape[0], chunk_size):
        zc = z_query[start : start + chunk_size]  # (c, d)
        diff = zc.unsqueeze(1) - posterior_mean.unsqueeze(0)  # (c, N, d)
        log_w = log_norm.unsqueeze(0) - 0.5 * ((diff**2) / tau_sq.unsqueeze(0)).sum(dim=-1)
        weights = torch.softmax(log_w, dim=-1)  # (c, N)
        nll = gaussian_nll(x_ref.unsqueeze(0), mu_q[start : start + chunk_size].unsqueeze(1), obs_var)
        v_parts.append((weights * nll).sum(dim=-1))
        w_parts.append(weights)
    v_hat = torch.cat(v_parts)
    if not return_info:
        return v_hat
    weights = torch.cat(w_parts)
    return v_hat, {"weights": weights, "ess": 1.0 / (weights**2).sum(dim=-1)}
