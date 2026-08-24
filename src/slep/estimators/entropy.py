"""潜态熵估计 Ŝ 双路（docs/plan_v2.md 第 4 节 p̂/Ŝ 行）。

路一（kNN，Kozachenko–Leonenko）：对样本集自身做留一 k 近邻，

    Ŝ_kNN = ψ(N) − ψ(k) + log c_d + (d/N)·Σ_i log r_k(i)

N 为样本数，d 为维数，c_d 为 d 维单位球体积，ψ 为 digamma 函数，
r_k(i) 为样本 i 到其第 k 近邻（不含自身）的欧氏距离。该式即
−(1/N)Σ log p̂(z_i) 用 density.log_density_knn 的留一版逐项代入。

路二（flow）：Ŝ_flow = −mean(log q̂(z))，q̂ 为 estimators.flow 拟合的
流密度，求值集须与训练集分离（否则过拟合使熵低估）。

双路互证是 EXP-P1（能量下降）主判据 Ŝ 的仪器要求；两路差值随结果
报告。自检见 tests/test_entropy.py：已知高斯恢复解析熵，容差由
log p 的方差与 kNN 波动闭式推导。
"""
from __future__ import annotations

import torch

from slep.estimators.density import unit_ball_log_volume
from slep.estimators.flow import FlowDensity


def entropy_knn(
    z_samples: torch.Tensor,
    k: int,
    chunk_size: int = 2048,
    return_info: bool = False,
    standardize: bool = False,
) -> float | tuple[float, dict[str, torch.Tensor]]:
    """kNN 留一熵估计，公式见模块 docstring。z_samples 形状 (N, d)。

    standardize 置真时在逐维标准化坐标下估计后按仿射变换换算回原坐标：
    H_z = H_std + Σ_j log σ_j（微分熵在可逆仿射下的精确关系）。各向异性
    强的分布（如聚合后验）里球形邻域在原坐标退化，标准化口径偏差更小；
    两口径的选择随结果注明。
    """
    n, d = z_samples.shape
    if not 1 <= k <= n - 1:
        raise ValueError(f"k={k} 超出留一可用范围 [1, {n - 1}]")
    correction = 0.0
    if standardize:
        std = z_samples.std(dim=0).clamp_min(1e-12)
        correction = float(torch.log(std).sum())
        z_samples = (z_samples - z_samples.mean(dim=0)) / std
    radii = []
    for i in range(0, n, chunk_size):
        dists = torch.cdist(z_samples[i : i + chunk_size], z_samples)
        # 第 k+1 小值即剔除自身（距离 0）后的第 k 近邻
        radii.append(dists.kthvalue(k + 1, dim=-1).values)
    r_k = torch.cat(radii)
    if not torch.all(r_k > 0):
        raise ValueError("样本集中存在重复点，留一 kNN 熵无定义")
    ent = (
        float(torch.digamma(torch.tensor(float(n))) - torch.digamma(torch.tensor(float(k))))
        + unit_ball_log_volume(d)
        + d * float(torch.log(r_k).mean())
        + correction
    )
    if not return_info:
        return ent
    return ent, {"radius": r_k}


def entropy_flow(flow_density: FlowDensity, z_eval: torch.Tensor) -> float:
    """流密度熵估计 −mean log q̂。z_eval 不得与流的训练集重叠。"""
    return float(-flow_density.log_prob(z_eval).mean())
