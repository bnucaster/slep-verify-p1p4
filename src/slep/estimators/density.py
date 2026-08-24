"""潜态密度与体积校正自信息估计（docs/plan_v2.md 第 4 节 p̂、Î 行）。

范围说明：本模块提供 CAL-P4（合成校准恢复仿射律）全链所需的 kNN 对数
密度路与体积校正自信息 Î。第 4 节要求的 kNN / normalizing flow 双路互证
与熵估计 Ŝ 属任务三，届时以本模块为其中一路。

kNN 对数密度（Kozachenko–Leonenko 型校正）：

    log p̂(z) = ψ(k) − ψ(N) − log c_d − d·log r_k(z)

z 为查询潜点（d 维实向量），k 为近邻数，N 为参照样本数，r_k(z) 为 z 到
参照集中第 k 近邻的欧氏距离，c_d = π^{d/2} / Γ(d/2 + 1) 为 d 维单位球
体积（Γ 为伽马函数），ψ 为 digamma 函数（ln Γ 的导数）。朴素估计
log[k / (N·c_d·r_k^d)] 在小 k 下有系统偏差；把 log k 与 log N 换成
ψ(k) 与 ψ(N) 后，局部密度恒定时估计的期望偏差为零，剩余偏差只来自
密度在 kNN 球内的变化。单点波动方差趋于 trigamma ψ₁(k)，自检容差由
此推导（tests/test_density.py）。

体积校正自信息（母论文 C2.1：密度须相对不变测度取）：

    Î(z) = −log p̂(z) + (1/2)·log det ĝ(z)

ĝ(z) 为 metric 模块的 Fisher 度量拉回闭式（d×d）。p̂ 是相对坐标体积的
密度，除以流形体积因子 sqrt(det ĝ)（对数域即加 (1/2)·log det ĝ）后，
Î 成为坐标不变量：潜坐标做可逆重参数化时 −log p̂ 与 (1/2)·log det ĝ 的
变化相互抵消。漏掉此项，P4（仿射律）的横纵轴都依赖参数化
（docs/plan_v2.md 第 4 节说明段）。

支撑度剔除等判定阈值待校准（来源缺口：docs/plan_v2.md 第 5 节校准阶段
尚未完成），本模块只产出数值与诊断，不做判定。
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from slep.estimators.metric import fisher_pullback_gaussian_batch


def unit_ball_log_volume(d: int) -> float:
    """log c_d = (d/2)·log π − log Γ(d/2 + 1)。"""
    return (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)


def log_density_knn(
    z_query: torch.Tensor,
    z_ref: torch.Tensor,
    k: int,
    return_info: bool = False,
    standardize: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """kNN 对数密度，公式见模块 docstring。返回 (q,)。

    z_query 形状 (q, d)，z_ref 形状 (N, d)。查询点不得与参照点重合
    （r_k = 0 时密度发散，直接报错）；对参照集内点估密度须由调用方
    先行去重或留一。

    standardize 置真时在逐维标准化坐标 x' = (z − μ)/σ（μ、σ 取自参照集）
    下取球形邻域，密度按仿射变换回原坐标：log p_z = log p_{x'} − Σ_j log σ_j。
    坐标尺度跨多个量级时（试点实测约 3 个量级）球形邻域在原坐标下退化，
    标准化是 kNN 路的推荐口径；两种口径的选择在结果中注明。

    return_info 附带 radius（(q,) 第 k 近邻距离；标准化口径下为标准化
    坐标的距离，供支撑度诊断与自检误差界使用）。
    """
    n_ref, d = z_ref.shape
    if not 1 <= k <= n_ref:
        raise ValueError(f"k={k} 超出参照集大小 {n_ref}")
    correction = 0.0
    if standardize:
        mean = z_ref.mean(dim=0)
        std = z_ref.std(dim=0).clamp_min(1e-12)
        z_query = (z_query - mean) / std
        z_ref = (z_ref - mean) / std
        correction = -torch.log(std).sum()
    dists = torch.cdist(z_query, z_ref)
    r_k = dists.kthvalue(k, dim=-1).values  # (q,)
    if not torch.all(r_k > 0):
        raise ValueError("存在查询点与参照点重合（r_k = 0），密度无定义")
    log_p = (
        torch.digamma(torch.tensor(float(k), dtype=z_query.dtype))
        - torch.digamma(torch.tensor(float(n_ref), dtype=z_query.dtype))
        - unit_ball_log_volume(d)
        - d * torch.log(r_k)
        + correction
    )
    if not return_info:
        return log_p
    return log_p, {"radius": r_k}


def self_information_knn(
    decoder_mean: Callable[[torch.Tensor], torch.Tensor],
    obs_var: float | torch.Tensor,
    z_query: torch.Tensor,
    z_ref: torch.Tensor,
    k: int,
    volume_correction: bool = True,
    chunk_size: int = 64,
    return_info: bool = False,
    standardize: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """体积校正自信息 Î = −log p̂ + (1/2)·log det ĝ，返回 (q,)。

    volume_correction 置假时退化为 −log p̂（含 / 不含校正双版本并报是
    EXP-P4 的预注册要求，docs/plan_v2.md 第 8 节第 7 条）。
    standardize 透传给 kNN 密度路（见 log_density_knn）。

    return_info 附带 log_density、logdet_g、radius，供诊断与消融。
    """
    log_p, info = log_density_knn(z_query, z_ref, k, return_info=True, standardize=standardize)
    if not volume_correction:
        if not return_info:
            return -log_p
        return -log_p, {"log_density": log_p, "logdet_g": None, "radius": info["radius"]}
    g = fisher_pullback_gaussian_batch(decoder_mean, z_query, obs_var, chunk_size=chunk_size)
    sign, logdet = torch.linalg.slogdet(g)
    if not torch.all(sign > 0):
        raise ValueError(
            "存在 det ĝ ≤ 0 的查询点，度量近奇异；体积校正无定义（几何门问题，"
            "阈值待校准）"
        )
    i_hat = -log_p + 0.5 * logdet
    if not return_info:
        return i_hat
    return i_hat, {"log_density": log_p, "logdet_g": logdet, "radius": info["radius"]}
