"""经验漂移分解与支撑度诊断（docs/plan_v2.md 第 4 节漂移分解、支撑度行）。

漂移估计（转移增量的 kNN 回归）：查询点 z 处

    b̂(z) = (1/k)·Σ_{i∈N_k(z)} (z'_i − z_i)/Δt

(z_i, z'_i) 为相邻两步的转移对，Δt 为步长，N_k(z) 为 z 在转移起点集中
的 k 近邻（标准化坐标选取，见 estimators.density 的口径说明）。

梯度占比（ĝ 度量下的梯度 / 无散分解）：

    frac_grad = 1 − min_ψ [ Σ_i ‖b̂(z_i) + ĝ⁻¹∇ψ(z_i)‖²_ĝ / Σ_i ‖b̂(z_i)‖²_ĝ ]

ψ 为可学习标量势（小 MLP，tanh 隐层）；纯梯度流的 b̂ = −ĝ⁻¹∇V 使残差
趋零、占比趋一。有限基函数族给出的是占比下界，该口径随结果注明。
EXP-P2 以此作分层变量（plan_v2 第 8 节 EXP-P2 第 5 条：占比低的轨迹群
失败归因于漂移结构，变分主张未被触及）。

支撑度：kNN 半径诊断；低支撑段剔除规则预注册、阈值待校准（来源缺口：
CAL-P2/P3 噪声地板尚未运行），本模块只产出半径。

自检见 tests/test_drift.py：合成 OU 漂移恢复（噪声 SE 与平滑偏差界
闭式）、纯梯度流占比趋一、注入无散分量后占比对 Lyapunov 闭式预期。
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def estimate_drift_knn(
    z_from: torch.Tensor,
    z_next: torch.Tensor,
    dt: float,
    z_eval: torch.Tensor,
    k: int,
    standardize: bool = True,
    chunk_size: int = 1024,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """kNN 漂移回归。z_from/z_next 形状 (N, d)，z_eval 形状 (q, d)。"""
    if z_from.shape != z_next.shape:
        raise ValueError("转移对形状不一致")
    if dt <= 0:
        raise ValueError("Δt 须为正")
    increments = (z_next - z_from) / dt
    ref = z_from
    query = z_eval
    if standardize:
        mean = ref.mean(dim=0)
        std = ref.std(dim=0).clamp_min(1e-12)
        ref = (ref - mean) / std
        query = (query - mean) / std
    drifts, radii = [], []
    for i in range(0, query.shape[0], chunk_size):
        dists = torch.cdist(query[i : i + chunk_size], ref)
        knn = dists.topk(k, dim=-1, largest=False)
        drifts.append(increments[knn.indices].mean(dim=1))
        radii.append(knn.values[:, -1])
    drift = torch.cat(drifts)
    if not return_info:
        return drift
    return drift, {"radius": torch.cat(radii)}


class _ScalarPotential(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # 输出头零初始化：初始 ψ ≡ 0 → 初始残差比 = 1，留出最优值不会
        # 高于 1，占比获得构造性下界 ≥ 0。非零初始化在近奇异度量方向上
        # g⁻¹∇ψ 初始即爆炸（S2 真实几何实测占比 −60），下界性质失效。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def gradient_fraction(
    z_points: torch.Tensor,
    drift: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    seed: int,
    hidden: int = 64,
    n_iters: int = 1500,
    lr: float = 1e-2,
    val_fraction: float = 0.3,
) -> dict:
    """ĝ 度量下经验漂移的梯度占比（拟合口径，见模块 docstring）。

    ψ 在训练子集上优化，占比在留出子集上报告：梯度场与无散场的正交性
    只在总体期望下成立，训练集上的经验残差会因过参数化被压低（占比
    虚高），留出口径消除该偏差。seed 控制切分、初始化与优化（须先过
    guard，由调用方负责）。
    """
    torch.manual_seed(seed)
    n, d = z_points.shape
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(int(n * val_fraction), 1)
    idx_val, idx_train = perm[:n_val], perm[n_val:]

    g_all = metric_fn(z_points).detach()
    mean = z_points.mean(dim=0)
    std = z_points.std(dim=0).clamp_min(1e-12)
    x_all = ((z_points - mean) / std).detach()

    psi = _ScalarPotential(d, hidden).double()
    opt = torch.optim.Adam(psi.parameters(), lr=lr, weight_decay=1e-4)

    def residual_ratio_on(idx: torch.Tensor, create_graph: bool) -> torch.Tensor:
        x_req = x_all[idx].detach().requires_grad_(True)
        val = psi(x_req)
        (grad_x,) = torch.autograd.grad(val.sum(), x_req, create_graph=create_graph)
        grad_z = grad_x / std  # 链式法则回原坐标
        g = g_all[idx]
        nat_grad = torch.linalg.solve(g, grad_z.unsqueeze(-1)).squeeze(-1)
        resid = drift[idx] + nat_grad
        num = torch.einsum("ti,tij,tj->", resid, g, resid)
        den = torch.einsum("ti,tij,tj->", drift[idx], g, drift[idx])
        return num / den

    # ψ=0 基线先入榜：留出最优值 ≤ 1，占比下界 ≥ 0 构造性成立
    best_val = float(residual_ratio_on(idx_val, create_graph=False).detach())
    for it in range(n_iters):
        loss = residual_ratio_on(idx_train, create_graph=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == n_iters - 1:
            val_ratio = float(residual_ratio_on(idx_val, create_graph=False).detach())
            best_val = min(best_val, val_ratio)

    train_ratio = float(residual_ratio_on(idx_train, create_graph=False).detach())
    return {
        "fraction": 1.0 - best_val,
        "residual_ratio_val": best_val,
        "residual_ratio_train": train_ratio,
        "note": "留出 + 早停选择口径；有限基函数族（MLP）给出的是梯度占比的近似",
    }


def support_radius(
    z_query: torch.Tensor,
    z_ref: torch.Tensor,
    k: int,
    standardize: bool = False,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """支撑度诊断：第 k 近邻半径。剔除阈值待校准，此处只产出数值。"""
    ref, query = z_ref, z_query
    if standardize:
        mean = ref.mean(dim=0)
        std = ref.std(dim=0).clamp_min(1e-12)
        ref = (ref - mean) / std
        query = (query - mean) / std
    radii = []
    for i in range(0, query.shape[0], chunk_size):
        dists = torch.cdist(query[i : i + chunk_size], ref)
        radii.append(dists.kthvalue(k, dim=-1).values)
    return torch.cat(radii)
