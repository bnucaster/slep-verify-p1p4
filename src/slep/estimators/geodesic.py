"""测地求解、唯一性认证与偏差估计（docs/plan_v2.md 第 4 节、第 3 节第 4 条）。

测地求解：固定端点 z_a、z_b，离散路径 z_0 = z_a, z_1, …, z_m = z_b
（内部点自由），最小化离散能量泛函

    E[path] = m · Σ_{t=0}^{m−1} (z_{t+1} − z_t)ᵀ · ĝ(z̄_t) · (z_{t+1} − z_t)

z̄_t 为段中点，m 为段数（参数间隔 Δτ = 1/m，E = Σ‖Δz‖²_ĝ/Δτ）。能量
泛函的极小元是以仿射参数化的测地线。优化：Adam 粗收敛后 LBFGS 精修，
残差取能量对内部点梯度的最大范数，作质检量报告。

多起点与唯一性认证（plan_v2 第 3 节第 4 条）：直线初始化加带种子的
平滑扰动初始化各跑一遍，全部收敛路径的两两最大逐点距离为
uniqueness_spread；认证阈值待校准（来源缺口：CAL-P2/P3 残差基准尚未
运行），本模块只报数值。多条近最短路径并存时 spread 大，认证应拒绝。

测地偏差：轨迹对同端点测地线的归一化偏差面积

    dev = [ Σ_t dist_ĝ(traj_t, geo)·Δs_t ] / L_geo²

dist_ĝ(p, geo) = min_q sqrt((p−q)ᵀ ĝ(p) (p−q)) 对测地离散点取最小（局部
度量近似，偏差远小于度量变化尺度时成立）；Δs_t 为轨迹相邻点的 ĝ 弧长；
L_geo 为测地线 ĝ 长度。除以 L² 使量纲为零、对整体缩放不变。

自检见 tests/test_geodesic.py：平坦度量退化为直线；合成校准系统上与
白化直线拉回的闭式测地对拍；多起点在双路几何上探出非唯一。
"""
from __future__ import annotations

from typing import Callable

import torch

MetricFn = Callable[[torch.Tensor], torch.Tensor]


def _energy(full_path: torch.Tensor, metric_fn: MetricFn) -> torch.Tensor:
    delta = full_path[1:] - full_path[:-1]  # (m, d)
    mids = 0.5 * (full_path[1:] + full_path[:-1])
    g = metric_fn(mids)
    m = delta.shape[0]
    return m * torch.einsum("ti,tij,tj->", delta, g, delta)


def path_length(full_path: torch.Tensor, metric_fn: MetricFn) -> torch.Tensor:
    delta = full_path[1:] - full_path[:-1]
    mids = 0.5 * (full_path[1:] + full_path[:-1])
    g = metric_fn(mids)
    return torch.einsum("ti,tij,tj->t", delta, g, delta).clamp_min(0).sqrt().sum()


def _optimize_path(
    init_interior: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    metric_fn: MetricFn,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    precond: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """在预条件坐标 w = δz·L 中优化（L 为参考度量的 Cholesky 因子）。

    潜坐标尺度可跨多个量级（试点实测约 3 个量级），直接在 z 坐标优化
    病态；δz = w·L⁻ᵀ 使 ‖δz‖²_g_ref = ‖w‖²，Adam/LBFGS 在近各向同性
    的 w 空间内收敛。
    """
    base = init_interior.detach()
    precond_inv = torch.linalg.inv(precond)  # L⁻¹，行约定 δz = w @ L⁻¹
    w = torch.zeros_like(base).requires_grad_(True)

    def full() -> torch.Tensor:
        interior = base + w @ precond_inv
        return torch.cat([z_a.unsqueeze(0), interior, z_b.unsqueeze(0)])

    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(n_adam):
        loss = _energy(full(), metric_fn)
        opt.zero_grad()
        loss.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS([w], max_iter=n_lbfgs, tolerance_grad=1e-12,
                              tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = _energy(full(), metric_fn)
        loss.backward()
        return loss

    lbfgs.step(closure)
    loss = _energy(full(), metric_fn)
    (grad_w,) = torch.autograd.grad(loss, w)
    return full().detach(), float(loss.detach()), float(grad_w.abs().max())


def solve_geodesic(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    metric_fn: MetricFn,
    n_segments: int = 32,
    n_starts: int = 4,
    perturb_scale: float = 0.3,
    generator: torch.Generator | None = None,
    n_adam: int = 300,
    n_lbfgs: int = 200,
    lr: float = 0.05,
) -> dict:
    """多起点测地求解。返回 dict：

    path（最低能量路径 (m+1, d)）、energy、residual（该路径梯度最大范数）、
    all_energies、uniqueness_spread（各收敛路径两两最大逐点欧氏距离的
    最大值；认证阈值待校准）、length（ĝ 弧长）。

    起点 0 为直线；其余为直线加正弦包络随机方向扰动（幅度 =
    perturb_scale × 端点距离，generator 控制，保证可复现）。
    """
    d = z_a.shape[0]
    tau = torch.linspace(0, 1, n_segments + 1, dtype=z_a.dtype).unsqueeze(-1)
    straight = z_a * (1 - tau) + z_b * tau  # (m+1, d)
    chord = float((z_b - z_a).norm())

    inits = [straight[1:-1]]
    for _ in range(n_starts - 1):
        direction = torch.randn((d,), generator=generator, dtype=z_a.dtype)
        direction = direction / direction.norm()
        envelope = torch.sin(torch.pi * tau[1:-1, 0]).unsqueeze(-1)
        inits.append(straight[1:-1] + perturb_scale * chord * envelope * direction)

    # 预条件：弦中点度量的 Cholesky 因子（残差亦在该白化坐标下报告）
    mid = (0.5 * (z_a + z_b)).unsqueeze(0)
    precond = torch.linalg.cholesky(metric_fn(mid)[0])

    paths, energies, residuals = [], [], []
    for init in inits:
        path, energy, resid = _optimize_path(
            init, z_a, z_b, metric_fn, n_adam, n_lbfgs, lr, precond
        )
        paths.append(path)
        energies.append(energy)
        residuals.append(resid)

    # 唯一性散布取度量口径：路径两两的归一化偏差面积（对称化取大者）。
    # 欧氏逐点距离在强各向异性几何下失真（软方向的大坐标差在度量下
    # 可忽略，CAL-P3 实测欧氏散布虚高两个量级），降为次要诊断。
    spread, spread_euclid = 0.0, 0.0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            dev_ij = geodesic_deviation(paths[i], paths[j], metric_fn)["normalized_area"]
            dev_ji = geodesic_deviation(paths[j], paths[i], metric_fn)["normalized_area"]
            spread = max(spread, dev_ij, dev_ji)
            spread_euclid = max(spread_euclid, float((paths[i] - paths[j]).norm(dim=-1).max()))

    best = int(torch.tensor(energies).argmin())
    return {
        "path": paths[best],
        "energy": energies[best],
        "residual": residuals[best],
        "all_energies": energies,
        "all_paths": paths,
        "uniqueness_spread": spread,
        "uniqueness_spread_euclid": spread_euclid,
        "length": float(path_length(paths[best], metric_fn)),
        "uniqueness_note": "认证阈值待校准（CAL-P2/P3 残差基准，任务四）",
    }


def geodesic_deviation(
    traj: torch.Tensor, geodesic: torch.Tensor, metric_fn: MetricFn
) -> dict:
    """轨迹对测地线的归一化偏差面积，定义见模块 docstring。

    traj (n, d)，geodesic (m+1, d)。返回 normalized_area 与中间量。
    """
    g_traj = metric_fn(traj)  # (n, d, d)
    diff = traj.unsqueeze(1) - geodesic.unsqueeze(0)  # (n, m+1, d)
    dist_sq = torch.einsum("nmi,nij,nmj->nm", diff, g_traj, diff)
    dist = dist_sq.clamp_min(0).sqrt().min(dim=1).values  # (n,)

    delta = traj[1:] - traj[:-1]
    mids_g = metric_fn(0.5 * (traj[1:] + traj[:-1]))
    seg_len = torch.einsum("ti,tij,tj->t", delta, mids_g, delta).clamp_min(0).sqrt()
    # 梯形式：段长乘段两端距离均值
    area = (seg_len * 0.5 * (dist[1:] + dist[:-1])).sum()
    length_geo = path_length(geodesic, metric_fn)
    return {
        "normalized_area": float(area / length_geo**2),
        "area": float(area),
        "geodesic_length": float(length_geo),
        "max_pointwise_dist": float(dist.max()),
    }
