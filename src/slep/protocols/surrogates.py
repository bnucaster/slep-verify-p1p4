"""EXP-P2 主代理：长度与速度分布匹配的平滑随机路径（plan_v2 第 8 节
EXP-P2 第 2 条）。

两个口径：

1. smooth_random_surrogate（欧氏口径）：逐步速率 s_t = ‖z_{t+1} − z_t‖
   打乱重排（速度分布精确匹配）；方向取低通平滑的高斯随机方向（滑动
   平均窗长默认按观测方向自相关时间估计）；端点桥修正。
2. smooth_random_surrogate_metric（度量口径）：速率与方向都在局部
   度量下取——逐步度量速率 ‖Δz‖_ĝ 打乱重排，方向为局部白化坐标
   （ĝ = LLᵀ 的 Cholesky 白化）中的平滑随机单位向量，逐步按代理自身
   位置的度量积分。动机（CAL-P2 实测，results/calibration/cal_p2p3/）：
   欧氏口径的代理在度量"陡"方向乱走，仅动能项就把代理与观测完全
   分开，常数势打分同样完美分离，势项消融判据失去分辨力；度量口径
   按构造匹配动能项，分离只能来自势对齐，消融判据才有内容。

桥修正会轻微改变速率分布，实际（度量）长度比随结果返回作质检量。
时间打乱拼接、桥插值、相位随机化按 plan_v2 降为稳健性检查，不在
本模块（时间打乱已在 tests/test_om_action.py 用作方向性检验）。
"""
from __future__ import annotations

from typing import Callable

import torch


def direction_autocorr_time(traj: torch.Tensor, max_lag: int = 50) -> int:
    """观测方向自相关时间：相邻步方向余弦相似度均值落到 1/e 以下的
    首个滞后；用于匹配代理平滑窗。"""
    delta = traj[1:] - traj[:-1]
    norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    dirs = delta / norms
    n = dirs.shape[0]
    for lag in range(1, min(max_lag, n - 1)):
        cos = (dirs[:-lag] * dirs[lag:]).sum(-1).mean()
        if float(cos) < 0.3679:
            return lag
    return min(max_lag, n - 1)


def smooth_random_surrogate(
    traj: torch.Tensor,
    generator: torch.Generator,
    smooth_window: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """生成一条匹配代理。traj 形状 (n+1, d)；返回 (代理路径, 质检量)。

    smooth_window 为 None 时按 direction_autocorr_time(traj) 取。
    质检量：length_ratio（代理与观测总长之比，桥修正引起的偏离）。
    """
    n, d = traj.shape[0] - 1, traj.shape[1]
    delta = traj[1:] - traj[:-1]
    speeds = delta.norm(dim=-1)
    perm = torch.randperm(n, generator=generator)
    speeds_perm = speeds[perm]

    window = smooth_window or direction_autocorr_time(traj)
    raw = torch.randn((n + 2 * window, d), generator=generator, dtype=traj.dtype)
    kernel = torch.ones(window, dtype=traj.dtype) / window
    smoothed = torch.stack(
        [
            torch.conv1d(
                raw[:, j].reshape(1, 1, -1), kernel.reshape(1, 1, -1), padding=0
            ).reshape(-1)
            for j in range(d)
        ],
        dim=-1,
    )[:n]
    dirs = smoothed / smoothed.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    steps = speeds_perm.unsqueeze(-1) * dirs
    path = torch.cat([traj[:1], traj[0] + steps.cumsum(dim=0)], dim=0)
    # 端点桥修正：线性漂移平移
    tau = torch.linspace(0, 1, n + 1, dtype=traj.dtype).unsqueeze(-1)
    path = path + tau * (traj[-1] - path[-1])

    length_ratio = float(
        (path[1:] - path[:-1]).norm(dim=-1).sum() / speeds.sum().clamp_min(1e-30)
    )
    return path, {"length_ratio": length_ratio, "smooth_window": window}


def _smooth_unit_dirs(
    n: int, d: int, window: int, generator: torch.Generator, dtype
) -> torch.Tensor:
    raw = torch.randn((n + 2 * window, d), generator=generator, dtype=dtype)
    kernel = torch.ones(window, dtype=dtype) / window
    smoothed = torch.stack(
        [
            torch.conv1d(raw[:, j].reshape(1, 1, -1), kernel.reshape(1, 1, -1)).reshape(-1)
            for j in range(d)
        ],
        dim=-1,
    )[:n]
    return smoothed / smoothed.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def smooth_random_surrogate_metric(
    traj: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
    smooth_window: int | None = None,
    pin_endpoint: bool = True,
) -> tuple[torch.Tensor, dict]:
    """度量口径主代理，动机见模块 docstring。

    逐步构造：v_w 为白化坐标平滑随机单位向量，s_t 为打乱后的观测度量
    速率，步进 δz = s_t · L(z)⁻ᵀ v_w（L 为当前点度量的 Cholesky 因子，
    使 ‖δz‖_ĝ = s_t）。端点桥修正（pin_endpoint，正式口径恒开；关闭
    仅供构造性质自检）后返回度量长度比作质检量。
    """
    n, d = traj.shape[0] - 1, traj.shape[1]
    delta = traj[1:] - traj[:-1]
    g_obs = metric_fn(traj[:-1])
    metric_speeds = torch.einsum("ti,tij,tj->t", delta, g_obs, delta).clamp_min(0).sqrt()
    perm = torch.randperm(n, generator=generator)
    speeds_perm = metric_speeds[perm]

    window = smooth_window or direction_autocorr_time(traj)
    dirs_w = _smooth_unit_dirs(n, d, window, generator, traj.dtype)

    points = [traj[0]]
    for t in range(n):
        g_here = metric_fn(points[-1].unsqueeze(0))[0]
        chol = torch.linalg.cholesky(g_here)
        step = torch.linalg.solve_triangular(
            chol.T, (speeds_perm[t] * dirs_w[t]).unsqueeze(-1), upper=True
        ).squeeze(-1)
        points.append(points[-1] + step)
    path = torch.stack(points)
    if pin_endpoint:
        tau = torch.linspace(0, 1, n + 1, dtype=traj.dtype).unsqueeze(-1)
        path = path + tau * (traj[-1] - path[-1])

    delta_s = path[1:] - path[:-1]
    g_s = metric_fn(path[:-1])
    sur_len = torch.einsum("ti,tij,tj->t", delta_s, g_s, delta_s).clamp_min(0).sqrt().sum()
    metric_length_ratio = float(sur_len / metric_speeds.sum().clamp_min(1e-30))
    return path, {"metric_length_ratio": metric_length_ratio, "smooth_window": window}
