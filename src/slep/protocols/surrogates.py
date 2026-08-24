"""EXP-P2 主代理：长度与速度分布匹配的平滑随机路径（plan_v2 第 8 节
EXP-P2 第 2 条）。

构造：观测轨迹 z_{0..n} 的逐步速率 s_t = ‖z_{t+1} − z_t‖ 打乱重排
（速度分布精确匹配）；方向取低通平滑的高斯随机方向（滑动平均窗长
默认按观测方向自相关时间估计，保证平滑度可比）；从 z_0 累积成路径后
做端点桥修正（线性漂移平移把末端钉到 z_n）。桥修正会轻微改变速率，
实际长度比随结果返回作质检量。

时间打乱拼接、桥插值、相位随机化按 plan_v2 降为稳健性检查，不在
本模块（时间打乱已在 tests/test_om_action.py 用作方向性检验）。
"""
from __future__ import annotations

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
