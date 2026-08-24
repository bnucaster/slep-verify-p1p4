"""效用平台检测器（docs/plan_v2.md 第 8 节 EXP-P1 第 1 条冻结规则的实现）。

规则（plan_v2 原文预注册的结构与阈值）：Û 滑窗斜率连续 3 窗 < 1%/窗
→ 平台点；平台后窗口 = min(2×平台点, 训练总长)。

操作化：效用序列 u[0..n−1]（等间隔检查点）切成互不重叠的长度 W 窗；
第 j 窗的相对斜率

    rel_slope_j = (LS 斜率 × W) / max(|窗内均值|, ε)

LS 斜率为窗内对检查点序号的最小二乘一次拟合斜率，乘 W 得"每窗变化
量"，除以窗内均值化为相对量；ε 防零除。首个满足"自身及其前两窗相对
斜率绝对值均 < θ"的窗，其末端检查点为平台点。θ = 0.01（plan_v2 冻结
值）；窗长 W 由 CAL-P1 定标（本模块不内置默认值，强制显式传入）。

口径说明一：plan_v2 原文"斜率 < 1%/窗"按字面是单侧；此处取绝对值双
侧——效用骤降不是平台，单侧字面口径会把崩塌判成平台。该解释已写入
协议 v1.1 第 1 节。

口径说明二（平滑）：检测前可对序列做长度 s 的滑动均值（末端对齐：
平滑点 i 由原始点 i−s+1..i 平均而来），返回的平台点始终以原始检查点
序号计。动机：dev_v3 实测 SGD 检查点间波动 σ_rel≈0.0106 高于 θ=0.01，
未平滑判据对真饱和曲线失效；s 与 W 一并在协议 v1.1 阈值表冻结
（docs/protocol_v1.1_thresholds.json 的 plateau 节）。

自检与工作特性定标见 tests/test_plateau.py 与 scripts/run_cal_p1.py。
"""
from __future__ import annotations

import torch

REL_SLOPE_THRESHOLD = 0.01  # plan_v2 第 8 节 EXP-P1 第 1 条冻结值：1%/窗
CONSECUTIVE_WINDOWS = 3  # 同上：连续 3 窗


def window_relative_slopes(u: torch.Tensor, window: int, eps: float = 1e-12) -> torch.Tensor:
    """逐窗相对斜率，见模块 docstring。u 形状 (n,)，返回 (n//window,)。"""
    if window < 2:
        raise ValueError("窗长须 ≥ 2")
    n_win = u.shape[0] // window
    if n_win == 0:
        raise ValueError("序列短于一个窗")
    seg = u[: n_win * window].reshape(n_win, window).to(torch.float64)
    t = torch.arange(window, dtype=torch.float64)
    t_c = t - t.mean()
    slope = (seg * t_c).sum(dim=1) / (t_c**2).sum()
    return slope * window / seg.mean(dim=1).abs().clamp_min(eps)


def moving_average(u: torch.Tensor, smoothing: int) -> torch.Tensor:
    """长度 smoothing 的滑动均值，末端对齐；smoothing=1 原样返回。"""
    if smoothing < 1:
        raise ValueError("平滑长度须 ≥ 1")
    if smoothing == 1:
        return u
    if u.shape[0] < smoothing:
        raise ValueError("序列短于平滑长度")
    return u.to(torch.float64).unfold(0, smoothing, 1).mean(dim=1)


def detect_plateau(u: torch.Tensor, window: int, smoothing: int = 1) -> dict:
    """平台检测。返回 dict：

    plateau_index：平台点（原始检查点序号，未检出为 None）；
    post_window：(起, 止) 平台后窗口 = (平台点, min(2×平台点, n))；
    rel_slopes：平滑序列的逐窗相对斜率（诊断用）。
    """
    rel = window_relative_slopes(moving_average(u, smoothing), window)
    flat = rel.abs() < REL_SLOPE_THRESHOLD
    plateau_index = None
    for j in range(CONSECUTIVE_WINDOWS - 1, flat.shape[0]):
        if bool(flat[j - CONSECUTIVE_WINDOWS + 1 : j + 1].all()):
            plateau_index = (j + 1) * window + smoothing - 1
            break
    out = {
        "plateau_index": plateau_index,
        "rel_slopes": rel,
        "window": window,
        "smoothing": smoothing,
    }
    if plateau_index is not None:
        out["post_window"] = (plateau_index, min(2 * plateau_index, int(u.shape[0])))
    return out
