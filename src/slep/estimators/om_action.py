"""OM（Onsager–Machlup）作用量估计 Â_OM（docs/plan_v2.md 第 4 节）。

主项离散化：

    Â_OM = Σ_{t=0}^{n−1} (1/(4·T̂)) · ‖ (z_{t+1} − z_t)/Δt + ∇_ĝV̂(z_t^†) ‖²_ĝ(z_t^†) · Δt

z_{0..n} 为均匀步长 Δt 的潜轨迹（每点 d 维）；∇_ĝV̂ = ĝ⁻¹·∂V̂/∂z 为势
在度量 ĝ 下的自然梯度（母论文 A3 的漂移形式）；‖v‖²_ĝ = vᵀ·ĝ·v；T̂ 为
语义温度（确证中只能来自 P4 拟合斜率，自检中用真值）。取值点 z_t^†
按 grad_point 口径：

- "left"：取 z_t。与 Euler–Maruyama 模拟的噪声项精确对应（白化坐标下
  逐步贡献恒等于 ‖ε_t‖²/2，ε_t 为该步标准正态增量），自检口径；
- "mid"：取中点 (z_t + z_{t+1})/2。一般轨迹的默认，离散化偏差更小。

数值含义：逐步残差是速度对负自然梯度漂移的偏离，平方后按度量加权累计；
数值越低表示路径越接近该随机动力学最可能实现的走法。

R 修正项（曲率标量相关）不进入主项，按 plan_v2 第 4 节报为 O(T) 不确定
度；幅度评估待校准（来源缺口：CAL-P2/P3 噪声地板尚未运行），结果字典
以 curvature_correction 字段注明未含。

自检见 tests/test_om_action.py：合成 Langevin 恢复（χ² 闭式统计）与
弯曲坐标一致性（dt 收敛方向）。
"""
from __future__ import annotations

from typing import Callable

import torch


def om_action(
    traj: torch.Tensor,
    dt: float,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    temperature: float,
    grad_point: str = "mid",
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """单条轨迹的 Â_OM 主项。traj 形状 (n+1, d)，返回标量张量。

    potential_fn: (m, d) → (m,) 批量势函数（autograd 可微）。
    metric_fn: (m, d) → (m, d, d) 批量度量。
    return_info 附带 per_step（(n,) 逐步贡献）与 residual_sq（(n,)
    残差平方 ‖r‖²_ĝ，未除 4T，供分位诊断）。
    """
    if traj.ndim != 2 or traj.shape[0] < 2:
        raise ValueError("traj 须为 (n+1, d) 且至少两点")
    if dt <= 0 or temperature <= 0:
        raise ValueError("Δt 与 T̂ 须为正")
    if grad_point not in ("left", "mid"):
        raise ValueError(f"未知 grad_point 口径 {grad_point!r}")

    velocity = (traj[1:] - traj[:-1]) / dt  # (n, d)
    z_eval = traj[:-1] if grad_point == "left" else 0.5 * (traj[:-1] + traj[1:])

    z_req = z_eval.detach().requires_grad_(True)
    v_val = potential_fn(z_req)
    (grad_v,) = torch.autograd.grad(v_val.sum(), z_req)
    g = metric_fn(z_eval.detach())  # (n, d, d)
    nat_grad = torch.linalg.solve(g, grad_v.unsqueeze(-1)).squeeze(-1)  # ĝ⁻¹∂V

    resid = velocity + nat_grad
    resid_sq = torch.einsum("ti,tij,tj->t", resid, g, resid)  # ‖r‖²_ĝ
    per_step = resid_sq * dt / (4.0 * temperature)
    action = per_step.sum()
    if not return_info:
        return action
    return action, {
        "per_step": per_step,
        "residual_sq": resid_sq,
        "curvature_correction": "未含；R 修正项按 O(T) 不确定度另报（待校准）",
    }


def om_action_batch(
    trajs: torch.Tensor,
    dt: float,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    temperature: float,
    grad_point: str = "mid",
) -> torch.Tensor:
    """批量轨迹 (B, n+1, d) → (B,) 作用量。逐条调用单条实现。"""
    return torch.stack(
        [om_action(t, dt, potential_fn, metric_fn, temperature, grad_point) for t in trajs]
    )
