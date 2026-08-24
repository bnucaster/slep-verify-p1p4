"""S2 规划段：CEM/MPC（plan_v2 第 2 节；P3 审议轨迹与导航效用共用）。

CEM（交叉熵方法）：维护每步动作的类别分布（horizon × 4），迭代
采样-评分-取精英-更新；MPC（模型预测控制）：每真实步重规划、只执行
首动作。

规划全部在学习的世界模型内部进行：从当前隐状态 h 与观测 o 出发，
候选动作序列经模型闭环推演（下一步观测用解码均值预测代入编码器），
代价取预测目标通道中心格激活的折扣和的负值——"预测自己站在目标上"
的概率越高代价越低。真实环境只用于执行首动作，符合审议轨迹是模型
自身隐状态演化的要求（plan_v2 第 2 节）。

导航效用（能力门用）：mpc_episode 在真实环境跑 MPC 至到达或超时，
成功率作 Û 的操作化（近距目标口径由调用方按 BFS 距离筛选）。
"""
from __future__ import annotations

import numpy as np
import torch

from slep.systems.s2_gridworld import GridWorld
from slep.systems.s2_world_model import S2WorldModel


class CEMPlanner:
    def __init__(
        self,
        model: S2WorldModel,
        view: int = 5,
        horizon: int = 12,
        n_samples: int = 64,
        n_elites: int = 8,
        n_iters: int = 3,
        gamma: float = 0.95,
        smoothing: float = 0.7,
    ):
        self.model = model
        self.view = view
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.gamma = gamma
        self.smoothing = smoothing

    def _rollout_cost(
        self, h: torch.Tensor, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """批量闭环推演。h (H,)，obs (D,)，actions (N, T) 整数；返回 (N,)。"""
        n = actions.shape[0]
        h_b = h.unsqueeze(0).expand(n, -1).contiguous()
        obs_b = obs.unsqueeze(0).expand(n, -1).contiguous()
        center = self.view * self.view + (self.view * self.view) // 2  # 目标通道中心格下标
        cost = torch.zeros(n)
        discount = 1.0
        for t in range(actions.shape[1]):
            a_onehot = torch.nn.functional.one_hot(actions[:, t], num_classes=4).float()
            inp = torch.cat([self.model.encoder(obs_b), a_onehot], dim=-1)
            h_b = self.model.gru(inp, h_b)
            obs_b = self.model.decoder_mean(h_b)
            cost = cost - discount * obs_b[:, center]
            discount *= self.gamma
        return cost

    def plan(
        self, h: torch.Tensor, obs: torch.Tensor, generator: torch.Generator
    ) -> tuple[int, dict]:
        """返回 (首动作, info)。info 含精英均值代价与最终分布（诊断）。"""
        probs = torch.full((self.horizon, 4), 0.25)
        best_cost = None
        with torch.no_grad():
            for _ in range(self.n_iters):
                actions = torch.multinomial(
                    probs, num_samples=self.n_samples, replacement=True, generator=generator
                ).T.contiguous()  # (N, T)
                cost = self._rollout_cost(h, obs, actions)
                elite_idx = cost.argsort()[: self.n_elites]
                elites = actions[elite_idx]  # (E, T)
                counts = torch.zeros(self.horizon, 4)
                for t in range(self.horizon):
                    counts[t] = torch.bincount(elites[:, t], minlength=4).float()
                probs = self.smoothing * (counts / self.n_elites) + (1 - self.smoothing) * probs
                best_cost = float(cost[elite_idx[0]])
        action = int(probs[0].argmax())
        return action, {"best_cost": best_cost, "first_step_probs": probs[0].tolist()}


def mpc_episode(
    model: S2WorldModel,
    env: GridWorld,
    planner: CEMPlanner,
    max_steps: int,
    generator: torch.Generator,
) -> dict:
    """MPC 导航回合：每步重规划。返回 success、steps、隐轨迹 (t, H)、动作表。

    隐轨迹是模型内部更新映射 Ψ 沿真实执行的演化（审议期间的候选推演
    不入轨迹），供 P3 的审议轨迹口径在确证阶段引用。
    """
    h = torch.zeros(model.hidden_dim)
    hs, actions = [], []
    obs = torch.from_numpy(env.observe())
    with torch.no_grad():
        for _ in range(max_steps):
            action, _ = planner.plan(h, obs, generator)
            a_onehot = torch.nn.functional.one_hot(torch.tensor(action), num_classes=4).float()
            inp = torch.cat([model.encoder(obs.unsqueeze(0)), a_onehot.unsqueeze(0)], dim=-1)
            h = model.gru(inp, h.unsqueeze(0)).squeeze(0)
            obs = torch.from_numpy(env.step(action))
            hs.append(h.clone())
            actions.append(action)
            if env.at_goal:
                break
    return {
        "success": env.at_goal,
        "steps": len(actions),
        "h_traj": torch.stack(hs) if hs else torch.zeros(0, model.hidden_dim),
        "actions": actions,
    }
