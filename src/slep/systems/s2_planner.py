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
        binarize_rollout: bool = True,
    ):
        """binarize_rollout：闭环推演时把预测观测按 0.5 取整后再送编码器。
        训练观测是二值的，sigmoid 均值直接回灌造成分布偏移并逐步放大；
        二值化抑制该偏移（开发族诊断实测缓解项，随配置记录）。"""
        self.model = model
        self.view = view
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.gamma = gamma
        self.smoothing = smoothing
        self.binarize_rollout = binarize_rollout

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
            pred = self.model.decoder_mean(h_b)
            cost = cost - discount * pred[:, center]
            obs_b = (pred > 0.5).float() if self.binarize_rollout else pred
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


class ExhaustiveMPCPlanner(CEMPlanner):
    """小动作空间的穷举式 MPC：4^horizon 条动作序列全评估，取最优首动作。

    动机（开发族配对评估实测）：CEM 的采样噪声叠加闭环推演误差使其在
    本任务上劣于随机基线；四动作 × 短视界可全枚举，消除优化器随机性。
    盲区回退：最优序列的预期目标激活低于 explore_threshold（模型未见
    目标、代价面平坦）时返回随机动作，保持探索性——此时 argmax 只会
    放大噪声。审议结构不变：候选未来仍由世界模型推演打分。
    """

    def __init__(
        self,
        model: S2WorldModel,
        view: int = 5,
        horizon: int = 1,
        gamma: float = 0.9,
        explore_threshold: float = 0.05,
        epsilon: float = 0.2,
        binarize_rollout: bool = False,
    ):
        """默认视界 1：开发族配对评估（n=60/段，dev_v2 s0 末检查点）实测
        视界 1 全面优于视界 4（盲区近段 0.42 对 0.17；多步闭环推演的
        墙体通道误差复利有害），且优于随机基线（可见段 0.58 对 0.53、
        盲区近段 0.42 对 0.25）。远段（BFS 5-6）与随机持平，受视野与
        回合步数预算限制，效用口径的距离带在冻结时定。"""
        super().__init__(
            model, view=view, horizon=horizon, gamma=gamma,
            binarize_rollout=binarize_rollout,
        )
        self.explore_threshold = explore_threshold
        self.epsilon = epsilon
        seqs = torch.cartesian_prod(*([torch.arange(4)] * horizon))
        self._all_seqs = seqs.reshape(-1, horizon)  # (4^h, h)

    def plan(
        self, h: torch.Tensor, obs: torch.Tensor, generator: torch.Generator
    ) -> tuple[int, dict]:
        """ε-greedy：plan 模式以 ε 概率随机化——确定性 argmax 在预测出错时
        会在相邻两格间死循环振荡至超时（开发族配对评估实测其把可见目标
        段成功率压到随机以下），ε 破环。"""
        with torch.no_grad():
            cost = self._rollout_cost(h, obs, self._all_seqs)
        best = int(cost.argmin())
        best_gain = float(-cost[best])
        if best_gain < self.explore_threshold or float(
            torch.rand(1, generator=generator)
        ) < self.epsilon:
            action = int(torch.randint(0, 4, (1,), generator=generator))
            mode = "explore" if best_gain < self.explore_threshold else "epsilon"
            return action, {"mode": mode, "best_gain": best_gain}
        return int(self._all_seqs[best, 0]), {"mode": "plan", "best_gain": best_gain}


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
