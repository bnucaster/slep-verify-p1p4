"""S3：小型 Transformer 序列世界模型（plan_v2 第 2 节；协议 v1.2 附录，选项 E）。

角色：只跑 P1（能量下降），回应架构普适性追问。与 S2（GRU 世界模型）
共享数据分布（gridworld 随机策略 rollout）与效用口径（确定性近距导航），
差异仅在内部更新映射 Ψ 的架构：因果自注意力替代循环单元。

结构：token_t = 线性嵌入(o_t) + 线性嵌入(a_t) + 正弦位置码 → 因果
TransformerEncoder（层数/头数/维数见构造参数）→ 隐状态 h_t = 末层
残差流在位置 t 的输出 → 解码头（两层 MLP + sigmoid）预测 o_{t+1}。
配对约定与 S2 同构：hs[:, i] 解码预测 obs[:, i+1]。

观测模型：对角高斯 p(o|h) = N(o; μ_θ(h), diag(σ²))，σ 逐通道固定
（墙体 sigma_dec、目标 goal_sigma_dec），估计器接口
（hidden_trajectory / decoder_mean / obs_var / hidden_dim）与
S2WorldModel 对齐，估计器与管线原样复用。

规划：S3ExhaustivePlanner 与 S2 定稿口径同构（穷举视界 1 + 盲区回退
+ ε=0.2 破振荡环）；Transformer 无递归态，每步对完整历史重前向。
"""
from __future__ import annotations

import math

import torch
from torch import nn

from slep.systems.s2_gridworld import GridWorld


class S3TransformerWM(nn.Module):
    def __init__(
        self,
        obs_dim: int = 50,
        action_dim: int = 4,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 2,
        ff_dim: int = 64,
        max_len: int = 128,
        sigma_dec: float = 0.2,
        goal_sigma_dec: float | None = None,
    ):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.act_embed = nn.Linear(action_dim, d_model)
        pos = torch.zeros(max_len, d_model)
        t = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pos[:, 0::2] = torch.sin(t * div)
        pos[:, 1::2] = torch.cos(t * div)
        self.register_buffer("pos_enc", pos)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_dim, batch_first=True,
            dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, obs_dim))
        self.hidden_dim = d_model
        self.obs_dim = obs_dim
        self.sigma_dec = sigma_dec
        self.goal_sigma_dec = goal_sigma_dec
        var = torch.full((obs_dim,), sigma_dec**2)
        if goal_sigma_dec is not None:
            if obs_dim % 2 != 0:
                raise ValueError("双通道方差要求观测维数为偶数")
            var[obs_dim // 2:] = goal_sigma_dec**2
        self.register_buffer("obs_var", var)

    def decoder_mean(self, h: torch.Tensor) -> torch.Tensor:
        """隐状态 (…, d_model) → 下一步观测均值 (…, obs_dim)。"""
        return torch.sigmoid(self.decoder(h))

    def decoder_mean_flat(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder_mean(h)

    def hidden_trajectory(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """因果前向。obs (E, T+1, D)、act (E, T, A) → 隐状态 (E, T, d_model)。

        token_t 由 (o_t, a_t) 组成，t = 0…T−1；因果掩码保证 h_t 只依赖
        前缀。h_t 解码预测 o_{t+1}（与 S2 配对约定同构）。
        """
        n_steps = act.shape[1]
        tok = (self.obs_embed(obs[:, :n_steps]) + self.act_embed(act)
               + self.pos_enc[:n_steps])
        mask = nn.Transformer.generate_square_subsequent_mask(n_steps, dtype=tok.dtype)
        return self.blocks(tok, mask=mask, is_causal=True)

    def rollout_loss(self, obs: torch.Tensor, act: torch.Tensor) -> dict[str, torch.Tensor]:
        """下一步预测 NLL（配分常数略去），逐样本逐步平均。"""
        hs = self.hidden_trajectory(obs, act)
        pred = self.decoder_mean(hs)
        target = obs[:, 1:]
        nll = (((target - pred) ** 2) / self.obs_var).sum(-1) / 2
        return {"total": nll.mean(), "mse_per_pixel": ((target - pred) ** 2).mean().detach()}


class S3ExhaustivePlanner:
    """S2 定稿口径的 S3 版：视界 1 穷举 + 盲区回退 + ε 破环。"""

    def __init__(self, model: S3TransformerWM, view: int = 5,
                 explore_threshold: float = 0.05, epsilon: float = 0.2):
        self.model = model
        self.center = view * view + (view * view) // 2  # 目标通道中心像素
        self.explore_threshold = explore_threshold
        self.epsilon = epsilon

    def plan(self, obs_hist: list, act_hist: list,
             generator: torch.Generator) -> tuple[int, dict]:
        t = len(obs_hist)
        obs = torch.stack(obs_hist).unsqueeze(0)  # (1, t, D)
        gains = torch.empty(4)
        with torch.no_grad():
            for a in range(4):
                acts = act_hist + [a]
                act = torch.nn.functional.one_hot(
                    torch.tensor(acts), num_classes=4).float().unsqueeze(0)
                # hidden_trajectory 只取 obs[:, :len(acts)]，obs 长 t 恰配
                hs = self.model.hidden_trajectory(obs, act)
                pred = self.model.decoder_mean(hs[0, t - 1])
                gains[a] = pred[self.center]
        best = int(gains.argmax())
        best_gain = float(gains[best])
        if best_gain < self.explore_threshold or float(
            torch.rand(1, generator=generator)
        ) < self.epsilon:
            action = int(torch.randint(0, 4, (1,), generator=generator))
            mode = "explore" if best_gain < self.explore_threshold else "epsilon"
            return action, {"mode": mode, "best_gain": best_gain}
        return best, {"mode": "plan", "best_gain": best_gain}


def s3_mpc_episode(model: S3TransformerWM, env: GridWorld,
                   planner: S3ExhaustivePlanner, max_steps: int,
                   generator: torch.Generator) -> dict:
    """S3 导航回合（口径同 s2_planner.mpc_episode）。"""
    obs_hist = [torch.from_numpy(env.observe()).float()]
    act_hist: list[int] = []
    with torch.no_grad():
        for _ in range(max_steps):
            action, _ = planner.plan(obs_hist, act_hist, generator)
            nxt = env.step(action)
            act_hist.append(action)
            obs_hist.append(torch.from_numpy(nxt).float())
            if env.at_goal:
                break
    return {"success": env.at_goal, "steps": len(act_hist)}
