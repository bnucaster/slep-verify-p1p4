"""S2：GRU 世界模型（docs/plan_v2.md 第 2 节）。

结构：观测编码器（线性 + ReLU）→ GRUCell 循环更新 → 观测解码器
（两层 MLP，sigmoid 输出）。循环更新即母论文 Setting 1（系统七元组）
中的内部更新映射 Ψ；推理轨迹是隐状态 h 的演化。

观测模型与 S1 同为高斯解码器 p(o|h) = N(o; μ_θ(h), σ_dec² I)：o 为
view² 维局部观测，h 为 GRU 隐状态（潜变量，配置取 16 维），σ_dec 为
固定标准差，保证 Fisher 度量闭式拉回可用。训练目标为教师强制下的
下一步观测预测 NLL：h_t = Ψ(h_{t-1}, enc(o_t), a_t)，解码 μ_θ(h_t)
对 o_{t+1} 计负对数似然。
"""
from __future__ import annotations

import torch
from torch import nn


class S2WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 4,
        embed_dim: int = 32,
        hidden_dim: int = 16,
        sigma_dec: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, embed_dim), nn.ReLU())
        self.gru = nn.GRUCell(embed_dim + action_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, obs_dim)
        )
        self.hidden_dim = hidden_dim
        self.sigma_dec = sigma_dec

    def decoder_mean(self, h: torch.Tensor) -> torch.Tensor:
        """隐状态 (…, H) → 观测均值 (…, obs_dim)，sigmoid 值域 (0,1)。"""
        return torch.sigmoid(self.decoder(h))

    def decoder_mean_flat(self, h: torch.Tensor) -> torch.Tensor:
        """单点 (H,) → (obs_dim,)，供 estimators.metric 拉回。"""
        return self.decoder_mean(h)

    def hidden_trajectory(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """教师强制推进 Ψ，返回隐状态序列 (E, T, H)。

        obs 形状 (E, T+1, obs_dim)，act 形状 (E, T, action_dim)。
        h_t 由 (h_{t-1}, enc(o_t), a_t) 得到，t = 1…T。
        """
        n_ep, n_steps = act.shape[0], act.shape[1]
        h = torch.zeros(n_ep, self.hidden_dim, dtype=obs.dtype, device=obs.device)
        hs = []
        for t in range(n_steps):
            inp = torch.cat([self.encoder(obs[:, t]), act[:, t]], dim=-1)
            h = self.gru(inp, h)
            hs.append(h)
        return torch.stack(hs, dim=1)

    def rollout_loss(self, obs: torch.Tensor, act: torch.Tensor) -> dict[str, torch.Tensor]:
        """下一步预测 NLL（与 z 无关的配分常数略去），逐样本逐步平均。"""
        hs = self.hidden_trajectory(obs, act)  # (E, T, H)
        pred = self.decoder_mean(hs)  # (E, T, obs_dim)
        target = obs[:, 1:]  # o_{t+1}
        nll = ((target - pred) ** 2).sum(-1) / (2 * self.sigma_dec**2)  # (E, T)
        return {"total": nll.mean(), "mse_per_pixel": ((target - pred) ** 2).mean().detach()}
