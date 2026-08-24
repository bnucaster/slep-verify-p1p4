"""S1：β-VAE / dSprites（docs/plan_v2.md 第 2 节）。

观测模型为高斯解码器 p(x|z) = N(x; μ_θ(z), σ_dec² I)：x 为 64×64 图像展平
的 4096 维向量，z 为 d 维潜变量（配置取 8），μ_θ 为卷积解码器（sigmoid
输出，值域 (0,1) 匹配二值像素），σ_dec 为固定标准差。固定方差使 Fisher
度量有闭式拉回（estimators/metric.py 的前提）。

训练目标（β-VAE）：

    L = E_q[ NLL(x | z) ] + β·KL( q(z|x) ‖ N(0, I) )

q(z|x) = N(z; μ_φ(x), diag σ_φ²(x)) 为编码器高斯后验，NLL 为上述高斯
观测模型的负对数似然，KL 为后验对标准正态先验的散度，β 为解耦压力
系数（plan_v2 取 {1, 4}）。β 增大压向后验坍缩，维度坍缩正是几何门
（度量近奇异）要防的情形，试点谱提取须如实反映。

本模块只定义模型与损失；训练、检查点与谱提取在 scripts/run_s1_pilot.py。
"""
from __future__ import annotations

import torch
from torch import nn


class S1Encoder(nn.Module):
    def __init__(self, latent_dim: int, channels: list[int], fc_dim: int):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 4, 2, 1), nn.ReLU(),   # 64 → 32
            nn.Conv2d(c1, c2, 4, 2, 1), nn.ReLU(),  # 32 → 16
            nn.Conv2d(c2, c3, 4, 2, 1), nn.ReLU(),  # 16 → 8
            nn.Conv2d(c3, c4, 4, 2, 1), nn.ReLU(),  # 8 → 4
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c4 * 16, fc_dim), nn.ReLU())
        self.head_mean = nn.Linear(fc_dim, latent_dim)
        self.head_logvar = nn.Linear(fc_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(self.conv(x))
        return self.head_mean(h), self.head_logvar(h)


class S1Decoder(nn.Module):
    def __init__(self, latent_dim: int, channels: list[int], fc_dim: int):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.c4 = c4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim), nn.ReLU(), nn.Linear(fc_dim, c4 * 16), nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, 2, 1), nn.ReLU(),  # 4 → 8
            nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.ReLU(),  # 8 → 16
            nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.ReLU(),  # 16 → 32
            nn.ConvTranspose2d(c1, 1, 4, 2, 1),              # 32 → 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(-1, self.c4, 4, 4)
        return torch.sigmoid(self.deconv(h))


class S1BetaVAE(nn.Module):
    def __init__(self, latent_dim: int, channels: list[int], fc_dim: int, sigma_dec: float, beta: float):
        super().__init__()
        self.encoder = S1Encoder(latent_dim, channels, fc_dim)
        self.decoder = S1Decoder(latent_dim, channels, fc_dim)
        self.sigma_dec = sigma_dec
        self.beta = beta
        self.latent_dim = latent_dim

    def decoder_mean_flat(self, z: torch.Tensor) -> torch.Tensor:
        """单潜点 (d,) → 展平观测均值 (4096,)，供 estimators.metric 拉回。"""
        return self.decoder(z.unsqueeze(0)).reshape(-1)

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回总损失与分项（逐样本平均）。x 形状 (B, 1, 64, 64)。"""
        mu_z, logvar_z = self.encoder(x)
        std_z = torch.exp(0.5 * logvar_z)
        z = mu_z + std_z * torch.randn_like(std_z)
        x_mean = self.decoder(z)
        # 高斯 NLL；与 z 无关的配分常数略去（对梯度与谱提取无影响，
        # 绝对势值在估计器层另行带常数）。
        recon = ((x - x_mean) ** 2).sum(dim=(1, 2, 3)) / (2 * self.sigma_dec**2)
        kl_per_dim = 0.5 * (mu_z**2 + logvar_z.exp() - 1 - logvar_z)  # (B, d)
        kl = kl_per_dim.sum(dim=-1)
        total = recon.mean() + self.beta * kl.mean()
        return {
            "total": total,
            "recon": recon.mean().detach(),
            "kl": kl.mean().detach(),
            "kl_per_dim": kl_per_dim.mean(dim=0).detach(),
        }
