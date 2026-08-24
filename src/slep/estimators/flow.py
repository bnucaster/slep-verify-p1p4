"""RealNVP 型归一化流密度估计（密度双路之二，docs/plan_v2.md 第 4 节 p̂ 行）。

选择自研最小实现而非外部库的原因：判定代码冻结（任务七）要求全部判定
链路可审计、可复现，仓库内约两百行的实现比第三方库更可控；该决定登记
于 pyproject.toml 注释。

结构与记账：

1. 输入标准化 x' = (x − μ)/σ（μ、σ 为训练集逐维均值与标准差）。这是
   仿射坐标变换，log p_x(x) = log p_{x'}(x') − Σ_j log σ_j，配套修正项
   在 log_prob 中一并返回。试点显示潜坐标尺度跨约 3 个量级，标准化是
   流训练与 kNN 球形邻域都需要的预处理。
2. K 个仿射耦合层：交替奇偶掩码；被掩码半边经两层 MLP 输出缩放 s 与
   平移 t，另半边变换 y = x·exp(s) + t；s 经 tanh 限幅（|s| ≤ clamp）
   保证数值稳定。log|det| = Σ s。
3. 基分布为标准正态。训练目标为最大似然（负对数似然的 Adam 下降），
   留出集 NLL 随训练报告，供质量审计；不做任何判定。

自检见 tests/test_entropy.py：已知高斯 / 相关高斯的密度与熵恢复。
"""
from __future__ import annotations

import math

import torch
from torch import nn


class _Coupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int, clamp: float):
        super().__init__()
        self.register_buffer("mask", mask)
        self.clamp = clamp
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        # 末层零初始化：初始为恒等变换，训练早期稳定
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x → (y, log|det|)。"""
        x_masked = x * self.mask
        s, t = self.net(x_masked).chunk(2, dim=-1)
        s = self.clamp * torch.tanh(s / self.clamp)
        keep = self.mask
        y = x_masked + (1 - keep) * (x * torch.exp(s) + t)
        log_det = ((1 - keep) * s).sum(dim=-1)
        return y, log_det


class RealNVP(nn.Module):
    def __init__(self, dim: int, n_couplings: int = 8, hidden: int = 64, clamp: float = 3.0):
        super().__init__()
        masks = []
        for i in range(n_couplings):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0
            masks.append(mask)
        self.couplings = nn.ModuleList(
            [_Coupling(dim, m, hidden, clamp) for m in masks]
        )
        self.dim = dim

    def log_prob_standardized(self, x: torch.Tensor) -> torch.Tensor:
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for coupling in self.couplings:
            x, ld = coupling(x)
            log_det = log_det + ld
        base = -0.5 * (x**2).sum(dim=-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + log_det


class FlowDensity:
    """标准化 + RealNVP 的密度估计器，log_prob 直接给原坐标密度。"""

    def __init__(self, flow: RealNVP, mean: torch.Tensor, std: torch.Tensor):
        self.flow = flow
        self.mean = mean
        self.std = std
        self._log_det_std = torch.log(std).sum()

    def log_prob(self, z: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, z.shape[0], chunk_size):
                x = (z[i : i + chunk_size] - self.mean) / self.std
                outs.append(self.flow.log_prob_standardized(x.float()).double())
        return torch.cat(outs) - self._log_det_std


def fit_flow_density(
    z_train: torch.Tensor,
    z_val: torch.Tensor,
    seed: int,
    n_couplings: int = 8,
    hidden: int = 64,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> tuple[FlowDensity, dict]:
    """最大似然训练流密度。seed 须先过 guard（调用方负责）。

    返回 (FlowDensity, 训练报告)。报告含逐 epoch 训练/留出 NLL（标准化
    坐标下，nats/样本），供质量审计；不做判定。
    """
    torch.manual_seed(seed)
    mean = z_train.mean(dim=0)
    std = z_train.std(dim=0).clamp_min(1e-12)
    x_train = ((z_train - mean) / std).float()
    x_val = ((z_val - mean) / std).float()

    flow = RealNVP(z_train.shape[1], n_couplings=n_couplings, hidden=hidden)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    gen = torch.Generator()
    gen.manual_seed(seed)

    history = []
    n = x_train.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = x_train[perm[i : i + batch_size]]
            nll = -flow.log_prob_standardized(batch).mean()
            opt.zero_grad()
            nll.backward()
            opt.step()
            total += float(nll.detach()) * batch.shape[0]
        with torch.no_grad():
            val_nll = float(-flow.log_prob_standardized(x_val).mean())
        history.append({"epoch": epoch, "train_nll": total / n, "val_nll": val_nll})

    flow.eval()
    report = {
        "history": history,
        "final_val_nll": history[-1]["val_nll"],
        "n_train": int(n),
        "n_val": int(x_val.shape[0]),
    }
    return FlowDensity(flow, mean, std), report
