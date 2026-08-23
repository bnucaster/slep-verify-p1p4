"""估计器自检用合成系统：已知势的线性高斯观测世界。

定位：估计器单元自检（CLAUDE.md 工程约定的"合成真值恢复"一类）。
docs/plan_v2.md 第 5 节第 1 条要求的几何匹配合成校准系统属任务二，
须匹配真实系统 ĝ 谱与各向异性；本系统几何规整，只用于捕捉实现错误，
不为任何仪器门或阈值提供依据（该条同时禁止以规整流形上的恢复充当
仪器门信心）。

构造。潜变量 z ~ N(0, I_d)，d 为潜维；真值潜样本 z_i 直接可用（oracle
编码，绕过编码器噪声）。观测按

    x = W z + α·‖z‖²·u + ε

生成：x 为 D 维观测，W 为 D×d 列正交矩阵（‖W δ‖ = ‖δ‖，简化误差界），
u 为固定 D 维单位向量，α ≥ 0 为失配强度，ε ~ N(0, σ_x² I_D) 为观测噪声
（σ_x 为噪声标准差）。被评估的解码器为线性 μ(z) = W z，假定观测方差
σ_dec²（与数据真实噪声 σ_x² 允许不同）。α·‖z‖²·u 一项落在解码器族外，
使解码负对数似然的局部均值随 z 变化，势场形状因此已知。

真势（k 近邻球收缩到点的稠密极限，对观测噪声取期望）：

    V_true(z) = (D/2)·log(2π σ_dec²) + (D·σ_x² + α²·‖z‖⁴) / (2 σ_dec²)

首项为高斯负对数似然的配分常数；D·σ_x² 项是观测噪声在解码器方差
σ_dec² 下的平均代价；α²·‖z‖⁴ 项是解码器系统性失配的平方偏差，随 ‖z‖
增大而增大，构成非平凡势形。

finite-k 的 oracle 参照（邻居集 N_k(z) 固定，只对各邻居的 ε_i 取期望）：

    V_oracle(z) = (D/2)·log(2π σ_dec²)
        + ( mean_{i∈N_k(z)} ‖W(z_i − z) + α·‖z_i‖²·u‖² + D·σ_x² ) / (2 σ_dec²)

估计值与 V_oracle 之差只剩观测噪声贡献，逐点方差有闭式（见
noise_se_for_neighbors），自检容差由此推导（tests/test_potential.py）。

oracle 编码器后验：q(z | x_i) = N(z; z_i, τ² I_d)，τ 为已知后验标准差。
第二操作化（编码器后验加权）的权重由此生成；τ 与解码噪声独立，自检里
权重噪声与 NLL 噪声不耦合。真实系统中两者共享噪声的问题在校准阶段
处理（docs/plan_v2.md 第 5 节第 2 条同源噪声注入）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from slep import guard


@dataclass
class LinearGaussianSelfCheck:
    """已知势合成系统。字段见模块 docstring 的构造式。"""

    weight: torch.Tensor  # (D, d) 列正交
    u: torch.Tensor  # (D,) 单位向量
    alpha: float
    sigma_x: float
    sigma_dec: float

    @property
    def latent_dim(self) -> int:
        return self.weight.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.weight.shape[0]

    @classmethod
    def build(
        cls,
        latent_dim: int,
        obs_dim: int,
        alpha: float,
        sigma_x: float,
        sigma_dec: float,
        generator: torch.Generator,
    ) -> "LinearGaussianSelfCheck":
        raw = torch.randn((obs_dim, latent_dim), generator=generator, dtype=torch.float64)
        weight, _ = torch.linalg.qr(raw)  # 列正交化
        u_raw = torch.randn((obs_dim,), generator=generator, dtype=torch.float64)
        u = u_raw / u_raw.norm()
        return cls(weight=weight, u=u, alpha=alpha, sigma_x=sigma_x, sigma_dec=sigma_dec)

    def sample(self, n: int, generator: torch.Generator, purpose: str) -> tuple[torch.Tensor, torch.Tensor]:
        """采 n 个 (z_i, x_i)。generator 的种子须先过评估族守卫。

        purpose 进入守卫审计信息。generator 已带种子，此处再查一次
        initial_seed，保证采样路径无守卫遗漏（CLAUDE.md 硬规则 1）。
        """
        guard.assert_seed_allowed(int(generator.initial_seed()), purpose=purpose)
        z = torch.randn((n, self.latent_dim), generator=generator, dtype=torch.float64)
        eps = torch.randn((n, self.obs_dim), generator=generator, dtype=torch.float64)
        x = self.decoder_mean(z) + self.mismatch(z) + self.sigma_x * eps
        return z, x

    def decoder_mean(self, z: torch.Tensor) -> torch.Tensor:
        """被评估的线性解码器 μ(z) = W z，支持批量（… × d → … × D）。"""
        return z @ self.weight.T

    def mismatch(self, z: torch.Tensor) -> torch.Tensor:
        """解码器族外偏置 α·‖z‖²·u（… × d → … × D）。"""
        sq = (z**2).sum(dim=-1, keepdim=True)
        return self.alpha * sq * self.u

    def nll_const(self) -> float:
        import math

        return 0.5 * self.obs_dim * math.log(2 * math.pi * self.sigma_dec**2)

    def v_true(self, z_query: torch.Tensor) -> torch.Tensor:
        """稠密极限真势 V_true，见模块 docstring 公式。z_query 形状 (q, d)。"""
        sq = (z_query**2).sum(dim=-1)
        return self.nll_const() + (
            self.obs_dim * self.sigma_x**2 + self.alpha**2 * sq**2
        ) / (2 * self.sigma_dec**2)

    def _residual_mean(self, z_query: torch.Tensor, z_neighbors: torch.Tensor) -> torch.Tensor:
        """m_i = W(z_i − z) + α‖z_i‖²u 逐邻居，形状 (q, k, D)。"""
        delta = z_neighbors - z_query.unsqueeze(1)  # (q, k, d)
        return delta @ self.weight.T + self.mismatch(z_neighbors)

    def v_oracle_knn(self, z_query: torch.Tensor, z_neighbors: torch.Tensor) -> torch.Tensor:
        """finite-k oracle（等权），z_neighbors 形状 (q, k, d)，返回 (q,)。"""
        m = self._residual_mean(z_query, z_neighbors)
        mean_sq = (m**2).sum(dim=-1).mean(dim=-1)
        return self.nll_const() + (
            mean_sq + self.obs_dim * self.sigma_x**2
        ) / (2 * self.sigma_dec**2)

    def v_oracle_weighted(
        self, z_query: torch.Tensor, z_ref: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """加权 oracle：权重 weights 形状 (q, N)，对全体参照点 z_ref (N, d)。"""
        m = self._residual_mean(z_query, z_ref.unsqueeze(0).expand(z_query.shape[0], -1, -1))
        sq = (m**2).sum(dim=-1)  # (q, N)
        return self.nll_const() + (
            (weights * sq).sum(dim=-1) + self.obs_dim * self.sigma_x**2
        ) / (2 * self.sigma_dec**2)

    def noise_se(
        self, z_query: torch.Tensor, z_neighbors: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """V̂ 相对 oracle 的观测噪声标准误，逐查询点，形状 (q,)。

        推导：单邻居贡献 ‖m_i + ε_i‖²/(2σ_dec²)，其中 2·m_i·ε_i 与 ‖ε_i‖²
        不相关（高斯奇偶矩），方差分别为 4σ_x²‖m_i‖² 与 2D·σ_x⁴，合计

            Var_i = ( σ_x²·‖m_i‖² + D·σ_x⁴/2 ) / σ_dec⁴

        等权平均时 SE = sqrt(Σ_i Var_i)/k；权重 w_i 时 SE = sqrt(Σ_i w_i²·Var_i)。
        """
        m = self._residual_mean(z_query, z_neighbors)
        m_sq = (m**2).sum(dim=-1)  # (q, k) 或 (q, N)
        var_i = (
            self.sigma_x**2 * m_sq + self.obs_dim * self.sigma_x**4 / 2
        ) / self.sigma_dec**4
        if weights is None:
            k = z_neighbors.shape[1]
            return var_i.sum(dim=-1).sqrt() / k
        return (weights**2 * var_i).sum(dim=-1).sqrt()
