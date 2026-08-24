"""几何匹配的合成 Langevin 校准系统（docs/plan_v2.md 第 5 节第 1、2 条）。

构造思路
--------
1. 扭曲映射 φ: R^d → R^d，两层复合 φ = φ_rad ∘ φ_ax：

   逐轴层  φ_ax(z) = diag(s)·ψ(R z + c)，ψ(v)_i = v_i + a·τ·log cosh(v_i/τ)

   s 为 d 维正尺度向量，R 为 d×d 正交矩阵，c 为 d 维偏置；ψ 逐元素作用，
   a ∈ [0, 1) 为逐轴扭曲强度，τ > 0 为其长度尺度。ψ 斜率
   m_i(v) = 1 + a·tanh(v_i/τ) ∈ (1−a, 1+a) 处处为正。

   径向层  φ_rad(y) = y·f(‖y‖)，f(r) = exp(η·tanh( log(r/r₀) / τ₀ ))

   η ≥ 0 为共同因子强度，r₀ 为中心半径，τ₀ 为对数半径的长度尺度
   （无量纲化后径向应变 ρ'/f = 1 + η·sech²(·)/τ₀ 与 r 的绝对尺度无关）。
   φ_rad 的 Jacobian 为 f·(I − ŷŷᵀ) + ρ'·ŷŷᵀ（ŷ = y/‖y‖，
   ρ(r) = r·f(r)，η ≥ 0 时 ρ' > 0），即 d−1 个切向方向共同乘 f、径向
   乘 ρ'——近共形的整体缩放。

   两层职责分开：试点实测显示真实系统的 log det 大幅波动来自整个
   Jacobian 的共同缩放（各方向相关），条件数波动温和；η（共同因子）
   贡献 log det 方差且对条件数影响有限，a（逐轴独立调制）加局部纹理，
   条件数基准由 s 的秩谱决定。构造性上限：共同因子的对数摆幅受
   log‖φ(z)‖ = log(σ_dec‖u‖) 的固有散布约束（f = ‖y‖/‖φ_ax‖），径向
   应变又随 η/τ₀ 增长，故对数半径散布小的目标系统无法完全复现特别大
   的 log det 波动；拟合器设 η ≤ 3τ₀，残余缺口在匹配报告中显式记录。
   φ 全局可逆：径向一维二分解 ρ(r) = ‖y‖，再逐元素 Newton 解 ψ。

2. 解码器 μ(z) = B·φ(z)，B 为 D×d 列正交矩阵，观测方差 σ_dec²。Fisher
   度量闭式

       g(z) = ∇φᵀ∇φ/σ_dec²，∇φ = J_rad·diag(s·m)·R

   log det g = 2[Σ_i log(s_i m_i) + (d−1)·log f + log ρ'] − 2d·log σ_dec。

3. 白化坐标 u(z) = φ(z)/σ_dec 下度量恒为单位阵。在 u 空间模拟平直
   过阻尼 Langevin：

       du = −∇W(u)·dt + sqrt(2T)·dB_t

   W(u) = (1/2)·Σ_i w_i·(u_i − u0_i)² 为各向异性二次势（w 为正刚度，
   u0 为阱心），T 为真温度，dB_t 为标准布朗增量。不变测度为
   N(u0, diag(T/w))，可精确采样（exact_sample，供区分离散化偏差与
   估计器误差）。

4. 拉回 z 坐标（z = φ⁻¹(σ_dec·u)）后坐标密度为

       p_z(z) ∝ exp(−W(u(z))/T)·|det ∇φ(z)|/σ_dec^d
              = exp(−V(z)/T)·sqrt(det g(z))

   V(z) := W(u(z))。体积校正自信息 Î = −log p_z + (1/2)·log det g
   = V/T + 常数：仿射律以真斜率 1/T 精确成立，全链恢复有已知真值。

5. 观测生成：x = μ(z) + b(z) + ε，ε ~ N(0, σ_x² I_D)，解码器族外偏置

       b(z) = sqrt(2 σ_dec²·W(u(z)))·û

   û ⊥ col(B) 的单位向量。由正交性，解码 NLL 的局部均值（V̂ 的估计
   对象）= W(u(z)) + [D·σ_x² + 邻域离散项]/(2σ_dec²) + 配分常数，与
   动力学势严格同源。

与 systems/selfcheck.py 的关系：selfcheck 是估计器单元自检（几何规整）；
本系统承担第 5 节第 1 条的几何匹配职责，谱形以试点实测为目标，不以
规整高斯流形充当仪器门信心。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class CalLangevinSystem:
    s: torch.Tensor  # (d,) 正尺度
    warp_a: float  # 扭曲强度 a ∈ [0, 1)
    warp_tau: float  # 变化长度尺度 τ
    rot: torch.Tensor  # (d, d) 正交
    shift: torch.Tensor  # (d,) 偏置 c
    basis: torch.Tensor  # (D, d) 列正交
    u_hat: torch.Tensor  # (D,) 单位向量，⊥ col(basis)
    sigma_dec: float
    sigma_x: float
    well_w: torch.Tensor  # (d,) 势阱刚度
    well_u0: torch.Tensor  # (d,) 势阱中心（u 空间）
    temperature: float
    radial_eta: float = 0.0  # 共同因子强度 η ≥ 0
    radial_r0: float = 1.0  # 共同因子中心 r₀
    radial_tau0: float = 1.0  # 共同因子长度尺度 τ₀

    def __post_init__(self):
        if not 0.0 <= self.warp_a < 1.0:
            raise ValueError("可逆性要求 0 ≤ a < 1")
        if self.warp_tau <= 0 or self.radial_tau0 <= 0:
            raise ValueError("τ 与 τ₀ 须为正")
        if self.radial_eta < 0:
            raise ValueError("径向层可逆性（ρ' > 0）按 η ≥ 0 论证，负值不支持")

    @property
    def latent_dim(self) -> int:
        return self.s.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.basis.shape[0]

    # ---- 映射与几何 ----

    def _v(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.rot.T + self.shift

    def _psi(self, v: torch.Tensor) -> torch.Tensor:
        # log cosh 的数值稳定形式：|x| + log((1+e^{-2|x|})/2)
        x = v / self.warp_tau
        logcosh = x.abs() + torch.log1p(torch.exp(-2 * x.abs())) - math.log(2.0)
        return v + self.warp_a * self.warp_tau * logcosh

    def _slope(self, v: torch.Tensor) -> torch.Tensor:
        """ψ 的斜率 m(v) = 1 + a·tanh(v/τ)。"""
        return 1.0 + self.warp_a * torch.tanh(v / self.warp_tau)

    def _inner(self, z: torch.Tensor) -> torch.Tensor:
        """逐轴层 φ_ax(z) = diag(s)·ψ(Rz + c)。"""
        return self.s * self._psi(self._v(z))

    def _radial_f(self, r: torch.Tensor) -> torch.Tensor:
        """切向共同因子 f(r) = exp(η·tanh(log(r/r₀)/τ₀))；r = 0 处取极限值。"""
        arg = torch.log(r.clamp_min(1e-300) / self.radial_r0) / self.radial_tau0
        return torch.exp(self.radial_eta * torch.tanh(arg))

    def _radial_rho_prime(self, r: torch.Tensor) -> torch.Tensor:
        """径向导数 ρ'(r) = f·(1 + η·sech²(log(r/r₀)/τ₀)/τ₀)；η ≥ 0 时为正。"""
        arg = torch.log(r.clamp_min(1e-300) / self.radial_r0) / self.radial_tau0
        f = torch.exp(self.radial_eta * torch.tanh(arg))
        return f * (1.0 + self.radial_eta * (1.0 - torch.tanh(arg) ** 2) / self.radial_tau0)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        y1 = self._inner(z)
        r = y1.norm(dim=-1, keepdim=True)
        return y1 * self._radial_f(r)

    def phi_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """∇φ = J_rad·diag(s·m)·R，(..., d) → (..., d, d)。"""
        sm = self.s * self._slope(self._v(z))
        jac_ax = sm.unsqueeze(-1) * self.rot
        y1 = self._inner(z)
        r = y1.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        y_hat = (y1 / r).unsqueeze(-1)  # (..., d, 1)
        f = self._radial_f(r).unsqueeze(-1)
        rho_p = self._radial_rho_prime(r).unsqueeze(-1)
        eye = torch.eye(self.latent_dim, dtype=z.dtype, device=z.device)
        jac_rad = f * eye + (rho_p - f) * (y_hat @ y_hat.transpose(-1, -2))
        return jac_rad @ jac_ax

    def phi_inverse(self, y: torch.Tensor, tol: float = 1e-10, max_iter: int = 200) -> torch.Tensor:
        """两段反演：径向一维二分解 ρ(r) = ‖y‖，再逐元素 Newton 解 ψ。

        径向段用二分而非 Newton：ρ 严格增但非凸，强 η 下 Newton 可振荡；
        f ∈ [e^{−η}, e^{η}] 给出必含根的区间 [‖y‖·e^{−η}, ‖y‖·e^{η}]，
        二分每步减半，100 步后区间宽度缩小 2^{-100}，达浮点极限。
        """
        r_y = y.norm(dim=-1, keepdim=True)
        lo = r_y * math.exp(-self.radial_eta)
        hi = r_y * math.exp(self.radial_eta)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            too_small = mid * self._radial_f(mid) < r_y
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)
        r = 0.5 * (lo + hi)
        y1 = torch.where(r_y > 0, y * (r / r_y.clamp_min(1e-30)), y)

        w = y1 / self.s
        v = w.clone()
        w_scale = w.abs().clamp_min(1.0)
        for _ in range(max_iter):
            resid = self._psi(v) - w
            if float((resid.abs() / w_scale).max()) < tol:
                break
            v = v - resid / self._slope(v)
        else:
            raise RuntimeError(f"phi_inverse 逐轴段未收敛，残差 {float(resid.abs().max()):.2e}")
        return (v - self.shift) @ self.rot

    def u_of_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.phi(z) / self.sigma_dec

    def z_of_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.phi_inverse(u * self.sigma_dec)

    def metric_true(self, z: torch.Tensor) -> torch.Tensor:
        """闭式 g(z) = ∇φᵀ∇φ / σ_dec²，(..., d, d)。"""
        jac = self.phi_jacobian(z)
        return jac.transpose(-1, -2) @ jac / self.sigma_dec**2

    def logdet_metric_true(self, z: torch.Tensor) -> torch.Tensor:
        """闭式 log det g（见模块 docstring 第 2 条）。"""
        sm = self.s * self._slope(self._v(z))
        y1 = self._inner(z)
        r = y1.norm(dim=-1)
        d = self.latent_dim
        return 2.0 * (
            torch.log(sm).sum(dim=-1)
            + (d - 1) * torch.log(self._radial_f(r))
            + torch.log(self._radial_rho_prime(r))
        ) - 2.0 * d * math.log(self.sigma_dec)

    # ---- 解码器与观测 ----

    def decoder_mean(self, z: torch.Tensor) -> torch.Tensor:
        return self.phi(z) @ self.basis.T

    def decoder_mean_flat(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_mean(z)

    def well(self, u: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.well_w * (u - self.well_u0) ** 2).sum(dim=-1)

    def potential_dyn(self, z: torch.Tensor) -> torch.Tensor:
        """动力学势 V(z) = W(u(z))。"""
        return self.well(self.u_of_z(z))

    def nll_const(self) -> float:
        return 0.5 * self.obs_dim * math.log(2 * math.pi * self.sigma_dec**2)

    def potential_estimand(self, z: torch.Tensor) -> torch.Tensor:
        """V̂ 的稠密极限估计对象：V(z) + (D σ_x²)/(2σ_dec²) + 配分常数。"""
        return (
            self.potential_dyn(z)
            + self.obs_dim * self.sigma_x**2 / (2 * self.sigma_dec**2)
            + self.nll_const()
        )

    def mismatch(self, z: torch.Tensor) -> torch.Tensor:
        """b(z) = sqrt(2σ_dec²·W(u(z)))·û。"""
        amp = torch.sqrt(2 * self.sigma_dec**2 * self.potential_dyn(z))
        return amp.unsqueeze(-1) * self.u_hat

    def sample_observations(self, z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        eps = torch.randn((z.shape[0], self.obs_dim), generator=generator, dtype=z.dtype)
        return self.decoder_mean(z) + self.mismatch(z) + self.sigma_x * eps

    # ---- 动力学与采样 ----

    def simulate_chains(
        self,
        n_chains: int,
        n_steps: int,
        dt: float,
        burn_in: int,
        thin: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Euler–Maruyama 模拟白化坐标 Langevin，返回 z 样本 (n_chains, m, d)。

        初值从不变测度精确采样（burn_in 仍保留以吸收离散化瞬态）。稳定性
        要求 dt < 2/max(w)，越限报错。
        """
        if dt >= 2.0 / float(self.well_w.max()):
            raise ValueError("dt 超出 Euler–Maruyama 稳定域 dt < 2/max(w)")
        d = self.latent_dim
        std0 = torch.sqrt(self.temperature / self.well_w)
        u = self.well_u0 + std0 * torch.randn((n_chains, d), generator=generator, dtype=self.s.dtype)
        noise_scale = math.sqrt(2 * self.temperature * dt)
        kept = []
        for step in range(n_steps):
            drift = -self.well_w * (u - self.well_u0)
            u = u + drift * dt + noise_scale * torch.randn(
                (n_chains, d), generator=generator, dtype=self.s.dtype
            )
            if step >= burn_in and (step - burn_in) % thin == 0:
                kept.append(u.clone())
        u_samples = torch.stack(kept, dim=1)  # (n_chains, m, d)
        flat = u_samples.reshape(-1, d)
        z_parts = [self.z_of_u(flat[i : i + 8192]) for i in range(0, flat.shape[0], 8192)]
        return torch.cat(z_parts).reshape(u_samples.shape)

    def exact_sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """不变测度的精确样本（诊断用，区分离散化偏差与估计器误差）。"""
        std0 = torch.sqrt(self.temperature / self.well_w)
        u = self.well_u0 + std0 * torch.randn(
            (n, self.latent_dim), generator=generator, dtype=self.s.dtype
        )
        return self.z_of_u(u)

    # ---- 真值参照 ----

    def self_information_true(self, z: torch.Tensor) -> torch.Tensor:
        """Î 闭式真值：−log p_z + (1/2)log det g = W(u(z))/T + log Z_u。

        log Z_u = Σ_i (1/2)·log(2πT/w_i) 为 u 空间高斯配分函数；推导见
        模块 docstring 第 4 条。
        """
        log_z_u = 0.5 * torch.log(2 * math.pi * self.temperature / self.well_w).sum()
        return self.potential_dyn(z) / self.temperature + log_z_u


def system_to_params(system: CalLangevinSystem) -> dict:
    """全部参数导出为纯 JSON 结构（审计与复现用）。"""
    return {
        "s": system.s.tolist(),
        "warp_a": system.warp_a,
        "warp_tau": system.warp_tau,
        "rot": system.rot.tolist(),
        "shift": system.shift.tolist(),
        "basis": system.basis.tolist(),
        "u_hat": system.u_hat.tolist(),
        "sigma_dec": system.sigma_dec,
        "sigma_x": system.sigma_x,
        "well_w": system.well_w.tolist(),
        "well_u0": system.well_u0.tolist(),
        "temperature": system.temperature,
        "radial_eta": system.radial_eta,
        "radial_r0": system.radial_r0,
        "radial_tau0": system.radial_tau0,
    }


def system_from_params(params: dict) -> CalLangevinSystem:
    t = lambda key: torch.tensor(params[key], dtype=torch.float64)  # noqa: E731
    return CalLangevinSystem(
        s=t("s"), warp_a=params["warp_a"], warp_tau=params["warp_tau"],
        rot=t("rot"), shift=t("shift"), basis=t("basis"), u_hat=t("u_hat"),
        sigma_dec=params["sigma_dec"], sigma_x=params["sigma_x"],
        well_w=t("well_w"), well_u0=t("well_u0"), temperature=params["temperature"],
        radial_eta=params.get("radial_eta", 0.0),
        radial_r0=params.get("radial_r0", 1.0),
        radial_tau0=params.get("radial_tau0", 1.0),
    )


def build_matched_system(
    target_eig_median_by_rank: list[float],
    target_logdet_std: float,
    obs_dim: int,
    sigma_dec: float,
    sigma_x: float,
    well_w: list[float],
    temperature: float,
    generator: torch.Generator,
    warp_tau: float = 2.0,
    warp_a: float = 0.3,
    n_fit_rounds: int = 6,
    n_fit_points: int = 512,
) -> tuple[CalLangevinSystem, dict]:
    """谱匹配拟合：目标为试点实测的逐秩特征值中位数与 logdet 标准差。

    旋钮分工：s 逐轮乘性修正命中逐秩中位数（条件数由秩谱自动带出）；
    η（径向共同因子）乘性迭代命中 logdet 标准差，对条件数近中性；
    a 固定小值（默认 0.3）只提供逐轴局部纹理——试点显示真实系统的
    logdet 大波动与温和条件数波动并存，对应共同缩放为主的调制结构。
    r₀、τ₀ 每轮取当前逐轴层像的范数中位数与标准差。匹配是构造步骤，
    不设通过阈值，仪器门在 CAL-P4 全链恢复处判。
    """
    d = len(target_eig_median_by_rank)
    dtype = torch.float64
    target_eig = torch.tensor(sorted(target_eig_median_by_rank), dtype=dtype)  # 升序
    s = sigma_dec * target_eig.sqrt()

    raw = torch.randn((d, d), generator=generator, dtype=dtype)
    rot, _ = torch.linalg.qr(raw)
    shift = torch.randn((d,), generator=generator, dtype=dtype)

    raw_b = torch.randn((obs_dim, d + 1), generator=generator, dtype=dtype)
    qb, _ = torch.linalg.qr(raw_b)
    basis, u_hat = qb[:, :d], qb[:, d]

    well_w_t = torch.tensor(well_w, dtype=dtype)
    well_u0 = torch.zeros(d, dtype=dtype)

    def make(s_vec: torch.Tensor, eta: float, r0: float, tau0: float) -> CalLangevinSystem:
        return CalLangevinSystem(
            s=s_vec, warp_a=warp_a, warp_tau=warp_tau, rot=rot, shift=shift,
            basis=basis, u_hat=u_hat, sigma_dec=sigma_dec, sigma_x=sigma_x,
            well_w=well_w_t, well_u0=well_u0, temperature=temperature,
            radial_eta=eta, radial_r0=r0, radial_tau0=tau0,
        )

    def eval_spectra(system: CalLangevinSystem):
        z = system.exact_sample(n_fit_points, generator)
        eig = torch.linalg.eigvalsh(system.metric_true(z))  # (n, d) 升序
        logdet = system.logdet_metric_true(z)
        r = system._inner(z).norm(dim=-1)
        return eig, logdet, r

    # r₀/τ₀ 一次性冻结：η=0 时 ‖φ_ax(z)‖ = ‖y‖ = σ_dec‖u‖，其分布只由
    # u 的不变测度决定。逐轮重估会与径向压缩形成正反馈（τ₀ 坍缩），冻结
    # 切断该回路。
    std0 = torch.sqrt(torch.tensor(temperature, dtype=dtype) / well_w_t)
    u_probe = well_u0 + std0 * torch.randn((n_fit_points, d), generator=generator, dtype=dtype)
    r_y = (sigma_dec * u_probe).norm(dim=-1)
    r0 = float(r_y.median())
    tau0 = max(float(torch.log(r_y).std()), 1e-6)
    eta_cap = 3.0 * tau0  # 径向应变 ρ'/f ≤ 1 + η/τ₀ ≤ 4，控制条件数污染

    eta = min(0.3, eta_cap)
    history = []
    for round_idx in range(n_fit_rounds):
        system = make(s, eta, r0, tau0)
        eig, logdet, r = eval_spectra(system)
        med = torch.quantile(eig, 0.5, dim=0)
        achieved_std = float(logdet.std())
        history.append(
            {
                "round": round_idx,
                "eta": eta,
                "achieved_logdet_std": achieved_std,
                "achieved_eig_median_sorted": med.tolist(),
            }
        )
        if achieved_std > 0:
            eta = min(eta * target_logdet_std / achieved_std, eta_cap)
        s = s * (target_eig / med).sqrt()

    system = make(s, eta, r0, tau0)
    eig, logdet, _ = eval_spectra(system)
    report = {
        "target_eig_median_by_rank": target_eig.tolist(),
        "achieved_eig_median_by_rank": torch.quantile(eig, 0.5, dim=0).tolist(),
        "target_logdet_std": target_logdet_std,
        "achieved_logdet_std": float(logdet.std()),
        "achieved_log10_cond_median": float(
            torch.quantile(torch.log10(eig[:, -1] / eig[:, 0]), 0.5)
        ),
        "warp_a": warp_a,
        "warp_tau": warp_tau,
        "radial_eta": eta,
        "radial_eta_cap": eta_cap,
        "radial_r0": r0,
        "radial_tau0": tau0,
        "s": s.tolist(),
        "fit_history": history,
        "matching_gaps": (
            "logdet_std 达成值低于目标时为构造族上限所致（见模块 docstring"
            "径向层说明）；缺口方向：合成系统的位置变率弱于真实系统，"
            "由位置变率引起的估计误差在校准中可能被低估，任务四仪器门"
            "解释时须计入。"
        ),
    }
    return system, report
