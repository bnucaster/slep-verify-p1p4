"""合成 Langevin 校准系统的自检（docs/plan_v2.md 第 5 节第 1、2 条配套）。

容差依据（推导值）：

- φ 反演回程：Newton 收敛判据 1e-10，回程断言 1e-8（留两个数量级）。
- 闭式度量对拍：metric_true 与 estimators.metric 的 autograd 拉回同为
  float64 精确计算，只差舍入，取 1e-9 相对容差。
- 仿射真值：self_information_true 与 potential_dyn 的线性关系是代数
  恒等式，回归斜率与 1/T 之差只含舍入，取 1e-10。
- 链平稳性（OU 过程）：白化坐标下每维是刚度 w_i 的 OU 过程，步进 dt 时
  自相关时间约 1/(w_i·dt) 步，有效样本量 n_eff ≈ n_kept·thin·w_i·dt
  （保守起见再除 3）。样本均值容差 6·sqrt(Var/n_eff)，方差容差按
  卡方近似 6·Var·sqrt(2/n_eff)。

种子纪律：种子经 guard.family_seeds 取校准族。
"""
import torch

from slep import guard
from slep.estimators import metric
from slep.systems.cal_langevin import CalLangevinSystem, build_matched_system

TARGET_EIG = [0.05, 0.4, 3.0, 20.0, 120.0, 600.0]  # 6 维示例目标（升序）
TARGET_LOGDET_STD = 2.0


def _gen(idx: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-cal-langevin")[idx]
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _system() -> tuple[CalLangevinSystem, dict]:
    gen = _gen(0)
    return build_matched_system(
        target_eig_median_by_rank=TARGET_EIG,
        target_logdet_std=TARGET_LOGDET_STD,
        obs_dim=16,
        sigma_dec=0.2,
        sigma_x=0.05,
        well_w=[0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
        temperature=0.5,
        generator=gen,
    )


def test_phi_inverse_roundtrip():
    system, _ = _system()
    gen = _gen(1)
    z = torch.randn((256, system.latent_dim), generator=gen, dtype=torch.float64)
    z_rec = system.phi_inverse(system.phi(z))
    assert float((z_rec - z).abs().max()) < 1e-8


def test_metric_closed_form_matches_autograd_pullback():
    system, _ = _system()
    gen = _gen(1)
    z = torch.randn((8, system.latent_dim), generator=gen, dtype=torch.float64)
    g_closed = system.metric_true(z)
    for i in range(z.shape[0]):
        g_auto = metric.fisher_pullback_gaussian(
            system.decoder_mean_flat, z[i], system.sigma_dec**2
        )
        assert torch.allclose(g_auto, g_closed[i], rtol=1e-9, atol=1e-12)


def test_affine_identity_exact():
    system, _ = _system()
    z = system.exact_sample(2048, _gen(1))
    v = system.potential_dyn(z)
    i_true = system.self_information_true(z)
    vc = v - v.mean()
    slope = (vc * (i_true - i_true.mean())).sum() / (vc**2).sum()
    assert abs(float(slope) - 1.0 / system.temperature) < 1e-10
    resid = i_true - i_true.mean() - slope * vc
    assert float(resid.abs().max()) < 1e-10


def test_matched_spectra_close_to_target():
    system, report = _system()
    z = system.exact_sample(1024, _gen(1))
    eig = torch.linalg.eigvalsh(system.metric_true(z))
    med = torch.quantile(eig, 0.5, dim=0)
    ratio = med / torch.tensor(TARGET_EIG, dtype=torch.float64)
    # 匹配为构造步骤，此处只作回归保护，界取自拟合器当前能力（fit_history
    # 落盘可查），与协议判定阈值无关：非最小秩在目标 3 倍以内；最小秩
    # 放宽到 [1/6, 3]——排序谱逐样本取轴最小值，强调制下其中位数被顺序
    # 统计压低属预期，方向偏难（更病态），对仪器门保守。
    assert torch.all(ratio[1:] < 3.0) and torch.all(ratio[1:] > 1 / 3.0), ratio
    assert 1 / 6.0 < float(ratio[0]) < 3.0, ratio
    assert 0.5 < report["achieved_logdet_std"] / TARGET_LOGDET_STD < 2.0


def test_chain_stationarity_matches_invariant_measure():
    system, _ = _system()
    gen = _gen(1)
    dt, thin, burn_in, n_steps, n_chains = 0.05, 10, 400, 4400, 8
    z = system.simulate_chains(n_chains, n_steps, dt, burn_in, thin, gen)
    u = system.u_of_z(z.reshape(-1, system.latent_dim))
    n_kept_per_chain = z.shape[1]
    n_total = n_chains * n_kept_per_chain

    var_target = system.temperature / system.well_w
    # OU 自相关时间 ~1/(w·dt) 步；保留样本间隔 thin 步 → 相邻保留样本
    # 相关性以 exp(-w·dt·thin) 计，n_eff 取 n_total·w·dt·thin/3（除 3 保守）。
    n_eff = n_total * system.well_w * dt * thin / 3.0
    mean_tol = 6.0 * (var_target / n_eff).sqrt()
    assert torch.all((u.mean(dim=0) - system.well_u0).abs() <= mean_tol)
    var_tol = 6.0 * var_target * (2.0 / n_eff).sqrt()
    assert torch.all((u.var(dim=0) - var_target).abs() <= var_tol)

    # Euler–Maruyama 的不变方差有 O(dt) 偏差：OU 精确离散不变方差为
    # T/(w·(1−w·dt/2))，dt=0.05、w≤2 时偏差 ≤ 5%，应被上面 var_tol 覆盖
    # （此注释记录偏差来源；若未来收紧 var_tol 须先修正此项）。
