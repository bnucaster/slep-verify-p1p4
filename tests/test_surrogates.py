"""P2 主代理生成器的健全性用例。

断言为构造性质：端点钉合（1e-9，纯代数）；速度分布匹配（桥修正前的
速率是观测速率的置换，桥修正后总长比落在 [0.5, 2] 的宽松质检带——
定值回归界）；平滑窗自动匹配返回正整数；同种子确定性。
"""
import torch

from slep import guard
from slep.protocols.surrogates import direction_autocorr_time, smooth_random_surrogate


def _traj(gen: torch.Generator, n: int = 120, d: int = 6) -> torch.Tensor:
    # 平滑随机轨迹：低通高斯增量累积
    raw = torch.randn((n + 10, d), generator=gen, dtype=torch.float64)
    inc = torch.stack([raw[i : i + 8].mean(0) for i in range(n)])
    return 0.3 * inc.cumsum(dim=0)


def _gen(offset: int = 0) -> torch.Generator:
    seed = guard.family_seeds("calibration", purpose="test-surrogates")[0]
    g = torch.Generator()
    g.manual_seed(seed + offset)
    return g


def test_endpoints_pinned_and_length_sane():
    traj = _traj(_gen())
    sur, info = smooth_random_surrogate(traj, _gen(1))
    assert torch.allclose(sur[0], traj[0], atol=1e-9)
    assert torch.allclose(sur[-1], traj[-1], atol=1e-9)
    assert 0.5 < info["length_ratio"] < 2.0
    assert info["smooth_window"] >= 1


def test_deterministic_given_generator():
    traj = _traj(_gen())
    s1, _ = smooth_random_surrogate(traj, _gen(2))
    s2, _ = smooth_random_surrogate(traj, _gen(2))
    assert torch.equal(s1, s2)


def test_autocorr_time_positive_and_bounded():
    traj = _traj(_gen())
    lag = direction_autocorr_time(traj)
    assert 1 <= lag <= 50


def test_metric_surrogate_matches_metric_speeds_flat():
    from slep.protocols.surrogates import smooth_random_surrogate_metric

    traj = _traj(_gen())

    def metric(z):
        return torch.eye(z.shape[-1], dtype=torch.float64).expand(z.shape[0], z.shape[-1], z.shape[-1])

    sur, info = smooth_random_surrogate_metric(traj, metric, _gen(3))
    assert torch.allclose(sur[0], traj[0], atol=1e-9)
    assert torch.allclose(sur[-1], traj[-1], atol=1e-9)
    # 平坦度量下桥修正前逐步度量速率是观测速率的置换；桥修正后总长比
    # 落在宽松质检带（定值回归界）
    assert 0.5 < info["metric_length_ratio"] < 2.0


def test_metric_surrogate_speeds_exact_before_bridge_anisotropic():
    from slep.protocols.surrogates import smooth_random_surrogate_metric

    traj = _traj(_gen())
    scale = torch.tensor([25.0, 1.0, 1.0, 1.0, 1.0, 0.04], dtype=torch.float64)

    def metric(z):
        return torch.diag(scale).expand(z.shape[0], 6, 6)

    # 构造性质自检：关闭桥修正时，代理逐步度量速率恰为观测度量速率的
    # 置换（排序后逐项相等）。
    sur, _ = smooth_random_surrogate_metric(traj, metric, _gen(4), pin_endpoint=False)
    d_obs = traj[1:] - traj[:-1]
    d_sur = sur[1:] - sur[:-1]
    s_obs = torch.einsum("ti,tij,tj->t", d_obs, metric(traj[:-1]), d_obs).sqrt()
    s_sur = torch.einsum("ti,tij,tj->t", d_sur, metric(sur[:-1]), d_sur).sqrt()
    assert torch.allclose(s_obs.sort().values, s_sur.sort().values, rtol=1e-9)
