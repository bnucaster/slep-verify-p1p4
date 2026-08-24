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
