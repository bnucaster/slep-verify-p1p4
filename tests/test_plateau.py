"""平台检测器的健全性用例（协议件正确性检查；工作特性定标见 CAL-P1）。

断言均为结构性 / 方向性：饱和曲线检出平台且位于膝点之后、效用接近
饱和值；持续增长不检出；持续下降（双侧口径）不检出；平台后窗口公式
正确。种子经 guard.family_seeds 取校准族。
"""
import torch

from slep import guard
from slep.protocols.plateau import detect_plateau


def _t(n: int = 400) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float64)


def test_saturating_curve_detected_after_knee():
    t = _t()
    u = 0.8 * (1 - torch.exp(-t / 60.0))
    out = detect_plateau(u, window=20)
    idx = out["plateau_index"]
    assert idx is not None
    assert u[idx] > 0.95 * 0.8  # 平台点处效用已接近饱和
    assert out["post_window"] == (idx, min(2 * idx, 400))


def test_linear_growth_not_detected():
    t = _t()
    u = 0.1 + 0.5 * t / 400.0
    assert detect_plateau(u, window=20)["plateau_index"] is None


def test_steady_decline_not_detected_two_sided():
    t = _t()
    u = 1.0 - 0.002 * t  # 每 20 步窗下降 4%，双侧口径不算平台
    assert detect_plateau(u, window=20)["plateau_index"] is None


def test_noisy_flat_curve_detected_early():
    seed = guard.family_seeds("calibration", purpose="test-plateau")[0]
    gen = torch.Generator()
    gen.manual_seed(seed)
    u = 0.7 + 0.002 * torch.randn(400, generator=gen, dtype=torch.float64)
    out = detect_plateau(u, window=20)
    assert out["plateau_index"] is not None
    assert out["plateau_index"] <= 5 * 20  # 前几窗内即应判平
