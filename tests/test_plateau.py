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


def test_smoothing_index_mapping_on_constant_curve():
    # 常数曲线：W=3 时首个可判窗 j=2，平台点=(2+1)*3=9；
    # 平滑 s=5 末端对齐使原始序号整体 +（s−1）。
    u = torch.full((60,), 0.5, dtype=torch.float64)
    assert detect_plateau(u, window=3)["plateau_index"] == 9
    out = detect_plateau(u, window=3, smoothing=5)
    assert out["plateau_index"] == 9 + 4
    assert out["smoothing"] == 5


def test_wobbly_saturation_needs_smoothing():
    # 构造对应 dev_v3 实测量级的检查点波动（σ_rel≈1.5%>θ=1%，
    # 见 results/description/s2_u_curve/dev_v3/）：真饱和曲线叠加
    # 波动后，未平滑 W=3 判据应失效，平滑 5 应在饱和段内检出。
    seed = guard.family_seeds("calibration", purpose="test-plateau")[0]
    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    t = _t(80)
    base = 0.32 * (1 - torch.exp(-t / 10.0))
    u = base + 0.0048 * torch.randn(80, generator=gen, dtype=torch.float64)
    assert detect_plateau(u, window=3)["plateau_index"] is None
    out = detect_plateau(u, window=3, smoothing=5)
    assert out["plateau_index"] is not None
    assert out["plateau_index"] >= 30  # 饱和段（膝点约 3×时间常数）之后
