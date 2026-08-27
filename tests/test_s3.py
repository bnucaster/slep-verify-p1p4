"""S3 Transformer 世界模型自检（协议 v1.2 附录，选项 E）。

结构性断言：前向形状与 S2 接口对齐；教师强制损失冒烟下降；因果性
——后缀 (obs, act) 扰动不得改变前缀隐状态（Ψ 的因果结构是推理轨迹
语义的前提）。种子经 guard 取校准族。
"""
import torch

from slep import guard
from slep.systems.s3_transformer import S3TransformerWM


def _data(n_ep=4, t=12, obs_dim=50, gen=None):
    obs = torch.rand(n_ep, t + 1, obs_dim, generator=gen)
    act = torch.nn.functional.one_hot(
        torch.randint(0, 4, (n_ep, t), generator=gen), num_classes=4).float()
    return obs, act


def test_shapes_align_with_s2_interface():
    seed = guard.family_seeds("calibration", purpose="test-s3")[0]
    gen = torch.Generator()
    gen.manual_seed(seed)
    model = S3TransformerWM()
    obs, act = _data(gen=gen)
    hs = model.hidden_trajectory(obs, act)
    assert hs.shape == (4, 12, model.hidden_dim)
    assert model.decoder_mean(hs).shape == (4, 12, 50)
    assert model.obs_var.shape == (50,)
    out = model.rollout_loss(obs, act)
    assert out["total"].dim() == 0


def test_loss_decreases_smoke():
    # 可学习结构：逐回合常值观测（看过 o_0 即可预测全部后续），损失应大幅下降
    seed = guard.family_seeds("calibration", purpose="test-s3")[0]
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    model = S3TransformerWM(d_model=16, n_layers=1, n_heads=2, ff_dim=32)
    base = torch.rand(16, 1, 50, generator=gen)
    obs = base.expand(16, 11, 50).contiguous()
    act = torch.nn.functional.one_hot(
        torch.randint(0, 4, (16, 10), generator=gen), num_classes=4).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        loss = model.rollout_loss(obs, act)["total"]
        if first is None:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.3 * first


def test_causality_suffix_perturbation():
    seed = guard.family_seeds("calibration", purpose="test-s3")[0]
    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    model = S3TransformerWM()
    model.eval()
    obs, act = _data(n_ep=2, t=12, gen=gen)
    with torch.no_grad():
        h0 = model.hidden_trajectory(obs, act)
        obs2, act2 = obs.clone(), act.clone()
        obs2[:, 8:] = torch.rand(obs2[:, 8:].shape, generator=gen)
        act2[:, 8:] = torch.nn.functional.one_hot(
            torch.randint(0, 4, act2[:, 8:].shape[:2], generator=gen), 4).float()
        h1 = model.hidden_trajectory(obs2, act2)
    assert torch.allclose(h0[:, :8], h1[:, :8], atol=1e-6)
    assert not torch.allclose(h0[:, 8:], h1[:, 8:], atol=1e-3)
