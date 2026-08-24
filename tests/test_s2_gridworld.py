"""S2 环境与世界模型的健全性用例（非估计器自检，属系统正确性检查）。

种子纪律：随机数只用校准族种子（guard.family_seeds）。
"""
import numpy as np
import torch

from slep import guard
from slep.systems import s2_gridworld as gw
from slep.systems.s2_world_model import S2WorldModel


def _rng() -> np.random.Generator:
    seed = guard.family_seeds("calibration", purpose="test-s2")[0]
    return np.random.default_rng(seed)


def test_maze_structure_and_connectivity():
    rng = _rng()
    n = 8
    maze = gw.generate_maze(n, rng)
    assert maze.shape == (2 * n + 1, 2 * n + 1)
    assert maze[0].all() and maze[-1].all() and maze[:, 0].all() and maze[:, -1].all()
    # 单元位置全开
    assert not maze[1::2, 1::2].any()
    # 连通性：从任一单元 BFS 应达全部 n² 个单元
    start = (1, 1)
    seen = {start}
    frontier = [start]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not maze[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                frontier.append((nr, nc))
    cells = {(r, c) for r, c in seen if r % 2 == 1 and c % 2 == 1}
    assert len(cells) == n * n


def test_observation_window_and_blocked_move():
    rng = _rng()
    maze = gw.generate_maze(4, rng)
    pos = gw.random_free_cell(maze, rng)
    env = gw.GridWorld(maze, pos, view=5)
    obs = env.observe()
    assert obs.shape == (25,)
    assert obs[12] == 0.0  # 窗口中心是当前位置，必为通路
    # 边界外补墙：把智能体放到 (1,1)，窗口左上角越界处应为墙
    env2 = gw.GridWorld(maze, (1, 1), view=5)
    obs2 = env2.observe().reshape(5, 5)
    assert obs2[0].all() and obs2[:, 0].all()
    # 撞墙不动：向已知墙方向走
    r, c = env2.position
    assert maze[r - 1, c]  # (0,1) 是外墙
    env2.step(0)  # 北
    assert env2.position == (r, c)


def test_rollout_shapes_and_onehot():
    rng = _rng()
    obs, act = gw.collect_rollouts(3, 10, 4, 5, rng)
    assert obs.shape == (3, 11, 25)
    assert act.shape == (3, 10, 4)
    assert np.all(act.sum(-1) == 1.0)
    assert set(np.unique(obs)) <= {0.0, 1.0}


def test_world_model_shapes_and_training_step_reduces_loss():
    seed = guard.family_seeds("calibration", purpose="test-s2")[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_np, act_np = gw.collect_rollouts(16, 12, 4, 5, rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)

    model = S2WorldModel(25, 4, 16, 8, 0.2)
    hs = model.hidden_trajectory(obs, act)
    assert hs.shape == (16, 12, 8)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss0 = model.rollout_loss(obs, act)["total"].item()
    for _ in range(30):
        out = model.rollout_loss(obs, act)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
    loss1 = model.rollout_loss(obs, act)["total"].item()
    # 同批数据 30 步优化后损失应明显下降；阈值 0.9 只检查方向性，
    # 属实现健全性检查，与协议判定阈值无关。
    assert loss1 < 0.9 * loss0, f"{loss0} -> {loss1}"
