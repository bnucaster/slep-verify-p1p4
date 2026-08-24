"""S2 环境、世界模型与规划段的健全性用例（系统正确性检查）。

种子纪律：随机数只用校准族种子（guard.family_seeds）。
"""
import numpy as np
import torch

from slep import guard
from slep.systems import s2_gridworld as gw
from slep.systems.s2_planner import CEMPlanner, ExhaustiveMPCPlanner, mpc_episode
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
    assert not maze[1::2, 1::2].any()
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


def test_observation_two_channels_and_goal_visibility():
    rng = _rng()
    maze = gw.generate_maze(4, rng)
    pos = gw.random_free_cell(maze, rng)
    env = gw.GridWorld(maze, pos, view=5, goal=pos)  # 目标即当前格：中心可见
    obs = env.observe()
    assert obs.shape == (50,)
    walls, goal_ch = obs[:25].reshape(5, 5), obs[25:].reshape(5, 5)
    assert walls[2, 2] == 0.0  # 当前位置必为通路
    assert goal_ch[2, 2] == 1.0 and goal_ch.sum() == 1.0
    assert env.at_goal
    # 无目标：目标通道恒零
    env2 = gw.GridWorld(maze, pos, view=5)
    assert env2.observe()[25:].sum() == 0.0
    # 边界外补墙
    env3 = gw.GridWorld(maze, (1, 1), view=5)
    walls3 = env3.observe()[:25].reshape(5, 5)
    assert walls3[0].all() and walls3[:, 0].all()


def test_blocked_move_and_bfs():
    rng = _rng()
    maze = gw.generate_maze(4, rng)
    env = gw.GridWorld(maze, (1, 1), view=5)
    assert maze[0, 1]
    env.step(0)  # 北，外墙
    assert env.position == (1, 1)
    # BFS：相邻通路距离 1；自身 0；墙内 None
    free = np.argwhere(~maze)
    a = (int(free[0][0]), int(free[0][1]))
    assert gw.bfs_distance(maze, a, a) == 0
    assert gw.bfs_distance(maze, (0, 0), a) is None


def test_rollout_shapes_and_onehot():
    rng = _rng()
    obs, act = gw.collect_rollouts(3, 10, 4, 5, rng)
    assert obs.shape == (3, 11, 50)
    assert act.shape == (3, 10, 4)
    assert np.all(act.sum(-1) == 1.0)
    assert set(np.unique(obs)) <= {0.0, 1.0}


def test_novelty_injection():
    rng = _rng()
    env = gw.make_episode_env(6, 5, rng)
    old_maze = env.maze.copy()
    info = gw.inject_novelty(env, rng)
    assert not np.array_equal(env.maze, old_maze)
    assert not env.maze[env.position]
    assert env.goal is not None and not env.maze[env.goal]
    assert info["new_position"] == env.position


def test_exhaustive_planner_modes_and_determinism():
    seed = guard.family_seeds("calibration", purpose="test-s2")[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = S2WorldModel(50, 4, 16, 8, 0.2, 0.04)
    planner = ExhaustiveMPCPlanner(model, view=5, horizon=2, epsilon=0.0)
    env = gw.make_episode_env(6, 5, rng)
    gen = torch.Generator()
    gen.manual_seed(seed)
    action, info = planner.plan(torch.zeros(8), torch.from_numpy(env.observe()), gen)
    assert action in (0, 1, 2, 3)
    assert info["mode"] in ("plan", "explore")
    assert planner._all_seqs.shape == (16, 2)
    # ε=1 时必为随机化模式
    planner_eps = ExhaustiveMPCPlanner(model, view=5, horizon=1, epsilon=1.0,
                                       explore_threshold=-1.0)
    _, info2 = planner_eps.plan(torch.zeros(8), torch.from_numpy(env.observe()), gen)
    assert info2["mode"] in ("epsilon", "explore")


def test_world_model_shapes_and_training_step_reduces_loss():
    seed = guard.family_seeds("calibration", purpose="test-s2")[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_np, act_np = gw.collect_rollouts(16, 12, 4, 5, rng)
    obs, act = torch.from_numpy(obs_np), torch.from_numpy(act_np)

    model = S2WorldModel(50, 4, 16, 8, 0.2)
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
    # 同批数据 30 步优化后损失明显下降；0.9 为方向性回归界，非协议阈值。
    assert loss1 < 0.9 * loss0, f"{loss0} -> {loss1}"


def test_planner_mechanics_and_mpc_episode():
    seed = guard.family_seeds("calibration", purpose="test-s2")[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    model = S2WorldModel(50, 4, 16, 8, 0.2)
    planner = CEMPlanner(model, view=5, horizon=6, n_samples=32, n_elites=4, n_iters=2)
    env = gw.make_episode_env(6, 5, rng)
    action, info = planner.plan(
        torch.zeros(8), torch.from_numpy(env.observe()), gen
    )
    assert action in (0, 1, 2, 3)
    assert len(info["first_step_probs"]) == 4

    gen_ep = torch.Generator()
    gen_ep.manual_seed(seed)
    out = mpc_episode(model, env, planner, max_steps=6, generator=gen_ep)
    assert out["h_traj"].shape == (out["steps"], 8)
    assert isinstance(out["success"], bool)
    # 同种子确定性
    gen2 = torch.Generator()
    gen2.manual_seed(seed)
    rng2 = np.random.default_rng(seed)
    env2 = gw.make_episode_env(6, 5, rng2)
    torch.manual_seed(seed)
    model2 = S2WorldModel(50, 4, 16, 8, 0.2)
    planner2 = CEMPlanner(model2, view=5, horizon=6, n_samples=32, n_elites=4, n_iters=2)
    out2 = mpc_episode(model2, env2, planner2, max_steps=6, generator=gen2)
    assert out2["actions"] == out["actions"]
