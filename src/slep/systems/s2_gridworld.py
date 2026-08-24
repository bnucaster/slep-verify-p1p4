"""S2 环境：程序生成迷宫 + 局部视野观测 + 目标通道（docs/plan_v2.md 第 2 节）。

迷宫用递归回溯（深度优先）在 (2n+1)×(2n+1) 栅格上生成，n 为单元数；
True 为墙。智能体动作为四方向绝对移动，撞墙原地不动。观测为两通道
局部窗口（各 view×view，展平相接，共 2·view² 维）：

- 通道 0：墙体占据（边界外记墙）；
- 通道 1：目标指示（目标在窗口内的格子记 1，否则全零）。

目标通道服务导航效用（plan_v2 第 3 节：S2 能力门 = 留出迷宫导航成功率）
与 CEM/MPC 规划段的代价函数；未设目标时通道恒零。试点阶段（任务二 2c）
的单通道观测由本版本取代，试点产物照旧存档。

新奇注入（EXP-P2 用，plan_v2 第 2 节）：已知时刻切换未见迷宫拓扑，
智能体钉到最近空格，目标重置；注入时刻与拓扑种子由调用方记录，
新奇锚独立于能量估计。
"""
from __future__ import annotations

import numpy as np

ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)  # 北南西东


def generate_maze(n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """递归回溯迷宫，返回 (2n+1, 2n+1) 布尔阵，True=墙。"""
    size = 2 * n_cells + 1
    maze = np.ones((size, size), dtype=bool)
    start = (rng.integers(n_cells), rng.integers(n_cells))
    stack = [start]
    visited = {start}
    while stack:
        r, c = stack[-1]
        maze[2 * r + 1, 2 * c + 1] = False
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < n_cells and 0 <= c + dc < n_cells and (r + dr, c + dc) not in visited
        ]
        if not neighbors:
            stack.pop()
            continue
        nr, nc = neighbors[rng.integers(len(neighbors))]
        maze[r + nr + 1, c + nc + 1] = False  # 打通两单元间的墙
        visited.add((nr, nc))
        stack.append((nr, nc))
    return maze


def random_free_cell(maze: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    free = np.argwhere(~maze)
    r, c = free[rng.integers(len(free))]
    return int(r), int(c)


def nearest_free_cell(maze: np.ndarray, position: tuple[int, int]) -> tuple[int, int]:
    free = np.argwhere(~maze)
    d = np.abs(free - np.array(position)).sum(axis=1)
    r, c = free[int(d.argmin())]
    return int(r), int(c)


def bfs_distance(maze: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    """通路步数（四邻域 BFS）；不可达返回 None。"""
    from collections import deque

    if maze[start] or maze[goal]:
        return None
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ACTIONS:
            nxt = (r + int(dr), c + int(dc))
            if not maze[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None


class GridWorld:
    """迷宫内的四方向移动智能体，双通道局部观测（墙体 + 目标指示）。"""

    def __init__(
        self,
        maze: np.ndarray,
        position: tuple[int, int],
        view: int = 5,
        goal: tuple[int, int] | None = None,
    ):
        if view % 2 != 1:
            raise ValueError("view 须为奇数")
        if maze[position]:
            raise ValueError("初始位置在墙内")
        if goal is not None and maze[goal]:
            raise ValueError("目标在墙内")
        self.maze = maze
        self.position = position
        self.view = view
        self.goal = goal

    @property
    def at_goal(self) -> bool:
        return self.goal is not None and self.position == self.goal

    def observe(self) -> np.ndarray:
        half = self.view // 2
        r, c = self.position
        padded = np.pad(self.maze, half, constant_values=True)
        walls = padded[r : r + self.view, c : c + self.view].astype(np.float32)
        goal_ch = np.zeros((self.view, self.view), dtype=np.float32)
        if self.goal is not None:
            gr, gc = self.goal[0] - r + half, self.goal[1] - c + half
            if 0 <= gr < self.view and 0 <= gc < self.view:
                goal_ch[gr, gc] = 1.0
        return np.concatenate([walls.reshape(-1), goal_ch.reshape(-1)])

    def step(self, action: int) -> np.ndarray:
        dr, dc = ACTIONS[action]
        nr, nc = self.position[0] + dr, self.position[1] + dc
        if not self.maze[nr, nc]:
            self.position = (nr, nc)
        return self.observe()


def make_episode_env(n_cells: int, view: int, rng: np.random.Generator) -> GridWorld:
    """新迷宫 + 随机起点 + 随机目标（异于起点）的回合环境。"""
    maze = generate_maze(n_cells, rng)
    start = random_free_cell(maze, rng)
    goal = start
    while goal == start:
        goal = random_free_cell(maze, rng)
    return GridWorld(maze, start, view, goal)


def inject_novelty(env: GridWorld, rng: np.random.Generator) -> dict:
    """新奇注入：切换未见迷宫拓扑，智能体钉最近空格，目标重置。

    返回记录（供新奇锚：注入前后位置与目标），独立于能量估计。
    """
    old_position = env.position
    env.maze = generate_maze((env.maze.shape[0] - 1) // 2, rng)
    env.position = nearest_free_cell(env.maze, old_position)
    goal = env.position
    while goal == env.position:
        goal = random_free_cell(env.maze, rng)
    env.goal = goal
    return {"old_position": old_position, "new_position": env.position, "new_goal": env.goal}


def collect_rollouts(
    n_episodes: int,
    episode_len: int,
    n_cells: int,
    view: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """随机策略轨迹：每回合新迷宫 + 随机目标。返回 obs (E, T+1, 2·view²)
    与动作 one-hot (E, T, 4)。到达目标不终止（世界模型训练数据）。"""
    obs_dim = 2 * view * view
    obs = np.empty((n_episodes, episode_len + 1, obs_dim), dtype=np.float32)
    act = np.zeros((n_episodes, episode_len, 4), dtype=np.float32)
    for e in range(n_episodes):
        env = make_episode_env(n_cells, view, rng)
        obs[e, 0] = env.observe()
        actions = rng.integers(0, 4, size=episode_len)
        for t, a in enumerate(actions):
            obs[e, t + 1] = env.step(int(a))
            act[e, t, a] = 1.0
    return obs, act
