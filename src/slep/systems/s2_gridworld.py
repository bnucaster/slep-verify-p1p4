"""S2 环境：程序生成迷宫 + 局部视野观测（docs/plan_v2.md 第 2 节）。

迷宫用递归回溯（深度优先）在 (2n+1)×(2n+1) 栅格上生成，n 为单元数；
True 为墙。智能体动作为四方向绝对移动，撞墙原地不动。观测为以智能体
为中心的 view×view 墙体占据窗口（边界外记墙），展平为 view² 维 0/1
向量。试点数据采集用均匀随机策略；规划段（P3 用的 CEM/MPC）属后续
任务。
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


class GridWorld:
    """迷宫内的四方向移动智能体，观测为局部墙体窗口。"""

    def __init__(self, maze: np.ndarray, position: tuple[int, int], view: int = 5):
        if view % 2 != 1:
            raise ValueError("view 须为奇数")
        if maze[position]:
            raise ValueError("初始位置在墙内")
        self.maze = maze
        self.position = position
        self.view = view

    def observe(self) -> np.ndarray:
        half = self.view // 2
        r, c = self.position
        padded = np.pad(self.maze, half, constant_values=True)
        window = padded[r : r + self.view, c : c + self.view]
        return window.astype(np.float32).reshape(-1)

    def step(self, action: int) -> np.ndarray:
        dr, dc = ACTIONS[action]
        nr, nc = self.position[0] + dr, self.position[1] + dc
        if not self.maze[nr, nc]:
            self.position = (nr, nc)
        return self.observe()


def random_free_cell(maze: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    free = np.argwhere(~maze)
    r, c = free[rng.integers(len(free))]
    return int(r), int(c)


def collect_rollouts(
    n_episodes: int,
    episode_len: int,
    n_cells: int,
    view: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """随机策略轨迹：每回合新迷宫。返回 obs (E, T+1, view²) 与
    动作 one-hot (E, T, 4)。"""
    obs = np.empty((n_episodes, episode_len + 1, view * view), dtype=np.float32)
    act = np.zeros((n_episodes, episode_len, 4), dtype=np.float32)
    for e in range(n_episodes):
        maze = generate_maze(n_cells, rng)
        env = GridWorld(maze, random_free_cell(maze, rng), view)
        obs[e, 0] = env.observe()
        actions = rng.integers(0, 4, size=episode_len)
        for t, a in enumerate(actions):
            obs[e, t + 1] = env.step(int(a))
            act[e, t, a] = 1.0
    return obs, act
