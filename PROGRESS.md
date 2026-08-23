# PROGRESS

更新纪律：每完成一个任务由执行者更新本文件；格式、权重与健康度判据见 CLAUDE.md 规则一（任务后进度回报）。

## 分阶段进度

| 阶段 | 权重 | 出口条件 | 进度 | 状态 |
|---|---|---|---|---|
| M1 校准 | 15% | 仪器门：全链恢复已知真值，阈值公式化 | 25% | 进行中 |
| M2 描述 | 20% | 预承诺页三分支之一触发 | 0% | 未开始 |
| 冻结与预注册 | 5% | 协议 v1.1 外部时间戳落地，判定代码冻结 | 0% | 未开始 |
| EXP-P4 仿射律 | 15% | 判定写入 metrics.json | 0% | 未开始 |
| EXP-P1 能量下降 | 10% | 判定写入 metrics.json | 0% | 未开始 |
| EXP-P3 近测地 | 12% | 判定写入 metrics.json | 0% | 未开始 |
| EXP-P2 低作用量 | 13% | 判定写入 metrics.json | 0% | 未开始 |
| M6 写作 | 10% | 论文成稿，工具箱开源 | 0% | 未开始 |

M1 进度依据：初始任务队列（docs/initial_prompt.md）把 M1 划为任务一至四，任务一已完成，按任务数等权计 1/4。任务一产物：

- 包骨架可安装（pyproject.toml，.venv Python 3.13.1 下 `pip install -e ".[dev]"` 通过）；
- 守卫用例含反向用例（tests/test_guard.py，6 过）；
- 度量估计 ĝ 双路与线性闭式对拍（src/slep/estimators/metric.py、tests/test_metric.py，5 过）；
- 势估计 V̂ 双操作化与已知势恢复（src/slep/estimators/potential.py、src/slep/systems/selfcheck.py、tests/test_potential.py，5 过）；
- k 敏感性曲线（results/calibration/potential_k_sensitivity/20260824-003733/，代码提交 2e6bff0）。

## 综合进度

约 3.8%（= 15% × 25%，其余阶段为 0）。

## 健康度

绿。判据核对：无协议违规（评估族种子未进入任何路径，守卫测试只改 tmp_path 副本；judge 与判定文件不存在手改；测试容差全部闭式推导并在测试文件注明依据，无手拍阈值）；无仪器门失败；进度与预算无落后。

## 信心

中。指"M1 能满足出口条件（全链恢复已知真值）"。依据——事实：已实现的 ĝ、V̂ 在自检系统上恢复真值，容差为推导值（tests/test_metric.py、tests/test_potential.py 共 16 用例通过）；推断：管线与落盘基建已就位，后续估计器可复用同一套自检模式。不确定部分：熵/密度估计（维数灾）、OM 作用量、测地求解与唯一性认证尚未开工，其自检能否通过、几何匹配合成系统上是否仍恢复，目前无证据。

## 事件日志

- 2026-08-23 启动包生成，仓库初始化前状态。
- 2026-08-24 仓库 git 初始化（工程约定要求产物记录提交哈希）。.venv（Python 3.13.1，torch 2.13.0+cpu）建立，`pip install -e ".[dev]"` 通过。
- 2026-08-24 任务一（骨架 + ĝ + V̂ + k 敏感性）完成：16 测试全过；k 敏感性产物落盘 results/calibration/potential_k_sensitivity/20260824-003733/。提交 239978f…59fa3a5。
