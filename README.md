# SLEP P1–P4 实证检验

检验母论文《The Semantic Least-Energy Principle》4.7 节（预测协议）的四条预注册预测：P1（能量随学习下降、拟合后压缩）、P2（推理轨迹集中于低作用量路径）、P3（低梯度区审议轨迹近测地）、P4（自信息对势的仿射律与温度可辨识）。

## 权威顺序

1. docs/plan_v2.md（综合实验方案 v2）：协议权威，一切实验设计与判据以它为准。
2. CLAUDE.md：执行纪律，Claude Code 每次会话自动加载。
3. 两者冲突时停下报告用户，不自行裁决。

## 启动步骤

1. 解压本包到工作目录，`git init` 并首次提交（协议审计依赖提交历史）。
2. 将母论文 PDF 放入 docs/（文件名建议 slep_paper.pdf）。
3. 安装：`pip install -e ".[dev]"`，跑 `pytest tests/` 确认评估族隔离守卫生效。
4. 在仓库根目录启动 Claude Code，粘贴 docs/initial_prompt.md 中的首条任务。

## 目录

- src/slep/：估计器（estimators）、测试系统（systems）、实验协议与判定（protocols）、守卫（guard.py）与工具（utils）
- configs/：种子分族与实验配置
- docs/：方案、proposal、预承诺页模板、预注册协议草稿、冻结状态、初始提示词
- results/：calibration / description / confirmation 三阶段产物；确证判定落在 confirmation/metrics.json
- PROGRESS.md：进度、健康度与信心，每任务后更新
