# SLEP P1–P4 预注册实证检验

对《The Semantic Least-Energy Principle》（母论文）第 4.7 节四条预测的
预注册检验：P1（能量随学习下降、拟合后压缩）、P2（推理轨迹集中于低作用
量路径）、P3（低梯度区审议轨迹近测地）、P4（自信息对势的仿射律与温度
可辨识）。判定三值（通过 / 失败 / 未构成检验），阈值全部溯源校准产物，
判定代码冻结并取得公开提交时间戳。

结果速览（详见 docs/paper/draft_v1.md 与 results/confirmation/scoreboard_v12.json）：
P2 × S2P 通过（五种子全票）；P4、P3、P1 各格未构成检验，理由逐格在案。

## 权威顺序

1. docs/plan_v2.md（综合实验方案 v2）：协议权威，一切实验设计与判据以它为准。
2. CLAUDE.md：执行纪律。
3. 两者冲突时停下报告用户，不自行裁决。

## 目录

- `src/slep/estimators/` —— 估计器工具箱（可独立复用）：Fisher 度量拉回、
  kNN 势、kNN/归一化流密度与熵、体积校正自信息、OM 作用量、测地求解与
  唯一性认证、经验漂移分解、支撑度诊断。全部带合成真值自检
  （`tests/`）。
- `src/slep/systems/` —— 测试系统：S1（β-VAE/dSprites）、S2 与 S2P
  （GRU 世界模型/网格世界，多步解码头变体）、S3（小 Transformer 序列
  世界模型）、几何匹配合成 Langevin 校准系统。
- `src/slep/protocols/` —— 入域门、平台检测器、仿射统计、代理构造、
  判定器（judge v1.1 与 v1.2）。
- `src/slep/guard.py` —— 种子分族守卫（校准 {99,100}、开发 0–4、评估
  5–9），冻结置位前机械阻断评估族。
- `docs/` —— 协议权威 `plan_v2.md`；预注册协议 v1.1
  （`protocol_v1.1_draft.md` + 阈值表）与披露性修订 v1.2
  （`protocol_v1.2_amendment.md` + 阈值表）；冻结清单与状态。
- `configs/` —— 全部实验配置；`scripts/` —— 校准、描述、确证三阶段
  运行脚本；`results/` —— 三阶段产物（判定文件 metrics.json 只由冻结
  judge 写入）。
- `PROGRESS.md` —— 全程进度板与事件日志（含每次修正的披露记录）。

## 复现

```
pip install -e ".[dev]"
pytest -q          # 估计器自检与判定器分支测试
```

确证阶段的完整重放顺序见 PROGRESS.md 事件日志；判定重放：

```
python scripts/assemble_judge_v12.py
```

时间戳锚点：v1.1 = f7f7910，v1.2 = 27df5af（`docs/freeze_status.json`）。
哈希按 LF 字节计算，审计时用 `git show` 取原始内容或以
`core.autocrlf=false` 克隆。

## 引用

论文草稿：`docs/paper/draft_v1.md`。工具箱作为公共引用锚随本仓库发布；
许可证待定（当前保留所有权利，引用请注明仓库与锚点提交）。
