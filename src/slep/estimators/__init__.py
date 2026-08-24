"""估计器层（docs/plan_v2.md 第 4 节）。

已实现：
- metric：潜点处 Fisher 度量估计 ĝ，Jacobian 拉回闭式与 MC 得分协方差双路。
- potential：语义势估计 V̂，潜空间 k 近邻主操作化与编码器后验加权第二操作化。
- density：kNN 对数密度（含标准化坐标口径）与体积校正自信息 Î。
- flow：自研最小 RealNVP 流密度（密度双路之二）。
- entropy：熵估计 Ŝ 双路（kNN 留一 / flow）。

待实现（任务三余项）：OM 作用量、测地求解与唯一性认证、经验漂移分解、
支撑度诊断。新估计器先过自检单元测试再接入实验（CLAUDE.md 工程约定）。
"""
