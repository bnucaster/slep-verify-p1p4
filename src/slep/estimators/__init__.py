"""估计器层（docs/plan_v2.md 第 4 节）。

已实现：
- metric：潜点处 Fisher 度量估计 ĝ，Jacobian 拉回闭式与 MC 得分协方差双路。
- potential：语义势估计 V̂，潜空间 k 近邻主操作化与编码器后验加权第二操作化。
- density：kNN 对数密度（含标准化坐标口径）与体积校正自信息 Î。
- flow：自研最小 RealNVP 流密度（密度双路之二）。
- entropy：熵估计 Ŝ 双路（kNN 留一 / flow）。
- om_action：OM 作用量 Â_OM 主项（左点 / 中点口径）。
- geodesic：测地求解（度量预条件）、唯一性散布、归一化偏差面积。
- drift：kNN 漂移回归、ĝ 度量下梯度占比（留出口径）、支撑度半径。

docs/plan_v2.md 第 4 节估计器层至此全部有实现与自检；操作化口径的
定夺（V̂ 邻域、lack-of-fit 效应量判据等）属任务四校准阈值工作。
新估计器先过自检单元测试再接入实验（CLAUDE.md 工程约定）。
"""
