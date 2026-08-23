"""SLEP P1–P4 实证检验核心包。

子包分工见 CLAUDE.md 目录地图：estimators（估计器层，docs/plan_v2.md 第 4 节）、
systems（测试系统 S1/S2/S3 与合成校准系统）、protocols（入域门、平台检测器、
四实验协议、聚合与 judge）、utils（落盘与配置工具）。

guard.py 为评估族隔离守卫（CLAUDE.md 硬规则 1）：任何解析随机种子的代码路径
都必须调用 guard.assert_seed_allowed，新路径遗漏该调用属协议违规。
"""

__version__ = "0.0.1"
