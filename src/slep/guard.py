"""评估族隔离守卫。

规则来源：CLAUDE.md 硬规则 1（数据分族）。评估族种子（configs/seeds.yaml
的 evaluation 项）在 docs/freeze_status.json 的 frozen 置位之前，禁止进入
任何训练、加载、读取或分析路径。所有解析种子的入口（训练脚本、数据加载、
结果聚合）都必须调用 assert_seed_allowed；新代码路径遗漏该调用属协议违规。

本模块只依赖标准库与 pyyaml，禁止为绕过检查而改写本文件或其读取的两个
配置文件；改动 seeds.yaml 属协议级变更，须用户书面确认。
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_FILE = _REPO_ROOT / "configs" / "seeds.yaml"
FREEZE_FILE = _REPO_ROOT / "docs" / "freeze_status.json"


class EvalFamilyIsolationError(RuntimeError):
    """冻结前触碰评估族种子。"""


class UnknownSeedError(RuntimeError):
    """种子不属于任何已登记分族；分族必须显式，不允许默认放行。"""


def _load_seed_families(seeds_file: Path = SEEDS_FILE) -> dict[str, list[int]]:
    with open(seeds_file, "r", encoding="utf-8") as f:
        families = yaml.safe_load(f)
    required = {"calibration", "development", "evaluation"}
    missing = required - set(families)
    if missing:
        raise RuntimeError(f"seeds.yaml 缺少分族: {sorted(missing)}")
    return families


def is_frozen(freeze_file: Path = FREEZE_FILE) -> bool:
    with open(freeze_file, "r", encoding="utf-8") as f:
        return bool(json.load(f).get("frozen", False))


def assert_seed_allowed(
    seed: int,
    purpose: str = "",
    seeds_file: Path = SEEDS_FILE,
    freeze_file: Path = FREEZE_FILE,
) -> str:
    """校验种子可用性，返回其所属分族名。

    参数
    ----
    seed: 待使用的随机种子。
    purpose: 调用场景说明（如 "train-s2"、"aggregate-p4"），进入报错信息，
        便于审计定位。
    seeds_file / freeze_file: 仅供单元测试注入临时副本；生产代码不得传入
        仓库外路径以外的任何替代品来规避检查。
    """
    families = _load_seed_families(seeds_file)
    for name, seeds in families.items():
        if seed in seeds:
            if name == "evaluation" and not is_frozen(freeze_file):
                raise EvalFamilyIsolationError(
                    f"种子 {seed} 属评估族，冻结（freeze_status.frozen）未置位，"
                    f"禁止使用。场景: {purpose or '未注明'}。"
                    "评估族只能在协议 v1.1 取得外部时间戳并置位后进入。"
                )
            return name
    raise UnknownSeedError(
        f"种子 {seed} 未登记于 configs/seeds.yaml 的任何分族。"
        f"场景: {purpose or '未注明'}。请先与用户确认分族归属再登记使用。"
    )
