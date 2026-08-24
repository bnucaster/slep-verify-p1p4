"""运行落盘工具。

CLAUDE.md 工程约定：每次运行在 results/<阶段>/<实验>/<run_id>/ 落盘
配置快照、git commit hash、日志与产物。本模块提供目录创建与元数据
写入；产物由调用方写入返回的目录。
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"


def git_state() -> dict[str, object]:
    """当前 git 提交哈希与工作区是否有未提交改动；取不到时记录原因。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def create_campaign_dir(stage: str, experiment: str, name: str, config: dict) -> Path:
    """可续跑的战役目录：results/<stage>/<experiment>/<name>/。

    多 run 训练战役须断点续跑（工程约定：长任务拆小步、每步落盘），
    时间戳目录做不到跨调用复用。首次创建写 config.json；再次进入时校验
    config 与既有快照一致（不一致报错，防同名目录混入不同配置的产物）。
    每次调用都追加一份 meta_<时间戳>.json（git 状态、环境）。
    """
    if stage not in {"calibration", "description", "confirmation"}:
        raise ValueError(f"未知阶段 {stage!r}")
    campaign = RESULTS_ROOT / stage / experiment / name
    campaign.mkdir(parents=True, exist_ok=True)
    cfg_path = campaign / "config.json"
    snapshot = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
    if cfg_path.exists():
        if cfg_path.read_text(encoding="utf-8") != snapshot:
            raise RuntimeError(
                f"战役目录 {campaign} 已存在且配置不一致；改配置须换战役名"
            )
    else:
        cfg_path.write_text(snapshot, encoding="utf-8")

    meta = {
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        meta["torch"] = None
    (campaign / f"meta_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return campaign


def create_run_dir(stage: str, experiment: str, config: dict) -> Path:
    """建 results/<stage>/<experiment>/<run_id>/，写 config.json 与 meta.json。

    run_id 为本地时间戳，同秒冲突时追加序号。config 应包含复现该次运行
    所需的全部参数（种子、系统参数、扫描网格等）。
    """
    if stage not in {"calibration", "description", "confirmation"}:
        raise ValueError(f"未知阶段 {stage!r}")
    base = RESULTS_ROOT / stage / experiment
    base.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / run_id
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{run_id}-{suffix}"
    run_dir.mkdir()

    meta = {
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        meta["torch"] = None

    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir
