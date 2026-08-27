"""冻结清单生成（协议 v1.1 第 8 节冻结程序第 2 步）。

对冻结范围内文件计算 SHA-256 写入 docs/freeze_manifest.json：协议正文、
机读阈值表、judge、全部估计器与协议源码、seeds.yaml、守卫。清单生成
要求阈值表无占位、judge 测试通过（调用方负责先跑测试）。

用法：.venv/Scripts/python.exe scripts/freeze_manifest.py [--v12]

--v12：生成 v1.2 清单（docs/freeze_manifest_v12.json）——v1.1 全集 +
v1.2 修订件（amendment、v1.2 阈值表、judge_v12、S2'/S3 系统实现）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

from slep.utils.runs import REPO_ROOT

FREEZE_SET = [
    "docs/protocol_v1.1_draft.md",
    "docs/protocol_v1.1_thresholds.json",
    "docs/manipulation_signature.md",
    "configs/seeds.yaml",
    "src/slep/guard.py",
    "src/slep/protocols/judge.py",
    "src/slep/protocols/affine.py",
    "src/slep/protocols/plateau.py",
    "src/slep/protocols/surrogates.py",
    "src/slep/estimators/metric.py",
    "src/slep/estimators/potential.py",
    "src/slep/estimators/density.py",
    "src/slep/estimators/flow.py",
    "src/slep/estimators/entropy.py",
    "src/slep/estimators/om_action.py",
    "src/slep/estimators/geodesic.py",
    "src/slep/estimators/drift.py",
]


FREEZE_SET_V12_EXTRA = [
    "docs/protocol_v1.2_amendment.md",
    "docs/protocol_v1.2_thresholds.json",
    "src/slep/protocols/judge_v12.py",
    "src/slep/systems/s2_world_model.py",
    "src/slep/systems/s3_transformer.py",
]


def main() -> None:
    v12 = "--v12" in sys.argv
    th_file = ("docs/protocol_v1.2_thresholds.json" if v12
               else "docs/protocol_v1.1_thresholds.json")
    th = json.loads((REPO_ROOT / th_file).read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        raise SystemExit(f"阈值表仍有占位 {th['_meta']['pending']}，不可生成冻结清单")
    files = FREEZE_SET + (FREEZE_SET_V12_EXTRA if v12 else [])
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "protocol_version": "1.2" if v12 else "1.1", "files": {}}
    for rel in files:
        data = (REPO_ROOT / rel).read_bytes()
        manifest["files"][rel] = hashlib.sha256(data).hexdigest()
    out = REPO_ROOT / "docs" / ("freeze_manifest_v12.json" if v12 else "freeze_manifest.json")
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"冻结清单已写入 {out}（{len(files)} 个文件）")


if __name__ == "__main__":
    main()
