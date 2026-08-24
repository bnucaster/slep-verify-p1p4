"""冻结清单生成（协议 v1.1 第 8 节冻结程序第 2 步）。

对冻结范围内文件计算 SHA-256 写入 docs/freeze_manifest.json：协议正文、
机读阈值表、judge、全部估计器与协议源码、seeds.yaml、守卫。清单生成
要求阈值表无占位、judge 测试通过（调用方负责先跑测试）。

用法：.venv/Scripts/python.exe scripts/freeze_manifest.py
"""
from __future__ import annotations

import hashlib
import json
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


def main() -> None:
    th = json.loads((REPO_ROOT / "docs/protocol_v1.1_thresholds.json").read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        raise SystemExit(f"阈值表仍有占位 {th['_meta']['pending']}，不可生成冻结清单")
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
    for rel in FREEZE_SET:
        data = (REPO_ROOT / rel).read_bytes()
        manifest["files"][rel] = hashlib.sha256(data).hexdigest()
    out = REPO_ROOT / "docs" / "freeze_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"冻结清单已写入 {out}（{len(FREEZE_SET)} 个文件）")


if __name__ == "__main__":
    main()
