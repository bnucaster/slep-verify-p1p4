"""评估族隔离守卫的用例（CLAUDE.md 硬规则 1）。

正向：冻结前评估族种子被拦截、未登记种子被拒绝。
反向：冻结置位后评估族放行。反向用例用 tmp_path 复制 freeze_status.json
并改写副本，仓库内真实文件不动，用例末尾校验真实文件仍未冻结。
"""
import json

import pytest

from slep import guard


def test_dev_seed_allowed():
    guard.assert_seed_allowed(0, purpose="unit-test")


def test_calibration_seed_allowed():
    guard.assert_seed_allowed(99, purpose="unit-test")


def test_eval_seed_blocked_before_freeze():
    with pytest.raises(guard.EvalFamilyIsolationError):
        guard.assert_seed_allowed(5, purpose="unit-test")


def test_unknown_seed_rejected():
    with pytest.raises(guard.UnknownSeedError):
        guard.assert_seed_allowed(42, purpose="unit-test")


def test_eval_seed_allowed_after_freeze_on_tmp_copy(tmp_path):
    frozen_copy = tmp_path / "freeze_status.json"
    payload = json.loads(guard.FREEZE_FILE.read_text(encoding="utf-8"))
    payload["frozen"] = True
    frozen_copy.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    family = guard.assert_seed_allowed(
        5, purpose="unit-test-frozen-copy", freeze_file=frozen_copy
    )
    assert family == "evaluation"

    # 副本置位不得影响仓库内真实文件：真实文件仍未冻结，评估族仍被拦截。
    assert guard.is_frozen() is False
    with pytest.raises(guard.EvalFamilyIsolationError):
        guard.assert_seed_allowed(5, purpose="unit-test-real-file-still-blocks")


def test_calibration_and_development_unaffected_by_freeze_state(tmp_path):
    frozen_copy = tmp_path / "freeze_status.json"
    payload = json.loads(guard.FREEZE_FILE.read_text(encoding="utf-8"))
    payload["frozen"] = True
    frozen_copy.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert guard.assert_seed_allowed(100, freeze_file=frozen_copy) == "calibration"
    assert guard.assert_seed_allowed(3, freeze_file=frozen_copy) == "development"
