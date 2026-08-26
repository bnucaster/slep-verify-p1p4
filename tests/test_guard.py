"""评估族隔离守卫的用例（CLAUDE.md 硬规则 1）。

冻结前后两种状态各用 tmp_path 副本显式构造，用例不依赖仓库内
freeze_status.json 的现值（协议冻结日 2026-08-26 起真实文件为
frozen=true）；另设一条一致性用例断言真实文件的 frozen 标志与守卫
实际放行行为相符。
"""
import json

import pytest

from slep import guard


def _freeze_copy(tmp_path, frozen: bool):
    copy = tmp_path / "freeze_status.json"
    payload = json.loads(guard.FREEZE_FILE.read_text(encoding="utf-8"))
    payload["frozen"] = frozen
    copy.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return copy


def test_dev_seed_allowed():
    guard.assert_seed_allowed(0, purpose="unit-test")


def test_calibration_seed_allowed():
    guard.assert_seed_allowed(99, purpose="unit-test")


def test_eval_seed_blocked_before_freeze(tmp_path):
    unfrozen = _freeze_copy(tmp_path, frozen=False)
    with pytest.raises(guard.EvalFamilyIsolationError):
        guard.assert_seed_allowed(5, purpose="unit-test", freeze_file=unfrozen)


def test_unknown_seed_rejected():
    with pytest.raises(guard.UnknownSeedError):
        guard.assert_seed_allowed(42, purpose="unit-test")


def test_eval_seed_allowed_after_freeze_on_tmp_copy(tmp_path):
    frozen_copy = _freeze_copy(tmp_path, frozen=True)
    family = guard.assert_seed_allowed(
        5, purpose="unit-test-frozen-copy", freeze_file=frozen_copy
    )
    assert family == "evaluation"


def test_real_freeze_file_state_matches_guard_behaviour():
    # 一致性：真实文件的 frozen 标志与守卫对评估族的实际行为同向。
    if guard.is_frozen():
        assert guard.family_seeds("evaluation", purpose="unit-test-consistency") == [5, 6, 7, 8, 9]
    else:
        with pytest.raises(guard.EvalFamilyIsolationError):
            guard.family_seeds("evaluation", purpose="unit-test-consistency")


def test_family_seeds_calibration_matches_registry():
    assert guard.family_seeds("calibration") == [99, 100]


def test_family_seeds_unknown_family_rejected():
    with pytest.raises(KeyError):
        guard.family_seeds("holdout")


def test_family_seeds_evaluation_blocked_before_freeze(tmp_path):
    unfrozen = _freeze_copy(tmp_path, frozen=False)
    with pytest.raises(guard.EvalFamilyIsolationError):
        guard.family_seeds("evaluation", purpose="unit-test", freeze_file=unfrozen)


def test_family_seeds_evaluation_allowed_on_frozen_copy(tmp_path):
    frozen_copy = _freeze_copy(tmp_path, frozen=True)
    assert guard.family_seeds("evaluation", freeze_file=frozen_copy) == [5, 6, 7, 8, 9]


def test_calibration_and_development_unaffected_by_freeze_state(tmp_path):
    for frozen in (False, True):
        copy = _freeze_copy(tmp_path, frozen=frozen)
        assert guard.assert_seed_allowed(100, freeze_file=copy) == "calibration"
        assert guard.assert_seed_allowed(3, freeze_file=copy) == "development"
