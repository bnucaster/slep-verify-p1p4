"""judge v1.2 分支自检（表驱动门 + P1-only 系统 + v1.1 函数复用）。

用 tmp 阈值表副本运行（真实 v1.2 表冻结前可含占位，不依赖其状态）。
"""
import json

import pytest

from slep.protocols import judge_v12
from slep.protocols.judge import FAIL, NA, PASS


TH = {
    "_meta": {"pending": []},
    "capability_gate": {"by_system": {"S1": 0.45, "S2": 0.29, "S2P": 0.29, "S3": 0.29}},
    "geometry_gate": {"log10_cond_median_max": 7.0, "participation_dim_median_min": 1.5},
    "stationarity_gate": {"rhat_max": 1.05, "n_chains_min": 4},
    "p4": {"lack_of_fit_p_min": 0.05, "delta_bic_max": 2.0, "r2_min": 0.25,
           "curvature_effect_ratio_max": 0.17, "t_split_half_rel_max": 0.20,
           "t_rel_err_context": 0.12},
    "p1": {"dual_shape_rho_min": 0.9, "s_drop_band_multiplier": 2.0},
    "p2": {"frac_below_q1_min": 0.75, "n_traj_min": 1000, "ablation_wilcoxon_p_max": 0.01,
           "novelty_logistic_p_max": 0.01},
    "geodesic": {"mannwhitney_p_max": 0.01, "deviation_min_effect": 0.0125,
                 "certified_pairs_min": 8},
}

GOOD_GATES = {"capability": {"value": 0.35}, "geometry": {"log10_cond_median": 4.0,
              "participation_median": 3.0}}


def test_gate_table_per_system():
    g = judge_v12.judge_gates_v12("S2P:s5", GOOD_GATES, TH)
    assert g["ok"]
    g2 = judge_v12.judge_gates_v12("S1", GOOD_GATES, TH)  # S1 阈值 0.45 > 0.35
    assert not g2["ok"] and not g2["detail"]["capability"]


def test_gate_unknown_system_class_missing():
    g = judge_v12.judge_gates_v12("SX:s0", GOOD_GATES, TH)
    assert not g["ok"] and "capability_gate.by_system.SX" in g["missing"]


def test_gate_geometry_v12_boundary():
    gates = {"capability": {"value": 0.5},
             "geometry": {"log10_cond_median": 6.5, "participation_median": 2.0}}
    assert judge_v12.judge_gates_v12("S2P:s5", gates, TH)["ok"]  # 6.5 ≤ 7.0（v1.1 为 6.0）
    gates["geometry"]["log10_cond_median"] = 7.5
    assert not judge_v12.judge_gates_v12("S2P:s5", gates, TH)["ok"]


def test_run_judge_v12_p1_only_for_s3(tmp_path, monkeypatch):
    inp = {"systems": {"S3:s5": {
        "gates": GOOD_GATES,
        "exp_p4": {"p_lack_of_fit": 0.5, "delta_bic": -5, "r_squared": 0.4,
                   "curvature_effect_ratio": 0.05, "slope": 1.0, "t_split_rel": 0.05,
                   "decoupling": {"uniform_decoder_slope_shift_rel": 0.01,
                                  "policy_swap_slope_shift_rel": 0.01}},
        "exp_p1": {"seeds": {}},
    }}}
    f_in = tmp_path / "judge_input.json"
    f_in.write_text(json.dumps(inp), encoding="utf-8")
    f_th = tmp_path / "th.json"
    f_th.write_text(json.dumps(TH), encoding="utf-8")
    monkeypatch.setattr(judge_v12, "THRESHOLDS_FILE_V12", f_th)
    out = judge_v12.run_judge_v12(f_in, tmp_path / "metrics.json")
    block = out["systems"]["S3:s5"]
    assert "P4" not in block  # S3 只判 P1
    assert out["protocol_version"] == "1.2"
    assert block["P1"]["verdict"] == NA  # 无种子输入


def test_run_judge_v12_p4_reuses_v11_semantics(tmp_path, monkeypatch):
    exp_p4_pass = {"p_lack_of_fit": 0.5, "delta_bic": -5, "r_squared": 0.4,
                   "curvature_effect_ratio": 0.05, "slope": 1.0, "t_split_rel": 0.05,
                   "decoupling": {"uniform_decoder_slope_shift_rel": 0.01,
                                  "policy_swap_slope_shift_rel": 0.01}}
    inp = {"systems": {"S2P:s5": {"gates": GOOD_GATES, "exp_p4": exp_p4_pass}}}
    f_in = tmp_path / "judge_input.json"
    f_in.write_text(json.dumps(inp), encoding="utf-8")
    f_th = tmp_path / "th.json"
    f_th.write_text(json.dumps(TH), encoding="utf-8")
    monkeypatch.setattr(judge_v12, "THRESHOLDS_FILE_V12", f_th)
    out = judge_v12.run_judge_v12(f_in, tmp_path / "metrics.json")
    assert out["systems"]["S2P:s5"]["P4"]["verdict"] == PASS
    inp["systems"]["S2P:s5"]["exp_p4"]["r_squared"] = 0.1  # 低于 r2_min → NA
    f_in.write_text(json.dumps(inp), encoding="utf-8")
    out = judge_v12.run_judge_v12(f_in, tmp_path / "metrics.json")
    assert out["systems"]["S2P:s5"]["P4"]["verdict"] == NA


def test_pending_thresholds_block(tmp_path, monkeypatch):
    f_th = tmp_path / "th.json"
    f_th.write_text(json.dumps({"_meta": {"pending": ["x"]}}), encoding="utf-8")
    monkeypatch.setattr(judge_v12, "THRESHOLDS_FILE_V12", f_th)
    with pytest.raises(RuntimeError):
        judge_v12.load_thresholds_v12()
