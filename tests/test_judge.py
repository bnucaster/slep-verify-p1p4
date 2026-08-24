"""judge 判定分支的穷举测试（冻结前自检；纯函数注入完整阈值表）。"""
import json

import pytest

from slep.protocols import judge
from slep.protocols.judge import FAIL, NA, PASS


def _th() -> dict:
    th = json.loads(judge.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    th["capability_gate"]["s2_nav_success_min"] = 0.3  # 测试注入占位值
    th["plateau"]["s2_window"] = 3
    th["_meta"]["pending"] = []
    return th


def test_load_thresholds_refuses_pending():
    # 真文件仍有占位时 judge 不可运行（冻结闸门）
    th = json.loads(judge.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        with pytest.raises(RuntimeError):
            judge.load_thresholds()


def test_gates_missing_and_pass():
    th = _th()
    g = judge.judge_gates("S1", {}, th)
    assert not g["ok"] and g["missing"]
    g2 = judge.judge_gates(
        "S1",
        {"capability": {"value": 0.6},
         "geometry": {"log10_cond_median": 5.0, "participation_median": 2.5},
         "stationarity": {"rhat": 1.01, "n_chains": 4}},
        th,
    )
    assert g2["ok"]
    g3 = judge.judge_gates(
        "S1",
        {"capability": {"value": 0.2},
         "geometry": {"log10_cond_median": 5.0, "participation_median": 2.5}},
        th,
    )
    assert not g3["ok"] and not g3["detail"]["capability"]


def _p4_base() -> dict:
    return {"p_lack_of_fit": 0.5, "delta_bic": -3.0, "r_squared": 0.4,
            "curvature_effect_ratio": 0.05, "slope": 2.0, "t_split_rel": 0.05,
            "decoupling": {"uniform_decoder_slope_shift_rel": 0.03,
                           "policy_swap_slope_shift_rel": 0.02}}


def test_p4_branches():
    th = _th()
    assert judge.judge_p4(_p4_base(), False, th)["verdict"] == NA  # 门未过
    assert judge.judge_p4(_p4_base(), True, th)["verdict"] == PASS
    r2low = {**_p4_base(), "r_squared": 0.1}
    assert judge.judge_p4(r2low, True, th)["verdict"] == NA
    dec = {**_p4_base(), "decoupling": {"uniform_decoder_slope_shift_rel": 0.3,
                                        "policy_swap_slope_shift_rel": 0.02}}
    assert judge.judge_p4(dec, True, th)["verdict"] == NA  # 解耦失败→不构成支持
    excused = {**_p4_base(), "p_lack_of_fit": 0.001, "curvature_effect_ratio": 0.05}
    assert judge.judge_p4(excused, True, th)["verdict"] == PASS  # 统计拒绝但效应量豁免
    bent = {**_p4_base(), "p_lack_of_fit": 0.001, "curvature_effect_ratio": 0.3}
    assert judge.judge_p4(bent, True, th)["verdict"] == FAIL
    split = {**_p4_base(), "t_split_rel": 0.5}
    assert judge.judge_p4(split, True, th)["verdict"] == FAIL
    missing = {k: v for k, v in _p4_base().items() if k != "slope"}
    assert judge.judge_p4(missing, True, th)["verdict"] == NA


def _p1_seed(plateau=2000):
    return {"dual_shape": {"rho": 0.98, "delta_same_sign": True},
            "sigma_u": 0.001, "sigma_u_max": 0.0017, "plateau_step": plateau,
            "s_drop": -1.0, "s_band": 0.3, "v_post_trend": "down_beyond_band",
            "s_drop_peak_window": -0.8, "peak_step": 1500}


def test_p1_seed_branches():
    th = _th()
    assert judge.judge_p1_seed(_p1_seed(), th)["verdict"] == PASS
    no_plat = judge.judge_p1_seed(_p1_seed(plateau=None), th)
    assert no_plat["verdict"] == NA and no_plat["supplement_peak_window"] is True
    bad_shape = {**_p1_seed(), "dual_shape": {"rho": 0.5, "delta_same_sign": True}}
    assert judge.judge_p1_seed(bad_shape, th)["verdict"] == NA
    noisy = {**_p1_seed(), "sigma_u": 0.01}
    assert judge.judge_p1_seed(noisy, th)["verdict"] == NA
    v_up = {**_p1_seed(), "v_post_trend": "up_beyond_band"}
    assert judge.judge_p1_seed(v_up, th)["verdict"] == FAIL
    s_flat = {**_p1_seed(), "s_drop": 0.1}
    assert judge.judge_p1_seed(s_flat, th)["verdict"] == FAIL


def test_p1_aggregate_majority_and_insufficient():
    th = _th()
    seeds = {str(i): _p1_seed() for i in range(4)}
    seeds["3"] = {**_p1_seed(), "v_post_trend": "up_beyond_band"}
    inp = {"seeds": seeds}
    gates = {s: True for s in seeds}
    r = judge.judge_p1(inp, gates, th)
    assert r["verdict"] == PASS  # 3 通过对 1 失败
    r2 = judge.judge_p1({"seeds": {"0": _p1_seed()}}, {"0": True}, th)
    assert r2["verdict"] == NA  # 构成判定的种子不足


def test_p3_branches():
    th = _th()
    base = {"terrain_ratio_ci_low": 1.4, "mw_p": 0.001, "effect": 0.05,
            "low_lt_high": True, "n_certified_pairs": 20}
    assert judge.judge_p3(base, True, th)["verdict"] == PASS
    assert judge.judge_p3(base, False, th)["verdict"] == NA
    assert judge.judge_p3({**base, "terrain_ratio_ci_low": 0.9}, True, th)["verdict"] == NA
    assert judge.judge_p3({**base, "n_certified_pairs": 5}, True, th)["verdict"] == NA
    assert judge.judge_p3({**base, "low_lt_high": False}, True, th)["verdict"] == FAIL
    assert judge.judge_p3({**base, "effect": 0.001}, True, th)["verdict"] == FAIL


def test_p2_branches():
    th = _th()
    base = {"frac_below_q1": 0.9, "n_traj": 1200,
            "ablation": {"wilcoxon_p": 0.001, "median_diff": 0.5},
            "novelty": {"logistic_p": 0.001, "positive": True}}
    assert judge.judge_p2(base, True, th)["verdict"] == PASS
    assert judge.judge_p2({**base, "n_traj": 100}, True, th)["verdict"] == NA
    novel_bad = {**base, "novelty": {"logistic_p": 0.5, "positive": True}}
    assert judge.judge_p2(novel_bad, True, th)["verdict"] == FAIL  # 母论文原判据
    abl_bad = {**base, "ablation": {"wilcoxon_p": 0.5, "median_diff": 0.0}}
    assert judge.judge_p2(abl_bad, True, th)["verdict"] == FAIL  # 常数势同样集中
    conc_bad = {**base, "frac_below_q1": 0.5}
    assert judge.judge_p2(conc_bad, True, th)["verdict"] == FAIL
