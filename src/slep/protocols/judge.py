"""判定器 judge（冻结件；docs/protocol_v1.1_draft.md 第 5、7 节）。

results/confirmation/metrics.json 的唯一合法写入者（CLAUDE.md 硬规则 3）。
数值只从 docs/protocol_v1.1_thresholds.json 读取；输入契约见协议第 7 节；
任何必需字段缺失一律记"未构成检验"并注明缺失项。三值：通过 / 失败 /
未构成检验。

设计约束：本文件冻结后不得修改（freeze_manifest 哈希审计）；判定函数
均为纯函数，便于冻结前单元测试穷举分支。
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_FILE = _REPO_ROOT / "docs" / "protocol_v1.1_thresholds.json"

PASS, FAIL, NA = "通过", "失败", "未构成检验"


def load_thresholds() -> dict:
    th = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        raise RuntimeError(f"阈值表仍有占位 {th['_meta']['pending']}，冻结前 judge 不可运行")
    return th


def _missing(d: dict, keys: list[str]) -> list[str]:
    out = []
    for k in keys:
        cur = d
        ok = True
        for part in k.split("."):
            if not isinstance(cur, dict) or part not in cur or cur[part] is None:
                ok = False
                break
            cur = cur[part]
        if not ok:
            out.append(k)
    return out


def judge_gates(system: str, gates: dict, th: dict) -> dict:
    """入域门。返回 {ok, detail, missing}。"""
    need = ["capability.value", "geometry.log10_cond_median", "geometry.participation_median"]
    miss = _missing(gates, need)
    if miss:
        return {"ok": False, "missing": miss, "detail": {}}
    cap_min = (th["capability_gate"]["s1_u_composite_min"] if system == "S1"
               else th["capability_gate"]["s2_nav_success_min"])
    detail = {
        "capability": gates["capability"]["value"] >= cap_min,
        "geometry_cond": gates["geometry"]["log10_cond_median"]
        <= th["geometry_gate"]["log10_cond_median_max"],
        "geometry_dim": gates["geometry"]["participation_median"]
        >= th["geometry_gate"]["participation_dim_median_min"],
    }
    if "stationarity" in gates and gates["stationarity"] is not None:
        detail["stationarity"] = (
            gates["stationarity"]["rhat"] <= th["stationarity_gate"]["rhat_max"]
            and gates["stationarity"]["n_chains"] >= th["stationarity_gate"]["n_chains_min"]
        )
    return {"ok": all(detail.values()), "missing": [], "detail": detail}


def judge_p4(inp: dict, gate_ok: bool, th: dict) -> dict:
    if not gate_ok:
        return {"verdict": NA, "reason": "入域门未过"}
    need = ["p_lack_of_fit", "delta_bic", "r_squared", "curvature_effect_ratio",
            "slope", "t_split_rel", "decoupling.uniform_decoder_slope_shift_rel",
            "decoupling.policy_swap_slope_shift_rel"]
    miss = _missing(inp, need)
    if miss:
        return {"verdict": NA, "reason": f"字段缺失 {miss}"}
    t = th["p4"]
    if inp["r_squared"] < t["r2_min"]:
        return {"verdict": NA, "reason": "R² 低于仪器分辨力下限", "layer": {"r2": inp["r_squared"]}}
    dec_shift = max(inp["decoupling"]["uniform_decoder_slope_shift_rel"],
                    inp["decoupling"]["policy_swap_slope_shift_rel"])
    if dec_shift > t["t_rel_err_context"]:
        return {"verdict": NA, "reason": "解耦对照失败：斜率随占据统计漂移，表观仿射不构成支持",
                "layer": {"decoupling_shift": dec_shift}}
    stat_reject = (inp["p_lack_of_fit"] <= t["lack_of_fit_p_min"]
                   or inp["delta_bic"] >= t["delta_bic_max"])
    original_fail = ((stat_reject and inp["curvature_effect_ratio"] > t["curvature_effect_ratio_max"])
                     or inp["t_split_rel"] >= t["t_split_half_rel_max"]
                     or inp["slope"] <= 0)
    layer = {
        "stat_reject": stat_reject,
        "curvature_excused": stat_reject
        and inp["curvature_effect_ratio"] <= t["curvature_effect_ratio_max"],
        "split_half_ok": inp["t_split_rel"] < t["t_split_half_rel_max"],
    }
    return {"verdict": FAIL if original_fail else PASS, "layer": layer}


def judge_p1_seed(seed_inp: dict, th: dict) -> dict:
    """单种子 P1。两层：original（平台前提）与 supplement_peak（另列）。"""
    t = th["p1"]
    need = ["dual_shape.rho", "dual_shape.delta_same_sign", "sigma_u", "sigma_u_max"]
    miss = _missing(seed_inp, need)
    if miss:
        return {"verdict": NA, "reason": f"字段缺失 {miss}"}
    if not (seed_inp["dual_shape"]["rho"] >= t["dual_shape_rho_min"]
            and seed_inp["dual_shape"]["delta_same_sign"]):
        return {"verdict": NA, "reason": "Ŝ 双路形状判据未过（仪器）"}
    if seed_inp["sigma_u"] > seed_inp["sigma_u_max"]:
        return {"verdict": NA, "reason": "σ_Û 超数据条件（CAL-P1）"}

    # 增补峰值窗口层（不入标题）
    supplement_peak = None
    if seed_inp.get("s_drop_peak_window") is not None and seed_inp.get("s_band") is not None:
        supplement_peak = seed_inp["s_drop_peak_window"] < -seed_inp["s_band"]

    if seed_inp.get("plateau_step") is None:
        return {"verdict": NA, "reason": "效用平台未检出（母论文层前提）",
                "supplement_peak_window": supplement_peak}
    need2 = ["s_drop", "s_band", "v_post_trend"]
    miss2 = _missing(seed_inp, need2)
    if miss2:
        return {"verdict": NA, "reason": f"字段缺失 {miss2}"}
    original_pass = seed_inp["v_post_trend"] == "down_beyond_band"
    supplement_pass = seed_inp["s_drop"] < -seed_inp["s_band"]
    verdict = PASS if (original_pass and supplement_pass) else FAIL
    return {"verdict": verdict,
            "layer": {"original_v": original_pass, "supplement_s": supplement_pass},
            "supplement_peak_window": supplement_peak}


def judge_p1(inp: dict, gate_ok_by_seed: dict, th: dict) -> dict:
    """P1 聚合：过门种子多数（协议第 3 节）。inp["seeds"] = {seed: seed_inp}。"""
    if "seeds" not in inp or not inp["seeds"]:
        return {"verdict": NA, "reason": "无种子输入"}
    per_seed = {}
    votes = []
    for seed, seed_inp in inp["seeds"].items():
        if not gate_ok_by_seed.get(seed, False):
            per_seed[seed] = {"verdict": NA, "reason": "入域门未过"}
            continue
        r = judge_p1_seed(seed_inp, th)
        per_seed[seed] = r
        if r["verdict"] in (PASS, FAIL):
            votes.append(r["verdict"])
    if len([s for s in per_seed.values() if s["verdict"] != NA or "入域门" not in
            s.get("reason", "")]) < 2 or len(votes) < 2:
        agg = NA
        reason = "构成判定的种子不足两个"
    else:
        agg = PASS if votes.count(PASS) * 2 >= len(votes) + 1 else (
            FAIL if votes.count(FAIL) * 2 >= len(votes) + 1 else NA)
        reason = f"票型 {votes}"
    return {"verdict": agg, "reason": reason, "per_seed": per_seed}


def judge_p3(inp: dict, gate_ok: bool, th: dict) -> dict:
    if not gate_ok:
        return {"verdict": NA, "reason": "入域门未过"}
    need = ["terrain_ratio_ci_low", "mw_p", "effect", "low_lt_high", "n_certified_pairs"]
    miss = _missing(inp, need)
    if miss:
        return {"verdict": NA, "reason": f"字段缺失 {miss}"}
    t = th["geodesic"]
    if inp["terrain_ratio_ci_low"] <= 1.0:
        return {"verdict": NA, "reason": "地形过平，检验未运行"}
    if inp["n_certified_pairs"] < t["certified_pairs_min"]:
        return {"verdict": NA, "reason": "认证端点对不足"}
    ok = (inp["low_lt_high"] and inp["mw_p"] < t["mannwhitney_p_max"]
          and inp["effect"] >= t["deviation_min_effect"])
    return {"verdict": PASS if ok else FAIL,
            "layer": {"low_lt_high": inp["low_lt_high"], "mw_p": inp["mw_p"],
                      "effect": inp["effect"]}}


def judge_p2(inp: dict, gate_ok: bool, th: dict) -> dict:
    if not gate_ok:
        return {"verdict": NA, "reason": "入域门未过"}
    need = ["frac_below_q1", "n_traj", "ablation.wilcoxon_p", "ablation.median_diff",
            "novelty.logistic_p", "novelty.positive"]
    miss = _missing(inp, need)
    if miss:
        return {"verdict": NA, "reason": f"字段缺失 {miss}"}
    t = th["p2"]
    if inp["n_traj"] < t["n_traj_min"]:
        return {"verdict": NA, "reason": "轨迹数不足"}
    original_fail = not (inp["novelty"]["positive"]
                         and inp["novelty"]["logistic_p"] < t["novelty_logistic_p_max"])
    concentration = inp["frac_below_q1"] > t["frac_below_q1_min"]
    ablation_pass = (inp["ablation"]["median_diff"] > 0
                     and inp["ablation"]["wilcoxon_p"] < t["ablation_wilcoxon_p_max"])
    supplement_fail = (not concentration) or (not ablation_pass)
    verdict = FAIL if (original_fail or supplement_fail) else PASS
    return {"verdict": verdict,
            "layer": {"novelty_ok": not original_fail, "concentration": concentration,
                      "ablation": ablation_pass}}


def run_judge(input_file: Path, output_file: Path) -> dict:
    """读 judge_input.json，写 metrics.json（唯一合法写入路径）。"""
    th = load_thresholds()
    inp = json.loads(Path(input_file).read_text(encoding="utf-8"))
    out: dict = {"thresholds_file": str(THRESHOLDS_FILE.name), "systems": {}, "input_echo": inp}
    for system, sys_inp in inp["systems"].items():
        gates = judge_gates(system, sys_inp.get("gates", {}), th)
        gate_ok = gates["ok"]
        block = {"gates": gates}
        if "exp_p4" in sys_inp:
            block["P4"] = judge_p4(sys_inp["exp_p4"], gate_ok, th)
        if "exp_p1" in sys_inp:
            gate_by_seed = sys_inp.get("gates_by_seed",
                                       {s: gate_ok for s in sys_inp["exp_p1"].get("seeds", {})})
            block["P1"] = judge_p1(sys_inp["exp_p1"], gate_by_seed, th)
        if "exp_p3" in sys_inp:
            block["P3"] = judge_p3(sys_inp["exp_p3"], gate_ok, th)
        if "exp_p2" in sys_inp:
            block["P2"] = judge_p2(sys_inp["exp_p2"], gate_ok, th)
        out["systems"][system] = block
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
