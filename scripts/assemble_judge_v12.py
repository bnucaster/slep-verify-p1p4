"""终局判定汇总（确证 11.5 收官）：judge_v12 全量复判 + 记分牌。

组装 judge_input.json 的全部系统键并运行冻结 judge_v12 写 metrics.json
（protocol_version 1.2）：

- S2:s5..s9——v1.1 遗留条目原样保留（几何门在 v1.2 门 7.0 下仍不过，
  复判维持未构成检验）；
- S1——EXP-P1 × S1 条目已在（v1.2 幅度地板在此重算 delta_same_sign，
  与修订第 4 节一致）；
- S2P:s5..s9——gates（能力池化 + P4' 主链几何/平稳）、exp_p4
  （main + controls 解耦）、exp_p3（p3_summary）、exp_p2（p2_summary）；
- S2P / S3 系统键——EXP-P1 条目已由 run_exp_p1_wm 写入。

聚合（协议第 3 节：过门种子多数）对 P4/P3/P2 的逐种子判定计票，
连同 P1 的 judge 内聚合一起写入记分牌 scoreboard_v12.json（metrics.json
之外；判定字段只由 judge 产生，记分牌是对其的机械计数与排版）。

用法：.venv/Scripts/python.exe scripts/assemble_judge_v12.py
"""
from __future__ import annotations

import json

from slep import guard
from slep.protocols.judge import FAIL, NA, PASS
from slep.protocols.judge_v12 import run_judge_v12
from slep.utils.runs import REPO_ROOT

CONF = REPO_ROOT / "results" / "confirmation"
JUDGE_INPUT = CONF / "judge_input.json"
METRICS = CONF / "metrics.json"
P4_DIR = CONF / "exp_p4" / "s2p_eval_v1"
P3_DIR = CONF / "exp_p3" / "eval_v1"
P2_DIR = CONF / "exp_p2" / "eval_v1"
TH_V12 = REPO_ROOT / "docs" / "protocol_v1.2_thresholds.json"


def majority(votes: list[str]) -> tuple[str, str]:
    if len(votes) < 2:
        return NA, "构成判定的种子不足两个"
    if votes.count(PASS) * 2 >= len(votes) + 1:
        return PASS, f"票型 {votes}"
    if votes.count(FAIL) * 2 >= len(votes) + 1:
        return FAIL, f"票型 {votes}"
    return NA, f"无多数（票型 {votes}）"


def main() -> None:
    th = json.loads(TH_V12.read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        raise SystemExit("v1.2 阈值表有占位，不可判定")
    inp = json.loads(JUDGE_INPUT.read_text(encoding="utf-8"))
    seeds = guard.family_seeds("evaluation", purpose="judge-v12")

    # v1.2 幅度地板对 S1 条目重算 delta_same_sign（修订第 4 节；v1.1 版
    # 判定存档于其锚点提交，不受影响）
    s1 = inp["systems"].get("S1")
    if s1 is not None:
        band = s1.get("bands", {}).get("s_band")
        floor = th["p1"]["dual_delta_floor_frac"]
        agg = json.loads((CONF / "exp_p1_s1" / "eval_v1" / "aggregate.json")
                         .read_text(encoding="utf-8"))
        for name, a in agg.get("analyses", {}).items():
            si = s1["exp_p1"]["seeds"].get(name)
            if not si:
                continue
            d_f, d_k = a["dual_shape"]["delta_flow"], a["dual_shape"]["delta_knn"]
            same = bool(d_f * d_k > 0)
            if band is not None and max(abs(d_f), abs(d_k)) < floor * band:
                same = True
            si["dual_shape"]["delta_same_sign"] = same

    # S2P 逐种子条目
    caps = {s: json.loads((P4_DIR / f"s{s}" / "capability.json").read_text(encoding="utf-8"))
            for s in seeds}
    cap_pooled = sum(caps[s]["success"] for s in seeds) / sum(caps[s]["n"] for s in seeds)
    for s in seeds:
        m = json.loads((P4_DIR / f"s{s}" / "main.json").read_text(encoding="utf-8"))
        c = json.loads((P4_DIR / f"s{s}" / "controls.json").read_text(encoding="utf-8"))
        p3 = json.loads((P3_DIR / f"s{s}" / "p3_summary.json").read_text(encoding="utf-8"))
        block = inp["systems"].setdefault(f"S2P:s{s}", {})
        block["gates"] = {
            "capability": {"value": cap_pooled, "system": "S2P",
                           "per_seed_rate": caps[s]["rate"]},
            "geometry": m["geometry"],
            "stationarity": {"rhat": m["stationarity"]["rhat"],
                             "n_chains": m["stationarity"]["n_chains"]},
        }
        block["exp_p4"] = {
            "p_lack_of_fit": m["affine_main"]["p_lack_of_fit"],
            "delta_bic": m["affine_main"]["delta_bic_lin_minus_quad"],
            "r_squared": m["affine_main"]["r_squared"],
            "curvature_effect_ratio": m["affine_main"]["curvature_effect_ratio"],
            "slope": m["affine_main"]["slope"],
            "t_split_rel": m["t_split_rel"],
            "decoupling": {
                "uniform_decoder_slope_shift_rel": c["uniform_decoder"]["shift_rel"],
                "policy_swap_slope_shift_rel": c["policy_swap"]["shift_rel"],
            },
            "temperature_hat": m["affine_main"]["temperature_hat"],
        }
        block["exp_p3"] = {k: p3.get(k) for k in
                           ("terrain_ratio_ci_low", "mw_p", "effect", "low_lt_high",
                            "n_certified_pairs")}
        p2_file = P2_DIR / f"s{s}" / "p2_summary.json"
        if p2_file.exists():
            p2 = json.loads(p2_file.read_text(encoding="utf-8"))
            block["exp_p2"] = {
                "frac_below_q1": p2["frac_below_q1"], "n_traj": p2["n_traj"],
                "ablation": {"wilcoxon_p": p2["ablation"]["wilcoxon_p"],
                             "median_diff": p2["ablation"]["median_diff"]},
                "novelty": {"logistic_p": p2["novelty"]["logistic_p"],
                            "positive": p2["novelty"]["positive"]},
                "drift_fraction": p2["drift_fraction"],
            }
    JUDGE_INPUT.write_text(json.dumps(inp, ensure_ascii=False, indent=2), encoding="utf-8")

    out = run_judge_v12(JUDGE_INPUT, METRICS)

    board = {"protocol_version": "1.2", "cells": {}}
    for pred in ("P4", "P3", "P2"):
        votes = []
        per = {}
        for s in seeds:
            v = out["systems"].get(f"S2P:s{s}", {}).get(pred, {})
            per[f"s{s}"] = {"verdict": v.get("verdict"), "reason": v.get("reason")}
            if v.get("verdict") in (PASS, FAIL):
                votes.append(v["verdict"])
        agg, reason = majority(votes)
        board["cells"][f"{pred}×S2P"] = {"verdict": agg, "reason": reason, "per_seed": per}
        s2_votes = []
        per2 = {}
        for s in seeds:
            v = out["systems"].get(f"S2:s{s}", {}).get(pred, {})
            if v:
                per2[f"s{s}"] = {"verdict": v.get("verdict"), "reason": v.get("reason")}
                if v.get("verdict") in (PASS, FAIL):
                    s2_votes.append(v["verdict"])
        if per2:
            agg2, r2 = majority(s2_votes)
            board["cells"][f"{pred}×S2(基线)"] = {"verdict": agg2, "reason": r2,
                                                  "per_seed": per2}
    for sysname in ("S1", "S2P", "S3"):
        v = out["systems"].get(sysname, {}).get("P1", {})
        if v:
            board["cells"][f"P1×{sysname}"] = {
                "verdict": v.get("verdict"), "reason": v.get("reason"),
                "per_seed": {k: {"verdict": r.get("verdict"), "reason": r.get("reason"),
                                 "supplement_peak_window": r.get("supplement_peak_window")}
                             for k, r in v.get("per_seed", {}).items()}}
    (CONF / "scoreboard_v12.json").write_text(
        json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    for cell, v in board["cells"].items():
        print(f"{cell}: {v['verdict']}  {v.get('reason', '')}")


if __name__ == "__main__":
    main()
