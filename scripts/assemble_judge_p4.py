"""EXP-P4 judge 汇总（任务八/10e）：产物 → judge_input → 冻结 judge。

结构约定（后续 EXP-P1/P3/P2 沿用）：规范输入 results/confirmation/
judge_input.json 按系统键增量合并——本脚本写入 "S2:s<种子>" 五个键的
gates 与 exp_p4 段；后续实验脚本向相应键并入 exp_p1/p3/p2 段后重跑
judge。metrics.json 始终由冻结 judge 整体重写（唯一合法写入者），
输入全量回显在案。

聚合（协议 v1.1 第 3 节）：P4 × S2 标题格 = 过门种子多数（≥ 半数同
向）；judge 逐种子判定，多数计票是对 metrics.json 判定字段的机械计
数，写入本实验目录 aggregate.json（metrics.json 之外）。

能力门取池化口径（阈值 0.29 的推导口径）：五种子总成功 / 总任务，
同一数值写入五个种子键。

用法：.venv/Scripts/python.exe scripts/assemble_judge_p4.py
"""
from __future__ import annotations

import json

from slep import guard
from slep.protocols.judge import NA, PASS, FAIL, run_judge
from slep.utils.runs import REPO_ROOT

EXP_DIR = REPO_ROOT / "results" / "confirmation" / "exp_p4" / "eval_v1"
JUDGE_INPUT = REPO_ROOT / "results" / "confirmation" / "judge_input.json"
METRICS = REPO_ROOT / "results" / "confirmation" / "metrics.json"


def main() -> None:
    seeds = guard.family_seeds("evaluation", purpose="exp-p4-judge")
    mains, caps, ctrls = {}, {}, {}
    for s in seeds:
        sd = EXP_DIR / f"s{s}"
        mains[s] = json.loads((sd / "main.json").read_text(encoding="utf-8"))
        caps[s] = json.loads((sd / "capability.json").read_text(encoding="utf-8"))
        ctrls[s] = json.loads((sd / "controls.json").read_text(encoding="utf-8"))

    cap_pooled = sum(caps[s]["success"] for s in seeds) / sum(caps[s]["n"] for s in seeds)

    inp = {"systems": {}}
    if JUDGE_INPUT.exists():
        inp = json.loads(JUDGE_INPUT.read_text(encoding="utf-8"))
    for s in seeds:
        m = mains[s]
        key = f"S2:s{s}"
        block = inp["systems"].setdefault(key, {})
        block["gates"] = {
            "capability": {"value": cap_pooled, "system": "S2",
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
                "uniform_decoder_slope_shift_rel":
                    ctrls[s]["uniform_decoder"]["shift_rel"],
                "policy_swap_slope_shift_rel": ctrls[s]["policy_swap"]["shift_rel"],
            },
            "temperature_hat": m["affine_main"]["temperature_hat"],
        }
    JUDGE_INPUT.write_text(json.dumps(inp, ensure_ascii=False, indent=2), encoding="utf-8")

    out = run_judge(JUDGE_INPUT, METRICS)

    verdicts = {s: out["systems"][f"S2:s{s}"]["P4"]["verdict"] for s in seeds}
    votes = [v for v in verdicts.values() if v in (PASS, FAIL)]
    if len(votes) < 2:
        agg = NA
        reason = "构成判定的种子不足两个"
    elif votes.count(PASS) * 2 >= len(votes) + 1:
        agg, reason = PASS, f"票型 {votes}"
    elif votes.count(FAIL) * 2 >= len(votes) + 1:
        agg, reason = FAIL, f"票型 {votes}"
    else:
        agg, reason = NA, f"无多数（票型 {votes}）"
    aggregate = {
        "prediction": "P4",
        "system": "S2",
        "verdict_by_seed": verdicts,
        "aggregate_verdict": agg,
        "aggregate_rule": "协议 v1.1 第 3 节：过门种子多数（≥ 半数同向）",
        "reason": reason,
        "capability_pooled": cap_pooled,
        "temperature_by_seed": {s: mains[s]["affine_main"]["temperature_hat"]
                                for s in seeds},
    }
    (EXP_DIR / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"能力门池化 {cap_pooled:.3f}")
    for s in seeds:
        print(f"s{s}: {verdicts[s]}  T̂={mains[s]['affine_main']['temperature_hat']:.3f}")
    print(f"P4 × S2 聚合：{agg}（{reason}）")


if __name__ == "__main__":
    main()
