"""A4 复制族 P2×S2P 判定（协议 v1.3 增补）。

组装复制族 judge_input 并跑冻结 judge_v12，只对 P2×S2P（复制）出判定，
按增补 §3：判据与 v1.2 完全相同，过门种子多数（≥3/5 同向）。

- 门（能力/几何/平稳）：取自 results/confirmation/exp_p4/s2p_repl_v1（run_exp_p4
  --config exp_p4_repl 产，与评估族同源同口径）。P4 仿射统计不进 A4 判定。
- P2 统计：取自 results/confirmation/exp_p2/repl_v1/s<seed>/p2_summary.json。
- 判定器 judge_v12 冻结，写 results/confirmation/metrics_repl.json（不覆盖
  评估族 metrics.json）；记分牌 scoreboard_repl.json。

三值结局（增补 §3 预注册）：
- 通过：P2×S2P（复制）多数过门种子三层成立 → 新鲜样本外确证成立。
- 失败：任一层多数不成立且构成 → 复制未确证。
- 未构成检验：过门构成种子 < 2。

用法：.venv/Scripts/python.exe scripts/assemble_judge_repl.py
"""
from __future__ import annotations

import json

from slep import guard
from slep.protocols.judge import FAIL, NA, PASS
from slep.protocols.judge_v12 import run_judge_v12
from slep.utils.runs import REPO_ROOT

CONF = REPO_ROOT / "results" / "confirmation"
JUDGE_INPUT = CONF / "judge_input_repl.json"
METRICS = CONF / "metrics_repl.json"
P4_DIR = CONF / "exp_p4" / "s2p_repl_v1"
P2_DIR = CONF / "exp_p2" / "repl_v1"
TH_V12 = REPO_ROOT / "docs" / "protocol_v1.2_thresholds.json"
SEEDS_V1_3 = REPO_ROOT / "configs" / "seeds_v1_3.yaml"


def majority(votes: list[str]) -> tuple[str, str]:
    if len(votes) < 2:
        return NA, "构成判定的种子不足两个（增补 §3 第三值）"
    if votes.count(PASS) * 2 >= len(votes) + 1:
        return PASS, f"票型 {votes}"
    if votes.count(FAIL) * 2 >= len(votes) + 1:
        return FAIL, f"票型 {votes}"
    return NA, f"无多数（票型 {votes}）"


def main() -> None:
    th = json.loads(TH_V12.read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        raise SystemExit("v1.2 阈值表有占位，不可判定")
    seeds = guard.family_seeds("replication", purpose="judge-repl", seeds_file=SEEDS_V1_3)
    inp = {"systems": {}}
    for s in seeds:
        m = json.loads((P4_DIR / f"s{s}" / "main.json").read_text(encoding="utf-8"))
        cap = json.loads((P4_DIR / f"s{s}" / "capability.json").read_text(encoding="utf-8"))
        p2 = json.loads((P2_DIR / f"s{s}" / "p2_summary.json").read_text(encoding="utf-8"))
        block = inp["systems"].setdefault(f"S2P:s{s}", {})
        block["gates"] = {
            "capability": {"value": cap["rate"], "system": "S2P"},
            "geometry": m["geometry"],
            "stationarity": {"rhat": m["stationarity"]["rhat"],
                             "n_chains": m["stationarity"]["n_chains"]},
        }
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

    votes, per = [], {}
    for s in seeds:
        v = out["systems"].get(f"S2P:s{s}", {}).get("P2", {})
        g = out["systems"].get(f"S2P:s{s}", {}).get("gates", {})
        per[f"s{s}"] = {"verdict": v.get("verdict"), "reason": v.get("reason"),
                        "gate_ok": g.get("ok"), "gate_detail": g.get("detail")}
        if v.get("verdict") in (PASS, FAIL):
            votes.append(v["verdict"])
    agg, reason = majority(votes)
    board = {"protocol_version": "1.3", "family": "replication",
             "cells": {"P2×S2P(复制)": {"verdict": agg, "reason": reason, "per_seed": per}}}
    (CONF / "scoreboard_repl.json").write_text(
        json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"P2×S2P(复制): {agg}  {reason}")
    for s, d in per.items():
        print(f"  {s}: {d['verdict']} gate_ok={d['gate_ok']} :: {d['reason']}")


if __name__ == "__main__":
    main()
