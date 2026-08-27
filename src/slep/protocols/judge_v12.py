"""判定器 judge v1.2（协议 v1.2 修订件；docs/protocol_v1.2_amendment.md）。

与 v1.1 的关系：P4/P1/P3/P2 判定函数逐字沿用 v1.1 冻结实现（自
judge.py 导入，阈值经参数注入，v1.1 文件零改动）；重写的只有两处——

1. 入域门表驱动：能力门阈值按系统键查 capability_gate 表（v1.1 实现
   把非 S1 系统硬编码到 S2 阈值，S3/S2' 无法正确判门）。
2. run_judge_v12 读 docs/protocol_v1.2_thresholds.json，输出
   metrics.json 带 protocol_version 字段；系统键约定：判定系统类别
   取键中冒号前段（"S2P:s5" → "S2P"）。

v1.2 对 P1 的一处判据修订（披露）：双路符号一致性加幅度地板——
max(|Δ_flow|, |Δ_knn|) < dual_delta_floor_frac × s_band 时视为
"变化量与零不可分"，不因符号分歧记仪器未过（v1.1 下 b1_s9 因
+0.054 对 −0.018 的噪声级分歧记 NA；v1.2 下该情形进入实质判定，
方向上使检验更严而非更宽）。实现方式：汇总脚本在生成 judge_input
时按该规则预处理 delta_same_sign 字段，判定函数本身不改。

冻结属性：本文件随 v1.2 清单冻结；metrics.json 自 v1.2 置位起由本
文件的 run_judge_v12 写入。
"""
from __future__ import annotations

import json
from pathlib import Path

from slep.protocols.judge import (  # v1.1 冻结实现，零改动复用
    FAIL,
    NA,
    PASS,
    _missing,
    judge_p1,
    judge_p2,
    judge_p3,
    judge_p4,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_FILE_V12 = _REPO_ROOT / "docs" / "protocol_v1.2_thresholds.json"

__all__ = ["PASS", "FAIL", "NA", "judge_gates_v12", "run_judge_v12"]


def load_thresholds_v12() -> dict:
    th = json.loads(THRESHOLDS_FILE_V12.read_text(encoding="utf-8"))
    if th["_meta"].get("pending"):
        raise RuntimeError(f"v1.2 阈值表仍有占位 {th['_meta']['pending']}，冻结前不可运行")
    return th


def system_class(system_key: str) -> str:
    """系统键 → 类别（"S2P:s5" → "S2P"；"S1" → "S1"）。"""
    return system_key.split(":")[0]


def judge_gates_v12(system_key: str, gates: dict, th: dict) -> dict:
    """入域门（表驱动能力阈值）。返回 {ok, detail, missing}。"""
    need = ["capability.value", "geometry.log10_cond_median", "geometry.participation_median"]
    miss = _missing(gates, need)
    if miss:
        return {"ok": False, "missing": miss, "detail": {}}
    cls = system_class(system_key)
    cap_table = th["capability_gate"]["by_system"]
    if cls not in cap_table:
        return {"ok": False, "missing": [f"capability_gate.by_system.{cls}"], "detail": {}}
    detail = {
        "capability": gates["capability"]["value"] >= cap_table[cls],
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


P1_ONLY_CLASSES = {"S3"}  # S3 只跑 P1（plan_v2 第 2 节）


def run_judge_v12(input_file: Path, output_file: Path) -> dict:
    """读 judge_input.json，写 metrics.json（v1.2 置位后的唯一合法写入路径）。"""
    th = load_thresholds_v12()
    inp = json.loads(Path(input_file).read_text(encoding="utf-8"))
    out: dict = {"protocol_version": "1.2", "thresholds_file": THRESHOLDS_FILE_V12.name,
                 "systems": {}, "input_echo": inp}
    for system_key, sys_inp in inp["systems"].items():
        cls = system_class(system_key)
        gates = judge_gates_v12(system_key, sys_inp.get("gates", {}), th)
        gate_ok = gates["ok"]
        block = {"gates": gates}
        p1_only = cls in P1_ONLY_CLASSES
        if "exp_p4" in sys_inp and not p1_only:
            block["P4"] = judge_p4(sys_inp["exp_p4"], gate_ok, th)
        if "exp_p1" in sys_inp:
            gate_by_seed = sys_inp.get("gates_by_seed",
                                       {s: gate_ok for s in sys_inp["exp_p1"].get("seeds", {})})
            block["P1"] = judge_p1(sys_inp["exp_p1"], gate_by_seed, th)
        if "exp_p3" in sys_inp and not p1_only:
            block["P3"] = judge_p3(sys_inp["exp_p3"], gate_ok, th)
        if "exp_p2" in sys_inp and not p1_only:
            block["P2"] = judge_p2(sys_inp["exp_p2"], gate_ok, th)
        out["systems"][system_key] = block
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
