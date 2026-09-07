"""P2 种子级统计（补充实验 A2，回应审稿 critical 2 的推断单元质疑）。

审稿指出:势项消融的 Wilcoxon 在 1000 条轨迹上做(轨迹级),而归因是种子
级。本脚本从已缓存的逐轨迹数据重聚合到种子级,不重跑管线:

- 每种子的中位配对差(含势间隙 − 常数势间隙)与自助 95% CI;
- 跨 5 种子的符号检验与 Wilcoxon 符号秩(把种子当推断单元);
- 集中性(低于 Q1 比例)与新奇关联的逐种子值。

对哪个评估口径做:默认 v1.2 评估族(exp_p2/eval_v1);--campaign 可指
复制族(exp_p2/repl_v1)。产物 <exp_dir>/seedlevel.json。

用法:.venv/Scripts/python.exe scripts/run_p2_seedlevel.py [--campaign eval_v1]
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

from slep.utils.runs import REPO_ROOT


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = [float(np.median(rng.choice(x, size=len(x), replace=True))) for _ in range(n)]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main() -> None:
    campaign = "eval_v1"
    if "--campaign" in sys.argv:
        campaign = sys.argv[sys.argv.index("--campaign") + 1]
    exp = REPO_ROOT / "results" / "confirmation" / "exp_p2" / campaign
    seed_dirs = sorted(p for p in exp.glob("s*") if p.is_dir())
    per_seed = {}
    seed_median_diffs = []
    for sd in seed_dirs:
        main, abl = {}, {}
        for p in sorted((sd / "main_cache").glob("b*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                main[r["traj"]] = r["gap"]
        for p in sorted((sd / "abl_cache").glob("b*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                abl[r["traj"]] = r["gap"]
        diffs = np.array([main[t] - abl[t] for t in main if t in abl])
        below_q1 = float(np.mean([r < 0 for r in []])) if False else None
        p2 = json.loads((sd / "p2_summary.json").read_text(encoding="utf-8"))
        lo, hi = boot_ci(diffs, seed=int(sd.name[1:]))
        per_seed[sd.name] = {
            "n_pairs": int(len(diffs)),
            "median_diff": float(np.median(diffs)),
            "median_diff_ci95": [lo, hi],
            "frac_positive_pairs": float(np.mean(diffs > 0)),
            "frac_below_q1": p2["frac_below_q1"],
            "novelty_p": p2["novelty"]["logistic_p"],
            "novelty_placebo_p": p2["novelty"]["placebo_p"],
        }
        seed_median_diffs.append(float(np.median(diffs)))

    smd = np.array(seed_median_diffs)
    # 种子级检验:5 个种子中位配对差 > 0 的符号检验与 Wilcoxon 符号秩
    n_pos = int(np.sum(smd > 0))
    sign_p = float(stats.binomtest(n_pos, len(smd), 0.5, alternative="greater").pvalue)
    try:
        w = stats.wilcoxon(smd, alternative="greater")
        wilcoxon_p = float(w.pvalue)
    except ValueError:
        wilcoxon_p = None
    out = {
        "campaign": campaign,
        "inferential_note": "种子为推断单元:每种子取 1000 对的中位配对差,"
                            "跨种子做符号检验/Wilcoxon;轨迹级 p 值另见 p2_summary",
        "per_seed": per_seed,
        "seed_level": {
            "n_seeds": len(smd),
            "seed_median_diffs": seed_median_diffs,
            "mean_of_seed_medians": float(smd.mean()),
            "n_seeds_positive": n_pos,
            "sign_test_p": sign_p,
            "wilcoxon_across_seeds_p": wilcoxon_p,
            "frac_below_q1_all_seeds": [per_seed[s]["frac_below_q1"] for s in per_seed],
            "novelty_p_range": [min(per_seed[s]["novelty_p"] for s in per_seed),
                                max(per_seed[s]["novelty_p"] for s in per_seed)],
        },
    }
    (exp / "seedlevel.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    sl = out["seed_level"]
    print(f"[{campaign}] 种子级:中位配对差 {[round(x,4) for x in seed_median_diffs]}")
    print(f"  {sl['n_seeds_positive']}/{sl['n_seeds']} 种子为正,符号检验 p={sl['sign_test_p']:.4f},"
          f" Wilcoxon p={sl['wilcoxon_across_seeds_p']}")
    print(f"  低于Q1比例 {sl['frac_below_q1_all_seeds']},新奇 p 范围 "
          f"{[f'{x:.1e}' for x in sl['novelty_p_range']]}")


if __name__ == "__main__":
    main()
