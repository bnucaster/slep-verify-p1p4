"""几何匹配：以 S1/S2 试点实测谱为目标构造合成 Langevin 校准系统（任务二 2d）。

对两个目标各拟合一个系统：
- matched_s1：d=8，目标为 S1 试点 β=1 的 ĝ 谱统计；
- matched_s2：d=16，目标为 S2 试点主种子的 ĝ 谱统计（CAL-P4 主实例）。

验证两件：
1. 匹配质量：目标 / 达成 谱统计对照（match_report.json + spectra_match.png）；
2. 平稳性：多链 Langevin 模拟后，白化坐标均值 / 方差对 OU 闭式目标的
   偏差（容差推导同 tests/test_cal_langevin.py 文件头），并报告分链
   Gelman–Rubin 型 R̂（stationarity.json）。

产物落盘 results/calibration/geometry_match/<run_id>/，其中
matched_s1.json / matched_s2.json 含全部系统参数（构造亦可由
config + 种子确定性复现）。

用法：.venv/Scripts/python.exe scripts/run_geometry_match.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import yaml

from slep import guard
from slep.systems.cal_langevin import build_matched_system, system_to_params
from slep.utils.runs import REPO_ROOT, create_run_dir

CONFIG_FILE = REPO_ROOT / "configs" / "geometry_match.yaml"


def load_target(cfg: dict, which: str, log) -> dict:
    """从试点 summary.json 读取谱匹配目标。"""
    if which == "s1":
        summary = json.loads(
            (REPO_ROOT / cfg["s1_pilot_run"] / "summary.json").read_text(encoding="utf-8")
        )
        entry = next(e for e in summary if e["beta"] == cfg["s1_target_beta"])
    else:
        summary = json.loads(
            (REPO_ROOT / cfg["s2_pilot_run"] / "summary.json").read_text(encoding="utf-8")
        )
        seeds = guard.family_seeds("calibration", purpose="geometry-match-target")
        entry = next(e for e in summary if e["seed"] == seeds[cfg["s2_target_seed_index"]])
    target = {
        "eig_median_by_rank": entry["eig_median_by_rank"],
        "logdet_std": entry["logdet_std"],
        "source": {k: entry[k] for k in ("seed", "log10_cond_quantiles", "logdet_mean")},
    }
    log(
        f"目标[{which}] d={len(target['eig_median_by_rank'])} "
        f"logdet_std={target['logdet_std']:.2f} "
        f"log10cond中位={entry['log10_cond_quantiles'][1]:.2f}"
    )
    return target


def stationarity_check(system, cfg: dict, generator: torch.Generator) -> dict:
    """多链模拟，白化坐标对 OU 闭式目标的一致性与分链 R̂。"""
    z = system.simulate_chains(
        cfg["chains"], cfg["steps"], cfg["dt"], cfg["burn_in"], cfg["thin"], generator
    )  # (C, m, d)
    n_chains, m, d = z.shape
    u = system.u_of_z(z.reshape(-1, d)).reshape(n_chains, m, d)

    var_target = system.temperature / system.well_w
    n_total = n_chains * m
    # OU 自相关时间 ~1/(w·dt) 步，保留间隔 thin 步 → n_eff ≈ n·w·dt·thin/3（保守除 3）
    n_eff = n_total * system.well_w * cfg["dt"] * cfg["thin"] / 3.0
    mean_tol = 6.0 * (var_target / n_eff).sqrt()
    var_tol = 6.0 * var_target * (2.0 / n_eff).sqrt()

    u_flat = u.reshape(-1, d)
    mean_err = (u_flat.mean(0) - system.well_u0).abs()
    var_err = (u_flat.var(0) - var_target).abs()

    chain_means = u.mean(dim=1)  # (C, d)
    within = u.var(dim=1).mean(dim=0)  # (d,)
    between = m * chain_means.var(dim=0)  # (d,)
    var_plus = ((m - 1) * within + between) / m
    rhat = (var_plus / within).sqrt()

    return {
        "mean_abs_err_max": float(mean_err.max()),
        "mean_tol_min": float(mean_tol.min()),
        "mean_within_tol": bool((mean_err <= mean_tol).all()),
        "var_abs_err_max": float(var_err.max()),
        "var_within_tol": bool((var_err <= var_tol).all()),
        "rhat_max": float(rhat.max()),
        "n_samples": int(n_total),
        "note": "均值/方差容差由 OU 有效样本量闭式推导（见 tests/test_cal_langevin.py 文件头）",
    }


def plot_match(entries: list[dict], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    colors = {"s1": "#2a78d6", "s2": "#eb6834"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for entry in entries:
        c = colors[entry["name"]]
        target = np.array(entry["report"]["target_eig_median_by_rank"])[::-1]
        achieved = np.array(entry["report"]["achieved_eig_median_by_rank"])[::-1]
        ranks = np.arange(1, len(target) + 1)
        ax.plot(ranks, target, color=c, linewidth=2, marker="o", markersize=6,
                label=f"{entry['name'].upper()} 目标")
        ax.plot(ranks, achieved, color=c, linewidth=2, linestyle="--", marker="s",
                markersize=5, label=f"{entry['name'].upper()} 达成")
    ax.set_yscale("log")
    ax.set_xlabel("特征值序（大 → 小）", color=ink2)
    ax.set_ylabel("ĝ 特征值中位数", color=ink2)
    ax.set_title("几何匹配：试点目标谱与合成系统达成谱", color=ink, fontsize=11)
    ax.grid(True, axis="y", color=muted, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink2)
    ax.legend(frameon=False, labelcolor=ink2, ncols=2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    run_dir = create_run_dir("calibration", "geometry_match", cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"run_dir: {run_dir}")
    seed = guard.family_seeds(cfg["seed_family"], purpose="geometry-match")[cfg["seed_index"]]
    entries, match_report, stationarity = [], {}, {}

    for name in ("s1", "s2"):
        target = load_target(cfg, name, log)
        d = len(target["eig_median_by_rank"])
        gen = torch.Generator()
        gen.manual_seed(seed)
        well_w = np.exp(
            np.linspace(math.log(cfg["well_w_min"]), math.log(cfg["well_w_max"]), d)
        ).tolist()
        system, report = build_matched_system(
            target_eig_median_by_rank=target["eig_median_by_rank"],
            target_logdet_std=target["logdet_std"],
            obs_dim=cfg["obs_dim"],
            sigma_dec=cfg["sigma_dec"],
            sigma_x=cfg["sigma_x"],
            well_w=well_w,
            temperature=cfg["temperature"],
            generator=gen,
            warp_tau=cfg["warp_tau"],
        )
        report["target_source"] = target["source"]
        match_report[name] = report
        (run_dir / f"matched_{name}.json").write_text(
            json.dumps(system_to_params(system), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(
            f"匹配[{name}] a={report['warp_a']:.3f} "
            f"logdet_std 目标 {report['target_logdet_std']:.2f} → 达成 {report['achieved_logdet_std']:.2f}, "
            f"log10cond中位达成 {report['achieved_log10_cond_median']:.2f}"
        )
        stat = stationarity_check(system, cfg, gen)
        stationarity[name] = stat
        log(
            f"平稳[{name}] 均值达标={stat['mean_within_tol']} 方差达标={stat['var_within_tol']} "
            f"R̂max={stat['rhat_max']:.4f} n={stat['n_samples']}"
        )
        entries.append({"name": name, "report": report})

    (run_dir / "match_report.json").write_text(
        json.dumps(match_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "stationarity.json").write_text(
        json.dumps(stationarity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_match(entries, run_dir / "spectra_match.png")
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"产物已写入 {run_dir}")


if __name__ == "__main__":
    main()
