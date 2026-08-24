"""S1 三曲线：逐检查点 ⟨V̂⟩、Ŝ（flow/kNN 双路）、Û（任务六 6a，plan_v2 第 6 节）。

操作化（全部随配置记录）：
- Û：冻结检查点上因子探针留出精度的复合——形状逻辑回归精度、
  尺度/posX/posY 岭回归 R²（负值截零）四项均值；组分逐项归档。
  探针在后验均值上训练与评估。
- ⟨V̂⟩：留出同分布查询点上 k 近邻主口径 V̂ 的均值；参照集与查询潜态
  取后验采样（潜态分布口径）。
- Ŝ：后验采样潜态上的熵，flow 主口径与 kNN 留一双路并报（预承诺页
  分支判据引用两路分歧）。
- 平台检测：均匀检查点段（每 2k 步）上冻结版检测器（W 见配置）；
  σ_Û 在末检查点按测试折实测，与 CAL-P1 数据条件对照。

产物 results/description/s1_curves/<out_campaign>/：
b<β>_s<种子>/curves.csv 与 summary.json；跨种子图 curves_b<β>.png；
campaign 级 summary.json。

用法：.venv/Scripts/python.exe scripts/run_s1_curves.py [--beta 1|4]
（--beta 只算一个 β 臂，供两进程并行）
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
import torch
import yaml

from slep import guard
from slep.estimators import potential
from slep.estimators.entropy import entropy_flow, entropy_knn
from slep.estimators.flow import fit_flow_density
from slep.protocols.plateau import detect_plateau
from slep.systems.s1_beta_vae import S1BetaVAE
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "s1_curves.yaml"


def load_data(train_cfg: dict):
    packed = np.load(REPO_ROOT / train_cfg["packed_cache"])
    lat = np.load(REPO_ROOT / train_cfg["data_file"])
    return packed, lat["latents_classes"], lat["latents_values"]


def batch_imgs(packed: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    imgs = np.unpackbits(packed[idx], axis=1).astype(np.float32)
    return torch.from_numpy(imgs).reshape(-1, 1, 64, 64)


def encode(model: S1BetaVAE, packed: np.ndarray, idx: np.ndarray, gen: torch.Generator,
           sample: bool) -> torch.Tensor:
    """后验均值（sample=False）或后验采样（True）。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            mu, logvar = model.encoder(batch_imgs(packed, idx[i : i + 512]))
            if sample:
                eps = torch.randn(mu.shape, generator=gen, dtype=mu.dtype)
                outs.append(mu + torch.exp(0.5 * logvar) * eps)
            else:
                outs.append(mu)
    return torch.cat(outs)


def probes(z_train, z_test, cls_train, cls_test, val_train, val_test) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score

    zt, zv = z_train.numpy(), z_test.numpy()
    out = {}
    clf = LogisticRegression(max_iter=300).fit(zt, cls_train[:, 1])
    out["shape_acc"] = float(clf.score(zv, cls_test[:, 1]))
    out["_shape_pred"] = clf.predict(zv)
    for name, col in (("scale_r2", 2), ("posx_r2", 4), ("posy_r2", 5)):
        reg = Ridge().fit(zt, val_train[:, col])
        out[name] = float(r2_score(val_test[:, col], reg.predict(zv)))
        out[f"_{name}_pred"] = reg.predict(zv)
    out["u_composite"] = float(
        np.mean([out["shape_acc"]] + [max(out[k], 0.0) for k in ("scale_r2", "posx_r2", "posy_r2")])
    )
    return out


def sigma_u_from_folds(probe_out: dict, cls_test, val_test, n_folds: int) -> float:
    """末检查点 σ_Û 实测：同一探针模型在测试折上的复合 Û 标准差。"""
    n = len(cls_test)
    fold = n // n_folds
    comps = []
    for f in range(n_folds):
        sl = slice(f * fold, (f + 1) * fold)
        shape_acc = float((probe_out["_shape_pred"][sl] == cls_test[sl, 1]).mean())
        r2s = []
        for name, col in (("scale_r2", 2), ("posx_r2", 4), ("posy_r2", 5)):
            pred = probe_out[f"_{name}_pred"][sl]
            y = val_test[sl, col]
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2s.append(max(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 0.0))
        comps.append(float(np.mean([shape_acc] + r2s)))
    return float(np.std(comps)) / np.sqrt(n_folds)  # 折均值的标准误≈全测试集 Û 的 σ


def run_curves(cfg, train_cfg, packed, classes, values, beta: float, seed: int,
               out_campaign, log) -> dict:
    run_train = REPO_ROOT / "results" / "description" / "s1_train" / cfg["train_campaign"] / f"b{beta:g}_s{seed}"
    out_dir = out_campaign / f"b{beta:g}_s{seed}"
    out_dir.mkdir(exist_ok=True)
    if (out_dir / "summary.json").exists():
        log(f"跳过已完成 b{beta:g}_s{seed}")
        return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(seed + 777_000)
    total = (cfg["n_ref"] + cfg["n_query"] + cfg["n_probe_train"] + cfg["n_probe_test"]
             + cfg["n_flow_train"] + cfg["n_flow_val"] + cfg["n_entropy_eval"])
    pool = rng.choice(packed.shape[0], size=total, replace=False)
    ofs = 0

    def take(n):
        nonlocal ofs
        out = pool[ofs : ofs + n]
        ofs += n
        return out

    idx_ref, idx_q = take(cfg["n_ref"]), take(cfg["n_query"])
    idx_pt, idx_pv = take(cfg["n_probe_train"]), take(cfg["n_probe_test"])
    idx_ft, idx_fv = take(cfg["n_flow_train"]), take(cfg["n_flow_val"])
    idx_ent = take(cfg["n_entropy_eval"])
    x_ref = batch_imgs(packed, idx_ref).reshape(len(idx_ref), -1)

    ckpt_dir = run_train / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    rows = []
    sigma_u = None
    t0 = time.time()
    for ci, ck in enumerate(ckpts):
        payload = torch.load(ck, weights_only=True)
        model = S1BetaVAE(train_cfg["latent_dim"], train_cfg["channels"],
                          train_cfg["fc_dim"], train_cfg["sigma_dec"], beta)
        model.load_state_dict(payload["model"])
        model.eval()
        gen = torch.Generator()
        gen.manual_seed(seed * 1000 + payload["step"])

        z_ref = encode(model, packed, idx_ref, gen, sample=True).double()
        z_q = encode(model, packed, idx_q, gen, sample=True).double()
        z_pt = encode(model, packed, idx_pt, gen, sample=False)
        z_pv = encode(model, packed, idx_pv, gen, sample=False)
        z_ft = encode(model, packed, idx_ft, gen, sample=True).double()
        z_fv = encode(model, packed, idx_fv, gen, sample=True).double()
        z_ent = encode(model, packed, idx_ent, gen, sample=True).double()

        pr = probes(z_pt, z_pv, classes[idx_pt], classes[idx_pv], values[idx_pt], values[idx_pv])
        v_parts = []
        for i in range(0, z_q.shape[0], 256):
            def dec(z):
                return model.decoder(z.float()).reshape(z.shape[0], -1).double()
            v_parts.append(potential.potential_knn(
                dec, train_cfg["sigma_dec"] ** 2, z_q[i : i + 256], z_ref,
                x_ref.double(), k=cfg["k_potential"]))
        v_hat = torch.cat(v_parts)
        fd, _ = fit_flow_density(z_ft, z_fv, seed=seed, n_couplings=cfg["flow"]["couplings"],
                                 hidden=cfg["flow"]["hidden"], epochs=cfg["flow"]["epochs"],
                                 batch_size=cfg["flow"]["batch"])
        s_flow = entropy_flow(fd, z_ent)
        s_knn = entropy_knn(z_ent, k=cfg["k_entropy"])

        row = {
            "step": int(payload["step"]),
            "u_composite": pr["u_composite"], "shape_acc": pr["shape_acc"],
            "scale_r2": pr["scale_r2"], "posx_r2": pr["posx_r2"], "posy_r2": pr["posy_r2"],
            "v_mean": float(v_hat.mean()), "v_std": float(v_hat.std()),
            "s_flow": s_flow, "s_knn": s_knn,
        }
        rows.append(row)
        if ck == ckpts[-1]:
            sigma_u = sigma_u_from_folds(pr, classes[idx_pv], values[idx_pv], cfg["sigma_u_folds"])
        if ci % 5 == 0 or ck == ckpts[-1]:
            log(f"  b{beta:g}_s{seed} ckpt {ci + 1}/{len(ckpts)} step={row['step']} "
                f"U={row['u_composite']:.3f} V={row['v_mean']:.1f} S_flow={row['s_flow']:.2f} "
                f"({time.time() - t0:.0f}s)")

    with open(out_dir / "curves.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 平台检测：均匀段（每 checkpoint_every 步）
    uni = [r for r in rows if r["step"] % train_cfg["checkpoint_every"] == 0]
    u_curve = torch.tensor([r["u_composite"] for r in uni], dtype=torch.float64)
    plat = detect_plateau(u_curve, window=cfg["plateau_window"])
    plat_step = (
        int(uni[plat["plateau_index"] - 1]["step"]) if plat["plateau_index"] is not None else None
    )
    # 平台后 Ŝ 变化（flow 主口径）：平台点到末检查点
    s_drop = None
    if plat_step is not None:
        s_at = {r["step"]: r["s_flow"] for r in rows}
        s_drop = s_at[rows[-1]["step"]] - s_at[plat_step]

    summary = {
        "beta": beta, "seed": seed, "n_checkpoints": len(rows),
        "sigma_u_final": sigma_u,
        "sigma_u_requirement_w": cfg["plateau_window"],
        "sigma_u_requirement": 0.01 * (cfg["plateau_window"] ** 0.5) / (3 * 12 ** 0.5),
        "plateau_step": plat_step,
        "post_plateau_s_flow_change": s_drop,
        "final": rows[-1],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"完成 b{beta:g}_s{seed}: σ_Û={sigma_u:.4f}, 平台={plat_step}, "
        f"平台后ΔŜ={'%.3f' % s_drop if s_drop is not None else '未检出'}")
    return summary


def main() -> None:
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((REPO_ROOT / "configs" / "s1_train.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg["torch_threads"]))
    only_beta = None
    if "--beta" in sys.argv:
        only_beta = float(sys.argv[sys.argv.index("--beta") + 1])

    out_campaign = create_campaign_dir("description", "s1_curves", cfg["out_campaign"], cfg)
    log_path = out_campaign / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    packed, classes, values = load_data(train_cfg)
    seeds = guard.family_seeds("development", purpose="s1-curves")
    summaries = []
    for beta in train_cfg["betas"]:
        if only_beta is not None and float(beta) != only_beta:
            continue
        for seed in seeds:
            guard.assert_seed_allowed(seed, purpose="s1-curves")
            summaries.append(run_curves(cfg, train_cfg, packed, classes, values,
                                        float(beta), seed, out_campaign, log))
    (out_campaign / f"summary_{'all' if only_beta is None else f'b{only_beta:g}'}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("三曲线批次完成")


if __name__ == "__main__":
    main()
