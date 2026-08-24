"""S1 正式开发族训练（M2 描述，任务五 5b）。

β∈{1,4} × 开发族 5 种子 = 10 个 run，可断点续跑（子 run 目录含 DONE
标记即跳过）。每 run：密集检查点（前期对数间隔、后期每 2k 步，plan_v2
第 2 节）+ 逐段损失日志 + 末检查点因子探针 sanity（形状分类精度与
posX 回归 R²，只记录不判定；正式 Û 曲线在任务六按 CAL-P1 数据条件
配评估批量后计算）。

产物 results/description/s1_train/<campaign>/：config.json、逐调用
meta_*.json、b<β>_s<种子>/（losses.csv、probe_sanity.json、DONE、
checkpoints/ 为 git 忽略）。

用法：.venv/Scripts/python.exe scripts/run_s1_train.py [--smoke]
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
from slep.systems.s1_beta_vae import S1BetaVAE
from slep.utils.runs import REPO_ROOT, create_campaign_dir

CONFIG_FILE = REPO_ROOT / "configs" / "s1_train.yaml"


def checkpoint_steps(cfg: dict) -> list[int]:
    steps = set()
    k = 1
    while k <= cfg["checkpoint_log2_until"]:
        steps.add(k)
        k *= 2
    steps.update(range(cfg["checkpoint_every"], cfg["train_steps"] + 1, cfg["checkpoint_every"]))
    steps.add(cfg["train_steps"])
    return sorted(steps)


def load_packed(cfg: dict) -> np.ndarray:
    cache = REPO_ROOT / cfg["packed_cache"]
    if cache.exists():
        return np.load(cache)
    raw = np.load(REPO_ROOT / cfg["data_file"])["imgs"]
    packed = np.packbits(raw.reshape(raw.shape[0], -1), axis=1)
    np.save(cache, packed)
    return packed


def batch_from_packed(packed: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    imgs = np.unpackbits(packed[idx], axis=1).astype(np.float32)
    return torch.from_numpy(imgs).reshape(-1, 1, 64, 64)


def probe_sanity(model: S1BetaVAE, packed: np.ndarray, cfg: dict, rng) -> dict:
    """末检查点因子探针 sanity：形状分类精度 + posX 岭回归 R²。"""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score

    lat = np.load(REPO_ROOT / cfg["data_file"])
    classes = lat["latents_classes"]  # (N, 6)：color/shape/scale/orient/posX/posY
    values = lat["latents_values"]
    n = cfg["probe_sanity_n"]
    idx = rng.integers(0, packed.shape[0], size=n)
    with torch.no_grad():
        z = []
        for i in range(0, n, 512):
            mu, _ = model.encoder(batch_from_packed(packed, idx[i : i + 512]))
            z.append(mu)
        z = torch.cat(z).numpy()
    split = int(0.8 * n)
    shape_y = classes[idx, 1]
    clf = LogisticRegression(max_iter=500).fit(z[:split], shape_y[:split])
    shape_acc = float(clf.score(z[split:], shape_y[split:]))
    posx_y = values[idx, 4]
    reg = Ridge().fit(z[:split], posx_y[:split])
    posx_r2 = float(r2_score(posx_y[split:], reg.predict(z[split:])))
    return {"shape_acc_holdout": shape_acc, "posx_r2_holdout": posx_r2, "n": n}


def train_one(cfg: dict, beta: float, seed: int, packed: np.ndarray, campaign, log) -> None:
    run_dir = campaign / f"b{beta:g}_s{seed}"
    if (run_dir / "DONE").exists():
        log(f"跳过已完成 run b{beta:g}_s{seed}")
        return
    run_dir.mkdir(exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = S1BetaVAE(cfg["latent_dim"], cfg["channels"], cfg["fc_dim"], cfg["sigma_dec"], beta)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    ckpts = set(checkpoint_steps(cfg))

    t0 = time.time()
    with open(run_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "recon", "kl"])
        for step in range(1, cfg["train_steps"] + 1):
            idx = rng.integers(0, packed.shape[0], size=cfg["batch_size"])
            out = model.loss(batch_from_packed(packed, idx))
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            if step % cfg["log_every"] == 0 or step == 1:
                writer.writerow(
                    [step, f"{out['total'].item():.4f}", f"{out['recon'].item():.4f}",
                     f"{out['kl'].item():.4f}"]
                )
            if step in ckpts:
                torch.save(
                    {"model": model.state_dict(), "step": step, "beta": beta, "seed": seed,
                     "kl_per_dim": out["kl_per_dim"].tolist()},
                    ckpt_dir / f"ckpt_{step:06d}.pt",
                )
            if step % 5000 == 0:
                log(
                    f"  b{beta:g}_s{seed} {step}/{cfg['train_steps']} "
                    f"total={out['total'].item():.1f} kl={out['kl'].item():.2f} "
                    f"({time.time() - t0:.0f}s)"
                )

    sanity = probe_sanity(model, packed, cfg, rng)
    (run_dir / "probe_sanity.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    wall = time.time() - t0
    (run_dir / "DONE").write_text(f"wall_seconds={wall:.0f}\n", encoding="utf-8")
    log(
        f"完成 b{beta:g}_s{seed}：{wall:.0f}s，探针 sanity 形状精度 "
        f"{sanity['shape_acc_holdout']:.3f}，posX R² {sanity['posx_r2_holdout']:.3f}"
    )


def main() -> None:
    config_file = CONFIG_FILE
    if "--config" in sys.argv:
        config_file = REPO_ROOT / sys.argv[sys.argv.index("--config") + 1]
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    smoke = "--smoke" in sys.argv
    if smoke:
        cfg = {**cfg, "campaign": "smoke_v1", "train_steps": 60, "checkpoint_log2_until": 16,
               "checkpoint_every": 30, "probe_sanity_n": 400, "log_every": 20}
    torch.set_num_threads(int(cfg["torch_threads"]))
    campaign = create_campaign_dir("description", "s1_train", cfg["campaign"], cfg)
    log_path = campaign / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    seeds = guard.family_seeds(cfg["seed_family"], purpose="s1-train-dev")
    if smoke:
        seeds = seeds[:1]
    log(f"campaign: {campaign}，betas={cfg['betas']}，seeds={seeds}")
    packed = load_packed(cfg)
    for beta in cfg["betas"]:
        for seed in seeds:
            guard.assert_seed_allowed(seed, purpose=f"s1-train-b{beta:g}")
            train_one(cfg, float(beta), seed, packed, campaign, log)
    log("S1 战役完成")


if __name__ == "__main__":
    main()
