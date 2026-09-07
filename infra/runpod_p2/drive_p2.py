"""EXP-P2 复制族 pod 端并行编排器（A4 加速；只在 RunPod CPU pod 上跑）。

把 scripts/run_exp_p2.py 的 5 种子 × (50 main + 50 abl) 块拆成分片子进程并行。
分片只改"哪个进程算哪些块",不改数值——逐轨迹代理流由 seed*1e6+ti 决定、
与块边界和并发数无关,故结果与本机顺序跑逐字一致。torch_threads 仍由冻结
配置定(10),每子进程各自 10 线程。

三相(避免竞争):
1. prep——逐种子跑 field + main 块 0(触发确定性扩池),并发到种子数。
   扩池写 traj_pool.npz;放在并行分片前做,分片时池已就绪、不再重扩。
2. shards——所有 (种子, 阶段, 块区间) 分片,并发 C 跑。main 块从 1 起
   (块 0 已在 prep),abl 块从 0 起。
3. finalize——逐种子 novelty + assemble,并发到种子数。

用法:
  python infra/runpod_p2/drive_p2.py --concurrency 24 --shard-blocks 5
产物同本机:results/confirmation/exp_p2/repl_v1/s<seed>/;末尾写 DONE_ALL。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = [sys.executable, "scripts/run_exp_p2.py", "--config", "configs/exp_p2_repl.yaml"]
SEEDS = [20, 21, 22, 23, 24]
N_BLOCKS = 50  # n_traj 1000 / block 20


def sh(args: list[str], tag: str) -> tuple[str, int, str]:
    t0 = time.time()
    p = subprocess.run(RUN + args, cwd=REPO, capture_output=True, text=True)
    dt = time.time() - t0
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    msg = f"[{tag}] rc={p.returncode} {dt:.0f}s :: {tail[0][:90]}"
    if p.returncode != 0:
        msg += f"\n  STDERR: {(p.stderr or '')[-400:]}"
    print(msg, flush=True)
    return tag, p.returncode, p.stderr or ""


def run_pool(jobs: list[tuple[list[str], str]], concurrency: int) -> list[tuple[str, int, str]]:
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(sh, a, t): t for a, t in jobs}
        for f in as_completed(futs):
            out.append(f.result())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--shard-blocks", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    a = ap.parse_args()
    seeds, S = a.seeds, a.shard_blocks
    t_start = time.time()

    # 相 1:prep(field + main 块 0),并发到种子数,避免扩池竞争
    print(f"=== 相1 prep（{len(seeds)} 种子）===", flush=True)
    prep = [([f"--seed", str(s), "--stage", "main", "--block-start", "0", "--block-end", "1"],
             f"prep-s{s}") for s in seeds]
    r1 = run_pool(prep, min(len(seeds), a.concurrency))
    bad = [t for t, rc, _ in r1 if rc != 0]
    if bad:
        print(f"✗ prep 失败: {bad}，中止", flush=True)
        sys.exit(1)

    # 相 2:分片。main 块 1..49,abl 块 0..49
    print(f"=== 相2 shards（并发 {a.concurrency}，每片 {S} 块）===", flush=True)
    jobs = []
    for s in seeds:
        for lo in range(1, N_BLOCKS, S):
            hi = min(lo + S, N_BLOCKS)
            jobs.append(([f"--seed", str(s), "--stage", "main",
                          "--block-start", str(lo), "--block-end", str(hi)],
                         f"main-s{s}-{lo}:{hi}"))
        for lo in range(0, N_BLOCKS, S):
            hi = min(lo + S, N_BLOCKS)
            jobs.append(([f"--seed", str(s), "--stage", "ablation",
                          "--block-start", str(lo), "--block-end", str(hi)],
                         f"abl-s{s}-{lo}:{hi}"))
    print(f"  共 {len(jobs)} 分片", flush=True)
    r2 = run_pool(jobs, a.concurrency)
    bad = [t for t, rc, _ in r2 if rc != 0]
    # 重试一轮失败分片(瞬时错误)
    if bad:
        print(f"  {len(bad)} 分片失败,重试一轮: {bad[:8]}...", flush=True)
        retry = [(a_, t) for a_, t in jobs if t in set(bad)]
        r2b = run_pool(retry, a.concurrency)
        bad = [t for t, rc, _ in r2b if rc != 0]
    if bad:
        print(f"✗ 分片仍失败: {bad}，中止(可重跑,已缓存块会跳过)", flush=True)
        sys.exit(2)

    # 相 3:novelty + assemble,并发到种子数
    print(f"=== 相3 finalize（novelty+assemble）===", flush=True)
    fin = [([f"--seed", str(s), "--stage", "assemble"], f"asm-s{s}") for s in seeds]
    r3 = run_pool(fin, min(len(seeds), a.concurrency))
    bad = [t for t, rc, _ in r3 if rc != 0]
    if bad:
        print(f"✗ finalize 失败: {bad}", flush=True)
        sys.exit(3)

    dt = time.time() - t_start
    done = REPO / "results/confirmation/exp_p2/repl_v1/DONE_ALL"
    done.write_text(f"all seeds done in {dt:.0f}s\n", encoding="utf-8")
    print(f"=== 全部完成，用时 {dt/60:.1f} 分钟。DONE_ALL 已写 ===", flush=True)


if __name__ == "__main__":
    main()
