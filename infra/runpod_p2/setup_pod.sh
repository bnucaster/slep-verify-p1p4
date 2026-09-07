#!/usr/bin/env bash
# EXP-P2 复制族 pod 端一键 setup（A4 加速）。从已 clone 的仓库根运行：
#   bash infra/runpod_p2/setup_pod.sh [concurrency=24] [shard_blocks=5]
# 做：还原冻结模型 → venv + 精确同版本依赖 → 验证 v1_3 冻结 → nohup 启动
# drive_p2.py。日志 drive.log；完成写 results/confirmation/exp_p2/repl_v1/DONE_ALL。
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== repo: $(pwd)  commit $(git rev-parse --short HEAD) ==="

echo "=== [1/5] 还原 5 个冻结模型（base64 bundle）==="
base64 -d infra/runpod_p2/models_bundle.tar.gz.b64 | tar xzf -
ls results/confirmation/s2_train/s2p_repl_v1/s2*/checkpoints/ckpt_020000.pt | wc -l | xargs echo "  检查点数(应为5):"

echo "=== [2/5] uv + py3.13 venv + 精确同版本依赖（numpy 2.5.2 需 py3.12+，本机 py3.13）==="
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy==2.5.2 scipy==1.18.1 scikit-learn==1.9.0 pyyaml
. .venv/bin/activate
export PYTHONPATH="$PWD/src"

echo "=== [3/5] 版本校验（须与本机一致以保数值可比）==="
python - <<'PY'
import torch, numpy, scipy, sklearn
print("  torch", torch.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| sklearn", sklearn.__version__)
assert torch.__version__.startswith("2.13.0"), "torch 版本不符"
PY

echo "=== [4/5] 协议校验：v1_3 冻结 + 复制族守卫可放行 ==="
python - <<'PY'
import json
d = json.load(open("docs/freeze_status.json"))
assert d["v1_3"]["frozen"], "v1_3 未冻结,复制族运行将被拒"
print("  v1_3 frozen OK, anchor", d["v1_3"]["anchor_commit"][:12])
PY
echo "  vCPU: $(nproc)"

echo "=== [5/5] nohup 启动 drive_p2 ==="
CONC="${1:-24}"; SHARD="${2:-5}"
export PYTHONPATH="$PWD/src"
nohup python infra/runpod_p2/drive_p2.py --concurrency "$CONC" --shard-blocks "$SHARD" > drive.log 2>&1 &
echo "  drive_p2 PID $!  并发 $CONC  每片 $SHARD 块  日志 drive.log"
echo "  监控: tail -f drive.log ; 完成标志文件 results/confirmation/exp_p2/repl_v1/DONE_ALL"
