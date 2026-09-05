#!/usr/bin/env bash
set -euo pipefail
ROOT="/xingxiaoyang/HyperTrace"
cd "$ROOT"
LOGDIR="runtime/p2_formal_package/audits/base_v4_ablation/shards"
mkdir -p "$LOGDIR"
for shard in $(seq 0 11); do
  gpu=$((shard % 4))
  nohup env GPU="$gpu" SHARD="$shard" SHARDS=12 EPOCHS=50 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    HYPERTRACE_ROOT="$ROOT" bash scripts/run_v4_shard.sh \
    >"$LOGDIR/shard_${shard}.log" 2>&1 < /dev/null &
done
sleep 8
echo "--- workers ---"
pgrep -af 'train_base.py|run_v4_shard' || true
echo "--- gpu ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- metrics ---"
find runtime/p2_formal_package/output/base_v4_ablation -name metrics.json | wc -l
