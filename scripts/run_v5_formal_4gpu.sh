#!/usr/bin/env bash
set -euo pipefail

# Four-GPU dispatcher for the formal v5 LOSO runs.  Each worker owns one GPU
# and processes its assigned scenario/seed runs sequentially.  Every run has
# an independent last_checkpoint.pt, so a killed window can be resumed safely.
ROOT="${HYPERTRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-p2/bin/python}"
PACKAGE="${PACKAGE:-$ROOT/runtime/p2_formal_package}"
PLAN="${PLAN:-$PACKAGE/plans/base_coordination_plan.json}"
CACHE="${CACHE:-$PACKAGE/cache/graphs_formal}"
REAL_INIT="${REAL_INIT:-$PACKAGE/output/real_pretrain_cached_preflight_honduras/best_checkpoint.pt}"
OUTPUT="${OUTPUT:-$PACKAGE/output/base_formal_v5_honduras}"
AUDITS="${AUDITS:-$PACKAGE/audits/base_formal_v5_honduras}"
EPOCHS="${EPOCHS:-50}"
SEEDS=(7 17 27)
SCENARIOS=(adaptive_evasion bridge_infiltration leader_amplifier persona_drift synchronized_boosting)

[[ -s "$PLAN" ]] || { echo "missing plan: $PLAN" >&2; exit 2; }
[[ -s "$REAL_INIT" ]] || { echo "missing real warm-start: $REAL_INIT" >&2; exit 2; }
mkdir -p "$OUTPUT" "$AUDITS"

run_one() {
  local gpu="$1" scenario="$2" seed="$3"
  local run_dir="$OUTPUT/$scenario/seed_$seed"
  local log="$AUDITS/${scenario}_seed${seed}.log"
  mkdir -p "$run_dir"
  if [[ -s "$run_dir/metrics.json" ]]; then
    echo "[$gpu] skip completed $scenario seed=$seed"
    return 0
  fi
  local resume=()
  if [[ -s "$run_dir/last_checkpoint.pt" ]]; then resume=(--resume); fi
  echo "[$gpu] start $scenario seed=$seed resume=${#resume[@]}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$PYTHON_BIN" "$ROOT/scripts/train_base.py" \
      --plan "$PLAN" \
      --output-dir "$run_dir" \
      --held-out-scenario "$scenario" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --device cuda:0 \
      --graph-cache-dir "$CACHE" \
      --real-init "$REAL_INIT" \
      "${resume[@]}" > "$log" 2>&1
  echo "[$gpu] finished $scenario seed=$seed"
}

worker() {
  local gpu="$1"; shift
  while (($#)); do
    local scenario="$1" seed="$2"; shift 2
    run_one "$gpu" "$scenario" "$seed"
  done
}

# Round-robin assignment keeps the four workers busy while preserving one
# process per GPU.  Do not use an unbounded xargs fan-out: each graph episode
# is large and concurrent CSV/cache reads would increase contention.
tasks=()
for scenario in "${SCENARIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do tasks+=("$scenario" "$seed"); done
done
for gpu in 0 1 2 3; do
  args=()
  for ((i=gpu*2; i<${#tasks[@]}; i+=8)); do
    args+=("${tasks[i]}" "${tasks[i+1]}")
  done
  worker "$gpu" "${args[@]}" &
done
wait
echo "all v5 formal workers completed"
