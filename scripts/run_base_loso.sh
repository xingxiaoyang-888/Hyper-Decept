#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYPERTRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PACKAGE="${PACKAGE:-$ROOT/runtime/p2_formal_package}"
PLAN="${PLAN:-$PACKAGE/plans/base_coordination_plan.json}"
CACHE="${CACHE:-$PACKAGE/cache/graphs_formal}"
OUTPUT="${OUTPUT:-$PACKAGE/output/base_formal_v3}"
AUDITS="${AUDITS:-$PACKAGE/audits}"
EPOCHS="${EPOCHS:-50}"
DEVICE="${DEVICE:-cuda:0}"
UK_INIT="${UK_INIT:-$PACKAGE/pretrain_formal_uk_v1/last_checkpoint.pt}"

scenarios=(
  adaptive_evasion
  bridge_infiltration
  leader_amplifier
  persona_drift
  synchronized_boosting
)
seeds=(7 17 27)

mkdir -p "$OUTPUT" "$AUDITS"
if [[ ! -s "$UK_INIT" ]]; then
  echo "missing formal UK initialization checkpoint: $UK_INIT" >&2
  exit 2
fi
for scenario in "${scenarios[@]}"; do
  for seed in "${seeds[@]}"; do
    run_dir="$OUTPUT/$scenario/seed_$seed"
    log="$AUDITS/base_formal_v3_${scenario}_seed${seed}.log"
    if [[ -s "$run_dir/metrics.json" ]]; then
      echo "skip completed $scenario seed=$seed"
      continue
    fi
    mkdir -p "$run_dir"
    resume=()
    if [[ -s "$run_dir/last_checkpoint.pt" ]]; then
      resume=(--resume)
    fi
    echo "start $scenario seed=$seed resume=${#resume[@]}"
    "$PYTHON_BIN" "$ROOT/scripts/train_base.py" \
      --plan "$PLAN" \
      --output-dir "$run_dir" \
      --held-out-scenario "$scenario" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --graph-cache-dir "$CACHE" \
      --uk-init "$UK_INIT" \
      "${resume[@]}" \
      > "$log" 2>&1
  done
done
