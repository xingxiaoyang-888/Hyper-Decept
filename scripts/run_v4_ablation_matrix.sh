#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYPERTRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PACKAGE="${PACKAGE:-$ROOT/runtime/p2_formal_package}"
PLAN="${PLAN:-$PACKAGE/plans/base_coordination_plan.json}"
CACHE="${CACHE:-$PACKAGE/cache/graphs_formal}"
OUTPUT="${OUTPUT:-$PACKAGE/output/base_v4_ablation}"
AUDITS="${AUDITS:-$PACKAGE/audits/base_v4_ablation}"
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

# Keep all comparisons on the same 20-episode DatasetPlan and only vary one
# explanatory factor at a time. The first entry is the no-message-passing
# shortcut baseline; the last is the frozen v3-style 26D reference.
variants=(
  "mlp_observable18:mlp:observable18:original:none"
  "euclidean_hgt_observable18:euclidean_hgt:observable18:original:none"
  "lorentz_random_observable18:lorentz_hgt:observable18:original:none"
  "lorentz_uk_observable18:lorentz_hgt:observable18:original:uk"
  "lorentz_uk_observable18_edge_shuffle:lorentz_hgt:observable18:degree_preserving_edge_shuffle:uk"
)

mkdir -p "$OUTPUT" "$AUDITS"
for spec in "${variants[@]}"; do
  IFS=: read -r name model_variant feature_contract graph_intervention init <<< "$spec"
  for scenario in "${scenarios[@]}"; do
    for seed in "${seeds[@]}"; do
      run_dir="$OUTPUT/$name/$scenario/seed_$seed"
      log="$AUDITS/${name}_${scenario}_seed${seed}.log"
      if [[ -s "$run_dir/metrics.json" ]]; then
        echo "skip completed $name $scenario seed=$seed"
        continue
      fi
      mkdir -p "$run_dir"
      resume=()
      if [[ -s "$run_dir/last_checkpoint.pt" ]]; then
        resume=(--resume)
      fi
      uk_args=()
      if [[ "$init" == "uk" ]]; then
        [[ -s "$UK_INIT" ]] || { echo "missing UK init: $UK_INIT" >&2; exit 2; }
        uk_args=(--uk-init "$UK_INIT")
      fi
      echo "start $name $scenario seed=$seed resume=${#resume[@]}"
      command=(
        "$PYTHON_BIN" "$ROOT/scripts/train_base.py"
        --plan "$PLAN"
        --output-dir "$run_dir"
        --held-out-scenario "$scenario"
        --seed "$seed"
        --epochs "$EPOCHS"
        --device "$DEVICE"
        --graph-cache-dir "$CACHE"
        --feature-contract "$feature_contract"
        --model-variant "$model_variant"
        --graph-intervention "$graph_intervention"
        "${uk_args[@]}"
        "${resume[@]}"
      )
      "${command[@]}" > "$log" 2>&1
    done
  done
done

echo "v4 ablation matrix complete"
