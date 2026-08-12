#!/usr/bin/env bash
set -euo pipefail

# Final four-GPU CHI evidence window. It is idempotent and resume-safe:
# completed explanation packets and training folds are skipped, while partial
# folds resume from their exact last_checkpoint.pt.
ROOT="${HYPERTRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-p2/bin/python}"
PACKAGE="${PACKAGE:-$ROOT/runtime/p2_formal_package}"
PLAN="${PLAN:-$PACKAGE/plans/base_coordination_plan.json}"
CACHE="${CACHE:-$PACKAGE/cache/graphs_formal}"
REAL_INIT="${REAL_INIT:-$PACKAGE/output/real_pretrain_cached_preflight_honduras/best_checkpoint.pt}"
EXPLANATIONS="${EXPLANATIONS:-$PACKAGE/explanations/v5_consensus_preflight}"
ABLATION_ROOT="${ABLATION_ROOT:-$PACKAGE/output/base_chi_ablation}"
AUDIT_ROOT="${AUDIT_ROOT:-$PACKAGE/audits/chi_final_window}"
EPOCHS="${EPOCHS:-50}"
GPU_LIST="${GPU_LIST:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
SKIP_FINAL_AUDIT="${SKIP_FINAL_AUDIT:-0}"
SKIP_EXPLANATION_GATE="${SKIP_EXPLANATION_GATE:-0}"
INTRAOP_THREADS="${INTRAOP_THREADS:-4}"
INTEROP_THREADS="${INTEROP_THREADS:-1}"

read -r -a GPUS <<< "$GPU_LIST"
if ((${#GPUS[@]} == 0)); then
  echo "GPU_LIST must contain at least one CUDA device" >&2
  exit 2
fi
if ! [[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_GPU must be a positive integer" >&2
  exit 2
fi
if ! [[ "$INTRAOP_THREADS" =~ ^[1-9][0-9]*$ && "$INTEROP_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "thread counts must be positive integers" >&2
  exit 2
fi

SCENARIOS=(adaptive_evasion bridge_infiltration leader_amplifier persona_drift synchronized_boosting)
SEEDS=(7 17 27)

required=(
  "$PLAN"
  "$PACKAGE/frozen/base_formal_v5_honduras/manifest.json"
  "$PACKAGE/frozen/base_formal_v5_uae/manifest.json"
  "$PACKAGE/audits/v5_rank_consensus/honduras/rank_consensus_scores.csv.gz"
  "$PACKAGE/audits/v5_rank_consensus/uae/rank_consensus_scores.csv.gz"
  "$REAL_INIT"
)
for path in "${required[@]}"; do [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }; done
mkdir -p "$EXPLANATIONS/honduras" "$EXPLANATIONS/uae" "$ABLATION_ROOT" "$AUDIT_ROOT"

run_explanation() {
  local gpu="$1" operation="$2" source="$3"
  local output="$EXPLANATIONS/$operation"
  if [[ -s "$output/manifest.json" ]] && "$PYTHON_BIN" -c \
    "import json; assert json.load(open('$output/manifest.json'))['status'] == 'passed'" 2>/dev/null; then
    echo "[explanation:$gpu] skip passed $operation"
    return 0
  fi
  rm -f "$output/packets.jsonl" "$output/manifest.json" "$output/run.log"
  echo "[explanation:$gpu] start $operation"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$PYTHON_BIN" -u "$ROOT/scripts/generate_consensus_evidence_preflight.py" \
      --freeze-manifest "$PACKAGE/frozen/base_formal_v5_${source}/manifest.json" \
      --bundle "$PACKAGE/external/$operation" \
      --consensus-scores "$PACKAGE/audits/v5_rank_consensus/$operation/rank_consensus_scores.csv.gz" \
      --output-dir "$output" --device cuda:0 --top-n 3 > "$output/run.log" 2>&1
  echo "[explanation:$gpu] finished $operation"
}

# Explanation integrity is the hard gate for all later paper-facing outputs.
explanation_audit="$AUDIT_ROOT/explanation_preflight_audit.json"
if [[ "$SKIP_EXPLANATION_GATE" == "1" ]]; then
  "$PYTHON_BIN" -c \
    "import json; assert json.load(open('$explanation_audit'))['status'] == 'passed'" \
    || { echo "cannot skip a missing or failed explanation gate" >&2; exit 2; }
elif ((${#GPUS[@]} >= 2)); then
  run_explanation "${GPUS[0]}" uae honduras & explanation_uae=$!
  run_explanation "${GPUS[1]}" honduras uae & explanation_honduras=$!
  wait "$explanation_uae" "$explanation_honduras"
  PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/audit_consensus_evidence_preflight.py" \
    --root "$EXPLANATIONS" --output "$explanation_audit"
else
  # One-card mode is intentionally sequential: both bundles are large and
  # concurrent inference would only compete for the same device and host I/O.
  run_explanation "${GPUS[0]}" uae honduras
  run_explanation "${GPUS[0]}" honduras uae
  PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/audit_consensus_evidence_preflight.py" \
    --root "$EXPLANATIONS" --output "$explanation_audit"
fi

# Reuse the already audited scratch Lorentz suite when available. The final
# audit rechecks every run contract and hash before accepting this symlink.
baseline="$ABLATION_ROOT/lorentz_observable18"
formal_baseline="$PACKAGE/output/base_formal_v5_honduras"
if [[ ! -e "$baseline" ]] && [[ $(find "$formal_baseline" -name metrics.json 2>/dev/null | wc -l) -eq 15 ]]; then
  # The formal v5 Honduras-direction suite is the frozen M0 Lorentz baseline.
  ln -s "$formal_baseline" "$baseline"
fi

variants=(
  "lorentz_observable18|lorentz_hgt|observable18|original|real"
  "lorentz_scratch_observable18|lorentz_hgt|observable18|original|scratch"
  "euclidean_observable18|euclidean_hgt|observable18|original|scratch"
  "lorentz_no_temporal|lorentz_hgt|observable17_no_temporal|remove_temporal_information|real"
)
tasks=()
for spec in "${variants[@]}"; do
  IFS='|' read -r variant model feature intervention initialization <<< "$spec"
  if [[ -n "$VARIANT_FILTER" && "$variant" != "$VARIANT_FILTER" ]]; then
    continue
  fi
  for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_dir="$ABLATION_ROOT/$variant/$scenario/seed_$seed"
      [[ -s "$run_dir/metrics.json" ]] || tasks+=("$spec|$scenario|$seed")
    done
  done
done

run_fold() {
  local gpu="$1" task="$2"
  IFS='|' read -r variant model feature intervention initialization scenario seed <<< "$task"
  local run_dir="$ABLATION_ROOT/$variant/$scenario/seed_$seed"
  local log="$AUDIT_ROOT/${variant}_${scenario}_seed${seed}.log"
  mkdir -p "$run_dir"
  local resume=()
  [[ -s "$run_dir/last_checkpoint.pt" ]] && resume=(--resume)
  local init_args=()
  if [[ "$initialization" == "real" ]]; then
    init_args=(--real-init "$REAL_INIT")
  fi
  echo "[ablation:$gpu] start $variant $scenario seed=$seed resume=${#resume[@]}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    OMP_NUM_THREADS="$INTRAOP_THREADS" MKL_NUM_THREADS="$INTRAOP_THREADS" OPENBLAS_NUM_THREADS=1 \
    "$PYTHON_BIN" "$ROOT/scripts/train_base.py" \
      --plan "$PLAN" --output-dir "$run_dir" --held-out-scenario "$scenario" \
      --seed "$seed" --epochs "$EPOCHS" --device cuda:0 --graph-cache-dir "$CACHE" \
      --feature-contract "$feature" --model-variant "$model" \
      --graph-intervention "$intervention" \
      --intraop-threads "$INTRAOP_THREADS" --interop-threads "$INTEROP_THREADS" \
      "${init_args[@]}" "${resume[@]}" > "$log" 2>&1
  echo "[ablation:$gpu] finished $variant $scenario seed=$seed"
}

worker() {
  local gpu="$1" offset="$2" stride="$3"
  # The stride must match the actual worker count.  Keeping this dynamic is
  # essential when increasing WORKERS_PER_GPU; a fixed stride would make
  # multiple workers revisit the same folds.
  for ((index=offset; index<${#tasks[@]}; index+=stride)); do run_fold "$gpu" "${tasks[index]}"; done
}
workers=()
total_workers=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
for ((worker_index=0; worker_index<total_workers; worker_index++)); do
  gpu_index=$(( worker_index % ${#GPUS[@]} ))
  worker "${GPUS[$gpu_index]}" "$worker_index" "$total_workers" & workers[$worker_index]=$!
done
wait "${workers[@]}"

if [[ "$SKIP_FINAL_AUDIT" != "1" ]]; then
  PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/audit_chi_model_ablation.py" \
    --root "$ABLATION_ROOT" --output-dir "$AUDIT_ROOT/model_ablation" --epochs "$EPOCHS"
fi
echo "CHI final-window evidence completed: $AUDIT_ROOT"
