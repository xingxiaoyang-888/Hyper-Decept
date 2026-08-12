#!/usr/bin/env bash
set -euo pipefail
ROOT="${HYPERTRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-p2/bin/python}"
PACKAGE="${PACKAGE:-$ROOT/runtime/p2_formal_package}"
FREEZE="${FREEZE:-$PACKAGE/frozen/base_formal_v5_honduras}"
OUT="${OUT:-$PACKAGE/audits/v5_external_matrix}"
BUNDLES="${BUNDLES:-honduras uae}"
mkdir -p "$OUT"
[[ -s "$FREEZE/manifest.json" ]] || { echo "missing freeze manifest" >&2; exit 2; }
mapfile -t checkpoints < <(find "$FREEZE" -mindepth 3 -maxdepth 3 -name best_checkpoint.pt | sort)
run_one() {
  local gpu="$1" checkpoint="$2" bundle="$3"
  local rel="${checkpoint#$FREEZE/}"; rel="${rel%.pt}"; rel="${rel//\//_}"
  local output="$OUT/${rel}_${bundle}.json"
  [[ -s "$output" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$PYTHON_BIN" "$ROOT/scripts/run_frozen_external_io.py" --checkpoint "$checkpoint" \
      --bundle "$PACKAGE/external/$bundle" --output "$output" --device cuda:0 > "$output.log" 2>&1
}
tasks=()
for checkpoint in "${checkpoints[@]}"; do
  for bundle in $BUNDLES; do tasks+=("$checkpoint" "$bundle"); done
done
worker() {
  local gpu="$1"; shift
  while (($#)); do run_one "$gpu" "$1" "$2"; shift 2; done
}
for gpu in 0 1 2 3; do
  args=()
  for ((i=gpu*2; i<${#tasks[@]}; i+=8)); do args+=("${tasks[i]}" "${tasks[i+1]}"); done
  worker "$gpu" "${args[@]}" &
done
wait
echo "external matrix completed"
