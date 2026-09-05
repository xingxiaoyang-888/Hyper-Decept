#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYPERTRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PACKAGE="${PACKAGE:-$ROOT/runtime/p2_formal_package}"
UK_BUNDLE="$PACKAGE/bundles/coordination/uk2019"
UK_OUTPUT="$PACKAGE/pretrain_formal_uk_v1"
UK_LOG="$PACKAGE/audits/pretrain_formal_uk_v1.log"

if [[ ! -s "$UK_OUTPUT/metrics.json" ]]; then
  mkdir -p "$UK_OUTPUT" "$PACKAGE/audits"
  resume=()
  if [[ -s "$UK_OUTPUT/last_checkpoint.pt" ]]; then
    resume=(--resume)
  fi
  "$PYTHON_BIN" "$ROOT/scripts/pretrain_lorentz_graph.py" \
    --bundle-root "$UK_BUNDLE" \
    --output-dir "$UK_OUTPUT" \
    --device cuda:0 \
    --epochs 50 \
    --max-edges 276775 \
    "${resume[@]}" \
    > "$UK_LOG" 2>&1
fi

UK_INIT="$UK_OUTPUT/last_checkpoint.pt" "$ROOT/scripts/run_base_loso.sh"
