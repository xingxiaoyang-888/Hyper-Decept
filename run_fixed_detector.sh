#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$ROOT_DIR/.runtime/matplotlib"
export HF_HOME="$ROOT_DIR/.runtime/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$MPLCONFIGDIR"

if [ "$(uname -s)" = "Darwin" ] && [ -z "${DYLD_LIBRARY_PATH:-}" ]; then
  SKLEARN_LIBOMP=$(
    "$ROOT_DIR/.venv/bin/python" -c \
      'import pathlib, sklearn; print(next(pathlib.Path(sklearn.__file__).parent.glob(".dylibs/libomp.dylib"), ""))'
  )
  if [ -n "$SKLEARN_LIBOMP" ]; then
    export DYLD_LIBRARY_PATH=$(dirname "$SKLEARN_LIBOMP")
  fi
fi

exec "$ROOT_DIR/.venv/bin/python" \
  "$ROOT_DIR/Character Classification/fixed_detector.py" "$@"
