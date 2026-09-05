#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runtime/p2_formal_package/audits

if ! pgrep -f 'data_processing/materialize_crypto_full.py' >/dev/null; then
  rm -rf runtime/p2_formal_package/bundles/coordination/crypto_campaign_full
  nohup .venv-sim/bin/python data_processing/materialize_crypto_full.py \
    --source raw_datasets/coordination/crypto_campaign/extracted_full/crypto_full \
    --output runtime/p2_formal_package/bundles/coordination/crypto_campaign_full \
    --chunksize 100000 --edges-per-campaign 32 \
    > runtime/p2_formal_package/audits/crypto_full_materialize.stdout.json \
    2> runtime/p2_formal_package/audits/crypto_full_materialize.stderr.log \
    </dev/null &
  echo $! > runtime/p2_formal_package/audits/crypto_full_materialize.pid
fi

rm -rf runtime/p2_formal_package/pretrain_calibration_uk
CUDA_VISIBLE_DEVICES=0 nohup .venv-sim/bin/python scripts/pretrain_lorentz_graph.py \
  --bundle-root runtime/p2_formal_package/bundles/coordination/uk2019 \
  --output-dir runtime/p2_formal_package/pretrain_calibration_uk \
  --device cuda --epochs 3 --max-steps 3 --max-edges 8192 \
  > runtime/p2_formal_package/audits/pretrain_calibration_uk.log 2>&1 \
  </dev/null &
echo $! > runtime/p2_formal_package/audits/pretrain_calibration_uk.pid

rm -rf runtime/p2_formal_package/pretrain_calibration_crypto_sample
CUDA_VISIBLE_DEVICES=1 nohup .venv-sim/bin/python scripts/pretrain_lorentz_graph.py \
  --bundle-root runtime/p2_formal_package/bundles/coordination/crypto_campaign \
  --output-dir runtime/p2_formal_package/pretrain_calibration_crypto_sample \
  --device cuda --epochs 3 --max-steps 3 --max-edges 8192 \
  > runtime/p2_formal_package/audits/pretrain_calibration_crypto.log 2>&1 \
  </dev/null &
echo $! > runtime/p2_formal_package/audits/pretrain_calibration_crypto.pid

