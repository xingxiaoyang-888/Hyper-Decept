# HyperDecept P2 server run order

This document is the execution order after the relocatable manifest contract
has been merged into `main`. It does not replace the local data audit or the
institutional data-use requirements.

## Before starting the server

1. Prepare and audit the official TwiBot-22 and MGTAB bundles locally.
2. Generate the synthetic episode bundles and their manifests.
3. Confirm every manifest declares
   `path_contract=hyperdecept.manifest-relative.v1`.
4. Copy each complete bundle with its manifest; do not copy a manifest alone.
5. Keep raw datasets, model caches, checkpoints, and participant data outside
   Git.

## On the server

1. Clone `main`, install the pinned environment, and set `HD_ROOT`, `REPO`,
   `DATA`, `RUNS`, and `HF_HOME`.
2. Copy the complete bundles into `DATA` while preserving the directory
   layout relative to their manifests.
3. Run the manifest validation command with `--require-files`.
4. Run `pytest tests -q`, `compileall`, and `git diff --check`.
5. Run `scripts/run_p2_smoke.py` on TwiBot-22, MGTAB, and one synthetic
   episode. Stop if forward, loss, backward, checkpoint, or three-source
   loading fails.
6. Run `scripts/train_p2.py` for one fold with `--epochs 1 --max-steps 1`.
   Inspect `metrics.json`, `config.json`, `data_plan.json`, and the checkpoint.
7. Only after the one-fold dry run passes, run the planned multi-epoch folds:
   `P2_multisource_real`, held-out scenario, and all configured model seeds.
8. Archive metrics, configs, data-plan hashes, code revision, runtime details,
   and portable audit reports. Do not upload raw data or model weights to Git.

## Relocation check

The server must be able to load a manifest after its entire bundle is moved to
another directory. A manifest can be inspected as JSON, but training must read
it through `DatasetPlan.read()` or `EpisodeManifest.read()` so relative paths
are resolved from the manifest directory.
