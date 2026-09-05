# HyperDecept P2 Smoke 组员执行指令

本流程用于最新 `main` 分支的三源接口验收，不是论文正式训练。禁止上传原始数据、数据库、CSV、模型权重或 API key。

## 1. 更新代码与路径

```bash
git clone https://github.com/xingxiaoyang-888/Hyper-Decept.git
cd Hyper-Decept
git checkout main
git pull --ff-only origin main
export REPO=$PWD
export RAW_TWIBOT=/path/to/TwiBot-22
export RAW_MGTAB=/path/to/MGTAB
export CORE_IDS=/path/to/twibot_1000_core_ids.txt
export OUT=$REPO/output/p2_smoke_data
export RUN=$REPO/output/p2_smoke_run
```

PowerShell 示例：

```powershell
$env:REPO = (Get-Location).Path
$env:RAW_TWIBOT = "D:\path\to\TwiBot-22"
$env:RAW_MGTAB = "D:\path\to\MGTAB"
$env:CORE_IDS = "D:\path\to\twibot_1000_core_ids.txt"
$env:OUT = "$env:REPO\output\p2_smoke_data"
$env:RUN = "$env:REPO\output\p2_smoke_run"
```

## 2. TwiBot smoke bundle

原始目录至少包含 `edge.csv`、`label.csv`、`split.csv`、`user.json` 和 `tweet_0.json` 到 `tweet_8.json`。核心 ID 必须是 1000 个且无重复。

```bash
mkdir -p "$OUT/audit"
python -m data_processing.twibot_raw_audit --twibot-dir "$RAW_TWIBOT" --core-ids "$CORE_IDS" --output "$OUT/audit/twibot_audit.json" --sha256
python -m data_processing.prepare_p2_smoke_data twibot --twibot-dir "$RAW_TWIBOT" --core-ids "$CORE_IDS" --output-dir "$OUT/twibot_bundle" --expected-core-count 1000
python -m data_processing.prepare_p2_smoke_data twibot-features --bundle-dir "$OUT/twibot_bundle" --expected-core-count 1000 --model-cache "$REPO/.runtime/huggingface/hub" --embedding-model sentence-transformers/all-mpnet-base-v2
```

必须生成 `core_users.csv`、`boundary_users.csv`、`labels.csv`、`follow_edges.csv`、`actions.csv`、`relations.csv`、`posts.csv`、`node_features_26d.csv` 和 `adapter_manifest.json`。

## 3. MGTAB

原始目录必须包含 `edge_index.pt`、`edge_type.pt`、`edge_weight.pt`、`features.pt`、`labels_bot.pt`、`labels_stance.pt`。

```bash
python -m data_processing.prepare_p2_smoke_data mgtab --mgtab-dir "$RAW_MGTAB" --output-dir "$OUT/mgtab/derived" --split-seed 42 --multiedge-policy coalesce_with_count
python -m data_processing.mgtab_raw_audit --mgtab-dir "$RAW_MGTAB" --output "$OUT/audit/mgtab_audit.json" --markdown-output "$OUT/audit/mgtab_audit.md"
```

必须生成 `split_seed42.csv`、`labels.csv` 和 `adapter_manifest_seed42.json`。不要重新随机划分。

## 4. 仿真 smoke episode

准备 `leader_amplifier × 500 agents × seed 11` 和 `independent_attack × 500 agents × seed 11` 两个 episode，每个需要 DB、profiles CSV 和 node-features CSV。

```bash
mkdir -p "$OUT/simulation"
python -m data_processing.materialize_simulation_episode --db /path/to/leader_amplifier.db --profiles /path/to/leader_amplifier_profiles.csv --node-features /path/to/leader_amplifier_node_features.csv --scenario leader_amplifier --seed 11 --num-agents 500 --time-steps 50 --output-prefix "$OUT/simulation/leader_amplifier_s11"
python -m data_processing.materialize_simulation_episode --db /path/to/independent_attack.db --profiles /path/to/independent_attack_profiles.csv --node-features /path/to/independent_attack_node_features.csv --scenario independent_attack --seed 11 --num-agents 500 --time-steps 50 --output-prefix "$OUT/simulation/independent_attack_s11"
```

Smoke 阶段不要传 `--future-trace` 或 `--cutoff-time`；`next_action_targets` 必须 disabled，不能把最后一次历史动作当作未来动作。只有存在独立 cutoff 后轨迹时，才同时提供这两个参数和 `--input-is-cutoff-snapshot`。

## 5. 生成并验证 plan

```bash
python -m data_processing.prepare_p2_smoke_data plan --twibot-dir "$RAW_TWIBOT" --twibot-bundle "$OUT/twibot_bundle" --mgtab-dir "$RAW_MGTAB" --leader-manifest "$OUT/simulation/leader_amplifier_s11.manifest.json" --independent-manifest "$OUT/simulation/independent_attack_s11.manifest.json" --output "$OUT/p2_dataset_plan.json"
python -m data_processing.episode_manifest validate --input "$OUT/p2_dataset_plan.json" --require-files
```

要求 `artifact_contract_valid=true` 且 `errors=[]`。

## 6. 三源 P2 smoke

```bash
mkdir -p "$RUN"
python scripts/run_p2_smoke.py --twibot-manifest "$OUT/twibot22.episode.manifest.json" --mgtab-manifest "$OUT/mgtab.episode.manifest.json" --simulation-manifest "$OUT/simulation/leader_amplifier_s11.manifest.json" --output-dir "$RUN" --device cpu
```

有 CUDA 时将 `--device cpu` 改为 `--device cuda`。该步骤必须覆盖 `TwiBot train -> synthetic`、`MGTAB train -> synthetic`、独立 TwiBot validation 和独立 MGTAB validation。必须生成 `p2_smoke_summary.json` 和 `p2_smoke_checkpoint.pt`；summary 必须为 `status=passed`，且 train/validation 标签数为正、`synthetic_users=500`。

## 7. 验收

```bash
pytest tests -q
python -m compileall -q data_processing scripts "Character Classification" tests
git diff --check
```

预期：`180 passed`；允许 warning，不允许 failed/error。

## 8. 回传结果

只回传：`p2_smoke_summary.json`、`p2_dataset_plan.json`、`audit/twibot_audit.json`、`audit/mgtab_audit.json`、`audit/mgtab_audit.md`、`twibot_bundle/adapter_manifest.json`、`mgtab/derived/adapter_manifest_seed42.json`、两个 simulation manifest。

禁止回传 `*.db`、`*.csv`、`*.pt`、原始 TwiBot JSON、`p2_smoke_checkpoint.pt` 和任何 API key。回复时附：commit、pytest 结果、TwiBot 核心/邻居数、train/validation 标签数、MGTAB 节点及标签数、仿真用户数、summary 路径、设备和错误信息。
