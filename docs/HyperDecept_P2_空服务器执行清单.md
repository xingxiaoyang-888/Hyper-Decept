# HyperDecept P2 空服务器执行清单

本清单描述从一台全新的 Linux GPU 服务器开始，到完成 P2 联合双曲训练前的全部准备。命令中的 `/srv/hyperdecept` 是服务器工作目录，可按实际磁盘位置整体替换。

## 0. 目标和边界

P2 主路线为：

```text
TwiBot-22 train + MGTAB train + DeepPersona/OASIS synthetic episodes
    -> domain-aware Intrinsic Lorentz-HGT
    -> bot detection + synthetic role/campaign/next-action auxiliary tasks
```

不要把测试集用于训练，也不要把模型权重、原始数据集或受试者数据提交到 Git。

## 1. 服务器规格

- Ubuntu 22.04/24.04，x86_64，Python 3.10 或 3.11。
- 推荐 NVIDIA GPU：24 GB 显存（RTX 4090、L40S 或同级）；最低 16 GB 显存可先跑 smoke test。
- 32-64 vCPU，128 GB RAM，500 GB SSD 起步；TwiBot-22 原始 edge/tweet 文件建议额外预留 300-500 GB。
- 网络可以通过 HTTP/HTTPS 代理访问 Hugging Face；训练阶段切换为离线模式。

检查硬件：

```bash
nvidia-smi
python3 --version
df -h /srv
```

## 2. 系统和 Python 环境

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential libsqlite3-dev libgl1
mkdir -p /srv/hyperdecept
cd /srv/hyperdecept
git clone <仓库地址> Hyper-Decept
cd Hyper-Decept
git lfs install
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

安装 PyTorch（按照服务器 CUDA 版本选择官方命令；下面是 CUDA 12.1 示例）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
pip install pandas numpy scipy scikit-learn networkx igraph
pip install torch-geometric
pip install transformers sentence-transformers spacy tqdm shap xgboost
pip install pytest ijson pyarrow
python -m spacy download en_core_web_sm
```

确认 GPU 和关键包：

```bash
python - <<'PY'
import torch, transformers, sentence_transformers, spacy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('sentence-transformers', sentence_transformers.__version__)
print('spacy', spacy.__version__)
PY
```

## 3. 目录约定

```bash
export HD_ROOT=/srv/hyperdecept
export REPO=$HD_ROOT/Hyper-Decept
export DATA=$HD_ROOT/datasets
export RUNS=$HD_ROOT/runs
export HF_HOME=$HD_ROOT/model_cache
mkdir -p "$DATA/twibot22" "$DATA/mgtab" "$DATA/synthetic" "$RUNS" "$HF_HOME"
cd "$REPO"
```

最终目录应至少包含：

```text
datasets/twibot22/{edge.csv,label.csv,split.csv,user.json,tweet_*.json,README/license}
datasets/mgtab/{users,edges,labels,official_split,README/license}
datasets/synthetic/<episode_id>/{seed_XX.db,seed_XX.csv,seed_XX.features.csv,seed_XX.labels.csv,seed_XX.event_targets.csv,seed_XX.manifest.json}
```

## 4. 下载和审计真实数据

### TwiBot-22

从数据集官方来源下载并解压到 `$DATA/twibot22`，不能只上传之前的 1000 人 DB。必须先确认：

```bash
cd "$DATA/twibot22"
ls edge.csv label.csv split.csv user.json tweet_*.json
python - <<'PY'
import pandas as pd
for f in ['label.csv','split.csv']:
    x = pd.read_csv(f, dtype=str)
    print(f, x.shape, list(x.columns))
print('edge columns:', pd.read_csv('edge.csv', nrows=0).columns.tolist())
PY
```

运行仓库审计（按实际目录调整）：

```bash
cd "$REPO"
python -m data_processing.twibot_raw_audit --twibot-dir "$DATA/twibot22" --output "$RUNS/twibot_audit.json"
```

如果该模块版本没有命令行入口，则使用项目已有的审计脚本/测试，并保存 `twibot_audit.json`；审计未通过时停止，不补造缺失字段。

### MGTAB

从官方来源下载原始用户特征、关系边、bot/human 标签和官方 train/validation/test 划分，放入 `$DATA/mgtab`。先只检查文件和列名：

```bash
find "$DATA/mgtab" -maxdepth 2 -type f -printf '%P\n'
```

MGTAB adapter 需要以原始字段为准实现；在 adapter 完成并通过测试前，不得把 MGTAB 行直接拼进 TwiBot 表格。

## 5. 生成 DeepPersona/OASIS 仿真数据

仿真数据不是模型权重。使用现有 DeepPersona/OASIS 框架和已配置的 LLM API（可经本地代理访问），每个 episode 必须同时产出：

```text
seed_XX.db
seed_XX.csv
seed_XX.features.csv
seed_XX.labels.csv
seed_XX.event_targets.csv
seed_XX.manifest.json
```

manifest 必须记录 `scenario_id`、`simulation_seed`、`num_agents`、`time_steps`、LLM model name、prompt hash、role、campaign、attack_phase、next_action、feature extractor version、graph builder version。

先跑小规模验收：

```text
2 个场景 × 1 seed × 500 agents
```

小规模通过后再生成主训练集：

```text
5 个场景 × 4 seeds × 2000 agents = 20 个主 episode
```

## 6. 创建并验证数据计划

```bash
cd "$REPO"
python -m data_processing.episode_manifest create \
  --output "$RUNS/p2_dataset_plan.json" \
  --simulation-root "$DATA/synthetic" \
  --twibot-root "$DATA/twibot22" \
  --mgtab-root "$DATA/mgtab" \
  --anchor-num-agents 2000 \
  --main-seeds 11,22,33,44

python -m data_processing.episode_manifest validate \
  --input "$RUNS/p2_dataset_plan.json"
```

仿真 artifact 全部生成后再执行严格检查：

```bash
python -m data_processing.episode_manifest validate \
  --input "$RUNS/p2_dataset_plan.json" --require-files
```

检查结果中 `errors` 必须为空。计划中的训练/验证/测试划分必须按 episode、用户和 campaign 隔离，禁止泄漏。

## 7. 下载并固定情感/心理模型权重

这一步下载的是预训练权重，不是数据集。配置代理后执行：

```bash
cd "$REPO"
export HTTP_PROXY=http://你的代理:端口
export HTTPS_PROXY=http://你的代理:端口
python -m scripts.prepare_model_cache \
  --cache-dir "$HF_HOME" \
  --proxy "$HTTP_PROXY"
```

脚本会准备：

```text
sentence-transformers/all-mpnet-base-v2
SamLowe/roberta-base-go_emotions
gpt2
facebook/bart-large-mnli
microsoft/deberta-v3-base-mnli（备用）
en_core_web_sm
```

确认 manifest 存在：

```bash
test -s "$HF_HOME/model_cache_manifest.json"
cat "$HF_HOME/model_cache_manifest.json"
```

下载成功后关闭在线访问，保证复现：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export AFG_PSYCHOLOGY_MODE=full
```

## 8. 特征和图构建 smoke test

先使用 500-agent episode 和小规模 TwiBot/MGTAB 子集，只验证管线，不记录最终论文结果：

```bash
pytest tests -q
python -m compileall -q explainability emotional_analysis data_processing \
  'Character Classification' scripts
git diff --check
```

验收要求：模型能从本地缓存加载；用户节点 ID 与标签行序一致；每个图保留 `data['user'].node_ids`；缺失角色/行动标签使用 mask，不得伪造。

## 9. P2 联合训练

`Character Classification/joint_training.py` 是训练库，负责 Domain-Aware Intrinsic Lorentz-HGT、bot/role/campaign/next-action 多任务、双曲 supervised alignment、NeighborLoader、AUROC/AUPRC/F1/Brier/ECE 和 checkpoint。正式训练前必须有一个项目 runner 将第 4-6 步生成的 episode 适配为 `EpisodeBatch`；runner 的输入必须只读取上述 manifest 和 artifact，不得隐式扫描其他目录。

runner 的正式参数固定为：

```text
real domain: TwiBot-22 train + MGTAB train
synthetic domain: 20 个主 episode 的 train 部分
validation: 两个真实数据集 val + 保留 synthetic validation episodes
test: TwiBot test、MGTAB test、LOSO synthetic test
geometry: Lorentz curvature 可学习，记录 checkpoint geometry metadata
```

每个 fold 保存：

```text
runs/p2/<fold>/checkpoint.pt
runs/p2/<fold>/metrics.json
runs/p2/<fold>/config.json
runs/p2/<fold>/data_plan.json
```

如果 runner 尚未合并，服务器上只运行到第 8 步，不要用旧的单数据集分类器冒充 P2 联合训练。

## 10. 最终验收与备份

```bash
cd "$REPO"
pytest tests -q
sha256sum "$HF_HOME/model_cache_manifest.json" > "$RUNS/model_cache_manifest.sha256"
sha256sum "$RUNS/p2_dataset_plan.json" > "$RUNS/p2_dataset_plan.sha256"
tar -czf "$RUNS/p2_run_metadata.tgz" \
  "$RUNS/p2_dataset_plan.json" "$HF_HOME/model_cache_manifest.json"
```

最终报告至少包括：数据审计结果、artifact 数量、模型 manifest、代码版本、训练 seed、GPU/CUDA/PyTorch 版本、每个 fold 的指标和数据泄漏检查结果。
