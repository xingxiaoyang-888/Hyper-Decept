# HyperDecept P2 空服务器执行清单

本清单对应 `main` 分支的 P2 多源 Lorentz-HGT 管线。Smoke 仅验证数据契约、
三源加载、前向、损失、反向和 checkpoint，不产生可写入论文的实验指标。

## 0. 执行边界

- Smoke 数据包约 140 MiB，不包含完整 TwiBot-22 原始数据。
- Smoke DatasetPlan 只有 2 个场景、每场景 1 个 seed，只能运行
  `scripts/run_p2_smoke.py`。
- 不得使用 Smoke DatasetPlan 运行 `scripts/train_p2.py`。LOSO 划分后它没有
  足够的合成训练、验证和测试 episode。
- 当前训练读取预计算特征，不调用外部 LLM API，也不下载 Hugging Face 权重。
- 原始数据、数据包、模型缓存、checkpoint 和运行日志均不得提交到 Git。

## 1. 推荐服务器和镜像

Smoke 最低配置：

- Ubuntu 22.04 x86_64；
- Python 3.10；
- 8 vCPU、32 GiB RAM；
- NVIDIA GPU 24 GiB 显存（也可先在 CPU 验证，但会更慢）；
- 100 GiB 可用磁盘。

正式 15 折实验建议：

- 1 张 48 GiB GPU，或先用 24 GiB GPU 做显存基准；
- 16 vCPU、64 GiB RAM；
- 500 GiB 以上独立数据盘；
- NVIDIA 驱动支持 CUDA 12.4（Linux driver 550 系列或更新版本）。

不要在实验中途升级 Python、PyTorch、PyG 或 CUDA。若云平台已有兼容的
PyTorch 镜像，可保留系统驱动，但仍应创建项目独立虚拟环境。

## 2. 系统初始化

以下命令在 Ubuntu 22.04 执行。没有 `sudo` 权限时，让平台镜像预装 Git、
Python 3.10 和 `venv`。

```bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git git-lfs unzip rsync tmux build-essential \
  python3.10 python3.10-venv python3-pip

sudo mkdir -p /data/hyperdecept/{data,runs,cache}
sudo chown -R "$USER":"$USER" /data/hyperdecept

export HD_ROOT=/data/hyperdecept
export REPO="$HD_ROOT/repo"
export DATA="$HD_ROOT/data/p2_smoke_package"
export RUNS="$HD_ROOT/runs"
export HF_HOME="$HD_ROOT/cache/huggingface"
mkdir -p "$RUNS" "$HF_HOME"
```

建议把上述五个 `export` 写入服务器用户自己的 `~/.bashrc`。不要写入 Git。

## 3. 获取代码并锁定环境

```bash
set -euo pipefail

git clone --branch main --single-branch \
  https://github.com/xingxiaoyang-888/Hyper-Decept.git "$REPO"
cd "$REPO"
git pull --ff-only origin main
git status --short --branch
git rev-parse HEAD

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1

# CUDA 12.4 wheel；不要同时安装 requirements.txt 中未固定版本的 torch。
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-p2.txt
```

确认 GPU 和关键版本：

```bash
nvidia-smi
python - <<'PY'
import sys
import numpy, pandas, sklearn, torch, torch_geometric

print("python=", sys.version)
print("torch=", torch.__version__)
print("torch_cuda_runtime=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("numpy=", numpy.__version__)
print("pandas=", pandas.__version__)
print("sklearn=", sklearn.__version__)
print("pyg=", torch_geometric.__version__)
assert torch.cuda.is_available(), "GPU Smoke 要求 PyTorch 能识别 CUDA"
PY
```

如果云平台驱动不支持 CUDA 12.4，不要自行混装系统 CUDA。停止并记录
`nvidia-smi` 输出后再选择与驱动匹配的官方 PyTorch wheel。

## 4. 上传并解压 Smoke 数据包

组员应上传 `hyperdecept_p2_smoke_package.zip`。解压后必须正好存在：

```text
$DATA/
├── bundles/
├── plans/
├── audits/
└── checksums/
```

示例命令：

```bash
set -euo pipefail
mkdir -p "$HD_ROOT/data"
unzip -q "$HD_ROOT/hyperdecept_p2_smoke_package.zip" -d "$HD_ROOT/data"

test -d "$DATA/bundles"
test -d "$DATA/plans"
test -d "$DATA/audits"
test -d "$DATA/checksums"
test -f "$DATA/plans/p2_smoke_dataset_plan.json"
test -f "$DATA/checksums/checksums.sha256"
```

如果压缩包额外嵌套了一层目录，只调整 `DATA`，不要移动或重命名内部文件。

## 5. 校验 SHA-256 和相对路径契约

```bash
set -euo pipefail
cd "$DATA"
sha256sum -c checksums/checksums.sha256 \
  | tee checksums/server_checksum_verification.txt
```

必须是 44 个受检文件全部 `OK`。出现一个 `FAILED` 就停止，不得继续训练。

使用项目解析器验证 DatasetPlan：

```bash
cd "$REPO"
source .venv/bin/activate
python -m data_processing.episode_manifest validate \
  --input "$DATA/plans/p2_smoke_dataset_plan.json" \
  --require-files \
  | tee "$RUNS/smoke_dataset_plan_validation.json"
```

再确认解析后的所有路径都位于数据包中：

```bash
python - "$DATA" <<'PY'
from pathlib import Path
import sys
from data_processing.episode_manifest import DatasetPlan

root = Path(sys.argv[1]).resolve()
plan = DatasetPlan.read(root / "plans/p2_smoke_dataset_plan.json")
checked = 0
for episode in plan.episodes:
    for name, value in {"source": episode.source_path, **episode.artifacts}.items():
        path = Path(value).resolve()
        assert path.is_relative_to(root), (episode.episode_id, name, path)
        assert path.exists(), (episode.episode_id, name, path)
        checked += 1
print({"status": "passed", "episodes": len(plan.episodes), "paths": checked})
PY
```

## 6. 代码和依赖验收

运行 P2 相关测试，不要求安装白盒 UI、SHAP、XGBoost、ChromaDB 等非训练依赖：

```bash
cd "$REPO"
source .venv/bin/activate

pytest -q \
  tests/test_episode_manifest.py \
  tests/test_twibot22_raw_adapter.py \
  tests/test_mgtab_adapter.py \
  tests/test_lorentz_hgt.py \
  tests/test_joint_training.py \
  tests/test_p2_smoke_report.py \
  tests/test_simulation_episode_materialization.py \
  tests/test_train_p2.py \
  tests/test_simulation_smoke_wiring.py::test_p2_smoke_runner_declares_three_sources_and_separate_validation \
  | tee "$RUNS/p2_server_tests.txt"

python -m compileall -q \
  scripts data_processing tests \
  "Character Classification/joint_training.py" \
  "Character Classification/lorentz_hgt.py"

git diff --check
git status --short --branch
```

验收条件：pytest、compileall、`git diff --check` 全部返回 0，并且工作区无代码修改。

## 7. 执行三源 P2 Smoke

先在 `tmux` 中运行，避免 SSH 断开终止任务：

```bash
tmux new -s hyperdecept-smoke
```

在 tmux 会话内执行：

```bash
set -euo pipefail
cd "$REPO"
source .venv/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export CUDA_VISIBLE_DEVICES=0

SMOKE_RUN="$RUNS/p2_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SMOKE_RUN/output" "$SMOKE_RUN/reports"

python scripts/run_p2_smoke.py \
  --twibot-manifest "$DATA/plans/twibot22.episode.manifest.json" \
  --mgtab-manifest "$DATA/plans/mgtab.episode.manifest.json" \
  --simulation-manifest \
    "$DATA/bundles/synthetic/leader_amplifier/n500/seed_11/seed_11.manifest.json" \
  --output-dir "$SMOKE_RUN/output" \
  --report-dir "$SMOKE_RUN/reports" \
  --device cuda:0 \
  2>&1 | tee "$SMOKE_RUN/run.log"

test -s "$SMOKE_RUN/output/p2_smoke_checkpoint.pt"
test -s "$SMOKE_RUN/output/p2_smoke_summary.json"

python - "$SMOKE_RUN/output/p2_smoke_summary.json" <<'PY'
import json, math, sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "passed"
assert summary["checkpoint_bytes"] > 0
for domain, metrics in summary["train_metrics_smoke_only"].items():
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), (domain, key, value)
print({
    "status": summary["status"],
    "twibot_users": summary["twibot_users"],
    "mgtab_users": summary["mgtab_users"],
    "synthetic_users": summary["synthetic_users"],
    "checkpoint_bytes": summary["checkpoint_bytes"],
})
PY

python -m pip freeze > "$SMOKE_RUN/pip-freeze.txt"
git rev-parse HEAD > "$SMOKE_RUN/git-commit.txt"
nvidia-smi -q > "$SMOKE_RUN/nvidia-smi.txt"
```

Smoke 通过标准：

- 三种数据均成功加载；
- 两次联合训练 step 和两次真实验证均完成；
- 所有损失为有限数值；
- checkpoint、summary、JSON/Markdown 审计报告均生成；
- 日志无 traceback、CUDA OOM 或 checksum/path 错误。

Smoke 指标只能标注为 `smoke_only`，不得进入论文表格。

## 8. Smoke 后的正式数据准备（后续阶段，本次不要执行）

正式训练不能复用当前两条 500-Agent Smoke episode。推荐的不完全因子设计为：

- 真实数据：正式 TwiBot-22 可审计物化 bundle；完整标准 MGTAB 10,199 用户；
- 主模拟集：5 场景 × 4 seeds × 2,000 agents = 20 episodes、40,000 agent 实例；
- 规模泛化集：2 个代表场景 × 3 个规模（500/1,000/5,000）× 3 seeds
  = 18 episodes、39,000 agent 实例；
- 合成数据合计 38 episodes、79,000 agent 实例；规模泛化集只用于测试，
  不参与模型参数训练。

正式场景：

```text
leader_amplifier
bridge_infiltration
synchronized_boosting
persona_drift
adaptive_evasion
```

正式数据在本地或生成服务器物化并审计后，放入新的
`$HD_ROOT/data/p2_formal_package`，不得覆盖 Smoke 包。每个 episode 必须包含
DB、profiles、26 维特征、标签、event targets 和相对路径 manifest。

以下命令描述当前正式 DatasetPlan 的目标接口。只有当正式 TwiBot bundle 的
26 维非占位特征、原始关系物化文件、官方标签/划分和 adapter manifest 均已
生成并通过审计后才可执行，不能直接把 Smoke bundle 改名后代入：

```bash
python -m data_processing.episode_manifest create \
  --output "$HD_ROOT/data/p2_formal_package/plans/p2_formal_dataset_plan.json" \
  --simulation-root "$HD_ROOT/data/p2_formal_package/bundles/synthetic" \
  --twibot-root "$HD_ROOT/data/p2_formal_package/bundles/twibot22" \
  --mgtab-root "$HD_ROOT/data/p2_formal_package/bundles/mgtab" \
  --scenarios \
    leader_amplifier,bridge_infiltration,synchronized_boosting,persona_drift,adaptive_evasion \
  --main-seeds 11,22,33,44 \
  --anchor-num-agents 2000 \
  --scale-scenarios leader_amplifier,adaptive_evasion \
  --scale-sizes 500,1000,5000 \
  --scale-seeds 101,102,103 \
  --time-steps 50

python -m data_processing.episode_manifest validate \
  --input "$HD_ROOT/data/p2_formal_package/plans/p2_formal_dataset_plan.json" \
  --require-files
```

正式数据包尚未生成时，上述 `--require-files` 失败是正确行为，不得用空文件
或补造标签绕过。

## 9. 正式训练的 dry-run 与主实验

正式 DatasetPlan 验证通过后，先只跑一折、一步：

```bash
FORMAL_PLAN="$HD_ROOT/data/p2_formal_package/plans/p2_formal_dataset_plan.json"
DRY_RUN="$RUNS/p2_formal_dry_run"

python scripts/train_p2.py \
  --plan "$FORMAL_PLAN" \
  --output-dir "$DRY_RUN" \
  --protocol P2_multisource_real \
  --held-out-scenario leader_amplifier \
  --seed 7 \
  --epochs 1 \
  --max-steps 1 \
  --device cuda:0

test -s "$DRY_RUN/checkpoint.pt"
test -s "$DRY_RUN/metrics.json"
test -s "$DRY_RUN/config.json"
test -s "$DRY_RUN/data_plan.json"
```

dry-run 通过并记录显存峰值后，执行 5 个 LOSO 场景 × 3 个模型 seed，共 15 个
主实验。以下是基线命令；正式 epoch 数应根据预注册/实验方案固定，不能看到
测试集结果后再修改。

```bash
FORMAL_PLAN="$HD_ROOT/data/p2_formal_package/plans/p2_formal_dataset_plan.json"
SCENARIOS=(
  leader_amplifier
  bridge_infiltration
  synchronized_boosting
  persona_drift
  adaptive_evasion
)
MODEL_SEEDS=(7 17 27)

for scenario in "${SCENARIOS[@]}"; do
  for seed in "${MODEL_SEEDS[@]}"; do
    run_dir="$RUNS/p2_formal/${scenario}/seed_${seed}"
    mkdir -p "$run_dir"
    python scripts/train_p2.py \
      --plan "$FORMAL_PLAN" \
      --output-dir "$run_dir" \
      --protocol P2_multisource_real \
      --held-out-scenario "$scenario" \
      --seed "$seed" \
      --epochs 50 \
      --device cuda:0 \
      --hidden-dim 64 \
      --num-heads 4 \
      --num-layers 2 \
      --dropout 0.1 \
      --learning-rate 0.001 \
      2>&1 | tee "$run_dir/run.log"
  done
done
```

当前 runner 按验证集 AUPRC（不可用时退回 balanced accuracy）选择最佳 epoch，
最后一次性评估测试集。每折必须保留 `checkpoint.pt`、`metrics.json`、
`config.json`、`data_plan.json`、日志、Git commit、环境版本和 GPU 信息。

## 10. 停止条件

出现以下任一情况立即停止，不继续批量训练：

- checksum 不一致或 manifest 路径越出数据包；
- 代码 commit 与计划记录不一致或工作区存在未知修改；
- DatasetPlan 有 `errors`；
- 训练、验证或测试出现 NaN/Inf；
- CUDA OOM、GPU 不可见或 checkpoint 无法重新读取；
- Smoke 被误当成正式训练指标；
- 正式数据缺失时间截断、标签来源或 episode provenance。
