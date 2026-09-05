# HyperDecept P2 正式数据服务器执行方案

> 接收人：数据与服务器执行组员  
> 执行基线：`main`，至少为 `7a20f52`  
> 服务器：`xxy@114.55.248.113`  
> 原则：原始数据只在服务器私有目录处理一次；Git 只存代码，不存原始数据、数据包、模型权重或密钥。

## 1. 本轮决定

可以跳过耗时的本地抽取，直接将获得授权的 TwiBot-22 和 MGTAB 原始数据上传到服务器，一次性完成审计、抽取、特征构建、manifest 和 checksum。这样可避免本地算力不足及反复上传。

但必须区分三类 seed：

1. **真实数据没有 simulation seed**：TwiBot-22 和 MGTAB 各保留一个正式数据包，所有训练 fold 和模型 seed 共同读取。
2. **simulation seed**：只属于 DeepPersona/OASIS 仿真 episode，决定一次独立世界的生成过程。
3. **model seed**：只决定模型初始化和训练随机性，不重新抽取真实数据，也不重新生成仿真 DB。

因此，`twibot22/` 下绝对不需要为每个 seed 建文件夹。正确结构如下：

```text
/home/xxy/hyperdecept-p2/
├── repo/                              # main 代码工作树
├── package/                           # 可搬迁的最终正式数据包
│   ├── bundles/
│   │   ├── twibot22/                  # 只有一份，共享给全部训练
│   │   ├── mgtab/                     # 只有一份，共享给全部训练
│   │   └── synthetic/
│   │       └── main/
│   │           ├── leader_amplifier/n5000/seed_11.*
│   │           ├── leader_amplifier/n5000/seed_22.*
│   │           ├── leader_amplifier/n5000/seed_33.*
│   │           ├── leader_amplifier/n5000/seed_44.*
│   │           └── <其余四个场景>/n5000/seed_<simulation_seed>.*
│   ├── plans/
│   ├── audits/
│   └── checksums/
├── work/                              # 中间文件、缓存、临时仿真输出
├── model-cache/                       # Hugging Face/心理特征模型缓存
└── runs/
    └── p2_formal/
        └── <held_out_scenario>/model_seed_<7|17|27>/

/data/hyperdecept/xxy/                 # 管理员授权后使用的数据盘
└── raw-private/                       # 临时原始数据，禁止提交 Git
    ├── twibot22/
    └── mgtab/
```

代码当前使用 `seed_11.db`、`seed_11.csv`、`seed_11.features.csv` 等同前缀文件，而不是强制 `seed_11/` 子目录。二者都能组织数据，但正式执行应遵守 DatasetPlan 当前声明的**同前缀文件合同**，不要自行改成嵌套 seed 目录后让路径失配。

## 2. 规模设计与先行条件

### 2.1 正式主语料

如果 5000-agent pilot 通过，主仿真语料为：

```text
5 个场景 × 4 个 simulation seeds × 5000 agents × 50 time steps
= 20 个独立 episode
= 100,000 个 synthetic agent instances
```

场景固定为：

```text
leader_amplifier
bridge_infiltration
synchronized_boosting
persona_drift
adaptive_evasion
```

simulation seeds 固定为 `11,22,33,44`。模型 seeds 固定为 `7,17,27`。

### 2.2 必须先跑一个 pilot

不得直接并发生成 20 个 5000-agent episode。先执行：

```text
leader_amplifier × seed 11 × 5000 agents × 50 steps
```

pilot 必须验证：

- 仿真程序能够真正完成 50 steps，而不是只创建 5000 个静态 profile。
- cutoff 前快照不包含 cutoff 后事件。
- future trace 中存在 cutoff 后动作，可生成非空 `next_action`。
- 26 维特征不是 Smoke 占位值，并与 5000 个 `user_id` 一一对齐。
- DB 通过 SQLite integrity check；角色、campaign、阶段、动作和 provenance 来源明确。
- 记录墙钟时间、峰值 RAM/GPU、DB/CSV 大小、LLM/API 调用量与费用。

只有 pilot 全部通过，才批准剩余 19 个 episode。若运行时间、API 成本或磁盘不可接受，先调整生成策略，不得静默减少步数或伪造未来动作。

### 2.3 规模外推实验

由于主语料已全部使用 5000 agents，不再执行原方案的完整 `500/1000/5000` 大笛卡尔积。论文若需要规模泛化曲线，后续只增加一个低成本子集：

```text
leader_amplifier + adaptive_evasion
× N={500,1000,2000}
× simulation seed={101}
= 6 个 scale-test episode
```

该子集不进入主训练，只用于 scale test；应在 20 个主 episode 完成后再决定是否生成。

## 3. 登录、代码与私有目录

```bash
ssh -i ~/.ssh/hyperdecept_p2 xxy@114.55.248.113

set -euo pipefail
export HD=/home/xxy/hyperdecept-p2
export REPO=$HD/repo
# 原始大数据优先放数据盘；管理员需先创建并授权该目录。
export RAW=/data/hyperdecept/xxy/raw-private
export PKG=$HD/package
export WORK=$HD/work
export RUNS=$HD/runs
export HF_HOME=$HD/model-cache

umask 077
mkdir -p "$RAW/twibot22" "$RAW/mgtab" \
  "$PKG/bundles/twibot22" "$PKG/bundles/mgtab" \
  "$PKG/bundles/synthetic/main" \
  "$PKG/plans" "$PKG/audits" "$PKG/checksums" \
  "$WORK" "$RUNS" "$HF_HOME"
chmod 700 "$RAW" "$WORK" "$HF_HOME"
```

如果 `mkdir -p "$RAW"` 报 `Permission denied`，立即停止并让管理员执行一次：

```bash
sudo mkdir -p /data/hyperdecept/xxy/raw-private
sudo chown -R xxy:xxy /data/hyperdecept/xxy
sudo chmod 700 /data/hyperdecept/xxy/raw-private
```

不要在权限未修好时自动退回系统盘上传完整 TwiBot-22。

不要修改服务器上已经封存的 Smoke 工作树。为正式任务新建干净工作树：

```bash
git clone <HyperDecept Git URL> "$REPO"
cd "$REPO"
git fetch origin --prune
git checkout main
git pull --ff-only origin main
git rev-parse HEAD | tee "$PKG/audits/code_commit.txt"
git status --porcelain
```

验收：`git status --porcelain` 必须为空，commit 至少包含 `7a20f52`。

进入已配置的 Python 环境；如果正式工作树单独建环境：

```bash
cd "$REPO"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install torch-geometric sentence-transformers transformers spacy \
  pandas numpy scipy scikit-learn ijson pyarrow tqdm pytest
```

服务器已知环境为 Python 3.10、PyTorch 2.6.0+cu124、PyG 2.6.1、NVIDIA A10。不要为追求“更新”随意升级已通过 Smoke 的 CUDA/PyTorch 组合。

## 4. 上传原始 TwiBot-22 与 MGTAB

仅在数据集许可、学校政策和服务器用途允许时上传。原始数据禁止放入仓库目录。

在数据持有者电脑执行，推荐 `rsync`，中断后可以续传：

```bash
rsync -avhP --partial --append-verify \
  /local/path/TwiBot-22/ \
  -e "ssh -i ~/.ssh/hyperdecept_p2" \
  xxy@114.55.248.113:/data/hyperdecept/xxy/raw-private/twibot22/

rsync -avhP --partial --append-verify \
  /local/path/MGTAB/ \
  -e "ssh -i ~/.ssh/hyperdecept_p2" \
  xxy@114.55.248.113:/data/hyperdecept/xxy/raw-private/mgtab/
```

若 Windows 没有 `rsync`，可用 WSL；最后才使用 `scp -r`。上传前后分别对原始文件生成 SHA-256 清单并比较。服务器端：

```bash
cd "$RAW"
find twibot22 mgtab -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$PKG/audits/raw_server.sha256"

du -sh "$RAW/twibot22" "$RAW/mgtab" | tee "$PKG/audits/raw_sizes.txt"
find "$RAW/twibot22" -maxdepth 1 -type f -printf '%f\n' | sort
find "$RAW/mgtab" -maxdepth 2 -type f -printf '%P\n' | sort
```

当前 `/home/xxy` 所在系统盘约剩 153GB，而 `/data/hyperdecept` 约剩 274GB、但 `xxy` 当前无写权限。必须先修复数据盘权限，再上传完整原始数据，并预留解包和派生文件的峰值空间。上传前先在持有端统计压缩包和解压后大小；若原始数据、解压副本和临时文件的预计峰值接近 274GB，也不能直接上传，需扩容或采用边解压边处理且及时删除已确认不再使用的压缩副本。

- 只有原始数据 + 中间文件峰值可稳定低于约 120GB 时，才可由管理员批准暂用 `/home/xxy`。
- 正常方案是使用 `/data/hyperdecept/xxy/raw-private`，不要硬塞满系统盘。

## 5. TwiBot-22 一次性正式抽取

### 5.1 先做只读审计

```bash
cd "$REPO"
source .venv/bin/activate

python -m data_processing.twibot_raw_audit \
  --twibot-dir "$RAW/twibot22" \
  --output "$PKG/audits/twibot22_raw_audit.json"
```

完整 `--sha256` 会再次读取全部 tweet shards，可能非常慢；已有上传校验清单时可不重复全量 hash，但必须保留 `raw_server.sha256`。

### 5.2 确定正式 core ID 集合

TwiBot-22 适配器是**有界核心用户 + 全部匹配一跳外部邻居**抽取，不是默认把整个 TwiBot-22 的所有用户都实例化为训练节点。正式 core ID 文件必须由固定、可复现的规则生成，并保留：

```text
$PKG/bundles/twibot22/core_ids.txt
$PKG/audits/twibot22_core_selection.json
```

选择原则：

- 首选覆盖官方 train/validation/test 的、类别分层的较大 core 集，而不是沿用 Smoke 的固定 1000 人。
- core 数量由磁盘、特征提取耗时和 A10 训练吞吐的 pilot 决定，不要在未测量时声称“全量 TwiBot”。
- 无论 core 规模多少，抽取出的相连外部邻居可以保留为无标签上下文，不算作 core 数量。
- `core_selection.json` 记录原始 label/split hash、采样算法、随机种子、各 split/label 数量和代码 commit。

当前仓库没有“论文正式 core ID 采样器”的独立一键 CLI。因此组员不能临时凭手工顺序截取 ID；先产出并回传选择统计，待负责人确认 core 规模与规则后再正式抽取。

### 5.3 运行一次流式抽取

确认 `core_ids.txt` 后执行：

```bash
CORE_COUNT=$(grep -cv '^[[:space:]]*$' "$PKG/bundles/twibot22/core_ids.txt")

python -m data_processing.prepare_p2_smoke_data twibot \
  --twibot-dir "$RAW/twibot22" \
  --core-ids "$PKG/bundles/twibot22/core_ids.txt" \
  --expected-core-count "$CORE_COUNT" \
  --edge-chunksize 250000 \
  --output-dir "$PKG/bundles/twibot22"
```

模块名称保留了 `smoke_data`，但 `twibot` 子命令调用的是流式 `TwiBot22RawAdapter`，`expected-core-count` 可自定义，并非固定 1000。它会扫描原始大文件一次并物化所需表。

### 5.4 正式 26 维特征

禁止直接把 `prepare_p2_smoke_data twibot-features` 的结果当论文正式特征；该子命令文档明确标注为 `smoke-only`，没有运行完整四个心理/情感证据引擎。

正式特征必须使用现有 `MultimodalExtractor`/四个 `emotional_analysis` 引擎生成并保存为：

```text
$PKG/bundles/twibot22/node_features_26d.csv
```

强制合同：

- 一行一个 core `user_id`，不能重复、不能缺行。
- 精确包含训练器的 26 个字段：8 个语义、10 个行为、8 个心理代理特征。
- `Empathy_Gap_*`、`Dark_Triad_*`、`Contagion_*`、`Volatility_*` 不能整列占位为零。
- 保存模型名/版本、模型缓存 hash、提取器 commit、输入文本覆盖率、失败率和每列统计。
- 心理字段只能表述为可观察语言/行为代理特征，不是真实人格诊断。

当前代码有完整引擎和 `--features-only` 路径，但尚无针对 TwiBot 原始 bundle 的正式批处理封装命令。不要凭空使用 Smoke 特征替代。完成原始抽取后先停在这里，把 bundle 统计发给负责人；由负责人合并正式批处理 wrapper 后再运行。

此外，当前 `episode_manifest create` 的 TwiBot 正式条目仍按原始目录布局声明 `label.csv/split.csv`，没有完整声明已物化 bundle 的 `adapter_manifest_json`。而训练器只有拿到该字段才会调用 `load_materialized_bundle`，避免重新扫描原始 TwiBot。正式 DatasetPlan 前必须修复此合同。该问题不妨碍上传、审计和一次性抽取，但未修复前禁止删除原始 TwiBot、禁止开始正式训练。

## 6. MGTAB 一次性正式准备

MGTAB 是静态的 10,199 节点标准张量包，只需一份。它的 788 维输入由 20 维账号字段和 768 维预计算 LaBSE 表示组成，不运行 TwiBot 的 26 维心理特征流程。

```bash
cd "$REPO"
source .venv/bin/activate

python -m data_processing.mgtab_raw_audit \
  --mgtab-dir "$RAW/mgtab" \
  --output "$PKG/audits/mgtab_raw_audit.json" \
  --markdown-output "$PKG/audits/mgtab_raw_audit.md"

cp "$RAW/mgtab/edge_index.pt" "$PKG/bundles/mgtab/"
cp "$RAW/mgtab/edge_type.pt" "$PKG/bundles/mgtab/"
cp "$RAW/mgtab/edge_weight.pt" "$PKG/bundles/mgtab/"
cp "$RAW/mgtab/features.pt" "$PKG/bundles/mgtab/"
cp "$RAW/mgtab/labels_bot.pt" "$PKG/bundles/mgtab/"
cp "$RAW/mgtab/labels_stance.pt" "$PKG/bundles/mgtab/"

python -m data_processing.prepare_p2_smoke_data mgtab \
  --mgtab-dir "$PKG/bundles/mgtab" \
  --output-dir "$PKG/bundles/mgtab/derived" \
  --split-seed 42 \
  --multiedge-policy coalesce_with_count
```

验收至少包括：10,199 节点；特征 `[10199,788]`；bot/stance 标签与节点对齐；7 种关系合法；固定 split 和 adapter manifest 已生成。

## 7. 5000-agent 仿真 pilot

`data_processing.materialize_simulation_episode` 只负责把**已经完成的仿真**物化为 P2 合同，它不会生成 OASIS/DeepPersona 场景。必须先使用项目中实际可运行的场景入口完成仿真。

组员先确认并记录实际入口：

```text
<ACTUAL_SIMULATION_ENTRYPOINT>
```

不得把不存在的命令写进报告。使用实际入口生成：

```text
scenario=leader_amplifier
simulation_seed=11
num_agents=5000
time_steps=50
```

仿真原始输出必须至少提供：

```text
cutoff_snapshot.db
profiles.csv
node_features_26d.csv
future_trace.csv
generation_config.json
generation.log
```

`future_trace.csv` 必须包含 `user_id,created_at,action`；`profiles.csv` 必须包含唯一的 `user_id,user_type`；`node_features_26d.csv` 必须是一人一行的真实非占位 26 维特征。

选定 cutoff 后先把最终 DB 放入正式 bundle：

```bash
SCENARIO=leader_amplifier
SIM_SEED=11
N=5000
STEPS=50
IN=$WORK/simulation-pilot/$SCENARIO/n$N/seed_$SIM_SEED
OUT=$PKG/bundles/synthetic/main/$SCENARIO/n$N/seed_$SIM_SEED

mkdir -p "$(dirname "$OUT")"
cp "$IN/cutoff_snapshot.db" "$OUT.db"
cp "$IN/profiles.csv" "$OUT.csv"

python -m data_processing.materialize_simulation_episode \
  --db "$OUT.db" \
  --profiles "$OUT.csv" \
  --node-features "$IN/node_features_26d.csv" \
  --future-trace "$IN/future_trace.csv" \
  --cutoff-time '<ISO-8601 cutoff>' \
  --input-is-cutoff-snapshot \
  --scenario "$SCENARIO" \
  --seed "$SIM_SEED" \
  --num-agents "$N" \
  --time-steps "$STEPS" \
  --output-prefix "$OUT"
```

注意：物化器将输入 DB 作为 `source_path`，不会替你复制 DB；所以必须先复制到 `$OUT.db` 并传入同一路径。不要让 manifest 指向之后会被清理的 `$WORK` 文件。

**代码门槛：** 当前 `materialize_simulation_episode.py` 仍将 episode ID 后缀和 `generator_metadata.smoke_only` 写为 Smoke 值，且 provenance 字段不足以支撑正式论文数据。因此上面的物化命令是正式接口的目标模板；必须先由负责人升级并合并“formal mode”，测试确认不再出现 `:smoke` 和 `smoke_only=true` 后，组员才可以物化正式 pilot。原始 OASIS/DeepPersona pilot 可以先运行并保留在 `$WORK`，但不能将旧物化产物登记为正式数据。

pilot 验收：

```bash
python - <<'PY'
from pathlib import Path
import json, pandas as pd, sqlite3

p = Path('/home/xxy/hyperdecept-p2/package/bundles/synthetic/main/leader_amplifier/n5000/seed_11')
m = json.loads(Path(f'{p}.manifest.json').read_text(encoding='utf-8'))
profiles = pd.read_csv(f'{p}.csv')
features = pd.read_csv(f'{p}.features.csv')
targets = pd.read_csv(f'{p}.event_targets.csv')
assert len(profiles) == len(features) == len(targets) == 5000
assert profiles.user_id.astype(str).is_unique
assert features.user_id.astype(str).is_unique
assert targets.user_id.astype(str).is_unique
assert {'next_action','target_time','cutoff_time'}.issubset(targets.columns)
assert targets.next_action.notna().sum() > 0
assert m['capabilities']['next_action_targets'] is True
with sqlite3.connect(f'{p}.db') as c:
    assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
print('pilot contract passed; next_action labels=', targets.next_action.notna().sum())
PY
```

同时将运行时间、资源、文件大小、有效动作覆盖率和成本写入：

```text
$PKG/audits/synthetic_pilot_report.md
```

## 8. 批量生成剩余 19 个正式 episode

pilot 获得负责人批准后，按以下矩阵生成，不改变 prompt、模型、feature extractor 或 time-step 定义：

```text
scenarios = leader_amplifier, bridge_infiltration,
            synchronized_boosting, persona_drift, adaptive_evasion
simulation_seeds = 11,22,33,44
num_agents = 5000
time_steps = 50
```

每个 episode 的最终同前缀文件必须齐全：

```text
seed_<s>.db
seed_<s>.csv
seed_<s>.features.csv
seed_<s>.labels.csv
seed_<s>.event_targets.csv
seed_<s>.manifest.json
```

此外，`manifest.json`/provenance 至少记录：

- scenario、simulation seed、人数、50 steps、cutoff。
- LLM/API 或本地模型名称、版本、推理参数。
- prompt/config SHA-256、代码 commit、DeepPersona/OASIS 版本。
- 角色、campaign、attack phase、next action 的来源。
- feature extractor 和 graph builder 版本。
- DB、profiles、features、targets 的 SHA-256。

每生成一个 episode 立即单独验收并登记，不要等 20 个全部结束后才发现合同错误。建议串行跑 pilot，正式批次最多按资源和 API 限额做小并发；禁止多个任务写同一 DB 或 cache 文件。

## 9. DatasetPlan、相对路径与 checksum

所有正式 bundle 完成、且两个代码门槛（正式 TwiBot bundle contract、正式 simulation materializer）已经合并并测试后生成计划。由于主语料已是 N=5000，先关闭额外 scale entries：

```bash
cd "$REPO"
source .venv/bin/activate

python -m data_processing.episode_manifest create \
  --output "$PKG/plans/p2_formal_dataset_plan.json" \
  --simulation-root "$PKG/bundles/synthetic" \
  --twibot-root "$PKG/bundles/twibot22" \
  --mgtab-root "$PKG/bundles/mgtab" \
  --scenarios leader_amplifier,bridge_infiltration,synchronized_boosting,persona_drift,adaptive_evasion \
  --main-seeds 11,22,33,44 \
  --anchor-num-agents 5000 \
  --scale-scenarios leader_amplifier \
  --scale-sizes 5000 \
  --scale-seeds 101 \
  --time-steps 50

python -m data_processing.episode_manifest validate \
  --input "$PKG/plans/p2_formal_dataset_plan.json" \
  --require-files \
  | tee "$PKG/audits/p2_formal_plan_validation.txt"
```

当前 main 已实现相对路径 manifest contract。计划文件应能随整个 `package/` 移动，而不是固化生成机器绝对路径。检查：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/home/xxy/hyperdecept-p2/package/plans/p2_formal_dataset_plan.json')
d = json.loads(p.read_text(encoding='utf-8'))
paths = []
for e in d['episodes']:
    paths.append(e['source_path'])
    paths.extend(e.get('artifacts', {}).values())
absolute = [x for x in paths if Path(x).is_absolute()]
assert not absolute, absolute[:5]
print('relative manifest paths passed:', len(paths))
PY
```

生成最终校验：

```bash
cd "$PKG"
find bundles plans audits -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > checksums/checksums.sha256
sha256sum -c checksums/checksums.sha256 \
  | tee checksums/checksum_verification.txt
```

注意：不要把 `checksums.sha256` 自己包含进待校验列表。

## 10. 清理原始数据的条件

TwiBot/MGTAB 可以在服务器完成最终抽取后删除，以节省空间，但必须同时满足：

1. DatasetPlan `--require-files` 无错误。
2. checksum 全部 `OK`。
3. TwiBot 正式 26 维特征非占位且审计通过。
4. MGTAB 10,199 节点完整通过适配审计。
5. 20 个仿真 episode 的 `next_action` 不为空并通过逐集审计。
6. 用正式 package 完成 P2 one-step dry-run。
7. 数据持有者确认原始文件在其他受控位置仍有备份，或明确批准删除。

满足后只删除 `$RAW/twibot22` 和 `$RAW/mgtab`，绝对不要删除 `$PKG/bundles`。删除前后记录：

```bash
du -sh "$RAW/twibot22" "$RAW/mgtab" | tee "$PKG/audits/raw_before_deletion.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "$PKG/audits/raw_deletion_approved_at.txt"
```

实际删除属于不可逆操作，必须由服务器管理员/数据负责人确认后执行，本方案不自动执行删除命令。

## 11. P2 正式训练前 dry-run

先确认计划有效，然后每个 LOSO fold 至少跑一个训练 step。示例：

```bash
cd "$REPO"
source .venv/bin/activate

python scripts/train_p2.py \
  --plan "$PKG/plans/p2_formal_dataset_plan.json" \
  --output-dir "$RUNS/p2_dry_run/leader_amplifier/model_seed_7" \
  --protocol P2_multisource_real \
  --held-out-scenario leader_amplifier \
  --seed 7 \
  --epochs 1 \
  --max-steps 1 \
  --device cuda
```

验收：forward、loss、backward、checkpoint 成功；`action_vocabulary` 不是只有 `unavailable`；synthetic action labels 大于 0；所有实际使用的输入 adapter 已初始化；输出记录 plan、commit、CUDA/PyTorch 和 seed。

随后对其余四个 held-out scenario 重复 one-step dry-run。五个 fold 都通过才允许正式训练。

## 12. 正式训练矩阵

正式训练为 5 个 LOSO held-out scenarios × 3 个 model seeds，共 15 次；不重新生成任何数据：

```text
held_out_scenarios =
  leader_amplifier
  bridge_infiltration
  synchronized_boosting
  persona_drift
  adaptive_evasion

model_seeds = 7,17,27
```

每次输出：

```text
$RUNS/p2_formal/<held_out_scenario>/model_seed_<seed>/
├── checkpoint.pt
├── metrics.json
├── config.json
└── data_plan.json
```

命令模板：

```bash
python scripts/train_p2.py \
  --plan "$PKG/plans/p2_formal_dataset_plan.json" \
  --output-dir "$RUNS/p2_formal/<HELD_OUT>/model_seed_<MODEL_SEED>" \
  --protocol P2_multisource_real \
  --held-out-scenario <HELD_OUT> \
  --seed <MODEL_SEED> \
  --epochs <经 validation 预注册的正式 epoch 数> \
  --device cuda
```

不要在 test 指标出来后修改 epoch、损失权重或阈值。正式 epoch、early-stopping/selection 规则和超参数应在 dry-run 后、正式 15 次训练前一次性冻结。

## 13. 必须停止并汇报的情况

发生以下任一情况立即停止，不要自行补造数据：

- 数据许可不允许上传云服务器或做派生处理。
- 系统盘剩余空间低于 30GB，或预计峰值会占满磁盘。
- TwiBot/MGTAB 原始 SHA-256 与上传端不一致。
- TwiBot core ID 缺标签/划分，或抽取后的 ID/行数无法对齐。
- 正式 26 维特征仍是 Smoke 占位值或整列异常为零。
- 仿真没有真正的 cutoff snapshot/future trace，导致 `next_action` 不可用。
- DB 用户数不是 5000、未完成 50 steps，或 SQLite integrity check 失败。
- manifest 含绝对路径、指向 `$WORK` 临时文件，或 checksum 失败。
- DatasetPlan `--require-files` 报错或 LOSO 泄漏审计失败。
- 正式 TwiBot bundle contract 或 simulation formal mode 尚未合并；此时最多执行原始数据上传、审计、抽取和仿真原始 pilot，不得进入正式 DatasetPlan/训练。
- A10 显存不足、出现 NaN/Inf、validation 无有限指标或 checkpoint 不可加载。

## 14. 每个阶段回传给负责人的内容

只回传审计和统计，不通过聊天工具发送原始用户文本、原始 DB、API key 或私钥：

1. 代码 commit、`git status`、Python/PyTorch/PyG/CUDA/GPU。
2. 原始数据文件清单、总大小、SHA-256 校验结论。
3. TwiBot core 选择规则及 train/val/test × bot/human 计数、边/邻居/帖子覆盖率。
4. MGTAB 节点、标签、关系、split 统计。
5. pilot 的时长、资源、成本、文件大小、有效 `next_action` 数量。
6. 20 个 episode 的场景/seed/人数/steps/状态表。
7. DatasetPlan summary、相对路径检查和 checksum 结果。
8. 五个 dry-run 与十五个正式训练的输出目录和关键指标。

最终保留：`package/bundles/`、`package/plans/`、`package/audits/`、`package/checksums/`、`runs/` 和 `model-cache/`。原始数据只在完整验收及明确批准后清理。
