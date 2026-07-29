# TwiBot-22 1000-Agent + DeepPersona 集成运行说明

## 本次验证结论

- 固定核心账号：1000（500 human、500 bot），ID 全部唯一且均在 `label.csv` 命中。
- TwiBot 原始数据适配：5549 个账号节点、4238 条关注边、37639 条帖子、37502 条历史轨迹。
- 最终 Agent CSV：1000 行，1000 个非空人格描述，包含 24 小时活跃度、历史帖子和原始 ID。
- DeepPersona/RAG：1000 个账号生成 4000 个证据块；本次 8 个活跃 Agent 全部注入成功。
- API：本地 `qwen3.5:9b` 共 8 次请求，8 次成功、0 次失败。
- 新仿真 DB：SQLite `integrity_check=ok`，1000 个 Agent、500 good、500 bad。

## 数据链路

```text
TwiBot-22 原始文件
  -> 固定 1000 个核心 ID
  -> TwiBot/OASIS 基础 DB + 基础 CSV + ID 映射 + manifest
  -> 交大框架需要的最终 Agent CSV
  -> 账号证据结构化 + DeepPersona chunk + Chroma 向量库
  -> YAML（人数、时间步、活跃比例、模型接口）
  -> MultiAgent4Collusion 仿真
  -> 新仿真 DB + 原始动作 + 检测输入 + 审计 manifest
```

## 复现命令

在仓库根目录执行：

```bash
.venv/bin/python data_processing/twibot_oasis_adapter.py \
  --twibot-dir TwiBot-22 \
  --output-dir output/twibot_oasis_1000 \
  --total-sample-size 1000 \
  --core-ids twibot_1000_core_ids.txt \
  --output-tag 1000 \
  --max-actions 50 \
  --max-follows 100 \
  --random-seed 42

.venv/bin/python data_processing/twibot_simulation_csv.py \
  --adapter-db output/twibot_oasis_1000/twibot_1000_v5.db \
  --adapter-csv output/twibot_oasis_1000/twibot_1000_multimodal_v5.csv \
  --output output/twibot_oasis_1000/twibot_1000_simulation.csv \
  --manifest output/twibot_oasis_1000/twibot_1000_simulation_manifest.json

.venv/bin/python deeppersona_ai/prepare_twibot_profiles.py \
  --csv output/twibot_oasis_1000/twibot_1000_simulation.csv \
  --output output/twibot_oasis_1000/twibot_1000_profiles.json

.venv/bin/python deeppersona_ai/profile_chunker.py \
  --input output/twibot_oasis_1000/twibot_1000_profiles.json \
  --output output/twibot_oasis_1000/twibot_1000_chunks.json

.venv/bin/python deeppersona_ai/build_vector_store.py \
  --input output/twibot_oasis_1000/twibot_1000_chunks.json \
  --output-dir output/twibot_oasis_1000/vector_store

cd MultiAgent4Collusion-master
DEEP_PERSONA_VECTOR_STORE=../output/twibot_oasis_1000/vector_store \
../.venv/bin/python \
  scripts/twitter_simulation/align_with_real_world/twitter_simulation_large.py \
  --config_path scripts/misinformation_simulation/twibot_1000_t1_validation.yaml
```

## 方法边界

本次 TwiBot-22 是真实账号验证，因此 DeepPersona 层只组织可追溯的公开证据：简介、账号指标、关系与历史帖子；没有随机给真实账号编造年龄、职业或价值观。官方 DeepPersona 随机生成的完整合成人格更适合单独的合成仿真实验，不应当未经标注地混入 TwiBot-22 真实数据实验。

当前 YAML 是链路验证配置：1000 个 Agent 全部实例化，但 `activation_scale=0.02`，一个时间步只抽中 8 个 Agent 调用本地 API。30/50 时间步正式实验应复制该 YAML、固定随机种子，并根据服务器并发能力提高活跃比例。
