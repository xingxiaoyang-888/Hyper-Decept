# HyperDecept 代码架构与关键文件清单

> 更新日期：2026-08-04  
> 本文档描述**当前仓库真实状态**。拟议但尚未实现的训练调整见
> `docs/HyperDecept_联合双曲训练与数据构建方案_v3.md`。

## 1. 当前唯一论文主线

```text
真实图：TwiBot-22 + MGTAB
受控仿真：DeepPersona + MultiAgent4Collusion/OASIS episodes
                         ↓
              audited DatasetPlan / EpisodeBatch
                         ↓
          dataset-specific heterogeneous inputs
                         ↓
              shared Intrinsic Lorentz-HGT
                         ↓
        shared Lorentz bot decision geometry
                         ↓
 synthetic privileged supervision + white-box explanation
                         ↓
       realtime trace + moderator decision support
```

P2 是正式训练协议。P1 是移除 MGTAB 等训练来源的控制实验，不是第二个项目。
`new_main_classifier.py` 的 XGBoost/单 DB 流程是兼容基线和白盒验收入口，不是 P2。

## 2. 数据与 manifest 层：`data_processing/`

| 文件 | 当前职责 | 状态 |
|---|---|---|
| `dataset_adapter.py` | 统一能力声明、bundle 和 ID/provenance 契约；不再读取旧 TwiBot V5 | 已完成 |
| `twibot_raw_audit.py` | 审计 TwiBot 原始文件、字段、覆盖率与哈希 | 已完成，待完整 TwiBot 正式运行 |
| `twibot22_raw_adapter.py` | 读取 TwiBot 用户、帖子、关系和官方 split；不补造角色 | 已完成，待完整数据物化 |
| `mgtab_raw_audit.py` | 审计 MGTAB tensor schema、节点、边、标签和能力缺失 | 已完成 |
| `mgtab_adapter.py` | 将 MGTAB 标准张量转换为训练 episode；保留 stance 但不冒充战术角色 | 已完成并通过 Smoke |
| `episode_manifest.py` | DatasetPlan、LOSO、P1/P2 assignment、artifact 与泄漏审计 | 已完成 |
| `materialize_simulation_episode.py` | 从 DB/CSV/sidecar 物化 synthetic features、labels、event targets 和 manifest | 已实现，待全部正式 DB 运行 |
| `audit_simulation_run.py` | 审计仿真 DB 的用户、动作、角色和基本完整性 | 已实现 |
| `prepare_p2_smoke_data.py` | 生成 Smoke bundle/manifest | 已完成；非正式数据入口 |

旧的 `twinbot_adapter.py`、`twinbot_adapter_dynamic.py`、`twibot_oasis_adapter.py`
以及 `TwiBotStaticAdapter` 已删除。它们输出的静态 `twibot_*_v5.db/csv` 不属于当前
P2 数据合同；误传给 `graph_builder.py` 会显式报错。当前 TwiBot-22 唯一正式入口是
`twibot_raw_audit.py` → `twibot22_raw_adapter.py` → 物化 bundle → DatasetPlan。

关键约束：

- 真实数据只有其原始标签；
- 外部邻居是上下文节点，不是第三类别；
- MGTAB 没有原始时间戳、campaign 和战术角色真值；
- synthetic user ID 的作用域是 episode，真实 user ID 的作用域是 dataset；
- DatasetPlan 是训练入口，不允许运行时扫描目录猜测数据。

## 3. 双曲模型层：`Character Classification/`

### 3.1 当前主模型文件

| 文件 | 当前实现 | 状态 |
|---|---|---|
| `lorentz_hgt.py` | Lorentz 运算、可学习曲率、关系消息传递、可靠性边属性、加权 Lorentz centroid、`LorentzPrototypeClassifier` | 核心完成，待正式规模训练 |
| `joint_training.py` | Episode loader、dataset-specific adapter、共享 Lorentz-HGT、共享 bot prototype、synthetic privileged heads、当前联合损失、评估、checkpoint、NeighborLoader 工具 | 核心完成；训练调度待 v3 升级 |
| `graph_builder.py` | 构建异构图并保持 node ID、relation 与 evidence 对齐 | 已完成 |
| `new_feature_extractor.py` | 多模态与 psychology-inspired proxy 特征、缓存和 provenance | 已完成 |

### 3.2 当前模型结构

`DomainAwareLorentzHGT` 实际使用：

```text
dataset-specific feature adapters
    → shared IntrinsicLorentzHGT
    → weighted Lorentz aggregation
    → one shared Lorentz bot prototype head
    → synthetic-only role/campaign/next-action heads
```

这已经替代早期的“只分 real/synthetic adapter + 两套 bot head”设计。

### 3.3 当前损失的真实实现

当前 `JointLossConfig` 仍为：

```text
detection_weight = 1.0
privileged_weight = 0.3
alignment_weight = 0.05
role/campaign/action share = 1.0/1.0/1.0
```

主损失只有 detection、privileged package 和 alignment 三组；旧 relation loss 已删除。
但损失量级尚未自适应归一化，三个 privileged share 仍是人工参数。该部分明确标记为
**待项目负责人审阅 v3 后修改**。

### 3.4 兼容/历史文件

| 文件 | 用途 | 禁止混淆 |
|---|---|---|
| `new_main_classifier.py` | XGBoost、旧 DB、白盒 packet 和传统基线 | 不是 P2 runner |
| `new_role_assigner.py` | 几何角色分析和旧角色实验 | 真实数据输出不能当角色真值 |
| `new_gang_detection.py` | 历史群体检测/可视化路径 | 非论文主模型 |

## 4. 正式训练入口：`scripts/`

| 文件 | 职责 | 当前状态 |
|---|---|---|
| `train_p2.py` | 读取 DatasetPlan、执行 P2 LOSO fold、训练/验证/测试、保存 checkpoint、metrics、config 和 data plan | 已完成基础 runner；待 v3 调度升级 |
| `run_p2_smoke.py` | TwiBot/MGTAB/synthetic 小规模端到端 Smoke | 已通过；不能作为论文结果 |
| `generate_formal_simulation_plan.py` | 生成 5 场景 × 4 seeds 的正式仿真 CSV/YAML/plan | 已完成 |
| `run_formal_simulation_plan.py` | 串行、可恢复地生成正式仿真 DB | 已完成并用于服务器续跑 |
| `calibrate_formal_simulation.py` | 运行 1 timestep API/吞吐校准 | 已完成并通过一次正式校准 |
| `audit_formal_simulation_inputs.py` | 审计人格哈希、CSV/YAML 和配置契约 | 已完成 |
| `prepare_model_cache.py` | 固定特征模型缓存 | 已实现，按需要运行 |

`train_p2.py` 当前已具备：

- P2 真实/仿真 batch 加载；
- LOSO assignment 和 artifact 强审计；
- 训练 schema 只能由 train graphs 决定；
- validation AUPRC 优先选最佳 epoch；
- 保存模型、优化器、损失配置、geometry metadata 和 DatasetPlan；
- 按 episode 输出测试指标。

当前不足：

- `make_user_neighbor_loader()` 已存在，但正式 runner 尚未真正调用；
- 无数据源/场景分层 mini-batch scheduler；
- positive class weight 没有由 fold train mask 自动填写；
- 固定 0.5 threshold，未实现 validation-only temperature scaling；
- 无 scheduler、early stopping、AMP 边界和完整断点恢复；
- 当前 CLI 只实现 P2，P1 控制实验 runner 尚未接入。

## 5. 心理/语言行为特征：`emotional_analysis/`

| 文件 | 作用 |
|---|---|
| `empathy_gap_analyzer.py` | Empathy-gap-inspired language proxy |
| `dark_triad_analyzer.py` | Dark-triad-inspired NLI proxy |
| `contagion_analyzer.py` | 情绪/语义传播相似性 proxy |
| `volatility_analyzer.py` | 情绪波动 proxy |

这些模块是可观测输入和解释概念，不是 bot 标签、真实人格诊断或角色真值。论文统一使用
`psychology-inspired linguistic/behavioral proxies`。它们可用于 TwiBot/synthetic 的
特征与解释；MGTAB 使用其公开预计算特征，不能补造缺失原文或心理字段。

## 6. 仿真与人格引擎

### `Deeppersona/`、`deeppersona_ai/`

- 生成和检查结构化 DeepPersona；
- 为 OASIS Agent 提供人格检索上下文；
- 当前正式人口是 2,000 个独立 profile，20 个 episode 共享同一人口哈希；
- 人格引擎属于数据生成上游，不属于白盒解释器。

### `MultiAgent4Collusion-master/`

- OASIS 社交平台、Agent、动作和 SQLite 仿真；
- 当前正式设置为 2,000 Agents、30 timesteps、budgeted activation；
- Agent 动作由 DeepSeek API（或经验证的兼容本地 LLM）动态生成；
- DB 生成后仍需物化为可训练 artifact，不能直接把裸 DB 视为最终 DatasetPlan。

## 7. 白盒解释与实时追踪：`explainability/`

| 文件 | 职责 | 状态 |
|---|---|---|
| `schemas.py` | explanation/evidence/geometry/role 数据契约 | 已完成 |
| `evidence_registry.py` | 白名单证据注册与可追溯 ID | 已完成 |
| `adapters.py` | Predictor/Explainer/EvidenceProvider 统一接口 | 已完成基础接口 |
| `additive_risk.py` | 可加性风险头与概念贡献 | 传统白盒基线已完成 |
| `hyperbolic_geometry.py` | Poincaré 投影和几何保真指标 | 已完成 |
| `geo_pgexplainer.py` | prediction/geometry/role-aware edge mask | 模块完成，待正式 benchmark |
| `geometry_benchmark.py` | 普通解释与 Geo-PG 对比 | 待正式模型与数据 |
| `realtime.py` | 解释状态机、hash-chain 和 SQLite trace | 后端基础完成 |
| `packet_builder.py` | 汇总 prediction、geometry、role 和 evidence | 已完成基础结构 |

解释器损失与检测器损失分开。实时追踪与人类受试者实验属于模型训练后的 HCI 评估，
不能用来替代检测性能和跨域泛化实验。

## 8. 九阶段当前状态

| 阶段 | 关键代码 | 当前状态 |
|---|---|---|
| M1 数据与证据 | `data_processing/`、`evidence_registry.py` | 核心完成；正式 TwiBot/仿真物化未完成 |
| M2 统一解释协议 | `schemas.py`、`adapters.py`、`packet_builder.py` | 基础完成 |
| M3 真正双曲 HGT | `lorentz_hgt.py`、`joint_training.py` | 几何核心完成；待 v3 训练升级和正式训练 |
| M4 白盒决策 | `additive_risk.py`、传统分类入口 | 基线完成；待连接 P2 checkpoint |
| M5 几何解释 | `geo_pgexplainer.py`、`geometry_benchmark.py` | 模块完成；待 benchmark |
| M6 实时追踪 | `realtime.py` | 后端状态机/hash-chain 完成 |
| M7 反事实 | schema 预留 | 生成器待实现 |
| M8 审核工作台 | adapter/realtime backend | UI 和完整交互待实现 |
| M9 实验与论文 | DatasetPlan、P2 runner、受试者方案 | 数据生成中；正式训练/解释/人类实验未运行 |

## 9. 当前数据状态

| 数据 | 状态 |
|---|---|
| MGTAB | 10,199 用户完整 tensor bundle 已完成 adapter 和 Smoke |
| TwiBot-22 | 完整原始数据待上传服务器并正式审计/物化 |
| DeepPersona population | 2,000 个有效且内容唯一的 profile 已完成 |
| Synthetic plan | 5 场景 × 4 seeds × 2,000 Agents × 30 timesteps |
| Synthetic DB | 部分完成，服务器重启后按 resumable state 继续；最终以 state/audit 为准 |
| Scale/OOD extras | 不再作为主训练启动条件，暂未冻结 |

## 10. 下一步顺序

在仿真仍未跑完时：

1. 项目负责人审阅 v3 训练规范；
2. 不启动正式 GPU 训练；
3. 继续完成 20 个仿真 DB；
4. 上传并物化完整 TwiBot-22；
5. 对全部数据生成可重定位 DatasetPlan/checksums；
6. 审阅通过后再修改训练损失、NeighborLoader、scheduler、校准和 resume；
7. 先运行 `epochs=1, max_steps=1`；
8. 再运行单 fold/单 seed 试训；
9. 最后执行 5 LOSO × 3 model seeds 正式训练。

未经审阅不得将 v3 中“拟修改”的内容写成已完成，也不得根据当前部分 DB 提前报告论文
性能数字。
