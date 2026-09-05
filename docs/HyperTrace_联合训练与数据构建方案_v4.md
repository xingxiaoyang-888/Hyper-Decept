# HyperTrace 联合训练与数据构建方案 v4

> 状态：服务器正式训练前冻结的研究契约
>
> 版本目标：修正 CIKM 评审指出的数据—任务错配、模块脱节、标签不可解释和实验不足问题。
>
> 若 IO-26 在实验冻结前仍处于 restricted/unavailable 状态，切换到 [IO-26 替代数据与主实验冻结方案](HyperTrace_IO26_替代数据与主实验冻结方案.md)，不得用 TwiBot-22/MGTAB 冒充协同标签数据。

## 1. 论文主任务

HyperTrace 不把主任务定义为“账号是不是 bot”，也不声称识别账号的真实心理人格。主任务定义为：

> 给定决策截止时间 `t` 之前可观察到的账号、帖子、交互和时间信息，判断一个账号或局部网络是否与经过验证的协同信息行动相关，并生成可供审核员核验的、带时间边界和来源记录的解释。

主标签只能来自数据集已有的可靠标注，或来自仿真器明确记录的 `coordination_label`。以下信息不得作为检测器输入：`persona`、心理分数、生成器内部状态、scenario 名称、campaign ID、`role`、`next_action` 及未来事件。

论文的核心主张只有三条：

1. 动态协同检测需要同时处理异构关系、时间截止和层级结构；
2. 受控 episode 能提供真实数据缺失的时间、协同和反事实监督，并可用于 sim-to-real 预训练；
3. HyperTrace 将预测转换为可回放、可追溯、可被审核员纠正的证据，而不是只输出一个风险分数。

## 2. 数据职责

| 数据源 | 正式职责 | 是否进入主训练 | 可支持的结论 |
|---|---|---:|---|
| IO-26 | 真实信息行动主数据；按 campaign 构造时间异构图 | 是，微调和主验证 | 未见真实 campaign 的协同检测泛化 |
| DeepPersona/OASIS | 受控动态 episode；提供协同、角色、策略、反事实和 provenance 真值 | 可用于预训练；须有/无仿真消融 | 动态机制、早期检测和解释校准 |
| Fox8-23 | 真实 ChatGPT-powered botnet | 否 | LLM botnet 迁移压力测试 |
| BotSim-24 | 独立 LLM botnet 仿真 | 否 | 外部生成器压力测试 |
| TwiBot-22 | 历史 bot benchmark；当前范围外 | 否 | 不支持协同信息行动主张，因此不进入本论文实验 |
| MGTAB | 历史多关系 bot benchmark；当前范围外 | 否 | 不支持协同信息行动主张，因此不进入本论文实验 |

本论文实验默认保留 IO-26 与经过审计的 DeepPersona/OASIS episode。若 IO-26 未获授权，则真实主数据切换为公开的 Crypto-Campaign 数据，任务名称同步改为 `coordinated deceptive campaign detection`；Fox8-23、BotSim-24、TwiBot-22 和 MGTAB 均不进入当前主实验矩阵。不得把真实数据与仿真数据直接拼接为一个无来源区分的训练表。

### 2.1 IO-26 时间契约

IO-26 的逐帖字段包含 `post_time`、回复、转发、提及、URL、话题、账号和账号创建时间，可按时间窗口构造：

```text
account --posts--> post --mentions/replies/reposts--> account
post --contains--> url/hashtag
```

每个样本必须带 `cutoff_time`，图构建只允许读取 `post_time <= cutoff_time` 的记录。IO-26 的 control timeline 可能只覆盖入选日期、最多约 100 条消息，因此必须报告活动跨度匹配规则，并加入时间泄漏测试；不得让“历史更长”成为标签捷径。

### 2.2 正式仿真契约

正式目标为 5 个场景 × 4 个仿真 seed：

```text
leader_amplifier
bridge_infiltration
synchronized_boosting
persona_drift
adaptive_evasion
```

每个 episode 使用 2,000 个 Agent、30 个 timestep。若服务器资源允许，可另做规模实验 `N ∈ {500, 2000}`；规模实验不替代主实验，也不改变主结论。

每个 episode 必须包含：

- 非占位的 26 维可观察特征；
- 带时间戳的 action/event 记录；
- `coordination_label` 和其生成来源；
- `role`、`campaign`、`next_action` 作为评估真值或解释校准字段，不进入主检测器输入；
- 每个 cutoff 的 `features`、`edges`、`labels` 和 `provenance`；
- 场景、仿真 seed、Agent 数、timestep、模型版本、提示词版本、文件 SHA-256；
- 可从中断处恢复的 episode checkpoint。

仿真标签必须和真实标签分开命名。禁止把生成器的角色命名包装为真实世界角色标注。

## 3. 统一输入特征

新版主模型使用统一的 26 维可观察特征。建议固定为：

```text
semantic_0 ... semantic_7                         8
follower_following_ratio, action_frequency,
like_ratio, retweet_ratio, reply_ratio,
temporal_entropy, url_ratio, mention_ratio,
hashtag_ratio, media_ratio                       10
active_span, inter_event_mean, inter_event_std,
burstiness, unique_target_ratio, reciprocity_ratio,
cross_user_sync, temporal_recency                  8
                                                     --
                                                     26
```

所有特征必须由 cutoff 之前的公开文本、资料、事件和图关系计算。心理分析文件夹可以保留用于历史复现和附录探索，但不进入 v4 主检测输入或损失函数。

## 4. 训练任务与模型

### 4.1 两阶段训练

```text
Stage A: DeepPersona/OASIS synthetic temporal pretraining (optional but testable)
    ↓ same 26D observable schema and same detector target
Stage B: IO-26 campaign-disjoint real-data fine-tuning
    ↓ frozen checkpoint and threshold
IO-26 unseen campaigns → primary test
Fox8/BotSim → external stress tests
```

Stage A 只有在仿真 episode 通过完整 provenance 和非占位特征检查后才启用。必须比较 `real-only` 与 `synthetic-pretrain → real-finetune`；如果仿真预训练不改善未见 campaign，则不把它写成有效性结论。

### 4.2 主检测器

主模型为 intrinsic Lorentz heterogeneous graph encoder：

```text
26D observable features + typed temporal graph
    → Intrinsic Lorentz-HGT
    → Lorentz prototype binary detector
    → coordination risk + uncertainty
```

隐藏状态、关系消息传递、距离和聚合在 Lorentz 流形中完成。Poincaré 坐标只用于解释和可视化输出，不把“投影到 Poincaré”当作训练本身。

### 4.3 主损失

主检测实验只使用一个 class-balanced binary classification loss：

```text
L = class-balanced cross-entropy(coordination_label, prediction)
```

暂时移除主损失中的心理辅助项、role head、campaign embedding、next-action head、synthetic privileged loss 和跨域 alignment loss。它们会把研究变成难以解释的多任务堆叠，也无法解决真实标签不足。

如果后续需要证明角色解释，角色只在仿真评估阶段作为外部真值，不反向更新主检测器。

## 5. 划分和实验矩阵

### 5.1 IO-26

IO-26 的发布描述提供了 26 个经过平台核验的信息行动 campaign 及其时间匹配的 control 数据，但不应把它理解为已经替我们冻结好论文所需的 train/validation/test 三分。我们必须自行按 campaign 切分，而不是按帖子或账号随机切分。使用固定脚本和固定 seed 生成约 `18 train / 4 validation / 4 test` 的 campaign-disjoint split；实际分配以 campaign 样本量和标签比例为约束，并写入不可变 manifest。账号不得跨 split，模型选择、阈值和标准化参数只读取 train/validation。

若 state actor 元数据可用，额外报告 actor overlap；若不可用，不声称 actor-disjoint 泛化。

### 5.2 Synthetic LOSO

每个 held-out scenario 的 4 个仿真 seed全部作为测试；其余场景中一个 seed 用于 validation，其余 seed 用于 pretraining。共 5 个 LOSO folds。模型 seed 只改变优化随机性，不重新生成 episode。

### 5.3 最小但足够的模型矩阵

| 编号 | 模型 | 目的 |
|---|---|---|
| E0 | Euclidean HGT，IO-26 real-only | 几何基线，不使用双曲空间 |
| E1 | Intrinsic Lorentz-HGT，IO-26 real-only | 双曲结构对照，直接在 Lorentz 流形训练 |
| E2 | Intrinsic Lorentz-HGT，synthetic pretrain → IO-26 finetune | 双曲 sim-to-real 方案；预训练和微调均使用 Lorentz-HGT |
| E3 | E2 去掉时间字段/随机打乱时间 | 时间有效性消融 |

每个模型使用至少 3 个 model seeds。若 E2 没有稳定改善外部泛化，不声称仿真预训练提高检测性能，而将仿真定位为解释和压力测试环境。

## 6. 评估指标

### 6.1 检测

- AUROC、AUPRC、Macro-F1、Balanced Accuracy；
- Brier Score 和 ECE；
- account-level micro 与 campaign-level macro；
- 3 个 model seeds 的均值、标准差和 95% bootstrap 置信区间。

### 6.2 动态检测

在每个 episode/campaign 的 10%、25%、50%、75% 时间进度报告：

- early Recall/Precision；
- time-to-detection 和提前发现时间；
- false alarm rate；
- 新事件到来后的风险变化；
- cutoff 泄漏率。

IO-26 或外部数据没有可靠时间字段时，相应指标必须标记为 unavailable，而不是补造时间。

### 6.3 解释

主解释指标为：

1. prediction fidelity；
2. Lorentz geometry fidelity（距离失真、径向顺序保持）；
3. temporal validity（解释不引用 cutoff 之后的事件）；
4. evidence traceability（evidence ID、来源记录、时间、版本、hash 可核验）。

role fidelity 只在 DeepPersona 有生成真值的 episode 上作为附加指标。解释延迟、失败率和 unresolved evidence warning 必须一并报告。

### 6.4 审核员实验：验证实时可审计工作流

贡献三不能只用界面截图证明。实验分为“离线回放审计”和“人类受试者研究”两层。

#### 6.4.1 案例生成与冻结

从未见过的 IO-26 test campaigns 和留出的 synthetic episodes 生成脱敏案例。每个案例包含一个冻结的决策截止时间、该时刻之前的异构图、模型风险输出、ExplanationPacket 和真实标签。IO-26 案例只使用 campaign-association 标签；角色真值只在 synthetic 案例中用于解释评估。

案例至少覆盖：正确高置信建议、正确低置信建议、错误高置信建议和证据相互冲突的边界案例。若自然模型错误数量不足，可离线构造“错误 AI 建议”版本，但必须在伦理材料中说明，实验后向参与者完整告知，且不得把这种建议当作模型真实性能。

每个案例在实验前冻结：数据版本、cutoff、模型 checkpoint、阈值、解释 packet、来源 hash 和案例标签。参与者永远看不到未来事件、persona、生成器内部状态或隐藏标签。

#### 6.4.2 实验条件

采用被试内、拉丁方 counterbalancing；不同条件使用内容和标签匹配但不重复的案例集：

1. **Manual**：只看脱敏的事件和关系材料，不显示 AI 建议；
2. **Score-only**：显示风险分数、置信度和二元建议；
3. **Static explanation**：显示风险分数和固定 top-k 证据，不显示时间状态、来源核验、反证和认知强制提示；
4. **HyperTrace**：显示 cutoff 时间线、Lorentz/Poincaré 结构摘要、支持与反证、证据来源、反事实和审核员确认/纠正流程；
5. **HyperTrace-no-forcing**：与 HyperTrace 相同但去掉要求参与者先核对反证再决定的认知强制提示。

第五个条件用于单独估计 cognitive forcing 的作用；如果受试者数量或时长有限，先保留 1–4 条件，并在 HyperTrace 条件内随机开启/关闭提示。

#### 6.4.3 参与者任务

每个 trial 要求参与者：

1. 判断账号/局部网络是否与协同信息行动相关；
2. 选择接受、拒绝或升级 AI 建议；
3. 给出 0–100 的置信度；
4. 在解释条件下指出至少一条支持证据、一条反证，并核对证据是否早于 cutoff；
5. 记录最终审核结论和可选的简短理由。

参与者先完成教程和练习题，再进入正式试次。正式试次中不要让同一个案例出现在多个条件，避免记忆造成泄漏。

#### 6.4.4 研究问题与假设

- **RQ1/H1**：HyperTrace 是否比 score-only 和 static explanation 提高人机联合准确率与证据核验准确率？
- **RQ2/H2**：HyperTrace 是否降低错误 AI 建议的接受率，并提高错误建议拒绝率？
- **RQ3/H3**：时间线和 provenance 是否降低 cutoff 泄漏判断、提高来源追溯成功率？
- **RQ4/H4**：认知强制提示是否提高适当依赖，而不是简单提高总体 AI 信任？
- **RQ5/H5**：HyperTrace 是否提高置信度校准，并在可接受的时间和工作负荷内完成任务？

#### 6.4.5 主要指标

主指标预先固定为：

- 人机联合判断的 balanced accuracy；
- 正确 AI 建议接受率；
- 错误 AI 建议拒绝率；
- 适当依赖率 = 正确接受 + 正确拒绝的机会数占比；
- 过度依赖率 = 错误建议被接受的比例；
- 证据来源与 cutoff 核验准确率。

次指标包括决策时间、置信度 Brier/ECE、NASA-TLX 或等价工作负荷量表、主观信任，以及时间线、证据展开、反事实查看等交互日志。报告“校准信任”，不要只报告更高的主观信任分。

#### 6.4.6 样本量与统计

受试者数量应由先导数据进行事前 power analysis 决定；在没有先导效应量时，建议目标为 36–48 名受试者，并记录专业审核经验作为协变量。主要分析使用带受试者和案例随机截距的 mixed-effects logistic regression；决策时间对数变换后使用 mixed-effects model。报告条件间 odds ratio/均值差、95% 置信区间、效应量和多重比较校正，而不是只报告 p 值。

#### 6.4.7 离线回放审计

在人因实验前，先对所有案例自动验证：每个解释 evidence ID 可解析、所有记录时间不晚于 cutoff、hash 可重算、状态机事件顺序合法、增量回放不读取未来事件，并测量从事件到 ExplanationPacket 的延迟。离线审计失败的案例不得进入受试者实验。

因此，CHI 论文中的证据链是：

```text
技术回放：无未来泄漏 + 可追溯 + 可恢复
        ↓
受试者研究：准确判断 + 拒绝错误建议 + 适当依赖
```

## 7. 实时解释的准确定位

HyperTrace 的“实时”指事件流回放或在线事件到达后：

```text
INGESTED → PREDICTED → EXPLAINING → EVIDENCE_LINKED
→ READY_FOR_REVIEW → REVIEWED → CONFIRMED/CORRECTED → ARCHIVED
```

系统不在每个事件到来时重新训练模型，而是用冻结模型更新当前 cutoff 的预测、证据和解释快照，并通过 hash-chain 保存全过程。解释对象是：

> “为什么模型在当前时间点认为该账号/局部网络与协同信息行动相关，以及哪些证据支持或反驳该判断？”

## 8. 服务器执行顺序

服务器开启后严格按以下顺序执行，禁止直接运行旧版 `scripts/train_p2.py` 正式训练：

1. **环境与版本冻结**：记录 GPU、CUDA、PyTorch、PyG、git commit 和依赖 lock；建立可写 runtime 目录。
2. **数据审计**：审计 IO-26 字段、时间覆盖、标签映射、campaign 数量和 PII 处理；审计 20 个 synthetic episode 是否满足 26D/provenance/cutoff 契约。
3. **构建 v4 manifest**：所有路径使用相对路径；记录每个 artifact 的大小、SHA-256、schema 和来源；`--require-files` 必须通过。
4. **物化时间图**：按 cutoff 构建 IO-26 与 synthetic 的统一异构图；生成磁盘图缓存，禁止每个 epoch 重建全图。
5. **数据 smoke**：1 个 IO-26 campaign、1 个 synthetic episode、1 个 cutoff，运行前向、反向、时间泄漏测试和 checkpoint 恢复测试。
6. **小型 pilot**：E0/E1/E2 各 1 个 seed、1 epoch；确认显存、neighbor sampling、指标和解释 packet 完整。
7. **正式检测训练**：先完成 IO-26 E0/E1，再执行 E2；每个 fold/seed 独立 checkpoint，支持从最近 step 恢复。
8. **主实验测试**：冻结 checkpoint、阈值和特征契约后，只读取 IO-26 的未见 campaign；Fox8-23、BotSim-24、TwiBot-22 和 MGTAB 当前不运行。
9. **解释 benchmark 与人因案例**：对冻结模型生成 Geo-PG、时间截断、provenance 和反事实结果，最后准备审核员实验案例。

服务器窗口不足时，优先级为：`可访问真实 campaign 数据审计 > 时间图缓存 > smoke/恢复 > E0/E1 pilot > E2`。如果 IO-26 仍受限，按替代方案中的 Crypto-Campaign 顺序执行；不要用旧模型占满窗口。

## 9. 代码改造门槛

当前 `Character Classification/joint_training.py` 和 `scripts/train_p2.py` 仍包含旧版心理特征、多头和联合损失，不能称为 v4 已实现。正式训练前必须完成：

- 新的 `OBSERVABLE_FEATURE_COLUMNS_V4` 和所有 adapter 的 26D 对齐；
- `coordination_label` 统一映射；
- 无 privileged input 的单检测头；
- campaign-disjoint/time-cutoff split runner；
- neighbor mini-batching、磁盘图缓存和断点恢复；
- E0/E1/E2 可复现实验清单和结果 manifest。

在这些门槛通过前，服务器只做数据审计和缓存，不产生论文正式结果。

## 10. 论文结果的诚实写法

只有以下条件同时满足，才能在摘要中填写结果：

- E2 在未见 IO-26 campaign 上优于 E1 或至少保持相当性能；
- E1 相对 E0 的几何/泛化收益有稳定多 seed 证据；
- 解释的时间有效性和 evidence traceability 达到预设阈值；
- 审核员实验显示适当依赖改善，而非单纯更高的 AI 接受率。

否则应缩小主张：把双曲模型写为候选后端，把仿真写为受控评估环境，把实时解释写为原型系统，而不把未验证模块写成已证明贡献。
