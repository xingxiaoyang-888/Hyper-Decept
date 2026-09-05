# HyperDecept 联合双曲训练与数据构建方案 v2

> **归档说明（2026-08-04）**：本文档保留用于追溯早期设计，不再作为正式训练规范。
> 其中 50 timestep、38 个仿真 DB、分离 real/synthetic bot head、relation loss、
> 七项人工加权损失、MGTAB adapter 未完成和 P2 runner 未完成等描述均已过时。
> 当前待审阅规范见
> `docs/HyperDecept_联合双曲训练与数据构建方案_v3.md`。

## 1. 研究目标

训练一个面向动态协同欺骗的域感知 Lorentz-HGT：真实数据负责现实 bot/human
监督，DeepPersona/OASIS 仿真数据提供角色、campaign、攻击阶段和下一动作等机制监督。
所有数据源保留 provenance，不将生成标签伪装成真实世界标注。

论文中的方法贡献定义为 **domain-aware episodic sim-to-real hyperbolic
multi-task training**，而不是简单的数据拼接或数据增强。

## 2. 数据总体设计

### 2.1 真实域

| 数据集 | 用途 | 划分 |
|---|---|---|
| TwiBot-22 | 主要真实训练和内部测试 | 使用官方 node-level train/val/test |
| MGTAB | 外部真实验证；多源真实训练消融 | P1 完全外测，P2 只使用其 train split 联合训练 |
| Fox8-23 | 新时期外部测试候选 | 获得数据、许可证和标签审计通过后启用 |
| BotSim-24 | 独立仿真 OOD 测试 | 不替代真实数据，不和 DeepPersona 混作同一生成域 |

TwiBot-22 的 1000 用户文件只用于本地验收。最终服务器实验使用自定义 ID
manifest 或官方完整划分；1000 不是固定核心用户规模。

真实数据允许的监督：bot/human 标注。心理特征是模型提取特征，不是心理学真值；
角色、campaign 和攻击阶段在没有人工标注时必须保持 unavailable。

### 2.2 仿真主语料

主规模固定为 `N=2000`：

```text
5 scenario families × 4 simulation seeds × 2000 agents
= 20 main DBs
= 40,000 synthetic agent instances
```

默认场景：

1. `leader_amplifier`：Leader–Amplifier 层级传播；
2. `bridge_infiltration`：跨社区 Bridge 渗透；
3. `synchronized_boosting`：同步转发/点赞；
4. `persona_drift`：长期人格和表达漂移；
5. `adaptive_evasion`：被检测后降低同步性并改变策略。

默认 simulation seeds：`11, 22, 33, 44`。模型初始化 seeds 单独使用
`7, 17, 27`；模型 seed 不需要重新生成 DB。

### 2.3 规模外推语料

人数递进不覆盖所有场景，只覆盖两个具有代表性的场景：

```text
leader_amplifier + adaptive_evasion
× N={500, 1000, 5000}
× seeds={101, 102, 103}
= 18 scale DBs
= 39,000 synthetic agent instances
```

主语料中的 `N=2000` 作为规模曲线的第四个点。因此完整规模测试是
`N={500,1000,2000,5000}`，但避免了 5 场景 × 4 人数的全笛卡尔积。

默认核心执行计划（当前确定可构建的数据线）：

```text
2 core real dataset entries
20 main simulation DBs
18 scale simulation DBs
= 40 plan entries
= 38 synthetic DBs / 79,000 synthetic agent instances
```

完整论文数据蓝图再加入 Fox8-23 和 BotSim-24，共 42 条 plan entries。二者在
完成下载、许可和标签来源审计前保持 optional external benchmark 状态，不计入
当前必须生成的 38 个 DeepPersona DB。

### 2.4 两套真实数据协议

```text
P1 External Holdout:
  train = TwiBot-22 train + DeepPersona/OASIS train episodes
  test  = TwiBot-22 test + MGTAB + Fox8-23 + BotSim-24

P2 Multi-source Real:
  train = TwiBot-22 train + MGTAB train + DeepPersona/OASIS train episodes
  test  = TwiBot-22 test + MGTAB test + Fox8-23 + BotSim-24
```

**P2 是我们的主训练路线**：它回答“真实多源数据和仿真机制监督联合训练，是否
能提高检测、跨域泛化和解释质量”。P1 不是第二个项目，而是严格外部留出对照：
完全不使用 MGTAB 的训练标签，再测试 MGTAB，以量化 P2 的多源数据增益。
论文中只维护一套模型和一套联合训练器，分别用不同的 manifest assignment 运行
P1/P2；不能在 P2 使用 MGTAB 训练后仍把 MGTAB 称为“完全未见外部数据”。

## 3. 每个仿真 Episode 的文件契约

每个 episode 必须产生：

```text
seed_XX.db                 完整事件数据库
seed_XX.csv                Agent/profile 输入
seed_XX.features.csv       明确列名的 26 维用户特征
seed_XX.labels.csv         bot、role 等生成标签及来源
seed_XX.event_targets.csv  当前因果时间窗的 campaign/next_action 目标
seed_XX.manifest.json      scenario、seed、N、模型、prompt 和阶段 provenance
```

`event_targets.csv` 在一个训练窗口内必须是一名用户一行。若一个 DB 需要多个
时间截点，应输出多个 window artifact/manifest；训练器禁止自动选择最后一行，
避免用未来事件预测过去。

建议仿真阶段固定 50 个时间步，并显式记录：

```text
benign_warmup
coordination_onset
propagation_escalation
adaptive_evasion
aftermath
```

阶段标签、campaign ID 和角色必须来自生成配置或仿真日志，不能在事后根据模型
输出补造。

## 4. 划分与防泄漏

### 4.1 真实图

- 按官方 node split 建立 train/validation/test mask；
- 相同真实用户不得跨数据集派生文件进入多个 split；
- 外部邻居是无标签上下文，不是第三分类标签；
- 同时报告 transductive 和 inductive 设置时，必须明确两者边界。

### 4.2 仿真图

主实验采用 Leave-One-Scenario-Out：

```text
held-out scenario 的 4 个 seeds → test
其余 4 个场景的 seed 44       → validation
其余 4 个场景的 seeds 11/22/33 → train
```

五个场景依次作为 held-out scenario。规模 DB 只用于 scale test，不进入主训练。
仿真本地 user ID 可以在不同 DB 中重复，因为身份作用域是 episode；scenario + seed
+ N 的组合不得重复或跨 split。

## 5. 模型架构

```text
TwiBot-22 --------------------┐
                              ├─ domain-specific feature adapter
MGTAB ------------------------┤
                              ├─ shared intrinsic Lorentz-HGT
DeepPersona/OASIS ------------┘
                                  ├─ real bot head
                                  ├─ synthetic bot head
                                  ├─ role head (synthetic only)
                                  ├─ campaign metric head (synthetic only)
                                  ├─ next-action head (synthetic only)
                                  └─ relation reconstruction objective
```

### 5.1 情感/心理特征的定位

`emotional_analysis/` 是上游可观测特征引擎，不是一个被删除的旧分支，也不直接
产生“真实人格真值”。它输出八个可追溯的行为/语言代理特征，进入 26 维用户输入：

```text
Empathy_Gap_Mean/Max
Dark_Triad_Mean/Max
Contagion_Mean/Max
Volatility_Mean/Max
```

这些特征用于：

1. user feature adapter 的输入；
2. 白盒风险头的概念贡献；
3. 解释卡片中的语言/行为证据链接；
4. 去掉心理特征的消融实验。

论文表述使用 “psychology-inspired linguistic/behavioral proxies”，不把模型输出
写成受试者真实人格诊断。对 TwiBot/MGTAB 只报告检测和解释相关指标，不声称角色
或人格有现实世界真值。

### 5.2 角色分类的定位

角色分类保留，但不作为与 bot/human 平行的最终分类结论：

- 仿真域：`Leader`、`Amplifier`、`Bridge`、`Member` 等由生成器明确提供，作为
  privileged auxiliary supervision；
- 真实域：没有可靠角色标注时，角色是 Lorentz 表示上的结构发现/不确定性结果，
  不能当作第三种真实标签；
- 角色头、半径和关系注意力用于解释“模型为什么认为该用户处于某种协同位置”，
  而不是把一个聚类名称包装成事实。

因此论文中可以说 **role-aware hyperbolic explanation**，不能说“在 TwiBot 上
准确识别真实 Opinion Leader”，除非另有人工或官方标注。

共享隐藏状态保持在 Lorentz 流形上，分类时通过 origin log map 得到共同切空间表示。
真实与仿真使用不同输入 adapter 和 bot head，避免输入分布及概率基率被强制混合。

## 6. 损失函数

```text
L = λreal Lbot_real
  + λsyn Lbot_syn
  + λrole Lrole
  + λcampaign Lcampaign
  + λtemporal Lnext_action
  + λrelation Lrelation
  + λdomain Lcross_domain_hyperbolic_supcon
```

默认 domain alignment 直接使用 Lorentz 测地距离进行跨域有监督对比学习，只把
同类别的真实/仿真表示作为正对，不把 bot 和 human 无条件对齐，从而降低 domain
collapse 风险。class-conditional CORAL 仅保留为消融基线。真实类别不平衡通过
训练集统计得到的 positive class weight 处理，不能使用 test prevalence 调参。

几何解释损失属于解释器训练阶段，不和检测主损失混写。真实数据没有角色真值时，
不施加 Leader 半径排序监督；角色径向约束只用于有生成真值的仿真数据或独立消融。

## 7. 训练阶段

1. **数据预检**：文件、哈希、schema、能力声明、ID 和 split 审计；
2. **真实图自监督预训练**：关系重构、masked relation，并使用心理代理特征作为
   可观测输入；
3. **仿真多任务预训练**：bot、role、campaign、next action；
4. **域交替联合训练**：每一步一批真实 seed users + 一批仿真 seed users；
5. **真实域校准**：只使用真实 validation 调整阈值和 calibration；
6. **LOSO 场景测试**：未见攻击场景；
7. **MGTAB 外部测试**：真实跨域泛化；
8. **规模外推测试**：500/1000/2000/5000；
9. **Geo-PGExplainer 与实时证据追踪评估**。

服务器训练使用异构 NeighborLoader，以 user 为 seed node。监督损失只覆盖 seed
users，采样的上下文用户只参与消息传递和关系重构。

## 8. 必做基线与消融

```text
Euclidean HGT
Euclidean HGT + Poincare projection
Intrinsic Lorentz-HGT real-only
Intrinsic Lorentz-HGT synthetic-only
synthetic → real sequential transfer
naive real/synthetic concatenation
proposed shared encoder + domain-specific heads
proposed - conditional domain alignment
proposed - role/campaign/temporal auxiliary tasks
proposed - relation objective
```

SAHG、HypHGT 官方实现/可复现实现在数据与许可证允许时作为强基线。

## 9. 评价指标

真实检测：AUROC、AUPRC、Macro-F1、Balanced Accuracy、Brier、ECE。

跨域：TwiBot→MGTAB、LOSO 未见场景、未见 simulation seed、规模外推性能下降。

仿真辅助任务：role accuracy/NMI/ARI、campaign recovery、next-action accuracy。

解释：prediction fidelity、geometry fidelity、role fidelity、evidence traceability、
解释稳定性及审核员适当依赖指标。

## 10. 服务器执行顺序

先生成计划文件：

```bash
python -m data_processing.episode_manifest create \
  --output /srv/hyperdecept/dataset_plan.json \
  --simulation-root /srv/hyperdecept/simulation \
  --twibot-root /srv/datasets/twibot22 \
  --mgtab-root /srv/datasets/mgtab \
  --fox8-root /srv/datasets/fox8-23 \
  --botsim-root /srv/datasets/botsim-24
```

若 Fox8-23 或 BotSim-24 尚未通过下载/许可审计，先省略对应参数，生成 40 条核心
执行计划；审计通过后重新生成 42 条完整论文计划。

上传/生成数据后执行强制预检：

```bash
python -m data_processing.episode_manifest validate \
  --input /srv/hyperdecept/dataset_plan.json \
  --require-files
```

只有预检无错误、LOSO 划分审计无错误、所有 26 维特征列和 node ID 对齐后，才
开始 GPU 训练。

## 11. 当前代码状态

- `data_processing/episode_manifest.py`：计划生成、序列化、artifact 预检、LOSO
  划分、用户/campaign/scenario 泄漏审计；
- `Character Classification/joint_training.py`：manifest loader、domain-specific
  adapter、多任务 Lorentz-HGT、双曲跨域对比学习、条件 CORAL 消融、关系负采样、邻居采样 target 对齐、
  检测与 calibration 指标、checkpoint；
- `Character Classification/graph_builder.py`：显式保留 user node ID 顺序；
- 当前仍需在服务器运行前完成：MGTAB 原始适配器、仿真阶段/campaign sidecar
  输出、端到端 fold runner 和服务器真实数据 smoke test。
