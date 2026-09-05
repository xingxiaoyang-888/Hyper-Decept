# HyperTrace CHI Best Paper 写作与 Methods 总纲

> 版本：2026-08-14  
> 用途：作为正文写作、实验补齐、图表制作和投稿审计的唯一上位写作蓝图。  
> 证据纪律：本文严格区分已经完成的形成性访谈、已经完成的模型实验，以及尚待完成的正式解释评估和人因对照实验。方括号字段在获得真实材料前不得自行补写。

## 0. Best Paper 级论文的中心判断

HyperTrace 不应被写成“一种更复杂的双曲图检测器”。CHI 论文的中心问题应是：

> 当协调信息行动检测器可能正确也可能错误时，如何让审核员看到一个足够紧凑、时间范围明确、几何上忠实且来源可核验的证据集合，从而作出自己的最终判断？

论文研究对象是审核员如何核验证据、质疑模型并形成适当依赖。检测器、Lorentz 几何和图算法是实现这一目标的技术材料，而不是论文价值的终点。

建议全文始终维持一条叙事链：

```text
真实审核困难
→ 形成性访谈揭示证据核验需求
→ 提炼设计要求
→ HyperTrace 将模型输出重构为可核验证据
→ 离线实验验证检测与解释是否可信
→ 人因实验验证审核员能否作出更好的最终判断
→ 总结适用于高风险 AI 审核工具的设计知识
```

### 0.1 推荐标题

首选：

> **Beyond Risk Scores: HyperTrace for Evidence-Centered Human Review of Coordinated Information Operations**

备选：

> **HyperTrace: Supporting Auditable Human Review of Coordinated Information Operations**

标题避免使用 `novel hyperbolic detector`、`social bot detection` 或 `role-aware detection`。论文目标是协调信息行动审核，而不是传统 bot/human 分类。

### 0.2 一句话定位

> HyperTrace transforms a coordination detector's alert into a compact, time-scoped, geometry-consistent, and source-verifiable evidence packet that supports reviewers in accepting correct AI recommendations and rejecting incorrect ones.

### 0.3 第一页必须回答的问题

开头一页应让非图学习审稿人立即得到四个答案：

1. **问题是什么？** 风险分数和静态 top-k 图解释不足以让审核员独立核验协调信息行动建议。
2. **为什么重要？** 漏检会放任行动，无依据的升级或封禁也会伤害正常用户；解释还可能放大对错误 AI 建议的依赖。
3. **我们发现了什么？** 形成性访谈揭示审核员需要哪些证据、时间边界、来源信息和纠错操作；最终主题必须来自真实编码结果。
4. **我们贡献了什么？** 一套由形成性证据驱动的可审计工作流，以及验证其技术忠实度和人类决策价值的混合方法研究。

## 1. 三项主要贡献

### C1. 关于协调信息行动审核需求的形成性实证洞见

通过对 `[FI-N]` 名 `[参与者类型]` 的形成性访谈，我们刻画审核员如何判断协调行为、如何处理模型不确定性、如何核验时间与来源，以及现有工具在哪些环节阻碍独立判断。

最终贡献必须写成真实发现，而不是“我们访谈后设计了界面”。建议形成 3-4 个可迁移洞见，例如：

- `[FI-F1：审核员如何从关系、时间和来源之间建立证据链]`；
- `[FI-F2：风险分数或解释如何诱发过度依赖]`；
- `[FI-F3：何种缺失、冲突或时间范围警告会改变最终判断]`；
- `[FI-F4：审核员需要怎样的纠正、升级和请求更多证据流程]`。

这些主题目前只是结构占位符，只有与访谈代码本和引文一致时才能使用。

### C2. 面向审核员的受约束证据重构工作流

HyperTrace 不把解释定义成一组重要性分数，而是定义成满足以下要求的最小证据集合：

- 保持冻结检测器判断；
- 保持 Lorentz 原型距离和测地边际的主要含义；
- 不读取决策时间范围之外的信息；
- 每条证据都能解析到来源记录和版本哈希；
- 当约束无法同时满足时，显示失败或缺失警告，而不是静默生成貌似完整的解释。

工作流把证据重构、反事实删除、时间范围、来源核验和审核员状态流连接在一起。核心创新是解释契约与审核流程，而不是声称发明通用 HGT 或证明双曲空间一定提高准确率。

### C3. 关于证据界面与适当依赖的联合技术-人因评估

论文通过三层证据回答不同问题：

1. 检测器能否在未见真实 operation 上识别协调成员；
2. HyperTrace 证据是否紧凑、忠实、可追溯；
3. 审核员是否因此更准确地接受正确建议、拒绝错误建议并核验来源。

最终贡献不是“用户更信任系统”，而应是关于何种证据设计改善或未改善适当依赖、付出了何种时间和工作负担代价的经验性发现。

## 2. 研究问题

建议使用三个主 RQ，避免技术问题与人因问题各自扩张成一长串假设。

### RQ1：审核需求与设计知识

> How do reviewers currently assess coordination alerts, and what information and interaction mechanisms do they need to independently verify, challenge, and act on an AI recommendation?

证据来源：形成性访谈、编码主题、负例和设计要求映射。

### RQ2：解释的技术可信度

> Can constrained evidence reconstruction produce compact explanations that preserve the detector's prediction and Lorentz decision geometry while respecting temporal scope and provenance requirements?

证据来源：comprehensiveness、sufficiency、sparsity、geometry fidelity、temporal validity、provenance coverage、稳定性和延迟。

### RQ3：审核员的最终判断

> Compared with score-only and conventional static explanations, how does HyperTrace affect reviewers' decision accuracy, rejection of erroneous AI recommendations, evidence verification, decision time, and workload?

主预注册对比：

- HyperTrace vs. Score-only；
- HyperTrace vs. Static explanation。

不把“信任提高”设置为成功标准。正确建议应被适当接受，错误建议应被拒绝。

## 3. 推荐全文结构与篇幅纪律

最终格式使用当届 CHI 指定的 ACM `acmart` 模板；投稿前再次核对当届 CFP。以下页数仅是内部写作预算，不是对官方限制的陈述。

| 板块 | 内部篇幅目标 | 审稿人应获得的信息 |
|---|---:|---|
| Title, Abstract, CCS, Keywords | 0.5 页 | 问题、受众、方法、三类证据、核心发现 |
| 1 Introduction | 1.0-1.3 页 | 为什么是 HCI 问题、研究缺口、贡献 |
| 2 Related Work | 1.2-1.6 页 | 三条文献线如何汇合到本文缺口 |
| 3 Formative Study | 1.5-2.0 页 | 参与者、方法、分析、真实洞见 |
| 4 Design Requirements | 0.6-0.9 页 | 洞见如何约束系统设计 |
| 5 HyperTrace | 2.0-2.8 页 | 工作流、证据重构、几何、时间、来源、UI |
| 6 Evaluation Method | 1.8-2.5 页 | 离线实验、解释实验、人因实验 |
| 7 Results | 1.8-2.5 页 | 技术结果、解释结果、人因结果 |
| 8 Discussion | 1.0-1.4 页 | 设计知识、负结果、实践意义 |
| 9 Limitations, Ethics, Reproducibility | 0.7-1.0 页 | 边界、风险、伦理、材料开放 |
| 10 Conclusion | 0.3-0.5 页 | 回答问题，不重复摘要 |

写作原则：每个技术细节必须服务一个研究问题；每张图表必须能独立说明一个主张；无法改变主张的实现细节放入补充材料。

## 4. Title, Abstract, CCS 与关键词

### 4.1 摘要结构

建议保持 180-230 英文词，单段完成五个动作：

1. **Problem**：协调信息行动审核不能只依赖风险分数或无法核验的图解释。
2. **Formative evidence**：通过 `[FI-N]` 名 `[参与者]` 的访谈发现 `[2-3 个真实主题]`。
3. **System**：介绍 HyperTrace 的最小证据重构、时间范围、几何和来源契约。
4. **Evaluation**：说明受控 LLM-driven coordination、未见真实 operations、解释指标和 `[HS-N]` 人因实验。
5. **Findings/implications**：填写真实检测、解释和人因结果，并指出对高风险审核界面的设计含义。

摘要骨架：

> Reviewers of coordinated information operations must decide whether to act on AI recommendations that may be incomplete or wrong. Existing detectors typically expose risk scores or static graph evidence without clarifying when evidence became available, whether it preserves the detector's structural reasoning, or how it can be independently verified. Through formative interviews with `[FI-N]` `[participants]`, we found `[finding 1]`, `[finding 2]`, and `[finding 3]`. We designed HyperTrace, an evidence-centered review workflow that reconstructs a compact explanation under prediction, Lorentz-geometry, temporal-scope, and provenance constraints. We evaluate HyperTrace using controlled LLM-driven coordination episodes, two unseen real information operations, explanation-fidelity tests, and a `[HS-N]`-participant decision study comparing score-only, static explanation, and HyperTrace interfaces. HyperTrace achieves `[qualified model result]`, `[explanation result]`, and `[human result]`, while `[cost or negative result]`. These findings show `[design implication]` and position auditable explanation as support for independent judgment rather than a mechanism for increasing trust.

### 4.2 CCS 候选

- Human-centered computing → Empirical studies in HCI；
- Human-centered computing → Human computer interaction (HCI)；
- Human-centered computing → Interactive systems and tools；
- Computing methodologies → Artificial intelligence；
- Security and privacy → Social network security and privacy。

最终 CCS 必须通过 ACM CCS 工具生成，不手工伪造代码。

### 4.3 关键词

`human-AI decision making; appropriate reliance; coordinated information operations; explainable graph neural networks; auditable AI; content moderation; provenance; hyperbolic representation learning`

## 5. 1 Introduction

建议使用六段结构。

### 第 1 段：现实任务与受众

描述 trust-and-safety reviewers、平台审核员或调查分析人员面对的任务：从分散在账号、内容、关系和时间中的证据判断是否与协调行动相关。不要以“近年来 GNN 快速发展”开头。

### 第 2 段：为什么现有输出不够

风险分数不能说明证据，普通 top-k 解释也可能：

- 使用事后才出现的信息；
- 展示与模型判断相关但不可核验的边；
- 隐藏证据缺失和模型错误；
- 增加解释带来的错误说服力。

把问题明确转成“支持独立判断”，而不是“让模型更透明”。

### 第 3 段：形成性研究动机

简述 `[FI-N]` 名参与者与访谈目的，给出 2-3 个真实高层发现。形成性研究必须在第一页出现，否则系统设计看起来仍是研究者主观堆叠。

### 第 4 段：方法概念

介绍 HyperTrace 的核心：给定冻结判断和决策时间范围，寻找满足预测、几何、时间和来源约束的紧凑证据，并将约束状态暴露给审核员。

### 第 5 段：评估与诚实结果

可提前写入已经完成的结果：

- 未见 Honduras/UAE operations 上 AUROC 分别为 `0.8850` 和 `0.9005`；
- 60-fold 消融中 Euclidean-HGT 在若干预测指标上优于 Lorentz-HGT；
- 因此本文不声称 Lorentz 普遍提高预测准确率，而检验其是否支持更忠实的几何审计。

解释和人因结果只能在实验完成后填写。

### 第 6 段：贡献列表

使用 C1-C3 的 HCI 表述。贡献中避免出现“首次”“最先进”或“显著提高”，除非有完整检索与统计证据。

## 6. 2 Related Work

相关工作只保留三条主线，每小节结尾明确指出本文继承什么、缺口是什么。

### 2.1 Human Review of Coordinated Information Operations

涵盖：CIB/IO 调查、社交平台审核、分析人员如何处理跨账号证据、协调检测与传统 bot detection 的区别。

缺口句：现有检测工作主要优化分类或活动发现，较少研究审核员如何在模型建议可能出错时核验证据链并形成最终判断。

### 2.2 Explanations and Appropriate Reliance in Human-AI Decisions

涵盖：解释对过度依赖的双刃剑效应、认知强制、错误建议拒绝、适当依赖、内容审核或高风险决策界面。

缺口句：通用解释研究较少处理图证据的时间范围、来源可追溯性与结构语义同时失真的问题。

### 2.3 Graph and Hyperbolic Explanations

涵盖：GNNExplainer、PGExplainer、动态图解释、反事实解释、双曲/异构图表示。明确 HGT 与 Lorentz-HGT 是方法背景，不是 HCI 贡献本身。

缺口句：预测保真并不自动保证解释保持 Lorentz 原型距离、径向组织、时间可用性和来源可验证性。

## 7. 3 Formative Study

这是当前论文最需要先写实的部分。不得先写主题再反向挑引文。

### 3.1 Study Aim

形成性研究回答：审核员如何处理协调行动建议、哪些证据改变判断、现有工具如何导致不确定性或过度依赖、他们如何希望纠正模型。

### 3.2 Participants and Recruitment

正文必须报告：

- `[FI-N]`；
- 职业/研究角色与审核相关经验；
- 纳入和排除标准；
- 招募渠道；
- 地区和语言；
- 报酬；
- 访谈日期和平均时长；
- 是否存在作者与参与者的上下级或利益关系。

用表格报告匿名编号、角色、经验区间和相关任务，不报告可重新识别的单位与精确职位组合。

### 3.3 Interview Procedure

建议按真实流程描述，不得补做不存在的环节：

1. 知情同意与背景问卷；
2. 参与者回忆最近一次或典型的协调行为核验任务；
3. 围绕证据、时间、来源、错误建议和升级决策的半结构化访谈；
4. 若实际使用过原型或概念卡片，说明展示顺序和问题；若未使用，不得写成原型评估；
5. 总结确认和补充意见。

访谈提纲建议在补充材料公开。

### 3.4 Qualitative Analysis

必须依据实际分析选择一种方法，不能混用术语。

**如果已经采用 codebook thematic analysis：**

- 两名研究者先独立开放编码 `[比例或访谈数]`；
- 讨论形成代码本与定义；
- 在代码本稳定后双重编码 `[20%-30%]` 材料；
- 如计算一致性，优先报告适合名义编码的 Krippendorff's alpha，并同时说明协商过程；
- 其余材料由主编码者完成、第二研究者审计；
- 记录反例、代码合并、主题变化和审计轨迹。

**如果采用 reflexive thematic analysis：**

- 报告熟悉、编码、主题生成、审查、命名和写作阶段；
- 说明研究者立场和反思；
- 不把 inter-rater reliability 当作质量标准；
- 通过协作主题讨论、负例和审计备忘录提高透明度。

最终正文只保留真实采用的一种。不要用“达到饱和”作为自动质量声明；若使用 saturation，必须说明定义和判断过程。

### 3.5 Findings

每个主题建议采用统一结构：

```text
主题主张
→ 行为或工作实践描述
→ 2-3 条去标识化引文
→ 反例/参与者差异
→ 对设计的含义
```

建议控制为 3-4 个主题。不要把主题标题直接写成界面功能，例如“用户想要时间线”；应写成更高层的工作实践或张力。

### 3.6 Validity and Reflexivity

报告研究团队与议题的关系、翻译过程、研究者先验、成员检查是否进行、哪些参与者群体没有覆盖，以及形成性发现不能代表所有平台审核工作。

## 8. 4 Design Requirements

形成性主题与设计要求之间必须有显式证据矩阵。

| 形成性发现 | 设计要求 | HyperTrace 实现 | 后续验证 |
|---|---|---|---|
| `[FI-F1]` | DR1 `[例如：证据优先于分数]` | 局部关系、原始事件和证据 ID | evidence verification accuracy |
| `[FI-F2]` | DR2 `[例如：暴露时间范围]` | observed window、scope warning | temporal-scope verification |
| `[FI-F3]` | DR3 `[例如：支持质疑模型]` | incident-edge deletion、reject/request actions | erroneous-advice rejection |
| `[FI-F4]` | DR4 `[例如：可复核来源]` | provenance hash、source record | provenance verification |

正文只保留由访谈证据支持的 DR。没有访谈支撑的技术功能可作为实现选择，但不能包装成 user-derived design requirement。

## 9. 5 HyperTrace: System and Technical Method

Methods 的目标是让 HCI 审稿人理解系统怎样落实设计要求，并让技术审稿人可以复核。不要按代码文件夹顺序介绍。

### 5.1 Workflow and Scope

定义输入：截止时间或明确数据范围内的异构图 `G_{≤t}`、冻结检测器 `f_θ`、待审核账号或局部网络 `v`。

定义输出：

```text
ExplanationPacket = {
  review priority and model recommendation,
  compact typed evidence subgraph,
  temporal scope,
  geometry summary,
  counterfactual effect,
  provenance links and hashes,
  warnings and review state
}
```

当前 Honduras/UAE 真实外测采用 `full-release snapshot`，不是历史在线早期检测。界面和论文必须保留这一警告。

### 5.2 Frozen Coordination Detector

只说明与解释和研究问题有关的内容：

- 用户/内容节点及 typed relations；
- `observable18` 可观察特征；
- intrinsic Lorentz-HGT 表示；
- coordination-only class-balanced loss；
- 到正常/协调原型的 Lorentz 距离；
- 15-checkpoint rank consensus；
- 不使用心理特征、情感分析、角色分类、campaign head 或 next-action head。

定义测地决策边际：

```text
Δ(v) = d_L(z_v, p_normal) - d_L(z_v, p_coordination)
```

`Δ(v)>0` 表示表示点相对更接近 coordination prototype。论文聚合跨 checkpoint 的距离、半径、margin 和排序稳定性，不平均不同模型坐标系中的原始 Poincaré 坐标。

Lorentz-HGT 的角色必须诚实表述：它提供可审计的原型距离和层级表示；现有消融不支持“它普遍比 Euclidean-HGT 更准确”。

### 5.3 Constrained Evidence Reconstruction

这是 Methods 的理论核心。避免把预测、几何、时间、来源和稀疏度写成五个任意加权 loss。建议采用约束优化：

```text
minimize        |E'|
subject to      D(fθ(G≤t), fθ(G[E'])) ≤ εpred
                |ΔG(v) - ΔE'(v)| ≤ εgeo
                timestamp(e) ≤ t          for every e ∈ E'
                resolvable_source(e) = 1  for every e ∈ E'
```

其中：

- `E'` 是展示给审核员的证据边/事件；
- 第一项限制 prediction divergence；
- 第二项保持 Lorentz decision geometry；
- 时间和来源是硬约束，不通过权重进行折中；
- 目标只最小化证据量，因而自然对应 sparsity。

如果不存在满足约束的解释，系统返回 `insufficient evidence`、`temporal scope unavailable` 或 `unresolved provenance`，而不是降低约束后继续展示。

建议实现一个确定性的 budgeted greedy procedure：

1. 从审核账号的 typed candidate neighborhood 开始；
2. 计算删除候选证据后的预测和 geodesic margin 变化；
3. 按最小影响依次剪枝；
4. 一旦继续删除会违反 prediction 或 geometry constraint 即停止；
5. 输出保留证据、被删除证据、约束状态和完整 provenance。

该算法目前需要在最后服务器窗口前完成实现和单元测试；正文不能在实现之前声称已完成正式 constrained explainer。

### 5.4 Counterfactual and Explanation Metrics

对最终 `E'` 定义：

- **Comprehensiveness**：从完整图删除 `E'` 后预测/排序下降；
- **Sufficiency**：只保留 `E'` 时相对完整图的预测差异；
- **Sparsity**：`|E'| / |E_candidate|`；
- **Geometry fidelity**：完整图与解释子图的 `Δ(v)`、原型距离或径向顺序差异；
- **Temporal validity**：超出 cutoff/range 的证据比例，目标为 0；
- **Provenance coverage**：可解析 evidence ID 和哈希一致的证据比例；
- **Stability**：跨冻结 checkpoint 的 rank、margin 和证据集合重合；
- **Latency**：解释生成和数据解析耗时。

与 random-edge 和 degree-matched evidence 对照，防止仅因为保留高连接度边而获得较好结果。

### 5.5 ExplanationPacket and Audit Contract

每个 packet 必须记录：

- case ID 和不可逆匿名账号 ID；
- dataset-relative review percentile，而不是冒充校准概率；
- 模型版本、checkpoint manifest hash；
- evidence ID、timestamp、relation type、source file hash；
- 数据时间范围或 cutoff；
- prediction/geometry constraint 状态；
- 删除与 keep-only 反事实结果；
- unresolved evidence 与 scope warnings；
- append-only review actions。

真实标签与私有 label key 永不进入前端 packet。

### 5.6 Human Review Interface

正文为当前英文 UI 预留一张跨栏图。界面按审核任务而不是模型模块组织：

1. Review queue and model recommendation；
2. Critical account neighborhood；
3. Observed activity window；
4. Lorentz geometry summary；
5. Incident-edge counterfactual；
6. Source evidence and provenance；
7. `Confirm escalation`、`Reject recommendation`、`Request more evidence`。

界面不展示心理画像、角色名称、训练 loss、epoch 或无法解释给审核员的内部特征。

## 10. 6 Evaluation Method

### 6.1 Evaluation Overview

用一张表说明三种评估分别支持哪项贡献：

| 评估 | 数据/参与者 | 回答的问题 | 不支持的主张 |
|---|---|---|---|
| Detector evaluation | 20 synthetic episodes + Honduras/UAE | 是否学到可迁移协调模式 | 不证明 Lorentz 更准确，也不证明人类受益 |
| Explanation evaluation | 冻结模型与盲化案例 | 解释是否忠实、紧凑、可追溯 | 不证明审核员会正确使用 |
| Human-subject study | `[HS-N]` reviewers/proxies | 是否改善最终判断和适当依赖 | 不证明所有平台/国家都泛化 |

### 6.2 Detector Evaluation

训练：DeepPersona/OASIS 20 个受控 LLM-driven coordination episodes，5 个场景，LOSO，3 个 model seeds。

真实图：Honduras/UAE 无标签图只用于 schema-safe warm-start；目标 operation 标签不参与训练、checkpoint 选择、阈值或排序。

外测：

- UAE warm-start → synthetic coordination-only training → Honduras test；
- Honduras warm-start → synthetic coordination-only training → UAE test。

指标：AUROC、AUPRC、Macro-F1、Balanced Accuracy、Brier、ECE；外测重点解释类别不平衡下的 AUPRC 和排序价值。

消融：

- Lorentz-HGT + real warm-start；
- Lorentz-HGT scratch；
- Euclidean-HGT scratch；
- Lorentz-HGT without temporal input。

使用相同 scenario/seed 配对 bootstrap。已知结果必须如实写明 Euclidean scratch 在若干预测指标上更高。

### 6.3 Explanation Evaluation

在读取用于案例分层的真实标签前，先冻结模型、解释预算、约束阈值和 review queue。若需要用标签构造正确/错误 AI 建议案例，必须在模型与解释选择完全冻结后进行。

建议每个 operation 至少覆盖：

- high-priority cases；
- medium-priority controls；
- true/false positives and true/false negatives；
- 不同关系密度和活动跨度。

比较：

- Static top-k explanation；
- HyperTrace constrained evidence；
- Random-edge control；
- Degree-matched control。

报告均值、95% CI、按 case/checkpoint 配对的 bootstrap，以及失败率和生成延迟。解释参数不得根据最终人因结果回调。

### 6.4 Human-Subjects Study

#### Participants

目标人群优先为具有 trust-and-safety、调查、事实核查、平台治理或相关研究经验的参与者；若采用受训代理审核员，必须说明其与专业审核员的差异，并把专业经验作为协变量。

样本量由 pilot effect size 和事前 power analysis 决定。内部初始目标可设为 48-72 人，但最终论文只能报告实际 power analysis 和招募结果。

#### Design

推荐三条件 within-subject 设计：

1. **Score-only**：review percentile + AI recommendation；
2. **Static explanation**：分数 + 固定重要边/事件；
3. **HyperTrace**：分数 + 关系、时间、几何、反事实、来源和审核状态流。

使用 Latin-square 或 balanced incomplete block 分配条件与案例。每名参与者不能在不同条件重复看到同一个 case。

建议每人完成：

- 2 个不计分练习案例；
- 12 个正式案例，每条件 4 个；
- 各条件平衡 AI 正确/错误和正/负建议；
- 总时长控制在 45-60 分钟。

错误建议优先来自冻结模型自然产生的 FP/FN。若必须实验性翻转建议，应明确标记为 controlled erroneous-advice manipulation，且三条件使用相同底层证据。

#### Task

每个案例要求参与者：

1. 判断是否与协调信息行动相关；
2. 接受或拒绝 AI 建议；
3. 可选择请求更多证据；
4. 给出置信度；
5. 完成一个来源/时间范围核验问题；
6. 可选填写一句证据理由。

主要任务是最终判断，不是回忆模型训练流程。

#### Outcomes

预注册主要指标：

- balanced decision accuracy；
- erroneous-advice rejection rate；
- appropriate reliance；
- evidence verification accuracy。

次要指标：

- decision time；
- confidence calibration；
- request-more-evidence behavior；
- NASA-TLX；
- usability 与主观信任，仅作为辅助解释。

#### Statistical Analysis

- 二元正确性：mixed-effects logistic regression；
- 固定效应：condition、AI correctness、二者交互、experience；
- 随机效应：participant 和 case 随机截距，数据支持时加入 condition 随机斜率；
- 决策时间：log-transform 后的 linear mixed-effects model；
- 置信度校准：confidence-correctness gap 或 participant-level calibration model；
- 主对比使用 Holm 校正；
- 报告 odds ratio/估计差异、95% CI、效应量和原始分布，不只报告 p 值。

对开放回答或实验后访谈使用与形成性研究区分开的质性分析，并说明如何与定量结果整合。

#### Frontend Deployment

正式实验前端可部署在 Hugging Face Space，但必须使用后端隔离：

- 前端只有 blinded case pool；
- 私有标签、条件分配和答案键不进入静态 JavaScript；
- 每个 session 使用匿名研究 ID；
- 记录条件、case、回答、置信度、耗时和界面操作；
- 原始社交内容按许可和伦理要求截断、改写或去标识化；
- 正式实验优先使用私有 Space 或带访问控制的后端。

## 11. 7 Results 的推荐组织

不要按脚本或指标堆叠结果。每个小节先给一个回答 RQ 的结论句，再给图表和边界。

### 7.1 Detector Validity

报告 synthetic LOSO 与真实 operation-disjoint 结果。强调真实 AUROC 与 AUPRC 的差异和方差，不用 synthetic `~0.98` 指标替代真实表现。

### 7.2 What Lorentz Geometry Did and Did Not Provide

明确负结果：Euclidean scratch 在 AUPRC、AUROC、Balanced Accuracy 和 ECE 上更高。接下来检验 Lorentz 是否在 geometry fidelity 与审核证据组织上提供价值。

### 7.3 Explanation Fidelity and Auditability

填入 comprehensiveness、sufficiency、sparsity、geometry fidelity、temporal/provenance 结果和随机/度数对照。

### 7.4 Reviewer Decisions and Appropriate Reliance

先报告主要行为指标，再报告时间、工作负担和信心。重点分析 condition × AI correctness：HyperTrace 是否主要帮助参与者识别错误建议，而不是无差别提高 AI 接受率。

### 7.5 Qualitative Accounts of Use and Failure

说明参与者怎样使用或误用时间线、几何指标、反事实和 provenance；保留负面案例，例如界面复杂度、误解测地边际或证据过载。

## 12. 8 Discussion

Discussion 不重复结果，应产出可迁移的 HCI 知识。

### 8.1 From Explanation to Evidence Verification

讨论为什么高风险审核界面不应以“解释模型”为唯一目标，而应支持用户验证来源、发现缺失并保留拒绝路径。

### 8.2 Appropriate Reliance Requires Actionable Disagreement

讨论哪些设计让审核员能够反对 AI：错误建议案例、反事实、范围警告和请求更多证据。若人因结果无显著提升，分析是否由复杂度、专业知识或证据质量导致。

### 8.3 The Role of Geometry When Accuracy Does Not Improve

把 Euclidean 更好的结果当作重要边界：双曲表示的价值不能由准确率预设，而应由结构语义、解释 fidelity 和人类使用证据验证。若这些指标也无优势，应降低 Lorentz 的论文地位。

### 8.4 Design Implications

建议提炼 3-4 条跨系统设计含义：

- 将时间和来源设为不可静默放松的解释约束；
- 显示模型错误可能性并提供明确纠正动作；
- 使用 dataset-relative ranking 和稳定性而不是伪精确概率；
- 当解释不满足约束时展示失败，而不是生成完整感幻觉。

### 8.5 Implications for Practice

讨论 trust-and-safety 团队怎样把 HyperTrace 接入现有队列、证据保全、升级和复核流程，以及需要哪些组织政策和权限控制。

## 13. 9 Limitations, Ethics, and Reproducibility

### 9.1 Limitations

必须主动写明：

- synthetic supervision 与真实 operation 之间存在 domain gap；
- Honduras/UAE 是 full-release snapshot，不能证明实时早期检测；
- Lorentz 没有全面提高预测准确率；
- 真实 operation 数量有限；
- 形成性样本和人因样本的地区、平台与专业代表性有限；
- 界面可能增加工作负担；
- 解释约束仍不能证明因果协调意图。

### 9.2 Ethics and Privacy Statement

正文和 ACM Ethics and Privacy Statement 中报告：

- 形成性访谈伦理许可/豁免 `[FI-IRB]`；
- 正式人因实验伦理许可 `[HS-IRB]`；
- 知情同意、退出权、报酬和数据保存期限；
- 录音、转录、匿名化和访问权限；
- 公开社交数据的去标识化与最小披露；
- 误报对用户权益和调查升级的风险；
- 系统不自动执行封禁，只支持人工审核；
- 数据与模型可能被用于监控或针对政治表达的风险。

### 9.3 Reproducibility Package

匿名补充材料建议包含：

- 训练、外测、解释和审计代码；
- aggregate-only 结果和 hashes；
- feature/manifest/ExplanationPacket schema；
- 去标识化访谈提纲、代码本和主题审计说明；
- 人因实验预注册、任务脚本、条件材料和分析代码；
- 无标签或合成演示案例；
- 3-5 分钟无旁白或英文字幕演示视频；
- 环境和依赖 lock。

不公开原始受限数据、真实标签键、模型私钥、参与者身份或未经许可的原始内容。

## 14. 图表计划

### Teaser Figure：Reviewer-Centered Workflow

展示：AI alert → constrained evidence packet → reviewer verifies/challenges → final decision。模型架构只占小面积，审核员与证据交互是视觉中心。

### Figure 1：Formative Findings to Design Requirements

形成性主题 → DR1-DR4 → HyperTrace 功能 → 评估指标。该图是 HCI 叙事的桥梁。

### Figure 2：Constrained Evidence Reconstruction

完整异构图、决策时间范围、候选证据、最小证据子图、prediction/geometry/provenance constraints 和 failure warnings。

### Figure 3：HyperTrace Review Interface

当前英文 UI 的跨栏截图，标注七个关键区域。截图值若为演示值必须写明 illustrative，不可伪装为实验结果。

### Figure 4：Human Study Conditions and Procedure

Score-only、Static explanation、HyperTrace 三个条件，以及 case allocation、最终判断和结果记录流程。

### Figure 5：Main Human Results

使用 effect-size forest plot 或 predicted-probability plot，重点展示 AI 正确/错误情况下三条件的适当依赖，不只画柱状均值。

### 核心表格

1. Formative participants；
2. Findings → design requirements；
3. Data and evaluation protocol；
4. Detector and mechanism ablations；
5. Explanation fidelity；
6. Human-study mixed-effects results。

## 15. 主张-证据冻结表

| 论文主张 | 当前状态 | 投稿前要求 |
|---|---|---|
| 模型学到了受控协调模式 | 已支持 | 保留 LOSO 结果与划分审计 |
| 模型迁移到未见真实 operations | 有条件支持 | 使用 operation-disjoint AUROC/AUPRC 和范围限定 |
| Lorentz 比 Euclidean 更准确 | 不支持 | 不得声称 |
| Lorentz 支持更好的几何审计 | 尚待完整支持 | geometry fidelity + human/explanation evidence |
| 解释可追溯且不读取标签 | preflight 支持 | 扩展到正式案例与 provenance coverage |
| 解释忠实且紧凑 | 未完成 | comprehensiveness/sufficiency/sparsity |
| HyperTrace 改善审核最终判断 | 未完成 | 正式人因实验 |
| 形成性研究产生设计洞见 | 用户报告已完成 | 填入真实样本、分析、主题、引文和伦理信息 |
| 系统适用于实时早期检测 | 不支持 | 当前只能写 full-release snapshot；需另做 cutoff replay 才能升级主张 |

## 16. 形成性访谈待填信息表

在开始撰写 Section 3 前，需要整理以下真实信息：

```text
Study dates:
Ethics/IRB status and ID:
Participant count:
Participant roles:
Relevant experience range:
Countries/regions and interview language:
Recruitment channel:
Compensation:
Mean/range interview duration:
Recording/transcription procedure:
Interview guide:
Whether a prototype or scenario was shown:
Analysis approach actually used:
Number and role of coders:
Double-coded proportion:
Reliability statistic, if appropriate:
Final themes and definitions:
Negative/deviant cases:
De-identified quotations:
Data retention and access policy:
```

如果只有一名编码者或尚未完成系统编码，不应虚构双编码；可以在投稿前补做审计性第二编码或采用透明的 reflexive thematic analysis，但必须与实际过程一致。

## 17. 人因实验前端预留位置

正文预留：

- Section 5.6：界面设计与形成性需求映射；
- Figure 3：HyperTrace UI；
- Section 6.4：实验条件、任务、记录和统计；
- Figure 4：三条件流程；
- Supplementary Video：完整审核案例；
- Appendix：界面字段、事件日志 schema、任务说明和理解检查。

Hugging Face 部署不写成技术贡献。它只是可复现实验基础设施。正文关注界面如何支持核验、质疑和最终决策。

## 18. 写作顺序

### 现在即可完成

1. 整理形成性访谈 metadata、代码本、主题和去标识化引文；
2. 写 Section 3 Formative Study；
3. 从真实主题写 Section 4 Design Requirements；
4. 重写 Introduction，使形成性发现进入第一页；
5. 写 Related Work 和 Section 5.1-5.2；
6. 完成 UI 图位与人因方法预注册草案。

### 最后服务器窗口后完成

1. 补写 constrained explainer 的实际算法与参数；
2. 填写正式 explanation metrics；
3. 更新 Results 和 Discussion 中 Lorentz 的边界；
4. 生成 blinded human-study case pool。

### 人因实验后完成

1. 锁定统计分析与结果；
2. 写 appropriate reliance 与错误建议拒绝的主发现；
3. 完成定量-质性整合；
4. 更新摘要、贡献三、Discussion 和 Conclusion；
5. 提交匿名材料、视频和伦理声明。

## 19. Best Paper 级投稿自检

- 第一页是否在模型名称之前说明审核员和现实决策风险？
- 三项贡献是否至少两项属于 HCI 知识或人机系统贡献？
- 形成性主题是否有真实引文、反例和分析过程？
- 每个设计要求是否能回溯到形成性证据？
- 每个系统功能是否能映射到一个研究问题和评估指标？
- 是否避免把 Lorentz 的预测负结果藏起来？
- 是否证明解释不仅可生成，而且忠实、紧凑、可核验？
- 人因实验是否包含模型错误案例和最终判断？
- 是否报告效应量、95% CI、随机效应和多重比较？
- 是否区分 full-release snapshot 与 real-time cutoff？
- 是否彻底排除心理特征、情感分析、角色分类和旧 XGBoost 叙事？
- UI 图、算法图和数据表能否脱离正文独立理解？
- Ethics and Privacy Statement 是否覆盖参与者、公开社交数据和误报风险？
- 匿名材料是否足以复核主张且不泄漏受限数据？
- 摘要中的每个结果是否都能指向正式表格或统计模型？

Best Paper 的目标不应转化为“加入更多模块”。真正需要追求的是：问题重要、形成性证据可信、设计逻辑连贯、技术机制可证伪、人因结果回答现实决策问题，并且负结果和伦理边界都被清楚呈现。
