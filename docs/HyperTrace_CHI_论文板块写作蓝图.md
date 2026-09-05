# HyperTrace: CHI 论文板块写作蓝图

> 用途：把当前 HyperTrace / HyperDecept P2 项目映射到 ACM CHI 主论文结构，作为写作、实验和审稿风险控制的单一蓝图。
>
> 状态约定：`[已实现]` 表示代码或协议已有依据；`[待实验]` 表示必须在服务器正式运行后填写；`[待审批]` 表示必须以学校/机构实际伦理审批为准。所有百分比、显著性检验和用户数量在获得真实结果前都不得填写为具体值。

## 0. 论文定位

### 建议工作标题

**HyperTrace: Geometry-Consistent and Evidence-Traceable Explanations for Coordinated Deception Detection**

中文工作名可写作：**HyperTrace：面向协同欺骗检测的几何一致、时间可追溯解释系统**。

标题强调解释系统和人机协作，不把“使用双曲 HGT”单独包装成全部贡献，也不声称首次提出双曲 HGT。

### 一句话定位

HyperTrace 将面向协同欺骗检测的 intrinsic Lorentz heterogeneous encoder、几何一致解释、时间证据追踪和审核员界面整合为一个可审计的解释工作流，并通过离线实验和受试者研究检验解释是否提高人类的适当依赖，而不是只提高对 AI 的盲目信任。

### 核心研究问题

> 当协同欺骗行为具有层级结构、时间适应和人格伪装时，解释系统如何同时保留模型预测、双曲几何、网络角色和原始证据的含义，并帮助审核员形成与模型正确性相匹配的依赖？

### 三项主贡献

1. **四维解释质量框架。** 将解释质量从单一 prediction fidelity 扩展为 prediction fidelity、geometry fidelity、role fidelity 和 evidence traceability 四个可计算维度。
2. **HyperTrace 系统。** 提出面向 Lorentz 异构表示的 Geo-PGExplainer 风格解释、时间截断/未来事件隔离、统一 `ExplanationPacket` 和可复核证据链。安全表述是“在双曲协同图上实证显示欧式解释可能保持预测保真，却损伤层级/测地线保真”，不要在没有形式证明时写成数学必然性。
3. **以适当依赖为目标的人机评估。** 通过审核员工作台比较无 AI、概率、传统解释和 HyperTrace 四种条件，测量正确接受、错误拒绝、纠错率、决策时间、工作负荷和信任校准。

### 不应使用的表述

- 不写“首次提出真正的双曲 HGT”。已有 HypHGT 预印本，应将其作为最新架构/基线引用。
- 不写“解释一定会发生几何失真”，除非另有完成的形式证明。建议写“普通 PGExplainer 的欧式 mask 优化没有显式几何约束，因此存在几何失真风险，我们在实验中量化该风险”。
- 不把 TwiBot/MGTAB 的公开 profile 特征称为真实心理诊断；统一写作 `psychology-inspired linguistic/behavioral proxies`。
- 不把仿真生成的 role、campaign 或 attack phase 当作真实世界标签；明确标记为 synthetic privileged supervision。
- 不把“用户更信任 AI”作为主要成功标准；CHI 重点是 appropriate reliance 和 error-adjusted performance。

## 1. ACM/CHI 模板执行规则

模板 PDF 的关键要求如下，最终 LaTeX 必须以 CHI 当年 CFP 和 ACM TAPS 检查器为准：

- 使用 `\documentclass[sigchi,review,anonymous]{acmart}`（投稿阶段的匿名参数按当年 CHI 要求调整）；camera-ready 再替换为正式 rights 信息。
- 不手工修改页边距、字体、字号、行距、段间距或用 `\vspace` 强行压缩版面。
- 每位作者单独使用 `\author`、`\affiliation`、`\email`；投稿版不能泄露身份、仓库地址或可识别的机构信息。
- 第一页包含 title、author metadata、abstract、CCS Concepts、keywords；正式版本还包含 ACM Reference Format 和 rights block。
- 每张图使用 `figure`，图注放在图下；每张图必须有不超过 2000 字符的 `\Description{}`，描述信息而非重复图注。
- 表格使用 `booktabs` 风格，表题放在表上方；宽表使用 `table*`，不要用截图代替可访问表格。
- 参考文献使用 `ACM-Reference-Format` BibTeX；补齐作者、年份、venue、页码、DOI 或稳定 URL。
- 致谢使用 `acks` 环境，放在参考文献前；伦理与隐私/社会影响声明应单独覆盖公开数据、受试者、误导性 AI 建议和潜在滥用风险。
- 附录可放任务界面、完整指标、数据 schema、提示词摘要、统计检验和额外消融，但不能把核心贡献全部移到附录。

## 2. 推荐全文结构与写作任务

### 2.1 Title, Abstract, CCS, Keywords

#### Abstract 四段式（约 180-250 words，英文最终稿）

1. **Problem:** 协同欺骗网络呈现层级传播、异构关系和时间适应，现有 GNN 解释通常在欧式空间中给出稀疏 mask，难以说明层级结构和证据来源。
2. **Method:** HyperTrace 使用 intrinsic Lorentz 异构编码、几何一致解释、时间证据状态机和统一 ExplanationPacket，并把解释呈现给审核员。
3. **Evaluation:** 在 TwiBot-22、MGTAB、DeepPersona/OASIS synthetic episodes 上进行跨域/LOSO 评估，并以四维解释指标和受试者研究评估适当依赖。
4. **Finding:** 用占位符写真实结论，例如“HyperTrace improves geometry fidelity by [X.X] points and reduces acceptance of incorrect AI recommendations by [X.X] points without sacrificing [metric]”。

摘要不要列出实现细节、旧 CIKM 数字或未完成的 p-value。

#### CCS 与关键词候选

- CCS 候选：Human-centered computing -> Human computer interaction (HCI); Computing methodologies -> Machine learning; Information systems -> Social networks; Security and privacy -> Social network security and privacy。
- Keywords：graph explainability; hyperbolic representation learning; heterogeneous graphs; coordinated deception; evidence traceability; appropriate reliance; content moderation。

最终 CCS 必须通过 ACM CCS Concept Explorer 生成，不要直接照抄候选文字。

### 2.2 1 Introduction

建议用 5 个自然段和一个贡献列表：

1. **现实问题。** 水军/协同欺骗已从静态 bot 识别转向多账号协调、角色分工、跨关系传播和被检测后的策略改变；审核员不仅需要分数，还需要知道何时、为何、依据哪条记录作出决定。
2. **表示问题。** 用户、帖子、转发、点赞、评论和时间事件构成异构图；层级协同结构不一定适合平直欧式空间。说明 Lorentz 表示的中心-边缘/指数容量直觉，但不要把几何直觉直接当成性能结论。
3. **解释问题。** GNNExplainer、PGExplainer 等主要优化预测保真和稀疏度，未显式约束测地线距离、径向排序、角色关系和时间因果边界。指出“预测看起来对”不等于“网络结构解释正确”。
4. **人因问题。** 内容审核属于高风险决策。解释可能帮助纠错，也可能增加错误自动化偏见，因此应测量 appropriate reliance，而不是只问 trust 或 satisfaction。
5. **方案和贡献。** 用一段概述四维评价、HyperTrace 系统和人类研究，再列三项贡献。

引言末尾可使用如下贡献列表（英文稿中改为平行句式）：

- We formulate four measurable dimensions of explanation quality for hierarchical heterogeneous deception graphs.
- We implement HyperTrace, a geometry-consistent and temporally traceable explanation pipeline with provenance-aware evidence packets.
- We evaluate both model/explanation behavior and human appropriate reliance in a moderator decision task.

### 2.3 2 Related Work

不要按“算法名清单”堆砌。每一小节结尾明确本文缺口。

#### 2.1 Coordinated deception and social bot detection

介绍 TwiBot-22、MGTAB 以及协同/时间攻击检测工作，区分 bot/human supervision、角色标签和 synthetic labels。强调我们的目标不是重新发布一个 bot 分类器，而是研究检测结果如何被解释和审计。

#### 2.2 Hyperbolic and heterogeneous graph learning

引用：HGCN (NeurIPS 2019)、HGT (WWW 2020)、Hypformer (KDD 2024)、HypHGT (2026 arXiv preprint)。说明本文采用 intrinsic Lorentz heterogeneous encoder，并在可行时与 HypHGT 或公开实现对比；若最终代码没有完整复现 HypHGT 的 global-linear hyperbolic Transformer，不要声称等价复现。

#### 2.3 GNN explanations and graph counterfactuals

引用：GNNExplainer (NeurIPS 2019)、PGExplainer (NeurIPS 2020)、TempME (NeurIPS 2023)、D4Explainer (NeurIPS 2023)、Adversarial Mask Explainer (WWW 2024)、GNNX-Bench/GraphXAI；可补充 AAAI 2026 counterfactual work。明确现有工作通常未同时评估 geometry fidelity、role fidelity 和 evidence provenance。

#### 2.4 Human-AI reliance and moderation

引用：Buçinca et al. (PACM HCI 2021)、Lai et al. (CHI 2022)、Vasconcelos et al. (PACM HCI 2023)、Schemmer et al. (CHI 2023)、Chen et al. (PACM HCI 2023)、Lee and Chew (PACM HCI 2023)、Morrison et al. (PACM HCI 2024)、de Jong et al. (PACM HCI 2025)、Kou and Gui (PACM HCI 2020)。结尾指出 HyperTrace 把“解释正确性”和“依赖是否校准”放进同一审核任务。

### 2.4 3 Research Questions and Hypotheses

建议同时给出研究问题和可证伪假设：

- **RQ1:** 在相同检测性能预算下，几何约束是否保持更好的层级/测地线结构？
- **RQ2:** provenance 和时间截断是否提高解释的可追溯性、稳定性和审核速度？
- **RQ3:** HyperTrace 是否让审核员在 AI 正确时接受、AI 错误时拒绝，而不是无条件提高接受率？

可注册的假设：

- **H1:** Geo-PGExplainer 的 geometry fidelity 高于普通 PGExplainer，且 prediction fidelity 不显著下降。
- **H2:** HyperTrace 的 evidence traceability 高于无 provenance 的解释条件。
- **H3:** HyperTrace 提高 error-adjusted moderation accuracy 和 wrong-AI rejection rate。
- **H4:** HyperTrace 降低不适当接受错误建议的比例；trust calibration 或 calibration error 改善。

如果数据不足以支持统计检验，应把 H1-H4 改写为探索性 RQ，不强行报告显著性。

### 2.5 4 System Overview and Design Goals

先给一张端到端系统图，建议包含：

`data adapters -> feature/provenance layer -> Lorentz heterogeneous encoder -> detector/auxiliary heads -> Geo-PGExplainer -> temporal evidence state -> ExplanationPacket -> moderator dashboard`。

定义四个设计目标：

1. **G1 Geometry consistency:** 解释不能只保留能提高 logit 的边，还要量化保留的测地线和中心-边缘秩序。
2. **G2 Evidence traceability:** 每项解释映射到 source table、record ID、timestamp/cutoff、model version 和 provenance hash。
3. **G3 Temporal validity:** 训练/解释只能使用 cutoff 之前的信息，未来事件只能作为 next-action 或事后评估目标。
4. **G4 Appropriate reliance:** 界面提供不确定性、反事实和审计入口，支持纠错而不是诱导服从。

这一节不要展示所有代码目录；只展示概念模块，工程文件清单放在 artifact README 或附录。

### 2.6 5 Technical Method

#### 5.1 Data sources and provenance

用表格列出：

| 来源 | 角色 | 监督可用性 | 允许的结论 |
|---|---|---|---|
| TwiBot-22 | 主要真实数据 | bot/human 与官方 split | 静态/跨域检测与解释 |
| MGTAB | 多源真实训练或外部验证 | bot/stance 与官方字段 | 多源泛化，需遵守 adapter contract |
| DeepPersona/OASIS | 合成 episode | role/campaign/phase/next_action 可用 | 时序机制、LOSO、消融 |

明确 1000 只是早期 smoke/抽取规模，不是论文固定核心用户数。正式 TwiBot 使用经审计的 core ID manifest；合成主实验按已批准 DatasetPlan（当前目标为 5 场景 x 4 simulation seeds x 5000 agents，若服务器资源或正式协议变更，以最终 manifest 为准）。

#### 5.2 Intrinsic Lorentz heterogeneous encoder

介绍节点/边类型、关系专属参数、Lorentz manifold 运算、映射到切空间进行分类的必要性，以及 real/synthetic domain adapters 和 task heads。给出核心符号，但避免把未经实验验证的模块写成贡献结论。

建议报告的模型组：

1. Euclidean HGT；
2. Euclidean HGT + Poincare projection；
3. intrinsic Lorentz-HGT real-only；
4. intrinsic Lorentz-HGT synthetic-only；
5. sequential synthetic-to-real；
6. proposed shared encoder + domain-specific heads；
7. HypHGT（若许可证和实现可用）。

#### 5.3 Geo-PGExplainer

说明它继承 PGExplainer 的可学习边 mask，但增加可单独消融的几何项：

- geodesic-distance preservation；
- radial/center-periphery ordering preservation；
- relation/role consistency；
- prediction fidelity and sparsity。

不要把“几何损失”写成装饰项。给出每一项的定义、归一化方式、训练/推理阶段、权重选择规则和消融。权重应在 validation 或预注册规则中确定，不能看 test 结果后调参。

#### 5.4 Temporal evidence state machine

定义至少五个状态：`benign_warmup -> coordination_onset -> propagation_escalation -> adaptive_evasion -> aftermath`。说明状态只使用 cutoff 以前观测，未来事件不进入当前解释；状态转移记录触发证据、时间戳、置信度和版本。

#### 5.5 ExplanationPacket and provenance contract

一个 packet 至少包含：预测值/不确定性、模型和数据版本、节点/边 mask、几何摘要、角色/关系证据、时间状态、counterfactual、evidence IDs、source table/record ID、cutoff、hash 和可复核链接。解释生成失败时必须显式返回 warning 或 unavailable，不能留下悬空 evidence ID。

#### 5.6 Moderator dashboard

展示顺序建议是：

1. AI 判定、概率和不确定性；
2. 可切换的双曲结构图（中心-边缘、关系类型、被选证据）；
3. 时间线和 cutoff；
4. 原始证据/来源跳转；
5. “如果去掉该证据/改变该行为，预测如何变化”的反事实；
6. 审核员最终决定和理由记录。

界面要支持“查看证据后拒绝 AI”，不能只有一个接受按钮。论文中用 storyboard 或实际原型图，报告操作路径和响应时间。

### 2.7 6 Offline Evaluation

#### 数据和划分

- TwiBot-22：使用官方 train/validation/test 或审计后的等价 manifest；核心用户和邻居边界写入附录。
- MGTAB：按 adapter 审计结果报告节点数、特征维度、关系类型和 split；不得把 test 标签用于调参。
- Synthetic：5 scenario families，simulation seeds 与 model seeds 分离；主评估采用 LOSO，held-out scenario 的所有 seed 只进入 test。
- Scale evaluation：500/1000/2000/5000 只在预先指定的代表性场景上做规模曲线，不把 scale test 混入训练。

#### 主要比较

检测：AUROC、AUPRC、Macro-F1、Balanced Accuracy、Brier、ECE。

解释：prediction fidelity、sparsity、geometry fidelity、radial-order preservation、role fidelity、evidence traceability、explanation stability、median latency。

时序：cutoff leakage rate、next-action accuracy/F1、future-event exclusion rate、状态转移准确度。

#### 消融

- 去掉 geometry loss；
- 去掉 role/relationship consistency；
- 去掉 temporal cutoff/state machine；
- 去掉 provenance registry；
- Euclidean mask 与 Lorentz-aware mask；
- 共享 encoder 与简单拼接训练；
- synthetic auxiliary heads 开/关；
- psychology-inspired proxies 开/关。

每个消融只回答一个机制问题，避免把多个改动合成“全家桶 vs 空模型”。

#### 结果表占位符

```text
TwiBot AUPRC: [XX.X]
MGTAB AUPRC: [XX.X]
LOSO macro-F1 improvement: [+X.X percentage points]
ECE: [X.XX]
Geometry fidelity: [XX.X]
Role fidelity: [XX.X]
Evidence traceability: [XX.X%]
Median explanation latency: [XX ms]
```

### 2.8 7 Human-Subjects Study

#### 研究对象与伦理

写明实际批准机构、审批编号、招募渠道、纳入/排除标准、补偿金额、知情同意、匿名化、存储期限和退出权。若使用故意错误的 AI 建议，必须说明欺骗的必要性、最小风险、停止规则和事后告知，并严格按照已批准方案执行。不能在尚未审批时写成“已获批准”。

#### 四种条件

1. **No AI:** 只看案件和证据，自主判断。
2. **AI score only:** 显示 bot 概率/推荐，不提供解释。
3. **Conventional explanation:** 显示传统特征重要性或欧式子图。
4. **HyperTrace:** 显示双曲层级、角色证据、时间追踪、provenance、反事实和不确定性。

条件之间保持案件、标签难度和模型输出分布一致；采用被试内/被试间设计时要预先说明，随机化条件顺序并平衡顺序效应。

#### 任务

审核员阅读一批合成或去标识化案件，回答：是否升级/拦截、置信度、依据、是否接受 AI 建议。案件应包含 AI 正确和 AI 故意错误两类，错误建议比例、难度和场景在方案中固定。

#### 指标

主指标：error-adjusted accuracy、appropriate reliance、correct-AI acceptance、wrong-AI rejection/correction rate。

次指标：trust calibration/ECE、决策时间、NASA-TLX 或等价工作负荷、感知有用性、解释满意度、证据点击/回溯次数。

“信任增加”不能作为唯一正向结果；如果信任上升但错误建议接受率也上升，应如实报告为风险。

#### 统计计划

在数据收集前固定样本量依据、排除规则、主要比较和多重比较校正。根据设计使用 mixed-effects logistic model（准确/拒绝）、mixed-effects linear model（时间/工作负荷）或非参数替代；报告效应量、95% CI 和个体/案件随机效应。样本量和 p-value 在实验前不得虚构。

### 2.9 8 Results

结果按“检测 -> 解释 -> 人因 -> 性能/审计”顺序写，每节先给事实，再给解释，不把讨论提前写成结论。

建议小节：

1. **Detection and cross-domain generalization:** 主表、LOSO、MGTAB 外部测试。
2. **Geometry and role fidelity:** 普通 PGExplainer vs Geo-PGExplainer；报告预测保真不下降的证据和几何指标变化。
3. **Temporal and provenance reliability:** cutoff leakage、traceability、失败率、状态转移和延迟。
4. **Moderator outcomes:** 四条件准确率、wrong-AI rejection、时间、工作负荷和校准。
5. **Ablations and sensitivity:** 几何损失权重、邻居规模、场景规模、模型 seed。

每张结果图注明数据 split、seed 聚合方式、误差条定义和样本量。旧 CIKM 论文中的数值仅可作为历史背景，不能当作本稿新实验结果。

### 2.10 9 Discussion

围绕三个问题解释结果：

- 为什么层级协同图中的几何约束会影响解释，而不只是影响分类分数？
- 哪些证据链最能帮助审核员纠错，哪些只增加认知负担？
- HyperTrace 在域迁移、攻击适应和缺失时间戳时何时失效？

避免从小规模仿真直接外推现实平台治理效果；把结果解释为“在所审计数据和任务设置下的证据”。讨论模型性能、解释质量和人因收益可能互相冲突的情况。

### 2.11 10 Limitations, Ethics, and Broader Impact

至少覆盖：

- TwiBot/MGTAB 的时间陈旧、平台偏差和标签定义；
- synthetic episodes 与真实攻击的机制差异；
- psychology-inspired proxies 的误读、偏见和不应被当成诊断；
- 双曲嵌入和解释的不确定性；
- 证据链可能暴露用户隐私；
- 解释系统被攻击者反向利用、模型探测或针对性规避的风险；
- 受试者欺骗研究的伦理边界和去标识策略。

伦理声明要说明我们不提供个人心理诊断、不自动做最终封禁决定，并保留人工复核和审计记录。

### 2.12 11 Conclusion

三段即可：

1. 重申问题：协同欺骗解释必须同时正确、符合几何、可追溯并可被人正确使用。
2. 重申证据：HyperTrace 的系统和离线/人因评估回答了哪些 RQ；用真实结果替换占位符。
3. 给出边界：未来工作包括更完整的 hyperbolic Transformer、实时部署和更多平台数据，但不要把未来工作写成已完成贡献。

## 3. 建议图表清单

| 编号 | 内容 | 放置章节 | 必须回答的问题 |
|---|---|---|---|
| Fig. 1 | HyperTrace 端到端架构 | Overview | 数据如何流到审核员？ |
| Fig. 2 | Lorentz 中心-边缘与解释 mask | Method | 为什么欧式 mask 可能丢失层级结构？ |
| Fig. 3 | 时间证据状态机 | Method | cutoff 如何防止未来信息泄漏？ |
| Fig. 4 | Moderator dashboard | System/Human Study | 审核员实际看到什么、能做什么？ |
| Fig. 5 | 几何/角色保真对比 | Results | Geo-PGExplainer 是否改善目标指标？ |
| Fig. 6 | 四条件人因结果 | Results | 是否减少错误依赖？ |
| Table 1 | 数据集与划分 | Evaluation | 数据来源、标签和规模是否清楚？ |
| Table 2 | 检测与校准指标 | Results | 是否跨域泛化？ |
| Table 3 | 四维解释指标 | Results | 解释是否不仅仅是 fidelity？ |
| Table 4 | 用户研究统计 | Results | 人类收益、风险和效应量是什么？ |

每张图准备英文 caption、`Description` 和生成脚本；不要用手工截图替代可复现实验图。

## 4. 证据到主张的约束表

| 论文主张 | 必须有的证据 | 不能替代的证据 |
|---|---|---|
| 几何一致解释优于普通 mask | 同 split/seed 的 geometry fidelity 与消融 | 仅展示一张好看的可视化 |
| 解释不损伤预测保真 | fidelity、稀疏度和置信区间 | 只报告一个案例 |
| 证据可追溯 | evidence ID 解析成功率、source/cutoff 覆盖率 | 只展示 JSON 样例 |
| 适当依赖改善 | 正确接受、错误拒绝、校准和统计检验 | 只问“是否信任 AI” |
| 联合训练有价值 | P2 与 real-only/synthetic-only/naive concat 消融 | 仅报告参数量或训练时间 |
| 双曲表示有价值 | Euclidean、projection、intrinsic 对照及几何指标 | 仅因为使用 Lorentz 就声称有效 |

## 5. 当前代码与论文的对应关系

以下是论文层面的模块名，不要求在正文暴露全部旧目录：

- 数据适配和审计：`data_processing/`、DatasetPlan、relative manifest、checksum/audit artifacts。
- 26 维用户特征和行为代理：`Character Classification/`、`emotional_analysis/`；正文写作时统一称 psychology-inspired proxies。
- Lorentz 编码器和角色头：`Character Classification/lorentz_hgt.py`、`new_role_assigner.py`；最终实验前必须以代码审计记录为准描述实现范围。
- 证据注册和 ExplanationPacket：`explainability/`。
- P2 训练入口：`scripts/train_p2.py`；报告实际 commit、配置、CUDA/PyTorch 版本和 seed。
- 受试者界面：若 dashboard 尚未完成，正文暂写为 prototype，并将完成度、可用性和截图状态明确标注为 `[待实现]`，不能把概念图当成已部署系统。

## 6. 结果占位符和写作纪律

统一使用以下格式，直到正式实验结束：

```text
AUPRC: [XX.X]
Macro-F1: [XX.X]
ECE: [X.XX]
LOSO improvement: [+X.X percentage points]
Geometry fidelity: [XX.X]
Role fidelity: [XX.X]
Evidence traceability: [XX.X%]
Median explanation latency: [XX ms]
Moderator accuracy improvement: [+X.X percentage points]
Wrong-AI acceptance reduction: [X.X percentage points]
Participant count: [N]
Statistical test/effect size: [pre-registered analysis]
```

每一个数字都必须能回溯到 `runs/`、`package/audits/`、统计脚本和最终 commit。若实验失败、指标不可用或数据许可限制报告，应写明原因而不是用估计值填空。

## 7. 写作与实验执行顺序

1. 先冻结题目、RQ/H1-H4、数据 manifest、模型 seed、评价指标和伦理方案。
2. 在服务器完成正式 bundle 审计、5 场景 LOSO、模型对照和 Geo-PGExplainer 消融。
3. 锁定结果表和图的生成脚本，再实现/冻结 dashboard 交互。
4. 按批准方案进行受试者研究，保存匿名原始记录、条件随机化信息和分析脚本。
5. 先写 Method、Evaluation、Results，再写 Introduction、Abstract 和 Discussion，避免先写结论。
6. 用 `sigchi` 模板编译匿名稿，执行 ACM TAPS/可访问性检查，检查图注、`Description`、表格和引用。
7. 最后逐条审查“主张-证据表”，删除所有超出数据支持范围的 novelty 或因果表述。

## 8. 参考文献起始清单

以下是正文应优先核对并加入 BibTeX 的文献类别；最终版本需补齐 DOI、页码、作者全名和准确 venue：

- Chami et al. Hyperbolic Graph Convolutional Neural Networks. NeurIPS 2019.
- Hu et al. Heterogeneous Graph Transformer. WWW 2020.
- Yang et al. Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space. KDD 2024.
- Park et al. Hyperbolic Heterogeneous Graph Transformer. arXiv:2601.08251, 2026 preprint.
- Ying et al. GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS 2019.
- Luo et al. Parameterized Explainer for Graph Neural Network. NeurIPS 2020.
- TempME. Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery. NeurIPS 2023.
- D4Explainer. In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion. NeurIPS 2023.
- Adversarial Mask Explainer for Graph Neural Networks. The Web Conference 2024.
- Buçinca et al. To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI. PACM HCI 2021.
- Lai et al. Human-AI Collaboration via Conditional Delegation: A Case Study of Content Moderation. CHI 2022.
- Vasconcelos et al. Explanations Can Reduce Overreliance on AI Systems During Decision-Making. PACM HCI 2023.
- Schemmer et al. Appropriate Reliance on AI Advice: Conceptualization and the Effect of Explanations. CHI 2023.
- Chen et al. Understanding the Role of Human Intuition on Reliance in Human-AI Decision-Making with Explanations. PACM HCI 2023.
- Lee and Chew. Understanding the Effect of Counterfactual Explanations on Trust and Reliance on AI. PACM HCI 2023.
- Morrison et al. The Impact of Imperfect XAI on Human-AI Decision-Making. PACM HCI 2024.
- de Jong et al. Cognitive Forcing for Better Decision-Making: Reducing Overreliance on AI Systems Through Partial Explanations. PACM HCI 2025.
- Kou and Gui. Mediating Community-AI Interaction Through Situated Explanation: The Case of AI-Led Moderation. PACM HCI 2020.
- GNNX-Bench and GraphXAI. 用于解释评估和基准语境，具体版本和引用需按最终采用的实现核对。

## 9. 投稿前硬性验收清单

- [ ] 论文没有声称首次提出 hyperbolic HGT，也没有把 2026 预印本当作不存在。
- [ ] 所有新数字来自正式运行记录；旧 CIKM 数字已明确标为历史或删除。
- [ ] 真实数据标签、合成 privileged labels 和心理代理特征被严格区分。
- [ ] 四维解释指标都有定义、实现、基线和消融。
- [ ] 所有时间解释通过 cutoff/leakage 审计。
- [ ] ExplanationPacket 中每个 evidence ID 都能解析或显式标记 unavailable。
- [ ] 人类研究有真实伦理批准、知情同意、补偿、欺骗说明和 debriefing；稿件表述与批准版本一致。
- [ ] 人因主要结果包含错误 AI 建议下的纠错/拒绝，而非只有 trust/satisfaction。
- [ ] 代码、数据包、manifest、checksum、配置和图表脚本可按许可复现。
- [ ] `sigchi` 模板、CCS、关键词、图 Description、表题、引用和匿名化全部通过检查。

