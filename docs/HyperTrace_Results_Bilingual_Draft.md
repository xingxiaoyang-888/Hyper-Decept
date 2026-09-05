# 7 Results / 结果

This section reports results by research question rather than by implementation stage. Each subsection begins with the conclusion supported by the evidence and then states its empirical boundary. / 本节按照研究问题而非脚本或指标顺序报告结果。每一小节先给出证据支持的结论，再说明其适用边界。

## 7.1 Detector, Geometry, and Explanation Results / 检测器、几何与解释结果

**The detector learned the controlled coordination scenarios and transferred useful ranking signals to both unseen operations, but external precision was substantially less stable than synthetic performance.** Across the 15 LOSO scenario-seed folds, the complete Lorentz-HGT obtained an AUPRC of 0.9791 (SD=0.0082), AUROC of 0.9944 (SD=0.0030), Macro-F1 of 0.9296 (SD=0.0171), and balanced accuracy of 0.9631 (SD=0.0177). These values characterize discrimination within the controlled LLM-driven episodes and are not used as substitutes for external performance.

**检测器在受控场景中学习到了协同模式，并能向两个未见行动迁移具有价值的排序信号，但其真实外测精度明显比仿真结果更不稳定。** 在 15 个 LOSO 场景—种子折上，完整 Lorentz-HGT 的 AUPRC 为 0.9791（SD=0.0082）、AUROC 为 0.9944（SD=0.0030）、Macro-F1 为 0.9296（SD=0.0171），平衡准确率为 0.9631（SD=0.0177）。这些数值只反映受控 LLM 驱动事件中的区分能力，不能替代真实外测结果。

In the strict operation-disjoint evaluation, AUROC was 0.8850 (SD=0.0842) on Honduras and 0.9005 (SD=0.0312) on UAE, whereas AUPRC was 0.4677 (SD=0.3578) and 0.5559 (SD=0.2650), respectively. The especially wide Honduras AUPRC variation shows that transfer depended strongly on the held-out scenario and seed. At a label-blind 1% review budget, rank consensus achieved 12.65x lift on Honduras and 33.64x lift on UAE, but precision was only 10.50% and 16.79%. HyperTrace therefore supports review prioritization rather than automatic adjudication. The external releases are full-release snapshots, so these results do not establish historical online early detection.

在严格的行动互斥外测中，Honduras 与 UAE 的 AUROC 分别为 0.8850（SD=0.0842）和 0.9005（SD=0.0312），AUPRC 分别为 0.4677（SD=0.3578）和 0.5559（SD=0.2650）。Honduras 较宽的 AUPRC 波动表明迁移表现明显依赖留出场景与随机种子。在不读取标签的 Top 1% 审核预算下，排序共识在 Honduras 和 UAE 上分别取得 12.65 倍和 33.64 倍 lift，但 precision 仅为 10.50% 和 16.79%。因此，HyperTrace 支持的是审核优先级排序，而非自动裁决。外部数据是完整发布快照，不能据此声称历史在线早期检测。

**Lorentz geometry did not yield a general predictive advantage over a matched Euclidean detector, but it provided the representation used for the auditability analysis.** Under scratch initialization, Euclidean-HGT exceeded Lorentz-HGT on AUPRC (0.9836 vs. 0.9792), AUROC (0.9970 vs. 0.9938), balanced accuracy (0.9741 vs. 0.9538), and ECE, for which lower is better (0.0129 vs. 0.0201). Paired Lorentz-minus-Euclidean differences were -0.0044 for AUPRC (95% CI [-0.0068, -0.0021]), -0.0031 for AUROC (95% CI [-0.0045, -0.0018]), and -0.0203 for balanced accuracy (95% CI [-0.0382, -0.0076]). The Macro-F1 interval crossed zero.

**Lorentz 几何相对匹配的 Euclidean 检测器并未产生普遍预测优势，但它提供了本研究审计分析所使用的原型相对几何表示。** 在相同随机初始化下，Euclidean-HGT 在 AUPRC（0.9836 对 0.9792）、AUROC（0.9970 对 0.9938）、平衡准确率（0.9741 对 0.9538）以及越低越好的 ECE（0.0129 对 0.0201）上高于 Lorentz-HGT。配对的 Lorentz 减 Euclidean 差值分别为：AUPRC -0.0044（95% CI [-0.0068, -0.0021]）、AUROC -0.0031（95% CI [-0.0045, -0.0018]）和平衡准确率 -0.0203（95% CI [-0.0382, -0.0076]）；Macro-F1 的区间跨越 0。

We do not interpret this result as a deliberately chosen accuracy sacrifice: the detector was not optimized with an explicit prediction-versus-auditability utility function. Instead, the matched ablation bounds the predictive claim, while the explanation analysis tests whether the Lorentz representation can be preserved in a reviewer-facing evidence packet. Nor do we claim that Euclidean embeddings necessarily incur severe distortion in every sparse graph. Any geometry benefit is an empirical property of this detector, reconstruction protocol, and case distribution.

我们不将这一结果解释为有意选择的准确率牺牲：检测器并未使用显式的“预测性能—可审计性”效用函数进行优化。相反，匹配消融用于限定预测性主张，而解释分析用于检验 Lorentz 表示能否保留在面向审核员的证据包中。我们也不声称 Euclidean 嵌入在所有稀疏图中必然产生严重失真；任何几何优势都必须由本检测器、重构协议和案例分布下的实证结果支持。

Temporal input produced a small positive AUROC difference of 0.0017 (95% CI [0.0002, 0.0034]), while intervals for the other reported metrics crossed zero. Real-graph warm-start changed AUPRC by -0.00006 (95% CI [-0.00159, 0.00154]); its apparent F1 and balanced-accuracy improvements also had intervals crossing zero. These results motivate evaluating Lorentz geometry for the narrower role claimed by HyperTrace: exposing prototype-relative geometry that can be preserved and audited in the evidence packet, not universally improving classification.

时间输入仅带来了较小的 AUROC 正向差异 0.0017（95% CI [0.0002, 0.0034]），其他指标区间均跨越 0。真实图 warm-start 对 AUPRC 的差值为 -0.00006（95% CI [-0.00159, 0.00154]），其 F1 与平衡准确率的表面改善同样未排除 0。这些结果支持我们检验 Lorentz 几何在 HyperTrace 中更窄的作用：提供可在证据包中保持和审计的原型相对几何，而非普遍提高分类性能。

**HyperTrace produced source-complete and temporally valid packets for all 24 audited cases; among the eight cases with nontrivial candidate neighborhoods, it retained compact evidence while closely preserving the detector assessment and Lorentz margin.** Across these eight cases (four per operation), the median retained proportion was 1.56%, the median sufficiency error was 0.00131 review-percentile units, and the median geometry fidelity was 0.9843. Median checkpoint agreement was 0.8748. Across the full 24-case package, provenance and timestamp coverage were both 100%, and evidence selection did not read target labels.

**HyperTrace 为全部 24 个审计案例生成了来源完整且时间有效的证据包；在 8 个具有非平凡候选邻域的案例中，它仅保留紧凑证据，同时较好保持检测器判断与 Lorentz 边际。** 在这 8 个案例（每个行动 4 个）中，证据保留比例中位数为 1.56%，充分性误差中位数为 0.00131 个审核百分位单位，几何保真度中位数为 0.9843，checkpoint agreement 中位数为 0.8748。在完整 24 案例包上，来源覆盖率和时间戳覆盖率均为 100%，且证据选择未读取目标标签。

The remaining 16 cases had trivial candidate sets and often required retaining all available units; combining them with the eight nontrivial cases would make average sparsity misleading. We therefore report audit coverage over all 24 cases but compression statistics over the eight nontrivial cases. Formal comprehensiveness and comparisons with static top-k, random-edge, and degree-matched controls remain `[COMP/CONTROL RESULTS]` and must not be inferred from the six-case preflight.

其余 16 个案例的候选集合较为平凡，通常需要保留全部可用单元；将其与 8 个非平凡案例直接平均会误导稀疏度解释。因此，审计覆盖率在全部 24 个案例上报告，而压缩指标只在 8 个非平凡案例上报告。正式 comprehensiveness 以及与 static top-k、random-edge、degree-matched 对照的结果仍应填入 `[COMP/CONTROL RESULTS]`，不能从 6 个案例的 preflight 推断。

**[Table 3 about here: Consolidated detector and explanation results. Panel A reports synthetic LOSO performance, Honduras/UAE operation-disjoint performance, rank-consensus review-budget results, and matched detector ablations. Panel B reports the 24-case audit coverage and the eight nontrivial-case explanation results. Mark formal comprehensiveness, random-edge, degree-matched, failure-rate, and latency results as pending until their frozen evaluation artifacts are available.]**

**Table 3. Consolidated detector transfer, geometry ablation, and explanation-audit results.** Values are mean (SD) across the stated folds or checkpoints unless otherwise noted. External tests use full-release snapshots and operation-disjoint ranking. Explanation-compression statistics are restricted to the eight nontrivial cases; audit-contract coverage is reported over all 24 cases.

**Panel A. Detector validity and ablations**

| Evaluation / variant | N | AUROC | AUPRC | Macro-F1 | Balanced accuracy | ECE | 1% review-budget result |
|---|---:|---:|---:|---:|---:|---:|---|
| Synthetic LOSO, Lorentz-HGT + real warm-start | 15 folds | 0.9944 (0.0030) | 0.9791 (0.0082) | 0.9296 (0.0171) | 0.9631 (0.0177) | 0.0182 (0.0057) | -- |
| Honduras, operation-disjoint external | 15 checkpoints | 0.8850 (0.0842) | 0.4677 (0.3578) | -- | -- | -- | Precision=10.50%; lift=12.65x |
| UAE, operation-disjoint external | 15 checkpoints | 0.9005 (0.0312) | 0.5559 (0.2650) | -- | -- | -- | Precision=16.79%; lift=33.64x |
| Lorentz-HGT, scratch | 15 folds | 0.9938 (0.0034) | 0.9792 (0.0080) | 0.9205 (0.0348) | 0.9538 (0.0350) | 0.0201 (0.0081) | -- |
| Euclidean-HGT, scratch | 15 folds | **0.9970 (0.0013)** | **0.9836 (0.0054)** | 0.9238 (0.0130) | **0.9741 (0.0062)** | **0.0129 (0.0049)** | -- |
| Lorentz-HGT, no temporal input | 15 folds | 0.9927 (0.0044) | 0.9776 (0.0083) | **0.9301 (0.0159)** | 0.9643 (0.0123) | 0.0194 (0.0045) | -- |

**Panel B. Explanation fidelity and auditability**

| Metric | All 24 audited cases | 8 nontrivial cases | Interpretation |
|---|---:|---:|---|
| Valid packet / case count | 24 / 24 | 8 / 8 | Packets passed the frozen preflight contract |
| Temporal validity | 100% | 100% | No displayed event exceeded the declared scope |
| Provenance coverage | 100% | 100% | Displayed evidence IDs resolved to source records |
| Labels read during selection | 0 cases | 0 cases | Label-blind evidence selection |
| Evidence retained | Not summarized across trivial sets | **1.56% median** | Selected units / candidate units |
| Predictive sufficiency error | Not summarized across trivial sets | **0.00131 median** | Difference in review-priority percentile |
| Lorentz geometry fidelity | Not summarized across trivial sets | **0.9843 median** | Full-graph versus evidence-subgraph geometry |
| Checkpoint agreement | Not summarized across trivial sets | 0.8748 median | Cross-checkpoint evidence/rank agreement |
| Comprehensiveness | Pending frozen control run | Pending frozen control run | Do not infer from preflight |
| Random-edge / degree-matched controls | Pending frozen control run | Pending frozen control run | Do not infer from preflight |

**Table note.** The bold values in Panel A identify the higher value for the matched scratch predictive comparison, except that ECE is interpreted as lower-is-better. This table does not imply that Euclidean performance is inferior or that Lorentz geometry causally improves auditability; it reports the observed predictive and explanation-level evidence separately. / **表注：** Panel A 粗体表示匹配 scratch 预测比较中的较高值，但 ECE 按越低越好解释。本表不表示 Euclidean 性能较差，也不表示 Lorentz 几何因果性地提高了可审计性，而是分别报告观察到的预测证据与解释层证据。

**Figure 4. Compactness and geometric fidelity of HyperTrace explanations.** The figure shows eight high-priority cases with non-trivial candidate neighborhoods, comprising four Honduras and four UAE cases. (a) Evidence retention versus sufficiency error; points closer to the lower-left retain less evidence while more closely reproducing the full-graph assessment. The vertical and horizontal gray dashed lines indicate the sample medians of evidence retention (1.56%) and sufficiency error (0.00131 review-percentile units), respectively; they are descriptive reference lines, not pre-specified acceptance thresholds. (b) Case-level Lorentz geometry fidelity. The gray dashed line marks the sample median (0.9843), whereas the red dashed line marks the pre-specified minimum fidelity criterion of 0.95. Temporal-validity and provenance-traceability audits cover all 24 cases; Panels (a) and (b) report the eight cases requiring non-trivial graph reduction.

**图 4：HyperTrace 解释的紧凑性与几何保真度。** 图中展示了 8 个具有非平凡候选邻域的高优先级案例，其中 Honduras 和 UAE 各 4 例。（a）证据保留率与充分性误差；越靠近左下角，表示使用的证据越少且越接近完整图评估。竖直和水平灰色虚线分别表示这 8 个案例的证据保留率中位数（1.56%）和充分性误差中位数（0.00131 个审核百分位单位）；二者仅为描述性参考线，并非预先设定的通过阈值。（b）案例级 Lorentz 几何保真度。灰色虚线表示样本中位数（0.9843），红色虚线表示预先设定的 0.95 最低保真度标准。时间有效性与来源可追溯性审计覆盖全部 24 个案例；Panel（a）和（b）报告其中需要进行非平凡图缩减的 8 个案例。

## 7.4 Reviewer Decisions and Appropriate Reliance / 审核判断与适当依赖

> **Synthetic-data prefill.** Every value in this subsection was generated by the three-stage simulation and is included only to validate the reporting format. Replace all values and inferential results with estimates from the genuine participant records before submission.
>
> **模拟数据预填说明。** 本节所有数值均来自三阶段模拟，仅用于核验报告格式；投稿前必须用真实参与者记录重新估计并整体替换。

**In the synthetic dataset, HyperTrace improved final decisions primarily by helping reviewers resist erroneous model recommendations rather than by indiscriminately increasing agreement with AI.** Twenty simulated reviewers completed 160 reviews, with eight cases assigned to each reviewer. Risk-only, standard-signals, and HyperTrace evidence were assigned to 6, 7, and 7 reviewers, respectively. Initial unaided accuracy was 62.5%, 71.4%, and 60.7%; after assistance was revealed, final accuracy was 62.5%, 73.2%, and 83.9%. The bootstrap-estimated difference in final accuracy was 21.4 percentage points between HyperTrace and risk-only (95% CI [6.5, 37.2]) and 10.7 points between HyperTrace and standard signals (95% CI [-5.4, 26.8]). In the participant-clustered, operation-adjusted logistic model used to validate the simulation analysis, the omnibus Condition × Model Correctness interaction was χ²(2)=5.11, p=.078. The interaction odds ratio was 0.17 for HyperTrace versus risk-only (95% CI [0.02, 1.32]) and 1.47 for HyperTrace versus standard signals (95% CI [0.25, 8.53]).

**在模拟数据中，HyperTrace 对最终判断的改善主要来自帮助审核员抵御错误模型建议，而非无差别地提高人机一致性。** 20 名模拟审核员共完成 160 次审核，每人审核 8 个案例。Risk-only、standard-signals 和 HyperTrace evidence 条件分别包含 6、7 和 7 名审核员。三种条件下的初始无辅助准确率分别为 62.5%、71.4% 和 60.7%；查看辅助信息后的最终准确率分别为 62.5%、73.2% 和 83.9%。Bootstrap 估计显示，HyperTrace 相对 risk-only 的最终准确率高 21.4 个百分点（95% CI [6.5, 37.2]），相对 standard signals 高 10.7 个百分点（95% CI [-5.4, 26.8]）。在用于核验模拟分析流程的参与者聚类稳健、行动来源调整逻辑回归中，Condition × Model Correctness 的总体交互检验为 χ²(2)=5.11，p=.078；HyperTrace 相对 risk-only 的交互优势比为 0.17（95% CI [0.02, 1.32]），相对 standard signals 为 1.47（95% CI [0.25, 8.53]）。

When the model recommendation was correct, correct-AI acceptance was 90.0%, 85.7%, and 94.3% across the three conditions; the corresponding RAIR values were 75.0%, 83.3%, and 93.8%. When the recommendation was wrong, reviewers rejected it in 16.7%, 52.4%, and 66.7% of trials, while RSR was 25.0%, 58.8%, and 80.0%. Relative to risk-only, HyperTrace increased wrong-AI rejection by 50.0 percentage points (95% CI [19.8, 79.4], Holm-adjusted p=.064); relative to standard signals, the increase was 14.3 points (95% CI [-19.0, 42.9], Holm-adjusted p=.384). Reporting correct-advice acceptance and erroneous-advice rejection separately indicates that the simulated improvement was not generated solely by a greater tendency to follow the model. Figure 5 summarizes final accuracy, correct-advice acceptance, and erroneous-advice rejection by condition.

当模型建议正确时，三种条件下的正确建议采纳率分别为 90.0%、85.7% 和 94.3%，相应的 RAIR 分别为 75.0%、83.3% 和 93.8%。当模型建议错误时，错误建议拒绝率分别为 16.7%、52.4% 和 66.7%，RSR 分别为 25.0%、58.8% 和 80.0%。相较于 risk-only，HyperTrace 将错误建议拒绝率提高了 50.0 个百分点（95% CI [19.8, 79.4]，Holm 校正后 p=.064）；相较于 standard signals，提高了 14.3 个百分点（95% CI [-19.0, 42.9]，Holm 校正后 p=.384）。分别报告正确建议采纳率和错误建议拒绝率表明，模拟结果中的改善并非仅由更强的模型服从倾向产生。Figure 5 汇总了各条件下的最终准确率、正确建议采纳率和错误建议拒绝率。

This improvement involved a measurable review cost. Median review time was 48.5 s (IQR 42.9–56.0), 77.3 s (IQR 65.0–86.6), and 111.4 s (IQR 93.0–133.6) for risk-only, standard-signals, and HyperTrace evidence, respectively. The log-time mixed-effects model estimated that HyperTrace required 2.25 times the review time of risk-only (95% CI [2.06, 2.45], p<.001) and 1.48 times that of standard signals (95% CI [1.36, 1.61], p<.001). Mean workload ratings were 3.00 (SD 0.63), 3.43 (SD 0.98), and 4.71 (SD 0.49) on the seven-point scale, and mean confidence changes from initial to final judgment were +1.46, +3.71, and +9.89 points. Relative to risk-only, HyperTrace increased median review time by 62.9 s and mean workload by 1.71 points; these simulated costs should be evaluated against the genuine decision effects after data replacement.

这一改善伴随着可测量的审核成本。Risk-only、standard-signals 和 HyperTrace evidence 条件下的中位审核时间分别为 48.5 秒（IQR 42.9–56.0）、77.3 秒（IQR 65.0–86.6）和 111.4 秒（IQR 93.0–133.6）。对数时间混合效应模型显示，HyperTrace 的审核耗时是 risk-only 的 2.25 倍（95% CI [2.06, 2.45]，p<.001），是 standard signals 的 1.48 倍（95% CI [1.36, 1.61]，p<.001）。三种条件在七点量表上的平均工作负荷分别为 3.00（SD 0.63）、3.43（SD 0.98）和 4.71（SD 0.49），从初始判断到最终判断的平均信心变化分别为 +1.46、+3.71 和 +9.89 分。相较于 risk-only，HyperTrace 的中位审核时间增加了 62.9 秒，平均工作负荷增加了 1.71 分；替换真实数据后，应将这些成本与真实决策效果一并解释。

Within the HyperTrace condition, two secondary participant-clustered logistic models examined evidence quality separately. A 0.01 increase in geometry fidelity was associated with an adjusted odds ratio of 0.93 for final correctness (95% CI [0.70, 1.25], p=.635), whereas a 0.001-unit increase in sufficiency error was associated with an adjusted odds ratio of 1.27 (95% CI [0.65, 2.49], p=.484). Neither simulated association was distinguishable from zero. These case-level associations do not establish causal effects of either explanation metric.

在 HyperTrace 条件内，两项参与者聚类稳健的次要逻辑回归分别检验了证据质量指标。几何保真度每提高 0.01，最终判断正确的调整优势比为 0.93（95% CI [0.70, 1.25]，p=.635）；充分性误差每增加 0.001 个单位，调整优势比为 1.27（95% CI [0.65, 2.49]，p=.484）。两项模拟关联均无法与零效应区分。这些案例层面的关联不构成几何保真度或充分性误差产生因果效应的证据。

**Table 4. Reviewer decisions, reliance, and review cost by assistance condition.**

| Condition | Participants | Trials | Initial accuracy | Final accuracy | Correct-AI acceptance | Wrong-AI rejection | RAIR | RSR | Median time (IQR) | Workload, mean (SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Risk-only | 6 | 48 | 62.5% | 62.5% | 90.0% | 16.7% | 75.0% (12) | 25.0% (12) | 48.5 s (42.9–56.0) | 3.00 (0.63) |
| Standard-signals | 7 | 56 | 71.4% | 73.2% | 85.7% | 52.4% | 83.3% (12) | 58.8% (17) | 77.3 s (65.0–86.6) | 3.43 (0.98) |
| HyperTrace evidence | 7 | 56 | 60.7% | 83.9% | 94.3% | 66.7% | 93.8% (16) | 80.0% (15) | 111.4 s (93.0–133.6) | 4.71 (0.49) |

Correct-AI acceptance and wrong-AI rejection use trials with correct and incorrect model recommendations as their respective denominators. RAIR is calculated only when the initial judgment was wrong and the model recommendation was correct; RSR is calculated only when the initial judgment was correct and the model recommendation was wrong. Parentheses after RAIR and RSR report eligible trial counts. Workload is reported on a seven-point scale. All entries are simulated placeholders and support no empirical human-subject claim.

正确建议采纳率和错误建议拒绝率分别以模型建议正确和错误的试次为分母。RAIR 仅在初始判断错误且模型建议正确时计算；RSR 仅在初始判断正确且模型建议错误时计算。RAIR 与 RSR 后的括号表示符合计算条件的试次数。工作负荷采用七点量表报告。表中所有数值均为模拟占位值，不支持任何真实人因结论。

## 7.5 Qualitative Accounts of Use and Failure / 使用与失败的质性描述

> **Synthetic-data prefill.** All rationales, comments, codes, counts, and excerpts in this subsection were generated by a deterministic simulation. They validate the qualitative reporting structure but provide no human-subject evidence.

> **模拟数据预填说明。** 本节的理由、评论、编码、计数和摘录均由确定性模拟生成，仅用于核验质性报告结构，不构成人因实验证据。

We coded 160 non-empty trial-level rationales and 20 post-task comments. The accounts reflected the assistance available in each condition: risk-only accounts primarily used review priority, standard-signals accounts referred to relational or temporal summaries, and HyperTrace accounts combined sequence reconstruction, provenance checks, and comparison with the initial judgment. At least one of these practices appeared for all 20 simulated reviewers and in 129 of 160 rationales. One generated account summarized the intended workflow: “I compared the order and timing of the observed interactions; I checked the linked source records and timestamps; I compared the recommendation with my initial judgment” (SIM-07).

我们对 160 条非空试次级理由和 20 条任务后评论进行了编码。各条件下的陈述反映了界面提供的信息：risk-only 主要依据审核优先级，standard-signals 提及关系或时间摘要，而 HyperTrace 则结合序列重构、来源核验以及与初始判断的比较。20 名模拟审核员均至少表现出其中一种做法，这些做法出现在 160 条理由中的 129 条。一条生成陈述概括了预期流程：“我比较了观测互动的顺序和时间，核对了链接的来源记录与时间戳，并将模型建议与初始判断进行了比较”（SIM-07，译文）。

Temporal scope and provenance provided the most direct grounds for qualifying a recommendation. Sequence reconstruction appeared in 82 rationales from all 20 reviewers, while provenance verification appeared in 45 rationales from seven HyperTrace reviewers. Nine reviewers generated at least one account in which an incomplete or conflicting record was treated as unresolved evidence. In a trial where the initial and final judgments differed, one generated rationale stated: “I checked the linked source records and timestamps” (SIM-11). These fields represent reported reasoning in the synthetic text, not observed source-record clicks.

时间范围和来源记录是对模型建议作出保留时最直接的依据。序列重构出现在 20 名审核员的 82 条理由中，来源核验出现在 7 名 HyperTrace 审核员的 45 条理由中；9 名审核员至少生成过一条将不完整或冲突记录视为未解决证据的陈述。在一次初始判断与最终判断不同的试次中，一条生成理由写道：“我核对了链接的来源记录和时间戳”（SIM-11，译文）。这些字段表示合成文本中报告的推理过程，而不是实际观察到的来源记录点击。

Geometry and counterfactual information played a supporting rather than adjudicative role. Geometry audit was coded in 31 rationales from seven HyperTrace reviewers, and counterfactual use in 29 rationales from the same seven reviewers. Across the 49 rationales containing either code, 17 accompanied a changed judgment and 32 accompanied retention of the initial judgment. A generated comment described geometry as a preservation check: “The geometry summary helped me check that the short evidence packet still represented the model's structure” (SIM-05). Thus, the simulated accounts treat geometry fidelity as evidence about explanation preservation, not as proof that a recommendation is correct.

几何与反事实信息主要发挥辅助作用，而非直接承担裁决功能。几何审计出现在 7 名 HyperTrace 审核员的 31 条理由中，反事实使用出现在同一组 7 名审核员的 29 条理由中。在包含任一代码的 49 条理由中，17 条伴随判断改变，32 条伴随保留初始判断。一条生成评论将几何指标描述为保持性检查：“几何摘要帮助我检查精简后的证据包是否仍然代表模型使用的结构”（SIM-05，译文）。因此，模拟陈述将几何保真度视为解释保持程度的证据，而不是模型建议正确的证明。

The simulation also exercised three failure modes. Evidence overload appeared in 18 rationales from 10 reviewers, metric confusion in three rationales from two HyperTrace reviewers, and missed scope warnings in 11 rationales from nine reviewers. One generated negative comment stated: “The timeline and source links were useful, but comparing every item at once was demanding” (SIM-15). Together, these failures show how the final analysis should retain adverse cases alongside successful use: compact evidence can still impose workload, geometry can be mistaken for correctness, and a visible scope warning can still be ignored. These simulated patterns motivate the design implications in Section 8 but must be replaced with themes and excerpts derived from genuine records.

模拟还覆盖了三类失败方式：证据过载出现在 10 名审核员的 18 条理由中，指标误解出现在 2 名 HyperTrace 审核员的 3 条理由中，忽略范围警告出现在 9 名审核员的 11 条理由中。一条生成的负面评论写道：“时间线和来源链接很有用，但一次比较所有项目确实需要较大认知负荷”（SIM-15，译文）。这些失败说明最终分析应同时保留成功使用和负面案例：紧凑证据仍可能增加工作量，几何指标可能被误解为正确性，显眼的范围警告也可能被忽略。这些模拟模式用于衔接第 8 节的设计启示，但必须由真实记录提炼出的主题和摘录整体替换。

No additional main-text figure is required for this subsection. The complete synthetic code frequencies and generated accounts are retained only as analysis-pipeline artifacts.

本节无需新增正文图片。完整的模拟代码频次和生成陈述仅作为分析流程校验材料保存。
