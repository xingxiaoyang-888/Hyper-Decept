# HyperTrace CHI 贡献论证与实验闭环

## 0. 一句话定位

HyperTrace 不是一个“再加几个解释模块的协同检测器”，而是一套让审核员能够在时间截止点重放、核验、纠正协同欺骗判断的可审计人机协作工作流。Lorentz-HGT 是其中用于产生结构化风险与证据的模型组件；论文的中心评价对象是审核员能否在不看到未来信息的情况下正确使用、质疑和追溯模型建议。

## 1. 三项主要贡献

面向 CHI 投稿，最终 contribution statement 应采用“概念/交互设计—技术实现—实证发现”的结构。下文的 C1 是论文首要贡献，C2 是实现该交互目标的技术支撑，C3 是必须由研究结果而非系统功能来成立的实证贡献。这样可以避免把界面、状态机和证据包重复包装成多个贡献。

### C1：截止时间安全的可审计解释交互

**问题。** 现有图解释器通常返回 top-k 节点/边或一组重要性分数，却不能保证解释只使用决策时刻之前的事件，也不能让审核员复核证据来源。

**方法。** HyperTrace 将解释定义为 cutoff-safe evidence reconstruction：每个解释包固定 cutoff、事件顺序、evidence ID、来源记录、内容 hash、模型版本和审核状态。Dashboard 以时间线、支持证据、反证、证据来源和可恢复审核状态组织信息；审核员在提交结论前必须核对 cutoff 与来源。

**可证伪预测。** 相比 score-only 与 static explanation，HyperTrace 应提高 cutoff 核验准确率、证据追溯成功率和错误建议拒绝率；若只增加界面复杂度而没有这些收益，C1 不成立。

**必须报告。** 时间泄漏率、悬空 evidence 率、来源 hash 校验率、解释生成延迟、关键证据删除后的预测变化，以及受试者的 balanced accuracy、appropriate reliance、错误建议拒绝率、决策时间和工作负担。

### C2：异构协同结构的内在 Lorentz 表示与安全初始化

**问题。** 协同网络包含层级、桥接和多关系结构；欧氏空间或只在最后投影到双曲空间的模型不能直接保证层级距离与消息传递的一致性。真实图通常缺少仿真器提供的机制标签，因此直接复制关系参数会造成 schema 错配。

**方法。** Intrinsic Lorentz-HGT 在 Lorentz 流形中完成关系消息传递、聚合和 prototype decision geometry。UK2019 只用于真实协同拓扑的结构预训练；DeepPersona/OASIS 提供受控 LLM-CIB episode 的协同监督。两种 schema 之间只迁移经过验证的全局 Lorentz 标量（曲率和输入尺度），不迁移 relation-specific、tweet adapter 或 privileged heads。

**可证伪预测。** 与 Euclidean HGT、Lorentz scratch 和不安全的全量 checkpoint 迁移相比，安全初始化应在几何保持、训练稳定性或跨场景泛化中的至少一个预注册指标上有可解释收益；若没有收益，应如实报告为负结果，而不把初始化协议宣传成性能提升。

**必须报告。** E0 Euclidean-HGT、E1 Lorentz-HGT real-only、E2 synthetic-pretrain→real-finetune、E3 时间打乱消融；同时报告 Lorentz 距离/径向顺序失真、AUROC/AUPRC/Macro-F1、Brier/ECE、LOSO 结果、梯度异常率、checkpoint 恢复成功率。当前服务器上的 E1-scratch/E2-UK-init 仅是协议 preflight，不是论文 E0–E3 的最终结果。

### C3：关于适当依赖与证据核验的实证发现

**问题。** “解释更漂亮”或“用户更信任模型”不能证明人机协作更安全；过度依赖会使错误高置信建议更危险。

**方法。** 通过四种最小条件产生可复现的人机协作证据：Manual、Score-only、Static explanation、HyperTrace。HyperTrace 额外提供时间边界、支持/反证、provenance 和确认/纠正流程；可选第五条件 HyperTrace-no-forcing 去除提交前的反证核验提示，用于估计认知强制的独立作用。研究对象是审核员的适当依赖，而不是总体信任或单纯点击率。

**可证伪预测。** HyperTrace 应在保持或提高联合判断准确率的同时，降低错误 AI 建议接受率，提高错误建议拒绝率和证据核验准确率；若只提高反应时间或主观信任而不改善适当依赖，C3 不成立。

**必须报告。** 被试内拉丁方设计、案例与条件不重复、正确/错误 AI 建议的平衡、教程与练习、预注册的主要指标及置信区间。若使用故意错误建议，伦理材料必须说明欺骗、风险控制和实验后告知。

在实验完成前，C3 只能写为研究目标或预注册假设，不能在摘要和贡献列表中声称为已经得到的发现。投稿时应把实际观察到的收益、无差异和代价写成经验性贡献；例如“何种证据核验机制帮助审核员拒绝错误建议，以及该收益在决策时间和工作负担上的代价”。

## 2. 数据与训练职责

| 数据 | 角色 | 可支持的论文结论 |
|---|---|---|
| DeepPersona/OASIS | 受控 LLM-driven synthetic CIB 主监督与解释校准 | 早期检测、机制真值、反事实和 cutoff/provenance 审计 |
| UK 2019 Coordinated Behavior | 真实用户协同拓扑的结构初始化 | 真实结构预训练可行性；不支持文本、LLM 来源或完整时间结论 |
| IO-26（授权后） | 真实信息行动主数据，按 campaign 切分 | 未见真实 campaign 的协同检测泛化；是主实验所需真实证据 |
| Crypto-Campaign | Base 冻结后的 campaign-disjoint 金融适配 | 跨领域适配；不能写成通用基础模型性能 |
| Fake Accounts、Fox8-23、BotSim-24 | 外部压力测试，按字段能力使用 | 外部稳健性；不得补造缺失标签或文本 |
| TwiBot-22、MGTAB | 当前 CHI 主线排除 | 不用于支持协同欺骗主张 |

在 IO-26 尚未授权前，服务器上的 Base 训练只能被称为 synthetic Base / initialization validation；不得把它写成真实数据泛化结果，也不得在摘要中填入最终百分比。

## 3. 最小实验闭环

1. **模型闭环：** E0–E3 使用相同 campaign/scenario split、特征列、阈值选择规则和 model seeds。
2. **解释闭环：** 对每个 test cutoff 生成 ExplanationPacket，执行时间泄漏、来源追溯、证据删除和几何失真审计。
3. **人因闭环：** 从冻结 test 案例抽取 Manual/Score-only/Static/HyperTrace 条件，测量 balanced accuracy、appropriate reliance、错误建议拒绝率、证据核验准确率、时间和工作负担。
4. **审计闭环：** 保存 data manifest、split manifest、checkpoint、metrics、解释包、受试者材料版本和 SHA-256；所有最终数字由脚本从审计产物汇总。

## 4. 不能写进论文的表述

- “首次检测 LLM 水军”——除非有公开且明确的 LLM-CIB 真实标签；当前只能说受控 LLM-driven episodes 和外部 LLM bot 压力测试。
- “UK2019 证明 LLM 协同欺骗”——UK 只提供真实拓扑结构。
- “心理特征解释真实人格”——心理字段不进入主检测输入，也不承担真实有效性结论。
- “实时”——只有在报告每个 cutoff 的增量延迟、无未来读取和恢复行为后，才能称为 real-time/auditable replay。
- “模块越多越先进”——每个组件必须对应消融和可证伪假设。

## 5. 当前状态与下一步

- 已完成：20 个 synthetic episodes 的 DatasetPlan 与文件审计；UK schema-validated 全局 Lorentz warm-start 前置；coordination-only Base、best-validation checkpoint 和精确恢复契约；相关本地测试。旧 v1/v2 LOSO 与使用 UK preflight checkpoint 的中间运行均标记为不可引用的协议产物。
- 待服务器重开：只重建 availability 标记特征并使用完整真实边的正式 UK 自监督 checkpoint；随后重新运行 5 场景 × 3 model seeds 的 v3 synthetic LOSO。
- 未完成：IO-26 授权与 campaign-disjoint 物化；E0–E3 真实主实验；解释包批量审计；Dashboard；受试者研究。
- 下一步：在不改变已冻结协议的前提下完成其余 synthetic Base folds；拿到 IO-26 后先做 split/provenance 审计，再运行 E0/E1/E2/E3 和 Crypto 独立 adapter；最后冻结人因案例并提交伦理版本。
