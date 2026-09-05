# HyperTrace CHI 贡献冻结与证据矩阵

## 贡献一：面向协同欺骗审核的时间边界解释工作流

HyperTrace 将解释对象定义为“截止时刻可观察证据的重构”，而不是静态 top-k 边或风险分数。每个 ExplanationPacket 同时记录 cutoff、事件顺序、evidence ID、来源 hash、模型版本和可恢复的审核状态。

必须用离线审计证明：解释不读取 cutoff 之后的事件；evidence ID 可回溯；删除解释中的关键证据会造成可量化的预测变化。

## 贡献二：面向异构协同结构的内在 Lorentz 表示与可迁移训练协议

HyperTrace 在 Lorentz 流形中完成异构关系消息传递、关系可靠性聚合和 prototype decision geometry。真实 UK2019 只用于拓扑结构预训练；DeepPersona/OASIS 提供受控 LLM-CIB 的机制级监督；二者之间只迁移经过 schema 验证的全局 Lorentz 参数，禁止复制不兼容的 relation-specific 参数。

必须用 E1/E2 对照证明：Lorentz 表示相对 Euclidean HGT 的几何保持、跨场景泛化和训练稳定性；warm-start 的收益或无收益都必须报告。

## 贡献三：将解释展示设计为审核员的证据核验与适当依赖机制

HyperTrace Dashboard 不只展示“模型为什么判高风险”，还要求审核员核对时间线、来源证据、支持证据和反事实证据，再接受、拒绝或升级建议。研究重点是 appropriate reliance，而不是让人盲目信任 AI。

实验至少比较 Manual、Score-only、Static explanation、HyperTrace；主要指标为审核 balanced accuracy、错误 AI 建议拒绝率、appropriate reliance、证据 cutoff 核验准确率、决策时间和工作负担。

## 明确边界

- DeepPersona/OASIS 的 role、campaign、next_action 是训练/离线评估真值，不是部署输入。
- Crypto-Campaign 是冻结 Base 后的 campaign-disjoint 金融适配，不是 Base 的 LLM 监督来源。
- Fake Accounts、Masquerade-23、Fox8-23 是外部评估数据，不能被写成主监督数据。
- 不把工程模块数量写成贡献；每个模块必须对应一个可检验假设和消融。
