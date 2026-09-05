# HyperTrace 初始化协议冻结说明

## 结论

UK2019 的 Lorentz 预训练 checkpoint 不能直接加载到 DeepPersona/OASIS 的完整异构模型。

原因是两者的图 schema 不同：

```text
UK2019:     user -> coordination -> user
Synthetic:  user/tweet + follows/posts/retweets/likes/comments 等多类关系
```

因此不能使用 `load_state_dict` 强行迁移整个 encoder。关系特定参数并不具有可比语义，直接复制会造成权重错配，也不能作为论文中的严谨 sim-to-real 证据。

## 冻结的主训练协议

下一次服务器窗口直接训练 `HyperTrace-Base`：

```text
输入：审计后的 DeepPersona/OASIS synthetic episodes
目标：coordination_label / is_bad 主监督
辅助真值：role、campaign、next_action（只在训练和离线解释评估使用）
模型：完整多关系 Intrinsic Lorentz-HGT
划分：5 场景 LOSO；被留出场景的 4 seeds 全部测试
```

Base 的主结果不依赖 TwiBot-22、MGTAB 或 Crypto-Campaign。

## UK2019 的正确位置

UK2019 保留为结构初始化候选和消融实验：

1. UK2019 先进行单关系 Lorentz-HGT 自监督图预训练；
2. 只有经过显式 `transfer_map` 验证的 schema-agnostic 参数才允许迁移，例如公共曲率或公共输入缩放；
3. relation-specific layers、tweet adapter、synthetic heads 必须重新初始化并由 synthetic 监督训练；
4. 记录迁移参数数量、未迁移参数数量和加载日志；
5. 若 warm-start 没有稳定改善验证集 AUPRC，则主论文不宣称 UK 初始化带来收益，将其作为负结果/消融记录。

因此，下次正式训练默认采用完整 synthetic 模型从头初始化；UK warm-start 作为独立可复现实验，不阻塞 Base 主训练。

## 数据集边界

```text
DeepPersona/OASIS      Base 主监督
UK2019                 真实拓扑预训练候选/结构消融
Crypto-Campaign        冻结 Base 后的 campaign-disjoint 金融适配
Fake Accounts          外部 CIB 评估；无图边时不进入 HGT 训练
TwiBot-22/MGTAB        当前研究路线排除
```

## 论文表述

论文应写成：

> HyperTrace-Base is trained on audited LLM-driven coordinated-deception episodes with explicit temporal and campaign supervision. We additionally evaluate whether a topology-only Lorentz warm-start from UK2019 transfers across heterogeneous schemas; relation-specific parameters are never copied without a validated mapping. Crypto-Campaign is treated as a separate campaign-disjoint domain-adaptation experiment.

这一区分保证真实数据来源不被篡改，也避免把金融 campaign 或普通协同拓扑误称为 LLM-CIB 标签。
