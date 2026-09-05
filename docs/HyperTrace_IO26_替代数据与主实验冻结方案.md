# HyperTrace：IO-26 不可用时的协同数据替代方案

> 用途：IO-26 访问受限、申请周期不可控时的主实验备选方案。
>
> 原则：不伪造数据访问，不把 bot/human 标签包装成协同标签，不用受限数据的汇总数字冒充可复现实验。

## 1. 推荐决策

若 IO-26 在代码冻结和服务器训练窗口前仍未获得明确下载授权，主实验切换为：

```text
Open Crypto-Campaign dataset
        + audited DeepPersona/OASIS episodes
        + optional public IRA/CIB holdout when the raw records are legally accessible
```

论文任务相应定义为：

> 在决策截止时间之前，预测账号或局部网络是否会参与一个具有公开组织记录和参与证据的协同欺骗 campaign。

这比“是否为 bot”更接近论文目标，也比“检测一般信息行动”更符合可获得标签。

## 2. 首选真实数据：Crypto-Campaign

论文： [A Dataset of Coordinated Cryptocurrency-Related Social Media Campaigns](https://arxiv.org/abs/2301.06601)

数据记录： [Zenodo 7813450](https://zenodo.org/records/7813450)（原论文链接 7539178 当前重定向到该记录）。当前 API 显示：

- CC BY 4.0；
- 完整压缩包约 7.46 GB；
- `samples.zip` 约 9.9 MB；
- 15,870 个 bounty/campaign events；
- 约 185,000 名参与者；
- 约 10M forum comments；
- 跨事件的社交媒体行动、参与证明和时间信息。

### 2.1 为什么它比 IO-26 更适合作为应急主数据

- campaign 由论坛组织帖和规则明确记录；
- 参与者和任务动作有来源证据；
- 活动开始、截止和奖励信息提供时间边界；
- 数据公开可下载，许可和 DOI 可引用；
- 可以构造事件级、账号级和局部网络级 provenance。

它不是平台核验的国家级信息行动数据，因此论文不能写“state-sponsored IO detection”。正确主张是“coordinated deceptive campaign detection”。

### 2.2 防止标签泄漏：前瞻性任务

不能把参与后的 proof URL、奖励金额、已完成状态或明确 campaign ID 直接作为输入。主任务采用前瞻性定义：

```text
cutoff = 用户首次被观察到参与 campaign 之前
input  = cutoff 之前的公开论坛/社交行动、时间和关系
label  = 用户是否在未来窗口内完成该 campaign 的高置信参与动作
```

高置信正例：存在可核验的参与注册、任务证明或奖励记录，并且记录时间晚于 cutoff。

控制样本：同一主题、同一时间窗和同一论坛活跃，但在观测范围内没有任何该 campaign 参与证明的用户。论文中称为 `matched non-participant controls`，不称为绝对正常用户。

同时报告：

1. 去掉显式任务文本/URL 的结果；
2. 只使用 cutoff 前行为的结果；
3. 随机时间打乱消融；
4. 仅使用结构、仅使用文本、结构+文本的结果。

## 3. 仿真数据如何与真实数据统一

DeepPersona/OASIS 不作为无来源混合表直接加入，而是生成与真实任务相同的前瞻性样本：

```text
cutoff t 的可观察图和事件
→ 预测未来 Δ 窗口内是否进入/参与 campaign
```

仿真额外提供：

- 真正的 coordination membership；
- role/strategy transition；
- next_action；
- 反事实干预结果；
- 完整 provenance。

这些字段用于预训练、LOSO、解释校准和反事实有效性评估，但 `persona`、生成器内部状态、未来动作和 `role` 不作为检测器输入。

## 4. 训练与测试协议

### R0：真实数据基线

```text
Crypto-Campaign train campaigns
→ Lorentz-HGT supervised training
→ validation campaigns
→ unseen campaign test
```

### R1：欧式几何对照

使用完全相同的输入、切分、训练预算和 seeds，将 Lorentz-HGT 换成 Euclidean HGT。R0/R1 只回答双曲几何是否有价值。

### R2：真实自监督预训练

只在真实训练 campaign 上做不使用标签的时间/关系预训练（masked event、temporal edge 或 future relation prediction），然后监督微调。它用于回答真实数据自身的时序预训练是否有帮助。

### R3：仿真 sim-to-real

```text
synthetic pretraining
→ Crypto-Campaign supervised fine-tuning
→ unseen real campaign test
```

只有 R3 稳定优于 R0/R2，才将“受控仿真增强真实泛化”写成贡献。

### R4：时间有效性消融

保持模型和数据不变，去掉时间字段或打乱事件顺序。R4 用于证明收益不是由静态图或数据规模造成。

## 5. 划分

禁止按账号随机拆分。推荐：

```text
campaign/event families → train 70%
campaign/event families → validation 15%
campaign/event families → test 15%
```

同一个 campaign 的组织帖、参与者、证明、关系和未来事件必须在同一 partition。所有 split 列表、随机种子、过滤规则和 SHA-256 写入 manifest。

如果无法构造高置信 matched controls，真实数据只用于正例事件预测和 ranking，并在论文中明确负例不具备“已验证自然用户”含义。

## 6. 可选第二真实来源

### 6.1 IRA / Internet Research Agency archive

若能合法获取完整账号/帖子记录，可作为未见 campaign 外部测试。不能只依赖 tweet IDs，因为无法恢复文本和关系。只有在实际文件可下载、许可证和字段可审计时才纳入。

### 6.2 Danish Election 2022

有精细时间行为和 likes/retweets，但论文作者明确没有发现 cluster size 与 bot 分数或删除状态的显著关联，缺少强协同真值。因此只能作为无监督结构压力测试，不能作为主监督训练。

### 6.3 MisBot / Fox8 / BotSim

这些数据可用于 misinformation、LLM bot 或传播压力测试，但没有与本任务完全一致的真实 campaign membership 标签，不进入主监督结果。它们的生成器字段不能作为输入。

## 7. IO-26 的处理方式

IO-26 如果后来获批，可以作为额外真实 campaign 外部测试或第二主数据源；但主论文不能依赖“远程合作方只提供汇总指标”的不可复现实验。除非具备：

- 正式数据使用协议；
- 机构伦理/数据治理批准；
- 指定执行人员和访问日志；
- 固定容器和代码哈希；
- 可复核的聚合结果生成流程。

否则论文中不能写“我们使用了 IO-26”，只能写“未来可在受限数据上复现”。

## 8. 论文定位变化

采用替代方案后，标题和摘要中的任务名称统一使用：

```text
coordinated deceptive campaign detection
```

不再使用未经标签支持的：

```text
state-sponsored information-operation detection
LLM-water-army detection in the wild
```

主要贡献仍为：

1. 时间截止、异构关系和证据 provenance 约束下的协同检测与解释任务；
2. Lorentz-HGT 与受控动态仿真的 sim-to-real 训练/验证协议；
3. HyperTrace 实时、可审计解释工作流及审核员适当依赖实验。

## 9. 服务器执行优先级

IO-26 未授权时：

1. 下载并校验 Crypto-Campaign `samples.zip`；
2. 解析其 schema 和样例；
3. 确认完整包预算和可用磁盘；
4. 生成前瞻性 cutoff/label/control manifest；
5. 用样例跑 Euclidean/Lorentz Smoke；
6. 再上传或下载完整包并物化缓存；
7. 先跑 R0/R1 pilot，再决定是否启动 R3。

IO-26 获批时：将其加入独立 adapter 和外部测试 manifest，不改变 Crypto-Campaign 的可复现实验链路。
