# HyperDecept 联合双曲训练与数据构建方案 v3

> 状态：**方案待审阅，尚未据此修改训练代码**  
> 更新日期：2026-08-04  
> 适用范围：P2 正式数据构建、Lorentz-HGT 联合训练、LOSO 实验与训练审计。

## 1. 论文训练主张

HyperDecept 的训练贡献不是把多个数据集直接拼接，也不是堆叠大量损失项，而是：

> 在共享 Lorentz 决策几何中，将真实数据的 bot/human 监督与受控多 Agent episode
> 的角色、campaign 和下一动作特权监督结合，进行可审计的 episodic sim-to-real
> 联合训练。

正式名称建议使用：

**Privileged Episodic Sim-to-Real Training in a Shared Lorentz Decision Geometry**

三个原则：

1. 真实数据只提供其实际存在的 bot/human 标签，不补造角色、人格或 campaign 真值；
2. 仿真标签明确标为 generated privileged supervision，不包装成真实世界心理诊断；
3. 检测器、解释器和审核工作台分阶段训练与评估，不将解释器损失混入检测主损失。

## 2. 当前数据基线

### 2.1 真实数据

| 数据集 | P2 用途 | 监督与限制 | 当前状态 |
|---|---|---|---|
| TwiBot-22 | 主要真实训练、验证和内部测试 | 使用官方 node split；外部邻居只作上下文 | 待上传完整原始数据并正式物化 |
| MGTAB | 第二真实来源及多关系结构验证 | 使用预注册、分层、固定哈希的 node split（不是官方 split）；无原始时间戳、campaign 或战术角色真值 | 10,199 用户标准张量包已完成 adapter/Smoke 验证 |

P2 主训练使用：

```text
TwiBot-22 official train
+ MGTAB adapter-generated train
+ DeepPersona/OASIS synthetic train episodes
```

验证和测试严格使用冻结的 train/validation/test 划分：TwiBot-22 使用官方 split，
MGTAB 使用 adapter 生成并写入 `split_seed42.csv` 的预注册分层 split。两者都不使用
test prevalence、test threshold 或 test 标签选择损失权重。

P1 仅作为控制实验：训练时移除 MGTAB，再在 MGTAB test 上评估。P1 不是另一条产品
路线，也不能和 P2 的 MGTAB 结果混称为“完全未见域”。由于当前模型有 dataset-specific
input adapter，P1 必须先实现“仅用无标签目标域统计量初始化 adapter”的明确协议；在此
之前不得把 MGTAB test-only 数字写成跨域结果。当前 `scripts/train_p2.py` 只实现 P2。

### 2.1.1 如何避免审稿人误解训练集与检测集

用 TwiBot-22 和 MGTAB 训练后在它们的 test mask 上检测不是泄漏，而是标准的
`seen-source held-out test`：训练只读 train mask，模型选择和校准只读 validation，
最终冻结 checkpoint 后一次性读取 test。论文必须准确命名：TwiBot-22 test 和 MGTAB
test 都是同源留出测试，不能称为外部域泛化。真正跨数据集结果只在数据集完全不参与
有标签训练、且目标域 adapter 的无标签适配协议已预注册时报告。

审核员会质疑的是：全量标签训练后再测同一批用户；用 test 选择阈值或损失权重；把边界
邻居标签混入监督；或把 synthetic role/persona 标签包装成真实心理真值。

### 2.2 仿真主语料

当前正式主语料冻结为：

```text
5 scenario families × 4 simulation seeds × 2,000 agents
= 20 independently executed episodes
每个 episode = 30 timesteps
```

场景：

1. `leader_amplifier`
2. `bridge_infiltration`
3. `synchronized_boosting`
4. `persona_drift`
5. `adaptive_evasion`

仿真 seeds：`11, 22, 33, 44`。模型初始化 seeds：`7, 17, 27`。模型 seed
不要求重新生成仿真 DB。

20 个 episode 复用同一批经过审计的 2,000 个独立 DeepPersona。这个设计控制了
参与者构成，使 LOSO 更集中地测量场景变化；但它不能证明未见人格泛化。因此论文
必须区分：

- LOSO：未见攻击场景泛化；
- simulation seed：同一场景的随机运行变化；
- model seed：优化随机性；
- persona-disjoint：当前主语料不支持，若要声称必须另生成独立人口。

### 2.4 暂不纳入主训练的数据扩展

旧 v2 中的 `N={500,1000,5000}`、38 个 synthetic DB、Fox8-23 和 BotSim-24
不再是启动正式 P2 训练的前置条件。它们只能在数据、许可、预算和主实验稳定后作为：

- 规模外推实验；
- 外部时间泛化实验；
- 额外稳健性分析。

论文主结果不得依赖尚未完成的数据扩展。

额外数据集的固定作用如下：

| 数据 | 是否进入 P2 主训练 | 可承担的结论 |
|---|---:|---|
| Fox8-23（若取得并独立审计） | 否 | 独立真实时间/来源 OOD；单列报告 |
| BotSim-24（若取得并确认生成器独立） | 否 | 独立仿真生成器 OOD；只测检测与鲁棒性 |
| `N={500,1000,5000}` 规模包 | 否 | size/吞吐/显存曲线，不增加独立样本数 |
| persona-disjoint population | 否（当前没有） | 未见人格泛化；需另生人口后才能声称 |

它们只进入 `external_test` 或 `scale_test` partition，不参与阈值、超参、标准化参数或
解释器选择。当前 runner 对完全未见 dataset adapter 会拒绝运行，这是防止伪造跨域结果
的安全门槛。

## 3. 正式 Episode Artifact 契约

原始 OASIS SQLite DB 不是可直接训练的完整 episode。每个正式 episode 在物化后必须
至少包含：

```text
seed_XX.db
seed_XX.csv
seed_XX.features.csv
seed_XX.labels.csv
seed_XX.event_targets.csv
seed_XX.manifest.json
seed_XX.activation_audit.json
```

要求：

- 26 维输入不得是占位零值；
- `next_action` 只来自对应时间截断前后的合法预测窗口；
- role/campaign/phase 必须来自生成配置或事件日志；
- manifest 保存 scenario、simulation seed、Agent 数、timesteps、模型、prompt、人口哈希、
  DB/CSV/features/labels 的 SHA-256 和 provenance；
- cutoff artifact 与完整 DB 分开记录，禁止用未来事件解释过去预测；
- DB 完成不等于 artifact 完成，只有 manifest 审计通过后才能进入 DatasetPlan。

## 4. 数据划分与统计单位

### 4.1 真实数据

- TwiBot-22 使用官方 train/validation/test node masks；MGTAB 使用
  `split_seed42.csv` 的固定、分层、可复现 node masks；
- 同一真实用户不得跨 split；
- 邻居节点可以作为无标签消息传递上下文，但不能进入监督损失；
- 训练、模型选择和校准只读取 train/validation；test 只在模型冻结后运行；
- 分别报告 dataset macro 与 sample-weighted 指标，避免大数据源掩盖小数据源。

### 4.2 Synthetic LOSO

每个 held-out scenario 构成一个 fold：

```text
held-out scenario 的 seeds 11/22/33/44  → test
其余四个场景的 seed 44                 → validation
其余四个场景的 seeds 11/22/33          → train
```

最终正式运行：

```text
5 LOSO folds × 3 model seeds = 15 training runs
```

synthetic 的统计单位是 episode/campaign，而不是把同一 episode 中 2,000 个 Agent
错误地视为 2,000 次完全独立实验。置信区间应在 fold、episode 或 model seed 层面计算。

## 5. 模型架构：当前已经实现的部分

```text
TwiBot-22 ────── dataset-specific feature adapter ─┐
MGTAB ───────── dataset-specific feature adapter ──┼─ Shared Intrinsic Lorentz-HGT
Synthetic ───── dataset-specific feature adapter ──┘
                                                      ↓
                                  weighted Lorentz centroid aggregation
                                                      ↓
                                    shared Lorentz bot prototypes
                                      ├─ bot/human decision
                                      └─ synthetic privileged heads
                                           ├─ Lorentz role prototypes
                                           ├─ campaign embedding
                                           └─ next-action classifier
```

当前实现要点：

1. 每个数据集使用独立输入 adapter，而不是只粗略区分 real/synthetic 两个输入域；
2. 编码器共享，隐藏状态保持在 Lorentz 流形；
3. 关系内、关系间及残差聚合使用加权 Lorentz centroid；论文中称
   `Fréchet-style aggregation`，不声称是迭代求解的精确 Fréchet mean；
4. bot/human 使用一个共享 Lorentz prototype classifier，不再使用 real/synthetic
   两套线性 bot head；
5. role 使用 Lorentz prototype classifier，但只在有生成真值的 synthetic 域监督；
6. campaign 和 next-action 当前在 origin tangent space 上输出；
7. 边可使用 multiplicity、temporal synchronization、recency 和 availability 等可靠性字段。

## 6. 损失函数

### 6.1 当前代码实际实现

当前代码不是旧 v2 的七项损失。实际为：

```text
Lcurrent = Ldet-real + Ldet-synthetic
         + 0.3 × schedule(t) × mean(Lrole, Lcampaign, Lnext-action)
         + 0.05 × Lhyperbolic-alignment
```

其中 relation reconstruction loss 已从当前主损失删除；class-conditional CORAL 只作为
alignment 方法的替代实现。当前仍存在的问题是：`1.0/0.3/0.05` 和三个 privileged
share 均为人工固定，且不同任务损失的数值量级未归一化。

### 6.2 v3 拟采用的简化目标

正式方案只保留三个概念组：

\[
\mathcal{L} = \mathcal{L}_{det}
  + \alpha(t)\,\overline{\mathcal{L}}_{priv}
  + \beta\,\overline{\mathcal{L}}_{geo}.
\]

定义：

1. `Ldet`：真实与仿真 bot 检测的域平衡平均；
2. `Lpriv`：role、campaign、next-action 的可用任务平均；
3. `Lgeo`：只把相同 bot/human 类别的真实—仿真 Lorentz 表示作为正对的跨域监督
   对比约束。

为避免不同量级导致人工权重失真：

- 每个 privileged 子任务先除以其 stop-gradient EMA loss scale，再对当前可用任务求平均；
- alignment 同样记录其 EMA scale，用于训练稳定性和审计；
- 检测目标固定为主任务系数 1，不参与自动降权；
- `alpha(t)` 使用短 warm-up 后逐步衰减，防止模型长期依赖 synthetic 特权标签；
- `beta` 只从一个预注册的小网格中用 validation 选择，并报告敏感性，不扩张成大量超参；
- 删除 `privileged_role_share`、`privileged_campaign_share` 和
  `privileged_action_share` 三个人工 share。

这不是把 EMA normalization 声称为论文核心算法。论文贡献仍是 privileged episodic
sim-to-real Lorentz training；归一化只是保证该训练目标可复现、可比较的实现机制。

### 6.3 不进入检测主损失的目标

以下项目分开训练或只作消融：

- Geo-PGExplainer 的 prediction/geometry/role fidelity；
- explanation sparsity；
- counterfactual validity；
- dashboard 人类依赖指标；
- 无可靠真实标签时的角色径向排序。

## 7. 训练调度

### 7.1 数据源平衡

正式训练不能直接把全部节点拼接后随机采样。每个优化 step：

1. 从 TwiBot-22/MGTAB 中按数据源均衡选择一个 real mini-batch；
2. 从当前 LOSO train 场景中按 scenario、seed 分层选择一个 synthetic mini-batch；
3. 使用 user seed nodes 的异构 NeighborLoader；
4. 上下文节点只参与消息传递，监督 mask 只覆盖 seed users；
5. `Ldet` 对 real/synthetic 两部分求域平衡平均，避免节点数量大的数据源支配梯度。

### 7.2 优化

正式实现目标：

```text
optimizer: AdamW
learning-rate schedule: linear warm-up + cosine decay
precision: AMP（数值敏感的 Lorentz 运算保持 FP32）
gradient clipping: 保留并记录
model selection: real validation AUPRC 优先
early stopping: 基于预注册 patience
checkpoint: model + optimizer + scheduler + scaler + RNG + DatasetPlan hash
```

训练集 positive class weight 必须分别从对应 fold 的 TwiBot/MGTAB/synthetic train mask
计算并写入 run config；禁止由 validation/test prevalence 推导。

## 8. 校准与最终评估

模型选择完成后，只使用真实 validation：

1. 拟合 temperature scaling；
2. 选择满足预注册目标的 operating threshold；
3. 冻结 temperature 和 threshold；
4. 运行 TwiBot-22 test、MGTAB test 和 held-out synthetic scenario；
5. 若存在 Fox8-23/BotSim-24，再以完全冻结 checkpoint 运行并单列为 external/OOD。

报告：

- Detection：AUROC、AUPRC、Macro-F1、Balanced Accuracy；
- Calibration：Brier、ECE、temperature、selected threshold；
- Synthetic mechanism：role accuracy、campaign retrieval/recovery、next-action accuracy；
- Generalization：LOSO macro、dataset macro、model-seed mean/CI；
- Evaluation provenance：每个数字标注 `seen-source holdout`、`held-out scenario`、
  `external dataset` 或 `scale test`，不把内部留出写成跨域泛化；
- Explanation：prediction fidelity、geometry fidelity、role fidelity、evidence traceability；
- Human study：审核准确率、纠错率、appropriate reliance、时间和主观负担。

## 9. 必做基线与最小消融

### 9.1 检测基线

```text
Euclidean HGT
Euclidean HGT + post-hoc Poincaré projection
Intrinsic Lorentz-HGT real-only
naive real + synthetic concatenation
synthetic pretrain → real finetune
proposed privileged episodic joint training
```

若官方实现、依赖和许可证可复现，再加入强双曲基线；不能仅凭论文描述重写一个弱版本
后称为公平比较。

### 9.2 关键消融

```text
- MGTAB train source
- synthetic privileged package
- hyperbolic class-conditional alignment
- temporal/reliability edge attributes
Lorentz prototype head → tangent linear head
Lorentz centroid → tangent/Euclidean aggregation
fixed raw loss weights → normalized v3 objective
```

不再把已经从主方法删除的 relation loss 列为 proposed component。

## 10. 当前完成状态与开训门槛

### 已完成

- MGTAB adapter、标准张量接入和 P2 Smoke；
- `scripts/train_p2.py` 正式 P2 runner；
- DatasetPlan、LOSO assignment、artifact/split audit；
- Intrinsic Lorentz-HGT、共享 bot prototype、Lorentz role prototype；
- 加权 Lorentz centroid 与边可靠性入口；
- 2,000 DeepPersona 人口和 20-episode 仿真计划；
- 部分正式仿真 DB 已生成，剩余仍在续跑计划中。

### 数据侧待完成

1. 完成全部 20 个 synthetic DB；
2. 将每个 DB 物化为 features/labels/event targets/manifest/audit；
3. 上传并审计完整 TwiBot-22；
4. 生成可重定位 DatasetPlan、checksums 和最终 `--require-files` 审计。

### 代码侧待审阅后修改

1. 用 v3 三组目标替换当前固定 raw loss 权重；
2. 将 NeighborLoader 真正接入 `scripts/train_p2.py`；
3. 实现数据源/场景分层 batch scheduler；
4. 从 fold train masks 自动计算 positive class weights；
5. 加入 warm-up、cosine decay、early stopping、AMP 安全边界和完整 resume；
6. 实现 validation-only temperature scaling 和 threshold selection；
7. 输出 dataset/episode/fold/model-seed 分层指标与审计记录。

### 正式 GPU 训练门槛

只有以下条件全部满足才启动 15 个正式训练 run：

- DatasetPlan `--require-files` 通过；
- TwiBot、MGTAB 与 20 个 synthetic episode 的 hash/schema/split 审计通过；
- 无 placeholder 26D 特征；
- next-action/cutoff 无未来泄漏；
- v3 runner 在 `--epochs 1 --max-steps 1` 上通过；
- 单 fold 小规模试训能保存、恢复并复现 checkpoint；
- 本文档经项目负责人确认。

## 11. 论文贡献边界

训练部分建议表述为：

> We introduce a privileged episodic sim-to-real training protocol that learns
> a shared Lorentz decision geometry from heterogeneous real-world supervision
> and controlled role-, campaign-, and action-level simulation signals.

不要表述为：

- “发明了一个由很多项组成的新损失”；
- “在真实用户上获得 Dark Triad 或 Opinion Leader 真值”；
- “20 个 episode 等于 40,000 个独立实验样本”；
- “MGTAB 在 P2 中是完全未见数据”；
- “weighted Lorentz centroid 是精确求解的 Fréchet mean”。
