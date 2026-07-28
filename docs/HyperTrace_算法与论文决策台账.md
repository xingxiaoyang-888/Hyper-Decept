# HyperTrace 算法与论文决策台账

本文件记录“论文证据 → 代码决策 → 可声明范围 → 必做实验”的对应关系。
任何模型升级先更新本台账，避免把预印本已有贡献误写成本文创新。

## 2026-07-27：真正双曲异构编码器

### 论文依据

1. Park et al. **Hyperbolic Heterogeneous Graph Transformer (HypHGT)**,
   arXiv:2601.08251, submitted 2026-01-13.
   - 使用 Lorentz 模型；
   - 关系专属可学习曲率；
   - 全局线性双曲注意力与局部异构 GNN 融合；
   - 研究任务为通用异构图节点分类。
2. Yang et al. **Hypformer: Exploring Efficient Transformer Fully in
   Hyperbolic Space**, KDD 2024.
   - 提供 curvature-aware HT/HR 运算；
   - 证明全双曲线性注意力可以避免频繁切换切空间。

### 对创新声明的影响

- 不再声明“首次提出真正双曲 HGT”。
- HypHGT 应作为最新骨干/强基线，而不是被当作本文贡献。
- HyperTrace 的候选贡献保持为：动态协同欺骗中的几何解释、时间证据追溯、
  统一 ExplanationPacket、解释状态演化，以及对审核员适当依赖的影响。

### 当前代码决策

- `Character Classification/lorentz_hgt.py`
  - Lorentz manifold 原生状态；
  - 多头、关系专属距离注意力；
  - 每种边类型独立可学习曲率；
  - 连续 edge mask 和关系门控可供 Geo-PGExplainer 审计；
  - 输出转换为单位 Poincare 坐标，兼容现有解释协议。
- `Character Classification/new_role_assigner.py`
  - `intrinsic_lorentz` 成为默认后端；
  - `projection_head` 保留为欧式 HGT + Poincare 投影消融基线；
  - 每次运行写出 `geometry_metadata.json`，明确后端和曲率。
- `Character Classification/graph_builder.py`
  - 删除随机 Tweet 节点初始化；
  - 默认使用可观察、确定性的帖子特征；
  - 支持通过 `--post-embeddings` 接入新的帖子语义模型输出。
  - 为 posts/retweets/likes/comments 建立成对反向语义边，确保帖子证据能够
    进入用户表示，而不是停留在不可回传的叶节点。

### 当前实现不能声称的内容

- 当前不是 HypHGT 的完整复现：尚未实现其全局线性双曲 Transformer 分支。
- 当前不能声称优于 HypHGT；必须在服务器数据上完成成对实验。
- 原生 Lorentz 后端通过单元和小图集成测试，只代表数值与接口正确，
  不代表检测效果或解释质量已经得到验证。

### 下一道论文验收门

1. 增加 HypHGT 风格的全局线性关系注意力，并与当前局部 Lorentz 编码器融合。
2. 在同一数据切分、同一维数与参数预算下比较：
   - Euclidean HGT；
   - Euclidean HGT + Poincare projection；
   - intrinsic local Lorentz HGT；
   - full global-local Lorentz HGT；
   - 若官方代码可得，HypHGT 官方实现。
3. 报告 AUROC/AUPRC/F1/ECE、训练时间、显存、距离失真、径向次序保持率、
   角色保真和解释稳定性。
4. Tweet semantic embedding 至少比较一个现代多语模型与无语义 fallback；
   具体模型在服务器显存和语言覆盖确认后冻结，不能只因发布时间选择。

## 算法采用规范

- “新”不等于“适合”：采用模型必须有与任务结构对应的理由和可复现实验。
- 经典算法允许作为基线；不能把旧基线包装成主要技术贡献。
- 对 2025–2026 预印本明确标注状态，不将未经同行评议结果视为既定事实。
- 每项创新声明必须绑定代码模块、对照组、消融项和可计算指标。

## 2026-07-27：TwiBot 静态数据适配

### 数据事实

- `twibot_1000_multimodal_v5.csv` 包含 1000 个有监督核心用户；
- `twibot_1000_v5.db` 额外包含 2904 个一阶外部邻居；
- 派生 DB 未保留原始 tweet ID 和时间戳，因此只能声明静态能力；
- `user_char` 在该导出中是公开 profile description，适配时重命名为 `bio`。

### 代码决策

- `data_processing/dataset_adapter.py` 明确分离公共节点、标签、外部邻居、
  静态动作与能力声明；
- 缺失时间保持为空，不生成伪时间；
- 派生内容节点使用显式 `derived_action_rowid` 或
  `content_hash_without_original_post_id` 标记；
- `graph_builder.py` 保留核心—外部邻居拓扑，并将外部邻居设为独立节点类型；
- 监督标签只覆盖核心用户，`boundary` 不作为第三分类标签；
- Evidence Registry 仅在 TwiBot 数据契约明确指定时，将 `user_char` 映射为
  公开 `bio`；其他数据集仍默认阻止该字段。

### 声明边界

- TwiBot 用于静态检测、开放邻域拓扑解释和跨数据源泛化；
- 不用 TwiBot 派生 DB 证明实时状态演化、团伙角色真值或攻击阶段变化；
- 若重新取得原始 TwiBot tweet JSON，可恢复真实 tweet ID/created_at，但必须记录
  原始来源和转换过程，不能从文本或行号反推后冒充原始字段。
