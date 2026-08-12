# HyperTrace CHI 收尾实验与交付方案

## 1. 冻结范围

论文的研究对象是：审核员如何核验协同信息行动检测器给出的结构、时间、反事实和来源证据。

主训练、消融、解释界面和论文主张均永久排除：

- 心理特征、情绪分析及人格推断；
- 角色分类、角色 head 和事后角色命名；
- campaign head、next-action head 等 privileged auxiliary heads；
- TwiBot-22/MGTAB 的普通 bot detection 结果；
- 旧 `ablation/` 心理特征/XGBoost 消融管线。

这些旧文件可作为历史代码保留，但不进入当前实验入口、结果表或审稿材料。

## 2. 九阶段完成状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| S1 | 数据、标签语义、operation-disjoint 与来源审计 | 完成 |
| S2 | 20 个 DeepPersona/OASIS synthetic episodes 与图缓存 | 完成 |
| S3 | coordination-only Lorentz-HGT v5 正式训练 | 完成 |
| S4 | Honduras/UAE 双向 operation-disjoint 外测与 checkpoint 冻结 | 完成 |
| S5 | 15-checkpoint label-blind rank consensus | 完成 |
| S6 | 修复后的解释 evidence preflight 与证据 ID 审计 | 完成，审计通过 |
| S7 | Lorentz/Euclidean、real warm-start 与 no-temporal 机制消融 | 完成，60 folds 审计通过 |
| S8 | 英文审核界面、实验条件界面与人因实验 | 界面第一版完成；正式受试者实验未完成 |
| S9 | 论文表格、哈希清单、公开结果包与 main 分支发布 | 结果包完成，等待 Git 发布 |

因此，“主模型训练已完成”是准确表述；“论文实验已全部完成”目前不准确。

## 3. 已冻结的主模型证据

正式 v5 训练包含两个严格方向：

1. UAE 无标签、cutoff-safe 图 warm-start → synthetic coordination-only LOSO → Honduras 外测；
2. Honduras 无标签、cutoff-safe 图 warm-start → synthetic coordination-only LOSO → UAE 外测。

每个方向包含 5 个 held-out scenarios × 3 个 model seeds，共 15 个 checkpoint。外测阶段不使用目标 operation 标签选 checkpoint、调阈值或改变排序。

已冻结外测结果：

| Operation | AUROC | AUPRC |
|---|---:|---:|
| Honduras | 0.8850 ± 0.0842 | 0.4677 ± 0.3578 |
| UAE | 0.9005 ± 0.0312 | 0.5559 ± 0.2650 |

已冻结 label-blind rank-consensus 审核预算结果：

| Operation | Top 1% Precision | Top 1% Recall | Top 1% Lift |
|---|---:|---:|---:|
| Honduras | 10.50% | 12.65% | 12.65× |
| UAE | 16.79% | 33.65% | 33.64× |

当前 Honduras/UAE 共识只能称为 retrospective stability analysis；冻结协议在未来新数据集上才能称 prospective protocol。

## 4. 最后一个四小时 GPU 窗口

统一入口：

```bash
cd /xingxiaoyang/HyperTrace
GPU_LIST=0 bash scripts/run_chi_final_window.sh
```

`GPU_LIST` 是可选的 CUDA 设备列表。单张 A800 使用 `GPU_LIST=0`；四张卡可使用 `GPU_LIST="0 1 2 3"`。单卡模式会顺序处理两个解释 bundle 和所有消融 fold，并通过每个 fold 的 `last_checkpoint.pt` 恢复。

`WORKERS_PER_GPU` 控制同一张卡上的独立 fold 并发数，默认值为 `2`。`INTRAOP_THREADS` 和 `INTEROP_THREADS` 默认分别为 `4` 和 `1`，防止多进程训练时每个 PyTorch worker 各自创建全部 CPU 线程。它们都是吞吐参数，不改变数据划分、模型、损失、epoch 或 checkpoint 协议。并发上限不得超过当前变体尚未完成的 fold 数。

### 4.1 硬门槛：解释 preflight

GPU0 与 GPU1 并行生成 Honduras/UAE Top-3 label-blind evidence packets。必须满足：

- `labels_consumed=false`；
- 明确披露 eligibility membership 来源，但不读取 target label values；
- 冻结 consensus percentile 与重新推理 baseline percentile 一致；
- 15 个 checkpoint 的旋转不变 Lorentz 半径、距离和 margin 可计算；
- incident-edge deletion 数值有限；
- evidence IDs 能在 `events.csv/edges.csv` 中解析；
- bundle、checkpoint freeze 和 score artifact SHA-256 一致；
- 明确标记真实外测是 `full_release_snapshot`，不冒称历史在线 cutoff。

任何一项失败都停止论文结果生成；不允许带警告继续填表。

### 4.2 模型机制消融

固定相同 DatasetPlan、5 个 LOSO scenarios、3 seeds、50 epochs、class-balanced coordination-only loss 和 validation AUPRC 选模规则：

| 变体 | 唯一变化 | 回答的问题 |
|---|---|---|
| Lorentz-HGT + real warm-start | `observable18` + 原始关系图 + 无标签真实图初始化 | 主基线 |
| Lorentz-HGT scratch | 与 Euclidean 相同的随机初始化 | 与 Euclidean 进行无初始化混杂的几何比较 |
| Euclidean-HGT scratch | 仅把 backbone 改为 Euclidean HGT | 内在 Lorentz 几何是否有预测增益 |
| Lorentz-HGT no temporal | 主基线删除 `Temporal_Entropy`，清零边级 temporal attributes | 时序输入是否贡献检测性能 |

每个变体 15 个完整 folds，共 60 个 folds。最终审计要求四个变体的 scenario/seed 一一配对，并分别计算 scratch Lorentz 对 scratch Euclidean、主基线对 no-temporal、real warm-start 对 Lorentz scratch 的 bootstrap delta。

真实图 warm-start checkpoint 的参数包含 Lorentz manifold 结构，代码会拒绝把它加载到 Euclidean-HGT。因此几何比较只能使用两个 scratch 条件；主基线与 Euclidean 的直接差异不能单独归因于几何。

最终 60-fold 结果显示：scratch Euclidean 在 AUPRC、AUROC、Balanced Accuracy 和 ECE 上优于 scratch Lorentz；Macro-F1 的配对区间跨零。时间输入对 AUROC 有小幅正贡献，其他指标区间跨零。real warm-start 对 AUPRC 基本无影响，对 F1、Balanced Accuracy 和校准的改善是区间跨零的趋势。论文不得声称 Lorentz 几何全面提高预测性能；其主要价值需要由几何一致的可审计解释与后续人因实验论证。

报告 AUROC、AUPRC、Macro-F1、Balanced Accuracy、Brier、ECE 的 mean、SD、95% CI，以及按同一 scenario/seed 配对的 bootstrap delta。Brier/ECE 统一按“越低越好”转换 delta 方向。

心理特征与角色模块没有消融，因为它们不属于主模型。

### 4.3 解释机制评估

Top-3 只用于协议 preflight，不能作为论文总体解释效果。preflight 通过后冻结正式解释抽样：

- 每个 operation 的高风险队列和中等风险对照分别固定抽样；
- 抽样在揭示标签前完成；
- 所有条件使用同一 case pool 和同一 frozen v5 predictions；
- 人因案例平衡在抽样冻结、标签揭示之后完成，且不反向改变模型或解释器。

论文解释指标：

1. `Comprehensiveness`：删除解释关系后的排序/分数下降；
2. `Sufficiency`：仅保留解释关系时相对完整图的变化；
3. `Sparsity`：保留证据边数 / 可用关系边数；
4. `Geometry fidelity`：解释子图对 geodesic margin 与相对原型距离的保持；
5. `Temporal validity`：未来信息读取率、cutoff/window 违规率；
6. `Provenance coverage`：可解析 evidence ID 比例与 hash 校验率；
7. `Stability`：15-checkpoint rank std、Top-K Jaccard 与解释重叠；
8. `Latency`：packet 生成与审核界面加载时间。

当前 preflight 已实现 incident-edge deletion；正式解释批次必须补充 keep-only sufficiency 和 sparsity 后才可填入论文解释结果表。

## 5. 英文审核界面

实现位置：`review_ui/index.html`。

默认 HyperTrace 条件包括：

- 风险队列及 dataset-relative percentile；
- 15-checkpoint consensus rank 与稳定性；
- critical account neighborhood；
- Lorentz geodesic margin、Poincaré radius 和 checkpoint agreement；
- incident-edge deletion counterfactual；
- observed activity window 与 event counts；
- source evidence table 与 evidence IDs；
- artifact hashes、append-only review state；
- Reject recommendation / Request more evidence / Confirm escalation。

界面禁止展示：

- 心理、人设、情绪或角色名称；
- 未校准的“检测概率”；
- target ground-truth label；
- 把 full release snapshot 描述成 real-time historical cutoff；
- 跨独立 checkpoint 平均 Poincaré 坐标。

桌面 1440×900 和移动端 390×844 已完成布局验证，页面无横向溢出，网络图、标签页和审核状态交互正常。

## 6. CHI 人因实验闭环

服务器计算不能替代人因实验。论文若把人机协作列为主要贡献，必须完成伦理审批和真实受试者研究，不能预填结果。

建议四个条件：

1. Manual：平台记录，无模型建议；
2. Score-only：模型审核优先级，无解释；
3. Static explanation：静态重要关系/事件，无反事实和 provenance workflow；
4. HyperTrace：完整时间、几何、反事实、来源与审核状态流。

主要指标：

- balanced decision accuracy；
- appropriate reliance；
- erroneous-recommendation rejection rate；
- evidence verification accuracy。

次要指标：decision time、NASA-TLX/工作负担、信心校准。实验应平衡正确/错误 AI 建议，使用 Latin-square 或等价顺序平衡，并保证同一受试者不重复看到同一 campaign/case。

UI 完成后还需制作三种受控降级条件、任务脚本、练习案例、随机化清单和可恢复日志；正式数据必须由统计脚本而非人工复制生成。

## 7. 最终结果包与 Git 发布

服务器运行完成后生成：

```text
runtime/p2_formal_package/audits/chi_final_window/
├── explanation_preflight_audit.json
├── model_ablation/
│   ├── audit.json
│   └── table_model_ablation.tex
└── public/
    ├── paper_results.json
    ├── paper_tables.md
    ├── table_model_ablation.tex
    ├── checksums.sha256
    └── environment.json
```

进入 main 的内容：

- 训练、解释、审计和 UI 代码；
- 小型 JSON/CSV/Markdown/LaTeX 结果摘要；
- 数据/模型 manifest 和 SHA-256；
- 可公开的图表与论文表格。

不进入 Git：

- 原始 Honduras/UAE 数据；
- `.pt` checkpoint；
- 图缓存和大体积 CSV；
- SSH/API 密钥；
- 受试者可识别信息。

推送前必须核对当前远端 main，避免覆盖组员改动。只在全部审计 `status=passed` 后提交，提交信息中记录冻结 manifest hash 与实验协议版本。

## 8. 论文写作边界

可以写：

- HyperTrace 在受控 LLM-driven synthetic CIB 上学习 coordination patterns；
- 在未见真实 operations 上进行 operation-disjoint 外测；
- Lorentz 几何、真实图初始化和时间输入由无混杂的配对消融分别检验；
- 解释 packet 提供可复核的关系、时间、反事实和来源证据。

当前不能写：

- 已证明所有真实平台上的 LLM-CIB 泛化；
- 已证明实时早期检测，除非完成历史 cutoff replay；
- 已提高审核员表现，除非正式人因实验完成；
- 心理或角色解释具有真实性；
- Top-3 preflight 代表总体解释性能。
