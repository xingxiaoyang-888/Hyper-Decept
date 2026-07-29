# 已作废：TwiBot-22 → DeepPersona → Agent 仿真

作废日期：2026-07-28

作废原因：该实验错误地把 DeepPersona 注入 TwiBot-22 真实账号并重新进行 Agent 仿真，混淆了“合成仿真路线”和“真实数据评测路线”。其中产生的 CSV、向量库、YAML、仿真 DB、日志和图表不得进入论文实验、统计结果或演示材料。

正确边界：

- DeepPersona 只注入 MultiAgent4Collusion 的合成仿真 Agent。
- TwiBot-22 直接进入预处理、检测和白盒解释，不经过 DeepPersona 或 Agent 仿真。

本目录仅作为错误实验的可追溯归档，不可恢复为有效实验结果。
