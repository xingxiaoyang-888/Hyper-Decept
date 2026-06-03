"""
ablation - 情感模块消融实验与独立性验证工具包
====================================================
本模块独立于主检测管线，提供以下功能：
  1. 独立性验证   — 验证四个情感模块之间的统计独立性
  2. 消融实验     — 逐一移除情感模块，评估分类性能变化
  3. 可视化       — 相关性热力图、VIF 柱状图、消融结果对比图

四个情感模块：
  - EmpathyGap      : 认知共情错位差 (EmpathyGapAnalyzer)
  - DarkTriad       : 暗黑三角人格    (DarkTriadAnalyzer)
  - Contagion       : 无阻尼情感传染  (ContagionAnalyzer)
  - Volatility      : 功能性情感瞬切  (EmotionVolatilityAnalyzer)

使用方式：
  python -m ablation.run_ablation --skip-independence  # 仅消融
  python -m ablation.run_ablation --skip-ablation       # 仅独立性验证
  python -m ablation.run_ablation                       # 全流程
"""

from .independence_verify import verify_independence
from .ablation_experiment import run_ablation
from .ablation_plot import (
    plot_ablation_bar_chart,
    plot_correlation_heatmap,
    plot_vif_chart,
    plot_pca_scree,
)

__all__ = [
    "verify_independence",
    "run_ablation",
    "plot_ablation_bar_chart",
    "plot_correlation_heatmap",
    "plot_vif_chart",
    "plot_pca_scree",
]