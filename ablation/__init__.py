"""
ablation — Emotion Module Ablation Study & Independence Verification Toolkit
=============================================================================

This module operates independently of the main detection pipeline and provides:

  1. Independence Verification — Validates statistical independence among the four emotion modules
  2. Ablation Experiment — Removes each emotion module in turn, measuring classification performance impact
  3. Visualization — Correlation heatmaps, VIF bar charts, ablation result comparison plots

Four emotion modules:
  - EmpathyGap      : Cognitive empathy gap          (EmpathyGapAnalyzer)
  - DarkTriad       : Dark Triad personality traits   (DarkTriadAnalyzer)
  - Contagion       : Emotional contagion             (ContagionAnalyzer)
  - Volatility      : Emotion volatility              (EmotionVolatilityAnalyzer)

Usage:
  python -m ablation.run_ablation --skip-independence   # ablation only
  python -m ablation.run_ablation --skip-ablation        # independence verification only
  python -m ablation.run_ablation                        # full pipeline
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