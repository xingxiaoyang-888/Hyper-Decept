"""
ablation_plot.py — Visualization module for ablation experiments.

Provides four chart types:
  1. Ablation bar chart        — Accuracy / AUC / F1 comparison
  2. Correlation heatmap       — Pairwise Pearson correlation matrix
  3. VIF bar chart             — Variance Inflation Factor
  4. PCA scree plot            — Explained variance per PC

All plots exported at 300 dpi PNG.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import Dict, List, Optional
import os

# =============================================================
# Global style configuration
# =============================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Attempt to load CJK fonts (Windows -> Mac -> Linux fallback)
_CN_FONT_LOADED = False
for _cn_font in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    try:
        matplotlib.font_manager.findfont(_cn_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _cn_font
        _CN_FONT_LOADED = True
        break
    except Exception:
        continue

if not _CN_FONT_LOADED:
    # Fallback to sans-serif if no CJK font found
    plt.rcParams["font.family"] = "sans-serif"
    print("[Plot] No CJK font detected, using English labels.")


# =============================================================
# Color constants
# =============================================================
COLOR_FULL_MODEL = "#2c3e50"       # dark blue-grey
COLOR_ABLATION_BASE = "#e74c3c"    # ablation baseline red
COLORS_ABLATION = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#95a5a6"]


def _use_cn_labels():
    """Check whether CJK font rendering is available."""
    return _CN_FONT_LOADED


# =============================================================

# Chart 1: Ablation bar chart (core output)
# =============================================================
def plot_ablation_bar_chart(
    ablation_results: List[Dict],
    metrics: List[str] = None,
    save_path: str = None,
    show_values: bool = True,
) -> plt.Figure:
    """
    Plot ablation experiment comparison bar chart.
    
    Parameters:
        ablation_results : list of dicts returned by run_ablation()
        metrics          : metrics to plot, default ["accuracy", "auc", "f1"]
        save_path        : output image path
        show_values      : whether to annotate bar tops with values
    
    Returns:
        fig : matplotlib Figure object
    """
    if metrics is None:
        metrics = ["accuracy", "auc", "f1"]
    
    if not ablation_results:
        print("[Plot] ablation_results is empty, skipping plot.")
        return None

    cn = _use_cn_labels()
    n_configs = len(ablation_results)
    n_metrics = len(metrics)
    config_names = [r["config"] for r in ablation_results]
    
    # Metric display name mapping
    metric_display = {
        "accuracy":  "Accuracy",
        "auc":       "AUC",
        "f1":        "F1 Score",
        "precision": "Precision",
        "recall":    "Recall",
    }
    
    # ---- Create subplots ----
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6), 
                              squeeze=False)
    axes = axes[0]  # first row
    
    bar_width = 0.65
    x_positions = np.arange(n_configs)
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        
        values = [r.get(metric, 0.0) for r in ablation_results]
        stds = [r.get(f"{metric}_std", 0.0) for r in ablation_results]
        colors = [r.get("color", "#2c3e50") for r in ablation_results]
        
        # Plot bars
        bars = ax.bar(
            x_positions, values, bar_width,
            yerr=stds, capsize=5,
            color=colors, edgecolor="white", linewidth=1.2,
            error_kw={"linewidth": 1.5, "capthick": 1.5}
        )
        
        # Annotate bar tops with values
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(stds) * 0.1 + 0.003,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color="#2c3e50"
                )
        
        # Styling
        ax.set_xlabel("")
        ax.set_ylabel(metric_display.get(metric, metric))
        ax.set_title(metric_display.get(metric, metric))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(config_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Draw baseline at first bar (Full Model)
        if values:
            baseline = values[0]
            ax.axhline(y=baseline, color="#2c3e50", linestyle="-", 
                       linewidth=1.5, alpha=0.6, label="Full Model Baseline")
            ax.legend(fontsize=8)
    
    fig.suptitle(
        "Emotion Module Ablation Study",
        fontsize=17, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", 
                    facecolor="white", edgecolor="none")
        print(f"[Plot] Ablation bar chart saved: {save_path}")
    
    return fig


# =============================================================
# Chart 2: Emotion feature correlation heatmap
# =============================================================
def plot_correlation_heatmap(
    correlation_matrix: np.ndarray,
    feature_names: List[str],
    save_path: str = None,
    cmap: str = "RdBu_r"
) -> plt.Figure:
    """
    Plot 8x8 Pearson correlation heatmap of emotion features.
    """
    # Shorten labels for readability: "EmpathyGap_Mean" -> "EG_Mean"
    short_labels = []
    for name in feature_names:
        replacements = {
            "EmpathyGap": "EG",
            "DarkTriad": "DT",
            "Contagion": "CT",
            "Volatility": "VL",
        }
        short = name
        for k, v in replacements.items():
            short = short.replace(k, v)
        short_labels.append(short)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot heatmap
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    
    # Set axis ticks
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    
    # Display values in cells; white text on dark cells, black on light ones
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = correlation_matrix[i, j]
            text_color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson Correlation", fontsize=10)
    
    # Add module boundary lines
    n_features = len(feature_names)
    for split in [2, 4, 6]:
        if split < n_features:
            ax.axhline(y=split - 0.5, color="black", linewidth=2)
            ax.axvline(x=split - 0.5, color="black", linewidth=2)
    
    ax.set_title(
        "Emotion Feature Correlation Matrix",
        fontsize=15, fontweight="bold", pad=15
    )
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] Correlation heatmap saved: {save_path}")
    
    return fig


# Chart 3: VIF bar chart
# =============================================================
def plot_vif_chart(
    vif_dict: Dict[str, float],
    save_path: str = None,
) -> plt.Figure:
    """
    Plot VIF bar chart with reference lines at VIF=5 and VIF=10.
    """
    names = list(vif_dict.keys())
    values = [vif_dict[n] for n in names]
    
    # Cap infinite values for display
    capped_values = [min(v, 20.0) for v in values]
    colors = []
    for v in values:
        if np.isinf(v) or v > 10:
            colors.append("#e74c3c")   # red: severe
        elif v > 5:
            colors.append("#f39c12")   # orange: moderate
        else:
            colors.append("#27ae60")   # green: good
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, capped_values, width=0.6, color=colors, edgecolor="white", linewidth=1.2)
    
    # Reference lines
    ax.axhline(y=5, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8,
               label="VIF=5 (Moderate)")
    ax.axhline(y=10, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8,
               label="VIF=10 (Severe)")
    ax.axhline(y=1, color="#27ae60", linestyle=":", linewidth=1, alpha=0.6,
               label="VIF=1 (Ideal)")
    
    # Annotate bar tops (use "inf" for infinite values)
    for bar, raw_val, capped_val in zip(bars, values, capped_values):
        label = "inf" if np.isinf(raw_val) else f"{raw_val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, capped_val + 0.3,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Variance Inflation Factor")
    ax.set_title("VIF of Emotion Features", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(capped_values) * 1.15 + 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] VIF chart saved: {save_path}")
    
    return fig

def plot_pca_scree(
    explained_variance_ratio: np.ndarray,
    cumulative_variance: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot PCA scree plot: per-component explained variance (bars)
    plus cumulative variance curve.
    """
    n_components = len(explained_variance_ratio)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    x = np.arange(1, n_components + 1)
    
    # Left y-axis: per-component explained variance (bars)
    colors_bar = ["#3498db"] * 4 + ["#bdc3c7"] * max(0, n_components - 4)
    ax1.bar(x, explained_variance_ratio, width=0.55, color=colors_bar,
            edgecolor="white", linewidth=1.2, alpha=0.85,
            label="Explained Variance Ratio")

    # Annotate values
    for xi, ratio in zip(x, explained_variance_ratio):
        ax1.text(xi, ratio + 0.01, f"{ratio:.1%}", ha="center", 
                 fontsize=9, fontweight="bold")

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_ylim(0, max(explained_variance_ratio) * 1.25)
    ax1.set_xticks(x)
    
    # Right y-axis: cumulative variance curve
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_variance, "o-", color="#e74c3c", linewidth=2.5,
             markersize=8, label="Cumulative Variance")
    for xi, cum in zip(x, cumulative_variance):
        ax2.text(xi, cum + 0.015, f"{cum:.1%}", ha="center",
                 fontsize=9, fontweight="bold", color="#e74c3c")
    ax2.set_ylabel(
        "Cumulative Variance Ratio",
        color="#e74c3c"
    )
    ax2.set_ylim(0, 1.08)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    
    # 80% reference line
    ax2.axhline(y=0.80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(n_components + 0.3, 0.81, "80%", fontsize=8, color="gray")
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, 
               loc="center right")

    ax1.set_title(
        "PCA Scree Plot of Emotion Features",
        fontsize=15, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.2, linestyle="--")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] PCA scree plot saved: {save_path}")

    return fig


# Utility: generate all plots at once
# =============================================================


def plot_all(
    ablation_results: List[Dict] = None,
    correlation_matrix: np.ndarray = None,
    feature_names: List[str] = None,
    vif_dict: Dict[str, float] = None,
    pca_var: np.ndarray = None,
    pca_cum: np.ndarray = None,
    save_dir: str = "./ablation_results",
):
    """


    Generate all plots and save to the specified directory.
    Pass data for each section; missing data skips the corresponding plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if ablation_results:
        plot_ablation_bar_chart(
            ablation_results,
            save_path=os.path.join(save_dir, "ablation_bar_chart.png")
        )
    
    if correlation_matrix is not None and feature_names is not None:
        plot_correlation_heatmap(
            correlation_matrix, feature_names,
            save_path=os.path.join(save_dir, "correlation_heatmap.png")
        )
    
    if vif_dict:
        plot_vif_chart(
            vif_dict,
            save_path=os.path.join(save_dir, "vif_chart.png")
        )
    
    if pca_var is not None and pca_cum is not None:
        plot_pca_scree(
            pca_var, pca_cum,
            save_path=os.path.join(save_dir, "pca_scree.png")
        )

    print(f"\n[Plot] All charts saved to: {save_dir}/")
