"""
ablation_plot.py - 消融实验可视化模块

提供四类图表：
  1. 消融柱状图    — 各消融配置的准确率 / AUC / F1 对比 (带误差线)
  2. 相关性热力图  — 8 维情感特征的两两 Pearson 相关
  3. VIF 柱状图     — 方差膨胀因子 (带阈值参考线)
  4. PCA 碎石图     — 各主成分方差解释比例

支持中文字体回退，导出 300 dpi PNG。
"""

import matplotlib
matplotlib.use("Agg")  # 无头后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import Dict, List, Optional
import os

# =============================================================
# 全局样式与中文字体配置
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

# 尝试加载中文字体 (Windows → Mac → Linux 回退)
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
    # 如果中文字体都找不到，使用英文渲染并给出提示
    plt.rcParams["font.family"] = "sans-serif"
    print("[Plot] 未检测到中文字体，将使用英文标签。")


# =============================================================
# 颜色常量
# =============================================================
COLOR_FULL_MODEL = "#2c3e50"       # 深蓝灰
COLOR_ABLATION_BASE = "#e74c3c"    # 消融基准红色
COLORS_ABLATION = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#95a5a6"]


def _use_cn_labels():
    """是否使用中文标签"""
    return _CN_FONT_LOADED


# =============================================================
# 图表 1：消融实验对比柱状图 (核心输出)
# =============================================================
def plot_ablation_bar_chart(
    ablation_results: List[Dict],
    metrics: List[str] = None,
    save_path: str = None,
    show_values: bool = True,
) -> plt.Figure:
    """
    绘制消融实验对比柱状图。
    
    参数:
        ablation_results : run_ablation() 返回的结果列表
        metrics          : 要绘制的指标，默认 ["accuracy", "auc", "f1"]
        save_path        : 图片保存路径
        show_values      : 是否在柱顶标注数值
    
    返回:
        fig : matplotlib Figure 对象
    """
    if metrics is None:
        metrics = ["accuracy", "auc", "f1"]
    
    if not ablation_results:
        print("[Plot] ablation_results 为空，跳过绘图")
        return None

    cn = _use_cn_labels()
    n_configs = len(ablation_results)
    n_metrics = len(metrics)
    config_names = [r["config"] for r in ablation_results]
    
    # 指标显示名映射
    metric_display = {
        "accuracy":  "准确率 Accuracy" if cn else "Accuracy",
        "auc":       "AUC" if cn else "AUC",
        "f1":        "F1 分数" if cn else "F1 Score",
        "precision": "精确率 Precision" if cn else "Precision",
        "recall":    "召回率 Recall" if cn else "Recall",
    }
    
    # ---- 创建子图 ----
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6), 
                              squeeze=False)
    axes = axes[0]  # 取第一行
    
    bar_width = 0.65
    x_positions = np.arange(n_configs)
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        
        values = [r.get(metric, 0.0) for r in ablation_results]
        stds = [r.get(f"{metric}_std", 0.0) for r in ablation_results]
        colors = [r.get("color", "#2c3e50") for r in ablation_results]
        
        # 绘制柱状图
        bars = ax.bar(
            x_positions, values, bar_width,
            yerr=stds, capsize=5,
            color=colors, edgecolor="white", linewidth=1.2,
            error_kw={"linewidth": 1.5, "capthick": 1.5}
        )
        
        # 在柱顶标注数值
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
        
        # 样式
        ax.set_xlabel("")
        ax.set_ylabel(metric_display.get(metric, metric))
        ax.set_title(metric_display.get(metric, metric))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(config_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # 在第一个柱 (Full Model) 上画基准线
        if values:
            baseline = values[0]
            ax.axhline(y=baseline, color="#2c3e50", linestyle="-", 
                       linewidth=1.5, alpha=0.6, label="完整模型基线" if cn else "Full Model Baseline")
            ax.legend(fontsize=8)
    
    fig.suptitle(
        "情感模块消融实验" if cn else "Emotion Module Ablation Study",
        fontsize=17, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", 
                    facecolor="white", edgecolor="none")
        print(f"[Plot] 消融柱状图已保存: {save_path}")
    
    return fig


# =============================================================
# 图表 2：情感特征相关性热力图
# =============================================================
def plot_correlation_heatmap(
    correlation_matrix: np.ndarray,
    feature_names: List[str],
    save_path: str = None,
    cmap: str = "RdBu_r"
) -> plt.Figure:
    """
    绘制 8×8 情感特征 Pearson 相关系数热力图。
    """
    cn = _use_cn_labels()
    
    # 使用更短的可读标签
    short_labels = []
    for name in feature_names:
        # 缩短标签: "EmpathyGap_Mean" → "EG_Mean"
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
    
    # 绘制热力图
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    
    # 设置坐标轴
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    
    # 在格子内显示数值
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = correlation_matrix[i, j]
            # 对角线用白色，其余用黑色或白色决定对比度
            if abs(val) > 0.7:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson 相关系数" if cn else "Pearson Correlation", fontsize=10)
    
    # 添加模块分界线
    n_features = len(feature_names)
    for split in [2, 4, 6]:
        if split < n_features:
            ax.axhline(y=split - 0.5, color="black", linewidth=2)
            ax.axvline(x=split - 0.5, color="black", linewidth=2)
    
    ax.set_title(
        "情感模块特征相关矩阵" if cn else "Emotion Feature Correlation Matrix",
        fontsize=15, fontweight="bold", pad=15
    )
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] 相关热力图已保存: {save_path}")
    
    return fig


# =============================================================
# 图表 3：VIF 方差膨胀因子柱状图
# =============================================================
def plot_vif_chart(
    vif_dict: Dict[str, float],
    save_path: str = None,
) -> plt.Figure:
    """
    绘制 VIF 柱状图，标注 5 和 10 两条参考线。
    """
    cn = _use_cn_labels()
    
    names = list(vif_dict.keys())
    values = [vif_dict[n] for n in names]
    
    # 截断无穷大值用于显示
    capped_values = [min(v, 20.0) for v in values]
    colors = []
    for v in values:
        if np.isinf(v) or v > 10:
            colors.append("#e74c3c")   # 红色：严重
        elif v > 5:
            colors.append("#f39c12")   # 橙色：中度
        else:
            colors.append("#27ae60")   # 绿色：良好
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, capped_values, width=0.6, color=colors, edgecolor="white", linewidth=1.2)
    
    # 参考线
    ax.axhline(y=5, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8,
               label="VIF=5 (中度共线性)" if cn else "VIF=5 (Moderate)")
    ax.axhline(y=10, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8,
               label="VIF=10 (严重共线性)" if cn else "VIF=10 (Severe)")
    ax.axhline(y=1, color="#27ae60", linestyle=":", linewidth=1, alpha=0.6,
               label="VIF=1 (理想正交)" if cn else "VIF=1 (Ideal)")
    
    # 在柱顶标注 (对于无穷大值用 "∞" 显示)
    for bar, raw_val, capped_val in zip(bars, values, capped_values):
        if np.isinf(raw_val):
            label = "∞"
        else:
            label = f"{raw_val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, capped_val + 0.3,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("方差膨胀因子 (VIF)" if cn else "Variance Inflation Factor")
    ax.set_title(
        "情感特征方差膨胀因子 (VIF)" if cn else "VIF of Emotion Features",
        fontsize=15, fontweight="bold"
    )
    ax.set_ylim(0, max(capped_values) * 1.15 + 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] VIF 图表已保存: {save_path}")
    
    return fig


# =============================================================
# 图表 4：PCA 碎石图
# =============================================================
def plot_pca_scree(
    explained_variance_ratio: np.ndarray,
    cumulative_variance: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    绘制 PCA 碎石图：并列显示单主成分方差解释 + 累计方差曲线。
    """
    cn = _use_cn_labels()
    n_components = len(explained_variance_ratio)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    x = np.arange(1, n_components + 1)
    
    # 左侧 y 轴：单成分方差解释比例 (柱状图)
    colors_bar = ["#3498db"] * 4 + ["#bdc3c7"] * max(0, n_components - 4)
    ax1.bar(x, explained_variance_ratio, width=0.55, color=colors_bar,
            edgecolor="white", linewidth=1.2, alpha=0.85,
            label="方差解释比例" if cn else "Explained Variance Ratio")
    
    # 标注数值
    for xi, ratio in zip(x, explained_variance_ratio):
        ax1.text(xi, ratio + 0.01, f"{ratio:.1%}", ha="center", 
                 fontsize=9, fontweight="bold")
    
    ax1.set_xlabel("主成分" if cn else "Principal Component")
    ax1.set_ylabel("方差解释比例" if cn else "Explained Variance Ratio")
    ax1.set_ylim(0, max(explained_variance_ratio) * 1.25)
    ax1.set_xticks(x)
    
    # 右侧 y 轴：累计方差曲线
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_variance, "o-", color="#e74c3c", linewidth=2.5, 
             markersize=8, label="累计方差" if cn else "Cumulative Variance")
    for xi, cum in zip(x, cumulative_variance):
        ax2.text(xi, cum + 0.015, f"{cum:.1%}", ha="center",
                 fontsize=9, fontweight="bold", color="#e74c3c")
    ax2.set_ylabel(
        "累计方差解释比例" if cn else "Cumulative Variance Ratio",
        color="#e74c3c"
    )
    ax2.set_ylim(0, 1.08)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    
    # 80% 参考线
    ax2.axhline(y=0.80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(n_components + 0.3, 0.81, "80%", fontsize=8, color="gray")
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, 
               loc="center right")
    
    ax1.set_title(
        "情感特征 PCA 方差分解" if cn else "PCA Scree Plot of Emotion Features",
        fontsize=15, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.2, linestyle="--")
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] PCA 碎石图已保存: {save_path}")
    
    return fig


# =============================================================
# 辅助：一次性绘制所有图表
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
    一次性生成所有图表并保存到指定目录。
    按需传入各部分数据，缺省则跳过对应图表。
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
    
    print(f"\n[Plot] 所有图表已保存至: {save_dir}/")
