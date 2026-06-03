"""
independence_verify.py - 情感模块统计独立性验证

验证四个情感模块输出的 8 维特征之间是否存在高度共线性。
如果某个模块可以由其他模块线性预测 (VIF > 10)，则其"独立贡献"存疑。

使用的方法：
  1. Pearson 相关矩阵 —— 检查成对线性相关
  2. 方差膨胀因子 VIF —— 检查多重共线性 (每个特征被其余特征回归后的膨胀程度)
  3. PCA 主成分分析 —— 如果 4 个模块确实独立，应贡献 4 个显著主成分
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# 将项目根目录加入 sys.path，确保能导入 emotional_analysis 和主项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def _load_psycho_features(
    db_path: str = None,
    csv_path: str = None,
    max_agents: int = 200
) -> Tuple[np.ndarray, List[str]]:
    """
    数据加载层：从主项目中提取 8 维情感特征矩阵。
    自动检测可用的数据源，优先复用主项目的特征提取管线。
    
    返回:
        feature_matrix : (N, 8) 的 numpy 数组
        feature_names  : 8 个特征名列表
    """
    module_names = ["EmpathyGap_Mean", "EmpathyGap_Max",
                    "DarkTriad_Mean",  "DarkTriad_Max",
                    "Contagion_Mean",  "Contagion_Max",
                    "Volatility_Mean", "Volatility_Max"]

    # ------------- 方案 A：直接读取已缓存的特征矩阵 ------------
    # 优先尝试从 main_detector.py 的路径配置中加载数据并实时提取
    try:
        from main_detector import DB_FILE, CSV_FILE

        import pandas as pd
        import ast

        db_path = db_path or DB_FILE
        csv_path = csv_path or CSV_FILE

        print(f"[IndependenceVerify] 从主项目配置加载数据:")
        print(f"  DB : {db_path}")
        print(f"  CSV: {csv_path}")

        # 载入标签数据
        df_labels = pd.read_csv(csv_path)
        df_labels['is_bad'] = df_labels['user_type'].apply(
            lambda x: 1 if 'bad' in str(x).lower() else 0
        )

        # 主键对齐
        if 'name' in df_labels.columns:
            df_labels['global_node_id'] = df_labels['name'].astype(str).str.strip()
        else:
            df_labels['global_node_id'] = df_labels['user_id'].astype(str).str.strip()

        # 重组推文文本
        summary_texts = []
        for _, row in df_labels.iterrows():
            bio = str(row.get('user_char', row.get('description', '')))
            tweets_raw = str(row.get('previous_tweets', '[]')).strip()
            if tweets_raw.startswith('[') and tweets_raw.endswith(']'):
                try:
                    tweet_list = ast.literal_eval(tweets_raw)
                    tweets_formatted = " | ".join(
                        [str(t).strip() for t in tweet_list]
                    ) if isinstance(tweet_list, list) else tweets_raw
                except (ValueError, SyntaxError):
                    tweets_formatted = tweets_raw
            else:
                tweets_formatted = tweets_raw
            summary_texts.append(f"Bio: {bio}. Recent actions: {tweets_formatted}")

        # 从 MultimodalExtractor 提取心理学特征
        from Character_Classification.new_feature_extractor import MultimodalExtractor
        extractor = MultimodalExtractor(psychology_mode="full", verbose_progress=False)
        user_ids = df_labels['user_id'].tolist()

        # 直接调用 _extract_llm_native_psychology 拿到 (N, 8) 矩阵
        import re
        parsed_tweets = []
        for s in summary_texts:
            m = re.search(
                r'Bio:\s*(.*?)\.\s*Recent actions:\s*(.*)$',
                str(s), flags=re.IGNORECASE
            )
            parsed_tweets.append(m.group(2).strip() if m else "")

        psycho_matrix = extractor._extract_llm_native_psychology(parsed_tweets)

        # 限制样本数以提速
        if max_agents and psycho_matrix.shape[0] > max_agents:
            rng = np.random.default_rng(42)
            indices = rng.choice(psycho_matrix.shape[0], size=max_agents, replace=False)
            psycho_matrix = psycho_matrix[indices]
        
        return psycho_matrix, module_names

    except Exception as e:
        print(f"[IndependenceVerify] 主项目管线加载失败: {e}")
        print("  请检查 main_detector.py 中的 DB_FILE / CSV_FILE 路径配置。")
        raise RuntimeError(f"无法加载情感特征数据: {e}")


def compute_vif(
    feature_matrix: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    计算每个特征的方差膨胀因子 (Variance Inflation Factor)
    
    VIF_j = 1 / (1 - R²_j)
    其中 R²_j 是用其余特征线性回归预测特征 j 的 R 方。
    VIF > 10 表示存在严重多重共线性。
    VIF > 5  表示可能存在共线性问题，需要关注。
    VIF 接近 1 表示该特征与其余特征几乎正交，独立性良好。
    """
    n_features = feature_matrix.shape[1]
    vif_dict = {}

    for j in range(n_features):
        # 将第 j 列作为因变量，其余列作为自变量
        y = feature_matrix[:, j]
        X_without_j = np.delete(feature_matrix, j, axis=1)

        # 检查 X 是否包含常数列 (方差为 0)
        if np.std(X_without_j) == 0:
            vif_dict[feature_names[j]] = np.inf
            continue

        # 线性回归
        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_without_j, y)
            y_pred = reg.predict(X_without_j)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot < 1e-12:
                # 因变量方差为零，完全可被其他变量解释
                vif_dict[feature_names[j]] = np.inf
            else:
                r_squared = 1 - ss_res / ss_tot
                # 防止 r_squared 因数值误差 >= 1
                if r_squared >= 1.0:
                    vif_dict[feature_names[j]] = np.inf
                else:
                    vif_dict[feature_names[j]] = float(1.0 / (1.0 - r_squared))
        except Exception:
            vif_dict[feature_names[j]] = np.inf

    return vif_dict


def compute_module_level_independence(
    feature_matrix: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    计算模块级独立性指标 (而非逐列 VIF)。
    将每个模块的 (Mean, Max) 两列合并为一个模块向量，
    然后用其余三个模块的六列预测该模块的两列，计算 R²。
    
    返回格式:
      {
        "EmpathyGap":  {"R²": 0.XX, "VIF_equiv": 1/(1-R²)},
        "DarkTriad":   {"R²": 0.XX, "VIF_equiv": 1/(1-R²)},
        ...
      }
    """
    # 模块与其列索引
    modules = {
        "EmpathyGap":  [0, 1],
        "DarkTriad":   [2, 3],
        "Contagion":   [4, 5],
        "Volatility":  [6, 7],
    }

    results = {}
    for mod_name, cols in modules.items():
        # 因变量：该模块的 Mean + Max
        Y = feature_matrix[:, cols]
        # 自变量：其余三模块的 6 列
        other_cols = [c for m, cs in modules.items() if m != mod_name for c in cs]
        X = feature_matrix[:, other_cols]

        # 对每个因变量列分别回归，取平均 R²
        r_squared_vals = []
        for col_idx in range(Y.shape[1]):
            y_col = Y[:, col_idx]
            reg = LinearRegression(fit_intercept=True)
            try:
                reg.fit(X, y_col)
                y_pred = reg.predict(X)
                ss_res = np.sum((y_col - y_pred) ** 2)
                ss_tot = np.sum((y_col - np.mean(y_col)) ** 2)
                if ss_tot > 1e-12:
                    r2 = max(0.0, min(1.0 - ss_res / ss_tot, 0.9999))
                else:
                    r2 = 1.0
            except Exception:
                r2 = 1.0
            r_squared_vals.append(r2)

        avg_r2 = float(np.mean(r_squared_vals))
        vif_equiv = float(1.0 / (1.0 - avg_r2)) if avg_r2 < 1.0 else np.inf

        results[mod_name] = {
            "R_squared": round(avg_r2, 4),
            "VIF_equivalent": round(vif_equiv, 4),
            "is_independent": avg_r2 < 0.5  # 经验阈值：R² < 0.5 视为独立性较好
        }

    return results


def compute_pca_decomposition(
    feature_matrix: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA 分解：检验 8 维特征是否可被 4 个主成分充分解释。
    如果四个模块独立，则方差应由 4 个主成分共同分担。
    
    返回:
        explained_variance_ratio : 各主成分方差解释比例
        cumulative_variance      : 累计方差解释比例
        loadings                 : 每个主成分对各原始特征的载荷
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=min(8, feature_matrix.shape[1]))
    pca.fit(scaled)

    loadings = pca.components_  # shape: (n_components, 8)

    return (pca.explained_variance_ratio_,
            np.cumsum(pca.explained_variance_ratio_),
            loadings)


def generate_independence_report(
    correlation_matrix: np.ndarray,
    vif_dict: Dict[str, float],
    module_independence: Dict[str, Dict[str, float]],
    pca_variance_ratio: np.ndarray,
    cumulative_variance: np.ndarray,
    feature_names: List[str],
    module_names: List[str],
) -> str:
    """生成人类可读的独立性验证报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("  情感模块统计独立性验证报告")
    lines.append("=" * 70)

    # -- 1. 成对相关系数矩阵 --
    lines.append("\n--- (1) Pearson 相关系数矩阵 (模块级) ---")
    lines.append(f"{'':>15} {'EmpathyGap':>12} {'DarkTriad':>12} {'Contagion':>12} {'Volatility':>12}")
    for i, mod_i in enumerate(module_names):
        vals = []
        for j, mod_j in enumerate(module_names):
            if i == j:
                vals.append("1.0000")
            else:
                # 取模块两列的相关系数均值
                cols_i = [i * 2, i * 2 + 1]
                cols_j = [j * 2, j * 2 + 1]
                cors = []
                for ci in cols_i:
                    for cj in cols_j:
                        cors.append(abs(correlation_matrix[ci, cj]))
                vals.append(f"{np.mean(cors):.4f}")
        lines.append(f"{mod_i:>15} " + " ".join(f"{v:>12}" for v in vals))

    # -- 2. VIF 逐列 --
    lines.append("\n--- (2) 方差膨胀因子 VIF (逐列) ---")
    lines.append(f"{'特征名':<22} {'VIF':>8}  {'判定'}")
    for name in feature_names:
        vif_val = vif_dict.get(name, float('inf'))
        if vif_val > 10:
            flag = "⚠ 严重共线性"
        elif vif_val > 5:
            flag = "⚡ 中度共线性"
        else:
            flag = "✅ 独立性良好"
        vif_str = f"{vif_val:.2f}" if not np.isinf(vif_val) else "∞"
        lines.append(f"  {name:<20} {vif_str:>8}  {flag}")

    # -- 3. 模块级独立性 --
    lines.append("\n--- (3) 模块级独立性 (其他三模块 → 该模块的 R²) ---")
    lines.append(f"{'模块':<15} {'R²':>8}  {'VIF_equiv':>10}  {'独立性'}")
    for mod_name, info in module_independence.items():
        flag = "✅ 独立" if info["is_independent"] else "⚠ 可能冗余"
        lines.append(
            f"  {mod_name:<13} {info['R_squared']:>8.4f}  "
            f"{info['VIF_equivalent']:>10.4f}  {flag}"
        )

    # -- 4. PCA 方差分解 --
    lines.append("\n--- (4) PCA 方差分解 (8 维 = 4 模块 × 2 统计量) ---")
    lines.append(f"{'主成分':<10} {'方差解释比例':>12}  {'累计方差比例':>14}")
    for i, (ratio, cum) in enumerate(zip(pca_variance_ratio, cumulative_variance)):
        lines.append(f"  PC{i+1:<8} {ratio:>12.4f}  {cum:>14.4f}")
    
    # 判断：前 4 个主成分应解释大部分方差
    pc4_cum = cumulative_variance[min(3, len(cumulative_variance)-1)]
    if pc4_cum >= 0.80:
        lines.append(f"\n  ✅ 前 4 个主成分累计解释 {pc4_cum:.1%} 方差，")
        lines.append(f"     说明 4 个情感模块共同贡献了绝大多数信息量，")
        lines.append(f"     未出现单一模块完全主导的情形。")
    elif pc4_cum >= 0.60:
        lines.append(f"\n  ⚡ 前 4 个主成分累计解释 {pc4_cum:.1%} 方差，")
        lines.append(f"     4 模块信息有一定重叠，但也可能存在冗余模块。")
    else:
        lines.append(f"\n  ⚠ 前 4 个主成分仅解释 {pc4_cum:.1%} 方差，")
        lines.append(f"     建议检查是否有模块贡献极小或高度冗余。")

    # -- 5. 综合结论 --
    lines.append("\n--- (5) 综合结论 ---")
    vif_ok = all(not np.isinf(v) and v < 10 for v in vif_dict.values())
    module_ok = all(info["is_independent"] for info in module_independence.values())
    pca_ok = pc4_cum >= 0.60

    all_ok = vif_ok and module_ok and pca_ok
    if all_ok:
        lines.append("  ✅ 四个情感模块总体独立性通过验证。")
        lines.append("     各模块捕捉了不同维度的心理信号，消融实验结论可信。")
    else:
        if not vif_ok:
            lines.append("  ⚠ 部分特征的 VIF 偏高，存在列级共线性风险。")
        if not module_ok:
            lines.append("  ⚠ 部分模块可由其他模块部分预测，消融时需关注交互效应。")
        if not pca_ok:
            lines.append("  ⚠ PCA 分解显示模块信息重叠度偏高。")
        lines.append("     建议在消融实验中额外报告模块组合消融 (Combination Ablation)。")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def verify_independence(
    db_path: str = None,
    csv_path: str = None,
    max_agents: int = 200,
    save_dir: str = "./ablation_results"
) -> Dict:
    """
    主入口：运行完整的情感模块独立性验证流程。

    参数:
        db_path     : SQLite 数据库路径 (默认从 main_detector 读取)
        csv_path    : CSV 标签文件路径
        max_agents  : 最大分析样本数 (加速)
        save_dir    : 图表保存目录

    返回:
        result_dict : 包含所有分析结果的字典，供消融实验参考
    """
    os.makedirs(save_dir, exist_ok=True)

    module_names = ["EmpathyGap", "DarkTriad", "Contagion", "Volatility"]
    feature_names = [
        "EmpathyGap_Mean",   "EmpathyGap_Max",
        "DarkTriad_Mean",    "DarkTriad_Max",
        "Contagion_Mean",    "Contagion_Max",
        "Volatility_Mean",   "Volatility_Max",
    ]

    # ---- Step 1: 加载数据 ----
    print("\n" + "█" * 60)
    print("  情感模块独立性验证 — 数据加载")
    print("█" * 60)
    feature_matrix, _ = _load_psycho_features(
        db_path=db_path, csv_path=csv_path, max_agents=max_agents
    )
    print(f"  [OK] 加载情感特征矩阵: {feature_matrix.shape}")

    # ---- Step 2: 相关矩阵 ----
    print("\n[Step 1/4] 计算 Pearson 相关系数矩阵 ...")
    correlation_matrix = np.corrcoef(feature_matrix.T)
    # 取绝对值用于判定强度
    abs_corr = np.abs(correlation_matrix)

    # ---- Step 3: VIF 分析 ----
    print("[Step 2/4] 计算逐列 VIF ...")
    vif_dict = compute_vif(feature_matrix, feature_names)

    # ---- Step 4: 模块级独立性 ----
    print("[Step 3/4] 计算模块级独立性 ...")
    module_independence = compute_module_level_independence(feature_matrix)

    # ---- Step 5: PCA ----
    print("[Step 4/4] PCA 方差分解 ...")
    pca_var, pca_cum, pca_loadings = compute_pca_decomposition(
        feature_matrix, feature_names
    )

    # ---- 生成报告 ----
    report = generate_independence_report(
        correlation_matrix, vif_dict, module_independence,
        pca_var, pca_cum, feature_names, module_names
    )
    print("\n" + report)

    # 保存报告
    report_path = os.path.join(save_dir, "independence_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[OK] 独立性验证报告已保存至: {report_path}")

    # ---- 绘图 ----
    from ablation.ablation_plot import (
        plot_correlation_heatmap,
        plot_vif_chart,
        plot_pca_scree,
    )
    plot_correlation_heatmap(
        correlation_matrix, feature_names,
        save_path=os.path.join(save_dir, "correlation_heatmap.png")
    )
    plot_vif_chart(
        vif_dict,
        save_path=os.path.join(save_dir, "vif_chart.png")
    )
    plot_pca_scree(
        pca_var, pca_cum,
        save_path=os.path.join(save_dir, "pca_scree.png")
    )

    # 返回结构化结果
    return {
        "feature_matrix": feature_matrix,
        "correlation_matrix": correlation_matrix,
        "vif_dict": vif_dict,
        "module_independence": module_independence,
        "pca_variance_ratio": pca_var,
        "pca_cumulative_variance": pca_cum,
        "pca_loadings": pca_loadings,
        "feature_names": feature_names,
        "module_names": module_names,
        "report": report,
    }