"""
independence_verify.py — Statistical Independence Verification of Emotion Modules

Validates whether the 8-dimensional features produced by the four emotion modules
exhibit high collinearity. If one module can be linearly predicted from the
others (VIF > 10), its "independent contribution" is questionable.

Methods used:
  1. Pearson correlation matrix — checks pairwise linear correlation
  2. Variance Inflation Factor (VIF) — checks multicollinearity (how much each
     feature is inflated when regressed on all other features)
  3. PCA decomposition — if the 4 modules are truly independent, they should
     contribute 4 significant principal components
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os




# Add project root and Character Classification directory to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CC_DIR = os.path.join(_PROJECT_ROOT, "Character Classification")


def _add_cc_to_syspath():
    """Handle directory names with spaces to avoid ModuleNotFoundError."""
    if _CC_DIR not in sys.path:
        sys.path.insert(0, _CC_DIR)


sys.path.insert(0, _PROJECT_ROOT)

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def _load_psycho_features(
    db_path: str = None,
    csv_path: str = None,
    dataset: str = None,
    max_agents: int = 200
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract the pure 8-dim emotion-psychology feature matrix from a dataset.
    Fully aligned with the Character Classification core pipeline data loading.

    Returns:
        feature_matrix : (N, 8) numpy array
        feature_names  : list of 8 feature names
    """
    module_names = ["EmpathyGap_Mean", "EmpathyGap_Max",
                    "DarkTriad_Mean",  "DarkTriad_Max",
                    "Contagion_Mean",  "Contagion_Max",
                    "Volatility_Mean", "Volatility_Max"]

    _add_cc_to_syspath()
    import pandas as pd
    from config import (
        resolve_dataset_paths, load_label_frame, build_text_fields,
    )
    from new_feature_extractor import MultimodalExtractor

    db_file, csv_file = resolve_dataset_paths(db_path, csv_path, dataset)


    print(f"[IndependenceVerify] Data paths:")
    print(f"  DB : {db_file}")
    print(f"  CSV: {csv_file}")

    df_labels = load_label_frame(csv_file, db_file)
    df_labels["user_id"] = df_labels["user_id"].astype(str)
    _, tweets_joined, _ = build_text_fields(df_labels)

    extractor = MultimodalExtractor(psychology_mode="full", verbose_progress=False)
    psycho_matrix = extractor._extract_llm_native_psychology(tweets_joined)

    if max_agents and psycho_matrix.shape[0] > max_agents:
        rng = np.random.default_rng(42)
        indices = rng.choice(psycho_matrix.shape[0], size=max_agents, replace=False)
        psycho_matrix = psycho_matrix[indices]

    return psycho_matrix, module_names


def compute_vif(
    feature_matrix: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    n_features = feature_matrix.shape[1]
    vif_dict = {}

    for j in range(n_features):
        y = feature_matrix[:, j]
        X_without_j = np.delete(feature_matrix, j, axis=1)
        if np.std(X_without_j) == 0:
            vif_dict[feature_names[j]] = np.inf
            continue
        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_without_j, y)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot < 1e-12:
                vif_dict[feature_names[j]] = np.inf
            else:
                r_squared = 1 - ss_res / ss_tot
                if r_squared >= 1.0:
                    vif_dict[feature_names[j]] = np.inf
                else:
                    vif_dict[feature_names[j]] = float(1.0 / (1.0 - r_squared))
        except Exception:
            vif_dict[feature_names[j]] = np.inf
    return vif_dict


def compute_module_level_independence(
) -> Dict[str, Dict[str, float]]:
    """
    Compute module-level independence (rather than per-column VIF).
    Merge each module's (Mean, Max) columns into a module vector,
    then predict that module's two columns using the six columns
    from the other three modules, computing R^2.
    
    Returns:
      {
        "EmpathyGap":  {"R_squared": 0.XX, "VIF_equivalent": 1/(1-R^2)},
        "DarkTriad":   {"R_squared": 0.XX, "VIF_equivalent": 1/(1-R^2)},
        ...
      }
    """
    # Module-to-column index mapping
    modules = {
        "EmpathyGap":  [0, 1],
        "DarkTriad":   [2, 3],
        "Contagion":   [4, 5],
        "Volatility":  [6, 7],
    }

    results = {}
    for mod_name, cols in modules.items():
        # Dependent variables: Mean + Max of this module
        Y = feature_matrix[:, cols]
        # Predictors: 6 columns from other three modules
        other_cols = [c for m, cs in modules.items() if m != mod_name for c in cs]
        X = feature_matrix[:, other_cols]

        # Regress each target column separately, then average R^2
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
            "is_independent": avg_r2 < 0.5 
        }

    return results


def compute_pca_decomposition(
    feature_matrix: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    
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
    lines = []
    lines.append("=" * 70)
    lines.append(" Independence verification report of sentiment module statistics")
    lines.append("=" * 70)

    lines.append("\n--- (1) Pearson correlation coefficient matrix ---")
    lines.append(f"{'':>15} {'EmpathyGap':>12} {'DarkTriad':>12} {'Contagion':>12} {'Volatility':>12}")
    for i, mod_i in enumerate(module_names):
        vals = []
        for j, mod_j in enumerate(module_names):
            if i == j:
                vals.append("1.0000")
            else:
                cols_i = [i * 2, i * 2 + 1]
                cols_j = [j * 2, j * 2 + 1]
                cors = []
                for ci in cols_i:
                    for cj in cols_j:
                        cors.append(abs(correlation_matrix[ci, cj]))
                vals.append(f"{np.mean(cors):.4f}")
        lines.append(f"{mod_i:>15} " + " ".join(f"{v:>12}" for v in vals))




    # -- 2. Per-column VIF --
    lines.append("\n--- (2) Variance Inflation Factor (per-column) ---")
    lines.append(f"{'Feature':<22} {'VIF':>8}  {'Judgment'}")
    for name in feature_names:
        vif_val = vif_dict.get(name, float('inf'))
        if vif_val > 10:

            flag = "SEVERE collinearity"
        elif vif_val > 5:

            flag = "MODERATE collinearity"
        else:


            flag = "OK (independent)"
        vif_str = f"{vif_val:.2f}" if not np.isinf(vif_val) else "inf"
        lines.append(f"  {name:<20} {vif_str:>8}  {flag}")

    #3
    lines.append("\n--- (3) Independence  ---")
    lines.append(f"{'module':<15} {'R²':>8}  {'VIF_equiv':>10}  {'Independence'}")
    for mod_name, info in module_independence.items():
        flag = " Independence" if info["is_independent"] else "redundancy"
        lines.append(
            f"  {mod_name:<13} {info['R_squared']:>8.4f}  "
            f"{info['VIF_equivalent']:>10.4f}  {flag}"
        )
    lines.append(f"{'principal component':<10} {'Variance Explained Proportion':>12}  {'Cumulative variance ratio':>14}")
    for i, (ratio, cum) in enumerate(zip(pca_variance_ratio, cumulative_variance)):
        lines.append(f"  PC{i+1:<8} {ratio:>12.4f}  {cum:>14.4f}")
    
    pc4_cum = cumulative_variance[min(3, len(cumulative_variance)-1)]
    if pc4_cum >= 0.80:
        lines.append(f"\n    {pc4_cum:.1%} variance")
        lines.append(f"redundant")
    else:
        lines.append(f"\n   {pc4_cum:.1%} variance")

    vif_ok = all(not np.isinf(v) and v < 10 for v in vif_dict.values())

    all_ok = vif_ok and module_ok and pca_ok
    if all_ok:
        lines.append("The overall independence of the four emotion modules was verified.")
        if not vif_ok:
            lines.append("The VIF of some features is too high, indicating a risk of column-level collinearity.")
        if not pca_ok:
            lines.append("The PCA decomposition display module has a high degree of information overlap.")
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def verify_independence(
    db_path: str = None,
    csv_path: str = None,
    dataset: str = None,
    max_agents: int = 200,
    save_dir: str = "./ablation_results"
) -> Dict:
    """
    Main entry point: run the complete emotion module independence verification.

    Parameters:
        db_path     : SQLite database path
        csv_path    : CSV label file path
        dataset     : dataset name (e.g., agent72, twibot1000)
        max_agents  : maximum samples to analyze (for speed)
        save_dir    : output directory for plots

    Returns:
        result_dict : dictionary containing all analysis results
    """
    os.makedirs(save_dir, exist_ok=True)

    module_names = ["EmpathyGap", "DarkTriad", "Contagion", "Volatility"]
    feature_names = [
        "EmpathyGap_Mean",   "EmpathyGap_Max",
        "DarkTriad_Mean",    "DarkTriad_Max",
        "Contagion_Mean",    "Contagion_Max",
        "Volatility_Mean",   "Volatility_Max",
    ]

    # ---- Step 1: Load data ----
    print("\n" + "=" * 60)
    print("  Emotion Module Independence Verification - Data Loading")
    print("=" * 60)
    feature_matrix, _ = _load_psycho_features(
        db_path=db_path, csv_path=csv_path, dataset=dataset, max_agents=max_agents
    )
    print(f"  [OK] Loaded emotion feature matrix: {feature_matrix.shape}")

    # ---- Step 2: Correlation matrix ----
    print("\n[Step 1/4] Computing Pearson correlation matrix ...")
    correlation_matrix = np.corrcoef(feature_matrix.T)
    # Take absolute values for strength assessment
    abs_corr = np.abs(correlation_matrix)

    # ---- Step 3: VIF analysis ----
    print("[Step 2/4] Computing per-column VIF ...")
    vif_dict = compute_vif(feature_matrix, feature_names)

    # ---- Step 4: Module-level independence ----
    print("[Step 3/4] Computing module-level independence ...")
    module_independence = compute_module_level_independence(feature_matrix)

    # ---- Step 5: PCA ----
    print("[Step 4/4] PCA variance decomposition ...")
    pca_var, pca_cum, pca_loadings = compute_pca_decomposition(
        feature_matrix, feature_names
    )

    # ---- Generate report ----
    report = generate_independence_report(
        correlation_matrix, vif_dict, module_independence,
        pca_var, pca_cum, feature_names, module_names
    )
    print("\n" + report)

    # Save report
    report_path = os.path.join(save_dir, "independence_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[OK] Independence report saved to: {report_path}")

    # ---- Plotting ----
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

    # Return structured results
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
    