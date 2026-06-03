"""
ablation_experiment.py - 情感模块消融实验核心逻辑

功能：
  1. 加载完整多模态特征矩阵
  2. 按四种策略逐一置零对应情感模块的特征列
  3. 使用 XGBoost + 重复分层 K 折交叉验证评估每种配置
  4. 汇总性能差异，输出结构化结果供绘图模块使用

消融配置：
  - Full Model           : 保留全部 4 个情感模块 (基线)
  - Without EmpathyGap   : 置零 EmpathyGap 列 (索引 0,1)
  - Without DarkTriad     : 置零 DarkTriad 列 (索引 2,3)
  - Without Contagion     : 置零 Contagion 列 (索引 4,5)
  - Without Volatility    : 置零 Volatility 列 (索引 6,7)
  - No Emotional Modules  : 置零全部 8 维情感特征 (用于对照)
"""

import os
import sys
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================
# 消融配置表：定义每个实验场景的名称与要置零的列索引
# ============================================================
ABLATION_CONFIGS = OrderedDict([
    ("Full Model", {
        "description": "完整模型 (基线)",
        "zero_indices": [],  # 不置零任何列
        "color": "#2c3e50"
    }),
    ("− EmpathyGap", {
        "description": "移除认知共情错位差",
        "zero_indices": [0, 1],
        "color": "#e74c3c"
    }),
    ("− DarkTriad", {
        "description": "移除暗黑三角人格",
        "zero_indices": [2, 3],
        "color": "#e67e22"
    }),
    ("− Contagion", {
        "description": "移除无阻尼情感传染",
        "zero_indices": [4, 5],
        "color": "#f1c40f"
    }),
    ("− Volatility", {
        "description": "移除功能性情感瞬切",
        "zero_indices": [6, 7],
        "color": "#3498db"
    }),
    ("No Emotional Modules", {
        "description": "移除全部情感模块 (对照)",
        "zero_indices": list(range(8)),
        "color": "#95a5a6"
    }),
])


def _load_full_feature_matrix(
    db_path: str = None,
    csv_path: str = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    复用主项目的 UltimateDeceptionDetector 管线，
    获取最终的 (X, y, feature_names) 用于消融实验。
    """
    from main_detector import UltimateDeceptionDetector

    # 临时补丁：如果 main_detector 使用全局路径常量，我们需要先设好
    if db_path and hasattr(main_detector, 'DB_FILE'):
        import main_detector
        main_detector.DB_FILE = db_path
    if csv_path and hasattr(main_detector, 'CSV_FILE'):
        import main_detector
        main_detector.CSV_FILE = csv_path

    print("\n[Ablation] 复用主检测器管线加载完整特征矩阵 ...")
    detector = UltimateDeceptionDetector()
    df_features, y = detector.load_and_fuse_data()
    
    X = detector.X
    feature_names = detector.feature_names
    
    print(f"[Ablation] 特征矩阵维度: {X.shape}")
    print(f"[Ablation] 正负样本分布: 负={int(np.sum(y == 0))}, 正={int(np.sum(y == 1))}")
    
    return X, y, feature_names


def _find_emotional_feature_columns(feature_names: List[str]) -> Dict[str, List[int]]:
    """
    根据列名定位 8 维情感特征在整体特征矩阵中的实际列索引。
    由于主矩阵还包含语义 PCA 列和行为统计列，情感特征不一定从索引 0 开始。
    
    返回:
      {
        "EmpathyGap":  [idx_mean, idx_max],
        "DarkTriad":   [idx_mean, idx_max],
        "Contagion":   [idx_mean, idx_max],
        "Volatility":  [idx_mean, idx_max],
      }
    """
    module_patterns = {
        "EmpathyGap":  ["Empathy_Gap_Mean",  "Empathy_Gap_Max"],
        "DarkTriad":   ["Dark_Triad_Mean",   "Dark_Triad_Max"],
        "Contagion":   ["Contagion_Mean",    "Contagion_Max"],
        "Volatility":  ["Volatility_Mean",   "Volatility_Max"],
    }
    
    # 建立列名到索引的快速查找表
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    result = {}
    for module, patterns in module_patterns.items():
        indices = []
        for pattern in patterns:
            # 先精确匹配
            if pattern in name_to_idx:
                indices.append(name_to_idx[pattern])
            else:
                # 模糊匹配以防拼写差异
                matched = [idx for name, idx in name_to_idx.items() 
                           if pattern.lower().replace("_", "") in name.lower().replace("_", "")]
                if matched:
                    indices.extend(matched[:1])  # 只取第一个匹配
                else:
                    raise KeyError(f"无法在特征矩阵中找到列 '{pattern}'，"
                                   f"可用列: {list(feature_names[:10])}...")
        
        if len(indices) != 2:
            raise ValueError(f"模块 {module} 应恰好有 2 列，实际找到 {len(indices)}: {indices}")
        result[module] = sorted(indices)
    
    return result


def _apply_ablation(X: np.ndarray, module_indices: Dict[str, List[int]], 
                    zero_modules: List[str]) -> np.ndarray:
    """
    返回 X 的副本，将指定模块的列置零。
    zero_modules 是模块名列表，如 ["EmpathyGap", "DarkTriad"]。
    """
    X_ablated = X.copy()
    for mod_name in zero_modules:
        if mod_name not in module_indices:
            raise ValueError(f"未知模块: {mod_name}，可选: {list(module_indices.keys())}")
        indices = module_indices[mod_name]
        X_ablated[:, indices] = 0.0
    return X_ablated


def run_single_config(
    X: np.ndarray,
    y: np.ndarray,
    module_indices: Dict[str, List[int]],
    config_name: str,
    zero_modules: List[str],
    n_repeats: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    执行单个消融配置的重复交叉验证评估。
    
    参数:
        X              : 完整特征矩阵
        y              : 标签
        module_indices : 模块→列索引映射
        config_name    : 配置名称 (用于打印)
        zero_modules   : 需要置零的模块列表
        n_repeats      : RepeatedStratifiedKFold 重复次数
        n_splits       : K 折数
    
    返回:
        {
            "config": str,
            "accuracy":  float, "accuracy_std":  float,
            "auc":       float, "auc_std":       float,
            "f1":        float, "f1_std":        float,
            "precision": float, "precision_std": float,
            "recall":    float, "recall_std":    float,
        }
    """
    X_abl = _apply_ablation(X, module_indices, zero_modules)
    
    # 自适应树深
    n_samples = len(y)
    max_depth = 3 if n_samples <= 20 else 5
    scale_pos = float(np.sum(y == 0)) / max(np.sum(y == 1), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=max_depth,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        verbosity=0,
    )
    
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    
    # 存储每折的指标
    scores = {"accuracy": [], "auc": [], "f1": [], "precision": [], "recall": []}
    
    for train_idx, test_idx in cv.split(X_abl, y):
        X_train, X_test = X_abl[train_idx], X_abl[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(np.unique(y_train)) < 2:
            continue  # 跳过只有单类的折
        
        model_clone = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=max_depth,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", random_state=random_state,
            eval_metric="logloss", scale_pos_weight=scale_pos,
            use_label_encoder=False, verbosity=0,
        )
        model_clone.fit(X_train, y_train)
        
        y_pred = model_clone.predict(X_test)
        y_proba = model_clone.predict_proba(X_test)[:, 1]
        
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["auc"].append(roc_auc_score(y_test, y_proba))
        scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
    
    result = {
        "config": config_name,
    }
    for metric in ["accuracy", "auc", "f1", "precision", "recall"]:
        vals = scores[metric]
        if vals:
            result[metric] = round(float(np.mean(vals)), 4)
            result[f"{metric}_std"] = round(float(np.std(vals)), 4)
        else:
            result[metric] = 0.0
            result[f"{metric}_std"] = 0.0
    
    # 计算相对于 Full Model 的变化量
    result["_zero_modules"] = zero_modules
    
    print(f"  [{config_name:<25}] ACC={result['accuracy']:.4f}±{result['accuracy_std']:.4f}  "
          f"AUC={result['auc']:.4f}±{result['auc_std']:.4f}  "
          f"F1={result['f1']:.4f}±{result['f1_std']:.4f}")
    
    return result


def run_ablation(
    db_path: str = None,
    csv_path: str = None,
    n_repeats: int = 5,
    n_splits: int = 5,
    save_dir: str = "./ablation_results",
    skip_independence: bool = False,
    skip_ablation: bool = False,
) -> List[Dict]:
    """
    主入口：运行完整的情感模块消融实验。
    
    参数:
        db_path           : 数据库路径 (覆盖 main_detector 中的默认值)
        csv_path          : CSV 路径
        n_repeats         : CV 重复次数
        n_splits          : K 折数
        save_dir          : 输出目录
        skip_independence : 跳过独立性验证步骤
        skip_ablation     : 跳过消融实验步骤 (仅运行独立性验证)
    
    返回:
        ablation_results  : List[Dict]，每个配置的完整指标
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== 前置步骤：独立性验证 ==========
    if not skip_independence:
        print("\n" + "█" * 60)
        print("  前置分析：情感模块统计独立性验证")
        print("█" * 60)
        try:
            from ablation.independence_verify import verify_independence
            ind_result = verify_independence(
                db_path=db_path, csv_path=csv_path, save_dir=save_dir
            )
        except Exception as e:
            print(f"[WARNING] 独立性验证失败 (将跳过): {e}")
    
    if skip_ablation:
        print("\n[INFO] --skip-ablation 已设置，跳过消融实验。")
        return []
    
    # ========== 消融实验主流程 ==========
    print("\n" + "█" * 60)
    print("  情感模块消融实验")
    print("█" * 60)
    
    # 1. 加载完整特征矩阵
    X, y, feature_names = _load_full_feature_matrix(
        db_path=db_path, csv_path=csv_path
    )
    
    # 2. 定位情感特征列
    print("\n[Ablation] 定位情感特征列索引 ...")
    try:
        module_indices = _find_emotional_feature_columns(feature_names)
        for mod, idxs in module_indices.items():
            print(f"  {mod:<13} → [{idxs[0]}, {idxs[1]}]  "
                  f"({feature_names[idxs[0]]}, {feature_names[idxs[1]]})")
    except KeyError as e:
        print(f"[ERROR] 列名匹配失败: {e}")
        print(f" 可用列名 (前20): {list(feature_names[:20])}")
        print("  将使用默认索引映射 (假设情感列在最后 8 列) ...")
        total_cols = X.shape[1]
        if total_cols < 8:
            raise RuntimeError(f"特征列数 ({total_cols}) 不足 8，无法进行消融实验")
        # 假设情感列在最后 8 位
        offset = total_cols - 8
        module_indices = {
            "EmpathyGap":  [offset + 0, offset + 1],
            "DarkTriad":   [offset + 2, offset + 3],
            "Contagion":   [offset + 4, offset + 5],
            "Volatility":  [offset + 6, offset + 7],
        }
        print("   → 回退索引映射:")
        for mod, idxs in module_indices.items():
            print(f"     {mod}: {idxs}")
    
    # 3. 遍历所有消融配置
    print(f"\n[Ablation] 开始评估 {len(ABLATION_CONFIGS)} 种配置 "
          f"(Repeat={n_repeats}, Fold={n_splits}) ...")
    print("-" * 70)
    
    all_module_names = list(module_indices.keys())
    ablation_results = []
    
    for config_name, config_info in ABLATION_CONFIGS.items():
        if config_name == "Full Model":
            zero_modules = []
        elif config_name == "No Emotional Modules":
            zero_modules = all_module_names
        else:
            # 从配置名推断要移除的模块 (如 "− EmpathyGap" → ["EmpathyGap"])
            zero_modules = [
                mod for mod in all_module_names
                if mod.lower() in config_name.lower()
            ]
            if len(zero_modules) == 0:
                print(f"[WARNING] 无法从配置名 '{config_name}' 中解析模块，跳过")
                continue
        
        result = run_single_config(
            X=X, y=y, module_indices=module_indices,
            config_name=config_name, zero_modules=zero_modules,
            n_repeats=n_repeats, n_splits=n_splits
        )
        result["color"] = config_info["color"]
        result["description"] = config_info["description"]
        ablation_results.append(result)
    
    # 4. 计算性能退化 (相对于 Full Model)
    print("\n" + "=" * 70)
    print("  消融实验 — 性能退化汇总 (相对于 Full Model)")
    print("=" * 70)
    full_result = ablation_results[0] if ablation_results else None
    if full_result:
        print(f"\n{'配置':<25} {'Δ ACC':>8} {'Δ AUC':>8} {'Δ F1':>8}")
        print("-" * 55)
        for res in ablation_results[1:]:
            d_acc = res["accuracy"] - full_result["accuracy"]
            d_auc = res["auc"] - full_result["auc"]
            d_f1 = res["f1"] - full_result["f1"]
            print(f"{res['config']:<25} {d_acc:>+8.4f} {d_auc:>+8.4f} {d_f1:>+8.4f}")
    
    # 5. 保存原始结果
    import json
    json_path = os.path.join(save_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 消融结果已保存至: {json_path}")
    
    return ablation_results