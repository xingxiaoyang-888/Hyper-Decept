"""
main_detector.py (终极审判庭)

AI 多智能体欺骗博弈检测系统 - 核心主控程序
- 数据融合：整合【大模型深度心理特征】与【异构图双曲战术角色】。
- 终极分类：基于 XGBoost 的高精度人机对抗分类器 (Bot vs. Human)。
- 可解释性：使用 SHAP (SHapley Additive exPlanations) 解构水军的伪装逻辑。
"""

import os
import argparse
import warnings
import shutil
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging

from detection_module.feature_extractor import MultimodalExtractor
from detection_module.visualizer import CognitiveVisualizer
from detection_module.db_adapter import (
    build_summary_texts,
    load_label_frame,
)
from detection_module.hyper_newtest.config import (
    DATASET_CHOICES,
    PROJECT_ROOT,
    configure_utf8_streams,
    make_experiment_dir,
    resolve_dataset_paths,
    write_manifest,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

configure_utf8_streams()

# ==========================================
# 全局配置
# ==========================================
DB_FILE, CSV_FILE = resolve_dataset_paths()
ROLE_CSV = os.getenv(
    "AFG_ROLE_CSV",
    str(PROJECT_ROOT / "data" / "hyperrole_results" / "hetero_hyperrole_assignments.csv"),
)
SAVE_DIR = os.getenv("AFG_SAVE_DIR", str(PROJECT_ROOT / "new_result"))

os.makedirs(SAVE_DIR, exist_ok=True)

class UltimateDeceptionDetector:
    def __init__(
        self,
        db_file=DB_FILE,
        csv_file=CSV_FILE,
        role_csv=ROLE_CSV,
        save_dir=SAVE_DIR,
        psychology_mode="full",
        max_tweets_per_user=None,
    ):
        logger.info("[终极审判庭] 启动...")
        self.db_file = db_file
        self.csv_file = csv_file
        self.role_csv = role_csv
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.extractor = MultimodalExtractor(
            psychology_mode=psychology_mode,
            max_tweets_per_user=max_tweets_per_user,
            cache_dir=os.path.join(self.save_dir, "feature_cache"),
        )
        
    def load_and_fuse_data(self):
        logger.info("开始融合【心理侧写特征】与【拓扑战术特征】...")
        
        # 1. 获取基础标签和文本
        df_labels = load_label_frame(self.csv_file, self.db_file)
        df_labels['user_id'] = df_labels['user_id'].astype(str)
        summary_texts = build_summary_texts(df_labels)
        user_ids_master = df_labels['user_id'].tolist()
            
        # 2. 调用战区一：获取心理学特征矩阵 (8维心理 + 行为 + 语义PCA)
        user_ids, psycho_matrix = self.extractor.fuse_multimodal_features(
            self.db_file,
            summary_texts,
            user_ids_master=user_ids_master,
        )
        
        # 定义战区一的特征列名 (需要与 feature_extractor 对齐)
        behavior_names = [
            'Follower_Following_Ratio', 'Action_Frequency', 'Like_Ratio', 'Retweet_Ratio', 'Reply_Ratio', # 假设的5维行为
            'Temporal_Entropy', 'URL_Ratio', 'Mention_Ratio', 'Hashtag_Ratio', 'Media_Ratio',             # 假设的5维行为
        ]
        psycho_names = [
            'Empathy_Gap_Mean', 'Empathy_Gap_Max', 
            'Dark_Triad_Mean', 'Dark_Triad_Max',
            'Contagion_Mean', 'Contagion_Max', 
            'Volatility_Mean', 'Volatility_Max'
        ]
        semantic_dim = psycho_matrix.shape[1] - len(behavior_names) - len(psycho_names)
        if semantic_dim < 0:
            raise ValueError(f"特征矩阵维度异常: {psycho_matrix.shape}")
        feature_names = [f"Semantic_PCA_{i}" for i in range(semantic_dim)]
        feature_names += behavior_names + psycho_names
        
        df_psycho = pd.DataFrame(psycho_matrix, columns=feature_names)
        df_psycho['user_id'] = pd.Series(user_ids).astype(str).values
        
        # 3. 调用战区二：获取异构图战术角色 (如果文件不存在，抛出异常提示先跑战区二)
        if not os.path.exists(self.role_csv):
            raise FileNotFoundError(f"未找到拓扑特征文件！请先运行 detection_module/hetero_hyperrole_classifier.py 生成 {self.role_csv}")
            
        df_roles = pd.read_csv(self.role_csv)
        logger.info(f"[DEBUG] HyperRole CSV 列名: {df_roles.columns.tolist()}")  # 验证 Tactical_Role → role 修复
        role_col = 'role' if 'role' in df_roles.columns else 'Tactical_Role'
        if role_col not in df_roles.columns:
            raise ValueError("拓扑特征文件中缺少 role 或 Tactical_Role 列。")

        # 将 user_id 转为字符串以保证 Merge 成功
        df_roles['user_id'] = df_roles['user_id'].astype(str)
        df_psycho['user_id'] = df_psycho['user_id'].astype(str)
        
        # 独热编码 (One-Hot Encoding) 战术角色
        df_roles_encoded = pd.get_dummies(df_roles[['user_id', role_col]], columns=[role_col])
        df_roles_encoded['poincare_radius'] = df_roles['poincare_radius']
        
        # 4. 终极矩阵 Join
        df_final = pd.merge(df_psycho, df_roles_encoded, on='user_id', how='inner')
        df_final = pd.merge(df_final, df_labels[['user_id', 'is_bad']], on='user_id', how='inner')
        if df_final.empty:
            raise ValueError("融合后的样本数为 0，请检查 CSV、数据库与 HyperRole 结果中的 user_id 是否一致。")

        # 保存角色标签用于可视化
        self.y_role = df_roles.set_index('user_id').loc[df_final['user_id'], role_col].values
        
        self.y = df_final['is_bad'].values
        self.X_df = df_final.drop(columns=['user_id', 'is_bad']).apply(pd.to_numeric, errors='coerce').fillna(0.0)
        self.feature_names = self.X_df.columns.tolist()
        self.X = self.X_df.to_numpy(dtype=float)
        
        logger.info(f"终极超级矩阵构建完毕！维度: {self.X.shape[0]} 样本 x {self.X.shape[1]} 特征")
        logger.info(f"类别分布: {sum(self.y == 0)} 良民 (Human), {sum(self.y == 1)} 水军 (Bot)")
        logger.info(f"[DEBUG] 全部特征名 ({len(self.feature_names)}): {self.feature_names}")
        logger.info(f"[DEBUG] 角色标签分布: {pd.Series(self.y_role).value_counts().to_dict()}")

        return self.X_df, self.y

    def train_and_evaluate(self):
        """
        核心战役二：XGBoost 降维打击与交叉验证
        """
        logger.info("启动 XGBoost 终极分类器 (Stratified 5-Fold CV)...")

        # 带 SMOTE 的 Pipeline（SMOTE 在 CV 折内执行，无数据泄露）
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('xgb', xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', random_state=42, eval_metric='logloss'
            ))
        ])

        # 交叉验证（对原始数据做 CV，Pipeline 内部自动 SMOTE 每折）
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipeline, self.X, self.y, cv=cv)
        y_pred_proba = cross_val_predict(pipeline, self.X, self.y, cv=cv, method='predict_proba')[:, 1]

        # 打印军报（用原始标签，不是 SMOTE 后的）
        logger.info("\n" + "="*40 + "\n终极分类测试报告 (Cross-Validation, 无数据泄露)\n" + "="*40)
        print(classification_report(self.y, y_pred, target_names=["Human (Good)", "Bot (Bad)"]))
        auc_score = roc_auc_score(self.y, y_pred_proba)
        logger.info(f"ROC-AUC Score (无泄漏): {auc_score:.4f}")

        # 在全量平衡数据上拟合最终模型，用于 SHAP
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X, self.y)
        self.model = pipeline.named_steps['xgb']
        self.model.fit(X_resampled, y_resampled)
        self.X_resampled_df = pd.DataFrame(X_resampled, columns=self.feature_names).apply(
            pd.to_numeric,
            errors='coerce',
        ).fillna(0.0)

        self._plot_confusion_matrix(self.y, y_pred)
        
    def explain_with_shap(self):
        """
        核心战役三：SHAP 白盒化解释引擎 (揭示虚假博弈的运作机制)
        """
        logger.info("启动 SHAP 归因分析引擎 (Explainable AI)...")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_resampled_df)
        
        # 1. SHAP Summary Plot (特征重要性总览)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, self.X_resampled_df, plot_type="dot", max_display=20, show=False)
        plt.title('SHAP Feature Attribution: How Bots are Detected\n(Top 20 Most Decisive Features)', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SHAP Bar Plot (宏观重要性排名)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_resampled_df, plot_type="bar", max_display=15, show=False)
        plt.title('Global Feature Importance (Mean Absolute SHAP Value)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可解释性报告已生成！请查看 {self.save_dir} 目录。")
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'], annot_kws={"size": 16})
        plt.title('Bot Detection Confusion Matrix', fontsize=15, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="AI 多智能体欺骗博弈检测系统")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--db", dest="db_file", default=None)
    parser.add_argument("--csv", dest="csv_file", default=None)
    parser.add_argument("--role-csv", dest="role_csv", default=ROLE_CSV)
    parser.add_argument("--save-dir", dest="save_dir", default=SAVE_DIR)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument(
        "--psychology-mode",
        choices=["full", "fast", "off"],
        default=os.getenv("AFG_PSYCHOLOGY_MODE", "fast"),
        help="full=完整四引擎；fast=跳过最慢的 GPT-2 困惑度和 DarkTriad NLI；off=心理指标置零。",
    )
    parser.add_argument(
        "--max-tweets-per-user",
        type=int,
        default=None,
        help="每个用户最多用于心理扫描的文本数；默认不截断。",
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    db_file, csv_file = resolve_dataset_paths(args.db_file, args.csv_file, args.dataset)
    run_save_dir = make_experiment_dir(
        args.save_dir,
        db_path=db_file,
        csv_path=csv_file,
        dataset=args.dataset,
        run_name=args.run_name,
        prefix="detector",
    )
    write_manifest(
        run_save_dir,
        script="new_main_detector.py",
        db_path=db_file,
        csv_path=csv_file,
        role_csv=args.role_csv,
        dataset=args.dataset or "auto",
        psychology_mode=args.psychology_mode,
        max_tweets_per_user=args.max_tweets_per_user,
    )
    logger.info(f"当前数据库: {db_file}")
    logger.info(f"当前标签/文本 CSV: {csv_file}")
    logger.info(f"当前实验输出目录: {run_save_dir}")

    detector = UltimateDeceptionDetector(
        db_file=db_file,
        csv_file=csv_file,
        role_csv=args.role_csv,
        save_dir=run_save_dir,
        psychology_mode=args.psychology_mode,
        max_tweets_per_user=args.max_tweets_per_user,
    )

    try:
        # 1. 熔断矩阵
        X_df, y_true = detector.load_and_fuse_data()
        if os.path.exists(detector.role_csv):
            shutil.copyfile(detector.role_csv, os.path.join(detector.save_dir, "role_assignments_used.csv"))
        # 2. 训练与评估
        detector.train_and_evaluate()
        # 3. 提取解释
        detector.explain_with_shap()

        # 4. 可视化心理特征
        logger.info("启动 CognitiveVisualizer 可视化引擎...")
        visualizer = CognitiveVisualizer(
            detector.X, detector.y, detector.y_role,
            detector.feature_names, save_dir=detector.save_dir
        )
        visualizer.generate_all_reports(trained_xgb_model=detector.model)

        print("\n系统运行圆满结束！所有防线均已打通，顶会级图表已输出。")
    except Exception as e:
        logger.exception(f"运行失败: {e}")
        if isinstance(e, FileNotFoundError) and "拓扑特征文件" in str(e):
            logger.error("提示：请确保你已经先运行了 detection_module/hetero_hyperrole_classifier.py！")
