"""
main_detector.py (终极审判庭 - 全量修复整合版)

AI 多智能体欺骗博弈检测系统 - 核心主控程序
- 【致命 Bug 修复】：引入 ast.literal_eval 动态解包，彻底解决 previous_tweets 无法切分导致情感指标归零的封印。
- 【弹性主键总线】：完美兼容 20agent 仿真库 (name 字段) 与 TwiBot 基准库 (user_id 字段) 的跨表级对齐。
- 【自适应评估】：根据样本规模（如 N<=20）自动在 留一法(LOOCV) 与 Stratified 5-Fold 之间平滑切分，无缝防爆。
- 【白盒归因雷达】：高度集成 SHAP 可解释性模型，输出学术论文专用的特征贡献大图。
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 平滑容错处理高级依赖库
try:
    import shap
except ImportError:
    shap = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 核心全局路径配置中心
# ==========================================
# 当切换数据集时，只需在此处更换对应的文件路径即可
DB_FILE = r"D:\Github\Multi-AFG-Detection\data\twibot_1000_v5.db"
CSV_FILE = r"D:\Github\Multi-AFG-Detection\data\twibot_1000_multimodal_v5.csv"
ROLE_CSV = r"D:\Github\Multi-AFG-Detection\data\hyperrole_results\hetero_hyperrole_assignments.csv"
SAVE_DIR = r"D:\Github\Multi-AFG-Detection\results"

os.makedirs(SAVE_DIR, exist_ok=True)

class UltimateDeceptionDetector:
    def __init__(self):
        logger.info("⚖️ [终极审判庭] 正在初始化多维博弈对抗主控台...")
        
        # 智能动态识别特征总线组件
        try:
            from detection_module.hyper_newtest.new_20agent_engine_test import Agent20Extractor
            self.extractor = Agent20Extractor()
            logger.info("   -> 成功挂载 20agent 仿真沙盒专属提取引擎")
        except Exception:
            from past_feature_extractor import MultimodalExtractor
            self.extractor = MultimodalExtractor()
            logger.info("   -> 成功挂载 TwiBot 基准大数据通用提取引擎")
        
    def load_and_fuse_data(self):
        """
        核心战役一：跨维度多模态特征安全融合总线 (Safe Data Fusion Bus)
        """
        logger.info("🔗 启动数据熔断总线，执行格式修正与跨表对齐...")
        
        # 1. 载入原始文本与标签数据
        df_labels = pd.read_csv(CSV_FILE)
        df_labels['is_bad'] = df_labels['user_type'].apply(lambda x: 1 if 'bad' in str(x).lower() else 0)
        
        # 【对齐保障】：智能判定主键命名空间
        if 'name' in df_labels.columns:
            df_labels['global_node_id'] = df_labels['name'].astype(str).str.strip()
            logger.info("   [Key Alignment] 仿真环境模式：锁定 'name' 字段 (User_X) 作为融合主键")
        else:
            df_labels['global_node_id'] = df_labels['user_id'].astype(str).str.strip()
            logger.info("   [Key Alignment] 基准环境模式：锁定 'user_id' 字段 (uXXXXX) 作为融合主键")

        # 2. 核心清洗：打破分隔符坍塌封印，重组推文文本流
        summary_texts = []
        for _, row in df_labels.iterrows():
            bio = str(row.get('user_char', row.get('description', '')))
            tweets_raw = str(row.get('previous_tweets', '[]')).strip()
            
            # 【致命 Bug 绝杀代码】：检查并利用 ast 将字符串化的 Python 列表还原为真实列表
            if tweets_raw.startswith('[') and tweets_raw.endswith(']'):
                try:
                    tweet_list = ast.literal_eval(tweets_raw)
                    if isinstance(tweet_list, list):
                        # 用底层的标准分隔符 " | " 进行拼装，确保下游分析时 split(" | ") 能够切分多条推文
                        tweets_formatted = " | ".join([str(t).strip() for t in tweet_list])
                    else:
                        tweets_formatted = tweets_raw
                except (ValueError, SyntaxError):
                    tweets_formatted = tweets_raw
            else:
                tweets_formatted = tweets_raw
                
            summary_texts.append(f"Bio: {bio}. Recent actions: {tweets_formatted}")
            
        # 3. 驱动战区一：获取复合心理学特征矩阵
        user_ids_master = df_labels['user_id'].tolist()
        _, psycho_matrix = self.extractor.fuse_multimodal_features(DB_FILE, summary_texts, user_ids_master=user_ids_master)
        
        # 【动态降维自适应】：动态解析 PCA 语义列名，拒绝死板切片
        total_dim = psycho_matrix.shape[1]
        psycho_core_names = [
            'Empathy_Gap_Mean', 'Empathy_Gap_Max', 
            'Dark_Triad_Mean', 'Dark_Triad_Max',
            'Contagion_Mean', 'Contagion_Max', 
            'Volatility_Mean', 'Volatility_Max'
        ]
        behavior_dim = 10 if total_dim > 138 else 3 # 弹性兼容小样本精简特征与大样本稠密特征
        pca_dim = total_dim - len(psycho_core_names) - behavior_dim
        
        feature_names = [f"Semantic_PCA_{i}" for i in range(pca_dim)]
        feature_names += [f"Behavior_Stat_{i}" for i in range(behavior_dim)]
        feature_names += psycho_core_names
        
        df_psycho = pd.DataFrame(psycho_matrix, columns=feature_names)
        df_psycho['global_node_id'] = df_labels['global_node_id'].values 
        
        # 4. 融合战区二：载入双曲几何拓扑特征报告
        if not os.path.exists(ROLE_CSV):
            raise FileNotFoundError(f"❌ 未找到拓扑雷达报告！请先跑检测模块生成: {ROLE_CSV}")
            
        df_roles = pd.read_csv(ROLE_CSV)
        df_roles['global_node_id'] = df_roles['user_id'].astype(str).str.strip()
        
        # 将结构角色转为 One-Hot 稠密矩阵，保留连续型特征 Poincare 半径
        df_roles_encoded = pd.get_dummies(df_roles[['global_node_id', 'Tactical_Role']], columns=['Tactical_Role'])
        df_roles_encoded['poincare_radius'] = df_roles['poincare_radius']
        
        # 5. 执行跨维度终极 Inner Join
        df_final = pd.merge(df_psycho, df_roles_encoded, on='global_node_id', how='inner')
        df_final = pd.merge(df_final, df_labels[['global_node_id', 'is_bad']], on='global_node_id', how='inner')
        
        if df_final.empty:
            raise ValueError("❌ [数据融断失败] 熔断后超级矩阵行数为0！请检查 CSV 中的 name/user_id 是否与拓扑结果完全一致。")
                             
        self.y = df_final['is_bad'].values
        self.X_df = df_final.drop(columns=['global_node_id', 'is_bad'])
        self.feature_names = self.X_df.columns.tolist()
        self.X = self.X_df.values
        
        logger.info(f"✅ 多模态矩阵交织完成！最终维度: {self.X.shape[0]} 样本 x {self.X.shape[1]} 特征")
        logger.info(f"   当前阵营权重分布: 人类(Human)={sum(self.y == 0)}, 机器人(Bot)={sum(self.y == 1)}")
        
        return self.X_df, self.y

    def train_and_evaluate(self):
        """
        核心战役二：基于自适应分流的 XGBoost 阵营对抗判决
        """
        n_samples = len(self.y)
        bad_count = sum(self.y)
        good_count = n_samples - bad_count
        
        if bad_count == 0 or good_count == 0:
            raise ValueError("数据集中仅包含单一种类标签，拒绝执行有监督二分类。")

        # 参数平衡控制：针对不同规模调整树深，引入 scale_pos_weight 代替小样本下易爆的 SMOTE
        max_depth = 3 if n_samples <= 20 else 5
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=max_depth,
            subsample=0.85,
            colsample_bytree=0.85,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=float(good_count) / bad_count
        )
        
        # 【工程防御闸】：自适应选择评估路径
        if n_samples <= 20:
            logger.info("🚨 [进入小样本保底机制] 触发留一法 (LOOCV) 分片验证策略...")
            loo = LeaveOneOut()
            y_eval_pred = np.zeros(n_samples)
            y_eval_proba = np.zeros(n_samples)
            
            for train_idx, test_idx in loo.split(self.X):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                if len(np.unique(y_train)) < 2:
                    y_eval_pred[test_idx] = y_train[0]
                    y_eval_proba[test_idx] = float(y_train[0])
                else:
                    clone_model = xgb.clone(self.model)
                    clone_model.fit(X_train, y_train)
                    y_eval_pred[test_idx] = clone_model.predict(X_test)[0]
                    y_eval_proba[test_idx] = clone_model.predict_proba(X_test)[0, 1]
            y_true_final = self.y
        else:
            logger.info("🚀 [进入标准学术机制] 触发 5-Fold 交叉验证 + SMOTE 均衡策略...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            X_res, y_res = self.X, self.y
            if SMOTE is not None:
                X_res, y_res = SMOTE(random_state=42).fit_resample(self.X, self.y)
                
            y_eval_pred = cross_val_predict(self.model, X_res, y_res, cv=cv)
            y_eval_proba = cross_val_predict(self.model, X_res, y_res, cv=cv, method='predict_proba')[:, 1]
            y_true_final = y_res
            
        # 汇报战果
        print("\n" + "⚔️" * 25 + "\n🔥 终极博弈人机分类报告\n" + "⚔️" * 25)
        print(classification_report(y_true_final, y_eval_pred, target_names=["Human (Good)", "Bot (Bad)"]))
        
        try:
            auc = roc_auc_score(y_true_final, y_eval_proba)
            logger.info(f"📊 系统综合辨识曲线面积 ROC-AUC: {auc:.4f}")
        except Exception:
            pass
            
        # 用全量数据拟合全局最终大脑，以备 SHAP 提取白盒规则
        self.model.fit(self.X, self.y)
        self._plot_confusion_matrix(y_true_final, y_eval_pred)

    def explain_with_shap(self):
        """
        核心战役三：SHAP 白盒化解释引擎 (心理特征与拓扑特征的交叉印证)
        """
        if shap is None:
            logger.warning("⚠️ 环境中缺乏 shap 库，跳过白盒归因。可执行 pip install shap 安装。")
            return
            
        logger.info("🔍 启动 SHAP 解释树，解构欺骗智能体的内部运作机制...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)
        
        # 1. 特征重要性点状分布图 (Summary Plot)
        plt.figure(figsize=(12, 9))
        shap.summary_plot(shap_values, self.X_df, plot_type="dot", max_display=15, show=False)
        plt.title('SHAP Attribution: Decision Boundaries for Cognitive Bot Detection\n(Top 15 Cross-Dimensional Features)', 
                  fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 审计报告渲染完毕！学术级别图表已存放于: {SAVE_DIR}")
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'], annot_kws={"size": 14})
        plt.title('Bot Detection Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    print("\n" + "★"*60)
    print(" ⚖️  AI 多智能体欺骗博弈检测系统 - 终极审判庭 (无错整合版)")
    print("★"*60 + "\n")
    
    detector = UltimateDeceptionDetector()
    
    try:
        # 1. 装载并跨维度熔断多模态特征
        detector.load_and_fuse_data()
        # 2. 自适应有监督分类测试
        detector.train_and_evaluate()
        # 3. 提取 SHAP 归因雷达
        detector.explain_with_shap()
        
        print("\n🎉 [大获全胜] 审判庭全链路测试通车！论文核心图表已全量交付。")
    except Exception as e:
        logger.error(f"❌ 审判庭流水线中断: {e}")
        logger.error("提示：若抛出文件找不到异常，请确保你已先行启动拓扑网络雷达 (hetero_hyperrole_classifier.py)！")