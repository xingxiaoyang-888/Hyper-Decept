import os
import torch
import numpy as np
import warnings
import functools
import logging
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)

class EmotionVolatilityAnalyzer:
    """
    重型特征引擎 V7.1：功能性情感瞬切率 (Functional Emotion Volatility)
    
    [理论物理意义与经验阈值参考]：
    本引擎输出原始的 28 维欧氏距离。虽然理论极限为 sqrt(28) ≈ 5.29，
    但由于自然语言情感分布的稀疏性（单句通常只激活 1-2 个维度），实际观测值域大不相同：
    - [0.0 ~ 0.5]：正常的情感波动（如从“平静”到“微小喜悦”）。人类绝大部分序列在此区间。
    - [0.5 ~ 1.0]：显著的情感跨度。
    - [1.0 ~ 1.5]：极端的情感反转（相当于单一情绪从 0.9 突变到另一情绪的 0.9，距离约 1.27）。
    - [ > 1.5 ]：非人类的超高频瞬切。这通常是无状态 (Stateless) 的 API 机器人在并发处理完全不同语境的任务。
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmotionVolatilityAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reload(cls, **kwargs):
        logger.info("正在卸载瞬切率引擎并清空显存...")
        if cls._instance is not None:
            if hasattr(cls._instance, 'model') and cls._instance.model is not None:
                del cls._instance.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cls._instance = None
        return cls(**kwargs)
    
    def __init__(self, 
                 device=None, 
                 # 英文原版: SamLowe/roberta-base-go_emotions
                 # 多语言/中文推荐: MilaNLProc/xlm-roberta-base-goemotions
                 emotion_model_name="SamLowe/roberta-base-go_emotions"):
        
        if getattr(self, '_initialized', False):
            return
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"[Volatility Engine ] 启动，使用设备: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
        print(f"   -> 尝试加载 28 维情感向量空间模型 ({emotion_model_name})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(self.device)
            self.model.eval()
            self.emotion_model_name = emotion_model_name
            self.emotion_dim = int(getattr(self.model.config, "num_labels", 28))
        except Exception as e:
            logger.error(f"模型加载失败！系统进入 Dummy 安全模式: {e}")
            self.emotion_dim = 28

        self._initialized = True
        if self.model is not None:
            print(f" [Volatility Engine ] 成功搭载！\n")
        else:
            print(f" [Volatility Engine ] 以安全降级模式运行！\n")

    def _get_default_scores(self, insufficient: bool = True) -> dict:
        """
        统一返回格式。
        insufficient=True 明确告诉下游：不是波动率为0，而是无法计算。
        """
        return {
            "Agent_Mean_Volatility": 0.0,
            "Agent_Max_Volatility": 0.0,
            "Insufficient_Data": insufficient
        }

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_emotion_vectors_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if self.model is None or self.tokenizer is None or not texts:
            return np.zeros((len(texts), self.emotion_dim))

        unique_texts = list(dict.fromkeys(texts))
        vector_by_text = {}
        for batch_start in tqdm(
            range(0, len(unique_texts), batch_size),
            desc="Volatility emotion batches",
            unit="batch",
            leave=False,
        ):
            batch_texts = unique_texts[batch_start : batch_start + batch_size]
            try:
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                for text, vector in zip(batch_texts, probs):
                    vector_by_text[text] = vector
            except Exception as e:
                logger.warning(f"Batch 情感向量提取失败: {e}")
                for text in batch_texts:
                    vector_by_text[text] = np.zeros(self.emotion_dim)
                
        return np.array([vector_by_text.get(text, np.zeros(self.emotion_dim)) for text in texts])

    def evaluate_agents_batch(self, agents_texts: List[List[str]], batch_size: int = 64) -> List[Dict]:
        """
        批量计算多个 Agent 的功能性情感瞬切率。
        与 evaluate_agent 使用同一套清洗、过滤和欧氏距离公式，只把情感向量推理提升为全局批处理。
        """
        prepared_agents = []
        all_valid_texts = []

        for texts in tqdm(agents_texts, desc="Volatility prepare agents", unit="agent", leave=False):
            if not texts or len(texts) < 2:
                prepared_agents.append([])
                continue

            cleaned_texts = [self._clean_text(t) for t in texts]
            valid_texts = [t for t in cleaned_texts if len(t) >= 5]
            prepared_agents.append(valid_texts)
            if len(valid_texts) >= 2:
                all_valid_texts.extend(valid_texts)

        default_results = [self._get_default_scores(insufficient=True) for _ in agents_texts]
        if not all_valid_texts:
            return default_results

        unique_texts = list(dict.fromkeys(all_valid_texts))
        unique_vectors = self._get_emotion_vectors_batch(unique_texts, batch_size=batch_size)
        vector_by_text = {
            text: unique_vectors[idx]
            for idx, text in enumerate(unique_texts)
        }

        results = []
        zero_vec = np.zeros(self.emotion_dim)
        for valid_texts in tqdm(prepared_agents, desc="Volatility aggregate agents", unit="agent", leave=False):
            if len(valid_texts) < 2:
                results.append(self._get_default_scores(insufficient=True))
                continue

            emotion_matrix = np.array([vector_by_text.get(text, zero_vec) for text in valid_texts])
            diff_matrix = emotion_matrix[1:] - emotion_matrix[:-1]
            euclidean_distances = np.linalg.norm(diff_matrix, axis=1)

            results.append({
                "Agent_Mean_Volatility": round(float(np.mean(euclidean_distances)), 4),
                "Agent_Max_Volatility": round(float(np.max(euclidean_distances)), 4),
                "Insufficient_Data": False
            })

        return results

    def evaluate_agent(self, texts: List[str], timestamps: List[float] = None) -> Dict:
        """
        Agent 维度宏观聚合接口：计算情感序列瞬切率
        """
        if not texts or len(texts) < 2:
            return self._get_default_scores(insufficient=True)
            
        # ---------------------------------------------------------
        # 时序对齐保护
        # ---------------------------------------------------------
        if timestamps is not None:
            if len(timestamps) != len(texts):
                logger.warning("timestamps 长度与 texts 不一致！将忽略时间戳并假设文本已按时间升序。")
            else:
                # 按照时间戳升序重排推文
                sorted_pairs = sorted(zip(timestamps, texts), key=lambda x: x[0])
                texts = [p[1] for p in sorted_pairs]
        
        # 1. 清洗并过滤过短的噪声文本
        cleaned_texts = [self._clean_text(t) for t in texts]
        valid_indices = [i for i, t in enumerate(cleaned_texts) if len(t) >= 5]
        
        if len(valid_indices) < 2:
            return self._get_default_scores(insufficient=True)
            
        valid_texts = [cleaned_texts[i] for i in valid_indices]
        
        # 2. 提取 28 维情感概率向量
        emotion_matrix = self._get_emotion_vectors_batch(valid_texts)
        
        # 3. 欧氏距离计算 (Euclidean Distance)
        diff_matrix = emotion_matrix[1:] - emotion_matrix[:-1]
        
        # 保留原始欧氏距离，不进行破坏性的极值归一化
        euclidean_distances = np.linalg.norm(diff_matrix, axis=1)
        
        mean_volatility = float(np.mean(euclidean_distances))
        max_volatility = float(np.max(euclidean_distances))
        
        # 正常计算完毕，Insufficient_Data 置为 False
        return {
            "Agent_Mean_Volatility": round(mean_volatility, 4),
            "Agent_Max_Volatility": round(max_volatility, 4),
            "Insufficient_Data": False
        }
