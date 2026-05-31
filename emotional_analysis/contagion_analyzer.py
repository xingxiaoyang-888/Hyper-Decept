import os
import torch
import numpy as np
import warnings
import functools
import logging
import math
import re
from typing import List, Dict
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)

class ContagionAnalyzer:
    """
    无阻尼靶向情感传染率 (Frictionless Emotional Contagion)
    
    使用高维稠密向量 (Dense Embeddings) 计算推文与“水军既定靶标”的语义逼近度。
    单例防 OOM、SentenceTransformer C++ 底层 C-API 批处理加速、LRU 缓存、异常降级。
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ContagionAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reload(cls, **kwargs):
        logger.info("正在卸载传染率引擎并清空缓存与显存...")
        if cls._instance is not None:
            if hasattr(cls._instance.analyze_agent_text, 'cache_clear'):
                cls._instance.analyze_agent_text.cache_clear()
            if hasattr(cls._instance, 'embedder') and cls._instance.embedder is not None:
                del cls._instance.embedder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cls._instance = None
        return cls(**kwargs)
    
    def __init__(self, 
                 device=None, 
                 model_name="all-mpnet-base-v2", 
                 payload_anchors: List[str] = None): 
        
        if getattr(self, '_initialized', False):
            return
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"[Contagion Engine ] 启动，使用设备: {self.device}")
        
        # ---------------------------------------------------------
        # 定义靶向操纵锚点 
        # 这里预设了“贩卖焦虑/煽动对立”的典型水军 Leader 话术。
        # 在实际部署时，可根据具体的虚假博弈场景传入不同的 Anchors。
        # ---------------------------------------------------------
        if payload_anchors is None:
            self.payload_anchors = [
                "I lost my job and feel completely hopeless about the future.",
                "The economy is ruined, and the system is rigged against ordinary people.",
                "Everything is getting worse, and there is no way to survive this crisis."
            ]
        else:
            self.payload_anchors = payload_anchors

        # 模型降级机制
        self.embedder = None
        print(f"   -> 尝试加载主语义计算模型 ({model_name})...")

        def load_sentence_embedder(candidate_name):
            from sentence_transformers import SentenceTransformer, util
            try:
                return SentenceTransformer(
                    candidate_name,
                    device=str(self.device),
                    local_files_only=True,
                ), util
            except Exception as local_exc:
                logger.warning(f"{candidate_name} 本地缓存加载失败，尝试默认加载: {local_exc}")
                return SentenceTransformer(candidate_name, device=str(self.device)), util

        try:
            self.embedder, self.util = load_sentence_embedder(model_name)
            self.model_name = model_name
        except Exception as e:
            fallback_model = "all-MiniLM-L6-v2"
            logger.warning(f"主模型加载失败: {e}。尝试加载轻量极速模型 {fallback_model}...")
            try:
                self.embedder, self.util = load_sentence_embedder(fallback_model)
                self.model_name = fallback_model
            except Exception as e2:
                logger.error(f"所有 Embedding 模型加载失败！系统进入 Dummy 安全模式: {e2}")

        # 预先计算并冻结 Anchors 的张量 
        if self.embedder is not None:
            self.anchor_embeddings = self.embedder.encode(self.payload_anchors, convert_to_tensor=True, show_progress_bar=False)
        else:
            self.anchor_embeddings = None

        self._initialized = True
        if self.embedder is not None:
            print(f" [Contagion Engine ] 成功搭载 {self.model_name}！\n")
        else:
            print(f" [Contagion Engine ] 以安全降级模式运行 (全 0 输出)！\n")

    def _get_default_scores(self) -> dict:
        return {
            "Max_Payload_Alignment": 0.0,
            "Mean_Payload_Alignment": 0.0,
            "Frictionless_Contagion_Score": 0.0
        }

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @functools.lru_cache(maxsize=10000)
    def analyze_agent_text(self, text: str, verbose: bool = False) -> dict:
        """
        单条推文靶向对齐分析 (支持 LRU 缓存)
        """
        text = self._clean_text(text)
        
        if not text or len(text) < 5 or self.embedder is None:
            return self._get_default_scores()

        try:
            # 编码单条文本
            query_emb = self.embedder.encode(text, convert_to_tensor=True, show_progress_bar=False)
            
            # 计算与所有预设 Anchors 的余弦相似度
            cos_scores = self.util.cos_sim(query_emb, self.anchor_embeddings)[0].cpu().numpy()
            
            # 过滤负相关（负数相似度对传染率无意义，归零）
            cos_scores = np.maximum(cos_scores, 0.0)
            
            max_align = float(np.max(cos_scores))
            mean_align = float(np.mean(cos_scores))
            
        except Exception as e:
            logger.warning(f"单条余弦相似度计算失败: {e}")
            return self._get_default_scores()

        # 对于单条静态文本，无阻尼得分默认等于最大对齐度
        result = {
            "Max_Payload_Alignment": round(max_align, 4),
            "Mean_Payload_Alignment": round(mean_align, 4),
            "Frictionless_Contagion_Score": round(max_align, 4) 
        }
        
        if verbose:
            result["_Cleaned_Text"] = text
            
        return result

    def analyze_batch(self, texts: List[str], verbose: bool = False, batch_size: int = 64) -> List[Dict]:
        """
        调用 SentenceTransformer 的底层 C++ 批处理引擎，
        一次性将整个矩阵丢入 GPU，完成数千条推文的余弦张量计算。
        """
        results = []
        if not texts:
            return results

        if self.embedder is None:
            return [self._get_default_scores() for _ in texts]

        cleaned_texts = [self._clean_text(t) for t in texts]
        
        # 提取有效文本及其原始索引，避免无意义计算
        valid_pairs = [(i, t) for i, t in enumerate(cleaned_texts) if len(t) >= 5]
        
        # 构建默认结果集
        final_results = [self._get_default_scores() for _ in texts]
        
        if not valid_pairs:
            return final_results

        unique_texts = list(dict.fromkeys(t for _, t in valid_pairs))

        try:
            # 高效批处理编码：内部实现了智能分块和 padding
            query_embs = self.embedder.encode(
                unique_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            
            # 矩阵乘法：计算 (Num_Texts, Embedding_Dim) x (Embedding_Dim, Num_Anchors)
            cos_scores_matrix = self.util.cos_sim(query_embs, self.anchor_embeddings).cpu().numpy()
            cos_scores_matrix = np.maximum(cos_scores_matrix, 0.0)

            result_by_text = {}
            for ptr, text in enumerate(tqdm(unique_texts, desc="Contagion scores", leave=False)):
                scores = cos_scores_matrix[ptr]
                max_align = float(np.max(scores))
                mean_align = float(np.mean(scores))
                
                res = {
                    "Max_Payload_Alignment": round(max_align, 4),
                    "Mean_Payload_Alignment": round(mean_align, 4),
                    "Frictionless_Contagion_Score": round(max_align, 4)
                }
                if verbose:
                    res["_Cleaned_Text"] = text
                result_by_text[text] = res

            for orig_idx, text in valid_pairs:
                final_results[orig_idx] = result_by_text[text]
                
        except Exception as e:
            logger.warning(f"Batch 张量相似度计算崩溃: {e}")

        return final_results

    def evaluate_agent(self, texts: List[str], response_delays: List[float] = None) -> Dict:
        """
        Agent 维度宏观聚合接口：真正的“无阻尼”计算
        
        参数:
        - texts: 该智能体的推文列表
        - response_delays: (可选) 该推文回复上一条推文的时间延迟（秒）。
          人类通常需要 60秒-几分钟来构思回复。如果水军大批量在 2秒内完美回复，
          那么无阻尼系数将爆表。
        """
        if not texts:
            return {"Agent_Mean_Alignment": 0.0, "Agent_Contagion_Spike": 0.0, "Agent_Frictionless_Index": 0.0}
            
        batch_results = self.analyze_batch(texts, batch_size=64)
        alignments = [res.get("Max_Payload_Alignment", 0.0) for res in batch_results]
        
        if not alignments:
            return {"Agent_Mean_Alignment": 0.0, "Agent_Contagion_Spike": 0.0, "Agent_Frictionless_Index": 0.0}
            
        mean_align = np.mean(alignments)
        max_align = np.max(alignments)
        
        # ---------------------------------------------------------
        # 时间阻尼惩罚公式 
        # 如果提供了回复延迟时间，计算真实的无阻尼传染指数。
        # ---------------------------------------------------------
        if response_delays is not None and len(response_delays) == len(alignments):
            frictionless_scores = []
            for align, delay in zip(alignments, response_delays):
                if delay <= 0:
                    delay = 1.0 # 避免除零
                # 公式：对齐度 * e^(-(延迟秒数 - 机器基准延迟5秒) / 60)
                # 延迟越小 (接近0)，衰减越小，得分越高。延迟超过 60 秒，得分大幅衰减。
                time_penalty = math.exp(-max(0, delay - 5.0) / 60.0)
                frictionless_scores.append(align * time_penalty)
            
            frictionless_index = np.max(frictionless_scores)
        else:
            # 缺乏时间维度时，退化为纯语义对齐度
            frictionless_index = max_align
        
        return {
            "Agent_Mean_Alignment": round(float(mean_align), 4),
            "Agent_Contagion_Spike": round(float(max_align), 4),
            "Agent_Frictionless_Index": round(float(frictionless_index), 4)
        }
