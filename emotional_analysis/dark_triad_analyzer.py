import os
import torch
import numpy as np
import warnings
import functools
import logging
import re
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)

class DarkTriadAnalyzer:
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DarkTriadAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reload(cls, **kwargs):
        """支持配置热重载，彻底释放旧模型与 LRU 缓存"""
        logger.info("正在卸载暗黑三角引擎并清空缓存与显存...")
        if cls._instance is not None:
            # 显式清空 LRU 缓存，释放绑定在旧实例上的内存引用
            if hasattr(cls._instance.analyze_agent_text, 'cache_clear'):
                cls._instance.analyze_agent_text.cache_clear()
                
            if hasattr(cls._instance, 'model') and cls._instance.model is not None:
                del cls._instance.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cls._instance = None
        return cls(**kwargs)
    
    def __init__(self, 
                 device=None, 
                 nli_model_name="facebook/bart-large-mnli", 
                 w_mach=0.70,   
                 w_narc=0.15,   
                 w_psych=0.15): 
        
        if getattr(self, '_initialized', False):
            return
            
        self.w_mach = w_mach
        self.w_narc = w_narc
        self.w_psych = w_psych
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"[Dark Triad Engine] 启动，使用设备: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
        print(f"   -> 尝试加载主 NLI 模型 ({nli_model_name})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
            self.model.eval()
            self.nli_model_name = nli_model_name
        except Exception as e:
            fallback_model = "microsoft/deberta-v3-base-mnli"
            logger.warning(f"主模型加载失败: {e}。尝试加载备用模型 {fallback_model}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model).to(self.device)
                self.model.eval()
                self.nli_model_name = fallback_model
            except Exception as e2:
                logger.error(f"所有 NLI 模型加载失败！系统进入 Dummy 安全模式: {e2}")

        self.traits = ["Machiavellianism", "Narcissism", "Psychopathy"]
        self.hypotheses = [
            "This text shows an intention to manipulate, exploit, or deceive the reader for hidden motives.",
            "This text expresses extreme self-importance, grandiosity, and a sense of entitlement.",
            "This text displays coldness, callousness, and a complete lack of empathy or remorse."
        ]
        
        self.entailment_idx = 2    
        self.contradiction_idx = 0 
        if self.model is not None:
            try:
                for idx, label in self.model.config.id2label.items():
                    if str(label).lower() == 'entailment':
                        self.entailment_idx = int(idx)
                    elif str(label).lower() == 'contradiction':
                        self.contradiction_idx = int(idx)
            except Exception as e:
                logger.warning(f"动态获取 NLI 索引失败，使用默认值 E:2, C:0: {e}")

        # 用于自恋特征补偿的第一人称代词集合
        self.first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself'}

        self._initialized = True
        if self.model is not None:
            print(f" [Dark Triad Engine] 成功搭载 {self.nli_model_name}！\n")
        else:
            print(f"[Dark Triad Engine] 以安全降级模式运行 (全 0 输出)！\n")

    def _get_default_scores(self) -> dict:
        return {
            "Machiavellianism_Score": 0.0,
            "Narcissism_Score": 0.0,
            "Psychopathy_Score": 0.0,
            "Dark_Triad_Index": 0.0
        }

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _narcissism_lexical_boost(self, text: str, base_score: float) -> float:
        """
        效度补偿机制
        短推文中，自恋往往表现为频繁的自我中心表达。
        计算第一人称代词密度，提供适当的权重补偿。
        """
        words = text.split()
        if not words:
            return base_score
            
        count = sum(1 for w in words if w in self.first_person_pronouns)
        density = count / len(words)
        
        # 密度越高，补偿越多（上限+0.2）
        boost = min(density * 0.5, 0.2)
        return min(base_score + boost, 1.0)

    def _compute_nli_entailment(self, premise: str, hypothesis: str) -> float:
        if self.model is None or self.tokenizer is None:
            return 0.0
            
        try:
            inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            p_entail = probs[self.entailment_idx]
            p_contra = probs[self.contradiction_idx]
            calibrated_score = (p_entail - p_contra + 1.0) / 2.0
            return float(calibrated_score)
            
        except Exception as e:
            logger.warning(f"单条 NLI 推理失败: {e}")
            return 0.0

    @functools.lru_cache(maxsize=10000)
    def analyze_agent_text(self, text: str, verbose: bool = False) -> dict:
        text = self._clean_text(text)
        
        # 显式的 tokenizer 检查
        if not text or len(text) < 10 or self.model is None or self.tokenizer is None:
            return self._get_default_scores()

        scores = {}
        for trait, hypothesis in zip(self.traits, self.hypotheses):
            scores[trait] = self._compute_nli_entailment(text, hypothesis)
            
        # 对自恋分数应用文本启发式补偿
        scores["Narcissism"] = self._narcissism_lexical_boost(text, scores["Narcissism"])
            
        dark_triad_index = (scores["Machiavellianism"] * self.w_mach) + \
                           (scores["Narcissism"] * self.w_narc) + \
                           (scores["Psychopathy"] * self.w_psych)

        result = {
            "Machiavellianism_Score": round(scores["Machiavellianism"], 4),
            "Narcissism_Score": round(scores["Narcissism"], 4),
            "Psychopathy_Score": round(scores["Psychopathy"], 4),
            "Dark_Triad_Index": round(dark_triad_index, 4)
        }
        
        if verbose:
            result["_Cleaned_Text"] = text
            
        return result

    def analyze_batch(self, texts: List[str], verbose: bool = False, batch_size: int = 32) -> List[Dict]:
        if not texts:
            return []

        # 显式的 tokenizer 检查
        if self.model is None or self.tokenizer is None:
            return [self._get_default_scores() for _ in texts]

        cleaned_texts = [self._clean_text(t) for t in texts]
        final_results = [self._get_default_scores() for _ in texts]
        unique_valid_texts = list(dict.fromkeys(t for t in cleaned_texts if len(t) >= 10))

        result_by_text = {}
        for batch_start in tqdm(
            range(0, len(unique_valid_texts), batch_size),
            desc="DarkTriad NLI batches",
            unit="batch",
        ):
            batch_texts = unique_valid_texts[batch_start: batch_start + batch_size]
            premises = []
            hypotheses_list = []
            for text in batch_texts:
                for hyp in self.hypotheses:
                    premises.append(text)
                    hypotheses_list.append(hyp)

            if not premises:
                continue

            try:
                inputs = self.tokenizer(premises, hypotheses_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                p_entail = probs[:, self.entailment_idx]
                p_contra = probs[:, self.contradiction_idx]
                calibrated_probs = (p_entail - p_contra + 1.0) / 2.0
                
                entailment_matrix = calibrated_probs.reshape(len(batch_texts), len(self.hypotheses))
                
            except Exception as e:
                logger.warning(f"Batch NLI 计算崩溃: {e}")
                entailment_matrix = np.zeros((len(batch_texts), len(self.hypotheses)))

            for ptr, text in enumerate(batch_texts):
                mach_score = float(entailment_matrix[ptr, 0])
                narc_score = float(entailment_matrix[ptr, 1])
                psych_score = float(entailment_matrix[ptr, 2])

                # 批量处理中同样应用自恋特征补偿
                narc_score = self._narcissism_lexical_boost(text, narc_score)

                dt_index = (mach_score * self.w_mach) + (narc_score * self.w_narc) + (psych_score * self.w_psych)

                res = {
                    "Machiavellianism_Score": round(mach_score, 4),
                    "Narcissism_Score": round(narc_score, 4),
                    "Psychopathy_Score": round(psych_score, 4),
                    "Dark_Triad_Index": round(dt_index, 4)
                }
                if verbose:
                    res["_Cleaned_Text"] = text
                result_by_text[text] = res

        for idx, text in enumerate(cleaned_texts):
            if len(text) >= 10 and text in result_by_text:
                final_results[idx] = result_by_text[text]

        return final_results

    def evaluate_agent(self, texts: List[str], anomaly_threshold: float = 0.65) -> Dict:
        if not texts:
            return {"Agent_Mean_Dark_Triad": 0.0, "Agent_Max_Dark_Triad": 0.0, "Agent_Manipulative_Ratio": 0.0}
            
        batch_results = self.analyze_batch(texts, batch_size=32)
        dt_scores = [res.get("Dark_Triad_Index", 0.0) for res in batch_results]
        
        if not dt_scores:
            return {"Agent_Mean_Dark_Triad": 0.0, "Agent_Max_Dark_Triad": 0.0, "Agent_Manipulative_Ratio": 0.0}
            
        mean_dt = np.mean(dt_scores)
        max_dt = np.max(dt_scores)
        anomaly_cnt = sum(1 for s in dt_scores if s >= anomaly_threshold)
        manipulative_ratio = anomaly_cnt / len(dt_scores)
        
        return {
            "Agent_Mean_Dark_Triad": round(float(mean_dt), 4),
            "Agent_Max_Dark_Triad": round(float(max_dt), 4),
            "Agent_Manipulative_Ratio": round(float(manipulative_ratio), 4)
        }
