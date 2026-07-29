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
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmotionVolatilityAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reload(cls, **kwargs):
        logger.info("Unloading Volatility engine and clearing VRAM...")
        if cls._instance is not None:
            if hasattr(cls._instance, 'model') and cls._instance.model is not None:
                del cls._instance.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cls._instance = None
        return cls(**kwargs)
    
    def __init__(self, 
                 device=None, 
                 # English default: SamLowe/roberta-base-go_emotions
                 # Multilingual fallback: MilaNLProc/xlm-roberta-base-goemotions
                 emotion_model_name="SamLowe/roberta-base-go_emotions"):
        
        if getattr(self, '_initialized', False):
            return
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Engine initialized on: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
        print(f"Loading 28-dimensional emotion model: {emotion_model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(self.device)
            self.model.eval()
            self.emotion_model_name = emotion_model_name
            self.emotion_dim = int(getattr(self.model.config, "num_labels", 28))
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.emotion_dim = 28

        self._initialized = True
        if self.model is not None:
            print("Model loaded successfully.\n")
        else:
            print("Running in dummy mode.\n")

    def _get_default_scores(self, insufficient: bool = True) -> dict:
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
            desc="Emotion extraction batches",
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
                logger.warning(f"Batch emotion extraction failed: {e}")
                for text in batch_texts:
                    vector_by_text[text] = np.zeros(self.emotion_dim)
                
        return np.array([vector_by_text.get(text, np.zeros(self.emotion_dim)) for text in texts])

    def evaluate_agents_batch(self, agents_texts: List[List[str]], batch_size: int = 64) -> List[Dict]:
        """
        Batch calculates functional emotion volatility for multiple agents.
        Uses the same cleaning, filtering, and Euclidean distance logic as evaluate_agent, 
        optimizing via global batch inference.
        """
        prepared_agents = []
        all_valid_texts = []

        for texts in tqdm(agents_texts, desc="Preparing agents", unit="agent", leave=False):
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
        for valid_texts in tqdm(prepared_agents, desc="Aggregating agents", unit="agent", leave=False):
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

    def evaluate_agent(self, texts: List[str], timestamps: List[float] = None,
                       return_evidence: bool = False, top_k_evidence: int = 5) -> Dict:
        """
        Aggregates emotion sequence volatility at the agent level.

        Parameters
        ----------
        texts : List[str]
            Agent tweet texts in chronological order.
        timestamps : List[float], optional
            Per-text timestamps for sorting.
        return_evidence : bool
            If True, include ``evidence`` key with the transition pairs that
            contributed highest volatility.  Default False.
        top_k_evidence : int
            Max number of transition-pair evidence items (default 5).
        """
        if not texts or len(texts) < 2:
            result = self._get_default_scores(insufficient=True)
            if return_evidence:
                result["evidence"] = []
            return result

        if timestamps is not None:
            if len(timestamps) != len(texts):
                logger.warning(
                    "Timestamps length mismatch. Ignoring timestamps and "
                    "assuming texts are chronologically ordered."
                )
            else:
                sorted_pairs = sorted(zip(timestamps, texts), key=lambda x: x[0])
                texts = [p[1] for p in sorted_pairs]

        # 1. Clean and filter short noisy texts
        cleaned_texts = [self._clean_text(t) for t in texts]
        valid_indices = [i for i, t in enumerate(cleaned_texts) if len(t) >= 5]

        if len(valid_indices) < 2:
            result = self._get_default_scores(insufficient=True)
            if return_evidence:
                result["evidence"] = []
            return result

        valid_texts = [cleaned_texts[i] for i in valid_indices]

        # 2. Extract 28-dimensional emotion probability vectors
        emotion_matrix = self._get_emotion_vectors_batch(valid_texts)

        # 3. Calculate Euclidean distance
        diff_matrix = emotion_matrix[1:] - emotion_matrix[:-1]

        # Retain raw Euclidean distances without destructive extreme normalization
        euclidean_distances = np.linalg.norm(diff_matrix, axis=1)

        mean_volatility = float(np.mean(euclidean_distances))
        max_volatility = float(np.max(euclidean_distances))

        result = {
            "Agent_Mean_Volatility": round(mean_volatility, 4),
            "Agent_Max_Volatility": round(max_volatility, 4),
            "Insufficient_Data": False,
        }

        if return_evidence:
            # Evidence for volatility = transition pairs with highest distance
            evidence: list = []
            # euclidean_distances[i] is the distance from valid_indices[i] to valid_indices[i+1]
            indexed = list(enumerate(euclidean_distances.tolist()))
            indexed.sort(key=lambda x: x[1], reverse=True)
            for transition_idx, dist in indexed[:top_k_evidence]:
                if dist <= 0.0:
                    continue
                before_idx = valid_indices[transition_idx]
                after_idx = valid_indices[transition_idx + 1]
                evidence.append({
                    "text_index": before_idx,
                    "text_index_next": after_idx,
                    "score": round(float(dist), 4),
                    "signal": "volatility_transition",
                    "details": {
                        "euclidean_distance": round(float(dist), 4),
                    },
                })
            result["evidence"] = evidence

        return result