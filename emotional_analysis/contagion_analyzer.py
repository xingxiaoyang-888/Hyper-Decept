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
    Frictionless Emotional Contagion Analyzer.
    
    Calculates semantic alignment between tweets and predefined target payloads 
    using dense embeddings. Features singleton pattern, batch processing, 
    LRU caching, and graceful degradation.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ContagionAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reload(cls, **kwargs):
        logger.info("Unloading contagion engine and clearing cache/VRAM...")
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
            
        print(f"Engine initialized on: {self.device}")
        
        # Define target manipulation anchors
        if payload_anchors is None:
            self.payload_anchors = [
                "I lost my job and feel completely hopeless about the future.",
                "The economy is ruined, and the system is rigged against ordinary people.",
                "Everything is getting worse, and there is no way to survive this crisis."
            ]
        else:
            self.payload_anchors = payload_anchors

        self.embedder = None
        print(f"Loading embedding model: {model_name}...")

        def load_sentence_embedder(candidate_name):
            from sentence_transformers import SentenceTransformer, util
            try:
                return SentenceTransformer(
                    candidate_name,
                    device=str(self.device),
                    local_files_only=True,
                ), util
            except Exception as local_exc:
                logger.warning(f"Local cache load failed for {candidate_name}, falling back to default: {local_exc}")
                return SentenceTransformer(candidate_name, device=str(self.device)), util

        try:
            self.embedder, self.util = load_sentence_embedder(model_name)
            self.model_name = model_name
        except Exception as e:
            fallback_model = "all-MiniLM-L6-v2"
            logger.warning(f"Primary model load failed: {e}. Trying fallback model {fallback_model}...")
            try:
                self.embedder, self.util = load_sentence_embedder(fallback_model)
                self.model_name = fallback_model
            except Exception as e2:
                logger.error(f"All embedding models failed. Entering dummy mode: {e2}")

        # Pre-compute anchor embeddings
        if self.embedder is not None:
            self.anchor_embeddings = self.embedder.encode(self.payload_anchors, convert_to_tensor=True, show_progress_bar=False)
        else:
            self.anchor_embeddings = None

        self._initialized = True
        if self.embedder is not None:
            print(f"Loaded {self.model_name} successfully.\n")
        else:
            print(f"Running in dummy mode (zero outputs).\n")

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
        """Analyze target alignment for a single tweet."""
        text = self._clean_text(text)
        
        if not text or len(text) < 5 or self.embedder is None:
            return self._get_default_scores()

        try:
            query_emb = self.embedder.encode(text, convert_to_tensor=True, show_progress_bar=False)
            cos_scores = self.util.cos_sim(query_emb, self.anchor_embeddings)[0].cpu().numpy()
            
            # Filter negative correlations
            cos_scores = np.maximum(cos_scores, 0.0)
            
            max_align = float(np.max(cos_scores))
            mean_align = float(np.mean(cos_scores))
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return self._get_default_scores()

        result = {
            "Max_Payload_Alignment": round(max_align, 4),
            "Mean_Payload_Alignment": round(mean_align, 4),
            "Frictionless_Contagion_Score": round(max_align, 4) 
        }
        
        if verbose:
            result["_Cleaned_Text"] = text
            
        return result

    def analyze_batch(self, texts: List[str], verbose: bool = False, batch_size: int = 64) -> List[Dict]:
        """Batch processing for cosine similarity computation."""
        results = []
        if not texts:
            return results

        if self.embedder is None:
            return [self._get_default_scores() for _ in texts]

        cleaned_texts = [self._clean_text(t) for t in texts]
        valid_pairs = [(i, t) for i, t in enumerate(cleaned_texts) if len(t) >= 5]
        final_results = [self._get_default_scores() for _ in texts]
        
        if not valid_pairs:
            return final_results

        unique_texts = list(dict.fromkeys(t for _, t in valid_pairs))

        try:
            query_embs = self.embedder.encode(
                unique_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            
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
            logger.warning(f"Batch similarity calculation failed: {e}")

        return final_results

    def evaluate_agent(self, texts: List[str], response_delays: List[float] = None,
                       return_evidence: bool = False, top_k_evidence: int = 5) -> Dict:
        """
        Aggregate alignments at the agent level.

        Args:
            texts: List of tweets from the agent.
            response_delays: (Optional) Response delays for time penalty.
            return_evidence: If True, include ``evidence`` key with top-scoring
                text-level results.  Default False preserves original behaviour.
            top_k_evidence: Max number of evidence items (default 5).
        """
        if not texts:
            result = {
                "Agent_Mean_Alignment": 0.0,
                "Agent_Contagion_Spike": 0.0,
                "Agent_Frictionless_Index": 0.0,
            }
            if return_evidence:
                result["evidence"] = []
            return result

        batch_results = self.analyze_batch(texts, batch_size=64)
        alignments = [res.get("Max_Payload_Alignment", 0.0) for res in batch_results]

        if not alignments:
            result = {
                "Agent_Mean_Alignment": 0.0,
                "Agent_Contagion_Spike": 0.0,
                "Agent_Frictionless_Index": 0.0,
            }
            if return_evidence:
                result["evidence"] = []
            return result

        mean_align = np.mean(alignments)
        max_align = np.max(alignments)

        if response_delays is not None and len(response_delays) == len(alignments):
            frictionless_scores = []
            for align, delay in zip(alignments, response_delays):
                if delay <= 0:
                    delay = 1.0
                time_penalty = math.exp(-max(0, delay - 5.0) / 60.0)
                frictionless_scores.append(align * time_penalty)

            frictionless_index = np.max(frictionless_scores)
        else:
            frictionless_index = max_align

        result = {
            "Agent_Mean_Alignment": round(float(mean_align), 4),
            "Agent_Contagion_Spike": round(float(max_align), 4),
            "Agent_Frictionless_Index": round(float(frictionless_index), 4),
        }

        if return_evidence:
            evidence: list = []
            indexed = list(enumerate(batch_results))
            indexed.sort(
                key=lambda x: x[1].get("Frictionless_Contagion_Score", 0.0),
                reverse=True,
            )
            for text_index, res in indexed[:top_k_evidence]:
                score = res.get("Frictionless_Contagion_Score", 0.0)
                if score <= 0.0:
                    continue
                evidence.append({
                    "text_index": text_index,
                    "score": score,
                    "signal": "contagion",
                    "details": {
                        "max_payload_alignment": res.get("Max_Payload_Alignment", 0.0),
                        "mean_payload_alignment": res.get("Mean_Payload_Alignment", 0.0),
                    },
                })
            result["evidence"] = evidence

        return result