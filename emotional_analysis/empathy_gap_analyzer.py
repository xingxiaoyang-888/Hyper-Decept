import os
import torch
import numpy as np
import spacy
import warnings
import math
import functools
import logging
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)

class EmpathyGapAnalyzer:
    """
    Cognitive-Affective Empathy Gap Analyzer.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmpathyGapAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reload(cls, **kwargs):
        logger.info("Unloading Empathy Gap engine and clearing VRAM...")
        if cls._instance is not None:
            if hasattr(cls._instance, 'emotion_model'):
                del cls._instance.emotion_model
            if hasattr(cls._instance, 'ppl_model'):
                del cls._instance.ppl_model
            if hasattr(cls._instance, 'nlp'):
                del cls._instance.nlp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cls._instance = None
        return cls(**kwargs)
    
    def __init__(self, 
                 device=None, 
                 spacy_model="en_core_web_sm", 
                 emotion_model_name="SamLowe/roberta-base-go_emotions", 
                 ppl_model_name="gpt2",
                 min_ppl=10.0, 
                 max_ppl=100.0, 
                 tree_depth_norm=6.0, 
                 w_tree=0.6, 
                 w_fluency=0.4,
                 enable_fluency=True):
        
        if getattr(self, '_initialized', False):
            return
            
        self.min_ppl = min_ppl
        self.max_ppl = max_ppl
        self.tree_depth_norm = tree_depth_norm
        self.w_tree = w_tree
        self.w_fluency = w_fluency
        self.enable_fluency = enable_fluency
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Engine initialized on: {self.device}")

        print(f"Loading emotion classifier: {emotion_model_name}...")
        self.emotion_model_name = emotion_model_name
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(self.emotion_model_name).to(self.device)
        self.emotion_model.eval()
        
        target_emotions = {'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'sadness'}
        self.high_arousal_negative_labels = []
        try:
            for idx, label in self.emotion_model.config.id2label.items():
                if str(label).lower() in target_emotions:
                    self.high_arousal_negative_labels.append(int(idx))
        except Exception as e:
            logger.warning(f"Failed to fetch emotion labels dynamically, using hardcoded fallback: {e}")
            self.high_arousal_negative_labels = [2, 3, 11, 12, 13, 14, 16, 25]

        print(f"Loading spaCy parser: {spacy_model}...")
        try:
            self.nlp = spacy.load(spacy_model)
            self.has_dependency_parser = "parser" in self.nlp.pipe_names
        except OSError:
            print(f"{spacy_model} not found, attempting download...")
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", spacy_model],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.nlp = spacy.load(spacy_model)
                self.has_dependency_parser = "parser" in self.nlp.pipe_names
                print(f"Downloaded and loaded {spacy_model} successfully.")
            except Exception:
                logger.warning("Download failed, falling back to blank spaCy English pipeline.")
                self.nlp = spacy.blank("en")
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
                self.has_dependency_parser = False

        self.ppl_model_name = ppl_model_name
        self.ppl_tokenizer = None
        self.ppl_model = None
        if self.enable_fluency:
            print(f"Loading causal LM for perplexity: {ppl_model_name}...")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_name)
            self.ppl_model = AutoModelForCausalLM.from_pretrained(self.ppl_model_name).to(self.device)
            self.ppl_model.eval()
        else:
            print("Fast mode enabled: Skipping GPT-2 perplexity check.")

        self._initialized = True
        print("Empathy Gap Engine loaded successfully.\n")

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing URLs, mentions, and extra whitespace."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _compute_emotional_arousal(self, text: str):
        try:
            inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            arousal_score = float(np.sum(probs[self.high_arousal_negative_labels]))
            return min(arousal_score, 1.0), probs
        except Exception as e:
            logger.warning(f"Emotion calculation failed: {e}")
            return 0.0, None

    def _extract_tree_depth_from_doc(self, doc):
        if not getattr(self, "has_dependency_parser", False):
            return 0.0, 0.0

        sent_depths = []
        for sent in doc.sents:
            token_depths = []
            for token in sent:
                depth = 0
                curr = token
                while curr.head != curr:
                    depth += 1
                    curr = curr.head
                token_depths.append(depth)
            
            if token_depths:
                sent_depths.append(sum(token_depths) / len(token_depths))
            
        if not sent_depths:
            return 0.0, 0.0
        
        overall_avg_depth = sum(sent_depths) / len(sent_depths)
        normalized_depth = min(overall_avg_depth / self.tree_depth_norm, 1.0)
        return normalized_depth, overall_avg_depth

    def _compute_syntactic_tree_depth(self, text: str):
        try:
            doc = self.nlp(text)
            return self._extract_tree_depth_from_doc(doc)
        except Exception as e:
            logger.warning(f"Syntactic parsing failed: {e}")
            return 0.0, 0.0

    def _compute_fluency_rigidity(self, text: str):
        if not self.enable_fluency or self.ppl_model is None or self.ppl_tokenizer is None:
            return 0.0, 0.0

        try:
            encodings = self.ppl_tokenizer(text, return_tensors="pt").to(self.device)
            max_length = self.ppl_model.config.n_positions
            seq_len = encodings.input_ids.size(1)

            if seq_len <= max_length:
                with torch.no_grad():
                    outputs = self.ppl_model(encodings.input_ids, labels=encodings.input_ids)
                nlls = [outputs.loss]
            else:
                stride = 512
                nlls = []
                prev_end_loc = 0
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc
                    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100

                    with torch.no_grad():
                        outputs = self.ppl_model(input_ids, labels=target_ids)
                    nlls.append(outputs.loss)
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

            if not nlls:
                return 0.0, 0.0

            ppl = torch.exp(torch.stack(nlls).mean()).item()
            clamped_ppl = max(min(ppl, self.max_ppl), self.min_ppl)
            
            log_ppl = math.log(clamped_ppl)
            log_min = math.log(self.min_ppl)
            log_max = math.log(self.max_ppl)
            
            rigidity_score = 1.0 - ((log_ppl - log_min) / (log_max - log_min))
            return max(min(rigidity_score, 1.0), 0.0), ppl
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 0.0, 0.0

    @functools.lru_cache(maxsize=2048)
    def analyze_agent_text(self, text: str, verbose: bool = False) -> dict:
        text = self._clean_text(text)
        if not text or len(text) < 5:
            return {"Empathy_Gap": 0.0}

        arousal, raw_probs = self._compute_emotional_arousal(text)
        tree_depth_score, raw_depth = self._compute_syntactic_tree_depth(text)
        
        try:
            token_len = len(self.ppl_tokenizer.encode(text)) if self.ppl_tokenizer is not None else 0
        except Exception:
            token_len = 0
            
        if token_len < 15:
            fluency_score, raw_ppl = 0.0, 0.0
            cognitive_rigidity = tree_depth_score
        else:
            fluency_score, raw_ppl = self._compute_fluency_rigidity(text)
            cognitive_rigidity = (self.w_tree * tree_depth_score) + (self.w_fluency * fluency_score)
            
        empathy_gap = arousal * cognitive_rigidity
        
        result = {
            "Affective_Arousal": round(arousal, 4),
            "Cognitive_Tree_Depth_Score": round(tree_depth_score, 4),
            "Cognitive_Fluency_Score": round(fluency_score, 4),
            "Composite_Cognitive_Rigidity": round(cognitive_rigidity, 4),
            "Empathy_Gap": round(empathy_gap, 4)
        }
        
        if verbose:
            result["_Raw_PPL"] = raw_ppl
            result["_Raw_Avg_Depth"] = raw_depth
            
        return result

    def analyze_batch(self, texts: List[str], verbose: bool = False, batch_size: int = 32) -> List[Dict]:
        if not texts:
            return []

        cleaned_texts = [self._clean_text(t) for t in texts]
        final_results = [{"Empathy_Gap": 0.0} for _ in texts]
        unique_texts = list(dict.fromkeys(t for t in cleaned_texts if t and len(t) >= 5))
        result_by_text = {}

        for batch_start in tqdm(
            range(0, len(unique_texts), batch_size),
            desc="Empathy batches",
            unit="batch",
        ):
            batch_texts = unique_texts[batch_start: batch_start + batch_size]
            
            batch_arousals = []
            try:
                inputs = self.emotion_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    logits = self.emotion_model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for i in range(len(batch_texts)):
                    score = float(np.sum(probs[i][self.high_arousal_negative_labels]))
                    batch_arousals.append(min(score, 1.0))
            except Exception as e:
                logger.warning(f"Batch emotion calculation failed: {e}")
                batch_arousals = [0.0] * len(batch_texts)

            try:
                docs = list(self.nlp.pipe(batch_texts, disable=["ner"]))
            except Exception as e:
                logger.warning(f"Batch syntactic parsing failed: {e}")
                docs = [None] * len(batch_texts)

            for i, text in enumerate(tqdm(batch_texts, desc="Empathy batch details", leave=False)):
                arousal = batch_arousals[i]
                
                if docs[i] is not None:
                    tree_depth_score, raw_depth = self._extract_tree_depth_from_doc(docs[i])
                else:
                    tree_depth_score, raw_depth = 0.0, 0.0
                
                try:
                    token_len = len(self.ppl_tokenizer.encode(text)) if self.ppl_tokenizer is not None else 0
                except Exception:
                    token_len = 0
                    
                if token_len < 15:
                    fluency_score, raw_ppl = 0.0, 0.0
                    cognitive_rigidity = tree_depth_score
                else:
                    fluency_score, raw_ppl = self._compute_fluency_rigidity(text)
                    cognitive_rigidity = (self.w_tree * tree_depth_score) + (self.w_fluency * fluency_score)
                    
                gap = arousal * cognitive_rigidity
                
                res = {
                    "Affective_Arousal": round(arousal, 4),
                    "Cognitive_Tree_Depth_Score": round(tree_depth_score, 4),
                    "Cognitive_Fluency_Score": round(fluency_score, 4),
                    "Composite_Cognitive_Rigidity": round(cognitive_rigidity, 4),
                    "Empathy_Gap": round(gap, 4)
                }
                if verbose:
                    res["_Raw_PPL"] = raw_ppl
                    res["_Raw_Avg_Depth"] = raw_depth
                result_by_text[text] = res

        for idx, text in enumerate(cleaned_texts):
            if text in result_by_text:
                final_results[idx] = result_by_text[text]

        return final_results

    def evaluate_agent(self, texts: List[str], anomaly_threshold: float = 0.6) -> Dict:
        """
        Aggregates empathy gap scores at the agent level.
        """
        if not texts:
            return {
                "Agent_Mean_Empathy_Gap": 0.0,
                "Agent_Max_Empathy_Gap": 0.0,
                "Agent_Anomaly_Ratio": 0.0
            }
            
        batch_results = self.analyze_batch(texts, batch_size=32)
        
        gaps = [res.get("Empathy_Gap", 0.0) for res in batch_results]
        
        if not gaps:
            return {"Agent_Mean_Empathy_Gap": 0.0, "Agent_Max_Empathy_Gap": 0.0, "Agent_Anomaly_Ratio": 0.0}
            
        mean_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        anomaly_cnt = sum(1 for g in gaps if g >= anomaly_threshold)
        anomaly_ratio = anomaly_cnt / len(gaps)
        
        return {
            "Agent_Mean_Empathy_Gap": round(float(mean_gap), 4),
            "Agent_Max_Empathy_Gap": round(float(max_gap), 4),
            "Agent_Anomaly_Ratio": round(float(anomaly_ratio), 4)
        }