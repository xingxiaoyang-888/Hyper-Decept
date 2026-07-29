import hashlib
import contextlib
import json
import os
import sqlite3
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import re
from tqdm import tqdm

try:
    from detection_module.hyper_newtest.config import (
        first_existing,
        get_table_columns,
        split_tweet_pool,
    )
except ImportError:
    from config import first_existing, get_table_columns, split_tweet_pool

try:
    from emotional_analysis.empathy_gap_analyzer import EmpathyGapAnalyzer
    from emotional_analysis.dark_triad_analyzer import DarkTriadAnalyzer
    from emotional_analysis.contagion_analyzer import ContagionAnalyzer
    from emotional_analysis.volatility_analyzer import EmotionVolatilityAnalyzer
except ImportError as e:
    warnings.warn(f"Please check the emotional_analysis directory: {e}")
    EmpathyGapAnalyzer, DarkTriadAnalyzer, ContagionAnalyzer, EmotionVolatilityAnalyzer = None, None, None, None

warnings.filterwarnings('ignore')

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

class MultimodalExtractor:
    def __init__(
        self,
        model_path="all-mpnet-base-v2",
        aligned_text_dim=128,
        psychology_mode=None,
        max_tweets_per_user=None,
        cache_dir=None,
        use_cache=True,
        verbose_progress=None,
    ):
       
        self.psychology_mode = (psychology_mode or os.getenv("AFG_PSYCHOLOGY_MODE", "full")).lower()
        if self.psychology_mode not in {"full", "fast", "off"}:
            raise ValueError("psychology_mode must be one of: full, fast, off.")

        max_tweets_env = os.getenv("AFG_MAX_TWEETS_PER_USER")
        max_tweets_value = max_tweets_per_user if max_tweets_per_user is not None else max_tweets_env
        self.max_tweets_per_user = int(max_tweets_value) if max_tweets_value not in (None, "", "none", "None") else None
        self.cache_dir = cache_dir or os.getenv("AFG_FEATURE_CACHE_DIR")
        self.use_cache = use_cache
        if verbose_progress is None:
            verbose_progress = str(os.getenv("AFG_VERBOSE_PROGRESS", "0")).lower() in {"1", "true", "yes", "on"}
        self.verbose_progress = bool(verbose_progress)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f" psychology mode: {self.psychology_mode}")
        if self.max_tweets_per_user is not None:
            print(f" max tweets per user: {self.max_tweets_per_user}")
        if not self.verbose_progress:
            print(" internal progress bars: quiet (set AFG_VERBOSE_PROGRESS=1 to show)")
        
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.text_model = SentenceTransformer(model_path, local_files_only=True)
            except Exception as local_exc:
                warnings.warn(f"Local sentence_transformers load failed, trying default loader: {local_exc}")
                self.text_model = SentenceTransformer(model_path)
        except Exception:
            warnings.warn("Unable to load sentence_transformers. Zero vector placeholder will be used.。")
            self.text_model = None
            
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=aligned_text_dim, random_state=42) 
        self.aligned_text_dim = aligned_text_dim
        
        if self.psychology_mode == "off":
            self.empathy_engine = None
            self.dark_triad_engine = None
            self.contagion_engine = None
            self.volatility_engine = None
        elif self.psychology_mode == "fast":
            self.empathy_engine = EmpathyGapAnalyzer(enable_fluency=False) if EmpathyGapAnalyzer else None
            self.dark_triad_engine = None
            self.contagion_engine = ContagionAnalyzer() if ContagionAnalyzer else None
            self.volatility_engine = EmotionVolatilityAnalyzer() if EmotionVolatilityAnalyzer else None
            warnings.warn("psychology_mode=fast skips the slowest fluency/NLI checks; use full for the original full scan.")
        else:
            self.empathy_engine = EmpathyGapAnalyzer() if EmpathyGapAnalyzer else None
            self.dark_triad_engine = DarkTriadAnalyzer() if DarkTriadAnalyzer else None
            self.contagion_engine = ContagionAnalyzer() if ContagionAnalyzer else None
            self.volatility_engine = EmotionVolatilityAnalyzer() if EmotionVolatilityAnalyzer else None

    def _run_engine(self, func, *args, **kwargs):
        if self.verbose_progress:
            return func(*args, **kwargs)
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return func(*args, **kwargs)

    def extract_behavior_features(self, db_path: str, user_ids_master: list = None):
        print(f"connnect: {os.path.basename(db_path)}")
        with sqlite3.connect(db_path) as conn:
            schema = get_table_columns(conn)

            if user_ids_master is not None:
                user_ids = [str(uid) for uid in user_ids_master]
            elif 'user' in schema and 'user_id' in schema['user']:
                user_ids = pd.read_sql_query("SELECT user_id FROM user", conn)['user_id'].astype(str).tolist()
            elif 'agent_actions' in schema:
                user_ids = pd.read_sql_query(
                    "SELECT DISTINCT agent_name AS user_id FROM agent_actions",
                    conn,
                )['user_id'].astype(str).tolist()
            else:
                user_ids = []

            df_base = pd.DataFrame({'user_id': user_ids})
            n = len(df_base)

            followers = np.zeros(n, dtype=float)
            following = np.zeros(n, dtype=float)
            tweet_cnt = np.zeros(n, dtype=float)
            listed = np.zeros(n, dtype=float)
            post_feat = np.zeros((n, 7), dtype=float)

            if n == 0:
                return user_ids, np.zeros((0, 10), dtype=float)
            if 'user' in schema and 'user_id' in schema['user']:
                df_users = pd.read_sql_query("SELECT * FROM user", conn)
                df_users['user_id'] = df_users['user_id'].astype(str)
                df_users = df_users.drop_duplicates('user_id')
                df_users = df_base.merge(df_users, on='user_id', how='left')

                user_cols = schema['user']
                followers_col = first_existing(user_cols, ['followers_count', 'num_followers', 'followers'])
                following_col = first_existing(user_cols, ['following_count', 'num_followings', 'following'])
                tweet_col = first_existing(user_cols, ['tweet_count', 'statuses_count', 'num_tweets'])
                listed_col = first_existing(user_cols, ['listed_count'])

                if followers_col:
                    followers = pd.to_numeric(df_users[followers_col], errors='coerce').fillna(0).to_numpy(float)
                if following_col:
                    following = pd.to_numeric(df_users[following_col], errors='coerce').fillna(0).to_numpy(float)
                if tweet_col:
                    tweet_cnt = pd.to_numeric(df_users[tweet_col], errors='coerce').fillna(0).to_numpy(float)
                if listed_col:
                    listed = pd.to_numeric(df_users[listed_col], errors='coerce').fillna(0).to_numpy(float)

            follower_ratio = followers / (following + 1.0)

         
            if 'post' in schema and 'user_id' in schema['post']:
                df_post = pd.read_sql_query("SELECT * FROM post", conn)
                if not df_post.empty:
                    df_post['user_id'] = df_post['user_id'].astype(str)
                    post_cols = schema['post']
                    aggregate_map = [
                        ('avg_like', first_existing(post_cols, ['like_count', 'num_likes'])),
                        ('avg_retweet', first_existing(post_cols, ['retweet_count', 'num_shares'])),
                        ('avg_reply', first_existing(post_cols, ['reply_count'])),
                        ('avg_url', first_existing(post_cols, ['url_count'])),
                        ('avg_mention', first_existing(post_cols, ['mention_count'])),
                        ('avg_hashtag', first_existing(post_cols, ['hashtag_count'])),
                        ('avg_media', first_existing(post_cols, ['media_count'])),
                    ]
                    for out_col, src_col in aggregate_map:
                        df_post[out_col] = (
                            pd.to_numeric(df_post[src_col], errors='coerce').fillna(0.0)
                            if src_col else 0.0
                        )

                    df_post_agg = (
                        df_post.groupby('user_id', as_index=False)
                        .agg(
                            post_count=('user_id', 'size'),
                            avg_like=('avg_like', 'mean'),
                            avg_retweet=('avg_retweet', 'mean'),
                            avg_reply=('avg_reply', 'mean'),
                            avg_url=('avg_url', 'mean'),
                            avg_mention=('avg_mention', 'mean'),
                            avg_hashtag=('avg_hashtag', 'mean'),
                            avg_media=('avg_media', 'mean'),
                        )
                    )
                    df_post_agg = df_base.merge(df_post_agg, on='user_id', how='left').fillna(0.0)
                    post_feat = df_post_agg[
                        ['avg_like', 'avg_retweet', 'avg_reply', 'avg_url', 'avg_mention', 'avg_hashtag', 'avg_media']
                    ].to_numpy(float)

                    if np.all(tweet_cnt == 0):
                        tweet_cnt = df_post_agg['post_count'].to_numpy(float)

         
            elif 'agent_actions' in schema and {'agent_name', 'action_type'}.issubset(set(schema['agent_actions'])):
                df_actions = pd.read_sql_query("SELECT agent_name, action_type FROM agent_actions", conn)
                if not df_actions.empty:
                    df_actions['user_id'] = df_actions['agent_name'].astype(str)
                    df_actions['action_type'] = df_actions['action_type'].astype(str).str.lower()
                    action_counts = (
                        df_actions.groupby(['user_id', 'action_type']).size().unstack(fill_value=0).reset_index()
                    )
                    total_counts = df_actions.groupby('user_id').size().reset_index(name='action_count')
                    action_counts = df_base.merge(action_counts, on='user_id', how='left').fillna(0.0)
                    total_counts = df_base.merge(total_counts, on='user_id', how='left').fillna(0.0)

                    denom = np.maximum(total_counts['action_count'].to_numpy(float), 1.0)
                    if np.all(tweet_cnt == 0):
                        tweet_cnt = total_counts['action_count'].to_numpy(float)

                    for out_idx, action_name in enumerate(['like', 'retweet', 'reply']):
                        if action_name in action_counts.columns:
                            post_feat[:, out_idx] = action_counts[action_name].to_numpy(float) / denom

            behavior_matrix = np.column_stack([
                follower_ratio,       # Follower_Following_Ratio
                tweet_cnt,            # Action_Frequency
                post_feat[:, 0],      # Like_Ratio
                post_feat[:, 1],      # Retweet_Ratio
                post_feat[:, 2],      # Reply_Ratio
                listed,               # Temporal_Entropy (proxy: listed_count)
                post_feat[:, 3],      # URL_Ratio
                post_feat[:, 4],      # Mention_Ratio
                post_feat[:, 5],      # Hashtag_Ratio
                post_feat[:, 6],      # Media_Ratio
            ])

        return user_ids, np.nan_to_num(behavior_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode_text_dual_stream(self, bios: list, tweets_list: list):
        if self.text_model is None:
            return np.zeros((len(bios), self.aligned_text_dim))

        bio_embs = np.asarray(self.text_model.encode(bios, show_progress_bar=False))
        emb_dim = bio_embs.shape[1] if bio_embs.ndim == 2 else self.aligned_text_dim

        tweet_groups = [split_tweet_pool(tweets_str, min_len=2) for tweets_str in tweets_list]
        if self.max_tweets_per_user is not None:
            tweet_groups = [
                group[: self.max_tweets_per_user] for group in tweet_groups
            ]
        flat_tweets = [tweet for group in tweet_groups for tweet in group]
        tweet_embs = np.zeros((len(tweet_groups), emb_dim), dtype=float)

        if flat_tweets:
            flat_embs = np.asarray(self.text_model.encode(flat_tweets, batch_size=64, show_progress_bar=False))
            cursor = 0
            for idx, group in enumerate(tweet_groups):
                group_len = len(group)
                if group_len:
                    tweet_embs[idx] = flat_embs[cursor: cursor + group_len].mean(axis=0)
                    cursor += group_len

        raw_text_matrix = np.hstack((bio_embs, tweet_embs))

        if raw_text_matrix.shape[0] > self.aligned_text_dim:
            aligned_matrix = self.pca.fit_transform(raw_text_matrix)
        else:
            aligned_matrix = raw_text_matrix
        return aligned_matrix

    def _extract_llm_native_psychology(self, tweets_list: list,
                                        return_evidence: bool = False):
        """Extract 8-dim psychology feature vectors.

        Parameters
        ----------
        tweets_list : list of str
            Per-user tweet pools.
        return_evidence : bool
            If True, also return per-user text-level evidence dicts from each
            analyser.  Default False preserves original behaviour.

        Returns
        -------
        psycho_matrix : np.ndarray  (N, 8)
        evidence_by_user : list of dict or None
            Only returned when *return_evidence* is True.  Each element is a
            dict mapping analyser name -> list of evidence items.
        """
        agent_tweets = [split_tweet_pool(tweets_str, min_len=5) for tweets_str in tweets_list]
        if self.max_tweets_per_user is not None:
            agent_tweets = [texts[:self.max_tweets_per_user] for texts in agent_tweets]

        if self.psychology_mode == "off":
            print(" [Psychological Stream] off mode: returning zero psychology features.")
            if return_evidence:
                return np.zeros((len(tweets_list), 8), dtype=float), [{} for _ in tweets_list]
            return np.zeros((len(tweets_list), 8), dtype=float)

        cache_prefix = self._psychology_cache_path(agent_tweets)
        matrix_cache = cache_prefix + ".npy" if cache_prefix else None
        evidence_cache = cache_prefix + ".evidence.json" if cache_prefix else None

        # Paired cache hit: both matrix AND evidence sidecar must exist.
        if matrix_cache and os.path.exists(matrix_cache):
            if not return_evidence:
                print(f" [Psychological Stream] hit feature cache: {matrix_cache}")
                return np.load(matrix_cache)
            if evidence_cache and os.path.exists(evidence_cache):
                print(f" [Psychological Stream] hit paired cache: {matrix_cache} + {evidence_cache}")
                with open(evidence_cache, "r", encoding="utf-8") as fh:
                    ev_data = json.load(fh)
                return np.load(matrix_cache), ev_data
            # Old cache: matrix exists but evidence sidecar missing.
            # Must re-generate evidence; re-use cached matrix.
            print(f" [Psychological Stream] cache hit (matrix only), re-generating evidence sidecar ...")
            psycho_matrix = np.load(matrix_cache)
            if return_evidence:
                _, evidence_by_user = self._run_psychology_engines(
                    agent_tweets, return_evidence=True
                )
                if evidence_cache:
                    with open(evidence_cache, "w", encoding="utf-8") as fh:
                        json.dump(evidence_by_user, fh, ensure_ascii=False)
                return psycho_matrix, evidence_by_user
            return psycho_matrix

        psycho_features, ev_out = self._run_psychology_engines(
            agent_tweets, return_evidence=return_evidence,
        )
        if return_evidence:
            evidence_by_user = ev_out
        psycho_matrix = np.array(psycho_features)
        if matrix_cache:
            np.save(matrix_cache, psycho_matrix)
            if return_evidence and evidence_cache:
                with open(evidence_cache, "w", encoding="utf-8") as fh:
                    json.dump(evidence_by_user, fh, ensure_ascii=False)
            print(f"  feature cache saved: {matrix_cache}")

        if return_evidence:
            return psycho_matrix, evidence_by_user
        return psycho_matrix

    def _run_psychology_engines(self, agent_tweets, return_evidence=False):
        """Run all four psychology engines and return features + evidence.

        Extracted so cache-hit path can re-run evidence without re-running
        the full feature computation.
        """
        psycho_features: list = []
        evidence_by_user: list = []

        for agent_tweets_array in tqdm(
            agent_tweets,
            desc="Deep Psycho-Scan",
            unit="agent",
            disable=not self.verbose_progress,
        ):
            user_evidence: dict = {}

            # 1. (Empathy Gap)
            if self.empathy_engine and agent_tweets_array:
                emp_res = self._run_engine(
                    self.empathy_engine.evaluate_agent,
                    agent_tweets_array,
                    return_evidence=return_evidence,
                )
                emp_mean = emp_res.get("Agent_Mean_Empathy_Gap", 0.0)
                emp_max = emp_res.get("Agent_Max_Empathy_Gap", 0.0)
                if return_evidence:
                    user_evidence["empathy_gap"] = emp_res.get("evidence", [])
            else:
                emp_mean, emp_max = 0.0, 0.0

            # 2. (Dark Triad)
            if self.dark_triad_engine and agent_tweets_array:
                dt_res = self._run_engine(
                    self.dark_triad_engine.evaluate_agent,
                    agent_tweets_array,
                    return_evidence=return_evidence,
                )
                dt_mean = dt_res.get("Agent_Mean_Dark_Triad", 0.0)
                dt_max = dt_res.get("Agent_Max_Dark_Triad", 0.0)
                if return_evidence:
                    user_evidence["dark_triad"] = dt_res.get("evidence", [])
            else:
                dt_mean, dt_max = 0.0, 0.0

            # 3.  (Frictionless Contagion)
            if self.contagion_engine and agent_tweets_array:
                cont_res = self._run_engine(
                    self.contagion_engine.evaluate_agent,
                    agent_tweets_array,
                    return_evidence=return_evidence,
                )
                cont_mean = cont_res.get("Agent_Mean_Alignment", 0.0)
                cont_max = cont_res.get("Agent_Contagion_Spike", 0.0)
                if len(psycho_features) == 0 and cont_mean > 0:
                    print(f"Contagion First non-zero value: mean={cont_mean:.4f}, max={cont_max:.4f}")
                if return_evidence:
                    user_evidence["contagion"] = cont_res.get("evidence", [])
            else:
                cont_mean, cont_max = 0.0, 0.0

            # 4.  (Emotion Volatility)
            if self.volatility_engine and agent_tweets_array:
                vol_res = self._run_engine(
                    self.volatility_engine.evaluate_agent,
                    agent_tweets_array,
                    return_evidence=return_evidence,
                )
                if not vol_res.get("Insufficient_Data", True):
                    vol_mean = vol_res.get("Agent_Mean_Volatility", 0.0)
                    vol_max = vol_res.get("Agent_Max_Volatility", 0.0)
                else:
                    vol_mean, vol_max = 0.0, 0.0
                if return_evidence:
                    user_evidence["volatility"] = vol_res.get("evidence", [])
            else:
                vol_mean, vol_max = 0.0, 0.0

            psycho_features.append([
                emp_mean, emp_max,
                dt_mean, dt_max,
                cont_mean, cont_max,
                vol_mean, vol_max,
            ])
            if return_evidence:
                evidence_by_user.append(user_evidence)

        return np.array(psycho_features), evidence_by_user if return_evidence else None

    def _psychology_cache_path(self, agent_tweets):
        if not self.use_cache or not self.cache_dir:
            return None
        digest = hashlib.sha256()
        digest.update(self.psychology_mode.encode("utf-8"))
        digest.update(str(self.max_tweets_per_user).encode("utf-8"))
        for texts in agent_tweets:
            digest.update(str(len(texts)).encode("utf-8"))
            for text in texts:
                encoded = str(text).encode("utf-8", errors="ignore")
                digest.update(str(len(encoded)).encode("utf-8"))
                digest.update(encoded)
        prefix = f"psychology_{digest.hexdigest()[:24]}"
        return os.path.join(self.cache_dir, prefix)  # no .npy suffix -- caller appends

    def fuse_multimodal_features(self, db_path: str, bios: list,
                                 tweets_list: list = None,
                                 user_ids_master: list = None,
                                 return_provenance: bool = False):
        """Fuse semantic, behavioural and psychological modalities.

        Parameters
        ----------
        db_path : str
            Path to SQLite database.
        bios : list of str
            Per-user biography / description strings.
        tweets_list : list of str, optional
            Per-user tweet pools.
        user_ids_master : list, optional
            Ordered user ids to align against.
        return_provenance : bool
            If True, returns a third element ``provenance`` mapping user_id
            to per-feature metadata.  Default False preserves the original
            (user_ids, fused_matrix) return signature.

        Returns
        -------
        user_ids : list of str
        fused_matrix : np.ndarray
        provenance : dict   (only when *return_provenance* is True)
        """
        print("\n" + "=" * 50)
        print("=" * 50)

        user_ids, behavior_matrix = self.extract_behavior_features(db_path, user_ids_master)

        if tweets_list is None:
            parsed_bios, parsed_tweets = [], []
            for s in bios:
                m = re.search(r'Bio:\s*(.*?)\.\s*Recent actions:\s*(.*)$', str(s), flags=re.IGNORECASE)
                if m:
                    parsed_bios.append(m.group(1).strip())
                    parsed_tweets.append(m.group(2).strip())
                else:
                    parsed_bios.append(str(s))
                    parsed_tweets.append("")
            bios_input, tweets_input = parsed_bios, parsed_tweets
        else:
            bios_input, tweets_input = bios, tweets_list

        aligned_text_matrix = self._encode_text_dual_stream(bios_input, tweets_input)

        _psycho_result = self._extract_llm_native_psychology(
            tweets_input, return_evidence=return_provenance,
        )
        if return_provenance:
            psycho_matrix, psycho_evidence = _psycho_result
        else:
            psycho_matrix = _psycho_result
            psycho_evidence = None

        min_len = min(len(user_ids), len(aligned_text_matrix), len(behavior_matrix), len(psycho_matrix))
        if min_len < len(user_ids) or min_len < len(aligned_text_matrix):
            warnings.warn(f"Input modality lengths differ; aligned to shortest length {min_len}.")
            user_ids = user_ids[:min_len]
            aligned_text_matrix = aligned_text_matrix[:min_len]
            behavior_matrix = behavior_matrix[:min_len]
            psycho_matrix = psycho_matrix[:min_len]
            if psycho_evidence is not None:
                psycho_evidence = psycho_evidence[:min_len]

        enhanced_behavior = np.hstack((behavior_matrix, psycho_matrix))
        behavior_matrix_scaled = self.scaler.fit_transform(enhanced_behavior)

        fused_matrix = np.hstack((aligned_text_matrix, behavior_matrix_scaled))

        print(f" complete: {fused_matrix.shape} ")

        if not return_provenance:
            return user_ids, fused_matrix

        # -- Build provenance -----------------------------------------------
        provenance: dict = {}
        # We record the *raw* (pre-scaling) psychology feature values in
        # provenance so they are interpretable.
        psycho_raw = np.hstack((behavior_matrix, psycho_matrix))
        # psycho_raw columns:
        #  0: Follower_Following_Ratio
        #  1: Action_Frequency
        #  2: Like_Ratio
        #  3: Retweet_Ratio
        #  4: Reply_Ratio
        #  5: Temporal_Entropy (listed_count)
        #  6: URL_Ratio
        #  7: Mention_Ratio
        #  8: Hashtag_Ratio
        #  9: Media_Ratio
        # 10: Empathy_Gap_Mean,  11: Empathy_Gap_Max
        # 12: Dark_Triad_Mean,   13: Dark_Triad_Max
        # 14: Contagion_Mean,    15: Contagion_Max
        # 16: Volatility_Mean,   17: Volatility_Max

        behaviour_names = [
            "Follower_Following_Ratio", "Action_Frequency",
            "Like_Ratio", "Retweet_Ratio", "Reply_Ratio",
            "Temporal_Entropy", "URL_Ratio", "Mention_Ratio",
            "Hashtag_Ratio", "Media_Ratio",
        ]
        psycho_names = [
            "Empathy_Gap_Mean", "Empathy_Gap_Max",
            "Dark_Triad_Mean", "Dark_Triad_Max",
            "Contagion_Mean", "Contagion_Max",
            "Volatility_Mean", "Volatility_Max",
        ]
        all_raw_names = behaviour_names + psycho_names

        psycho_evidence_sources = {
            "Empathy_Gap_Mean": "empathy_gap",
            "Empathy_Gap_Max": "empathy_gap",
            "Dark_Triad_Mean": "dark_triad",
            "Dark_Triad_Max": "dark_triad",
            "Contagion_Mean": "contagion",
            "Contagion_Max": "contagion",
            "Volatility_Mean": "volatility",
            "Volatility_Max": "volatility",
        }

        for i, uid in enumerate(user_ids):
            uid_str = str(uid)
            entry: dict = {}

            # -- behavioural features (no per-text evidence) ----------------
            for j, name in enumerate(behaviour_names):
                entry[name] = {
                    "value": float(psycho_raw[i, j]) if i < psycho_raw.shape[0] else 0.0,
                    "extractor": "MultimodalExtractor.extract_behavior_features",
                    "evidence_ids": [],
                    "text_indices": [],
                    "metadata": {"feature_group": "behavior"},
                }

            # -- psychology features with text evidence ---------------------
            for j, name in enumerate(psycho_names):
                prov: dict = {
                    "value": float(psycho_raw[i, 10 + j]) if i < psycho_raw.shape[0] else 0.0,
                    "extractor": "",
                    "evidence_ids": [],
                    "text_indices": [],
                    "metadata": {"feature_group": "psychology"},
                }

                source_key = psycho_evidence_sources.get(name, "")
                if source_key == "empathy_gap":
                    prov["extractor"] = "EmpathyGapAnalyzer"
                elif source_key == "dark_triad":
                    prov["extractor"] = "DarkTriadAnalyzer"
                elif source_key == "contagion":
                    prov["extractor"] = "ContagionAnalyzer"
                elif source_key == "volatility":
                    prov["extractor"] = "EmotionVolatilityAnalyzer"

                # Link evidence items from this user's text-level results.
                try:
                    if psycho_evidence is not None and i < len(psycho_evidence):
                        user_ev = psycho_evidence[i]
                        items = user_ev.get(source_key, [])
                        if items:
                            text_indices: list = []
                            evidence_ids: list = []
                            for item in items:
                                ti = item.get("text_index")
                                if ti is not None:
                                    text_indices.append(int(ti))
                                    # Synthetic stable evidence id when we
                                    # cannot resolve a real post_id from DB.
                                    evidence_ids.append(
                                        f"text:{uid_str}:{ti}"
                                    )
                            prov["text_indices"] = text_indices
                            prov["evidence_ids"] = evidence_ids
                except Exception:
                    pass

                entry[name] = prov

            provenance[uid_str] = entry

        return user_ids, fused_matrix, provenance
