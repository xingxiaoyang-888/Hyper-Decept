import hashlib
import contextlib
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

# ==========================================
# 核心架构层：导入四大重型社会心理学引擎
# ==========================================
try:
    from emotional_analysis.empathy_gap_analyzer import EmpathyGapAnalyzer
    from emotional_analysis.dark_triad_analyzer import DarkTriadAnalyzer
    from emotional_analysis.contagion_analyzer import ContagionAnalyzer
    from emotional_analysis.volatility_analyzer import EmotionVolatilityAnalyzer
except ImportError as e:
    warnings.warn(f"无法完整导入重型情感引擎，请检查 emotional_analysis 目录: {e}")
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
        """
        初始化异构多模态特征融合枢纽 (中央总线 终极大一统版)
        """
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

        print(f"[Feature Bus] 启动，加载双流文本基础语义模型: {model_path} ...")
        print(f"[Feature Bus] psychology mode: {self.psychology_mode}")
        if self.max_tweets_per_user is not None:
            print(f"[Feature Bus] max tweets per user: {self.max_tweets_per_user}")
        if not self.verbose_progress:
            print("[Feature Bus] internal progress bars: quiet (set AFG_VERBOSE_PROGRESS=1 to show)")
        
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.text_model = SentenceTransformer(model_path, local_files_only=True)
            except Exception as local_exc:
                warnings.warn(f"Local sentence_transformers load failed, trying default loader: {local_exc}")
                self.text_model = SentenceTransformer(model_path)
        except Exception:
            warnings.warn("无法加载 sentence_transformers。将使用零向量占位。")
            self.text_model = None
            
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=aligned_text_dim, random_state=42) 
        self.aligned_text_dim = aligned_text_dim
        
        # ==========================================
        # 挂载四大 LLM 原生社会心理学重型引擎
        # ==========================================
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
        """提取数据库中的统计行为特征 (兼容多表结构)"""
        print(f"[Behavioral Stream] 正在连接数据库: {os.path.basename(db_path)}")
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

            # ---------- 提取 user 表特征 ----------
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

            # ---------- 从 post 表聚合每用户发文特征 ----------
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

            # ---------- 无 post 表时，从 agent_actions 取同维度代理统计 ----------
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
        """双流文本池化编码 (PCA 降维)"""
        print(f"📝 [Semantic Stream] 启动双流文本编码 (1.静态 Bio | 2.动态推文池化)...")
        if self.text_model is None:
            return np.zeros((len(bios), self.aligned_text_dim))

        bio_embs = np.asarray(self.text_model.encode(bios, show_progress_bar=False))
        emb_dim = bio_embs.shape[1] if bio_embs.ndim == 2 else self.aligned_text_dim

        tweet_groups = [split_tweet_pool(tweets_str, min_len=2) for tweets_str in tweets_list]
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

    def _extract_llm_native_psychology(self, tweets_list: list):
        """
        【系统核心】全火力展开：并发调用四大重型引擎，抽取 8 维宏观心理特征
        """
        print(f" [Psychological Stream] 启动深度心理扫描，处理 {len(tweets_list)} 个智能体...")
        agent_tweets = [split_tweet_pool(tweets_str, min_len=5) for tweets_str in tweets_list]
        if self.max_tweets_per_user is not None:
            agent_tweets = [texts[:self.max_tweets_per_user] for texts in agent_tweets]

        if self.psychology_mode == "off":
            print(" [Psychological Stream] off mode: returning zero psychology features.")
            return np.zeros((len(tweets_list), 8), dtype=float)

        cache_path = self._psychology_cache_path(agent_tweets)
        if cache_path and os.path.exists(cache_path):
            print(f" [Psychological Stream] hit feature cache: {cache_path}")
            return np.load(cache_path)

        psycho_features = []
        
        for agent_tweets_array in tqdm(
            agent_tweets,
            desc="Deep Psycho-Scan",
            unit="agent",
            disable=not self.verbose_progress,
        ):
            # 1. 认知错位差 (Empathy Gap)
            if self.empathy_engine and agent_tweets_array:
                emp_res = self._run_engine(self.empathy_engine.evaluate_agent, agent_tweets_array)
                emp_mean = emp_res.get("Agent_Mean_Empathy_Gap", 0.0)
                emp_max = emp_res.get("Agent_Max_Empathy_Gap", 0.0)
            else:
                emp_mean, emp_max = 0.0, 0.0
                
            # 2. 暗黑三角激活度 (Dark Triad)
            if self.dark_triad_engine and agent_tweets_array:
                dt_res = self._run_engine(self.dark_triad_engine.evaluate_agent, agent_tweets_array)
                dt_mean = dt_res.get("Agent_Mean_Dark_Triad", 0.0)
                dt_max = dt_res.get("Agent_Max_Dark_Triad", 0.0)
            else:
                dt_mean, dt_max = 0.0, 0.0
                
            # 3. 无阻尼情感传染率 (Frictionless Contagion)
            # 默认降级为锚点模式计算基础均值与极值
            if self.contagion_engine and agent_tweets_array:
                cont_res = self._run_engine(self.contagion_engine.evaluate_agent, agent_tweets_array)
                cont_mean = cont_res.get("Agent_Mean_Alignment", 0.0)
                cont_max = cont_res.get("Agent_Contagion_Spike", 0.0)
                if len(psycho_features) == 0 and cont_mean > 0:
                    print(f"[DEBUG] Contagion 首个非零值: mean={cont_mean:.4f}, max={cont_max:.4f}")
            else:
                cont_mean, cont_max = 0.0, 0.0
                
            # 4. 功能性情感瞬切率 (Emotion Volatility)
            if self.volatility_engine and agent_tweets_array:
                vol_res = self._run_engine(self.volatility_engine.evaluate_agent, agent_tweets_array)
                # 检查数据是否充分 (大于等于两条文本才构成“瞬切”)
                if not vol_res.get("Insufficient_Data", True):
                    vol_mean = vol_res.get("Agent_Mean_Volatility", 0.0)
                    vol_max = vol_res.get("Agent_Max_Volatility", 0.0)
                else:
                    vol_mean, vol_max = 0.0, 0.0
            else:
                vol_mean, vol_max = 0.0, 0.0
            
            # 组装 8 维心理特征超级向量
            psycho_features.append([
                emp_mean, emp_max,   # 维度 1, 2: 错位差 (均值/极值)
                dt_mean, dt_max,     # 维度 3, 4: 暗黑三角 (均值/极值)
                cont_mean, cont_max, # 维度 5, 6: 传染率 (均值/极值)
                vol_mean, vol_max    # 维度 7, 8: 瞬切率 (均值/极值)
            ])
            
        psycho_matrix = np.array(psycho_features)
        if cache_path:
            np.save(cache_path, psycho_matrix)
            print(f" [Psychological Stream] feature cache saved: {cache_path}")
        return psycho_matrix

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
        return os.path.join(self.cache_dir, f"psychology_{digest.hexdigest()[:24]}.npy")

    def fuse_multimodal_features(self, db_path: str, bios: list, tweets_list: list = None, user_ids_master: list = None):
        """异构多模态融合枢纽"""
        print("\n" + "="*50)
        print("🔗 启动多模态特征融合总线 (Feature Bus V8.0)")
        print("="*50)
        
        # 1. 获取行为学特征矩阵
        user_ids, behavior_matrix = self.extract_behavior_features(db_path, user_ids_master)
        
        # 2. 文本字段预处理兼容
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

        # 3. 获取双流语义 PCA 矩阵
        aligned_text_matrix = self._encode_text_dual_stream(bios_input, tweets_input)
        
        # 4. 获取 8 维深度心理学矩阵
        psycho_matrix = self._extract_llm_native_psychology(tweets_input)

        min_len = min(len(user_ids), len(aligned_text_matrix), len(behavior_matrix), len(psycho_matrix))
        if min_len < len(user_ids) or min_len < len(aligned_text_matrix):
            warnings.warn(f"Input modality lengths differ; aligned to shortest length {min_len}.")
            user_ids = user_ids[:min_len]
            aligned_text_matrix = aligned_text_matrix[:min_len]
            behavior_matrix = behavior_matrix[:min_len]
            psycho_matrix = psycho_matrix[:min_len]
        
        # 5. 标准化融合 (仅对行为和心理指标进行标准化消除量纲)
        enhanced_behavior = np.hstack((behavior_matrix, psycho_matrix))
        behavior_matrix_scaled = self.scaler.fit_transform(enhanced_behavior)
        
        # 6. 终极超级矩阵构建
        print(" 正在拼接对齐后的语义向量、行为统计与【8维深度认知心理】特征...")
        fused_matrix = np.hstack((aligned_text_matrix, behavior_matrix_scaled))
        
        print(f"✅ 多模态超级矩阵构建完毕: {fused_matrix.shape} (降维文本 + 行为 + 8维心理侧写)")
        return user_ids, fused_matrix
