import os
import random
import sqlite3
import pandas as pd
import ijson
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TwiBotAdapterV5_3:
    def __init__(self, twibot_dir, output_dir, 
                 sample_size=500, 
                 max_actions=50, 
                 max_follows=100, 
                 max_chars_per_tweet=280, 
                 max_total_tweet_chars=5000, 
                 fetch_boundary_profiles=False,
                 random_seed=42):
        self.twibot_dir = twibot_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.max_actions = max_actions   
        self.max_follows = max_follows   
        self.max_chars = max_chars_per_tweet 
        self.max_total_chars = max_total_tweet_chars
        self.fetch_boundary_profiles = fetch_boundary_profiles
        self.rng = random.Random(random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def _normalize_id(self, value):
        if pd.isna(value) or not str(value).strip():
            return None
        return str(value).strip()

    def _step1_sample_core_users(self):
        logger.info(" sampling ...")
        label_path = os.path.join(self.twibot_dir, "label.csv")
        df_label = pd.read_csv(label_path)

        humans = df_label[df_label["label"] == "human"]["id"].astype(str).tolist()
        bots = df_label[df_label["label"] == "bot"]["id"].astype(str).tolist()

        actual_human = min(self.sample_size, len(humans))
        actual_bot = min(self.sample_size, len(bots))
        
        if actual_human < self.sample_size or actual_bot < self.sample_size:
            logger.warning(f"Insufficient actual samples available. Humans: {actual_human}, Bots: {actual_bot}")

        sampled_humans = self.rng.sample(humans, actual_human)
        sampled_bots = self.rng.sample(bots, actual_bot)

        core_users = set(sampled_humans + sampled_bots)
        return core_users, set(sampled_bots)

    def _step2_extract_open_topology(self, core_users):
        logger.info(" Single pass through edge.csv, collecting in/out degrees and interactions...")
        edge_path = os.path.join(self.twibot_dir, "edge.csv")
        
        raw_follow_out = defaultdict(list)
        raw_follow_in = defaultdict(list)
        raw_actions = defaultdict(list)
        
        chunk_size = 1_000_000
        for chunk in pd.read_csv(edge_path, chunksize=chunk_size):
            follow_mask = chunk["relation"].isin(["following", "followers", "follow"])
            for _, row in chunk[follow_mask].iterrows():
                src = self._normalize_id(row["source_id"])
                tgt = self._normalize_id(row["target_id"])
                
                if src in core_users:
                    raw_follow_out[src].append(tgt)
                if tgt in core_users and src not in core_users:
                    raw_follow_in[tgt].append(src)

            action_mask = chunk["relation"].isin(["post", "reply", "retweet", "like"])
            for _, row in chunk[action_mask].iterrows():
                src = self._normalize_id(row["source_id"])
                if src in core_users:
                    action = str(row["relation"]).lower()
                    tweet_id = self._normalize_id(row["target_id"])
                    raw_actions[src].append((action, tweet_id))

        logger.info("Executing unbiased graph sampling and degree truncation...")
        follow_edges = []
        boundary_users = set()
        
        for src, tgts in raw_follow_out.items():
            sampled = self.rng.sample(tgts, min(self.max_follows, len(tgts)))
            for tgt in sampled:
                follow_edges.append((src, tgt))
                if tgt not in core_users: boundary_users.add(tgt)
                
        for tgt, srcs in raw_follow_in.items():
            sampled = self.rng.sample(srcs, min(self.max_follows, len(srcs)))
            for src in sampled:
                follow_edges.append((src, tgt))
                boundary_users.add(src)

        user_tweet_edges = []
        target_tweets = set()
        for src, actions in raw_actions.items():
            k = min(self.max_actions, len(actions))
            sampled_indices = sorted(self.rng.sample(range(len(actions)), k))
            for idx in sampled_indices:
                action, tweet_id = actions[idx]
                user_tweet_edges.append((src, action, tweet_id))
                target_tweets.add(tweet_id)

        logger.info(f"Sampling complete: Follow edges={len(follow_edges)}, Action edges={len(user_tweet_edges)}")
        logger.info(f"Boundary Nodes: {len(boundary_users)}")
        return follow_edges, user_tweet_edges, target_tweets, boundary_users

    def _step3_extract_metadata(self, core_users, boundary_users, target_tweets):
        logger.info("Parsing JSON, extracting profiles and content...")
        user_profiles, user_metrics = {}, {}
        tweet_content = {}
        
        users_to_fetch = core_users.union(boundary_users) if self.fetch_boundary_profiles else core_users
        
        user_file = os.path.join(self.twibot_dir, "user.json")
        if os.path.exists(user_file):
            with open(user_file, "r", encoding="utf-8") as f:
                for user in ijson.items(f, "item"):
                    uid = self._normalize_id(user.get("id"))
                    if uid in users_to_fetch:
                        metrics = user.get("public_metrics", {}) or {}
                        desc = str(user.get("description", "") or "").replace("\n", " ")
                        user_profiles[uid] = desc[:self.max_chars]
                        user_metrics[uid] = {
                            "followers": int(metrics.get("followers_count", 0) or 0),
                            "following": int(metrics.get("following_count", 0) or 0)
                        }

        total_targets = len(target_tweets)
        for i in range(9):
            if len(tweet_content) >= total_targets: break
                
            tweet_file = os.path.join(self.twibot_dir, f"tweet_{i}.json")
            if not os.path.exists(tweet_file): continue
            
            logger.info(f"Retrieving {os.path.basename(tweet_file)} (Progress: {len(tweet_content)}/{total_targets})")
            
            start_size = len(tweet_content)
            with open(tweet_file, "r", encoding="utf-8") as f:
                for tweet in ijson.items(f, "item"):
                    tid = self._normalize_id(tweet.get("id"))
                    if tid in target_tweets:
                        text = str(tweet.get("text", "") or "").replace("\n", " ").replace("\r", " ")
                        tweet_content[tid] = text[:self.max_chars] 
                        
            if len(tweet_content) == start_size:
                logger.debug("This file contributed no new tweets.")
                
        return user_profiles, user_metrics, tweet_content

    def _batch_insert(self, cursor, query, data_list, batch_size=10000):
        for i in range(0, len(data_list), batch_size):
            cursor.executemany(query, data_list[i:i + batch_size])

    def _step4_build_outputs(self, core_users, boundary_users, sampled_bots, follow_edges, user_tweet_edges, user_profiles, user_metrics, tweet_content):
        logger.info("Building CSV and SQLite, establishing high-frequency indices...")
        
        csv_path = os.path.join(self.output_dir, "twibot_1000_multimodal_v5.csv")
        db_path = os.path.join(self.output_dir, "twibot_1000_v5.db")
        
        user_texts = defaultdict(list)
        for src, action, tid in user_tweet_edges:
            if action in ['post', 'retweet'] and tid in tweet_content:
                user_texts[src].append(tweet_content[tid])
                
        csv_rows = []
        for uid in core_users:
            combined_tweets = " | ".join(user_texts[uid])[:self.max_total_chars]
            
            csv_rows.append({
                "user_id": uid,
                "user_char": user_profiles.get(uid, ""),
                "followers_count": user_metrics.get(uid, {}).get("followers", 0),
                "following_count": user_metrics.get(uid, {}).get("following", 0),
                "previous_tweets": combined_tweets,
                "user_type": "bad" if uid in sampled_bots else "good",
            })
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        logger.info("CSV generated.")

        if os.path.exists(db_path): os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("CREATE TABLE user (user_id TEXT, user_type TEXT, followers INTEGER, following INTEGER)")
        cursor.execute("CREATE TABLE follow (follower_id TEXT, followee_id TEXT)")
        cursor.execute("CREATE TABLE agent_actions (agent_name TEXT, action_type TEXT, content TEXT)")
        
        all_user_records = []
        for uid in core_users.union(boundary_users):
            is_core = uid in core_users
            u_type = "bad" if (is_core and uid in sampled_bots) else ("good" if is_core else "boundary")
            
            fol = user_metrics.get(uid, {}).get("followers", 0)
            fwi = user_metrics.get(uid, {}).get("following", 0)
            all_user_records.append((uid, u_type, fol, fwi))
            
        self._batch_insert(cursor, "INSERT INTO user VALUES (?, ?, ?, ?)", all_user_records)
        self._batch_insert(cursor, "INSERT INTO follow VALUES (?, ?)", follow_edges)
        
        action_records = [(src, action, tweet_content[tid]) for src, action, tid in user_tweet_edges if tid in tweet_content]
        self._batch_insert(cursor, "INSERT INTO agent_actions VALUES (?, ?, ?)", action_records)
        
        logger.info("Building SQLite high-frequency query indices...")
        cursor.execute("CREATE INDEX idx_user_id ON user(user_id)")
        cursor.execute("CREATE INDEX idx_follow_follower ON follow(follower_id)")
        cursor.execute("CREATE INDEX idx_agent_actions_agent ON agent_actions(agent_name)")
        
        conn.commit()
        conn.close()
        logger.info("DB generated and indices built successfully.")

    def extract_and_convert(self):
        core_users, sampled_bots = self._step1_sample_core_users()
        follow_edges, user_tweet_edges, target_tweets, boundary_users = self._step2_extract_open_topology(core_users)
        user_profiles, user_metrics, tweet_content = self._step3_extract_metadata(core_users, boundary_users, target_tweets)
        self._step4_build_outputs(core_users, boundary_users, sampled_bots, follow_edges, user_tweet_edges, user_profiles, user_metrics, tweet_content)
if __name__ == "__main__":
    TWIBOT_DIR = r"D:\Github\Multi-AFG-Detection\data\tweets"
    OUTPUT_DIR = r"D:\Github\Multi-AFG-Detection\data"

    adapter = TwiBotAdapterV5_3(TWIBOT_DIR, OUTPUT_DIR, 
                                sample_size=500, max_actions=50, max_follows=100)
    adapter.extract_and_convert()