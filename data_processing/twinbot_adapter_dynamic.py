"""
TwiBot-22 Adapter Dynamic (Small-Sample Open-World Edition)

基于 TwiBot-22 Adapter V5.3，只新增“样本数量控制”：
1. 支持按总核心用户数 total_sample_size 随机抽样，适合生成小样本实验库。
2. 保留原 sample_size 语义：每类各抽 sample_size 个核心用户。
3. 输出文件名自动携带实际样本数，如 twibot_200_v5.db。

原始处理逻辑保持不变：
1. [I/O 极速化] 单次遍历 edge.csv，避免 6.5GB 文件的二次 I/O。
2. [防爆截断] 引入入度边 (In-degree) 的随机采样，彻底防止大V节点引发边数爆炸。
3. [特征防爆] 对拼接后的 previous_tweets 实施总长二次截断 (如 5000 字符)。
4. [内存安全] SQLite 分批执行 (Chunked Insert)，稳定内存水位。
5. [时序保护] 交互动作采样后保持原始相对顺序。
"""

import os
import argparse
import random
import sqlite3
import pandas as pd
import ijson
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TwiBotAdapterDynamic:
    def __init__(self, twibot_dir, output_dir, 
                 sample_size=500,
                 total_sample_size=None,
                 max_actions=50, 
                 max_follows=100, 
                 max_chars_per_tweet=280, 
                 max_total_tweet_chars=5000, # [改进6] 拼接文本总长控制
                 fetch_boundary_profiles=False,
                 output_tag=None,
                 random_seed=42):
        self.twibot_dir = twibot_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.total_sample_size = total_sample_size
        self.max_actions = max_actions   
        self.max_follows = max_follows   
        self.max_chars = max_chars_per_tweet 
        self.max_total_chars = max_total_tweet_chars
        self.fetch_boundary_profiles = fetch_boundary_profiles
        self.output_tag = output_tag
        self.rng = random.Random(random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def _normalize_id(self, value):
        if pd.isna(value) or not str(value).strip():
            return None
        return str(value).strip()

    def _step1_sample_core_users(self):
        logger.info("🌍 [Dynamic] 步骤 1: 安全采样核心锚点用户...")
        label_path = os.path.join(self.twibot_dir, "label.csv")
        df_label = pd.read_csv(label_path)

        humans = df_label[df_label["label"] == "human"]["id"].astype(str).tolist()
        bots = df_label[df_label["label"] == "bot"]["id"].astype(str).tolist()

        if self.total_sample_size is not None:
            actual_human, actual_bot = self._balanced_sample_counts(
                int(self.total_sample_size),
                len(humans),
                len(bots),
            )
        else:
            actual_human = min(int(self.sample_size), len(humans))
            actual_bot = min(int(self.sample_size), len(bots))
        
        requested = self.total_sample_size if self.total_sample_size is not None else int(self.sample_size) * 2
        actual_total = actual_human + actual_bot
        if actual_total < requested:
            logger.warning(
                f"   ⚠️ 警告: 实际可采样数不足。Requested={requested}, "
                f"Actual={actual_total}, Humans={actual_human}, Bots={actual_bot}"
            )
        logger.info(f"   -> 核心用户采样规模: total={actual_total}, humans={actual_human}, bots={actual_bot}")

        sampled_humans = self.rng.sample(humans, actual_human)
        sampled_bots = self.rng.sample(bots, actual_bot)

        core_users = set(sampled_humans + sampled_bots)
        return core_users, set(sampled_bots)

    def _balanced_sample_counts(self, total_sample_size, num_humans, num_bots):
        if total_sample_size <= 0:
            raise ValueError("total_sample_size 必须大于 0。")

        target_human = total_sample_size // 2 + total_sample_size % 2
        target_bot = total_sample_size // 2

        actual_human = min(target_human, num_humans)
        actual_bot = min(target_bot, num_bots)

        remaining = total_sample_size - actual_human - actual_bot
        if remaining > 0:
            human_capacity = max(num_humans - actual_human, 0)
            add_human = min(remaining, human_capacity)
            actual_human += add_human
            remaining -= add_human

        if remaining > 0:
            bot_capacity = max(num_bots - actual_bot, 0)
            add_bot = min(remaining, bot_capacity)
            actual_bot += add_bot

        return actual_human, actual_bot

    def _step2_extract_open_topology(self, core_users):
        logger.info("🌍 [Dynamic] 步骤 2: 单次遍历 edge.csv，收集出入度与交互行为...")
        edge_path = os.path.join(self.twibot_dir, "edge.csv")
        
        # [改进2] 使用字典在单次遍历中收集所有候选
        raw_follow_out = defaultdict(list)
        raw_follow_in = defaultdict(list)
        raw_actions = defaultdict(list)
        
        chunk_size = 1_000_000
        for chunk in pd.read_csv(edge_path, chunksize=chunk_size):
            # 1. 提取 Follow 边
            follow_mask = chunk["relation"].isin(["following", "followers", "follow"])
            for _, row in chunk[follow_mask].iterrows():
                src = self._normalize_id(row["source_id"])
                tgt = self._normalize_id(row["target_id"])
                
                # 核心向外关注 (出度)
                if src in core_users:
                    raw_follow_out[src].append(tgt)
                # 外部关注核心 (入度) [改进1：分离收集以便后续安全截断]
                if tgt in core_users and src not in core_users:
                    raw_follow_in[tgt].append(src)

            # 2. 提取 Action 边
            action_mask = chunk["relation"].isin(["post", "reply", "retweet", "like"])
            for _, row in chunk[action_mask].iterrows():
                src = self._normalize_id(row["source_id"])
                if src in core_users:
                    action = str(row["relation"]).lower()
                    tweet_id = self._normalize_id(row["target_id"])
                    raw_actions[src].append((action, tweet_id))

        # --- 执行无偏随机采样 ---
        logger.info("   -> 正在执行无偏图采样与度数防爆截断...")
        follow_edges = []
        boundary_users = set()
        
        # 处理出度
        for src, tgts in raw_follow_out.items():
            sampled = self.rng.sample(tgts, min(self.max_follows, len(tgts)))
            for tgt in sampled:
                follow_edges.append((src, tgt))
                if tgt not in core_users: boundary_users.add(tgt)
                
        # 处理入度 [改进1：防止大V百万粉丝撑爆内存]
        for tgt, srcs in raw_follow_in.items():
            sampled = self.rng.sample(srcs, min(self.max_follows, len(srcs)))
            for src in sampled:
                follow_edges.append((src, tgt))
                boundary_users.add(src)

        # 处理交互动作 [改进7：保持时序连续性，不打乱原有顺序]
        user_tweet_edges = []
        target_tweets = set()
        for src, actions in raw_actions.items():
            # 获取随机采样的索引，排序后按原顺序提取
            k = min(self.max_actions, len(actions))
            sampled_indices = sorted(self.rng.sample(range(len(actions)), k))
            for idx in sampled_indices:
                action, tweet_id = actions[idx]
                user_tweet_edges.append((src, action, tweet_id))
                target_tweets.add(tweet_id)

        logger.info(f"   🕸️ 采样完成: 边数 Follow={len(follow_edges)}, Action={len(user_tweet_edges)}")
        logger.info(f"   🕸️ 边界节点数(Boundary Nodes): {len(boundary_users)}")
        return follow_edges, user_tweet_edges, target_tweets, boundary_users

    def _step3_extract_metadata(self, core_users, boundary_users, target_tweets):
        logger.info("🌍 [Dynamic] 步骤 3: 解析 JSON，提取实体画像与内容...")
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
            
            logger.info(f"   -> 正在检索 {os.path.basename(tweet_file)} (进度: {len(tweet_content)}/{total_targets})")
            
            start_size = len(tweet_content)
            with open(tweet_file, "r", encoding="utf-8") as f:
                for tweet in ijson.items(f, "item"):
                    tid = self._normalize_id(tweet.get("id"))
                    if tid in target_tweets:
                        text = str(tweet.get("text", "") or "").replace("\n", " ").replace("\r", " ")
                        tweet_content[tid] = text[:self.max_chars] 
                        
            if len(tweet_content) == start_size:
                logger.debug(f"      此文件未贡献新推文。")
                
        return user_profiles, user_metrics, tweet_content

    def _batch_insert(self, cursor, query, data_list, batch_size=10000):
        """[改进8] 内存安全的 SQLite 分批写入工具"""
        for i in range(0, len(data_list), batch_size):
            cursor.executemany(query, data_list[i:i + batch_size])

    def _step4_build_outputs(self, core_users, boundary_users, sampled_bots, follow_edges, user_tweet_edges, user_profiles, user_metrics, tweet_content):
        logger.info("🌍 [Dynamic] 步骤 4: 构建 CSV 与 SQLite, 并建立高频索引...")
        
        sample_tag = self.output_tag or str(len(core_users))
        csv_path = os.path.join(self.output_dir, f"twibot_{sample_tag}_multimodal_v5.csv")
        db_path = os.path.join(self.output_dir, f"twibot_{sample_tag}_v5.db")
        
        # --- 1. 构建 CSV (严格核心) ---
        user_texts = defaultdict(list)
        for src, action, tid in user_tweet_edges:
            if action in ['post', 'retweet'] and tid in tweet_content:
                user_texts[src].append(tweet_content[tid])
                
        csv_rows = []
        for uid in core_users:
            # [改进6] 拼接文本防爆，二次截断
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
        logger.info(f"   ✅ CSV (监督信号层) 已生成.")

        # --- 2. 构建 SQLite (全图拓扑) ---
        if os.path.exists(db_path): os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("CREATE TABLE user (user_id TEXT, user_type TEXT, followers INTEGER, following INTEGER)")
        cursor.execute("CREATE TABLE follow (follower_id TEXT, followee_id TEXT)")
        cursor.execute("CREATE TABLE agent_actions (agent_name TEXT, action_type TEXT, content TEXT)")
        
        # [学术注释：GNN 注意事项] user_type 仅用于 Mask 分片，勿作为特征输入
        all_user_records = []
        for uid in core_users.union(boundary_users):
            is_core = uid in core_users
            u_type = "bad" if (is_core and uid in sampled_bots) else ("good" if is_core else "boundary")
            
            fol = user_metrics.get(uid, {}).get("followers", 0)
            fwi = user_metrics.get(uid, {}).get("following", 0)
            all_user_records.append((uid, u_type, fol, fwi))
            
        # 使用分批写入防止 OOM
        self._batch_insert(cursor, "INSERT INTO user VALUES (?, ?, ?, ?)", all_user_records)
        self._batch_insert(cursor, "INSERT INTO follow VALUES (?, ?)", follow_edges)
        
        action_records = [(src, action, tweet_content[tid]) for src, action, tid in user_tweet_edges if tid in tweet_content]
        self._batch_insert(cursor, "INSERT INTO agent_actions VALUES (?, ?, ?)", action_records)
        
        logger.info("   ⚡ 正在构建 SQLite 高频查询索引...")
        cursor.execute("CREATE INDEX idx_user_id ON user(user_id)")
        cursor.execute("CREATE INDEX idx_follow_follower ON follow(follower_id)")
        cursor.execute("CREATE INDEX idx_agent_actions_agent ON agent_actions(agent_name)")
        
        conn.commit()
        conn.close()
        logger.info(f"   ✅ DB (全景拓扑图) 已生成，索引构建完毕！")

    def extract_and_convert(self):
        print("\n" + "="*70)
        print("🚀 [TwiBot-22 数据底座 Dynamic 小样本版] 终极启动")
        print("="*70)
        
        core_users, sampled_bots = self._step1_sample_core_users()
        follow_edges, user_tweet_edges, target_tweets, boundary_users = self._step2_extract_open_topology(core_users)
        user_profiles, user_metrics, tweet_content = self._step3_extract_metadata(core_users, boundary_users, target_tweets)
        self._step4_build_outputs(core_users, boundary_users, sampled_bots, follow_edges, user_tweet_edges, user_profiles, user_metrics, tweet_content)
        
        print("\n🎉 Dynamic 小样本数据库生成完成。")


TwiBotAdapterV5_3 = TwiBotAdapterDynamic


def parse_args():
    parser = argparse.ArgumentParser(description="从 TwiBot-22 真实数据集中随机抽样生成小样本实验数据库")
    parser.add_argument("--twibot-dir", default=r"D:\Github\Multi-AFG-Detection\data\tweets")
    parser.add_argument("--output-dir", default=r"D:\Github\Multi-AFG-Detection\data")
    parser.add_argument("--total-sample-size", type=int, default=1000, help="核心用户总数，按 human/bot 尽量均衡抽样。")
    parser.add_argument("--per-class-sample-size", type=int, default=None, help="每类各抽多少个；设置后优先于 total-sample-size。")
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--max-follows", type=int, default=100)
    parser.add_argument("--max-chars-per-tweet", type=int, default=280)
    parser.add_argument("--max-total-tweet-chars", type=int, default=5000)
    parser.add_argument("--fetch-boundary-profiles", action="store_true")
    parser.add_argument("--output-tag", default=None, help="输出文件名标签；默认使用实际核心用户数。")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.per_class_sample_size is not None:
        total_sample_size = None
        sample_size = args.per_class_sample_size
    else:
        total_sample_size = args.total_sample_size
        sample_size = max(args.total_sample_size // 2, 1)

    adapter = TwiBotAdapterDynamic(
        args.twibot_dir,
        args.output_dir,
        sample_size=sample_size,
        total_sample_size=total_sample_size,
        max_actions=args.max_actions,
        max_follows=args.max_follows,
        max_chars_per_tweet=args.max_chars_per_tweet,
        max_total_tweet_chars=args.max_total_tweet_chars,
        fetch_boundary_profiles=args.fetch_boundary_profiles,
        output_tag=args.output_tag,
        random_seed=args.random_seed,
    )
    adapter.extract_and_convert()
