"""
从 deeppersonal_agents.json 生成仿真输入 CSV
"""
import json
import os
import random
import pandas as pd
from datetime import datetime

# ========== 配置区（按需修改）==========
NUM_BAD_LEADER = 1
NUM_BAD_MEMBER = 1
NUM_BAD = 0
CREATED_AT = "2026-04-20 15:06:01+00:00"
FOLLOW_EDGE_PROB = 0.15         # 关注图生成概率
RANDOM_SEED = 42
NUM_TWEETS_PER_AGENT = 5        # 每个 agent 初始推文数
# ======================================

random.seed(RANDOM_SEED)

# ---- 基于脚本位置的根目录计算（保证 repo 可移植） ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Hyper-Decept 根目录

# 输入：deeppersona_ai 生成的 JSON（位于项目根下）
DEEPERSONAL_PATH = os.path.join(PROJECT_ROOT, "deeppersona_ai",
                                "deeppersonal_agents.json")

# 输出：CSV 保存到 our_twitter_sim/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "our_twitter_sim")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "False_Business_0.csv")

# 推文 JSON 目录（用户已放置在 data/tweets/）
TWEETS_DIR = os.path.join(SCRIPT_DIR, "data", "tweets")


def load_tweet_pool():
    """从 data/tweets/ 加载推文池"""
    tweet_pool = {"good": [], "bad": []}
    good_files = ["real_tweets_COVID.json", "real_tweets_politics.json"]
    bad_files = ["fake_tweets_COVID.json", "fake_tweets_politics.json"]

    for fname in good_files:
        path = os.path.join(TWEETS_DIR, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                tweets = json.load(f)
                tweet_pool["good"].extend(tweets[:100])

    for fname in bad_files:
        path = os.path.join(TWEETS_DIR, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                tweets = json.load(f)
                tweet_pool["bad"].extend(tweets[:100])

    print(f"  推文池加载: good={len(tweet_pool['good'])}, bad={len(tweet_pool['bad'])}")
    return tweet_pool


def get_previous_tweets(user_type, tweet_pool, num=NUM_TWEETS_PER_AGENT):
    """根据 user_type 从对应推文池采样"""
    pool_key = "bad" if "bad" in user_type else "good"
    pool = tweet_pool.get(pool_key, [])
    if len(pool) >= num:
        return random.sample(pool, num)
    return pool[:]


def generate_activity():
    """生成 24 小时活动频率（uniform 0~1）"""
    freq = [round(random.uniform(0, 1), 3) for _ in range(24)]
    labels = ["active" if f >= 0.15 else "inactive" for f in freq]
    return freq, labels


def generate_follow_graph(num_agents):
    """在 agent 之间生成随机有向关注图"""
    following_ids = []
    for i in range(num_agents):
        followers = [
            j for j in range(num_agents)
            if i != j and random.random() < FOLLOW_EDGE_PROB
        ]
        following_ids.append(followers)
    return following_ids


def main():
    # ---- 读取深度人格 JSON ----
    if not os.path.exists(DEEPERSONAL_PATH):
        print(f"[ERROR] 未找到深度人格文件: {DEEPERSONAL_PATH}")
        return
    with open(DEEPERSONAL_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"读取 {len(profiles)} 个深度人格 profiles")

    # ---- 加载推文池 ----
    tweet_pool = load_tweet_pool()

    # ---- 分配 user_type ----
    n = len(profiles)
    total_bad = NUM_BAD_LEADER + NUM_BAD_MEMBER + NUM_BAD
    if n < total_bad:
        print(f"[ERROR] JSON 仅有 {n} 个 agent，不够分配 {total_bad} 个 bad")
        return
    user_type_list = (
        ["good"] * (n - total_bad)
        + ["bad_leader"] * NUM_BAD_LEADER
        + ["bad_member"] * NUM_BAD_MEMBER
        + ["bad"] * NUM_BAD
    )
    random.shuffle(user_type_list)

    # ---- 生成关注关系 ----
    following_ids = generate_follow_graph(n)

    # ---- 生成每行数据 ----
    rows = []
    for idx, profile in enumerate(profiles):
        user_type = user_type_list[idx]
        name = f"User_{idx + 1}"
        username = f"@User_{idx + 1}"
        summary = profile.get("Summary", "")

        activity_freq, activity_labels = generate_activity()
        follow_list = following_ids[idx]
        followers_count = sum(1 for fl in following_ids if idx in fl)

        previous_tweets = get_previous_tweets(user_type, tweet_pool)

        rows.append({
            "user_id": idx,
            "name": name,
            "username": username,
            "description": "",
            "created_at": CREATED_AT,
            "user_char": summary,
            "user_type": user_type,
            "followers_count": followers_count,
            "following_count": len(follow_list),
            "following_list": str(follow_list),
            "following_agentid_list": str(follow_list),
            "previous_tweets": str(previous_tweets),
            "tweets_id": str([]),
            "activity_level_frequency": str(activity_freq),
            "activity_level": str(activity_labels),
        })

    # ---- 输出 CSV ----
    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nCSV generated: {OUTPUT_CSV}")
    print(f"  总 agent: {len(df)}")
    print(f"  分布: {dict(df['user_type'].value_counts())}")
    print(f"  列: {list(df.columns)}")


if __name__ == "__main__":
    main()
