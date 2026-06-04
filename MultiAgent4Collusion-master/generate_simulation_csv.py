"""
Generate simulation input CSV from deeppersonal_agents.json
"""
import json
import os
import random
import pandas as pd
from datetime import datetime

# ========== Configuration ==========
NUM_BAD_LEADER = 1
NUM_BAD_MEMBER = 1
NUM_BAD = 0
CREATED_AT = "2026-04-20 15:06:01+00:00"
FOLLOW_EDGE_PROB = 0.15         # Probability of generating a follow edge
RANDOM_SEED = 42
NUM_TWEETS_PER_AGENT = 5        # Initial number of tweets per agent
# =================================

random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  

DEEPERSONAL_PATH = os.path.join(PROJECT_ROOT, "deeppersona_ai",
                                "deeppersonal_agents.json")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "our_twitter_sim")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "False_Business_0.csv")

TWEETS_DIR = os.path.join(SCRIPT_DIR, "data", "tweets")


def load_tweet_pool():
    """Load tweet pool from data/tweets/"""
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

    print(f"  Tweet pool loaded: good={len(tweet_pool['good'])}, bad={len(tweet_pool['bad'])}")
    return tweet_pool


def get_previous_tweets(user_type, tweet_pool, num=NUM_TWEETS_PER_AGENT):
    """Sample from the corresponding tweet pool based on user_type"""
    pool_key = "bad" if "bad" in user_type else "good"
    pool = tweet_pool.get(pool_key, [])
    if len(pool) >= num:
        return random.sample(pool, num)
    return pool[:]


def generate_activity():
    """Generate 24-hour activity frequency (uniform 0~1)"""
    freq = [round(random.uniform(0, 1), 3) for _ in range(24)]
    labels = ["active" if f >= 0.15 else "inactive" for f in freq]
    return freq, labels


def generate_follow_graph(num_agents):
    """Generate random directed follow graph between agents"""
    following_ids = []
    for i in range(num_agents):
        followers = [
            j for j in range(num_agents)
            if i != j and random.random() < FOLLOW_EDGE_PROB
        ]
        following_ids.append(followers)
    return following_ids


def main():
    if not os.path.exists(DEEPERSONAL_PATH):
        print(f"[ERROR] Deep persona file not found: {DEEPERSONAL_PATH}")
        return
    with open(DEEPERSONAL_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"Loaded {len(profiles)} deep persona profiles")

    tweet_pool = load_tweet_pool()

    n = len(profiles)
    total_bad = NUM_BAD_LEADER + NUM_BAD_MEMBER + NUM_BAD
    if n < total_bad:
        print(f"[ERROR] JSON only has {n} agents, insufficient to assign {total_bad} bad agents")
        return
    user_type_list = (
        ["good"] * (n - total_bad)
        + ["bad_leader"] * NUM_BAD_LEADER
        + ["bad_member"] * NUM_BAD_MEMBER
        + ["bad"] * NUM_BAD
    )
    random.shuffle(user_type_list)

    following_ids = generate_follow_graph(n)

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

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nCSV generated: {OUTPUT_CSV}")
    print(f"  Total agents: {len(df)}")
    print(f"  Distribution: {dict(df['user_type'].value_counts())}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()