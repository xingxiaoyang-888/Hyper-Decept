"""
Generate simulation input CSV from deeppersonal_agents.json
"""
import json
import os
import random
import argparse
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


def validate_profiles(profiles):
    """Validate the DeepPersona-to-OASIS contract."""

    if not isinstance(profiles, list) or not profiles:
        raise ValueError(
            "Deep persona input must be a non-empty JSON list"
        )
    invalid_profiles = [
        idx for idx, profile in enumerate(profiles)
        if not isinstance(profile, dict)
        or not isinstance(profile.get("Summary"), str)
        or not profile["Summary"].strip()
    ]
    if invalid_profiles:
        raise ValueError(
            "Every deep persona must contain a non-empty string Summary; "
            f"invalid profile indexes: {invalid_profiles}"
        )


def build_agent_dataframe(profiles, tweet_pool):
    """Convert validated DeepPersona profiles to the OASIS CSV schema."""

    validate_profiles(profiles)
    n = len(profiles)
    total_bad = NUM_BAD_LEADER + NUM_BAD_MEMBER + NUM_BAD
    if n < total_bad:
        raise ValueError(
            f"JSON only has {n} agents, insufficient to assign "
            f"{total_bad} bad agents"
        )
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
        summary = profile["Summary"].strip()

        activity_freq, activity_labels = generate_activity()
        follow_list = following_ids[idx]
        followers_count = sum(1 for fl in following_ids if idx in fl)

        previous_tweets = get_previous_tweets(user_type, tweet_pool)

        rows.append({
            "user_id": idx,
            "name": name,
            "username": username,
            # OASIS uses user_char in the agent prompt but uses the database
            # bio/description in personalized recommendation.  Store the
            # same DeepPersona summary in both places so the persona reaches
            # the complete simulation chain instead of only the LLM prompt.
            "description": summary,
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

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible OASIS population CSV from DeepPersona profiles."
    )
    parser.add_argument("--profiles", default=DEEPERSONAL_PATH)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--num-bad-leader", type=int, default=NUM_BAD_LEADER)
    parser.add_argument("--num-bad-member", type=int, default=NUM_BAD_MEMBER)
    parser.add_argument("--num-bad", type=int, default=NUM_BAD)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    global NUM_BAD_LEADER, NUM_BAD_MEMBER, NUM_BAD
    NUM_BAD_LEADER = args.num_bad_leader
    NUM_BAD_MEMBER = args.num_bad_member
    NUM_BAD = args.num_bad
    random.seed(args.seed)
    if min(NUM_BAD_LEADER, NUM_BAD_MEMBER, NUM_BAD) < 0:
        raise ValueError("bad-agent counts must be non-negative")
    if not os.path.exists(args.profiles):
        raise FileNotFoundError(
            f"Deep persona file not found: {args.profiles}"
        )
    with open(args.profiles, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if args.num_agents is not None:
        if args.num_agents <= 0:
            raise ValueError("num-agents must be positive")
        if len(profiles) < args.num_agents:
            raise ValueError(
                f"requested {args.num_agents} agents but only {len(profiles)} profiles exist"
            )
        # Preserve an already-sized, explicitly indexed population so the
        # CSV row/agent IDs stay aligned with DeepPersona RAG chunks.  Only a
        # larger source pool needs deterministic sampling.
        if len(profiles) > args.num_agents:
            profiles = random.sample(profiles, args.num_agents)
    validate_profiles(profiles)
    print(f"Loaded {len(profiles)} deep persona profiles")

    tweet_pool = load_tweet_pool()
    df = build_agent_dataframe(profiles, tweet_pool)
    output_csv = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nCSV generated: {output_csv}")
    print(f"  Total agents: {len(df)}")
    print(f"  Distribution: {dict(df['user_type'].value_counts())}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
