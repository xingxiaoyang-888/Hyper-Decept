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
SCENARIO_IDS = {
    "leader_amplifier",
    "bridge_infiltration",
    "synchronized_boosting",
    "persona_drift",
    "adaptive_evasion",
}
SCENARIO_ROLE_RATIOS = {
    "leader_amplifier": (0.005, 0.095, 0.0),
    "bridge_infiltration": (0.005, 0.055, 0.040),
    "synchronized_boosting": (0.005, 0.095, 0.0),
    "persona_drift": (0.005, 0.045, 0.050),
    "adaptive_evasion": (0.005, 0.095, 0.0),
}
# =================================

random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  

DEEPERSONAL_PATH = os.path.join(PROJECT_ROOT, "deeppersona_ai",
                                "deeppersonal_agents.json")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "our_twitter_sim")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "False_Business_0.csv")

TWEETS_DIR = os.path.join(SCRIPT_DIR, "data", "tweets")


def load_tweet_pool(pool_path=None):
    """Load tweet pool from data/tweets/"""
    if pool_path is not None:
        path = os.path.abspath(pool_path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("tweet pool must be a JSON object")
        tweet_pool = {}
        for label in ("good", "bad"):
            values = payload.get(label)
            if not isinstance(values, list):
                raise ValueError(f"tweet pool {label!r} must be a list")
            cleaned = [str(value).strip() for value in values if str(value).strip()]
            if not cleaned:
                raise ValueError(f"tweet pool {label!r} must not be empty")
            tweet_pool[label] = cleaned
        print(
            f"  Tweet pool loaded: good={len(tweet_pool['good'])}, "
            f"bad={len(tweet_pool['bad'])}"
        )
        return tweet_pool

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


def get_previous_tweets(user_type, tweet_pool, num=None):
    """Sample from the corresponding tweet pool based on user_type"""
    num = NUM_TWEETS_PER_AGENT if num is None else num
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


def _scenario_role_counts(scenario, num_agents):
    if scenario not in SCENARIO_IDS:
        raise ValueError(f"unsupported scenario: {scenario}")
    leader_ratio, member_ratio, bad_ratio = SCENARIO_ROLE_RATIOS[scenario]
    leaders = max(1, round(num_agents * leader_ratio))
    members = max(1, round(num_agents * member_ratio))
    bad = round(num_agents * bad_ratio)
    if leaders + members + bad >= num_agents:
        raise ValueError("scenario role ratios leave no organic agents")
    return leaders, members, bad


def generate_follow_graph(
    num_agents,
    user_types=None,
    scenario=None,
    mean_following=None,
):
    """Generate random directed follow graph between agents"""
    if mean_following is None:
        return [
            [
                target
                for target in range(num_agents)
                if source != target and random.random() < FOLLOW_EDGE_PROB
            ]
            for source in range(num_agents)
        ]
    if mean_following <= 0:
        raise ValueError("mean_following must be positive")
    if user_types is None or len(user_types) != num_agents:
        raise ValueError("formal sparse graph requires one user type per agent")
    leaders = [
        agent_id for agent_id, user_type in enumerate(user_types)
        if user_type == "bad_leader"
    ]
    members = [
        agent_id for agent_id, user_type in enumerate(user_types)
        if user_type == "bad_member"
    ]
    bad = [
        agent_id for agent_id, user_type in enumerate(user_types)
        if "bad" in user_type
    ]
    good = [
        agent_id for agent_id, user_type in enumerate(user_types)
        if user_type == "good"
    ]
    following_ids = []
    for source in range(num_agents):
        degree = max(
            1,
            min(
                num_agents - 1,
                round(random.gauss(mean_following, max(1.0, mean_following * 0.2))),
            ),
        )
        candidates = list(range(num_agents))
        candidates.remove(source)
        following = set(random.sample(candidates, degree))
        if scenario == "leader_amplifier" and user_types[source] == "bad_member":
            following.update(random.sample(leaders, min(2, len(leaders))))
        elif scenario == "bridge_infiltration" and "bad" in user_types[source]:
            following.update(random.sample(good, min(8, len(good))))
        elif scenario == "synchronized_boosting" and "bad" in user_types[source]:
            peers = [agent_id for agent_id in bad if agent_id != source]
            following.update(random.sample(peers, min(6, len(peers))))
        elif scenario == "persona_drift" and "bad" in user_types[source]:
            following.update(random.sample(good, min(6, len(good))))
        elif scenario == "adaptive_evasion" and "bad" in user_types[source]:
            # A sparse malicious backbone avoids an unrealistically dense bot clique.
            following.update(random.sample(members, min(2, len(members))))
        following.discard(source)
        following_ids.append(sorted(following))
    return following_ids


def _inject_scenario_activity(activity_freq, user_type, scenario):
    if "bad" not in user_type or scenario is None:
        return activity_freq
    adjusted = list(activity_freq)
    if scenario == "synchronized_boosting":
        burst_hours = (3, 9, 15, 21)
        for hour in burst_hours:
            adjusted[hour] = max(adjusted[hour], 0.95)
        for hour in set(range(24)).difference(burst_hours):
            adjusted[hour] = min(adjusted[hour], 0.35)
    elif scenario == "adaptive_evasion":
        adjusted = [min(value, 0.65) for value in adjusted]
    return [round(value, 3) for value in adjusted]


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


def build_agent_dataframe(
    profiles,
    tweet_pool,
    scenario=None,
    mean_following=None,
):
    """Convert validated DeepPersona profiles to the OASIS CSV schema."""

    validate_profiles(profiles)
    n = len(profiles)
    if scenario is None:
        role_counts = (NUM_BAD_LEADER, NUM_BAD_MEMBER, NUM_BAD)
    else:
        role_counts = _scenario_role_counts(scenario, n)
    num_bad_leader, num_bad_member, num_bad = role_counts
    total_bad = num_bad_leader + num_bad_member + num_bad
    if n < total_bad:
        raise ValueError(
            f"JSON only has {n} agents, insufficient to assign "
            f"{total_bad} bad agents"
        )
    user_type_list = (
        ["good"] * (n - total_bad)
        + ["bad_leader"] * num_bad_leader
        + ["bad_member"] * num_bad_member
        + ["bad"] * num_bad
    )
    random.shuffle(user_type_list)

    following_ids = generate_follow_graph(
        n,
        user_type_list,
        scenario,
        mean_following=mean_following,
    )

    rows = []
    for idx, profile in enumerate(profiles):
        user_type = user_type_list[idx]
        name = f"User_{idx + 1}"
        username = f"@User_{idx + 1}"
        summary = profile["Summary"].strip()

        activity_freq, activity_labels = generate_activity()
        activity_freq = _inject_scenario_activity(
            activity_freq, user_type, scenario
        )
        activity_labels = [
            "active" if value >= 0.15 else "inactive"
            for value in activity_freq
        ]
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
            "scenario_id": scenario or "legacy",
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
    parser.add_argument("--tweets-per-agent", type=int, default=10)
    parser.add_argument(
        "--tweet-pool",
        help="Audited JSON object containing non-empty good and bad text lists.",
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIO_IDS))
    parser.add_argument(
        "--mean-following",
        type=float,
        default=30.0,
        help="Mean out-degree for formal sparse scenario graphs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global NUM_BAD_LEADER, NUM_BAD_MEMBER, NUM_BAD, NUM_TWEETS_PER_AGENT
    NUM_BAD_LEADER = args.num_bad_leader
    NUM_BAD_MEMBER = args.num_bad_member
    NUM_BAD = args.num_bad
    NUM_TWEETS_PER_AGENT = args.tweets_per_agent
    random.seed(args.seed)
    if min(NUM_BAD_LEADER, NUM_BAD_MEMBER, NUM_BAD) < 0:
        raise ValueError("bad-agent counts must be non-negative")
    if NUM_TWEETS_PER_AGENT <= 0:
        raise ValueError("tweets-per-agent must be positive")
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

    tweet_pool = load_tweet_pool(args.tweet_pool)
    if args.scenario and not all(tweet_pool.values()):
        raise ValueError(
            "formal scenarios require --tweet-pool with non-empty good and bad lists"
        )
    df = build_agent_dataframe(
        profiles,
        tweet_pool,
        scenario=args.scenario,
        mean_following=args.mean_following if args.scenario else None,
    )
    output_csv = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nCSV generated: {output_csv}")
    print(f"  Total agents: {len(df)}")
    print(f"  Distribution: {dict(df['user_type'].value_counts())}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
