"""Build a MultiAgent4Collusion simulation CSV from a TwiBot OASIS export."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-db", required=True, type=Path)
    parser.add_argument("--adapter-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-seed-posts", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hour_of(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).hour
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    adapter_db = args.adapter_db.expanduser().resolve()
    adapter_csv = args.adapter_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    with adapter_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    base_by_id = {int(row["user_id"]): row for row in base_rows}
    num_agents = len(base_rows)
    expected_ids = list(range(num_agents))
    if sorted(base_by_id) != expected_ids:
        raise ValueError("Adapter CSV user_id values must be contiguous from 0")

    conn = sqlite3.connect(adapter_db)
    conn.row_factory = sqlite3.Row
    core_users = {
        int(row["agent_id"]): row
        for row in conn.execute(
            "SELECT user_id, agent_id, user_name, name, bio, created_at, "
            "num_followings, num_followers, user_type, user_char "
            "FROM user WHERE agent_id IS NOT NULL ORDER BY agent_id"
        )
    }
    if sorted(core_users) != expected_ids:
        raise ValueError("Adapter DB core agent IDs do not match the adapter CSV")

    following: dict[int, list[int]] = defaultdict(list)
    followers: Counter[int] = Counter()
    for follower_id, followee_id in conn.execute(
        "SELECT follower_id, followee_id FROM follow "
        "WHERE follower_id < ? AND followee_id < ?",
        (num_agents, num_agents),
    ):
        following[int(follower_id)].append(int(followee_id))
        followers[int(followee_id)] += 1

    seed_posts: dict[int, list[str]] = defaultdict(list)
    post_hours: dict[int, list[int]] = defaultdict(lambda: [0] * 24)
    for row in conn.execute(
        "SELECT user_id, content, created_at FROM post "
        "WHERE user_id < ? ORDER BY post_id",
        (num_agents,),
    ):
        agent_id = int(row["user_id"])
        content = str(row["content"] or "").strip()
        if content and len(seed_posts[agent_id]) < args.max_seed_posts:
            seed_posts[agent_id].append(content)
        hour = hour_of(row["created_at"])
        if hour is not None:
            post_hours[agent_id][hour] += 1
    conn.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id", "source_user_id", "name", "username", "description",
        "created_at", "user_char", "user_type", "followers_count",
        "following_count", "following_list", "following_agentid_list",
        "previous_tweets", "tweets_id", "activity_level_frequency",
        "activity_level",
    ]
    empty_bios = 0
    role_counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for agent_id in expected_ids:
            db_row = core_users[agent_id]
            base = base_by_id[agent_id]
            bio = str(db_row["bio"] or base.get("user_char") or "").strip()
            posts = seed_posts[agent_id]
            if not bio:
                empty_bios += 1
                bio = (
                    "No public biography is available. Recent public posts: "
                    + " ".join(posts[:2])
                    if posts
                    else "No public biography or recent public post is available."
                )
            counts = post_hours[agent_id]
            peak = max(counts) if counts else 0
            frequency = (
                [round(0.05 + 0.95 * count / peak, 3) for count in counts]
                if peak
                else [0.05] * 24
            )
            levels = ["active" if value >= 0.2 else "inactive" for value in frequency]
            internal_following = sorted(set(following[agent_id]))
            raw_following = int(db_row["num_followings"] or 0)
            raw_followers = int(db_row["num_followers"] or 0)
            role = str(base.get("user_type") or db_row["user_type"] or "good")
            role_counts[role] += 1
            writer.writerow(
                {
                    "user_id": agent_id,
                    "source_user_id": base.get("source_user_id") or db_row["user_name"],
                    "name": db_row["name"] or f"TwiBot_User_{agent_id}",
                    "username": f"@{db_row['user_name']}",
                    "description": bio,
                    "created_at": db_row["created_at"] or "2020-01-01T00:00:00+00:00",
                    "user_char": bio,
                    "user_type": role,
                    # The framework adds each preserved internal edge after signup.
                    "followers_count": max(raw_followers - followers[agent_id], 0),
                    "following_count": max(raw_following - len(internal_following), 0),
                    "following_list": repr(internal_following),
                    "following_agentid_list": repr(internal_following),
                    "previous_tweets": repr(posts),
                    "tweets_id": "[]",
                    "activity_level_frequency": repr(frequency),
                    "activity_level": repr(levels),
                }
            )

    manifest = {
        "converter": "twibot_simulation_csv",
        "adapter_db": str(adapter_db),
        "adapter_csv": str(adapter_csv),
        "output_csv": str(output),
        "num_agents": num_agents,
        "role_counts": dict(role_counts),
        "internal_follow_edges": sum(len(v) for v in following.values()),
        "seed_posts": sum(len(v) for v in seed_posts.values()),
        "empty_bios_replaced": empty_bios,
        "max_seed_posts_per_agent": args.max_seed_posts,
        "sha256": {
            "adapter_db": sha256(adapter_db),
            "adapter_csv": sha256(adapter_csv),
            "output_csv": sha256(output),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Simulation CSV: {output}")
    print(f"Agents: {num_agents}; roles: {dict(role_counts)}")
    print(f"Internal follows: {manifest['internal_follow_edges']}")
    print(f"Seed posts: {manifest['seed_posts']}; replaced empty bios: {empty_bios}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
