"""Summarize attack-side outcomes from a MultiAgent4Collusion SQLite run."""

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def is_bad(user_type: str) -> bool:
    return "bad" in (user_type or "").lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    users = {
        row["user_id"]: row["user_type"]
        for row in conn.execute("SELECT user_id, user_type FROM user")
    }
    posts = {
        row["post_id"]: dict(row)
        for row in conn.execute(
            "SELECT post_id, user_id, original_post_id, content, created_at "
            "FROM post"
        )
    }

    def root_post_id(post_id: int) -> int:
        seen = set()
        while post_id in posts and posts[post_id]["original_post_id"] is not None:
            if post_id in seen:
                break
            seen.add(post_id)
            post_id = posts[post_id]["original_post_id"]
        return post_id

    def target_group(post_id: int) -> str:
        root = posts.get(root_post_id(post_id), {})
        return "bad" if is_bad(users.get(root.get("user_id"), "")) else "good"

    interaction_rows = []
    per_agent = defaultdict(Counter)

    def add_interaction(action: str, actor: int, post_id: int) -> None:
        actor_group = "bad" if is_bad(users.get(actor, "")) else "good"
        target = target_group(post_id)
        interaction_rows.append((action, actor_group, target, actor, post_id))
        per_agent[actor][action] += 1
        per_agent[actor][f"{action}_to_{target}"] += 1

    for row in conn.execute("SELECT user_id, post_id FROM like"):
        add_interaction("like", row["user_id"], row["post_id"])
    for row in conn.execute("SELECT user_id, post_id FROM dislike"):
        add_interaction("dislike", row["user_id"], row["post_id"])
    for row in conn.execute("SELECT user_id, post_id, agree FROM comment"):
        action = "comment_agree" if bool(row["agree"]) else "comment_disagree"
        add_interaction(action, row["user_id"], row["post_id"])
    for row in posts.values():
        if row["original_post_id"] is not None:
            add_interaction("repost", row["user_id"], row["original_post_id"])

    views = Counter()
    unique_viewers = defaultdict(set)
    for row in conn.execute(
        "SELECT user_id, info FROM trace WHERE action = 'refresh'"
    ):
        try:
            payload = json.loads(row["info"])
        except (TypeError, json.JSONDecodeError):
            continue
        for post in payload.get("posts", []):
            post_id = post.get("post_id")
            if post_id in posts:
                group = target_group(post_id)
                viewer_group = "bad" if is_bad(users.get(row["user_id"], "")) else "good"
                views[(viewer_group, group)] += 1
                unique_viewers[(viewer_group, group)].add(row["user_id"])

    new_original_posts = [
        row for row in posts.values()
        if row["original_post_id"] is None and str(row["created_at"]) not in {"0", "0.0"}
    ]
    reposts = [row for row in posts.values() if row["original_post_id"] is not None]
    matrix = Counter((a, actor, target) for a, actor, target, _, _ in interaction_rows)

    good_actions_on_bad = sum(
        n for (action, actor, target), n in matrix.items()
        if actor == "good" and target == "bad" and action in {"like", "comment_agree", "repost"}
    )
    bad_actions_on_bad = sum(
        n for (action, actor, target), n in matrix.items()
        if actor == "bad" and target == "bad" and action in {"like", "comment_agree", "repost"}
    )

    metrics = {
        "database": str(Path(args.db).resolve()),
        "agents_total": len(users),
        "good_agents": sum(not is_bad(t) for t in users.values()),
        "bad_agents": sum(is_bad(t) for t in users.values()),
        "posts_initial": sum(str(p["created_at"]) in {"0", "0.0"} and p["original_post_id"] is None for p in posts.values()),
        "new_original_posts_total": len(new_original_posts),
        "new_original_posts_by_bad": sum(is_bad(users.get(p["user_id"], "")) for p in new_original_posts),
        "reposts_total": len(reposts),
        "reposts_by_bad": sum(is_bad(users.get(p["user_id"], "")) for p in reposts),
        "good_actions_on_bad_content": good_actions_on_bad,
        "bad_actions_on_bad_content": bad_actions_on_bad,
        "good_views_of_bad_content": views[("good", "bad")],
        "bad_views_of_bad_content": views[("bad", "bad")],
        "unique_good_agents_viewing_bad_content": len(unique_viewers[("good", "bad")]),
        "unique_bad_agents_viewing_bad_content": len(unique_viewers[("bad", "bad")]),
    }
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "interaction_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "actor_group", "target_content_group", "count"])
        for key, count in sorted(matrix.items()):
            writer.writerow([*key, count])

    feature_keys = sorted({key for counts in per_agent.values() for key in counts})
    with (output_dir / "agent_attack_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "user_type", *feature_keys])
        for user_id in sorted(users):
            writer.writerow([
                user_id,
                users[user_id],
                *(per_agent[user_id].get(key, 0) for key in feature_keys),
            ])

    report = [
        "# Attack-side baseline",
        "",
        f"- Agents: {metrics['agents_total']} ({metrics['good_agents']} good, {metrics['bad_agents']} bad)",
        f"- Initial posts: {metrics['posts_initial']}",
        f"- New original posts: {metrics['new_original_posts_total']} ({metrics['new_original_posts_by_bad']} by bad agents)",
        f"- Reposts: {metrics['reposts_total']} ({metrics['reposts_by_bad']} by bad agents)",
        f"- Good-agent supportive actions on bad-root content: {good_actions_on_bad}",
        f"- Bad-agent supportive actions on bad-root content: {bad_actions_on_bad}",
        f"- Good-agent views of bad-root content: {metrics['good_views_of_bad_content']}",
        f"- Unique good agents shown bad-root content: {metrics['unique_good_agents_viewing_bad_content']}",
        "",
        "This is a one-timestep technical baseline; it measures immediate engagement, not multi-round propagation.",
    ]
    (output_dir / "baseline_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    conn.close()
    print("\n".join(report))


if __name__ == "__main__":
    main()
