"""Convert raw TwiBot-22 data into the original OASIS/RogueAgent schema.

The existing TwiBot adapter emits a compact three-table database intended for
the detector.  This adapter instead executes the SQL schemas shipped with
MultiAgent4Collusion and preserves a source-to-internal ID mapping so the same
database can be consumed by the original platform utilities and the white-box
detection pipeline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import random
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import ijson


FOLLOW_RELATIONS = {"following", "followers"}
TWEET_RELATION_TO_ACTION = {
    "retweeted": "retweet",
    "quoted": "quote",
    "replied_to": "reply",
}
SCHEMA_FILES = (
    "user.sql",
    "post.sql",
    "follow.sql",
    "mute.sql",
    "like.sql",
    "dislike.sql",
    "trace.sql",
    "rec.sql",
    "comment.sql",
    "comment_like.sql",
    "comment_dislike.sql",
    "product.sql",
)


def normalize_id(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def load_id_file(path: Path) -> list[str]:
    seen = set()
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            key = next((name for name in ("source_user_id", "user_id", "id") if name in fields), None)
            if key is None:
                raise ValueError("Core-ID CSV must contain source_user_id, user_id, or id")
            iterator: Iterable[object] = (row.get(key) for row in reader)
        else:
            iterator = handle
        for raw in iterator:
            value = normalize_id(raw)
            if value is not None and value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError(f"Core-ID file is empty: {path}")
    return values


class Reservoirs:
    def __init__(self, limit: int, rng: random.Random):
        self.limit = max(int(limit), 0)
        self.rng = rng
        self.values: dict[str, list[object]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, key: str, value: object) -> None:
        if self.limit == 0:
            return
        self.counts[key] += 1
        bucket = self.values[key]
        if len(bucket) < self.limit:
            bucket.append(value)
            return
        index = self.rng.randrange(self.counts[key])
        if index < self.limit:
            bucket[index] = value


class TwiBotOasisAdapter:
    def __init__(
        self,
        twibot_dir: Path,
        output_dir: Path,
        total_sample_size: int = 1000,
        core_ids_path: Optional[Path] = None,
        max_actions: int = 50,
        max_follows: int = 100,
        max_chars_per_tweet: int = 280,
        max_total_tweet_chars: int = 5000,
        output_tag: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.twibot_dir = twibot_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.total_sample_size = int(total_sample_size)
        self.core_ids_path = core_ids_path.resolve() if core_ids_path else None
        self.max_actions = int(max_actions)
        self.max_follows = int(max_follows)
        self.max_chars = int(max_chars_per_tweet)
        self.max_total_chars = int(max_total_tweet_chars)
        self.output_tag = output_tag
        self.random_seed = int(random_seed)
        self.rng = random.Random(self.random_seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        required = ("label.csv", "edge.csv", "user.json")
        missing = [name for name in required if not (self.twibot_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing TwiBot-22 files: {', '.join(missing)}")

    def _sample_core_users(self) -> tuple[list[str], set[str]]:
        label_path = self.twibot_dir / "label.csv"
        requested_ids = load_id_file(self.core_ids_path) if self.core_ids_path else None
        requested_set = set(requested_ids or [])
        labels: dict[str, str] = {}
        humans: list[str] = []
        bots: list[str] = []
        with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                uid = normalize_id(row.get("id"))
                label = str(row.get("label") or "").strip().lower()
                if uid is None or label not in {"human", "bot"}:
                    continue
                if requested_ids is not None:
                    if uid in requested_set:
                        labels[uid] = label
                elif label == "human":
                    humans.append(uid)
                else:
                    bots.append(uid)

        if requested_ids is not None:
            missing = [uid for uid in requested_ids if uid not in labels]
            if missing:
                raise ValueError(f"{len(missing)} requested core IDs have no TwiBot label")
            core = requested_ids[: self.total_sample_size]
            bots_selected = {uid for uid in core if labels[uid] == "bot"}
        else:
            human_count = min((self.total_sample_size + 1) // 2, len(humans))
            bot_count = min(self.total_sample_size // 2, len(bots))
            remaining = self.total_sample_size - human_count - bot_count
            if remaining:
                extra_humans = min(remaining, len(humans) - human_count)
                human_count += extra_humans
                remaining -= extra_humans
            if remaining:
                bot_count += min(remaining, len(bots) - bot_count)
            selected_humans = self.rng.sample(humans, human_count)
            selected_bots = self.rng.sample(bots, bot_count)
            core = selected_humans + selected_bots
            bots_selected = set(selected_bots)

        if not core:
            raise ValueError("No labeled core users were selected")
        print(
            f"[adapter] core users: {len(core)} "
            f"(human={len(core) - len(bots_selected)}, bot={len(bots_selected)})",
            flush=True,
        )
        return core, bots_selected

    @staticmethod
    def _canonical_follow(source: str, relation: str, target: str) -> tuple[str, str]:
        if relation == "followers":
            return target, source
        return source, target

    def _scan_primary_edges(
        self, core_users: set[str]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], set[str]]:
        outgoing = Reservoirs(self.max_follows, self.rng)
        incoming = Reservoirs(self.max_follows, self.rng)
        # Keep separate reservoirs so a user's sparse likes are not silently
        # displaced by a much larger posting history.  The final per-user
        # sample remains capped and approximately follows the raw action mix.
        posts = Reservoirs(self.max_actions, self.rng)
        likes = Reservoirs(self.max_actions, self.rng)
        edge_path = self.twibot_dir / "edge.csv"
        print("[adapter] edge pass 1/3: follow, post, like", flush=True)
        with edge_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                relation = str(row.get("relation") or "").strip().lower()
                if relation not in FOLLOW_RELATIONS and relation not in {"post", "like"}:
                    continue
                source = normalize_id(row.get("source_id"))
                target = normalize_id(row.get("target_id"))
                if source is None or target is None:
                    continue
                if relation in FOLLOW_RELATIONS:
                    follower, followee = self._canonical_follow(source, relation, target)
                    if follower in core_users:
                        outgoing.add(follower, (follower, followee))
                    if followee in core_users:
                        incoming.add(followee, (follower, followee))
                elif source in core_users:
                    reservoir = posts if relation == "post" else likes
                    reservoir.add(source, (source, relation, target))

        follow_edges = {
            edge
            for reservoirs in (outgoing.values, incoming.values)
            for bucket in reservoirs.values()
            for edge in bucket
        }
        selected_actions = []
        for uid in sorted(core_users):
            post_bucket = posts.values.get(uid, [])
            like_bucket = likes.values.get(uid, [])
            post_count = posts.counts.get(uid, 0)
            like_count = likes.counts.get(uid, 0)
            total_count = post_count + like_count
            if total_count <= self.max_actions:
                selected_actions.extend(post_bucket)
                selected_actions.extend(like_bucket)
                continue

            like_quota = round(self.max_actions * like_count / total_count)
            if post_count and like_count and self.max_actions >= 2:
                like_quota = min(max(like_quota, 1), self.max_actions - 1)
            else:
                like_quota = min(max(like_quota, 0), self.max_actions)
            post_quota = self.max_actions - like_quota
            selected_actions.extend(self.rng.sample(post_bucket, min(post_quota, len(post_bucket))))
            selected_actions.extend(self.rng.sample(like_bucket, min(like_quota, len(like_bucket))))
        owned_tweets = {target for _, action, target in selected_actions if action == "post"}
        print(
            f"[adapter] sampled follow={len(follow_edges)}, "
            f"actions={len(selected_actions)}, owned_tweets={len(owned_tweets)}",
            flush=True,
        )
        return sorted(follow_edges), selected_actions, owned_tweets

    def _scan_tweet_relations(self, owned_tweets: set[str]) -> dict[str, tuple[str, str]]:
        relationships: dict[str, tuple[str, str]] = {}
        edge_path = self.twibot_dir / "edge.csv"
        print("[adapter] edge pass 2/3: retweet, quote, reply", flush=True)
        with edge_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                relation = str(row.get("relation") or "").strip().lower()
                if relation not in TWEET_RELATION_TO_ACTION:
                    continue
                source = normalize_id(row.get("source_id"))
                if source not in owned_tweets:
                    continue
                target = normalize_id(row.get("target_id"))
                if target is not None:
                    relationships[source] = (TWEET_RELATION_TO_ACTION[relation], target)
        print(f"[adapter] classified tweet relations={len(relationships)}", flush=True)
        return relationships

    def _scan_tweet_authors(self, target_tweets: set[str]) -> dict[str, str]:
        authors: dict[str, str] = {}
        edge_path = self.twibot_dir / "edge.csv"
        print("[adapter] edge pass 3/3: target tweet authors", flush=True)
        with edge_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("relation") != "post":
                    continue
                tweet_id = normalize_id(row.get("target_id"))
                if tweet_id in target_tweets:
                    author_id = normalize_id(row.get("source_id"))
                    if author_id is not None:
                        authors[tweet_id] = author_id
                        if len(authors) == len(target_tweets):
                            break
        print(f"[adapter] target authors={len(authors)}/{len(target_tweets)}", flush=True)
        return authors

    def _extract_tweets(self, target_tweets: set[str]) -> dict[str, dict]:
        tweets: dict[str, dict] = {}
        for index in range(9):
            if len(tweets) == len(target_tweets):
                break
            path = self.twibot_dir / f"tweet_{index}.json"
            if not path.is_file():
                continue
            print(
                f"[adapter] tweets {path.name}: {len(tweets)}/{len(target_tweets)} found",
                flush=True,
            )
            with path.open("rb") as handle:
                for record in ijson.items(handle, "item"):
                    tweet_id = normalize_id(record.get("id"))
                    if tweet_id in target_tweets:
                        metrics = record.get("public_metrics") or {}
                        entities = record.get("entities") or {}
                        attachments = record.get("attachments") or {}
                        tweets[tweet_id] = {
                            "author_id": normalize_id(record.get("author_id")),
                            "content": str(record.get("text") or "").replace("\n", " ").replace("\r", " ")[: self.max_chars],
                            "created_at": record.get("created_at"),
                            "like_count": int(metrics.get("like_count", 0) or 0),
                            "retweet_count": int(metrics.get("retweet_count", 0) or 0),
                            "reply_count": int(metrics.get("reply_count", 0) or 0),
                            "url_count": len(entities.get("urls") or []),
                            "mention_count": len(entities.get("mentions") or []),
                            "hashtag_count": len(entities.get("hashtags") or []),
                            "media_count": len(attachments.get("media_keys") or []),
                        }
        print(f"[adapter] tweet metadata={len(tweets)}/{len(target_tweets)}", flush=True)
        return tweets

    def _extract_users(self, users_to_fetch: set[str]) -> dict[str, dict]:
        users: dict[str, dict] = {}
        path = self.twibot_dir / "user.json"
        print(f"[adapter] user metadata target={len(users_to_fetch)}", flush=True)
        with path.open("rb") as handle:
            for record in ijson.items(handle, "item"):
                uid = normalize_id(record.get("id"))
                if uid not in users_to_fetch:
                    continue
                metrics = record.get("public_metrics") or {}
                users[uid] = {
                    "name": str(record.get("name") or uid),
                    "username": str(record.get("username") or uid),
                    "bio": str(record.get("description") or "").replace("\n", " ").replace("\r", " "),
                    "created_at": record.get("created_at"),
                    "followers": int(metrics.get("followers_count", 0) or 0),
                    "following": int(metrics.get("following_count", 0) or 0),
                }
                if len(users) == len(users_to_fetch):
                    break
        print(f"[adapter] user metadata={len(users)}/{len(users_to_fetch)}", flush=True)
        return users

    @staticmethod
    def _schema_dir() -> Path:
        return (
            Path(__file__).resolve().parents[1]
            / "MultiAgent4Collusion-master"
            / "oasis"
            / "social_platform"
            / "schema"
        )

    def _create_original_db(self, path: Path) -> sqlite3.Connection:
        if path.exists():
            path.unlink()
        schema_dir = self._schema_dir()
        missing = [name for name in SCHEMA_FILES if not (schema_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing original OASIS schemas: {', '.join(missing)}")
        conn = sqlite3.connect(path)
        for name in SCHEMA_FILES:
            conn.executescript((schema_dir / name).read_text(encoding="utf-8"))
        return conn

    def _build_outputs(
        self,
        core_order: list[str],
        bots: set[str],
        follow_edges: list[tuple[str, str]],
        actions: list[tuple[str, str, str]],
        relations: dict[str, tuple[str, str]],
        tweet_authors: dict[str, str],
        tweets: dict[str, dict],
        users: dict[str, dict],
    ) -> tuple[Path, Path]:
        core_set = set(core_order)
        boundary = {
            uid
            for edge in follow_edges
            for uid in edge
            if uid not in core_set
        }
        boundary.update(uid for uid in tweet_authors.values() if uid not in core_set)
        for metadata in tweets.values():
            author = metadata.get("author_id")
            if author and author not in core_set:
                boundary.add(author)
        all_users = core_order + sorted(boundary)
        user_map = {source_id: index for index, source_id in enumerate(all_users)}

        all_tweet_ids = sorted(tweets)
        tweet_map = {source_id: index for index, source_id in enumerate(all_tweet_ids)}
        tag = self.output_tag or str(len(core_order))
        db_path = self.output_dir / f"twibot_{tag}_v5.db"
        csv_path = self.output_dir / f"twibot_{tag}_multimodal_v5.csv"

        conn = self._create_original_db(db_path)
        cursor = conn.cursor()
        user_rows = []
        for source_id in all_users:
            metadata = users.get(source_id, {})
            if source_id in core_set:
                user_type = "bad" if source_id in bots else "good"
                agent_id = user_map[source_id]
            else:
                user_type = "boundary"
                agent_id = None
            user_rows.append(
                (
                    user_map[source_id],
                    agent_id,
                    source_id,
                    metadata.get("name", source_id),
                    metadata.get("bio", ""),
                    metadata.get("created_at"),
                    metadata.get("following", 0),
                    metadata.get("followers", 0),
                    user_type,
                    metadata.get("bio", ""),
                )
            )
        cursor.executemany(
            "INSERT INTO user (user_id, agent_id, user_name, name, bio, created_at, "
            "num_followings, num_followers, user_type, user_char) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            user_rows,
        )

        follow_rows = [
            (user_map[follower], user_map[followee], None)
            for follower, followee in follow_edges
            if follower in user_map and followee in user_map
        ]
        cursor.executemany(
            "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
            follow_rows,
        )

        relation_by_source = relations
        post_rows = []
        for source_tweet_id in all_tweet_ids:
            metadata = tweets[source_tweet_id]
            source_author = tweet_authors.get(source_tweet_id) or metadata.get("author_id")
            if source_author not in user_map:
                continue
            action, target_tweet_id = relation_by_source.get(source_tweet_id, ("post", None))
            original_post_id = (
                tweet_map.get(target_tweet_id) if action in {"retweet", "quote"} else None
            )
            quote_content = metadata.get("content", "") if action == "quote" else None
            post_rows.append(
                (
                    tweet_map[source_tweet_id],
                    user_map[source_author],
                    original_post_id,
                    metadata.get("content", ""),
                    quote_content,
                    metadata.get("created_at"),
                    metadata.get("like_count", 0),
                    0,
                    metadata.get("retweet_count", 0),
                )
            )
        cursor.executemany(
            "INSERT INTO post (post_id, user_id, original_post_id, content, quote_content, "
            "created_at, num_likes, num_dislikes, num_shares) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            post_rows,
        )

        like_rows = []
        comment_rows = []
        trace_rows = []
        user_texts: dict[str, list[str]] = defaultdict(list)
        trace_counter = 0
        for source_user, raw_action, source_tweet in actions:
            if source_user not in user_map:
                continue
            action = raw_action
            target_tweet = source_tweet
            if raw_action == "post" and source_tweet in relations:
                action, relation_target = relations[source_tweet]
                if action in {"retweet", "quote", "reply"}:
                    target_tweet = relation_target
            if raw_action == "post" and source_tweet in tweets:
                user_texts[source_user].append(tweets[source_tweet].get("content", ""))
            if action == "like" and target_tweet in tweet_map:
                like_rows.append((user_map[source_user], tweet_map[target_tweet], None))
            if action == "reply" and target_tweet in tweet_map:
                content = tweets.get(source_tweet, {}).get("content", "")
                created_at = tweets.get(source_tweet, {}).get("created_at")
                comment_rows.append(
                    (tweet_map[target_tweet], user_map[source_user], content, None, created_at, 0, 0)
                )
            trace_counter += 1
            created_at = tweets.get(source_tweet, {}).get("created_at")
            if not created_at:
                created_at = datetime.fromtimestamp(
                    trace_counter, tz=timezone.utc
                ).isoformat()
            info = json.dumps(
                {
                    "source_user_id": source_user,
                    "source_tweet_id": source_tweet,
                    "target_tweet_id": target_tweet,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            trace_rows.append((user_map[source_user], created_at, action, info))

        cursor.executemany(
            'INSERT INTO "like" (user_id, post_id, created_at) VALUES (?, ?, ?)',
            like_rows,
        )
        cursor.executemany(
            "INSERT INTO comment (post_id, user_id, content, agree, created_at, "
            "num_likes, num_dislikes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            comment_rows,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO trace (user_id, created_at, action, info) VALUES (?, ?, ?, ?)",
            trace_rows,
        )
        cursor.executescript(
            "CREATE INDEX IF NOT EXISTS idx_user_agent ON user(agent_id);"
            "CREATE INDEX IF NOT EXISTS idx_follow_follower ON follow(follower_id);"
            "CREATE INDEX IF NOT EXISTS idx_follow_followee ON follow(followee_id);"
            "CREATE INDEX IF NOT EXISTS idx_post_user ON post(user_id);"
            "CREATE INDEX IF NOT EXISTS idx_like_user ON 'like'(user_id);"
            "CREATE INDEX IF NOT EXISTS idx_trace_user ON trace(user_id);"
        )
        conn.commit()
        conn.close()

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = (
                "user_id",
                "source_user_id",
                "user_char",
                "followers_count",
                "following_count",
                "previous_tweets",
                "user_type",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for source_id in core_order:
                metadata = users.get(source_id, {})
                writer.writerow(
                    {
                        "user_id": user_map[source_id],
                        "source_user_id": source_id,
                        "user_char": metadata.get("bio", ""),
                        "followers_count": metadata.get("followers", 0),
                        "following_count": metadata.get("following", 0),
                        "previous_tweets": " | ".join(user_texts[source_id])[: self.max_total_chars],
                        "user_type": "bad" if source_id in bots else "good",
                    }
                )

        user_map_path = self.output_dir / f"twibot_{tag}_user_id_map.csv"
        with user_map_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("source_user_id", "user_id", "is_core", "user_type"))
            for source_id in all_users:
                writer.writerow(
                    (
                        source_id,
                        user_map[source_id],
                        int(source_id in core_set),
                        "bad" if source_id in bots else ("good" if source_id in core_set else "boundary"),
                    )
                )

        tweet_map_path = self.output_dir / f"twibot_{tag}_tweet_id_map.csv"
        with tweet_map_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("source_tweet_id", "post_id"))
            for source_id in all_tweet_ids:
                writer.writerow((source_id, tweet_map[source_id]))

        manifest = {
            "adapter": "twibot_oasis_adapter",
            "schema_source": str(self._schema_dir()),
            "twibot_dir": str(self.twibot_dir),
            "random_seed": self.random_seed,
            "core_users": len(core_order),
            "bots": len(bots),
            "all_users": len(all_users),
            "follow_rows": len(follow_rows),
            "post_rows": len(post_rows),
            "like_rows": len(like_rows),
            "comment_rows": len(comment_rows),
            "trace_rows": len(trace_rows),
            "db_path": str(db_path),
            "csv_path": str(csv_path),
            "user_map_path": str(user_map_path),
            "tweet_map_path": str(tweet_map_path),
        }
        manifest_path = self.output_dir / f"twibot_{tag}_oasis_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[adapter] DB: {db_path}", flush=True)
        print(f"[adapter] CSV: {csv_path}", flush=True)
        print(f"[adapter] manifest: {manifest_path}", flush=True)
        return db_path, csv_path

    def run(self) -> tuple[Path, Path]:
        core_order, bots = self._sample_core_users()
        core_set = set(core_order)
        follow_edges, actions, owned_tweets = self._scan_primary_edges(core_set)
        relations = self._scan_tweet_relations(owned_tweets)
        target_tweets = {target for _, _, target in actions}
        target_tweets.update(target for _, target in relations.values())
        tweet_authors = self._scan_tweet_authors(target_tweets)
        tweets = self._extract_tweets(target_tweets)
        for tweet_id, metadata in tweets.items():
            if metadata.get("author_id"):
                tweet_authors.setdefault(tweet_id, metadata["author_id"])
        follow_boundary = {uid for edge in follow_edges for uid in edge} - core_set
        users_to_fetch = core_set | follow_boundary | set(tweet_authors.values())
        users = self._extract_users(users_to_fetch)
        return self._build_outputs(
            core_order,
            bots,
            follow_edges,
            actions,
            relations,
            tweet_authors,
            tweets,
            users,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TwiBot-22 into the original MultiAgent4Collusion OASIS schema."
    )
    parser.add_argument("--twibot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-sample-size", type=int, default=1000)
    parser.add_argument("--core-ids", type=Path)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--max-follows", type=int, default=100)
    parser.add_argument("--max-chars-per-tweet", type=int, default=280)
    parser.add_argument("--max-total-tweet-chars", type=int, default=5000)
    parser.add_argument("--output-tag")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = TwiBotOasisAdapter(
        twibot_dir=args.twibot_dir,
        output_dir=args.output_dir,
        total_sample_size=args.total_sample_size,
        core_ids_path=args.core_ids,
        max_actions=args.max_actions,
        max_follows=args.max_follows,
        max_chars_per_tweet=args.max_chars_per_tweet,
        max_total_tweet_chars=args.max_total_tweet_chars,
        output_tag=args.output_tag,
        random_seed=args.random_seed,
    )
    adapter.run()


if __name__ == "__main__":
    main()
