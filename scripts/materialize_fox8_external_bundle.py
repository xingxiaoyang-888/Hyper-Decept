"""Materialize Fox8-23 as a provenance-preserving external transfer bundle.

Fox8 supplies bot/human labels, not coordinated-behavior ground truth.  This
adapter therefore produces a traceability bundle for external transfer checks;
it deliberately never creates CIB labels, campaign IDs, or LLM-origin claims.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_ID = "fox8-23"
FEATURE_COLUMNS = (
    *(f"Semantic_{index}" for index in range(8)),
    "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
    "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
    "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
    "Empathy_Gap_Mean", "Empathy_Gap_Max", "Dark_Triad_Mean",
    "Dark_Triad_Max", "Contagion_Mean", "Contagion_Max",
    "Volatility_Mean", "Volatility_Max",
)
BEHAVIOR_COLUMNS = {
    "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
    "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
    "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_ref(source: Path) -> str:
    """Store a relocatable source reference, never a machine absolute path."""
    source = source.resolve()
    root = Path.cwd().resolve()
    try:
        reference = source.relative_to(root)
    except ValueError:
        try:
            reference = Path(os.path.relpath(source, root))
        except ValueError:
            reference = Path(source.name)
        if str(reference).startswith(".."):
            reference = Path(source.name)
    return reference.as_posix()


def _id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tweet_id(tweet: dict[str, Any], fallback: str) -> str:
    return _id(tweet.get("id_str") or tweet.get("id")) or fallback


def _timestamp_hour(value: Any) -> int | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(timestamp) else int(timestamp.hour)


def _event_type(tweet: dict[str, Any]) -> str:
    if isinstance(tweet.get("retweeted_status"), dict):
        return "retweet"
    if _id(tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")):
        return "reply"
    if tweet.get("is_quote_status"):
        return "quote"
    return "tweet"


def _relationship_rows(actor_id: str, tweet: dict[str, Any], timestamp: Any, evidence_id: str) -> Iterable[dict[str, str]]:
    reply_target = _id(tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id"))
    if reply_target:
        yield {"source_id": actor_id, "target_id": reply_target, "edge_type": "reply_to", "timestamp": str(timestamp or ""), "evidence_id": evidence_id}
    retweet = tweet.get("retweeted_status")
    if isinstance(retweet, dict):
        original_user = retweet.get("user") if isinstance(retweet.get("user"), dict) else {}
        target = _id(original_user.get("id_str") or original_user.get("id"))
        if target:
            yield {"source_id": actor_id, "target_id": target, "edge_type": "retweets", "timestamp": str(timestamp or ""), "evidence_id": evidence_id}
    entities = tweet.get("entities") if isinstance(tweet.get("entities"), dict) else {}
    for mention in entities.get("user_mentions") or []:
        if not isinstance(mention, dict):
            continue
        target = _id(mention.get("id_str") or mention.get("id"))
        if target:
            yield {"source_id": actor_id, "target_id": target, "edge_type": "mentions", "timestamp": str(timestamp or ""), "evidence_id": evidence_id}


def _profile_row(user_id: str, record: dict[str, Any], *, in_source_cohort: bool) -> dict[str, Any]:
    profile = record.get("profile") or record.get("user") or {}
    if not profile and isinstance(record.get("user_tweets"), list):
        first_tweet = next((item for item in record["user_tweets"] if isinstance(item, dict)), {})
        profile = first_tweet.get("user") if isinstance(first_tweet.get("user"), dict) else {}
    return {
        "dataset_id": DATASET_ID,
        "node_id": user_id,
        "node_type": "twitter_account",
        "in_source_cohort": bool(in_source_cohort),
        "followers_count": profile.get("followers_count"),
        "following_count": profile.get("friends_count"),
        "statuses_count": profile.get("statuses_count"),
    }


def materialize(source: Path, output: Path, *, max_users: int | None = None) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    event_path, edge_path = output / "events.csv", output / "edges.csv"
    profiles: dict[str, dict[str, Any]] = {}
    source_labels: dict[str, str] = {}
    source_record_count = 0
    duplicate_source_ids: list[str] = []
    conflicting_source_labels: list[str] = []
    context_nodes: set[str] = set()
    activity: dict[str, Counter[str]] = defaultdict(Counter)
    hours: dict[str, Counter[int]] = defaultdict(Counter)
    event_count = edge_count = malformed_rows = 0

    with event_path.open("w", newline="", encoding="utf-8") as event_handle, edge_path.open("w", newline="", encoding="utf-8") as edge_handle:
        events = csv.DictWriter(event_handle, fieldnames=("dataset_id", "event_id", "evidence_id", "actor_id", "timestamp", "event_type", "text", "source_user_id"))
        edges = csv.DictWriter(edge_handle, fieldnames=("dataset_id", "source_id", "target_id", "edge_type", "timestamp", "evidence_id"))
        events.writeheader()
        edges.writeheader()
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if max_users is not None and len(source_labels) >= max_users:
                    break
                try:
                    record = json.loads(line)
                    source_record_count += 1
                    actor_id = _id(record.get("user_id"))
                    label = str(record.get("label", "")).strip().lower()
                    tweets = record.get("user_tweets") or []
                    if actor_id is None or label not in {"human", "bot"} or not isinstance(tweets, list):
                        raise ValueError("invalid Fox8 user record")
                except (json.JSONDecodeError, ValueError, TypeError):
                    malformed_rows += 1
                    continue
                if actor_id in source_labels:
                    duplicate_source_ids.append(actor_id)
                    if source_labels[actor_id] != label:
                        conflicting_source_labels.append(actor_id)
                else:
                    source_labels[actor_id] = label
                profiles.setdefault(actor_id, _profile_row(actor_id, record, in_source_cohort=True))
                for tweet_index, tweet in enumerate(tweets):
                    if not isinstance(tweet, dict):
                        continue
                    event_id = _tweet_id(tweet, f"row:{line_number}:tweet:{tweet_index}")
                    evidence_id = f"tweet:{event_id}"
                    timestamp = tweet.get("created_at") or ""
                    text = str(tweet.get("full_text") or tweet.get("text") or "")
                    event_type = _event_type(tweet)
                    events.writerow({
                        "dataset_id": DATASET_ID, "event_id": event_id, "evidence_id": evidence_id,
                        "actor_id": actor_id, "timestamp": timestamp, "event_type": event_type,
                        "text": text, "source_user_id": actor_id,
                    })
                    event_count += 1
                    activity[actor_id]["events"] += 1
                    activity[actor_id][event_type] += 1
                    activity[actor_id]["likes"] += int(tweet.get("favorite_count") or 0)
                    lowered = text.lower()
                    activity[actor_id]["urls"] += int("http://" in lowered or "https://" in lowered or "www." in lowered)
                    entities = tweet.get("entities") if isinstance(tweet.get("entities"), dict) else {}
                    activity[actor_id]["mentions"] += int(bool(entities.get("user_mentions")))
                    activity[actor_id]["hashtags"] += int(bool(entities.get("hashtags")))
                    activity[actor_id]["media"] += int(bool(tweet.get("extended_entities", {}).get("media") if isinstance(tweet.get("extended_entities"), dict) else False))
                    hour = _timestamp_hour(timestamp)
                    if hour is not None:
                        hours[actor_id][hour] += 1
                    for edge in _relationship_rows(actor_id, tweet, timestamp, evidence_id):
                        edges.writerow({"dataset_id": DATASET_ID, **edge})
                        context_nodes.add(edge["target_id"])
                        edge_count += 1

    nodes = [_profile_row(node_id, profiles[node_id], in_source_cohort=True) for node_id in sorted(profiles)]
    nodes.extend(_profile_row(node_id, {}, in_source_cohort=False) for node_id in sorted(context_nodes - set(profiles)))
    pd.DataFrame(nodes).to_csv(output / "nodes.csv", index=False)
    labels = pd.DataFrame({
        "dataset_id": DATASET_ID,
        "node_id": [node["node_id"] for node in nodes],
        "is_bot": [source_labels.get(node["node_id"]) == "bot" for node in nodes],
        "is_known": [node["node_id"] in source_labels for node in nodes],
        "label_task": "bot_human_transfer",
        "label_source": "fox8_23_release",
    })
    labels.to_csv(output / "labels.csv", index=False)

    features = pd.DataFrame(0.0, index=np.arange(len(nodes)), columns=FEATURE_COLUMNS)
    availability = pd.DataFrame(False, index=np.arange(len(nodes)), columns=FEATURE_COLUMNS)
    for index, node in enumerate(nodes):
        node_id = node["node_id"]
        count = activity[node_id]["events"]
        if not count:
            continue
        features.loc[index, "Action_Frequency"] = count
        features.loc[index, "Like_Ratio"] = activity[node_id]["likes"] / count
        for column, key in (("Retweet_Ratio", "retweet"), ("Reply_Ratio", "reply"), ("URL_Ratio", "urls"), ("Mention_Ratio", "mentions"), ("Hashtag_Ratio", "hashtags"), ("Media_Ratio", "media")):
            features.loc[index, column] = activity[node_id][key] / count
        distribution = np.asarray(list(hours[node_id].values()), dtype=float)
        if distribution.size:
            probability = distribution / distribution.sum()
            features.loc[index, "Temporal_Entropy"] = float(-(probability * np.log(probability + 1e-12)).sum())
        for column in BEHAVIOR_COLUMNS - {"Follower_Following_Ratio"}:
            availability.loc[index, column] = True
        followers, following = node.get("followers_count"), node.get("following_count")
        if pd.notna(followers) and pd.notna(following):
            features.loc[index, "Follower_Following_Ratio"] = float(followers) / (float(following) + 1.0)
            availability.loc[index, "Follower_Following_Ratio"] = True
    features.insert(0, "user_id", [node["node_id"] for node in nodes])
    availability.insert(0, "node_id", [node["node_id"] for node in nodes])
    features.to_csv(output / "features_26d.csv", index=False)
    availability.to_csv(output / "feature_availability.csv", index=False)

    manifest = {
        "schema_version": "hypertrace.external-bundle.v1",
        "dataset_id": DATASET_ID,
        "source_ref": _portable_source_ref(source),
        "source_path_kind": "repo_relative",
        "source_sha256": _sha256(source),
        "evaluation_scope": "bot_transfer_traceability",
        "label_semantics": "bot_human",
        "not_cib_supervision": True,
        "not_llm_origin_evaluation": True,
        "capabilities": {
            "temporal": True, "text": True, "graph_edges": True,
            "source_verified_labels": True, "explicit_cib_labels": False,
            "evidence_ids": True,
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_dim": len(FEATURE_COLUMNS),
        "missing_feature_policy": "zero_impute_with_feature_availability_sidecar",
        "counts": {
            "source_records": source_record_count,
            "source_users": len(source_labels), "context_nodes": len(context_nodes - set(profiles)),
            "nodes": len(nodes), "events": event_count, "edges": edge_count,
            "malformed_source_rows": malformed_rows,
            "duplicate_source_id_records": len(duplicate_source_ids),
            "conflicting_source_label_records": len(conflicting_source_labels),
        },
        "warnings": [
            "Fox8-23 bot/human labels are not CIB membership labels.",
            "No campaign, role, counterfactual, or LLM-origin label is inferred.",
        ],
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.output, max_users=args.max_users), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
