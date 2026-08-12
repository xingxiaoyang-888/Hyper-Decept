"""Materialize one Honduras/UAE operation as an auditable external bundle.

The source release provides malicious-information-operation and genuine-user
cohorts. It does not provide LLM-origin, tactical-role, or finer campaign IDs;
this adapter deliberately preserves those semantic limits.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data_processing.feature_contracts import OBSERVABLE_18
from scripts.audit_honduras_uae_source import EXPECTED_FILES


DATASET_ID = "honduras-uae-io"
BEHAVIOR_COLUMNS = tuple(OBSERVABLE_18[8:])
TWEET_FEATURE_COLUMNS = (
    "log_char_count", "log_token_count", "uppercase_ratio",
    "punctuation_ratio", "log_url_count", "log_mention_count",
    "log_hashtag_count", "has_quote", "is_reshare", "log_like_count",
    "log_dislike_count", "log_share_count", "hour_sin", "hour_cos",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _listed(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return [item for item in (_identifier(v) for v in parsed) if item]


def _iso_time(value: Any) -> str:
    try:
        number = int(value)
        return datetime.fromtimestamp(number / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _event_type(row: dict[str, Any]) -> str:
    if _identifier(row.get("retweet_tweetid")):
        return "retweet"
    if _identifier(row.get("in_reply_to_tweetid")):
        return "reply"
    if _identifier(row.get("quoted_tweet_tweetid")):
        return "quote"
    return "post"


def _truthy_list(value: Any) -> bool:
    return bool(_listed(value))


def _tweet_features(row: dict[str, Any]) -> dict[str, float]:
    content = str(row.get("tweet_text") or "")
    characters = len(content)
    tokens = re.findall(r"\S+", content)
    letters = [character for character in content if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters else 0.0
    )
    punctuation_ratio = sum(
        not character.isalnum() and not character.isspace()
        for character in content
    ) / max(characters, 1)
    try:
        hour = datetime.fromtimestamp(
            int(row.get("tweet_time")) / 1000, tz=timezone.utc
        ).hour
    except (TypeError, ValueError, OSError, OverflowError):
        hour = 0
    angle = 2.0 * np.pi * hour / 24.0
    values = (
        np.log1p(characters),
        np.log1p(len(tokens)),
        uppercase_ratio,
        punctuation_ratio,
        np.log1p(len(_listed(row.get("urls")))),
        np.log1p(len(_listed(row.get("user_mentions")))),
        np.log1p(len(_listed(row.get("hashtags")))),
        float(_identifier(row.get("quoted_tweet_tweetid")) is not None),
        float(_identifier(row.get("retweet_tweetid")) is not None),
        np.log1p(max(float(row.get("like_count") or 0), 0.0)),
        0.0,
        np.log1p(max(float(row.get("retweet_count") or 0), 0.0)),
        np.sin(angle),
        np.cos(angle),
    )
    return dict(zip(TWEET_FEATURE_COLUMNS, map(float, values)))


def _edge_rows(
    row: dict[str, Any], actor: str, tweet: str, timestamp: str, evidence: str
) -> Iterable[dict[str, str]]:
    common = {"timestamp": timestamp, "evidence_id": evidence}
    yield {
        **common,
        "source_type": "user", "source_id": actor,
        "target_type": "tweet", "target_id": tweet,
        "edge_type": "posts",
    }
    yield {
        **common,
        "source_type": "tweet", "source_id": tweet,
        "target_type": "user", "target_id": actor,
        "edge_type": "authored_by",
    }
    retweet = _identifier(row.get("retweet_tweetid"))
    if retweet:
        yield {
            **common,
            "source_type": "user", "source_id": actor,
            "target_type": "tweet", "target_id": retweet,
            "edge_type": "retweets",
        }
        yield {
            **common,
            "source_type": "tweet", "source_id": retweet,
            "target_type": "user", "target_id": actor,
            "edge_type": "retweeted_by",
        }
    reply = _identifier(row.get("in_reply_to_tweetid"))
    if reply:
        # DeepPersona/OASIS stores replies under the comments relation.
        yield {
            **common,
            "source_type": "user", "source_id": actor,
            "target_type": "tweet", "target_id": reply,
            "edge_type": "comments",
        }
        yield {
            **common,
            "source_type": "tweet", "source_id": reply,
            "target_type": "user", "target_id": actor,
            "edge_type": "commented_by",
        }
    reply_user = _identifier(row.get("in_reply_to_userid"))
    if reply_user:
        yield {
            **common,
            "source_type": "user", "source_id": actor,
            "target_type": "user", "target_id": reply_user,
            "edge_type": "reply_to_user",
        }
    retweet_user = _identifier(row.get("retweet_userid"))
    if retweet_user:
        yield {
            **common,
            "source_type": "user", "source_id": actor,
            "target_type": "user", "target_id": retweet_user,
            "edge_type": "retweets_user",
        }
    for mentioned in _listed(row.get("user_mentions")):
        yield {
            **common,
            "source_type": "user", "source_id": actor,
            "target_type": "user", "target_id": mentioned,
            "edge_type": "mentions",
        }


def _require_source_audit(path: Path, country: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "hypertrace.honduras-uae-source-audit.v1":
        raise ValueError("unsupported source audit schema")
    matching = [
        item for item in report.get("files", [])
        if item.get("country") == country
    ]
    if len(matching) != 2 or not all(item.get("passed") for item in matching):
        raise ValueError(f"source audit has not passed both {country} cohorts")
    return report


def materialize(
    source_root: Path,
    output: Path,
    *,
    country: str,
    source_audit: Path,
) -> dict[str, Any]:
    if country not in {"honduras", "uae"}:
        raise ValueError("country must be honduras or uae")
    _require_source_audit(source_audit, country)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contracts = [
        (filename, contract)
        for filename, contract in EXPECTED_FILES.items()
        if contract["country"] == country
    ]
    for filename, _ in contracts:
        if not (source_root / filename).is_file():
            raise FileNotFoundError(source_root / filename)

    event_path = output / "events.csv"
    edge_path = output / "edges.csv"
    tweet_feature_path = output / "tweet_features.csv"
    actor_nodes: set[str] = set()
    context_users: set[str] = set()
    tweet_nodes: set[str] = set()
    context_tweets: set[str] = set()
    labels: dict[str, int] = {}
    activity: dict[str, Counter[str]] = defaultdict(Counter)
    hours: dict[str, Counter[int]] = defaultdict(Counter)
    latest_profile: dict[str, tuple[int, Any, Any]] = {}
    counts: Counter[str] = Counter()
    source_files = []

    with event_path.open("w", newline="", encoding="utf-8") as event_handle, edge_path.open(
        "w", newline="", encoding="utf-8"
    ) as edge_handle, tweet_feature_path.open(
        "w", newline="", encoding="utf-8"
    ) as tweet_feature_handle:
        events = csv.DictWriter(event_handle, fieldnames=(
            "dataset_id", "operation_id", "event_id", "evidence_id",
            "actor_id", "timestamp", "event_type", "text", "source_file",
            "source_line",
        ))
        edges = csv.DictWriter(edge_handle, fieldnames=(
            "dataset_id", "operation_id", "source_type", "source_id",
            "target_type", "target_id", "edge_type", "timestamp",
            "evidence_id",
        ))
        tweet_features = csv.DictWriter(
            tweet_feature_handle,
            fieldnames=("tweet_id", *TWEET_FEATURE_COLUMNS),
        )
        events.writeheader()
        edges.writeheader()
        tweet_features.writeheader()
        for filename, contract in contracts:
            source = source_root / filename
            source_files.append({
                "filename": filename,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "cohort": contract["cohort"],
            })
            label = 1 if contract["cohort"] == "malicious_io" else 0
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    actor = _identifier(row.get("userid"))
                    tweet = _identifier(row.get("tweetid"))
                    if actor is None or tweet is None:
                        counts["skipped_missing_actor_or_tweet"] += 1
                        continue
                    previous = labels.setdefault(actor, label)
                    if previous != label:
                        raise ValueError(f"user appears in both cohorts: {actor}")
                    actor_nodes.add(actor)
                    tweet_nodes.add(tweet)
                    timestamp = _iso_time(row.get("tweet_time"))
                    evidence = f"tweet:{tweet}"
                    event_type = _event_type(row)
                    text = str(row.get("tweet_text") or "")
                    events.writerow({
                        "dataset_id": DATASET_ID,
                        "operation_id": country,
                        "event_id": tweet,
                        "evidence_id": evidence,
                        "actor_id": actor,
                        "timestamp": timestamp,
                        "event_type": event_type,
                        "text": text,
                        "source_file": filename,
                        "source_line": line_number,
                    })
                    tweet_features.writerow({
                        "tweet_id": tweet,
                        **_tweet_features(row),
                    })
                    counts["events"] += 1
                    activity[actor]["events"] += 1
                    activity[actor][event_type] += 1
                    activity[actor]["likes"] += int(row.get("like_count") or 0)
                    activity[actor]["urls"] += int(_truthy_list(row.get("urls")))
                    activity[actor]["mentions"] += int(_truthy_list(row.get("user_mentions")))
                    activity[actor]["hashtags"] += int(_truthy_list(row.get("hashtags")))
                    try:
                        epoch_ms = int(row.get("tweet_time"))
                        hours[actor][datetime.fromtimestamp(
                            epoch_ms / 1000, tz=timezone.utc
                        ).hour] += 1
                    except (TypeError, ValueError, OSError, OverflowError):
                        epoch_ms = -1
                    previous_profile = latest_profile.get(actor)
                    if previous_profile is None or epoch_ms > previous_profile[0]:
                        latest_profile[actor] = (
                            epoch_ms,
                            row.get("follower_count"),
                            row.get("following_count"),
                        )
                    for edge in _edge_rows(row, actor, tweet, timestamp, evidence):
                        edges.writerow({
                            "dataset_id": DATASET_ID,
                            "operation_id": country,
                            **edge,
                        })
                        counts[f"edge:{edge['edge_type']}"] += 1
                        if edge["target_type"] == "user":
                            context_users.add(edge["target_id"])
                        elif edge["target_id"] != tweet:
                            context_tweets.add(edge["target_id"])

    nodes = []
    for node_id in sorted(actor_nodes | context_users):
        nodes.append({
            "dataset_id": DATASET_ID,
            "operation_id": country,
            "node_id": node_id,
            "node_type": "user",
            "in_source_cohort": node_id in actor_nodes,
        })
    for node_id in sorted(tweet_nodes | context_tweets):
        nodes.append({
            "dataset_id": DATASET_ID,
            "operation_id": country,
            "node_id": node_id,
            "node_type": "tweet",
            "in_source_cohort": node_id in tweet_nodes,
        })
    pd.DataFrame(nodes).to_csv(output / "nodes.csv", index=False)

    label_rows = [{
        "dataset_id": DATASET_ID,
        "operation_id": country,
        "node_id": node_id,
        "is_bad": value,
        "is_known": True,
        "label_task": "real_information_operation_membership",
        "label_source": "zenodo_13912659_official_bad_good_cohort",
    } for node_id, value in sorted(labels.items())]
    pd.DataFrame(label_rows).to_csv(output / "labels.csv", index=False)

    feature_rows = []
    availability_rows = []
    for user_id in sorted(actor_nodes):
        total = activity[user_id]["events"]
        values = {name: 0.0 for name in OBSERVABLE_18}
        available = {name: False for name in OBSERVABLE_18}
        values["Action_Frequency"] = float(total)
        values["Like_Ratio"] = activity[user_id]["likes"] / total
        for column, key in (
            ("Retweet_Ratio", "retweet"),
            ("Reply_Ratio", "reply"),
            ("URL_Ratio", "urls"),
            ("Mention_Ratio", "mentions"),
            ("Hashtag_Ratio", "hashtags"),
        ):
            values[column] = activity[user_id][key] / total
        distribution = np.asarray(list(hours[user_id].values()), dtype=float)
        if distribution.size:
            probabilities = distribution / distribution.sum()
            values["Temporal_Entropy"] = float(
                -(probabilities * np.log(probabilities + 1e-12)).sum()
            )
        for column in BEHAVIOR_COLUMNS:
            if column != "Media_Ratio":
                available[column] = True
        _, followers, following = latest_profile[user_id]
        try:
            values["Follower_Following_Ratio"] = float(followers) / (
                float(following) + 1.0
            )
        except (TypeError, ValueError):
            available["Follower_Following_Ratio"] = False
        feature_rows.append({"user_id": user_id, **values})
        availability_rows.append({"node_id": user_id, **available})
    pd.DataFrame(feature_rows).to_csv(output / "features_observable18.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(
        output / "feature_availability.csv", index=False
    )

    manifest = {
        "schema_version": "hypertrace.external-bundle.v2",
        "dataset_id": DATASET_ID,
        "operation_id": country,
        "source": {
            "doi": "10.5281/zenodo.13912659",
            "license": "CC-BY-4.0",
            "source_audit_sha256": _sha256(source_audit),
            "files": source_files,
        },
        "evaluation_scope": "real_information_operation_external_evaluation",
        "label_semantics": "information_operation_membership",
        "not_generic_bot_labels": True,
        "not_llm_origin_evaluation": True,
        "not_campaign_level_cib_labels": True,
        "capabilities": {
            "temporal": True,
            "text": True,
            "typed_graph_edges": True,
            "source_verified_io_labels": True,
            "explicit_campaign_ids": False,
            "explicit_cib_roles": False,
            "evidence_ids": True,
        },
        "feature_contract": "observable18_partial",
        "feature_columns": list(OBSERVABLE_18),
        "semantic_feature_status": "unavailable_until_shared_train_only_projection",
        "missing_feature_policy": "zero_with_feature_availability_sidecar",
        "counts": {
            **dict(sorted(counts.items())),
            "source_users": len(actor_nodes),
            "context_users": len(context_users - actor_nodes),
            "source_tweets": len(tweet_nodes),
            "context_tweets": len(context_tweets - tweet_nodes),
            "malicious_users": sum(labels.values()),
            "genuine_users": len(labels) - sum(labels.values()),
        },
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--country", required=True, choices=("honduras", "uae"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.source_root,
        args.output,
        country=args.country,
        source_audit=args.source_audit,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
