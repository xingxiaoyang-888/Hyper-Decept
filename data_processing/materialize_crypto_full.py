"""Stream the complete Crypto-Campaign release into an auditable graph bundle.

All labelled comment rows contribute to user activity and temporal features.
To keep graph size tractable, each campaign retains a deterministic bottom-k
sample of consecutive cross-user interactions with source-row provenance.
No bot, CIB, or LLM-origin label is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHARACTER = ROOT / "Character Classification"
for value in (ROOT, CHARACTER):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from joint_training import DEFAULT_FEATURE_COLUMNS  # noqa: E402


COMMENT_FILES = (
    "comments_participation.tsv",
    "comments_registration.tsv",
    "comments_other.tsv",
    "comments_author.tsv",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def priority(*parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[:8], "big"
    )


def materialize(
    source: Path,
    output: Path,
    *,
    chunksize: int = 100_000,
    edges_per_campaign: int = 32,
) -> dict:
    labelled = source / "labeled"
    users_path = labelled / "users.tsv"
    events_path = labelled / "events.tsv"
    users = pd.read_csv(users_path, sep="\t", low_memory=False)
    users["user_id"] = users["user_id"].astype(str)
    if users["user_id"].duplicated().any():
        raise ValueError("full Crypto users contain duplicate user_id")
    node_ids = users["user_id"].tolist()
    node_map = {value: index for index, value in enumerate(node_ids)}
    n = len(users)
    action_count = np.zeros(n, dtype=np.int64)
    hour_count = np.zeros((n, 24), dtype=np.int64)
    link_count = np.zeros(n, dtype=np.int64)
    valid_time_count = np.zeros(n, dtype=np.int64)
    processed_rows = 0
    known_user_rows = 0
    malformed_rows = 0
    campaigns_seen: set[str] = set()
    last_actor: dict[str, tuple[str, str, str, str]] = {}
    edge_heaps: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    file_rows: dict[str, int] = {}

    for filename in COMMENT_FILES:
        path = labelled / filename
        rows = 0
        usecols = lambda name: name in {
            "thread_id", "comment_id", "post_time", "user_id",
            "twitter_links", "facebook_links", "instagram_links",
            "telegram_links", "reddit_links", "youtube_links", "medium_links",
            "linkedin_links", "discord_links", "tiktok_links", "steemit_links",
            "image_links", "other_links",
        }
        for chunk in pd.read_csv(
            path, sep="\t", chunksize=chunksize, low_memory=False,
            usecols=usecols, on_bad_lines="skip",
        ):
            rows += len(chunk)
            processed_rows += len(chunk)
            required = {"thread_id", "comment_id", "post_time", "user_id"}
            if not required.issubset(chunk.columns):
                raise ValueError(f"{filename} missing required columns")
            chunk["user_id"] = chunk["user_id"].astype(str)
            chunk["thread_id"] = chunk["thread_id"].astype(str)
            indices = chunk["user_id"].map(node_map)
            known = indices.notna()
            known_user_rows += int(known.sum())
            malformed_rows += int((~known).sum())
            if known.any():
                idx = indices[known].astype(int).to_numpy()
                np.add.at(action_count, idx, 1)
                times = pd.to_datetime(chunk.loc[known, "post_time"], errors="coerce", utc=True)
                timed = times.notna().to_numpy()
                if timed.any():
                    timed_idx = idx[timed]
                    hours = times[times.notna()].dt.hour.to_numpy(dtype=int)
                    np.add.at(hour_count, (timed_idx, hours), 1)
                    np.add.at(valid_time_count, timed_idx, 1)
                link_columns = [c for c in chunk.columns if c.endswith("_links")]
                if link_columns:
                    has_link = chunk.loc[known, link_columns].fillna("").astype(str).apply(
                        lambda row: any(value.strip() for value in row), axis=1
                    ).to_numpy()
                    np.add.at(link_count, idx[has_link], 1)

            for row in chunk.loc[known, ["thread_id", "comment_id", "post_time", "user_id"]].itertuples(index=False):
                campaign = str(row.thread_id)
                actor = str(row.user_id)
                campaigns_seen.add(campaign)
                previous = last_actor.get(campaign)
                if previous is not None and previous[0] != actor:
                    source_id, source_comment, source_time, source_file = previous
                    record = (
                        source_id, actor, campaign, str(row.post_time),
                        f"{source_file}:{source_comment}", f"{filename}:{row.comment_id}",
                    )
                    rank = priority(*record)
                    heap = edge_heaps.setdefault(campaign, [])
                    item = (-rank, record)
                    if len(heap) < edges_per_campaign:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                last_actor[campaign] = (
                    actor, str(row.comment_id), str(row.post_time), filename
                )
        file_rows[filename] = rows

    total = action_count.clip(min=1)
    probabilities = hour_count / valid_time_count.clip(min=1)[:, None]
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    feature = pd.DataFrame(0.0, index=np.arange(n), columns=list(DEFAULT_FEATURE_COLUMNS))
    feature["Action_Frequency"] = np.log1p(action_count)
    feature["Temporal_Entropy"] = entropy
    feature["URL_Ratio"] = link_count / total
    feature.insert(0, "user_id", node_ids)
    availability = pd.DataFrame(False, index=np.arange(n), columns=list(DEFAULT_FEATURE_COLUMNS))
    availability["Action_Frequency"] = action_count > 0
    availability["Temporal_Entropy"] = valid_time_count > 0
    availability["URL_Ratio"] = action_count > 0
    availability.insert(0, "node_id", node_ids)

    edge_rows = []
    for heap in edge_heaps.values():
        for _, record in heap:
            edge_rows.append(record)
    edges = pd.DataFrame(edge_rows, columns=(
        "source_id", "target_id", "campaign_id", "timestamp",
        "source_evidence_id", "target_evidence_id",
    ))
    edges.insert(0, "dataset_id", "crypto-campaign-full")
    edges["edge_type"] = "temporal_co_participation"
    edges["evidence_id"] = edges["source_evidence_id"] + "->" + edges["target_evidence_id"]
    edges = edges.sort_values(["campaign_id", "evidence_id"]).reset_index(drop=True)

    nodes = users.copy()
    nodes = nodes.rename(columns={"user_id": "node_id"})
    nodes.insert(0, "dataset_id", "crypto-campaign-full")
    nodes["node_type"] = "forum_user"
    labels = nodes[["dataset_id", "node_id", "bounty_participant"]].rename(
        columns={"bounty_participant": "label_value"}
    )
    labels["label_task"] = "bounty_participation"
    labels["label_source"] = "published_dataset_field"
    labels["is_known"] = labels["label_value"].notna()
    labels["campaign_id"] = pd.NA

    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "features_26d.csv": feature,
        "feature_availability.csv": availability,
        "nodes.csv": nodes,
        "edges.csv": edges,
        "events.csv": edges[[
            "dataset_id", "source_id", "target_id", "campaign_id", "timestamp",
            "edge_type", "evidence_id",
        ]],
        "labels.csv": labels,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(output / filename, index=False)

    coverage = known_user_rows / processed_rows if processed_rows else 0.0
    campaign_coverage = len(edge_heaps) / len(campaigns_seen) if campaigns_seen else 0.0
    manifest = {
        "schema_version": "hypertrace.crypto-full-materialized.v1",
        "dataset_id": "crypto-campaign-full",
        "source_root": str(source.resolve()),
        "source_files": {
            "users.tsv": {"bytes": users_path.stat().st_size, "sha256": sha256(users_path)},
            "events.tsv": {"bytes": events_path.stat().st_size, "sha256": sha256(events_path)},
            **{
                name: {"bytes": (labelled / name).stat().st_size,
                       "rows_processed": file_rows[name], "sha256": sha256(labelled / name)}
                for name in COMMENT_FILES
            },
        },
        "counts": {
            "nodes": n, "labelled_activity_rows": processed_rows,
            "known_user_activity_rows": known_user_rows,
            "unmatched_user_rows": malformed_rows,
            "campaigns_seen": len(campaigns_seen),
            "campaigns_with_sampled_edges": len(edge_heaps),
            "sampled_edges": len(edges),
        },
        "coverage": {
            "known_user_activity": coverage,
            "campaign_edge_coverage": campaign_coverage,
        },
        "feature_columns": list(DEFAULT_FEATURE_COLUMNS),
        "feature_dim": len(DEFAULT_FEATURE_COLUMNS),
        "edge_sampling": {
            "method": "deterministic_bottom_k_hash_per_campaign_over_consecutive_cross_user_events",
            "edges_per_campaign": edges_per_campaign,
            "all_labelled_rows_contribute_to_features": True,
        },
        "label_contract": {
            "available": ["bounty_participation"],
            "not_available": ["bot_human", "cib_binary", "llm_origin"],
        },
        "excluded_raw_text": {
            "path": str(source / "unprocessed" / "comments_raw.tsv"),
            "reason": "31GB raw HTML retained as evidence source; not required for structural pretraining",
        },
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)
            }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--edges-per-campaign", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(materialize(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

