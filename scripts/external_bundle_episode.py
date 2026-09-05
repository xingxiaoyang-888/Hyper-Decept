"""Read-only adapter from an audited external bundle to ``EpisodeBatch``.

The adapter deliberately keeps the source label semantics intact.  It exposes
the official bad/good cohort as an ``information_operation_membership`` target
for external evaluation; it does not relabel it as bot, campaign, role, or
LLM-origin supervision.  Unsupported edge types remain in the source bundle
for provenance but are not silently renamed into synthetic relations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (str(ROOT), str(CHARACTER_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from joint_training import EpisodeBatch  # noqa: E402


# These are the only relations with a learned frozen-Base module.  Other
# source relations are intentionally not projected onto a different relation.
MODEL_EDGE_TYPES = {
    ("user", "posts", "tweet"),
    ("tweet", "authored_by", "user"),
    ("user", "retweets", "tweet"),
    ("tweet", "retweeted_by", "user"),
    ("user", "comments", "tweet"),
    ("tweet", "commented_by", "user"),
    ("user", "likes", "tweet"),
    ("tweet", "liked_by", "user"),
    ("user", "follows", "user"),
    ("user", "similar", "user"),
}


def _normalise(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise ValueError("node identifiers must be non-empty")
    return text


def _empty_targets(n_users: int) -> dict[str, torch.Tensor]:
    return {
        "coordination_targets": torch.zeros(n_users, dtype=torch.float32),
        "coordination_mask": torch.zeros(n_users, dtype=torch.bool),
        "role_targets": torch.full((n_users,), -1, dtype=torch.long),
        "role_mask": torch.zeros(n_users, dtype=torch.bool),
        "campaign_targets": torch.full((n_users,), -1, dtype=torch.long),
        "campaign_mask": torch.zeros(n_users, dtype=torch.bool),
        "temporal_action_targets": torch.full((n_users,), -1, dtype=torch.long),
        "temporal_action_mask": torch.zeros(n_users, dtype=torch.bool),
    }


def _read_node_ids(path: Path) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(path, usecols=["node_id", "node_type"], low_memory=False)
    frame["node_id"] = frame["node_id"].map(_normalise)
    frame["node_type"] = frame["node_type"].astype(str)
    users = frame.loc[frame["node_type"] == "user", "node_id"].tolist()
    tweets = frame.loc[frame["node_type"] == "tweet", "node_id"].tolist()
    if len(users) != len(set(users)) or len(tweets) != len(set(tweets)):
        raise ValueError("nodes.csv contains duplicate typed node IDs")
    if not users:
        raise ValueError("external bundle contains no user nodes")
    return users, tweets


def _attach_features(data: HeteroData, bundle: Path, users: list[str], tweets: list[str]) -> None:
    user_index = {value: index for index, value in enumerate(users)}
    tweet_index = {value: index for index, value in enumerate(tweets)}
    user_features = pd.read_csv(bundle / "features_observable18.csv", low_memory=False)
    expected = {"user_id", "Semantic_0", "Semantic_1", "Semantic_2", "Semantic_3",
                "Semantic_4", "Semantic_5", "Semantic_6", "Semantic_7",
                "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
                "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
                "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio"}
    missing = sorted(expected - set(user_features.columns))
    if missing:
        raise ValueError(f"features_observable18.csv missing columns: {missing}")
    user_features["user_id"] = user_features["user_id"].map(_normalise)
    if user_features["user_id"].duplicated().any():
        raise ValueError("features_observable18.csv contains duplicate users")
    feature_columns = [f"Semantic_{i}" for i in range(8)] + [
        "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
        "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
        "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
    ]
    lookup = user_features.set_index("user_id")
    matrix = np.zeros((len(users), len(feature_columns)), dtype=np.float32)
    for user_id, row in lookup.iterrows():
        index = user_index.get(user_id)
        if index is not None:
            matrix[index] = pd.to_numeric(row[feature_columns], errors="raise").to_numpy(dtype=np.float32)
    data["user"].x = torch.from_numpy(matrix)
    data["user"].node_ids = list(users)

    # Tweet features are observed only for source tweets.  Context tweets are
    # retained as nodes and receive zero features with no fabricated semantics.
    tweet_matrix = np.zeros((len(tweets), 14), dtype=np.float32)
    tweet_path = bundle / "tweet_features.csv"
    if tweet_path.is_file():
        tweet_features = pd.read_csv(tweet_path, low_memory=False)
        tweet_features["tweet_id"] = tweet_features["tweet_id"].map(_normalise)
        if tweet_features["tweet_id"].duplicated().any():
            raise ValueError("tweet_features.csv contains duplicate tweet IDs")
        columns = [column for column in tweet_features.columns if column != "tweet_id"]
        if len(columns) != 14:
            raise ValueError(f"tweet_features.csv must contain 14 observed columns, got {len(columns)}")
        for tweet_id, row in tweet_features.set_index("tweet_id").iterrows():
            index = tweet_index.get(tweet_id)
            if index is not None:
                tweet_matrix[index] = pd.to_numeric(row[columns], errors="raise").to_numpy(dtype=np.float32)
    data["tweet"].x = torch.from_numpy(tweet_matrix)
    data["tweet"].node_ids = list(tweets)


def _attach_edges(data: HeteroData, bundle: Path, users: list[str], tweets: list[str], *, chunksize: int = 250_000) -> None:
    user_index = {value: index for index, value in enumerate(users)}
    tweet_index = {value: index for index, value in enumerate(tweets)}
    buckets: dict[tuple[str, str, str], list[np.ndarray]] = {key: [] for key in MODEL_EDGE_TYPES}
    relation_by_name = {key[1]: key for key in MODEL_EDGE_TYPES}
    names = set(relation_by_name)
    for chunk in pd.read_csv(
        bundle / "edges.csv",
        usecols=["source_type", "source_id", "target_type", "target_id", "edge_type"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["edge_type"].isin(names)]
        if chunk.empty:
            continue
        # Vectorized string conversion and dictionary mapping avoid a Python
        # callback per edge; UAE contains millions of edges.
        for edge_name, group in chunk.groupby("edge_type", sort=False):
            edge_type = relation_by_name[str(edge_name)]
            source_type, _, target_type = edge_type
            source_map = user_index if source_type == "user" else tweet_index
            target_map = user_index if target_type == "user" else tweet_index
            src = group["source_id"].astype(str).map(source_map)
            dst = group["target_id"].astype(str).map(target_map)
            valid = src.notna() & dst.notna()
            if valid.any():
                buckets[edge_type].append(np.vstack([
                    src.loc[valid].to_numpy(dtype=np.int64),
                    dst.loc[valid].to_numpy(dtype=np.int64),
                ]))
    for edge_type, parts in buckets.items():
        if not parts:
            continue
        edge_index = torch.from_numpy(np.concatenate(parts, axis=1)).to(torch.long)
        data[edge_type].edge_index = edge_index
        # Explicit defaults make the reliability contract auditable and avoid
        # treating absent source reliability fields as learned evidence.
        count = edge_index.shape[1]
        data[edge_type].base_weight = torch.ones(count, dtype=torch.float32)
        data[edge_type].multiplicity = torch.ones(count, dtype=torch.float32)
        data[edge_type].temporal_sync = torch.zeros(count, dtype=torch.float32)
        data[edge_type].temporal_recency = torch.zeros(count, dtype=torch.float32)
        data[edge_type].temporal_available = torch.zeros(count, dtype=torch.float32)


def load_external_bundle_episode(
    bundle: str | Path,
    *,
    episode_id: str | None = None,
    load_labels: bool = True,
) -> EpisodeBatch:
    """Load an audited bundle without modifying any source artifact."""
    bundle = Path(bundle).expanduser().resolve()
    required = ["manifest.json", "nodes.csv", "edges.csv", "features_observable18.csv", "labels.csv"]
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"external bundle missing artifacts: {missing}")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "hypertrace.external-bundle.v2":
        raise ValueError("unsupported external bundle schema")
    if manifest.get("evaluation_scope") != "real_information_operation_external_evaluation":
        raise ValueError("bundle is not an audited real-operation external evaluation")
    users, tweets = _read_node_ids(bundle / "nodes.csv")
    graph = HeteroData()
    _attach_features(graph, bundle, users, tweets)
    _attach_edges(graph, bundle, users, tweets)
    # Keep the model metadata stable across domains.  Empty stores are
    # semantically different from fabricated edges: they allow a checkpoint
    # trained with the full relation vocabulary to run on a source that lacks
    # some relation types.
    for edge_type in MODEL_EDGE_TYPES:
        if edge_type not in graph.edge_types:
            graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph.external_manifest = manifest
    graph.unsupported_source_relations = [
        "mentions", "reply_to_user", "retweets_user"
    ]
    targets = _empty_targets(len(users))
    if load_labels:
        labels = pd.read_csv(bundle / "labels.csv", low_memory=False)
        labels["node_id"] = labels["node_id"].map(_normalise)
        label_lookup = labels.set_index("node_id")
        for index, user_id in enumerate(users):
            if user_id not in label_lookup.index:
                continue
            row = label_lookup.loc[user_id]
            if bool(row.get("is_known", False)):
                targets["coordination_targets"][index] = float(row["is_bad"])
                targets["coordination_mask"][index] = True
    return EpisodeBatch(
        episode_id=episode_id or str(manifest.get("operation_id") or bundle.name),
        domain="real",
        graph=graph,
        dataset_name="honduras_uae_io",
        **targets,
    )


__all__ = ["MODEL_EDGE_TYPES", "load_external_bundle_episode"]
