"""Streaming adapter for the original TwiBot-22 release.

The adapter extracts only a requested core-user open neighborhood.  It never
loads the 170M-row edge file or the 88M tweets into memory at once, and it
keeps raw relation names/IDs alongside normalized action fields so that no
edge semantics are silently invented.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .dataset_adapter import (
    DatasetCapabilities,
    UnifiedDatasetBundle,
    normalize_id,
)

logger = logging.getLogger(__name__)


USER_RELATIONS = {"following", "followers", "followed"}
CONTENT_RELATIONS = {
    "post", "like", "retweeted", "quoted", "replied_to", "replied",
    "pinned", "own", "discuss", "mentioned", "contain", "membership",
}
ACTION_RELATION_MAP = {
    "post": "post",
    "like": "like",
    "retweeted": "retweet",
    "quoted": "quote",
    "replied_to": "reply",
    "replied": "reply",
}


def _canonical_id(value) -> Optional[str]:
    """Normalize IDs without converting large integers through float."""
    return normalize_id(value)


def _id_aliases(value: str) -> Set[str]:
    value = str(value)
    aliases = {value}
    if value.startswith("u") and value[1:].isdigit():
        aliases.add(value[1:])
    elif value.isdigit():
        aliases.add(f"u{value}")
    return aliases


def _iter_json_records(path: Path) -> Iterator[dict]:
    try:
        import ijson
    except ImportError:
        ijson = None
    if ijson is not None:
        # ijson consumes bytes.  Feeding it a text wrapper forces a costly
        # encode/decode bridge for every block, which is material when the
        # official TwiBot-22 tweet shards exceed 100 GB.
        with path.open("rb") as handle:
            try:
                first = b""
                while True:
                    char = handle.read(1)
                    if not char:
                        return
                    if not char.isspace():
                        first = char
                        break
                handle.seek(0)
                if first == b"[":
                    yield from ijson.items(handle, "item")
                else:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
                return
            except (ijson.common.JSONError, json.JSONDecodeError):
                pass
    with path.open("r", encoding="utf-8-sig") as handle:
        decoder = json.JSONDecoder()
        text = handle.read()
        if text.lstrip().startswith("["):
            values = decoder.decode(text)
            yield from values
        else:
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)


def _read_id_column(path: Path, column: str) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path.name} missing required column '{column}'")
        values = []
        for row in reader:
            value = _canonical_id(row.get(column))
            if value is not None:
                values.append(value)
        return values


class TwiBot22RawAdapter:
    """Read a bounded core-user neighborhood from original TwiBot-22 files.

    Parameters are deterministic.  ``max_edges_per_relation`` and
    ``max_actions_per_user`` are optional safety caps; ``None`` preserves all
    matching rows.  No timestamps, IDs, labels or roles are fabricated.
    """

    capabilities = DatasetCapabilities(
        temporal=True,
        stable_post_ids=True,
        external_neighbors=True,
        ground_truth_roles=False,
        raw_text=True,
        interaction_target_ids=True,
        supervised_labels=True,
    )

    def __init__(
        self,
        twibot_dir: str,
        core_user_ids: Iterable[str],
        *,
        max_edges_per_relation: Optional[int] = None,
        max_actions_per_user: Optional[int] = None,
        max_posts_per_user: Optional[int] = None,
        edge_chunksize: int = 250_000,
    ) -> None:
        self.root = Path(twibot_dir)
        self.core_user_ids = {
            value
            for item in core_user_ids
            if (value := _canonical_id(item)) is not None
        }
        if not self.core_user_ids:
            raise ValueError("core_user_ids must contain at least one ID")
        if any(
            value is not None and int(value) < 1
            for value in (max_edges_per_relation, max_actions_per_user, max_posts_per_user)
        ):
            raise ValueError("row caps must be positive or None")
        if edge_chunksize <= 0:
            raise ValueError("edge_chunksize must be positive")
        self.max_edges_per_relation = max_edges_per_relation
        self.max_actions_per_user = max_actions_per_user
        self.max_posts_per_user = max_posts_per_user
        self.edge_chunksize = edge_chunksize

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing TwiBot-22 file: {path}")
        return path

    def _load_labels(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        labels_path = self._path("label.csv")
        split_path = self._path("split.csv")
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        splits = pd.read_csv(split_path, dtype=str, keep_default_na=False)
        if not {"id", "label"}.issubset(labels.columns):
            raise ValueError("label.csv must contain id and label")
        if not {"id", "split"}.issubset(splits.columns):
            raise ValueError("split.csv must contain id and split")
        labels["user_id"] = labels["id"].map(_canonical_id)
        splits["user_id"] = splits["id"].map(_canonical_id)
        label_map = dict(zip(labels["user_id"], labels["label"].str.lower()))
        split_map = dict(zip(splits["user_id"], splits["split"].str.lower()))
        missing = self.core_user_ids.difference(label_map)
        if missing:
            raise ValueError(f"{len(missing)} core IDs are absent from label.csv")
        rows = []
        for user_id in sorted(self.core_user_ids):
            label = label_map[user_id]
            if label not in {"bot", "human"}:
                raise ValueError(f"Unexpected TwiBot label for {user_id}: {label}")
            rows.append({
                "user_id": user_id,
                "label": label,
                "user_type": "bad" if label == "bot" else "good",
                "is_bad": int(label == "bot"),
                "split": split_map.get(user_id),
            })
        return pd.DataFrame(rows), label_map

    def _scan_edges(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Set[str]]:
        edge_path = self._path("edge.csv")
        selected: List[dict] = []
        relation_counts = Counter()
        relation_limits = defaultdict(int)
        action_counts = defaultdict(int)
        boundary_users: Set[str] = set()
        reader = pd.read_csv(
            edge_path,
            dtype=str,
            keep_default_na=False,
            chunksize=self.edge_chunksize,
        )
        for chunk in reader:
            required = {"source_id", "target_id", "relation"}
            if not required.issubset(chunk.columns):
                raise ValueError("edge.csv must contain source_id, target_id, relation")
            for row in chunk.itertuples(index=False):
                source = _canonical_id(getattr(row, "source_id"))
                target = _canonical_id(getattr(row, "target_id"))
                relation = str(getattr(row, "relation")).strip().lower()
                if source is None or target is None or relation not in CONTENT_RELATIONS | USER_RELATIONS:
                    continue
                source_core = source in self.core_user_ids
                target_core = target in self.core_user_ids
                if not (source_core or target_core):
                    continue
                if relation in USER_RELATIONS:
                    if source_core and not target_core:
                        boundary_users.add(target)
                    if target_core and not source_core:
                        boundary_users.add(source)
                if source_core and not target_core and relation in USER_RELATIONS:
                    boundary_users.add(target)
                if target_core and not source_core and relation in USER_RELATIONS:
                    boundary_users.add(source)
                if self.max_edges_per_relation is not None and relation_limits[relation] >= self.max_edges_per_relation:
                    continue
                if relation in ACTION_RELATION_MAP and source_core:
                    if (
                        self.max_actions_per_user is not None
                        and action_counts[source] >= self.max_actions_per_user
                    ):
                        continue
                    action_counts[source] += 1
                relation_limits[relation] += 1
                relation_counts[relation] += 1
                selected.append({
                    "source_id": source,
                    "target_id": target,
                    "relation": relation,
                    "source_is_core": source_core,
                    "target_is_core": target_core,
                    "evidence_id": f"edge:{relation}:{len(selected)}",
                })
        edges = pd.DataFrame(selected)
        if edges.empty:
            edges = pd.DataFrame(columns=[
                "source_id", "target_id", "relation", "source_is_core",
                "target_is_core", "evidence_id",
            ])
        follow = edges[edges["relation"].isin(USER_RELATIONS)].copy()
        if not follow.empty:
            follow = follow.rename(columns={
                "source_id": "follower_id", "target_id": "followee_id",
            })
            follow["multiplicity"] = 1.0
            grouped = follow.groupby(
                ["follower_id", "followee_id", "relation"], sort=False
            ).agg(
                multiplicity=("multiplicity", "sum"),
                evidence_ids=("evidence_id", list),
            ).reset_index()
            grouped["source_scope"] = grouped["follower_id"].map(
                lambda value: "core" if value in self.core_user_ids else "boundary"
            )
            grouped["target_scope"] = grouped["followee_id"].map(
                lambda value: "core" if value in self.core_user_ids else "boundary"
            )
            follow = grouped
        else:
            follow = pd.DataFrame(columns=[
                "follower_id", "followee_id", "relation", "multiplicity",
                "evidence_ids", "source_scope", "target_scope",
            ])
        actions = edges[edges["relation"].isin(ACTION_RELATION_MAP)].copy()
        if not actions.empty:
            actions["action_type"] = actions["relation"].map(ACTION_RELATION_MAP)
        else:
            actions["action_type"] = pd.Series(dtype=str)
        return edges, follow, actions, boundary_users

    def _load_users(self, user_ids: Set[str]) -> pd.DataFrame:
        rows = []
        aliases = {
            alias: user_id
            for user_id in user_ids
            for alias in _id_aliases(user_id)
        }
        for record in _iter_json_records(self._path("user.json")):
            raw_id = _canonical_id(record.get("id"))
            user_id = aliases.get(raw_id) if raw_id is not None else None
            if user_id is None:
                continue
            metrics = record.get("public_metrics") or {}
            rows.append({
                "user_id": user_id,
                "followers": pd.to_numeric(metrics.get("followers_count"), errors="coerce"),
                "following": pd.to_numeric(metrics.get("following_count"), errors="coerce"),
                "bio": str(record.get("description") or ""),
                "created_at": record.get("created_at"),
                "username": record.get("username"),
                "verified": record.get("verified"),
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=[
                "user_id", "followers", "following", "bio", "created_at",
                "username", "verified",
            ])
        frame = frame.drop_duplicates("user_id", keep="first")
        for column in ("followers", "following"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        return frame

    def _load_posts(self, tweet_ids: Set[str], core_ids: Set[str]) -> pd.DataFrame:
        aliases = {alias for tweet_id in tweet_ids for alias in _id_aliases(tweet_id)}
        author_aliases = {
            alias: user_id for user_id in core_ids for alias in _id_aliases(user_id)
        }
        rows: List[dict] = []
        per_author = Counter()
        for index in range(9):
            path = self.root / f"tweet_{index}.json"
            if not path.exists():
                continue
            for record in _iter_json_records(path):
                tweet_id = _canonical_id(record.get("id"))
                author_raw = _canonical_id(record.get("author_id"))
                author_id = author_aliases.get(author_raw)
                # Edge scanning already selected the required tweet IDs. Do
                # not include every tweet authored by a core user; that could
                # turn a bounded open-neighborhood load into a full crawl.
                if tweet_id is None or tweet_id not in aliases:
                    continue
                if author_id is not None:
                    if (
                        self.max_posts_per_user is not None
                        and per_author[author_id] >= self.max_posts_per_user
                    ):
                        continue
                    per_author[author_id] += 1
                metrics = record.get("public_metrics") or {}
                references = record.get("referenced_tweets") or []
                rows.append({
                    "post_id": tweet_id,
                    "author_id": author_id or author_raw,
                    "content": str(record.get("text") or ""),
                    "created_at": record.get("created_at"),
                    "conversation_id": _canonical_id(record.get("conversation_id")),
                    "in_reply_to_user_id": _canonical_id(record.get("in_reply_to_user_id")),
                    "original_post_id": (
                        _canonical_id(references[0].get("id"))
                        if references and isinstance(references[0], dict) else None
                    ),
                    "reference_types": [
                        item.get("type") for item in references
                        if isinstance(item, dict) and item.get("type")
                    ],
                    "num_likes": metrics.get("like_count", 0),
                    "num_replies": metrics.get("reply_count", 0),
                    "num_retweets": metrics.get("retweet_count", 0),
                    "num_quotes": metrics.get("quote_count", 0),
                    "evidence_id": f"post:{tweet_id}",
                })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=[
                "post_id", "author_id", "content", "created_at",
                "conversation_id", "in_reply_to_user_id", "original_post_id",
                "reference_types", "num_likes", "num_replies", "num_retweets",
                "num_quotes", "evidence_id",
            ])
        return frame.drop_duplicates("post_id", keep="first")

    @staticmethod
    def _build_actions(actions: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
        if actions.empty:
            return pd.DataFrame(columns=[
                "actor_id", "target_id", "action_type", "relation",
                "content_node_id", "post_id", "content", "event_time",
                "evidence_id", "source_id", "source_is_core", "target_is_core",
            ])
        post_ids = set(posts["post_id"].astype(str)) if not posts.empty else set()
        post_lookup = posts.set_index("post_id").to_dict("index") if not posts.empty else {}
        rows = []
        for row in actions.itertuples(index=False):
            source = str(row.source_id)
            target = str(row.target_id)
            post_id = target if target in post_ids else (source if source in post_ids else None)
            post = post_lookup.get(post_id, {}) if post_id else {}
            rows.append({
                "actor_id": source if bool(row.source_is_core) else None,
                "target_id": target,
                "action_type": row.action_type,
                "relation": row.relation,
                "content_node_id": f"tweet:{post_id}" if post_id else f"entity:{target}",
                "post_id": post_id,
                "content": post.get("content", ""),
                "event_time": post.get("created_at"),
                "evidence_id": row.evidence_id,
                "source_id": source,
                "source_is_core": bool(row.source_is_core),
                "target_is_core": bool(row.target_is_core),
            })
        return pd.DataFrame(rows)

    def load(self) -> UnifiedDatasetBundle:
        labels, _label_map = self._load_labels()
        edges, follow, raw_actions, boundary_ids = self._scan_edges()
        user_ids = set(self.core_user_ids) | boundary_ids
        users = self._load_users(user_ids)
        core_users = users[users["user_id"].isin(self.core_user_ids)].copy()
        boundary_users = users[users["user_id"].isin(boundary_ids)].copy()
        for frame in (core_users, boundary_users):
            if "previous_tweets" not in frame.columns:
                frame["previous_tweets"] = ""
        missing_core = self.core_user_ids.difference(set(core_users["user_id"]))
        if missing_core:
            raise ValueError(
                f"{len(missing_core)} core users are absent from user.json"
            )
        tweet_ids = set()
        for row in raw_actions.itertuples(index=False):
            tweet_ids.update(_id_aliases(str(row.target_id)))
            tweet_ids.update(_id_aliases(str(row.source_id)))
        posts = self._load_posts(tweet_ids, self.core_user_ids)
        actions = self._build_actions(raw_actions, posts)
        labels = labels.copy()
        labels["data_split"] = labels["split"]
        labels = labels.drop(columns=["split"])
        warnings = [
            "TwiBot-22 relation semantics are preserved in raw relation/source/target columns.",
            "Tactical roles and campaign membership are unavailable, not inferred ground truth.",
            "Boundary users are limited to user-user relations incident on the requested core set.",
        ]
        bundle = UnifiedDatasetBundle(
            dataset_kind="twibot22_raw",
            capabilities=self.capabilities,
            core_users=core_users.reset_index(drop=True),
            boundary_users=boundary_users.reset_index(drop=True),
            labels=labels.reset_index(drop=True),
            follow_edges=follow.reset_index(drop=True),
            actions=actions.reset_index(drop=True),
            warnings=warnings,
        )
        # Optional raw relations/posts are attached without breaking the M1
        # bundle constructor or existing graph-builder callers.
        bundle.relations = edges.reset_index(drop=True)
        bundle.posts = posts.reset_index(drop=True)
        bundle.manifest_extra = {
            "raw_relation_counts": dict(Counter(edges["relation"])) if not edges.empty else {},
            "post_count": int(len(posts)),
            "stable_post_ids": True,
        }
        return bundle


def load_core_ids(path: str) -> List[str]:
    """Read the newline ID list produced for the 1000-user sample."""
    values = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = _canonical_id(line)
            if value is not None:
                values.append(value)
    if len(values) != len(set(values)):
        raise ValueError("core ID file contains duplicate IDs")
    return values


def load_materialized_bundle(bundle_dir: str | Path) -> UnifiedDatasetBundle:
    """Load a previously exported smoke bundle without rescanning raw shards.

    ``prepare_p2_smoke_data twibot`` already records the bounded raw-adapter
    result as CSV files plus an integrity manifest.  Training must consume
    those exact artifacts; rereading all nine 100-GB-scale tweet shards would
    be both wasteful and capable of drifting from the audited snapshot.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "adapter_manifest.json"
    required = {
        "core_users": root / "core_users.csv",
        "boundary_users": root / "boundary_users.csv",
        "labels": root / "labels.csv",
        "follow_edges": root / "follow_edges.csv",
        "actions": root / "actions.csv",
        "posts": root / "posts.csv",
    }
    missing = [
        str(path)
        for path in [manifest_path, *required.values()]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"materialized TwiBot bundle is incomplete: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_kind") != "twibot22_raw":
        raise ValueError("adapter manifest is not a raw TwiBot-22 bundle")

    declared_artifacts = manifest.get("artifacts") or {}
    for name, path in required.items():
        expected_sha256 = (declared_artifacts.get(name) or {}).get("sha256")
        if not expected_sha256:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"materialized {name} SHA-256 mismatch")

    frames = {
        name: pd.read_csv(path, low_memory=False)
        for name, path in required.items()
    }
    id_columns = {
        "core_users": ("user_id",),
        "boundary_users": ("user_id",),
        "labels": ("user_id",),
        "follow_edges": ("follower_id", "followee_id"),
        "actions": (
            "actor_id", "target_id", "content_node_id", "post_id", "source_id"
        ),
        "posts": (
            "post_id", "author_id", "conversation_id", "in_reply_to_user_id",
            "original_post_id",
        ),
    }
    for name, columns in id_columns.items():
        frame = frames[name]
        for column in columns:
            if column in frame.columns:
                frame[column] = frame[column].map(_canonical_id)

    def parse_list(value):
        if isinstance(value, list):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return [str(value)]
        return parsed if isinstance(parsed, list) else [parsed]

    if "evidence_ids" in frames["follow_edges"].columns:
        frames["follow_edges"]["evidence_ids"] = frames["follow_edges"][
            "evidence_ids"
        ].map(parse_list)
    if "reference_types" in frames["posts"].columns:
        frames["posts"]["reference_types"] = frames["posts"][
            "reference_types"
        ].map(parse_list)

    expected_counts = manifest.get("counts") or {}
    for name in ("core_users", "boundary_users", "follow_edges", "actions"):
        expected = expected_counts.get(name)
        if expected is not None and int(expected) != len(frames[name]):
            raise ValueError(
                f"materialized {name} count mismatch: {len(frames[name])} != {expected}"
            )
    if frames["core_users"]["user_id"].duplicated().any():
        raise ValueError("materialized core user IDs must be unique")

    bundle = UnifiedDatasetBundle(
        dataset_kind="twibot22_raw",
        capabilities=TwiBot22RawAdapter.capabilities,
        core_users=frames["core_users"],
        boundary_users=frames["boundary_users"],
        labels=frames["labels"],
        follow_edges=frames["follow_edges"],
        actions=frames["actions"],
        warnings=list(manifest.get("warnings") or []),
    )
    bundle.posts = frames["posts"]
    relations_path = root / "relations.csv"
    bundle.relations = (
        pd.read_csv(relations_path, low_memory=False)
        if relations_path.is_file() else pd.DataFrame()
    )
    bundle.manifest_extra = dict(manifest.get("extra") or {})
    bundle.materialized_manifest = manifest
    return bundle
