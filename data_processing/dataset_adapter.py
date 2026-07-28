"""Read-only dataset adapters for heterogeneous HyperTrace inputs.

Adapters preserve the difference between observed data, supervised labels and
unavailable information.  In particular, the TwiBot V5 export is a static
open-neighborhood snapshot: it must not receive fabricated timestamps, tweet
IDs, campaign memberships or tactical-role ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import sqlite3
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetCapabilities:
    temporal: bool
    stable_post_ids: bool
    external_neighbors: bool
    ground_truth_roles: bool
    raw_text: bool
    interaction_target_ids: bool
    supervised_labels: bool

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class UnifiedDatasetBundle:
    dataset_kind: str
    capabilities: DatasetCapabilities
    core_users: pd.DataFrame
    boundary_users: pd.DataFrame
    labels: pd.DataFrame
    follow_edges: pd.DataFrame
    actions: pd.DataFrame
    warnings: List[str] = field(default_factory=list)

    def manifest(self) -> dict:
        manifest = {
            "dataset_kind": self.dataset_kind,
            "capabilities": self.capabilities.to_dict(),
            "counts": {
                "core_users": int(len(self.core_users)),
                "boundary_users": int(len(self.boundary_users)),
                "follow_edges": int(len(self.follow_edges)),
                "actions": int(len(self.actions)),
            },
            "warnings": list(self.warnings),
        }
        extra = getattr(self, "manifest_extra", None)
        if isinstance(extra, dict):
            manifest["extra"] = extra
        return manifest


def normalize_id(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    return text


def detect_dataset_kind(db_path: str) -> str:
    """Identify a supported DB contract without mutating the database."""
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        columns = {
            table: {
                row[1] for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                )
            }
            for table in tables
        }
    if (
        {"user", "follow", "agent_actions"}.issubset(tables)
        and {"agent_name", "action_type", "content"}.issubset(
            columns["agent_actions"]
        )
        and "post" not in tables
    ):
        return "twibot_static_v5"
    if "post" in tables and "user" in tables:
        return "simulation_event_db"
    return "unknown"


class TwiBotStaticAdapter:
    """Load the existing TwiBot V5 CSV/DB pair into a unified static bundle."""

    capabilities = DatasetCapabilities(
        temporal=False,
        stable_post_ids=False,
        external_neighbors=True,
        ground_truth_roles=False,
        raw_text=True,
        interaction_target_ids=False,
        supervised_labels=True,
    )

    def __init__(
        self,
        db_path: str,
        csv_path: Optional[str] = None,
        core_user_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.core_user_ids = (
            {normalized for value in core_user_ids
             if (normalized := normalize_id(value)) is not None}
            if core_user_ids is not None else None
        )

    def _load_csv(self) -> Optional[pd.DataFrame]:
        if self.csv_path is None:
            return None
        if not self.csv_path.exists():
            raise FileNotFoundError(f"TwiBot CSV does not exist: {self.csv_path}")
        frame = pd.read_csv(self.csv_path, low_memory=False)
        required = {
            "user_id", "followers_count", "following_count",
            "previous_tweets", "user_type",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"TwiBot CSV missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["user_id"] = frame["user_id"].map(normalize_id)
        if frame["user_id"].isna().any() or frame["user_id"].duplicated().any():
            raise ValueError("TwiBot CSV user_id must be non-null and unique")
        return frame

    @staticmethod
    def _content_node_id(row) -> str:
        if str(row.action_type).lower() == "like":
            digest = hashlib.sha256(str(row.content).encode("utf-8")).hexdigest()
            return f"external_text:{digest[:24]}"
        return f"twibot_action:{int(row.source_rowid)}"

    def load(self) -> UnifiedDatasetBundle:
        if not self.db_path.exists():
            raise FileNotFoundError(f"TwiBot DB does not exist: {self.db_path}")
        if detect_dataset_kind(str(self.db_path)) != "twibot_static_v5":
            raise ValueError("Database does not match the TwiBot static V5 contract")

        csv_frame = self._load_csv()
        with sqlite3.connect(self.db_path) as connection:
            users = pd.read_sql_query(
                'SELECT rowid AS source_rowid, * FROM "user"', connection
            )
            follows = pd.read_sql_query(
                "SELECT rowid AS source_rowid, * FROM follow", connection
            )
            actions = pd.read_sql_query(
                "SELECT rowid AS source_rowid, * FROM agent_actions", connection
            )

        for column in ("user_id",):
            users[column] = users[column].map(normalize_id)
        follows["follower_id"] = follows["follower_id"].map(normalize_id)
        follows["followee_id"] = follows["followee_id"].map(normalize_id)
        actions["agent_name"] = actions["agent_name"].map(normalize_id)
        if users["user_id"].isna().any() or users["user_id"].duplicated().any():
            raise ValueError("TwiBot DB user_id must be non-null and unique")

        if csv_frame is not None:
            core_ids = set(csv_frame["user_id"])
        elif self.core_user_ids is not None:
            core_ids = set(self.core_user_ids)
        else:
            core_ids = set(
                users.loc[users["user_type"].astype(str) != "boundary", "user_id"]
            )
        db_ids = set(users["user_id"])
        missing_core = core_ids.difference(db_ids)
        if missing_core:
            raise ValueError(
                f"{len(missing_core)} core CSV users are absent from the TwiBot DB"
            )

        public_users = users[["user_id", "followers", "following"]].copy()
        public_users["followers"] = pd.to_numeric(
            public_users["followers"], errors="coerce"
        ).fillna(0.0)
        public_users["following"] = pd.to_numeric(
            public_users["following"], errors="coerce"
        ).fillna(0.0)

        if csv_frame is not None:
            public_csv = csv_frame[[
                "user_id", "followers_count", "following_count",
                "previous_tweets",
            ]].copy()
            public_csv = public_csv.rename(columns={
                "followers_count": "followers",
                "following_count": "following",
            })
            # ``user_char`` is the public TwiBot profile description, not a
            # private personality label. Rename it at the adapter boundary.
            public_csv["bio"] = (
                csv_frame["user_char"].fillna("").astype(str)
                if "user_char" in csv_frame.columns else ""
            )
            core_users = public_csv
            labels = csv_frame[["user_id", "user_type"]].copy()
        else:
            core_users = public_users[public_users["user_id"].isin(core_ids)].copy()
            core_users["bio"] = ""
            core_users["previous_tweets"] = ""
            db_labels = users.loc[
                users["user_id"].isin(core_ids), ["user_id", "user_type"]
            ].copy()
            labels = db_labels

        labels["is_bad"] = (
            labels["user_type"].astype(str).str.lower().eq("bad").astype(int)
        )
        boundary_users = public_users[
            ~public_users["user_id"].isin(core_ids)
        ].copy()

        follows = follows.dropna(subset=["follower_id", "followee_id"]).copy()
        follows["evidence_id"] = follows["source_rowid"].map(
            lambda value: f"follow:rowid:{int(value)}"
        )
        grouped_follow = (
            follows.groupby(["follower_id", "followee_id"], sort=False)
            .agg(
                multiplicity=("source_rowid", "size"),
                evidence_ids=("evidence_id", list),
            )
            .reset_index()
        )
        grouped_follow["source_scope"] = grouped_follow["follower_id"].map(
            lambda value: "core" if value in core_ids else "boundary"
        )
        grouped_follow["target_scope"] = grouped_follow["followee_id"].map(
            lambda value: "core" if value in core_ids else "boundary"
        )

        actions = actions.dropna(subset=["agent_name"]).copy()
        actions["action_type"] = actions["action_type"].astype(str).str.lower()
        actions["content"] = actions["content"].fillna("").astype(str)
        actions["evidence_id"] = actions["source_rowid"].map(
            lambda value: f"agent_actions:rowid:{int(value)}"
        )
        actions["content_node_id"] = actions.apply(
            self._content_node_id, axis=1
        )
        actions["event_time"] = pd.NA
        actions["id_kind"] = actions["action_type"].map(
            lambda value: (
                "content_hash_without_original_post_id"
                if value == "like" else "derived_action_rowid"
            )
        )
        actions = actions.rename(columns={"agent_name": "actor_id"})

        warnings = [
            "Static TwiBot export has no event timestamps; temporal replay is disabled.",
            "Original tweet IDs were not retained; content node IDs are explicit derived IDs.",
            "Tactical roles and campaign membership are unavailable, not inferred ground truth.",
        ]
        return UnifiedDatasetBundle(
            dataset_kind="twibot_static_v5",
            capabilities=self.capabilities,
            core_users=core_users.reset_index(drop=True),
            boundary_users=boundary_users.reset_index(drop=True),
            labels=labels.reset_index(drop=True),
            follow_edges=grouped_follow.reset_index(drop=True),
            actions=actions.reset_index(drop=True),
            warnings=warnings,
        )
