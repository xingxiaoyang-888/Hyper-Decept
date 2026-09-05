"""Unified, provenance-preserving adapters for real coordinated-behavior data.

The adapters deliberately keep coordination, bot/human, LLM-origin, and
campaign labels separate. Missing information is represented as ``NA`` and is
never inferred from a dataset's name or account category.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import pandas as pd


SCHEMA_VERSION = "hypertrace.coordination-bundle.v1"
DATASET_CATALOG = {
    "crypto-campaign": {
        "doi": "10.5281/zenodo.7813450",
        "license": "CC-BY-4.0",
        "intended_role": "real_campaign_adaptation",
    },
    "uk2019-coordinated-behavior": {
        "doi": "10.5281/zenodo.4647893",
        "license": "CC-BY-4.0",
        "intended_role": "real_topology_pretraining_or_external_evaluation",
    },
    "fake-accounts-activity": {
        "doi": "10.5281/zenodo.7391372",
        "license": "CC-BY-4.0",
        "intended_role": "real_cib_external_evaluation",
    },
}


@dataclass(frozen=True)
class CoordinationCapabilities:
    temporal: bool
    text: bool
    explicit_campaign_labels: bool
    explicit_cib_labels: bool
    graph_edges: bool
    llm_origin_labels: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class CoordinationBundle:
    dataset_id: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    events: pd.DataFrame
    labels: pd.DataFrame
    capabilities: CoordinationCapabilities
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "capabilities": self.capabilities.to_dict(),
            "counts": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "events": len(self.events),
                "labels": len(self.labels),
            },
            "columns": {
                "nodes": list(self.nodes.columns),
                "edges": list(self.edges.columns),
                "events": list(self.events.columns),
                "labels": list(self.labels.columns),
            },
            "warnings": list(self.warnings),
            "metadata": {**DATASET_CATALOG[self.dataset_id], **self.metadata},
        }


def _read(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def load_crypto_campaign(root: str | Path) -> CoordinationBundle:
    """Load the public Crypto-Campaign sample without inventing labels."""
    root = Path(root)
    sample = root / "extracted" / "samples"
    users = _read(sample / "users.tsv", sep="\t", low_memory=False)
    comments = _read(sample / "comments_raw.tsv", sep="\t", low_memory=False)
    events = _read(sample / "events.tsv", sep="\t", low_memory=False)
    nodes = users.rename(columns={"user_id": "node_id"}).copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes.insert(0, "dataset_id", "crypto-campaign")
    nodes["node_type"] = "forum_user"
    events = events.rename(columns={"user_id": "actor_id", "post_time ": "timestamp"})
    events["event_id"] = "event:" + events["comment_id"].astype(str)
    events["campaign_id"] = events["thread_id"].astype(str)
    events["event_type"] = "campaign_event"
    events["text"] = events.get("post_tex", pd.Series("", index=events.index)).fillna("")
    comments = comments.rename(columns={"user_id": "actor_id", "post_time": "timestamp"})
    comments["event_id"] = "comment:" + comments["comment_id"].astype(str)
    comments["campaign_id"] = comments["thread_id"].astype(str)
    comments["event_type"] = "comment"
    comments["text"] = comments.get("post_html", pd.Series("", index=comments.index)).fillna("")
    event_frame = pd.concat(
        [events[["event_id", "actor_id", "timestamp", "event_type", "text", "campaign_id"]],
         comments[["event_id", "actor_id", "timestamp", "event_type", "text", "campaign_id"]]],
        ignore_index=True,
    )
    event_frame.insert(0, "dataset_id", "crypto-campaign")
    event_frame["evidence_id"] = event_frame["event_id"]
    # Convert observed shared-thread participation into a user-user relation
    # for the current user-node Lorentz pretrainer. This is a derived topology
    # edge, not a fabricated campaign label; its evidence is the source thread.
    rows = []
    for thread_id, group in comments.dropna(subset=["thread_id", "actor_id"]).groupby("thread_id"):
        actors = sorted({str(value) for value in group["actor_id"]})
        timestamp = group["timestamp"].dropna().astype(str).min() if "timestamp" in group else ""
        for source_id, target_id in combinations(actors, 2):
            rows.append({
                "dataset_id": "crypto-campaign",
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": "co_participates_campaign",
                "campaign_id": str(thread_id),
                "timestamp": timestamp,
                "evidence_id": f"thread:{thread_id}:pair:{source_id}:{target_id}",
            })
    edges = pd.DataFrame(rows, columns=[
        "dataset_id", "source_id", "target_id", "edge_type", "campaign_id",
        "timestamp", "evidence_id",
    ])
    labels = nodes[["dataset_id", "node_id", "bounty_participant"]].rename(
        columns={"bounty_participant": "label_value"}
    ).copy()
    labels["label_task"] = "bounty_participation"
    labels["campaign_id"] = pd.NA
    labels["label_source"] = "annotated_or_observed_dataset_field"
    labels["is_known"] = labels["label_value"].notna()
    return CoordinationBundle(
        "crypto-campaign", nodes, edges, event_frame, labels,
        CoordinationCapabilities(True, True, True, False, True),
        [
            "Bounty participation is campaign evidence, not a generic CIB label.",
            "No LLM-origin label is present; use for real campaign adaptation/evaluation.",
        ],
        {"domain": "crypto_forum", "sample": True},
    )


def load_uk2019(root: str | Path) -> CoordinationBundle:
    """Load UK 2019 superspreader topology; tweet text is not fabricated."""
    root = Path(root) / "extracted"
    nodes = _read(root / "superspreader-nodes.csv")
    edges = _read(root / "superspreader-edges.csv")
    nodes = nodes.rename(columns={"id": "node_id", "cluster": "topology_cluster"})
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes.insert(0, "dataset_id", "uk2019-coordinated-behavior")
    nodes["node_type"] = "twitter_account"
    edges = edges.rename(columns={"source": "source_id", "target": "target_id"})
    edges.insert(0, "dataset_id", "uk2019-coordinated-behavior")
    edges["timestamp"] = pd.NA
    edges["campaign_id"] = pd.NA
    edges["evidence_id"] = [f"edge:row:{index}" for index in edges.index]
    edges["edge_type"] = "superspreader_relation"
    labels = nodes[["dataset_id", "node_id", "topology_cluster"]].rename(
        columns={"topology_cluster": "label_value"}
    ).copy()
    labels["label_task"] = "topology_cluster"
    labels["campaign_id"] = pd.NA
    labels["label_source"] = "annotated_cluster"
    labels["is_known"] = labels["label_value"].notna()
    return CoordinationBundle(
        "uk2019-coordinated-behavior", nodes, edges, pd.DataFrame(
            columns=["actor_id", "timestamp", "event_type", "text"]
        ), labels,
        CoordinationCapabilities(False, False, False, False, True),
        [
            "Cluster IDs are topology communities, not campaign or binary CIB ground truth.",
            "Public release contains IDs/topology but no complete tweet text or event timestamps.",
        ],
        {"domain": "twitter", "release": "superspreader_nodes_edges"},
    )


def load_fake_accounts(root: str | Path) -> CoordinationBundle:
    """Load Fake Accounts Activity, preserving unknown CIB as missing."""
    root = Path(root)
    users = _read(root / "users_ids.csv", low_memory=False)
    nodes = users.rename(columns={"user.id_str": "node_id", "cib": "campaign_id"})
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes.insert(0, "dataset_id", "fake-accounts-activity")
    nodes["node_type"] = "twitter_account"
    labels = nodes[["dataset_id", "node_id", "campaign_id"]].copy()
    labels["label_task"] = "cib_campaign_membership"
    labels["label_value"] = labels["campaign_id"]
    labels["is_known"] = labels["campaign_id"].notna()
    labels["label_source"] = "annotated_cib_field"
    events = pd.DataFrame(columns=["actor_id", "timestamp", "event_type", "text"])
    return CoordinationBundle(
        "fake-accounts-activity", nodes, pd.DataFrame(
            columns=["source_id", "target_id", "edge_type"]
        ), events, labels,
        CoordinationCapabilities(False, False, True, True, False),
        ["Blank CIB is unknown, not a human/negative label."],
        {"domain": "twitter", "tweets_file_optional": "tweets.csv"},
    )


def load_dataset(dataset_id: str, root: str | Path) -> CoordinationBundle:
    loaders = {
        "crypto-campaign": load_crypto_campaign,
        "uk2019-coordinated-behavior": load_uk2019,
        "fake-accounts-activity": load_fake_accounts,
    }
    try:
        return loaders[dataset_id](root)
    except KeyError as exc:
        raise ValueError(f"unsupported coordination dataset: {dataset_id}") from exc
