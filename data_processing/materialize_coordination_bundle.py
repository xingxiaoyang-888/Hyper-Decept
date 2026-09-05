"""Materialize real coordination data into a fixed 26D, auditable bundle.

Unavailable modalities are zero-imputed only in the model matrix and are
listed in ``feature_availability.csv``. No bot/human or LLM-origin labels are
created by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.coordination_adapter import load_dataset  # noqa: E402


FEATURE_COLUMNS = (
    *(f"Semantic_{i}" for i in range(8)),
    "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
    "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
    "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
    "Empathy_Gap_Mean", "Empathy_Gap_Max", "Dark_Triad_Mean",
    "Dark_Triad_Max", "Contagion_Mean", "Contagion_Max",
    "Volatility_Mean", "Volatility_Max",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _numeric(series, n: int) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(float) if series is not None else np.zeros(n)


def materialize(dataset_id: str, root: Path, output: Path) -> dict:
    bundle = load_dataset(dataset_id, root)
    nodes = bundle.nodes.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    ids = nodes["node_id"].tolist()
    n = len(nodes)
    values = pd.DataFrame(0.0, index=np.arange(n), columns=list(FEATURE_COLUMNS))
    available = pd.DataFrame(False, index=np.arange(n), columns=list(FEATURE_COLUMNS))
    available.insert(0, "node_id", ids)

    # Observable structural/activity covariates shared across sources.
    if {"followers", "following"}.issubset(nodes.columns):
        followers, following = _numeric(nodes["followers"], n), _numeric(nodes["following"], n)
        values["Follower_Following_Ratio"] = followers / (following + 1.0)
        available["Follower_Following_Ratio"] = True
    elif {"posts", "activity"}.issubset(nodes.columns):
        values["Action_Frequency"] = _numeric(nodes["activity"], n)
        available["Action_Frequency"] = True

    if not bundle.events.empty and "actor_id" in bundle.events.columns:
        counts = bundle.events["actor_id"].astype(str).value_counts()
        values["Action_Frequency"] = nodes["node_id"].map(counts).fillna(0).to_numpy(float)
        available["Action_Frequency"] = True
        if "text" in bundle.events.columns:
            text = bundle.events["text"].fillna("").astype(str)
            actor = bundle.events["actor_id"].astype(str)
            for col, pattern in (("URL_Ratio", r"https?://|www\."), ("Mention_Ratio", r"@[A-Za-z0-9_]"), ("Hashtag_Ratio", r"#[A-Za-z0-9_]"), ("Media_Ratio", r"pic\.twitter\.com|\.(?:jpg|jpeg|png|gif|mp4)\b")):
                hit = text.str.contains(pattern, case=False, regex=True).groupby(actor).mean()
                values[col] = nodes["node_id"].map(hit).fillna(0).to_numpy(float)
                available[col] = nodes["node_id"].isin(hit.index).to_numpy()
        if "timestamp" in bundle.events.columns:
            times = pd.to_datetime(bundle.events["timestamp"], errors="coerce", utc=True)
            hours = times.dt.hour
            entropy = {}
            for actor_id, group in pd.DataFrame({"actor": actor, "hour": hours}).dropna().groupby("actor"):
                p = group["hour"].value_counts(normalize=True).to_numpy()
                entropy[actor_id] = float(-(p * np.log(p + 1e-12)).sum())
            values["Temporal_Entropy"] = nodes["node_id"].map(entropy).fillna(0).to_numpy(float)
            available["Temporal_Entropy"] = nodes["node_id"].isin(entropy).to_numpy()

    # Topology degree is retained as an observable activity proxy. It is not a label.
    if not bundle.edges.empty and {"source_id", "target_id"}.issubset(bundle.edges.columns):
        degree = pd.concat([bundle.edges["source_id"].astype(str), bundle.edges["target_id"].astype(str)]).value_counts()
        values["Action_Frequency"] = nodes["node_id"].map(degree).fillna(values["Action_Frequency"]).to_numpy(float)
        available["Action_Frequency"] = available["Action_Frequency"] | nodes["node_id"].isin(degree.index).to_numpy()

    # Semantic/psychology fields are unavailable for the real bundles and stay
    # explicit zero + unavailable. This is not a hidden personality estimate.
    labels = bundle.labels.copy()
    values.insert(0, "user_id", ids)
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "features_26d.csv"
    availability_path = output / "feature_availability.csv"
    values.to_csv(features_path, index=False)
    available.to_csv(availability_path, index=False)
    nodes.to_csv(output / "nodes.csv", index=False)
    bundle.edges.to_csv(output / "edges.csv", index=False)
    bundle.events.to_csv(output / "events.csv", index=False)
    labels.to_csv(output / "labels.csv", index=False)
    manifest = {
        "schema_version": "hypertrace.coordination-materialized.v1",
        "dataset_id": dataset_id,
        "source_root": str(root.resolve()),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_dim": len(FEATURE_COLUMNS),
        "missing_feature_policy": "zero_impute_with_feature_availability_sidecar",
        "llm_origin_labels": False,
        "capabilities": bundle.capabilities.to_dict(),
        "warnings": bundle.warnings,
        "counts": {"nodes": len(nodes), "edges": len(bundle.edges), "events": len(bundle.events), "labels": len(labels)},
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("crypto-campaign", "uk2019-coordinated-behavior", "fake-accounts-activity"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.dataset, args.root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

