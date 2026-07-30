"""Create the auditable P2 artifact contract for one completed simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from data_processing.episode_manifest import EpisodeManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_features(source: Path, target: Path) -> None:
    frame = pd.read_csv(source, low_memory=False)
    if "user_id" not in frame.columns:
        raise ValueError("node feature file must contain user_id")
    if frame["user_id"].astype(str).duplicated().any():
        raise ValueError("node feature user_id must be unique")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--node-features", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-agents", required=True, type=int)
    parser.add_argument("--time-steps", required=True, type=int)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    db = args.db.expanduser().resolve()
    profiles_path = args.profiles.expanduser().resolve()
    features_source = args.node_features.expanduser().resolve()
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in (db, profiles_path, features_source):
        if not path.exists():
            raise FileNotFoundError(path)

    profiles = pd.read_csv(profiles_path, low_memory=False)
    required = {"user_id", "user_type"}
    if not required.issubset(profiles.columns):
        raise ValueError(f"profiles CSV missing {sorted(required - set(profiles.columns))}")
    profiles["user_id"] = profiles["user_id"].astype(str)
    if len(profiles) != args.num_agents or profiles["user_id"].duplicated().any():
        raise ValueError("profiles must contain exactly num-agents unique users")

    labels = profiles[["user_id", "user_type"]].copy()
    labels["is_bad"] = labels["user_type"].str.contains("bad", case=False).astype(int)
    labels["role"] = labels["user_type"].str.lower().map({
        "good": "organic",
        "bad": "independent_adversary",
        "bad_leader": "leader",
        "bad_member": "member",
    })
    is_bad = labels["is_bad"].eq(1)
    if args.scenario == "independent_attack":
        labels["campaign_id"] = ""
        labels.loc[is_bad, "campaign_id"] = labels.loc[is_bad, "user_id"].map(
            lambda value: f"{args.scenario}:s{args.seed}:user{value}"
        )
    else:
        labels["campaign_id"] = ""
        labels.loc[is_bad, "campaign_id"] = f"{args.scenario}:s{args.seed}"

    with sqlite3.connect(db) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        user_count = connection.execute("SELECT COUNT(*) FROM user").fetchone()[0]
        traces = pd.read_sql_query(
            "SELECT rowid AS source_rowid, user_id, created_at, action FROM trace",
            connection,
        )
    if user_count != args.num_agents:
        raise ValueError(f"DB contains {user_count} users, expected {args.num_agents}")
    traces["user_id"] = traces["user_id"].astype(str)
    latest = (
        traces.sort_values(["created_at", "source_rowid"], kind="stable")
        .drop_duplicates("user_id", keep="last")
    )
    event_targets = profiles[["user_id"]].merge(
        latest[["user_id", "action", "created_at", "source_rowid"]],
        on="user_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"action": "next_action", "created_at": "event_time"})
    event_targets["attack_phase"] = event_targets["next_action"].notna().map({
        True: "smoke_observed_action",
        False: "unavailable",
    })

    labels_path = Path(f"{prefix}.labels.csv")
    events_path = Path(f"{prefix}.event_targets.csv")
    features_path = Path(f"{prefix}.features.csv")
    labels.to_csv(labels_path, index=False)
    event_targets.to_csv(events_path, index=False)
    _copy_features(features_source, features_path)

    manifest_path = Path(f"{prefix}.manifest.json")
    manifest = EpisodeManifest(
        episode_id=f"sim:{args.scenario}:n{args.num_agents}:s{args.seed}:smoke",
        dataset_name="deeppersona_oasis",
        domain="synthetic",
        purpose="simulation_main",
        partition="pool",
        split_level="scenario",
        source_path=str(db),
        identity_scope="episode",
        scenario_id=args.scenario,
        simulation_seed=args.seed,
        num_agents=args.num_agents,
        time_steps=args.time_steps,
        attack_phases=("smoke_observed_action",),
        label_provenance={
            "bot": "generated",
            "role": "generated",
            "campaign": "generated",
            "next_action": "observed",
        },
        capabilities={
            "temporal": True,
            "raw_text": True,
            "ground_truth_roles": True,
            "ground_truth_campaigns": True,
        },
        artifacts={
            "profiles_csv": str(profiles_path),
            "features_csv": str(features_path),
            "labels_csv": str(labels_path),
            "event_targets_csv": str(events_path),
            "episode_manifest": str(manifest_path),
        },
        generator_metadata={
            "smoke_only": "true",
            "activation_policy": "fixed_smoke_agent_ids_plus_zero_random_activation",
            "forced_bad_leader_action": str(
                args.scenario == "leader_amplifier"
            ).lower(),
            "db_integrity": integrity,
            "observed_trace_users": str(int(latest["user_id"].nunique())),
        },
        source_sha256=_sha256(db),
        status="ready",
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "users": user_count,
        "observed_trace_rows": int(len(traces)),
        "observed_trace_users": int(latest["user_id"].nunique()),
        "labels": str(labels_path),
        "features": str(features_path),
        "event_targets": str(events_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
