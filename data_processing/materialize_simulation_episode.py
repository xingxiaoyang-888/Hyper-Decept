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


def _future_action_targets(
    profiles: pd.DataFrame,
    future_trace_path: Path | None,
    cutoff_time: str | None,
    cutoff_step: int | None = None,
    *,
    input_is_cutoff_snapshot: bool,
) -> tuple[pd.DataFrame, bool]:
    targets = profiles[["user_id"]].copy()
    if future_trace_path is None and cutoff_time is None and cutoff_step is None:
        return targets, False
    if future_trace_path is None:
        raise ValueError("future-trace is required for temporal supervision")
    if cutoff_step is None and cutoff_time is None:
        raise ValueError("cutoff-time is required when cutoff-step is unavailable")
    if not input_is_cutoff_snapshot:
        raise ValueError(
            "future action supervision requires --input-is-cutoff-snapshot"
        )
    if not future_trace_path.is_file():
        raise FileNotFoundError(future_trace_path)

    traces = pd.read_csv(future_trace_path, low_memory=False)
    required = {"user_id", "created_at", "action"}
    if not required.issubset(traces.columns):
        raise ValueError(
            f"future trace is missing columns: {sorted(required - set(traces.columns))}"
        )
    traces = traces.copy()
    traces["user_id"] = traces["user_id"].astype(str)
    if cutoff_step is not None:
        if "timestep" not in traces.columns:
            raise ValueError("formal future trace must contain timestep")
        numeric_steps = pd.to_numeric(traces["timestep"], errors="raise")
        if not (numeric_steps > cutoff_step).all():
            raise ValueError("future trace contains events at or before cutoff_step")
        traces["timestep"] = numeric_steps.astype(int)
        future = traces.copy()
        sort_columns = ["timestep", "_order"]
        cutoff_value = str(cutoff_time or f"step:{cutoff_step}")
    else:
        traces["created_at"] = pd.to_datetime(
            traces["created_at"], errors="raise", utc=True
        )
        cutoff = pd.to_datetime(cutoff_time, errors="raise", utc=True)
        future = traces[traces["created_at"] > cutoff].copy()
        sort_columns = ["created_at", "_order"]
        cutoff_value = cutoff.isoformat()
    future["_order"] = range(len(future))
    first = (
        future.sort_values(sort_columns, kind="stable")
        .drop_duplicates("user_id", keep="first")
    )
    target_columns = ["user_id", "action", "created_at"]
    if cutoff_step is not None:
        target_columns.append("timestep")
    targets = targets.merge(
        first[target_columns],
        on="user_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={
        "action": "next_action",
        "created_at": "target_time",
        "timestep": "target_timestep",
    })
    targets["cutoff_time"] = cutoff_value
    targets["attack_phase"] = targets["next_action"].notna().map({
        True: "post_cutoff_first_action",
        False: "unavailable",
    })
    return targets, True


def _validate_cutoff_snapshot(db: Path, cutoff_step: int) -> None:
    with sqlite3.connect(db) as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "trace" not in tables:
            raise ValueError("formal cutoff DB must contain a trace table")
        created_values = [
            row[0] for row in connection.execute(
                "SELECT created_at FROM trace WHERE created_at IS NOT NULL"
            )
        ]
    numeric = pd.to_numeric(pd.Series(created_values, dtype=object), errors="coerce")
    numeric = numeric.dropna()
    if not numeric.empty and (numeric / 3 > cutoff_step).any():
        raise ValueError("cutoff DB contains trace events after cutoff_step")


def _validate_activation_audit(
    path: Path,
    *,
    num_agents: int,
    time_steps: int,
    cutoff_step: int,
) -> dict:
    audit_path = path.expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "hyperdecept.activation-audit.v1",
        "policy": "budgeted_activity",
        "num_agents": num_agents,
        "time_steps": time_steps,
        "cutoff_step": cutoff_step,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"activation audit {key}={payload.get(key)!r}, expected {value!r}"
            )
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != time_steps:
        raise ValueError("activation audit must contain exactly one row per timestep")
    observed_before_cutoff: set[int] = set()
    for expected_step, row in enumerate(steps, start=1):
        if row.get("timestep") != expected_step:
            raise ValueError("activation audit timesteps must be contiguous and ordered")
        selected = row.get("selected")
        budget = row.get("budget")
        if not isinstance(selected, list) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("activation audit step has invalid selected list or budget")
        ids = [entry.get("agent_id") for entry in selected]
        if row.get("selected_count") != len(ids) or len(ids) > budget:
            raise ValueError("activation audit step violates its request budget")
        if len(set(ids)) != len(ids) or any(
            not isinstance(agent_id, int) or not 0 <= agent_id < num_agents
            for agent_id in ids
        ):
            raise ValueError("activation audit contains duplicate or invalid agent IDs")
        if expected_step <= cutoff_step:
            observed_before_cutoff.update(ids)
    if observed_before_cutoff != set(range(num_agents)):
        missing = num_agents - len(observed_before_cutoff)
        raise ValueError(
            f"activation audit misses {missing} agents before cutoff_step"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--node-features", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-agents", required=True, type=int)
    parser.add_argument("--time-steps", required=True, type=int)
    parser.add_argument("--cutoff-step", type=int)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--activation-audit", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--future-trace", type=Path)
    parser.add_argument("--cutoff-time")
    parser.add_argument(
        "--input-is-cutoff-snapshot",
        action="store_true",
        help="Assert that the DB/features contain no events after cutoff-time.",
    )
    args = parser.parse_args()
    if args.cutoff_step is not None and not 0 < args.cutoff_step < args.time_steps:
        raise ValueError("cutoff-step must be between 1 and time-steps - 1")
    if args.mode == "formal" and args.cutoff_step is None:
        raise ValueError("formal mode requires --cutoff-step")
    if args.mode == "formal" and args.activation_audit is None:
        raise ValueError("formal mode requires --activation-audit")

    db = args.db.expanduser().resolve()
    profiles_path = args.profiles.expanduser().resolve()
    features_source = args.node_features.expanduser().resolve()
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in (db, profiles_path, features_source):
        if not path.exists():
            raise FileNotFoundError(path)
    activation_audit = (
        args.activation_audit.expanduser().resolve()
        if args.activation_audit is not None else None
    )
    if args.mode == "formal":
        _validate_cutoff_snapshot(db, args.cutoff_step)
        _validate_activation_audit(
            activation_audit,
            num_agents=args.num_agents,
            time_steps=args.time_steps,
            cutoff_step=args.cutoff_step,
        )

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
    if user_count != args.num_agents:
        raise ValueError(f"DB contains {user_count} users, expected {args.num_agents}")
    future_trace = (
        args.future_trace.expanduser().resolve()
        if args.future_trace is not None else None
    )
    event_targets, temporal_supervision = _future_action_targets(
        profiles,
        future_trace,
        args.cutoff_time,
        args.cutoff_step,
        input_is_cutoff_snapshot=args.input_is_cutoff_snapshot,
    )
    if args.mode == "formal" and not temporal_supervision:
        raise ValueError("formal mode requires cutoff snapshot and future trace")
    if args.mode == "formal" and event_targets["next_action"].notna().sum() == 0:
        raise ValueError("formal mode requires at least one next_action target")

    labels_path = Path(f"{prefix}.labels.csv")
    events_path = Path(f"{prefix}.event_targets.csv")
    features_path = Path(f"{prefix}.features.csv")
    labels.to_csv(labels_path, index=False)
    event_targets.to_csv(events_path, index=False)
    _copy_features(features_source, features_path)

    manifest_path = Path(f"{prefix}.manifest.json")
    manifest = EpisodeManifest(
        episode_id=(
            f"sim:{args.scenario}:n{args.num_agents}:s{args.seed}:{args.mode}"
        ),
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
        cutoff_step=args.cutoff_step,
        attack_phases=("post_cutoff_first_action",) if temporal_supervision else (),
        label_provenance={
            "bot": "generated",
            "role": "generated",
            "campaign": "generated",
            **({"next_action": "observed"} if temporal_supervision else {}),
        },
        capabilities={
            "temporal": True,
            "raw_text": True,
            "ground_truth_roles": True,
            "ground_truth_campaigns": True,
            "next_action_targets": temporal_supervision,
        },
        artifacts={
            "profiles_csv": str(profiles_path),
            "features_csv": str(features_path),
            "labels_csv": str(labels_path),
            "event_targets_csv": str(events_path),
            "episode_manifest": str(manifest_path),
            **(
                {"activation_audit_json": str(activation_audit)}
                if activation_audit is not None else {}
            ),
            **(
                {"future_trace_csv": str(future_trace)}
                if future_trace is not None else {}
            ),
            **(
                {"cutoff_snapshot_db": str(db)}
                if args.mode == "formal" else {}
            ),
        },
        generator_metadata={
            "smoke_only": str(args.mode == "smoke").lower(),
            "activation_policy": (
                "fixed_smoke_agent_ids_plus_zero_random_activation"
                if args.mode == "smoke" else "budgeted_activity"
            ),
            "forced_bad_leader_action": str(
                args.scenario == "leader_amplifier"
            ).lower(),
            "db_integrity": integrity,
            "temporal_supervision": str(temporal_supervision).lower(),
            "cutoff_time": args.cutoff_time or "unavailable",
            "cutoff_step": str(args.cutoff_step or "unavailable"),
            "future_trace_sha256": (
                _sha256(future_trace) if future_trace is not None else "unavailable"
            ),
        },
        source_sha256=_sha256(db),
        status="ready",
    )
    manifest.write(manifest_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "users": user_count,
        "temporal_supervision": temporal_supervision,
        "future_action_targets": int(
            event_targets.get("next_action", pd.Series(dtype=str)).notna().sum()
        ),
        "labels": str(labels_path),
        "features": str(features_path),
        "event_targets": str(events_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
