"""Build deterministic, read-only time windows for audited external bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Support direct execution from the repository root on minimal server images.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_external_bundle import audit_bundle

SCHEMA_VERSION = "hypertrace.external-test-report.v1"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def build_external_report(bundle: Path, *, windows: int = 4,
                          max_events_per_window: int | None = None) -> dict[str, Any]:
    if windows <= 0:
        raise ValueError("windows must be positive")
    audit = audit_bundle(bundle)
    if audit["status"] == "not_ready":
        raise ValueError("external bundle failed audit")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    events = pd.read_csv(bundle / "events.csv", low_memory=False)
    required = {"event_id", "evidence_id", "actor_id", "timestamp"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events.csv missing required columns: {missing}")
    timestamps = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    if not timestamps.notna().any():
        # Older pandas releases do not support format="mixed" reliably.
        timestamps = events["timestamp"].map(
            lambda value: pd.to_datetime(value, errors="coerce", utc=True)
        )
    events["timestamp_utc"] = timestamps
    events = events.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc", kind="mergesort")
    if events.empty:
        raise ValueError("external bundle contains no parseable event timestamps")
    start, end = events["timestamp_utc"].min(), events["timestamp_utc"].max()
    span_ns = int((end - start).value)
    boundaries = [start + pd.to_timedelta(span_ns * i / windows, unit="ns") for i in range(windows + 1)]
    node_ids = set(pd.read_csv(bundle / "nodes.csv", usecols=["node_id"], low_memory=False)["node_id"].astype(str))
    rows = []
    for index in range(windows):
        left, right = boundaries[index], boundaries[index + 1]
        upper = events["timestamp_utc"] <= right if index == windows - 1 else events["timestamp_utc"] < right
        window = events[(events["timestamp_utc"] >= left) & upper]
        if max_events_per_window is not None:
            if max_events_per_window <= 0:
                raise ValueError("max_events_per_window must be positive")
            window = window.head(max_events_per_window)
        evidence = window["evidence_id"].dropna().astype(str)
        actors = window["actor_id"].dropna().astype(str)
        rows.append({
            "window_id": f"window:{index:03d}",
            "start_utc": left.isoformat(),
            "end_utc": right.isoformat(),
            "event_count": int(len(window)),
            "unique_actors": int(actors.nunique()),
            "evidence_count": int(evidence.nunique()),
            "evidence_coverage": float(evidence.nunique() / len(window)) if len(window) else 0.0,
            "actors_missing_from_nodes": int((~actors.isin(node_ids)).sum()),
        })
    checks = audit["checks"]
    artifacts = {}
    for name in ("nodes.csv", "edges.csv", "events.csv", "labels.csv", "manifest.json"):
        path = bundle / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle": str(bundle.resolve()),
        "dataset_id": manifest.get("dataset_id") or bundle.name,
        "audit_status": audit["status"],
        "evaluation_scope": audit.get("evaluation_scope"),
        "label_semantics": audit.get("label_semantics"),
        "supervised_metrics_allowed": bool(
            checks.get("supervised_metrics_allowed", False)
        ),
        "cib_supervised_metrics_allowed": bool(
            checks.get("cib_supervised_metrics_allowed", False)
        ),
        "io_supervised_metrics_allowed": bool(
            checks.get("io_supervised_metrics_allowed", False)
        ),
        "prediction_status": "not_run",
        "prediction_note": "No labels or predictions are inferred by this adapter.",
        "source_artifacts": artifacts,
        "time_range": {"start_utc": start.isoformat(), "end_utc": end.isoformat()},
        "windows": rows,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--max-events-per-window", type=int, default=None)
    args = parser.parse_args()
    report = build_external_report(args.bundle, windows=args.windows, max_events_per_window=args.max_events_per_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
