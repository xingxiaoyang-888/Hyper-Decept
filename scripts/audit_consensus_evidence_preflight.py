"""Audit label-blind consensus explanation packets without opening labels.csv."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


OPERATIONS = ("honduras", "uae")
FORBIDDEN_KEY_FRAGMENTS = ("psych", "empathy", "dark_triad", "role_label")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            errors.append(f"non-finite value at {path}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", errors)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                errors.append(f"excluded psychology/role key at {path}.{key}")
            _walk(child, f"{path}.{key}", errors)


def _source_ids(path: Path, wanted: set[str]) -> set[str]:
    found: set[str] = set()
    if not wanted:
        return found
    for chunk in pd.read_csv(
        path, usecols=["evidence_id"], dtype=str, chunksize=250_000,
        keep_default_na=False,
    ):
        found.update(set(chunk["evidence_id"]).intersection(wanted))
        if found == wanted:
            break
    return found


def _audit_operation(root: Path, operation: str) -> dict[str, Any]:
    directory = root / operation
    errors: list[str] = []
    manifest_path = directory / "manifest.json"
    packets_path = directory / "packets.jsonl"
    if not manifest_path.is_file() or not packets_path.is_file():
        return {
            "operation": operation,
            "status": "failed",
            "errors": [f"missing manifest or packets under {directory}"],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packets = [
        json.loads(line) for line in packets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest.get("status") != "passed":
        errors.append("generation manifest status is not passed")
    if manifest.get("packet_count") != len(packets) or len(packets) != 3:
        errors.append("packet_count must equal the preflight Top-3 budget")
    if manifest.get("packets_sha256") != _sha256(packets_path):
        errors.append("packets SHA-256 mismatch")
    for field in ("labels_consumed", "target_label_values_consumed"):
        if manifest.get(field) is not False:
            errors.append(f"manifest {field} must be false")
    if manifest.get("eligibility_membership_consumed") is not True:
        errors.append("eligibility membership disclosure is missing")

    event_ids: set[str] = set()
    edge_ids: set[str] = set()
    bundle: Path | None = None
    changes: list[float] = []
    margins: list[float] = []
    for index, packet in enumerate(packets):
        where = f"packet[{index}]"
        _walk(packet, where, errors)
        if packet.get("operation") != operation:
            errors.append(f"{where}: operation mismatch")
        if packet.get("labels_consumed") is not False:
            errors.append(f"{where}: labels_consumed must be false")
        reference = packet.get("reference_universe") or {}
        if reference.get("target_label_values_consumed") is not False:
            errors.append(f"{where}: target label values were not isolated")
        count = int(reference.get("account_count") or 0)
        if count <= 0 or reference.get("eligibility_membership_consumed") is not True:
            errors.append(f"{where}: invalid reference-universe disclosure")
        prediction = packet.get("prediction") or {}
        tolerance = max(1e-6, 2.0 / max(count, 1))
        if float(prediction.get("baseline_percentile_match_error", math.inf)) > tolerance:
            errors.append(f"{where}: recomputed baseline does not match frozen consensus")
        geometry = packet.get("geometry") or {}
        if geometry.get("checkpoint_count") != 15:
            errors.append(f"{where}: geometry must aggregate all 15 checkpoints")
        if geometry.get("coordinate_averaging_disabled") is not True:
            errors.append(f"{where}: coordinate averaging must remain disabled")
        margins.append(float(geometry.get("geodesic_margin_mean", math.nan)))
        counterfactual = packet.get("counterfactual") or {}
        if counterfactual.get("checkpoint_count") != 15:
            errors.append(f"{where}: counterfactual must aggregate all 15 checkpoints")
        if not math.isclose(
            float(counterfactual.get("consensus_percentile_before", math.nan)),
            float(prediction.get("recomputed_baseline_percentile", math.nan)),
            abs_tol=tolerance,
        ):
            errors.append(f"{where}: counterfactual baseline is inconsistent")
        changes.append(float(counterfactual.get("consensus_percentile_change", math.nan)))
        temporal = packet.get("temporal_summary") or {}
        if temporal.get("decision_time_scope") != "full_release_snapshot":
            errors.append(f"{where}: external evidence scope is not disclosed")
        event_ids.update(
            str(row.get("evidence_id", "")) for row in packet.get("evidence", [])
            if row.get("evidence_id")
        )
        edge_ids.update(
            str(row.get("evidence_id", "")) for row in packet.get("critical_edges", [])
            if row.get("evidence_id")
        )
        provenance = packet.get("provenance") or {}
        for key in ("bundle_manifest", "freeze_manifest", "consensus_scores"):
            source = Path(str(provenance.get(key, "")))
            expected = provenance.get(f"{key}_sha256")
            if not source.is_file() or expected != _sha256(source):
                errors.append(f"{where}: provenance mismatch for {key}")
            if key == "bundle_manifest" and source.is_file():
                bundle = source.parent

    if bundle is None:
        errors.append("bundle path could not be established from provenance")
    else:
        missing_events = event_ids - _source_ids(bundle / "events.csv", event_ids)
        missing_edges = edge_ids - _source_ids(bundle / "edges.csv", edge_ids)
        if missing_events:
            errors.append(f"{len(missing_events)} event evidence IDs are unresolved")
        if missing_edges:
            errors.append(f"{len(missing_edges)} edge evidence IDs are unresolved")
    return {
        "operation": operation,
        "status": "passed" if not errors else "failed",
        "packet_count": len(packets),
        "event_evidence_ids": len(event_ids),
        "edge_evidence_ids": len(edge_ids),
        "counterfactual_percentile_change": changes,
        "geodesic_margin_mean": margins,
        "errors": errors,
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "packets": str(packets_path.resolve()),
            "packets_sha256": _sha256(packets_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    operations = [_audit_operation(args.root, name) for name in OPERATIONS]
    result = {
        "schema_version": "hypertrace.consensus-evidence-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(row["status"] == "passed" for row in operations) else "failed",
        "labels_csv_opened": False,
        "excluded_modules": ["psychological_features", "role_classification"],
        "operations": operations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
