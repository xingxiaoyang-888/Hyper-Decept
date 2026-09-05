"""Audit external coordination bundles without running model inference.

The report classifies a bundle as supervised_reportable, traceability_only or
not_ready. No labels or features are inferred.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

REQUIRED = ("nodes.csv", "edges.csv", "events.csv", "labels.csv", "manifest.json")

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def audit_bundle(bundle: Path) -> dict:
    errors, warnings = [], []
    missing = [name for name in REQUIRED if not (bundle / name).is_file()]
    if missing:
        errors.append(f"missing artifacts: {missing}")
    manifest = {}
    if (bundle / "manifest.json").is_file():
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name in ("nodes", "edges", "events", "labels"):
        path = bundle / f"{name}.csv"
        if path.is_file():
            frames[name] = pd.read_csv(path, nrows=0, low_memory=False)
    node_cols = set(frames.get("nodes", pd.DataFrame()).columns)
    edge_cols = set(frames.get("edges", pd.DataFrame()).columns)
    event_cols = set(frames.get("events", pd.DataFrame()).columns)
    label_cols = set(frames.get("labels", pd.DataFrame()).columns)
    has_ids = "node_id" in node_cols and (("source_id" in edge_cols and "target_id" in edge_cols) or not edge_cols)
    has_events = {"actor_id", "timestamp"}.issubset(event_cols)
    has_evidence = "evidence_id" in event_cols or "evidence_id" in edge_cols
    # A bot/human label is not evidence of coordinated inauthentic behavior.
    # Keep the two semantics separate so a transfer dataset cannot silently
    # become a reportable CIB evaluation.
    has_cib_binary = "is_bad" in label_cols or "is_cib" in label_cols
    has_bot_binary = "is_bot" in label_cols
    has_known_mask = "is_known" in label_cols
    capabilities = dict(manifest.get("capabilities") or {})
    label_semantics = str(manifest.get("label_semantics", ""))
    cib_supervised = bool(
        has_ids
        and has_cib_binary
        and has_known_mask
        and label_semantics == "cib_membership"
        and capabilities.get("source_verified_labels", capabilities.get("explicit_cib_labels", False))
    )
    io_supervised = bool(
        has_ids
        and "is_bad" in label_cols
        and has_known_mask
        and label_semantics == "information_operation_membership"
        and capabilities.get("source_verified_io_labels", False)
    )
    supervised = cib_supervised or io_supervised
    traceability = bool(has_ids and has_evidence and (has_events or bool(edge_cols)))
    if not has_events:
        warnings.append("events lack actor_id/timestamp; temporal audit unavailable")
    if not has_evidence:
        warnings.append("no evidence_id in events/edges; provenance coverage unavailable")
    if not has_cib_binary:
        warnings.append("no explicit binary CIB label column")
    elif io_supervised and not cib_supervised:
        warnings.append(
            "information-operation labels support IO metrics, not generic "
            "bot or campaign-level CIB claims"
        )
    if has_bot_binary and not has_cib_binary:
        warnings.append("bot/human labels are transfer-only and cannot support CIB metrics")
    if has_known_mask:
        known = pd.read_csv(bundle / "labels.csv", usecols=["is_known"], low_memory=False)["is_known"].fillna(False)
        known_count, unknown_count = int(known.sum()), int((~known).sum())
    else:
        known_count = unknown_count = None
    status = "supervised_reportable" if supervised and traceability else ("traceability_only" if traceability else "not_ready")
    return {"schema_version": "hypertrace.external-bundle-audit.v1", "bundle": str(bundle.resolve()), "status": status, "evaluation_scope": manifest.get("evaluation_scope"), "label_semantics": label_semantics or None, "errors": errors, "warnings": warnings, "columns": {"nodes": sorted(node_cols), "edges": sorted(edge_cols), "events": sorted(event_cols), "labels": sorted(label_cols)}, "capabilities": capabilities, "label_counts": {"known": known_count, "unknown": unknown_count}, "checks": {"node_ids": has_ids, "events_temporal": has_events, "evidence_ids": has_evidence, "cib_binary_labels": has_cib_binary, "bot_binary_labels": has_bot_binary, "cib_supervised_metrics_allowed": cib_supervised, "io_supervised_metrics_allowed": io_supervised, "supervised_metrics_allowed": supervised, "traceability_allowed": traceability}, "artifacts": {name: {"bytes": (bundle / name).stat().st_size, "sha256": _sha256(bundle / name)} for name in REQUIRED if (bundle / name).is_file()}}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_bundle(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"] or report["status"] == "not_ready":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
