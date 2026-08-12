"""Verify and aggregate the frozen v5 Honduras/UAE evaluation matrix."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from datetime import datetime, timezone
from pathlib import Path

METRICS = ("auroc", "auprc", "f1", "balanced_accuracy", "brier", "ece")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-reports", type=int, default=30)
    args = p.parse_args()
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    errors = []
    frozen = {}
    for entry in freeze["checkpoints"]:
        path = Path(entry["frozen_checkpoint"])
        actual = sha256(path)
        if actual != entry["sha256"]:
            errors.append(f"checkpoint hash mismatch: {path}")
        frozen[str(path.resolve())] = entry
    rows = []
    for path in sorted(args.reports.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = str(Path(report["checkpoint"]).resolve())
        entry = frozen.get(checkpoint)
        if entry is None:
            errors.append(f"unfrozen checkpoint in report: {path}")
            continue
        target = Path(report["bundle"]).name.lower()
        # base-best.v2 predates embedded warm-start metadata; the immutable
        # freeze manifest records the source metrics contract and is therefore
        # authoritative for this already-frozen matrix.
        source = (entry.get("warm_start") or {}).get("source_operation")
        disjoint = bool(source and source != target)
        if report.get("status") != "passed" or report.get("checkpoint_updated") is not False:
            errors.append(f"invalid frozen evaluation contract: {path}")
        if report.get("training_labels_consumed") is not False:
            errors.append(f"real labels consumed: {path}")
        rows.append({
            "report": str(path.resolve()), "checkpoint": checkpoint,
            "held_out_scenario": entry["held_out_scenario"], "seed": entry["seed"],
            "source_operation": source, "target_operation": target,
            "operation_disjoint": disjoint,
            "evaluation_protocol": "strict_unseen_operation" if disjoint else "same_operation_unlabeled_warm_start",
            "metrics": report["metrics"],
        })
    if len(rows) != args.expected_reports:
        errors.append(f"expected {args.expected_reports} reports, found {len(rows)}")
    aggregates = {}
    for target in ("honduras", "uae"):
        subset = [r for r in rows if r["target_operation"] == target]
        if not subset:
            continue
        aggregates[target] = {
            "count": len(subset),
            "evaluation_protocol": subset[0]["evaluation_protocol"] if subset else None,
            "operation_disjoint": subset[0]["operation_disjoint"] if subset else None,
            "metrics": {
                name: {
                    "mean": statistics.mean(r["metrics"][name] for r in subset),
                    "std": statistics.stdev(r["metrics"][name] for r in subset),
                    "min": min(r["metrics"][name] for r in subset),
                    "max": max(r["metrics"][name] for r in subset),
                }
                for name in METRICS
            },
        }
    result = {
        "schema_version": "hypertrace.v5-external-matrix-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed", "errors": errors,
        "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "checkpoint_count": len(frozen), "report_count": len(rows),
        "real_operation_labels_consumed": False,
        "aggregates": aggregates, "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
