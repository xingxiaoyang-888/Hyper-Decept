"""Combine the two directions of the formal v5 unseen-operation protocol."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--honduras", type=Path, required=True)
    p.add_argument("--uae", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    h = json.loads(args.honduras.read_text(encoding="utf-8"))
    u = json.loads(args.uae.read_text(encoding="utf-8"))
    errors = []
    for name, report, target in (("honduras", h, "honduras"), ("uae", u, "uae")):
        aggregate = report.get("aggregates", {}).get(target, {})
        if report.get("status") != "passed" or not aggregate.get("operation_disjoint"):
            errors.append(f"{name} is not a passed strict operation-disjoint audit")
        if aggregate.get("count") != 15:
            errors.append(f"{name} expected 15 frozen runs")
    result = {
        "schema_version": "hypertrace.v5-bidirectional-operation-disjoint.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed", "errors": errors,
        "protocol": {
            "honduras": "UAE unlabeled cutoff-safe warm-start -> synthetic coordination-only LOSO -> frozen checkpoint -> Honduras external labels",
            "uae": "Honduras unlabeled cutoff-safe warm-start -> synthetic coordination-only LOSO -> frozen checkpoint -> UAE external labels",
            "test_labels_used_for_training_or_thresholding": False,
            "threshold": 0.5,
        },
        "honduras": h["aggregates"]["honduras"],
        "uae": u["aggregates"]["uae"],
        "source_audits": {
            "honduras": {"path": str(args.honduras.resolve()), "sha256": digest(args.honduras)},
            "uae": {"path": str(args.uae.resolve()), "sha256": digest(args.uae)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
