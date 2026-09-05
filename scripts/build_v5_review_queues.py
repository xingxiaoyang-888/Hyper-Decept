"""Build label-blind review queues from frozen selected-model rankings."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"schema_version": "hypertrace.v5-review-queues.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "queues": {}}
    for operation in ("honduras", "uae"):
        source = args.root / operation / "ranked_scores.csv.gz"
        audit = json.loads((args.root / operation / "audit.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(source)
        count = int(audit["known_accounts"])
        limit = math.ceil(count * 0.05)
        queue = frame.loc[frame["risk_rank"].between(1, limit), ["user_id", "risk_score", "risk_rank"]].copy()
        queue.insert(0, "operation", operation)
        queue["review_budget_tier"] = pd.cut(
            queue["risk_rank"],
            bins=[0, math.ceil(count * .005), math.ceil(count * .01), math.ceil(count * .02), limit],
            labels=["top_0.5pct", "top_1pct", "top_2pct", "top_5pct"], include_lowest=True,
        ).astype(str)
        target = args.output / operation / "review_queue_top5pct.csv.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        queue.to_csv(target, index=False, compression="gzip")
        top_half = next(x for x in audit["metrics"] if x["budget_fraction"] == 0.005)
        result["queues"][operation] = {
            "path": str(target.resolve()), "sha256": sha256(target), "rows": len(queue),
            "contains_ground_truth_labels": False,
            "top_0.5pct_lift": top_half["lift_at_k"],
            "readiness": "explanation_pilot_ready" if top_half["lift_at_k"] > 1 else "hold_for_stability_analysis",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
