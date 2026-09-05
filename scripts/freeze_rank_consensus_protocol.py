"""Freeze the prospective label-blind rank-consensus protocol and queues."""
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
    p.add_argument("--audits-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--implementation", type=Path, required=True)
    args = p.parse_args()
    gates = {"min_mean_pairwise_spearman": 0.80, "min_top1pct_pairwise_jaccard": 0.30, "max_p95_account_rank_std": 0.25}
    operations = {}
    for operation in ("honduras", "uae"):
        root = args.audits_root / operation
        audit_path = root / "audit.json"
        scores_path = root / "rank_consensus_scores.csv.gz"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        top1 = next(x for x in audit["review_budget_metrics"] if x["budget_fraction"] == 0.01)
        stability = audit["stability"]
        gate_passed = (
            stability["mean_pairwise_spearman"] >= gates["min_mean_pairwise_spearman"]
            and top1["mean_pairwise_checkpoint_jaccard"] >= gates["min_top1pct_pairwise_jaccard"]
            and stability["p95_account_rank_std"] <= gates["max_p95_account_rank_std"]
        )
        scores = pd.read_csv(scores_path)
        limit = math.ceil(len(scores) * 0.05)
        queue = scores.loc[scores["consensus_rank"] <= limit].copy()
        queue.insert(0, "operation", operation)
        queue_path = args.output / operation / "review_queue_top5pct.csv.gz"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue.to_csv(queue_path, index=False, compression="gzip")
        operations[operation] = {
            "gate_passed": gate_passed, "source_operation": audit["source_operation"],
            "target_operation": audit["target_operation"], "checkpoint_count": audit["checkpoint_count"],
            "stability": stability, "top1pct_pairwise_jaccard": top1["mean_pairwise_checkpoint_jaccard"],
            "audit": str(audit_path.resolve()), "audit_sha256": sha256(audit_path),
            "label_blind_queue": str(queue_path.resolve()), "queue_sha256": sha256(queue_path),
            "queue_rows": len(queue), "queue_contains_ground_truth_labels": False,
        }
    manifest = {
        "schema_version": "hypertrace.v5-rank-consensus-protocol.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_for_prospective_external_datasets",
        "current_honduras_uae_analysis_status": "retrospective_stability_analysis_not_primary_confirmatory_evidence",
        "algorithm": "run all 15 pre-frozen checkpoints; convert each score vector to within-dataset percentile ranks; arithmetic-mean ranks; deterministic user-id tie-break",
        "checkpoint_selection": "complete 15-checkpoint suite; no target-specific checkpoint selection",
        "review_budgets": [0.005, 0.01, 0.02, 0.05],
        "label_blind_stability_gates": gates,
        "future_dataset_order": [
            "freeze source checkpoint manifest and dataset audit",
            "run consensus and record stability before reading labels",
            "apply stability gate and freeze ranking",
            "only then unblind labels for AUROC/AUPRC and review-budget evaluation",
        ],
        "target_labels_used_for_model_selection_aggregation_or_ordering": False,
        "implementation": str(args.implementation.resolve()), "implementation_sha256": sha256(args.implementation),
        "operations": operations,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
