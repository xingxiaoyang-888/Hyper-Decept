"""Produce label-blind risk rankings and pre-registered review-budget metrics."""
from __future__ import annotations
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.external_bundle_episode import load_external_bundle_episode
from scripts.run_frozen_external_io import _build_model, _checkpoint_edge_types, _sha256

BUDGETS = (0.005, 0.01, 0.02, 0.05)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    batch = load_external_bundle_episode(args.bundle)
    metadata = (tuple(sorted(batch.graph.node_types)), _checkpoint_edge_types(checkpoint["model"]))
    model = _build_model(checkpoint, metadata).to(torch.device(args.device)).eval()
    batch.to(torch.device(args.device))
    with torch.no_grad():
        output = model(batch.graph, domain=batch.domain, dataset_name=batch.dataset_name)
        scores = torch.sigmoid(output["coordination_logits"]).cpu().numpy()
    targets = batch.coordination_targets.cpu().numpy().astype(np.int64)
    known = batch.coordination_mask.cpu().numpy().astype(bool)
    user_ids = np.asarray(batch.graph["user"].node_ids, dtype=str)
    known_indices = np.flatnonzero(known)
    # Risk score is the sole ordering signal; user ID is a deterministic tie-break.
    order = known_indices[np.lexsort((user_ids[known_indices], -scores[known_indices]))]
    positives = int(targets[known_indices].sum())
    prevalence = positives / max(len(known_indices), 1)
    metrics = []
    for fraction in BUDGETS:
        k = max(1, math.ceil(len(order) * fraction))
        selected = order[:k]
        hits = int(targets[selected].sum())
        precision = hits / k
        metrics.append({
            "budget_fraction": fraction, "reviewed_accounts": k, "true_io_hits": hits,
            "precision_at_k": precision, "recall_at_k": hits / max(positives, 1),
            "lift_at_k": precision / max(prevalence, np.finfo(float).eps),
        })
    rank_by_index = np.full(len(user_ids), -1, dtype=np.int64)
    rank_by_index[order] = np.arange(1, len(order) + 1)
    frame = pd.DataFrame({
        "user_id": user_ids, "risk_score": scores, "risk_rank": rank_by_index,
        "label_known": known, "information_operation_membership": np.where(known, targets, -1),
    }).sort_values(["risk_rank", "user_id"], kind="stable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "ranked_scores.csv.gz"
    frame.to_csv(ranking_path, index=False, compression="gzip")
    result = {
        "schema_version": "hypertrace.v5-review-budget-evaluation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256(args.checkpoint),
        "bundle": str(args.bundle.resolve()), "bundle_manifest_sha256": _sha256(args.bundle / "manifest.json"),
        "selection_and_ordering": "checkpoint selected on synthetic validation AUPRC; accounts ordered only by frozen risk score",
        "target_labels_used_for_selection_threshold_or_ordering": False,
        "threshold": None, "review_budgets": list(BUDGETS), "known_accounts": len(known_indices),
        "positive_accounts": positives, "prevalence": prevalence, "metrics": metrics,
        "ranking_artifact": str(ranking_path.resolve()), "ranking_sha256": _sha256(ranking_path),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
