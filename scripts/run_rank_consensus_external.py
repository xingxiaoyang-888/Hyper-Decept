"""Label-blind rank-consensus inference across every frozen v5 checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.external_bundle_episode import load_external_bundle_episode
from scripts.frozen_manifest import resolve_frozen_checkpoint
from scripts.run_frozen_external_io import _build_model, _checkpoint_edge_types, _sha256

BUDGETS = (0.005, 0.01, 0.02, 0.05)

def _percentile_ranks(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((ids, scores))
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = (np.arange(len(scores), dtype=np.float32) + 1.0) / max(len(scores), 1)
    return ranks

def _mean_pairwise_jaccard(rank_matrix: np.ndarray, fraction: float) -> float:
    k = max(1, math.ceil(rank_matrix.shape[1] * fraction))
    tops = [set(np.argpartition(row, -k)[-k:].tolist()) for row in rank_matrix]
    values = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            values.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    return float(np.mean(values))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    entries = list(freeze["checkpoints"])
    if len(entries) != 15:
        raise ValueError("rank consensus requires the complete 15-checkpoint suite")
    source_operations = {entry["warm_start"]["source_operation"] for entry in entries}
    if len(source_operations) != 1:
        raise ValueError("all consensus checkpoints must share one source operation")
    source_operation = next(iter(source_operations))
    batch = load_external_bundle_episode(args.bundle)
    target_operation = str(batch.episode_id)
    if source_operation == target_operation:
        raise ValueError("rank consensus external evaluation must be operation-disjoint")
    device = torch.device(args.device)
    batch.to(device)
    user_ids = np.asarray(batch.graph["user"].node_ids, dtype=str)
    known = batch.bot_mask.cpu().numpy().astype(bool)
    known_indices = np.flatnonzero(known)
    known_ids = user_ids[known_indices]
    rank_rows = []
    checkpoint_records = []
    metadata = None
    for entry in entries:
        checkpoint_path = resolve_frozen_checkpoint(args.freeze_manifest, entry)
        if _sha256(checkpoint_path) != entry["sha256"]:
            raise ValueError(f"frozen checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if metadata is None:
            metadata = (tuple(sorted(batch.graph.node_types)), _checkpoint_edge_types(checkpoint["model"]))
        model = _build_model(checkpoint, metadata).to(device).eval()
        with torch.no_grad():
            output = model(batch.graph, domain=batch.domain, dataset_name=batch.dataset_name)
            scores = torch.sigmoid(output["bot_logits"]).cpu().numpy()[known_indices]
        rank_rows.append(_percentile_ranks(scores, known_ids))
        checkpoint_records.append({
            "checkpoint": str(checkpoint_path.resolve()), "sha256": entry["sha256"],
            "held_out_scenario": entry["held_out_scenario"], "seed": entry["seed"],
            "best_validation_auprc": entry["best_validation_auprc"],
        })
        del model, output, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rank_matrix = np.stack(rank_rows).astype(np.float32, copy=False)
    consensus = rank_matrix.mean(axis=0)
    disagreement = rank_matrix.std(axis=0)
    order = np.lexsort((known_ids, -consensus))
    targets = batch.bot_targets.cpu().numpy().astype(np.int64)[known_indices]
    positives = int(targets.sum())
    prevalence = positives / len(targets)
    budget_metrics = []
    for fraction in BUDGETS:
        k = max(1, math.ceil(len(order) * fraction))
        selected = order[:k]
        hits = int(targets[selected].sum())
        precision = hits / k
        budget_metrics.append({
            "budget_fraction": fraction, "reviewed_accounts": k, "true_io_hits": hits,
            "precision_at_k": precision, "recall_at_k": hits / max(positives, 1),
            "lift_at_k": precision / max(prevalence, np.finfo(float).eps),
            "mean_pairwise_checkpoint_jaccard": _mean_pairwise_jaccard(rank_matrix, fraction),
        })
    pairwise = np.corrcoef(rank_matrix)
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    consensus_rank = np.empty(len(order), dtype=np.int64)
    consensus_rank[order] = np.arange(1, len(order) + 1)
    ranking = pd.DataFrame({
        "user_id": known_ids, "consensus_percentile": consensus,
        "consensus_rank": consensus_rank, "checkpoint_rank_std": disagreement,
    }).sort_values(["consensus_rank", "user_id"], kind="stable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "rank_consensus_scores.csv.gz"
    ranking.to_csv(ranking_path, index=False, compression="gzip")
    result = {
        "schema_version": "hypertrace.v5-rank-consensus-external.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "protocol_status": "retrospective_stability_analysis_for_current_operations; preregistered_candidate_for_future_operations",
        "source_operation": source_operation, "target_operation": target_operation, "operation_disjoint": True,
        "checkpoint_count": len(entries), "aggregation": "mean within-dataset percentile rank",
        "target_labels_used_for_checkpoint_selection_scoring_or_ordering": False,
        "known_accounts": len(targets), "positive_accounts": positives, "prevalence": prevalence,
        "ranking_metrics": {"auroc": float(roc_auc_score(targets, consensus)), "auprc": float(average_precision_score(targets, consensus))},
        "review_budget_metrics": budget_metrics,
        "stability": {
            "mean_pairwise_spearman": float(upper.mean()), "std_pairwise_spearman": float(upper.std()),
            "mean_account_rank_std": float(disagreement.mean()), "p95_account_rank_std": float(np.quantile(disagreement, 0.95)),
        },
        "checkpoints": checkpoint_records,
        "ranking_artifact": str(ranking_path.resolve()), "ranking_sha256": _sha256(ranking_path),
        "freeze_manifest": str(args.freeze_manifest.resolve()), "freeze_manifest_sha256": _sha256(args.freeze_manifest),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
