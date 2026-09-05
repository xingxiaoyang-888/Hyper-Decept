"""Evaluate budget-matched explanation controls on frozen HyperTrace cases.

The runner never reads target labels.  It reuses the exact case IDs, candidate
neighborhoods, evidence-unit budgets, reference universe, and 15 frozen
checkpoints used by the constrained HyperTrace evaluation.  Four conditions
are compared:

* ``hypertrace``: relation-stratified consensus-attention selection;
* ``static_topk``: a global consensus-attention top-k without constraints;
* ``random_edge``: a uniform sample without replacement;
* ``degree_matched``: relation-aware nearest-degree matching to HyperTrace.

Random and degree-matched controls are repeated with deterministic seeds.
Results are written incrementally at the case level and summarized with a
case-clustered bootstrap, so repeated draws are not treated as independent
cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (ROOT, CHARACTER_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.evaluate_constrained_evidence import (  # noqa: E402
    _attention_scores,
    _candidate_edges,
    _candidate_lookup,
    _edge_masks,
    _evidence_units,
    _fidelity,
    _load_entries,
    _mean_record,
    _prediction_record,
    _relation_stratified_prefix_for_fraction,
    _unit_attention_scores,
)
from scripts.external_bundle_episode import load_external_bundle_episode  # noqa: E402
from scripts.frozen_manifest import resolve_frozen_checkpoint  # noqa: E402
from scripts.run_frozen_external_io import (  # noqa: E402
    _build_model,
    _checkpoint_edge_types,
    _sha256,
)


METHODS = ("hypertrace", "static_topk", "random_edge", "degree_matched")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_seed(*parts: object) -> int:
    value = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _topk(scores: np.ndarray, units, count: int) -> np.ndarray:
    stable = np.asarray([unit.stable_id for unit in units], dtype=str)
    order = np.lexsort((stable, -np.asarray(scores, dtype=np.float64)))
    return order[:count].astype(np.int64, copy=False)


def _node_degrees(graph) -> dict[str, np.ndarray]:
    degrees = {
        node_type: np.zeros(len(graph[node_type].node_ids), dtype=np.int64)
        for node_type in graph.node_types
    }
    for (source_type, _, target_type), edge_index in graph.edge_index_dict.items():
        source = edge_index[0].detach().cpu().numpy()
        target = edge_index[1].detach().cpu().numpy()
        degrees[source_type] += np.bincount(
            source, minlength=len(degrees[source_type])
        )
        degrees[target_type] += np.bincount(
            target, minlength=len(degrees[target_type])
        )
    return degrees


def _unit_degree_scores(graph, candidates, units, degrees) -> np.ndarray:
    node_ids = {
        node_type: list(graph[node_type].node_ids) for node_type in graph.node_types
    }
    index = {
        node_type: {str(node_id): position for position, node_id in enumerate(ids)}
        for node_type, ids in node_ids.items()
    }
    candidate_degree = {}
    for candidate in candidates:
        source_degree = degrees[candidate.source_type][
            index[candidate.source_type][candidate.source_id]
        ]
        target_degree = degrees[candidate.target_type][
            index[candidate.target_type][candidate.target_id]
        ]
        candidate_degree[candidate.stable_id] = (
            math.log1p(float(source_degree)) + math.log1p(float(target_degree))
        ) / 2.0
    return np.asarray([
        float(np.mean([
            candidate_degree[member.stable_id] for member in unit.members
        ]))
        for unit in units
    ], dtype=np.float64)


def _degree_matched_sample(
    units,
    degree_scores: np.ndarray,
    target_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float]:
    """Match relation and degree against evidence outside the target selection.

    A degree-matched control must not trivially reproduce HyperTrace by matching
    each selected unit to itself.  We therefore exclude the complete target set
    whenever the remaining candidate pool is large enough.  For unusually
    large evidence budgets where that is impossible, all units are admitted
    but the resulting overlap is recorded for audit.
    """
    target_set = set(map(int, target_indices))
    non_target = set(range(len(units))) - target_set
    available = (
        non_target if len(non_target) >= len(target_indices)
        else set(range(len(units)))
    )
    selected: list[int] = []
    distances: list[float] = []
    exact = 0
    target_order = rng.permutation(np.asarray(target_indices, dtype=np.int64))
    for target_index in target_order:
        target = units[int(target_index)]
        same_relation = [
            index for index in available
            if units[index].relation == target.relation
        ]
        pool = same_relation if same_relation else sorted(available)
        differences = np.asarray([
            abs(degree_scores[index] - degree_scores[int(target_index)])
            for index in pool
        ])
        minimum = float(differences.min())
        tied = [
            pool[position] for position in np.flatnonzero(
                np.isclose(differences, minimum, rtol=0.0, atol=1e-12)
            )
        ]
        choice = int(rng.choice(np.asarray(tied, dtype=np.int64)))
        selected.append(choice)
        available.remove(choice)
        distances.append(minimum)
        exact += int(minimum <= 1e-12)
    selected_array = np.asarray(selected, dtype=np.int64)
    return (
        selected_array,
        float(np.mean(distances)),
        exact / max(len(target_indices), 1),
        len(set(map(int, selected_array)) & target_set)
        / max(len(target_indices), 1),
    )


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    return len(a & b) / max(len(a | b), 1)


def _selection_plan(
    operation: str,
    case: dict,
    units,
    aggregate_attention: np.ndarray,
    degree_scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[dict]:
    fraction = float(case["selection"]["edge_budget_fraction"])
    count = int(case["selection"]["selected_evidence_units"])
    hypertrace = _relation_stratified_prefix_for_fraction(
        aggregate_attention, units, fraction
    )
    if len(hypertrace) != count:
        raise ValueError(
            f"{case['case_id']}: reconstructed HyperTrace budget {len(hypertrace)} "
            f"does not match frozen count {count}"
        )
    result = [{
        "method": "hypertrace",
        "replicate": 0,
        "indices": hypertrace,
        "degree_match_mean_absolute_log_degree_error": 0.0,
        "degree_match_exact_fraction": 1.0,
        "selection_jaccard_with_hypertrace": 1.0,
    }]
    static = _topk(aggregate_attention, units, count)
    result.append({
        "method": "static_topk",
        "replicate": 0,
        "indices": static,
        "degree_match_mean_absolute_log_degree_error": None,
        "degree_match_exact_fraction": None,
        "selection_jaccard_with_hypertrace": _jaccard(static, hypertrace),
    })
    case_id = case["case_id"]
    for replicate in range(repetitions):
        random_rng = np.random.default_rng(
            _stable_seed(seed, operation, case_id, "random_edge", replicate)
        )
        random_indices = np.sort(random_rng.choice(
            len(units), size=count, replace=False
        )).astype(np.int64, copy=False)
        result.append({
            "method": "random_edge",
            "replicate": replicate,
            "indices": random_indices,
            "degree_match_mean_absolute_log_degree_error": None,
            "degree_match_exact_fraction": None,
            "selection_jaccard_with_hypertrace": _jaccard(
                random_indices, hypertrace
            ),
        })
        degree_rng = np.random.default_rng(
            _stable_seed(seed, operation, case_id, "degree_matched", replicate)
        )
        (
            degree_indices,
            mean_error,
            exact_fraction,
            target_overlap_fraction,
        ) = _degree_matched_sample(
            units, degree_scores, hypertrace, degree_rng
        )
        result.append({
            "method": "degree_matched",
            "replicate": replicate,
            "indices": degree_indices,
            "degree_match_mean_absolute_log_degree_error": mean_error,
            "degree_match_exact_fraction": exact_fraction,
            "degree_match_target_overlap_fraction": target_overlap_fraction,
            "selection_jaccard_with_hypertrace": _jaccard(
                degree_indices, hypertrace
            ),
        })
    return result


def _case_bootstrap(
    per_case: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    pivot = per_case.pivot(index="case_id", columns="method", values=metric)
    cases = pivot.index.to_numpy()
    rng = np.random.default_rng(_stable_seed(seed, metric, "bootstrap"))
    summaries: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        values = pivot[method].to_numpy(dtype=float)
        boot = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            indices = rng.integers(0, len(cases), size=len(cases))
            boot[draw] = float(np.mean(values[indices]))
        summaries[method] = {
            "n_cases": int(len(cases)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "case_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
            "case_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        }
    return summaries


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _load_entries(args.freeze_manifest.resolve())
    frozen_cases = _read_jsonl(args.hypertrace_cases.resolve())
    if args.scope == "nontrivial_high":
        frozen_cases = [
            row for row in frozen_cases
            if row["risk_stratum"] == "high"
            and row["selection"]["candidate_evidence_units"]
            > row["selection"]["selected_evidence_units"]
        ]
    elif args.scope != "all":
        raise ValueError(f"unsupported scope: {args.scope}")
    if not frozen_cases:
        raise ValueError("no frozen cases remain after scope filtering")

    consensus = pd.read_csv(args.consensus_scores)
    consensus["user_id"] = consensus["user_id"].astype(str)
    batch = load_external_bundle_episode(args.bundle.resolve(), load_labels=False)
    if torch.any(batch.coordination_mask):
        raise ValueError("control evaluator unexpectedly exposed target labels")
    operation = str(batch.episode_id)
    case_ids = [row["case_id"].split(":", 1)[-1] for row in frozen_cases]
    all_ids = np.asarray(batch.graph["user"].node_ids, dtype=str)
    index_lookup = {user_id: index for index, user_id in enumerate(all_ids)}
    case_indices = {user_id: index_lookup[user_id] for user_id in case_ids}
    reference_ids = consensus["user_id"].to_numpy(dtype=str)
    reference_indices = np.asarray(
        [index_lookup[user_id] for user_id in reference_ids], dtype=np.int64
    )
    reference_lookup = {
        user_id: position for position, user_id in enumerate(reference_ids)
    }
    candidates = {
        user_id: _candidate_edges(batch.graph, case_indices[user_id])
        for user_id in case_ids
    }
    units = {
        user_id: _evidence_units(candidates[user_id]) for user_id in case_ids
    }
    lookups = {
        user_id: _candidate_lookup(candidates[user_id]) for user_id in case_ids
    }
    degrees = _node_degrees(batch.graph)
    degree_scores = {
        user_id: _unit_degree_scores(
            batch.graph, candidates[user_id], units[user_id], degrees
        )
        for user_id in case_ids
    }

    device = torch.device(args.device)
    batch.to(device)
    baseline: dict[str, list[dict[str, float]]] = defaultdict(list)
    attention: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata = None
    for entry in entries:
        checkpoint_path = resolve_frozen_checkpoint(args.freeze_manifest, entry)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if metadata is None:
            metadata = (
                tuple(sorted(batch.graph.node_types)),
                _checkpoint_edge_types(checkpoint["model"]),
            )
        model = _build_model(checkpoint, metadata).to(device).eval()
        with torch.no_grad():
            output = model(
                batch.graph, domain=batch.domain, dataset_name=batch.dataset_name
            )
        for user_id in case_ids:
            baseline[user_id].append(_prediction_record(
                output,
                model,
                case_indices[user_id],
                reference_indices,
                reference_ids,
                reference_lookup[user_id],
            ))
            edge_scores = _attention_scores(
                model, lookups[user_id], len(candidates[user_id])
            )
            attention[user_id].append(_unit_attention_scores(
                edge_scores, candidates[user_id], units[user_id]
            ))
        # Relation audits contain full-edge attention tensors.  Releasing them
        # after extraction avoids retaining one graph-sized audit across the
        # next forward pass, which otherwise doubles peak memory on UAE.
        model.encoder._last_audits.clear()
        del model, output, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    case_by_id = {
        row["case_id"].split(":", 1)[-1]: row for row in frozen_cases
    }
    plans = {
        user_id: _selection_plan(
            operation,
            case_by_id[user_id],
            units[user_id],
            np.mean(np.stack(attention[user_id]), axis=0),
            degree_scores[user_id],
            args.repetitions,
            args.seed,
        )
        for user_id in case_ids
    }
    keep: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
    deleted: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
    for checkpoint_number, entry in enumerate(entries, start=1):
        checkpoint_path = resolve_frozen_checkpoint(args.freeze_manifest, entry)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model = _build_model(checkpoint, metadata).to(device).eval()
        for user_id in case_ids:
            for selection in plans[user_id]:
                key = (
                    user_id,
                    selection["method"],
                    int(selection["replicate"]),
                )
                for mode, destination in (("keep", keep), ("delete", deleted)):
                    masks = _edge_masks(
                        batch.graph,
                        units[user_id],
                        selection["indices"],
                        mode,
                    )
                    with torch.no_grad():
                        output = model(
                            batch.graph,
                            domain=batch.domain,
                            dataset_name=batch.dataset_name,
                            edge_mask_dict=masks,
                        )
                    destination[key].append(_prediction_record(
                        output,
                        model,
                        case_indices[user_id],
                        reference_indices,
                        reference_ids,
                        reference_lookup[user_id],
                    ))
                    model.encoder._last_audits.clear()
                    del output, masks
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "operation": operation,
            "checkpoint_completed": checkpoint_number,
            "checkpoint_total": len(entries),
        }), flush=True)

    rows = []
    for user_id in case_ids:
        base_mean = _mean_record(baseline[user_id])
        predicted_coordination = base_mean["geodesic_margin"] > 0
        for selection in plans[user_id]:
            key = (
                user_id,
                selection["method"],
                int(selection["replicate"]),
            )
            keep_mean = _mean_record(keep[key])
            delete_mean = _mean_record(deleted[key])
            fidelity = _fidelity(baseline[user_id], keep[key])
            raw_change = delete_mean["percentile"] - base_mean["percentile"]
            comprehensiveness = (
                -raw_change if predicted_coordination else raw_change
            )
            rows.append({
                "operation": operation,
                "case_id": f"user:{user_id}",
                "risk_stratum": case_by_id[user_id]["risk_stratum"],
                "method": selection["method"],
                "replicate": int(selection["replicate"]),
                "candidate_evidence_units": len(units[user_id]),
                "selected_evidence_units": len(selection["indices"]),
                "evidence_retained": (
                    len(selection["indices"]) / len(units[user_id])
                ),
                "sufficiency_percentile_error": fidelity[
                    "sufficiency_percentile_error"
                ],
                "geometry_fidelity": fidelity["geometry_fidelity"],
                "prototype_vote_agreement": fidelity[
                    "prototype_vote_agreement"
                ],
                "geodesic_margin_absolute_error": fidelity[
                    "geodesic_margin_absolute_error"
                ],
                "comprehensiveness": comprehensiveness,
                "selection_jaccard_with_hypertrace": selection[
                    "selection_jaccard_with_hypertrace"
                ],
                "degree_match_mean_absolute_log_degree_error": selection[
                    "degree_match_mean_absolute_log_degree_error"
                ],
                "degree_match_exact_fraction": selection[
                    "degree_match_exact_fraction"
                ],
                "degree_match_target_overlap_fraction": selection.get(
                    "degree_match_target_overlap_fraction"
                ),
                "checkpoint_count": len(entries),
                "labels_consumed": False,
            })
    row_path = output_dir / "case_replicates.jsonl"
    with row_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    frame = pd.DataFrame(rows)
    metric_columns = (
        "sufficiency_percentile_error",
        "geometry_fidelity",
        "prototype_vote_agreement",
        "comprehensiveness",
        "selection_jaccard_with_hypertrace",
    )
    per_case = frame.groupby(
        ["case_id", "method"], as_index=False
    )[list(metric_columns)].mean()
    per_case_path = output_dir / "per_case_method_means.csv"
    per_case.to_csv(per_case_path, index=False)
    summaries = {
        metric: _case_bootstrap(
            per_case,
            metric,
            seed=args.seed,
            draws=args.bootstrap_draws,
        )
        for metric in metric_columns
    }
    result = {
        "schema_version": "hypertrace.explanation-controls.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "operation": operation,
        "scope": args.scope,
        "case_count": len(case_ids),
        "checkpoint_count": len(entries),
        "random_and_degree_repetitions_per_case": args.repetitions,
        "selection_seed": args.seed,
        "selection_budget": "exact frozen HyperTrace evidence-unit count per case",
        "labels_consumed": False,
        "controls": {
            "static_topk": "global 15-checkpoint mean intrinsic-attention top-k",
            "random_edge": "uniform evidence units without replacement",
            "degree_matched": (
                "relation-aware nearest log-degree match without replacement, "
                "excluding HyperTrace-selected units whenever the remaining "
                "candidate pool can satisfy the budget; ties resolved by "
                "deterministic seeded random choice"
            ),
        },
        "metrics": summaries,
        "latency_seconds": time.perf_counter() - started,
        "artifacts": {
            "case_replicates": str(row_path),
            "case_replicates_sha256": _sha256(row_path),
            "per_case_method_means": str(per_case_path),
            "per_case_method_means_sha256": _sha256(per_case_path),
            "hypertrace_cases_sha256": _sha256(args.hypertrace_cases),
            "consensus_scores_sha256": _sha256(args.consensus_scores),
            "freeze_manifest_sha256": _sha256(args.freeze_manifest),
        },
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--consensus-scores", required=True, type=Path)
    parser.add_argument("--hypertrace-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scope", choices=("all", "nontrivial_high"), default="nontrivial_high"
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    run(args)


if __name__ == "__main__":
    main()
