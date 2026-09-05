"""Evaluate compact, geometry-faithful evidence on frozen real operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (ROOT, CHARACTER_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from lorentz_hgt import lorentz_distance  # noqa: E402
from scripts.external_bundle_episode import load_external_bundle_episode  # noqa: E402
from scripts.frozen_manifest import resolve_frozen_checkpoint  # noqa: E402
from scripts.generate_consensus_evidence_preflight import _percentile  # noqa: E402
from scripts.run_frozen_external_io import (  # noqa: E402
    _build_model,
    _checkpoint_edge_types,
    _sha256,
)


EDGE_BUDGET_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
PREDICTION_TOLERANCE = 0.02
GEOMETRY_FIDELITY_MINIMUM = 0.95
PROTOTYPE_VOTE_AGREEMENT_MINIMUM = 0.90


@dataclass(frozen=True)
class EdgeCandidate:
    edge_type: tuple[str, str, str]
    position: int
    source_id: str
    target_id: str

    @property
    def source_type(self) -> str:
        return self.edge_type[0]

    @property
    def relation(self) -> str:
        return self.edge_type[1]

    @property
    def target_type(self) -> str:
        return self.edge_type[2]

    @property
    def provenance_key(self) -> str:
        return "\x1f".join((
            self.source_type,
            self.source_id,
            self.target_type,
            self.target_id,
            self.relation,
        ))

    @property
    def stable_id(self) -> str:
        return f"{self.provenance_key}\x1f{self.position:012d}"


RECIPROCAL_RELATIONS = {
    "posts": "posts",
    "authored_by": "posts",
    "retweets": "retweets",
    "retweeted_by": "retweets",
    "comments": "comments",
    "commented_by": "comments",
    "likes": "likes",
    "liked_by": "likes",
}


@dataclass(frozen=True)
class EvidenceUnit:
    stable_id: str
    relation: str
    members: tuple[EdgeCandidate, ...]


def _evidence_units(candidates: list[EdgeCandidate]) -> list[EvidenceUnit]:
    grouped: dict[tuple[str, ...], list[EdgeCandidate]] = defaultdict(list)
    for candidate in candidates:
        relation = RECIPROCAL_RELATIONS.get(candidate.relation)
        typed_ids = {
            candidate.source_type: candidate.source_id,
            candidate.target_type: candidate.target_id,
        }
        if relation and "user" in typed_ids and "tweet" in typed_ids:
            key = (
                "user_tweet_interaction",
                relation,
                typed_ids["user"],
                typed_ids["tweet"],
            )
        else:
            key = (
                "typed_edge",
                candidate.source_type,
                candidate.relation,
                candidate.source_id,
                candidate.target_type,
                candidate.target_id,
            )
        grouped[key].append(candidate)
    return [
        EvidenceUnit(
            stable_id="\x1f".join(key),
            relation=(key[1] if key[0] == "user_tweet_interaction" else key[2]),
            members=tuple(sorted(values, key=lambda value: value.stable_id)),
        )
        for key, values in sorted(grouped.items())
    ]


def _select_cases(frame: pd.DataFrame, case_count: int) -> pd.DataFrame:
    if case_count < 3 or case_count % 3:
        raise ValueError("case_count must be a positive multiple of three")
    required = {
        "user_id",
        "consensus_percentile",
        "consensus_rank",
        "checkpoint_rank_std",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"consensus scores missing columns: {missing}")
    work = frame.copy()
    work["user_id"] = work["user_id"].astype(str)
    if work["user_id"].duplicated().any():
        raise ValueError("consensus scores contain duplicate user IDs")
    per_stratum = case_count // 3
    high = work.sort_values(
        ["consensus_rank", "checkpoint_rank_std", "user_id"], kind="stable"
    ).head(per_stratum).copy()
    high["risk_stratum"] = "high"
    used = set(high["user_id"])

    low = work.loc[~work["user_id"].isin(used)].sort_values(
        ["consensus_rank", "checkpoint_rank_std", "user_id"],
        ascending=[False, True, True],
        kind="stable",
    ).head(per_stratum).copy()
    low["risk_stratum"] = "low"
    used.update(low["user_id"])

    medium = work.loc[~work["user_id"].isin(used)].copy()
    medium["midpoint_distance"] = (
        medium["consensus_percentile"].astype(float) - 0.5
    ).abs()
    medium = medium.sort_values(
        ["midpoint_distance", "checkpoint_rank_std", "user_id"], kind="stable"
    ).head(per_stratum).copy()
    medium["risk_stratum"] = "medium"
    selected = pd.concat([high, medium, low], ignore_index=True)
    if len(selected) != case_count:
        raise ValueError("not enough consensus accounts for requested case pool")
    return selected.drop(columns=["midpoint_distance"], errors="ignore")


def _candidate_edges(graph, user_index: int) -> list[EdgeCandidate]:
    result: list[EdgeCandidate] = []
    node_ids = {
        node_type: list(graph[node_type].node_ids) for node_type in graph.node_types
    }
    for edge_type, edge_index in graph.edge_index_dict.items():
        source_type, _, target_type = edge_type
        incident = torch.zeros(edge_index.shape[1], dtype=torch.bool)
        if source_type == "user":
            incident |= edge_index[0].cpu() == user_index
        if target_type == "user":
            incident |= edge_index[1].cpu() == user_index
        for position in torch.nonzero(incident, as_tuple=False).flatten().tolist():
            source_index = int(edge_index[0, position])
            target_index = int(edge_index[1, position])
            result.append(EdgeCandidate(
                edge_type=edge_type,
                position=int(position),
                source_id=str(node_ids[source_type][source_index]),
                target_id=str(node_ids[target_type][target_index]),
            ))
    return sorted(result, key=lambda value: value.stable_id)


def _prefix_for_fraction(
    scores: np.ndarray,
    candidates: list[EvidenceUnit],
    fraction: float,
) -> np.ndarray:
    if not candidates:
        return np.asarray([], dtype=np.int64)
    stable = np.asarray([candidate.stable_id for candidate in candidates], dtype=str)
    order = np.lexsort((stable, -np.asarray(scores, dtype=np.float64)))
    if fraction >= 1.0:
        return order
    count = max(1, math.ceil(len(order) * fraction))
    return order[:count]


def _relation_stratified_prefix_for_fraction(
    scores: np.ndarray,
    candidates: list[EvidenceUnit],
    fraction: float,
) -> np.ndarray:
    """Select the same evidence fraction within every HGT relation."""
    if not candidates:
        return np.asarray([], dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    selected: list[int] = []
    relations = sorted({candidate.relation for candidate in candidates})
    for relation in relations:
        indices = np.asarray([
            index
            for index, candidate in enumerate(candidates)
            if candidate.relation == relation
        ], dtype=np.int64)
        stable = np.asarray(
            [candidates[int(index)].stable_id for index in indices], dtype=str
        )
        order = indices[np.lexsort((stable, -scores[indices]))]
        count = len(order) if fraction >= 1.0 else max(
            1, math.ceil(len(order) * fraction)
        )
        selected.extend(int(index) for index in order[:count])
    return np.asarray(selected, dtype=np.int64)


def _candidate_lookup(candidates: list[EdgeCandidate]):
    by_type: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for local_index, candidate in enumerate(candidates):
        grouped[candidate.edge_type].append((local_index, candidate.position))
    for edge_type, values in grouped.items():
        by_type[edge_type] = (
            np.asarray([value[0] for value in values], dtype=np.int64),
            np.asarray([value[1] for value in values], dtype=np.int64),
        )
    return by_type


def _attention_scores(model, lookup, candidate_count: int) -> np.ndarray:
    result = np.zeros(candidate_count, dtype=np.float64)
    for audit_name, audit in model.encoder._last_audits.items():
        encoded = audit_name.split(":", 1)[-1]
        edge_type = tuple(encoded.split("__"))
        if len(edge_type) != 3 or edge_type not in lookup:
            continue
        local_indices, edge_positions = lookup[edge_type]
        positions = torch.as_tensor(
            edge_positions, device=audit.attention.device, dtype=torch.long
        )
        values = audit.attention.index_select(0, positions).mean(dim=-1)
        result[local_indices] += values.detach().cpu().numpy()
    total = float(np.maximum(result, 0.0).sum())
    if total > np.finfo(float).eps:
        result /= total
    elif candidate_count:
        result.fill(1.0 / candidate_count)
    return result


def _unit_attention_scores(
    candidate_scores: np.ndarray,
    candidates: list[EdgeCandidate],
    units: list[EvidenceUnit],
) -> np.ndarray:
    candidate_index = {
        candidate.stable_id: index for index, candidate in enumerate(candidates)
    }
    result = np.zeros(len(units), dtype=np.float64)
    for unit_index, unit in enumerate(units):
        result[unit_index] = sum(
            candidate_scores[candidate_index[member.stable_id]]
            for member in unit.members
        )
    total = float(result.sum())
    if total > np.finfo(float).eps:
        result /= total
    elif len(result):
        result.fill(1.0 / len(result))
    return result


def _edge_masks(graph, units, selected_indices, mode: str):
    if mode not in {"keep", "delete"}:
        raise ValueError(f"unsupported mask mode: {mode}")
    all_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    selected_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    selected = set(int(value) for value in selected_indices)
    for unit_index, unit in enumerate(units):
        for candidate in unit.members:
            all_positions[candidate.edge_type].append(candidate.position)
            if unit_index in selected:
                selected_positions[candidate.edge_type].append(candidate.position)
    masks = {}
    for edge_type, positions in all_positions.items():
        edge_index = graph[edge_type].edge_index
        mask = torch.ones(edge_index.shape[1], device=edge_index.device)
        if mode == "keep":
            mask[torch.as_tensor(positions, device=mask.device)] = 0.0
            keep = selected_positions.get(edge_type, [])
            if keep:
                mask[torch.as_tensor(keep, device=mask.device)] = 1.0
        else:
            remove = selected_positions.get(edge_type, [])
            if remove:
                mask[torch.as_tensor(remove, device=mask.device)] = 0.0
        masks[edge_type] = mask
    return masks


def _prediction_record(
    output,
    model,
    global_index: int,
    reference_indices: np.ndarray,
    reference_ids: np.ndarray,
    reference_position: int,
) -> dict[str, float]:
    probabilities = torch.sigmoid(
        output["coordination_logits"]
    ).detach().cpu().numpy()
    percentile = _percentile(
        probabilities[reference_indices], reference_ids, reference_position
    )
    curvature = model.encoder.common_curvature()
    point = output["user_lorentz"][global_index : global_index + 1]
    prototypes = model.coordination_head.prototypes(curvature)
    distances = lorentz_distance(
        point.unsqueeze(1), prototypes.unsqueeze(0), curvature
    ).squeeze(-1).detach().cpu().numpy()[0]
    return {
        "percentile": float(percentile),
        "probability": float(probabilities[global_index]),
        "distance_to_normal": float(distances[0]),
        "distance_to_coordination": float(distances[1]),
        "geodesic_margin": float(distances[0] - distances[1]),
    }


def _mean_record(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def _fidelity(base_records, candidate_records) -> dict[str, float]:
    percentile_errors = [
        abs(base["percentile"] - candidate["percentile"])
        for base, candidate in zip(base_records, candidate_records)
    ]
    geometry_values = []
    vote_values = []
    margin_errors = []
    for base, candidate in zip(base_records, candidate_records):
        margin_error = abs(
            base["geodesic_margin"] - candidate["geodesic_margin"]
        )
        scale = max(
            abs(base["distance_to_normal"])
            + abs(base["distance_to_coordination"]),
            np.finfo(float).eps,
        )
        geometry_values.append(max(0.0, 1.0 - margin_error / scale))
        margin_errors.append(margin_error)
        vote_values.append(
            (base["geodesic_margin"] > 0)
            == (candidate["geodesic_margin"] > 0)
        )
    return {
        "sufficiency_percentile_error": float(np.mean(percentile_errors)),
        "geometry_fidelity": float(np.mean(geometry_values)),
        "geodesic_margin_absolute_error": float(np.mean(margin_errors)),
        "prototype_vote_agreement": float(np.mean(vote_values)),
    }


def _constraints_pass(fidelity: dict[str, float]) -> bool:
    return (
        fidelity["sufficiency_percentile_error"] <= PREDICTION_TOLERANCE
        and fidelity["geometry_fidelity"] >= GEOMETRY_FIDELITY_MINIMUM
        and fidelity["prototype_vote_agreement"]
        >= PROTOTYPE_VOTE_AGREEMENT_MINIMUM
    )


def _choose_budget(curve: dict[float, dict[str, float]]) -> float:
    for fraction in EDGE_BUDGET_FRACTIONS:
        if fraction in curve and _constraints_pass(curve[fraction]):
            return fraction
    raise AssertionError("the full evidence-unit set must reproduce baseline")


def _mean_pairwise_jaccard(score_rows: list[np.ndarray], count: int) -> float:
    if len(score_rows) < 2 or count <= 0:
        return 1.0
    tops = [set(np.argsort(-row, kind="stable")[:count]) for row in score_rows]
    values = []
    for left in range(len(tops)):
        for right in range(left + 1, len(tops)):
            union = tops[left] | tops[right]
            values.append(len(tops[left] & tops[right]) / max(len(union), 1))
    return float(np.mean(values))


def _scan_provenance(bundle: Path, wanted: set[str]):
    found: dict[str, list[dict[str, str]]] = defaultdict(list)
    columns = [
        "source_type",
        "source_id",
        "target_type",
        "target_id",
        "edge_type",
        "timestamp",
        "evidence_id",
    ]
    for chunk in pd.read_csv(
        bundle / "edges.csv",
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    ):
        keys = (
            chunk["source_type"] + "\x1f" + chunk["source_id"] + "\x1f"
            + chunk["target_type"] + "\x1f" + chunk["target_id"] + "\x1f"
            + chunk["edge_type"]
        )
        hit = chunk.loc[keys.isin(wanted)].copy()
        if hit.empty:
            continue
        hit["_key"] = keys.loc[hit.index]
        for row in hit.to_dict("records"):
            key = row.pop("_key")
            found[key].append(row)
    return found


def _summary(
    values: Iterable[float],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = 1.96 * std / math.sqrt(max(len(array), 1))
    ci_low = float(array.mean() - half)
    ci_high = float(array.mean() + half)
    if lower is not None:
        ci_low = max(lower, ci_low)
    if upper is not None:
        ci_high = min(upper, ci_high)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _aggregate_case_metrics(case_rows: list[dict]) -> dict:
    return {
        "comprehensiveness": _summary(
            (row["prediction"]["comprehensiveness"] for row in case_rows),
            lower=-1.0,
            upper=1.0,
        ),
        "sufficiency_percentile_error": _summary(
            (
                row["prediction"]["sufficiency_percentile_error"]
                for row in case_rows
            ),
            lower=0.0,
            upper=1.0,
        ),
        "sparsity": _summary(
            (row["selection"]["sparsity"] for row in case_rows),
            lower=0.0,
            upper=1.0,
        ),
        "geometry_fidelity": _summary(
            (row["prediction"]["geometry_fidelity"] for row in case_rows),
            lower=0.0,
            upper=1.0,
        ),
        "prototype_vote_agreement": _summary(
            (
                row["prediction"]["prototype_vote_agreement"]
                for row in case_rows
            ),
            lower=0.0,
            upper=1.0,
        ),
        "provenance_coverage": _summary(
            (row["provenance"]["coverage"] for row in case_rows),
            lower=0.0,
            upper=1.0,
        ),
        "timestamp_parse_coverage": _summary(
            (row["temporal"]["timestamp_parse_coverage"] for row in case_rows),
            lower=0.0,
            upper=1.0,
        ),
        "explanation_stability": _summary(
            (
                row["selection"]["checkpoint_topk_jaccard"]
                for row in case_rows
            ),
            lower=0.0,
            upper=1.0,
        ),
    }


def _load_entries(manifest_path: Path):
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(freeze.get("checkpoints", []))
    if len(entries) != 15:
        raise ValueError("formal evidence evaluation requires 15 checkpoints")
    for entry in entries:
        path = resolve_frozen_checkpoint(manifest_path, entry)
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"checkpoint hash mismatch: {path}")
    return entries


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    manifest_path = args.freeze_manifest.resolve()
    bundle = args.bundle.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _load_entries(manifest_path)
    score_frame = pd.read_csv(args.consensus_scores)
    cases = _select_cases(score_frame, args.case_count)
    batch = load_external_bundle_episode(bundle, load_labels=False)
    if torch.any(batch.coordination_mask):
        raise ValueError("label-blind evaluator unexpectedly exposed labels")

    all_ids = np.asarray(batch.graph["user"].node_ids, dtype=str)
    index_lookup = {user_id: index for index, user_id in enumerate(all_ids)}
    reference_ids = score_frame["user_id"].astype(str).to_numpy()
    missing = sorted(set(reference_ids) - set(index_lookup))
    if missing:
        raise ValueError(f"reference universe has {len(missing)} absent users")
    reference_indices = np.asarray(
        [index_lookup[user_id] for user_id in reference_ids], dtype=np.int64
    )
    reference_lookup = {
        user_id: index for index, user_id in enumerate(reference_ids)
    }
    case_ids = cases["user_id"].astype(str).tolist()
    case_indices = {user_id: index_lookup[user_id] for user_id in case_ids}
    candidates = {
        user_id: _candidate_edges(batch.graph, case_indices[user_id])
        for user_id in case_ids
    }
    if any(not values for values in candidates.values()):
        empty = [user_id for user_id, values in candidates.items() if not values]
        raise ValueError(f"selected cases have no model-supported incident edges: {empty}")
    lookups = {
        user_id: _candidate_lookup(values)
        for user_id, values in candidates.items()
    }
    units = {
        user_id: _evidence_units(values)
        for user_id, values in candidates.items()
    }

    device = torch.device(args.device)
    batch.to(device)
    baseline: dict[str, list[dict[str, float]]] = defaultdict(list)
    attention: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata = None
    baseline_started = time.perf_counter()
    for entry in entries:
        checkpoint_path = resolve_frozen_checkpoint(manifest_path, entry)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        probabilities = torch.sigmoid(
            output["coordination_logits"]
        ).detach().cpu().numpy()
        reference_scores = probabilities[reference_indices]
        curvature = model.encoder.common_curvature()
        prototypes = model.coordination_head.prototypes(curvature)
        points = output["user_lorentz"][[case_indices[value] for value in case_ids]]
        distances = lorentz_distance(
            points.unsqueeze(1), prototypes.unsqueeze(0), curvature
        ).squeeze(-1).detach().cpu().numpy()
        for local_index, user_id in enumerate(case_ids):
            global_index = case_indices[user_id]
            percentile = _percentile(
                reference_scores,
                reference_ids,
                reference_lookup[user_id],
            )
            baseline[user_id].append({
                "percentile": float(percentile),
                "probability": float(probabilities[global_index]),
                "distance_to_normal": float(distances[local_index, 0]),
                "distance_to_coordination": float(distances[local_index, 1]),
                "geodesic_margin": float(
                    distances[local_index, 0] - distances[local_index, 1]
                ),
            })
            candidate_scores = _attention_scores(
                model, lookups[user_id], len(candidates[user_id])
            )
            attention[user_id].append(_unit_attention_scores(
                candidate_scores,
                candidates[user_id],
                units[user_id],
            ))
        del model, output, checkpoint, probabilities, points, distances
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_seconds = time.perf_counter() - baseline_started

    selections: dict[str, dict[float, np.ndarray]] = {}
    for user_id in case_ids:
        aggregate = np.mean(np.stack(attention[user_id]), axis=0)
        selections[user_id] = {
            fraction: _relation_stratified_prefix_for_fraction(
                aggregate, units[user_id], fraction
            )
            for fraction in EDGE_BUDGET_FRACTIONS
        }

    keep_records: dict[str, dict[float, list[dict[str, float]]]] = {
        user_id: {fraction: [] for fraction in EDGE_BUDGET_FRACTIONS}
        for user_id in case_ids
    }
    keep_started = time.perf_counter()
    for entry in entries:
        checkpoint_path = resolve_frozen_checkpoint(manifest_path, entry)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = _build_model(checkpoint, metadata).to(device).eval()
        for user_id in case_ids:
            for fraction in EDGE_BUDGET_FRACTIONS:
                masks = _edge_masks(
                    batch.graph,
                    units[user_id],
                    selections[user_id][fraction],
                    "keep",
                )
                with torch.no_grad():
                    output = model(
                        batch.graph,
                        domain=batch.domain,
                        dataset_name=batch.dataset_name,
                        edge_mask_dict=masks,
                    )
                keep_records[user_id][fraction].append(_prediction_record(
                    output,
                    model,
                    case_indices[user_id],
                    reference_indices,
                    reference_ids,
                    reference_lookup[user_id],
                ))
                del output, masks
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    keep_seconds = time.perf_counter() - keep_started

    chosen: dict[str, float] = {}
    fidelity_by_case: dict[str, dict[float, dict[str, float]]] = {}
    for user_id in case_ids:
        fidelity_by_case[user_id] = {}
        for fraction in EDGE_BUDGET_FRACTIONS:
            fidelity = _fidelity(
                baseline[user_id], keep_records[user_id][fraction]
            )
            fidelity_by_case[user_id][fraction] = fidelity
        chosen[user_id] = _choose_budget(fidelity_by_case[user_id])

    deleted: dict[str, list[dict[str, float]]] = defaultdict(list)
    deletion_started = time.perf_counter()
    for entry in entries:
        checkpoint_path = resolve_frozen_checkpoint(manifest_path, entry)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = _build_model(checkpoint, metadata).to(device).eval()
        for user_id in case_ids:
            masks = _edge_masks(
                batch.graph,
                units[user_id],
                selections[user_id][chosen[user_id]],
                "delete",
            )
            with torch.no_grad():
                output = model(
                    batch.graph,
                    domain=batch.domain,
                    dataset_name=batch.dataset_name,
                    edge_mask_dict=masks,
                )
            deleted[user_id].append(_prediction_record(
                output,
                model,
                case_indices[user_id],
                reference_indices,
                reference_ids,
                reference_lookup[user_id],
            ))
            del output, masks
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    deletion_seconds = time.perf_counter() - deletion_started

    selected_units = {
        user_id: [
            units[user_id][int(index)]
            for index in selections[user_id][chosen[user_id]]
        ]
        for user_id in case_ids
    }
    selected_candidates = {
        user_id: [
            member
            for unit in selected_units[user_id]
            for member in unit.members
        ]
        for user_id in case_ids
    }
    wanted = {
        candidate.provenance_key
        for values in selected_candidates.values()
        for candidate in values
    }
    provenance_started = time.perf_counter()
    provenance = _scan_provenance(bundle, wanted)
    provenance_seconds = time.perf_counter() - provenance_started

    case_rows = []
    case_lookup = cases.set_index("user_id").to_dict("index")
    for user_id in case_ids:
        fraction = chosen[user_id]
        chosen_units = selected_units[user_id]
        chosen_candidates = selected_candidates[user_id]
        resolved_units = [
            unit for unit in chosen_units
            if all(
                member.provenance_key in provenance for member in unit.members
            )
        ]
        evidence_rows = [
            row for candidate in chosen_candidates
            for row in provenance.get(candidate.provenance_key, [])
        ]
        timestamps = pd.to_datetime(
            [row.get("timestamp") for row in evidence_rows],
            errors="coerce",
            utc=True,
        )
        valid_timestamps = int(pd.Series(timestamps).notna().sum())
        base_mean = _mean_record(baseline[user_id])
        delete_mean = _mean_record(deleted[user_id])
        fidelity = fidelity_by_case[user_id][fraction]
        selected_count = len(chosen_units)
        selected_edge_instances = len(chosen_candidates)
        raw_risk_change = delete_mean["percentile"] - base_mean["percentile"]
        predicted_coordination = base_mean["geodesic_margin"] > 0
        class_comprehensiveness = (
            -raw_risk_change if predicted_coordination else raw_risk_change
        )
        constraint_search = {}
        for budget in EDGE_BUDGET_FRACTIONS:
            budget_fidelity = fidelity_by_case[user_id][budget]
            budget_selected = len(selections[user_id][budget])
            constraint_search[str(budget)] = {
                "selected_evidence_units": budget_selected,
                "sparsity": budget_selected / len(units[user_id]),
                **budget_fidelity,
                "constraints_passed": _constraints_pass(budget_fidelity),
            }
        case_rows.append({
            "case_id": f"user:{user_id}",
            "operation": str(batch.episode_id),
            "risk_stratum": case_lookup[user_id]["risk_stratum"],
            "labels_consumed": False,
            "target_label_values_consumed": False,
            "eligibility_membership_consumed": True,
            "selection": {
                "proposal": "15-checkpoint mean intrinsic attention times reliability",
                "search": (
                    "smallest relation-stratified attention-ranked edge-budget "
                    "prefix satisfying constraints"
                ),
                "edge_budget_fraction": fraction,
                "candidate_evidence_units": len(units[user_id]),
                "candidate_edge_instances": len(candidates[user_id]),
                "selected_evidence_units": selected_count,
                "selected_edge_instances": selected_edge_instances,
                "sparsity": selected_count / len(units[user_id]),
                "relation_counts": dict(Counter(
                    unit.relation for unit in chosen_units
                )),
                "checkpoint_topk_jaccard": _mean_pairwise_jaccard(
                    attention[user_id], selected_count
                ),
                "constraint_search": constraint_search,
            },
            "prediction": {
                "frozen_consensus_percentile": float(
                    case_lookup[user_id]["consensus_percentile"]
                ),
                "recomputed_full_percentile": base_mean["percentile"],
                "keep_only_percentile": _mean_record(
                    keep_records[user_id][fraction]
                )["percentile"],
                "deleted_percentile": delete_mean["percentile"],
                "predicted_class": (
                    "coordination" if predicted_coordination else "normal"
                ),
                "risk_percentile_change_after_deletion": raw_risk_change,
                "comprehensiveness": class_comprehensiveness,
                **fidelity,
            },
            "geometry": {
                "full_geodesic_margin": base_mean["geodesic_margin"],
                "keep_only_geodesic_margin": _mean_record(
                    keep_records[user_id][fraction]
                )["geodesic_margin"],
                "coordinate_averaging_disabled": True,
                "checkpoint_count": len(entries),
            },
            "provenance": {
                "selected_evidence_units": selected_count,
                "selected_edge_instances": selected_edge_instances,
                "resolved_evidence_units": len(resolved_units),
                "coverage": len(resolved_units) / max(selected_count, 1),
                "unique_evidence_ids": sorted({
                    row.get("evidence_id") for row in evidence_rows
                    if row.get("evidence_id")
                }),
            },
            "temporal": {
                "scope": "full_release_snapshot",
                "historical_cutoff_claimed": False,
                "evidence_rows": len(evidence_rows),
                "timestamp_parse_coverage": (
                    valid_timestamps / max(len(evidence_rows), 1)
                ),
            },
        })

    case_path = output_dir / "cases.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = _aggregate_case_metrics(case_rows)
    metrics_by_stratum = {
        stratum: _aggregate_case_metrics([
            row for row in case_rows if row["risk_stratum"] == stratum
        ])
        for stratum in ("high", "medium", "low")
    }
    result = {
        "schema_version": "hypertrace.constrained-evidence-evaluation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(
            row["provenance"]["coverage"] == 1.0 for row in case_rows
        ) else "failed",
        "evaluation_scope": "label_blind_full_release_snapshot",
        "operation": str(batch.episode_id),
        "case_count": len(case_rows),
        "case_selection": "label-blind high/mid/low consensus strata",
        "checkpoint_count": len(entries),
        "constraints": {
            "prediction_percentile_error_max": PREDICTION_TOLERANCE,
            "geometry_fidelity_min": GEOMETRY_FIDELITY_MINIMUM,
            "prototype_vote_agreement_min": PROTOTYPE_VOTE_AGREEMENT_MINIMUM,
            "temporal_scope": "full_release_snapshot",
            "provenance_required": True,
        },
        "labels_consumed": False,
        "target_label_values_consumed": False,
        "eligibility_membership_consumed": True,
        "metrics": metrics,
        "metrics_by_risk_stratum": metrics_by_stratum,
        "edge_budget_choices": dict(Counter(
            str(value) for value in chosen.values()
        )),
        "latency_seconds": {
            "baseline_and_proposal": baseline_seconds,
            "keep_only_search": keep_seconds,
            "selected_edge_deletion": deletion_seconds,
            "provenance_scan": provenance_seconds,
            "total": time.perf_counter() - started,
        },
        "artifacts": {
            "cases": case_path.name,
            "cases_sha256": _sha256(case_path),
            "freeze_manifest_sha256": _sha256(manifest_path),
            "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
            "consensus_scores_sha256": _sha256(args.consensus_scores),
        },
        "errors": [],
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--consensus-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-count", type=int, default=12)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
