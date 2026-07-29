"""Utilities for the PGExplainer versus Geo-PGExplainer falsification study."""

from __future__ import annotations

import csv
import json
import os
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from .geo_pgexplainer import GeoPGExplanation, edge_type_name


@dataclass
class GeometryBenchmarkRow:
    run_id: str
    seed: int
    method: str
    prediction_fidelity_loss: float
    geodesic_distortion: float
    radial_order_agreement: float
    role_fidelity: float
    selected_edge_fraction_at_0_5: float
    geometry_backend: str = "projection_head"


@dataclass
class GeometryBudgetRow:
    run_id: str
    seed: int
    method: str
    edge_budget: float
    selected_edge_count: int
    total_edge_count: int
    prediction_fidelity_loss: float
    geodesic_distortion: float
    radial_order_agreement: float
    role_fidelity: float
    geometry_backend: str = "projection_head"


def benchmark_row(
    explanation: GeoPGExplanation,
    run_id: str,
    seed: int,
    geometry_backend: str = "projection_head",
) -> GeometryBenchmarkRow:
    metrics = explanation.metrics
    return GeometryBenchmarkRow(
        run_id=run_id,
        seed=seed,
        method=explanation.mode,
        prediction_fidelity_loss=metrics["prediction_fidelity_loss"],
        geodesic_distortion=metrics["geodesic_distortion"],
        radial_order_agreement=metrics["radial_order_agreement"],
        role_fidelity=metrics.get("role_fidelity", float("nan")),
        selected_edge_fraction_at_0_5=metrics["selected_edge_fraction_at_0.5"],
        geometry_backend=geometry_backend,
    )


def save_edge_masks(explanation: GeoPGExplanation, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, List[float]] = {
        edge_type_name(edge_type): mask.tolist()
        for edge_type, mask in explanation.edge_masks.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "method": explanation.mode,
            "losses": explanation.losses,
            "metrics": explanation.metrics,
            "edge_masks": payload,
        }, handle, ensure_ascii=False, indent=2)
    return path


def save_benchmark_csv(rows: Iterable[GeometryBenchmarkRow], path: str) -> str:
    rows = list(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(
        GeometryBenchmarkRow.__dataclass_fields__.keys()
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def save_budget_csv(rows: Iterable[GeometryBudgetRow], path: str) -> str:
    rows = list(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(
        GeometryBudgetRow.__dataclass_fields__.keys()
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def hard_masks_at_budget(edge_masks, budget: float):
    """Create globally matched hard masks while retaining relation keys."""
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    import torch

    flat = torch.cat([mask.reshape(-1) for mask in edge_masks.values()])
    keep = max(1, int(round(float(budget) * flat.numel())))
    selected = torch.topk(flat, k=min(keep, flat.numel())).indices
    hard_flat = torch.zeros_like(flat)
    hard_flat[selected] = 1.0
    result = {}
    offset = 0
    for edge_type, mask in edge_masks.items():
        count = mask.numel()
        result[edge_type] = hard_flat[offset:offset + count].reshape(mask.shape)
        offset += count
    return result


def summarize_paired_benchmark(
    rows: Iterable[GeometryBenchmarkRow],
    prediction_tolerance: float = 0.02,
) -> Dict[str, object]:
    """Summarize paired seeds without overstating an underpowered experiment."""
    grouped: Dict[int, Dict[str, GeometryBenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.seed, {})[row.method] = row
    pairs = [methods for methods in grouped.values()
             if {"pgexplainer", "geo_pgexplainer"} <= set(methods)]

    distortion_reduction = [
        pair["pgexplainer"].geodesic_distortion
        - pair["geo_pgexplainer"].geodesic_distortion
        for pair in pairs
    ]
    radial_gain = [
        pair["geo_pgexplainer"].radial_order_agreement
        - pair["pgexplainer"].radial_order_agreement
        for pair in pairs
    ]
    prediction_change = [
        pair["geo_pgexplainer"].prediction_fidelity_loss
        - pair["pgexplainer"].prediction_fidelity_loss
        for pair in pairs
    ]

    def mean(values):
        return statistics.fmean(values) if values else float("nan")

    n_pairs = len(pairs)
    positive_distortion_pairs = sum(value > 0 for value in distortion_reduction)
    nonnegative_radial_pairs = sum(value >= 0 for value in radial_gain)
    prediction_within_tolerance = all(
        value <= prediction_tolerance for value in prediction_change
    ) if pairs else False
    claim_ready = (
        n_pairs >= 5
        and positive_distortion_pairs >= math.ceil(0.8 * n_pairs)
        and nonnegative_radial_pairs >= math.ceil(0.8 * n_pairs)
        and prediction_within_tolerance
    )
    return {
        "paired_seed_count": n_pairs,
        "mean_geodesic_distortion_reduction": mean(distortion_reduction),
        "mean_radial_order_agreement_gain": mean(radial_gain),
        "mean_prediction_fidelity_loss_change": mean(prediction_change),
        "positive_distortion_reduction_pairs": positive_distortion_pairs,
        "nonnegative_radial_gain_pairs": nonnegative_radial_pairs,
        "prediction_tolerance": prediction_tolerance,
        "prediction_within_tolerance": prediction_within_tolerance,
        "claim_ready": claim_ready,
        "claim_note": (
            "claim_ready is a minimum engineering gate, not statistical proof; "
            "report confidence intervals and a paired significance test in the paper."
        ),
    }
