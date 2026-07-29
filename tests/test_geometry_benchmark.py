import csv
import json

import torch

from explainability.geo_pgexplainer import GeoPGExplanation
from explainability.geometry_benchmark import (
    benchmark_row,
    hard_masks_at_budget,
    save_benchmark_csv,
    save_edge_masks,
    summarize_paired_benchmark,
)


def _explanation():
    return GeoPGExplanation(
        edge_masks={("user", "follows", "user"): torch.tensor([0.2, 0.8])},
        losses={"total": 0.1},
        metrics={
            "prediction_fidelity_loss": 0.01,
            "geodesic_distortion": 0.2,
            "radial_order_agreement": 0.75,
            "role_fidelity": 0.8,
            "selected_edge_fraction_at_0.5": 0.5,
        },
        mode="pgexplainer",
    )


def test_benchmark_row_records_backend_and_four_metrics():
    row = benchmark_row(_explanation(), run_id="r1", seed=42)
    assert row.geometry_backend == "projection_head"
    assert row.geodesic_distortion == 0.2
    assert row.radial_order_agreement == 0.75
    assert row.role_fidelity == 0.8


def test_benchmark_artifacts_roundtrip(tmp_path):
    explanation = _explanation()
    mask_path = tmp_path / "masks.json"
    csv_path = tmp_path / "benchmark.csv"
    save_edge_masks(explanation, str(mask_path))
    save_benchmark_csv([benchmark_row(explanation, "r1", 42)], str(csv_path))
    with open(mask_path, encoding="utf-8") as handle:
        assert "user__follows__user" in json.load(handle)["edge_masks"]
    with open(csv_path, encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["method"] == "pgexplainer"


def test_paired_summary_requires_multiple_consistent_seeds():
    rows = []
    for seed in range(5):
        baseline = _explanation()
        geo = _explanation()
        geo.mode = "geo_pgexplainer"
        geo.metrics = dict(geo.metrics)
        geo.metrics["geodesic_distortion"] = 0.1
        geo.metrics["radial_order_agreement"] = 0.9
        rows.extend([
            benchmark_row(baseline, "r1", seed),
            benchmark_row(geo, "r1", seed),
        ])
    summary = summarize_paired_benchmark(rows)
    assert summary["paired_seed_count"] == 5
    assert summary["claim_ready"] is True


def test_hard_masks_use_same_global_budget_across_relations():
    masks = {
        ("user", "follows", "user"): torch.tensor([0.1, 0.9, 0.8]),
        ("user", "similar", "user"): torch.tensor([0.7, 0.2]),
    }
    hard = hard_masks_at_budget(masks, 0.4)
    assert sum(int(mask.sum()) for mask in hard.values()) == 2
    assert hard[("user", "follows", "user")].tolist() == [0.0, 1.0, 1.0]
