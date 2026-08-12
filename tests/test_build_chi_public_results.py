import json

from scripts.build_chi_public_results import _markdown, _top_one_percent


def test_top_one_percent_selects_exact_budget() -> None:
    audit = {"review_budget_metrics": [
        {"budget_fraction": 0.005, "reviewed_accounts": 1},
        {
            "budget_fraction": 0.01, "reviewed_accounts": 2,
            "true_io_hits": 1, "precision_at_k": 0.5, "recall_at_k": 0.2,
            "lift_at_k": 10.0, "mean_pairwise_checkpoint_jaccard": 0.4,
        },
    ]}
    assert _top_one_percent(audit)["reviewed_accounts"] == 2


def test_markdown_discloses_mixed_geometry_result() -> None:
    metrics = {
        name: {metric: {"mean": 0.5, "std": 0.1} for metric in (
            "auprc", "auroc", "f1", "balanced_accuracy", "brier", "ece"
        )}
        for name in (
            "lorentz_observable18", "lorentz_scratch_observable18",
            "euclidean_observable18", "lorentz_no_temporal",
        )
    }
    results = {
        "external_evaluation": {
            name: {"metrics": {metric: {"mean": 0.5, "std": 0.1} for metric in ("auroc", "auprc")}}
            for name in ("honduras", "uae")
        },
        "rank_consensus": {
            name: {"top_1_percent": {
                "precision_at_k": 0.1, "recall_at_k": 0.2, "lift_at_k": 10.0,
                "mean_pairwise_checkpoint_jaccard": 0.4,
            }} for name in ("honduras", "uae")
        },
        "model_ablation": {"summaries": metrics},
    }
    text = _markdown(results)
    assert "hyperbolic geometry is not claimed" in text
    assert "Psychological features and role classification are excluded" in text
