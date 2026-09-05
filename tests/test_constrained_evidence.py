import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from scripts.evaluate_constrained_evidence import (
    EdgeCandidate,
    _candidate_edges,
    _choose_budget,
    _evidence_units,
    _fidelity,
    _prefix_for_fraction,
    _relation_stratified_prefix_for_fraction,
    _select_cases,
)


def test_budget_choice_uses_complete_curve_and_first_passing_budget() -> None:
    failed = {
        "sufficiency_percentile_error": 0.03,
        "geometry_fidelity": 0.99,
        "prototype_vote_agreement": 1.0,
    }
    passed = {
        "sufficiency_percentile_error": 0.01,
        "geometry_fidelity": 0.99,
        "prototype_vote_agreement": 1.0,
    }
    curve = {
        fraction: (passed if fraction >= 0.10 else failed)
        for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
    }

    assert _choose_budget(curve) == 0.10
    assert len(curve) == 7


def test_candidate_edges_include_incoming_and_outgoing_user_relations() -> None:
    graph = HeteroData()
    graph["user"].node_ids = ["u0", "u1"]
    graph["tweet"].node_ids = ["t0", "t1"]
    graph[("user", "posts", "tweet")].edge_index = torch.tensor(
        [[0, 1], [0, 1]]
    )
    graph[("tweet", "authored_by", "user")].edge_index = torch.tensor(
        [[0, 1], [0, 1]]
    )

    candidates = _candidate_edges(graph, 0)

    assert [(row.relation, row.source_id, row.target_id) for row in candidates] == [
        ("authored_by", "t0", "u0"),
        ("posts", "u0", "t0"),
    ]
    units = _evidence_units(candidates)
    assert len(units) == 1
    assert units[0].relation == "posts"
    assert len(units[0].members) == 2


def test_case_selection_is_label_blind_and_stratified() -> None:
    frame = pd.DataFrame({
        "user_id": [str(value) for value in range(12)],
        "consensus_percentile": np.linspace(0.01, 1.0, 12),
        "consensus_rank": np.arange(12, 0, -1),
        "checkpoint_rank_std": np.linspace(0.1, 0.2, 12),
    })

    selected = _select_cases(frame, 6)

    assert set(selected["risk_stratum"]) == {"high", "medium", "low"}
    assert selected.groupby("risk_stratum").size().to_dict() == {
        "high": 2,
        "low": 2,
        "medium": 2,
    }
    assert not any("label" in column for column in selected.columns)


def test_prefix_selection_uses_smallest_deterministic_edge_budget() -> None:
    candidates = [
        EdgeCandidate(("user", "posts", "tweet"), index, "u", str(index))
        for index in range(4)
    ]
    scores = np.asarray([0.50, 0.30, 0.15, 0.05])

    assert _prefix_for_fraction(scores, candidates, 0.50).tolist() == [0, 1]
    assert _prefix_for_fraction(scores, candidates, 0.75).tolist() == [0, 1, 2]
    assert len(_prefix_for_fraction(scores, candidates, 1.0)) == 4


def test_relation_stratified_prefix_preserves_each_relation() -> None:
    candidates = [
        EdgeCandidate(("user", "posts", "tweet"), index, "u", str(index))
        for index in range(3)
    ] + [
        EdgeCandidate(("user", "retweets", "tweet"), index, "u", str(index + 3))
        for index in range(2)
    ]
    units = _evidence_units(candidates)
    scores = np.asarray([0.50, 0.30, 0.10, 0.09, 0.01])

    selected = _relation_stratified_prefix_for_fraction(scores, units, 0.20)
    relations = {units[int(index)].relation for index in selected}

    assert relations == {"posts", "retweets"}
    assert len(selected) == 2


def test_fidelity_reports_prediction_geometry_and_vote_preservation() -> None:
    base = [{
        "percentile": 0.90,
        "probability": 0.80,
        "distance_to_normal": 2.0,
        "distance_to_coordination": 1.0,
        "geodesic_margin": 1.0,
    }]
    kept = [{
        "percentile": 0.89,
        "probability": 0.79,
        "distance_to_normal": 1.98,
        "distance_to_coordination": 1.01,
        "geodesic_margin": 0.97,
    }]

    result = _fidelity(base, kept)

    assert np.isclose(result["sufficiency_percentile_error"], 0.01)
    assert np.isclose(result["geometry_fidelity"], 0.99)
    assert result["prototype_vote_agreement"] == 1.0
