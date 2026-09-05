from __future__ import annotations

import numpy as np

from scripts.evaluate_constrained_evidence import EvidenceUnit
from scripts.evaluate_explanation_controls import (
    _degree_matched_sample,
    _stable_seed,
    _topk,
)


def _units() -> list[EvidenceUnit]:
    return [
        EvidenceUnit("a", "posts", ()),
        EvidenceUnit("b", "posts", ()),
        EvidenceUnit("c", "retweets", ()),
        EvidenceUnit("d", "retweets", ()),
    ]


def test_topk_uses_stable_id_to_break_score_ties() -> None:
    selected = _topk(np.asarray([0.2, 0.2, 0.1, 0.4]), _units(), 3)
    assert selected.tolist() == [3, 0, 1]


def test_degree_matching_is_unique_relation_aware_and_reproducible() -> None:
    units = _units()
    degrees = np.asarray([1.0, 1.1, 2.0, 2.2])
    target = np.asarray([0, 2])
    first = _degree_matched_sample(
        units, degrees, target, np.random.default_rng(7)
    )
    second = _degree_matched_sample(
        units, degrees, target, np.random.default_rng(7)
    )
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    selected, mean_error, exact_fraction, overlap_fraction = first
    assert len(set(selected.tolist())) == len(target)
    assert {units[index].relation for index in selected} == {"posts", "retweets"}
    assert set(selected.tolist()).isdisjoint(set(target.tolist()))
    assert np.isclose(mean_error, 0.15)
    assert exact_fraction == 0.0
    assert overlap_fraction == 0.0


def test_stable_seed_changes_with_control_and_replicate() -> None:
    values = {
        _stable_seed(20260828, "honduras", "case", method, replicate)
        for method in ("random", "degree")
        for replicate in range(3)
    }
    assert len(values) == 6
