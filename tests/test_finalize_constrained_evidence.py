import hashlib
import json

import pytest

from scripts.finalize_constrained_evidence import _load_run, _rows


def _case(stratum: str, *, labels_consumed: bool = False) -> dict:
    return {
        "operation": "honduras",
        "risk_stratum": stratum,
        "labels_consumed": labels_consumed,
        "target_label_values_consumed": False,
        "selection": {
            "candidate_evidence_units": 10,
            "selected_evidence_units": 2,
            "sparsity": 0.2,
            "checkpoint_topk_jaccard": 0.9,
        },
        "prediction": {
            "comprehensiveness": 0.1,
            "sufficiency_percentile_error": 0.01,
            "geometry_fidelity": 0.97,
            "prototype_vote_agreement": 1.0,
        },
        "provenance": {"coverage": 1.0},
        "temporal": {"timestamp_parse_coverage": 1.0},
    }


def _write_run(directory, cases) -> None:
    directory.mkdir()
    payload = "".join(json.dumps(case) + "\n" for case in cases)
    cases_path = directory / "cases.jsonl"
    cases_path.write_text(payload, encoding="utf-8")
    audit = {
        "status": "passed",
        "operation": "honduras",
        "case_count": len(cases),
        "checkpoint_count": 15,
        "labels_consumed": False,
        "target_label_values_consumed": False,
        "artifacts": {
            "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()
        },
    }
    (directory / "audit.json").write_text(json.dumps(audit), encoding="utf-8")


def test_formal_finalizer_validates_and_summarizes_strata(tmp_path) -> None:
    directory = tmp_path / "formal"
    _write_run(directory, [_case("high"), _case("medium"), _case("low")])

    run = _load_run("honduras", directory, expected_cases=3)
    rows = _rows("honduras", run["cases"])

    assert [row["risk_stratum"] for row in rows] == [
        "all",
        "high",
        "medium",
        "low",
    ]
    assert rows[0]["sparsity"] == pytest.approx(0.2)


def test_formal_finalizer_rejects_case_level_label_access(tmp_path) -> None:
    directory = tmp_path / "formal"
    _write_run(
        directory,
        [_case("high", labels_consumed=True), _case("medium"), _case("low")],
    )

    with pytest.raises(ValueError, match="consumed labels"):
        _load_run("honduras", directory, expected_cases=3)
