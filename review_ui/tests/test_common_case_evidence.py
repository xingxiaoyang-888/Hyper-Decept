from __future__ import annotations

import json
from pathlib import Path

from review_ui.build_common_case_evidence import enrich


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_common_records_are_time_stratified_without_emitting_labels(tmp_path) -> None:
    package_path = tmp_path / "study_cases.private.json"
    package_path.write_text(json.dumps({
        "cases": [
            {"case_id": "HT-H-01", "operation": "honduras", "ground_truth": "coordinated"},
            {"case_id": "HT-U-01", "operation": "uae", "ground_truth": "not_coordinated"},
        ]
    }), encoding="utf-8")
    formal_root = tmp_path / "formal"
    write_jsonl(formal_root / "honduras" / "cases.jsonl", [{"case_id": "user:100"}])
    write_jsonl(formal_root / "uae" / "cases.jsonl", [{"case_id": "user:200"}])

    source_root = tmp_path / "source"
    source_root.mkdir()
    for filename in (
        "honduras-good-anonymized.jsonl",
        "uae-good-anonymized.jsonl",
    ):
        (source_root / filename).write_text("", encoding="utf-8")
    write_jsonl(source_root / "honduras-bad-anonymized.jsonl", [
        {
            "tweetid": str(1000 + index),
            "userid": "100",
            "tweet_text": f"event {index} @rawhandle https://example.test/{index}",
            "tweet_time": 1_600_000_000_000 + index * 1000,
            "good": 0,
        }
        for index in range(12)
    ])
    write_jsonl(source_root / "uae-bad-anonymized.jsonl", [{
        "tweetid": "2000",
        "userid": "200",
        "tweet_text": "single activity",
        "tweet_time": 1_600_000_000_000,
        "good": 1,
    }])

    output = tmp_path / "enriched.json"
    audit = enrich(package_path, formal_root, source_root, output, "test-salt")
    package = json.loads(output.read_text(encoding="utf-8"))
    first, second = package["cases"]

    assert audit["all_cases_populated"] is True
    assert len(first["common_case_evidence"]) == 8
    assert first["common_case_summary"]["source_activity_count"] == 12
    assert len(second["common_case_evidence"]) == 1
    assert first["common_case_contract"]["ground_truth_consumed"] is False
    assert first["common_case_contract"]["model_output_consumed"] is False
    assert first["common_case_contract"]["explainer_ranking_consumed"] is False
    serialized_records = json.dumps(first["common_case_evidence"])
    assert "@rawhandle" not in serialized_records
    assert "https://example.test" not in serialized_records
    assert "tweetid" not in serialized_records
    assert "userid" not in serialized_records
