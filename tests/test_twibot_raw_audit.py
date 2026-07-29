import json

from data_processing.twibot_raw_audit import (
    CHUNK_SIZE,
    audit_twibot_directory,
    iter_json_records,
    load_core_ids,
)


def _write_fixture(root):
    (root / "label.csv").write_text(
        "id,label\nu1,human\nu2,bot\n", encoding="utf-8"
    )
    (root / "split.csv").write_text(
        "id,split\nu1,train\nu2,test\n", encoding="utf-8"
    )
    (root / "edge.csv").write_text(
        "source_id,relation,target_id\n"
        "u1,following,u3\n"
        "u4,followers,u1\n"
        "u1,post,t1\n",
        encoding="utf-8",
    )
    (root / "user.json").write_text(
        json.dumps([
            {
                "id": "u1",
                "description": "DO-NOT-LEAK-PROFILE",
                "public_metrics": {"followers_count": 2},
            },
            {"id": "u3", "description": "boundary"},
        ]),
        encoding="utf-8",
    )
    (root / "tweet_0.json").write_text(
        json.dumps([
            {
                "id": "t1",
                "author_id": "u1",
                "text": "DO-NOT-LEAK-TWEET",
                "created_at": "2024-01-01T00:00:00Z",
                "referenced_tweets": [{"type": "retweeted", "id": "t0"}],
            }
        ]),
        encoding="utf-8",
    )
    (root / "README.md").write_text("fixture", encoding="utf-8")


def test_audit_reports_schema_counts_and_core_coverage_without_raw_text(tmp_path):
    _write_fixture(tmp_path)
    report = audit_twibot_directory(tmp_path, core_ids={"u1"})
    assert report["detected_kind"] == "twibot22_raw"
    edge = next(item for item in report["csv_files"] if item["path"].endswith("edge.csv"))
    assert edge["row_count"] == 3
    assert edge["distributions"]["relation"] == {
        "followers": 1,
        "following": 1,
        "post": 1,
    }
    assert edge["core_coverage"]["core_as_source_rows"] == 2
    assert edge["core_coverage"]["core_as_target_rows"] == 1
    assert edge["core_coverage"]["incident_noncore_ids"] == 3
    assert edge["core_coverage"]["core_source_ids"] == 1
    assert edge["core_coverage"]["core_target_ids"] == 1
    tweet = next(
        item for item in report["json_files"] if item["path"].endswith("tweet_0.json")
    )
    assert tweet["field_coverage"]["created_at_non_null"] == 1
    assert tweet["core_coverage"]["records_authored_by_core"] == 1
    serialized = json.dumps(report)
    assert "DO-NOT-LEAK-PROFILE" not in serialized
    assert "DO-NOT-LEAK-TWEET" not in serialized


def test_jsonl_and_core_id_csv_are_supported(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    assert [row["id"] for row in iter_json_records(path)] == ["1", "2"]
    core = tmp_path / "core.csv"
    core.write_text("user_id\n1\n2\n", encoding="utf-8")
    assert load_core_ids(core) == {"1", "2"}


def test_json_array_record_can_cross_streaming_chunk_boundary(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(
        json.dumps([{"id": "large", "text": "x" * (CHUNK_SIZE + 20)}]),
        encoding="utf-8",
    )
    records = list(iter_json_records(path))
    assert records[0]["id"] == "large"
    assert len(records[0]["text"]) == CHUNK_SIZE + 20


def test_incomplete_directory_is_reported(tmp_path):
    report = audit_twibot_directory(tmp_path)
    assert report["detected_kind"] == "incomplete_or_unknown"
    assert report["warnings"]
