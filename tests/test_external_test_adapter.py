import gzip
import json

import pandas as pd

from scripts.external_test_adapter import build_external_report
from scripts.materialize_fox8_external_bundle import materialize


def test_external_adapter_never_reports_cib_for_fox8(tmp_path):
    source = tmp_path / "fox8.ndjson.gz"
    record = {
        "user_id": 10,
        "label": "bot",
        "user_tweets": [
            {"id_str": "a", "created_at": "Wed Apr 18 14:09:01 +0000 2018", "text": "a"},
            {"id_str": "b", "created_at": "Thu Apr 19 14:09:01 +0000 2018", "text": "b"},
        ],
    }
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + chr(10))
    bundle = tmp_path / "bundle"
    materialize(source, bundle)
    report = build_external_report(bundle, windows=2)
    assert report["audit_status"] == "traceability_only"
    assert report["supervised_metrics_allowed"] is False
    assert report["prediction_status"] == "not_run"
    assert len(report["windows"]) == 2
    assert all(row["evidence_coverage"] == 1.0 for row in report["windows"] if row["event_count"])


def test_external_adapter_keeps_io_and_cib_metric_scopes_separate(tmp_path):
    bundle = tmp_path / "io_bundle"
    bundle.mkdir()
    pd.DataFrame([{"node_id": "u1", "node_type": "user"}]).to_csv(
        bundle / "nodes.csv", index=False
    )
    pd.DataFrame([{
        "source_id": "u1", "target_id": "t1", "edge_type": "posts",
        "timestamp": "2020-01-01T00:00:00Z", "evidence_id": "tweet:t1",
    }]).to_csv(bundle / "edges.csv", index=False)
    pd.DataFrame([{
        "event_id": "t1", "evidence_id": "tweet:t1", "actor_id": "u1",
        "timestamp": "2020-01-01T00:00:00Z", "text": "example",
    }]).to_csv(bundle / "events.csv", index=False)
    pd.DataFrame([{
        "node_id": "u1", "is_bad": 1, "is_known": True,
    }]).to_csv(bundle / "labels.csv", index=False)
    (bundle / "manifest.json").write_text(json.dumps({
        "dataset_id": "honduras-uae-io",
        "evaluation_scope": "real_information_operation_external_evaluation",
        "label_semantics": "information_operation_membership",
        "capabilities": {"source_verified_io_labels": True},
    }), encoding="utf-8")

    report = build_external_report(bundle, windows=1)

    assert report["audit_status"] == "supervised_reportable"
    assert report["supervised_metrics_allowed"] is True
    assert report["io_supervised_metrics_allowed"] is True
    assert report["cib_supervised_metrics_allowed"] is False
