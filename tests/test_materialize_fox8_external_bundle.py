import gzip
import json

import pandas as pd

from scripts.audit_external_bundle import audit_bundle
from scripts.materialize_fox8_external_bundle import materialize


def test_fox8_materializer_keeps_bot_labels_out_of_cib_metrics(tmp_path):
    source = tmp_path / "fox8.ndjson.gz"
    record = {
        "user_id": 10,
        "label": "bot",
        "user_tweets": [{
            "id_str": "tweet-1", "created_at": "Wed Apr 18 14:09:01 +0000 2018",
            "text": "hello @target https://example.test #tag", "favorite_count": 2,
            "in_reply_to_user_id_str": "11",
            "entities": {"user_mentions": [{"id_str": "12"}], "hashtags": [{"text": "tag"}]},
            "user": {"followers_count": 10, "friends_count": 2},
        }],
    }
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    output = tmp_path / "bundle"
    manifest = materialize(source, output)
    report = audit_bundle(output)

    assert manifest["evaluation_scope"] == "bot_transfer_traceability"
    assert report["status"] == "traceability_only"
    assert report["checks"]["bot_binary_labels"] is True
    assert report["checks"]["cib_supervised_metrics_allowed"] is False
    assert pd.read_csv(output / "events.csv")["evidence_id"].tolist() == ["tweet:tweet-1"]
    assert len(pd.read_csv(output / "edges.csv")) == 2
