import hashlib
import json

from scripts.audit_honduras_uae_source import audit_file, audit_root


def _write_jsonl(path, rows):
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    return {
        "country": "honduras",
        "cohort": "malicious_io",
        "good_value": 0,
        "bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
    }


def test_audit_file_checks_labels_time_text_and_relations(tmp_path):
    path = tmp_path / "sample.jsonl"
    contract = _write_jsonl(path, [
        {
            "tweetid": 10,
            "userid": 1,
            "tweet_text": "first",
            "tweet_time": 1_600_000_000_000,
            "good": 0,
            "in_reply_to_userid": 2,
            "retweet_userid": None,
            "quoted_tweet_tweetid": None,
            "user_mentions": "[2]",
        },
        {
            "tweetid": 11,
            "userid": 2,
            "tweet_text": "second",
            "tweet_time": 1_600_000_001_000,
            "good": 0,
            "in_reply_to_userid": None,
            "retweet_userid": 1.0,
            "quoted_tweet_tweetid": 10,
            "user_mentions": "[]",
        },
    ])

    report, users, tweets = audit_file(path, contract)

    assert report["passed"] is True
    assert users == {"1", "2"}
    assert tweets == {"10", "11"}
    assert report["relations"]["reply"]["internal_endpoint_fraction"] == 1.0
    assert report["relations"]["retweet"]["internal_endpoint_fraction"] == 1.0
    assert report["relations"]["quote"]["events"] == 1
    assert report["time"]["coverage"] == 1.0


def test_audit_file_rejects_label_semantic_mismatch(tmp_path):
    path = tmp_path / "sample.jsonl"
    contract = _write_jsonl(path, [{
        "tweetid": 10,
        "userid": 1,
        "tweet_text": "text",
        "tweet_time": 1_600_000_000_000,
        "good": 1,
        "in_reply_to_userid": None,
        "retweet_userid": None,
        "quoted_tweet_tweetid": None,
        "user_mentions": "[]",
    }])

    report, _, _ = audit_file(path, contract)

    assert report["passed"] is False
    assert "unexpected_good_label" in report["errors"]


def test_root_audit_is_partial_without_all_official_files(tmp_path):
    report = audit_root(tmp_path, require_all=False)

    assert report["status"] == "failed"
    assert report["passed"] is False
    assert len(report["missing_files"]) == 4
