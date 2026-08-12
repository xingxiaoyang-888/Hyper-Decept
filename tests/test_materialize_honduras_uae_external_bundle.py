import json

import pandas as pd

from scripts.materialize_honduras_uae_external_bundle import materialize


def _row(*, tweet, user, good, retweet_tweet=None, retweet_user=None,
         reply_tweet=None, reply_user=None, mentions="[]"):
    return {
        "tweetid": tweet,
        "userid": user,
        "follower_count": 10,
        "following_count": 4,
        "tweet_text": f"tweet {tweet}",
        "tweet_time": 1_600_000_000_000 + tweet,
        "in_reply_to_userid": reply_user,
        "in_reply_to_tweetid": reply_tweet,
        "quoted_tweet_tweetid": None,
        "retweet_userid": retweet_user,
        "retweet_tweetid": retweet_tweet,
        "like_count": 2,
        "hashtags": "['tag']",
        "urls": "[]",
        "user_mentions": mentions,
        "good": good,
    }


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materializes_operation_disjoint_bundle_without_semantic_fabrication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "honduras-bad-anonymized.jsonl", [
        _row(tweet=10, user=1, good=0, retweet_tweet=99,
             retweet_user=3, mentions="[2, 3]"),
    ])
    _write(source / "honduras-good-anonymized.jsonl", [
        _row(tweet=11, user=2, good=1, reply_tweet=10, reply_user=1),
    ])
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": "hypertrace.honduras-uae-source-audit.v1",
        "files": [
            {"country": "honduras", "cohort": "malicious_io", "passed": True},
            {"country": "honduras", "cohort": "genuine_control", "passed": True},
        ],
    }), encoding="utf-8")
    output = tmp_path / "bundle"

    manifest = materialize(
        source, output, country="honduras", source_audit=audit
    )

    labels = pd.read_csv(output / "labels.csv")
    edges = pd.read_csv(output / "edges.csv")
    features = pd.read_csv(output / "features_observable18.csv")
    tweet_features = pd.read_csv(output / "tweet_features.csv")
    availability = pd.read_csv(output / "feature_availability.csv")
    assert dict(zip(labels["node_id"].astype(str), labels["is_bad"])) == {
        "1": 1, "2": 0
    }
    assert {"posts", "authored_by", "retweets", "retweeted_by",
            "comments", "commented_by", "mentions"} <= set(edges["edge_type"])
    assert features.filter(regex=r"^Semantic_").to_numpy().sum() == 0
    assert len(tweet_features) == 2
    assert tweet_features["log_char_count"].gt(0).all()
    assert not availability.filter(regex=r"^Semantic_").to_numpy().any()
    assert manifest["semantic_feature_status"].startswith("unavailable")
    assert manifest["label_semantics"] == "information_operation_membership"
    assert manifest["counts"]["context_tweets"] == 1


def test_rejects_user_overlap_between_bad_and_good_cohorts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "honduras-bad-anonymized.jsonl", [
        _row(tweet=10, user=1, good=0),
    ])
    _write(source / "honduras-good-anonymized.jsonl", [
        _row(tweet=11, user=1, good=1),
    ])
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": "hypertrace.honduras-uae-source-audit.v1",
        "files": [
            {"country": "honduras", "passed": True},
            {"country": "honduras", "passed": True},
        ],
    }), encoding="utf-8")

    try:
        materialize(source, tmp_path / "bundle", country="honduras", source_audit=audit)
    except ValueError as error:
        assert "both cohorts" in str(error)
    else:
        raise AssertionError("cohort overlap must be rejected")
