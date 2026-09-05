import json

from scripts.build_framework_native_tweet_pool import build_pool


def test_native_pool_preserves_order_and_deduplicates(tmp_path):
    good_a = tmp_path / "real_covid.json"
    good_b = tmp_path / "real_politics.json"
    bad = tmp_path / "fake.json"
    good_a.write_text(json.dumps([" alpha ", "beta", "alpha"]), encoding="utf-8")
    good_b.write_text(json.dumps(["beta", "gamma"]), encoding="utf-8")
    bad.write_text(json.dumps(["claim one", "claim one", "claim two"]), encoding="utf-8")
    report = build_pool(
        good=[good_a, good_b],
        bad=[bad],
        output=tmp_path / "native_pool.json",
    )
    payload = json.loads((tmp_path / "native_pool.json").read_text(encoding="utf-8"))
    assert payload == {"good": ["alpha", "beta", "gamma"], "bad": ["claim one", "claim two"]}
    assert report["counts"] == {"good": 3, "bad": 2}
    assert len(report["source_files"]) == 3


def test_native_pool_rejects_non_list_source(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"text": "not a list"}), encoding="utf-8")
    try:
        build_pool(
            good=[source],
            bad=[source],
            output=tmp_path / "pool.json",
        )
    except TypeError as error:
        assert "JSON list" in str(error)
    else:
        raise AssertionError("non-list source should be rejected")
