import json

from scripts.prepare_formal_simulation_corpus import prepare_corpus


def test_prepare_corpus_builds_auditable_twenty_episode_plan(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "Profile_R1_A3_Count_200": {
            "Profile Index": 1,
            "Summary": "first persona",
        },
        "Profile_R2_A4_Count_250": {
            "Profile Index": 2,
            "Summary": "second persona",
        },
    }), encoding="utf-8")
    tweet_pool = tmp_path / "tweet_pool.json"
    tweet_pool.write_text(json.dumps({
        "good": ["audited organic seed"],
        "bad": ["audited adversarial seed"],
    }), encoding="utf-8")
    report = prepare_corpus(
        output_root=tmp_path / "formal",
        tweet_pool_path=tweet_pool,
        source=source,
        num_agents=20,
        min_attributes=200,
        parallel=64,
    )
    assert report["status"] == "passed"
    assert report["persona_populations"] == 4
    assert report["csv_files"] == 20
    assert report["episodes"] == 20
    assert report["audit"]["csv_rows_checked"] == 400
    assert report["audit"]["endpoint_parallel_slots"] == [64]
