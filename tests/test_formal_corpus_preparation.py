import hashlib
import json

from scripts.prepare_formal_simulation_corpus import prepare_corpus


def test_prepare_corpus_builds_auditable_twenty_episode_plan(tmp_path):
    source = tmp_path / "source.json"
    profiles = [{
        "Profile Index": index + 1,
        "Demographic Information": {
            f"field_{value}": f"agent_{index}_value_{value}"
            for value in range(200)
        },
        "Summary": " ".join([f"persona{index}"] * 80),
    } for index in range(20)]
    source.write_text(json.dumps(profiles), encoding="utf-8")
    generation_report = tmp_path / "generation_report.json"
    generation_report.write_text(json.dumps({
        "schema_version": "hyperdecept.formal-dp-personas.v1",
        "generator": "test DeepPersona pipeline",
        "design": "independently_generated_fixed_population_reused_across_episode_seeds",
        "requested_count": 20,
        "total_valid_profiles": 20,
        "attribute_count": 200,
        "consolidated_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
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
        generation_report=generation_report,
        num_agents=20,
        min_attributes=200,
        parallel=64,
    )
    assert report["status"] == "passed"
    assert report["persona_populations"] == 1
    assert report["independent_profiles"] == 20
    assert report["csv_files"] == 20
    assert report["episodes"] == 20
    assert report["audit"]["csv_rows_checked"] == 400
    assert report["audit"]["endpoint_parallel_slots"] == [64]
