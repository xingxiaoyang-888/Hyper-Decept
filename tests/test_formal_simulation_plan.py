from pathlib import Path

import yaml

from scripts.generate_formal_simulation_plan import build_configs


def test_formal_plan_has_twenty_bounded_configs(tmp_path):
    report = build_configs(
        output_dir=tmp_path / "configs",
        csv_root=tmp_path / "csv",
        db_root=tmp_path / "db",
        persona_root=tmp_path / "personas",
        tweet_pool_path=tmp_path / "tweet_pool.json",
        model_type="deepseek-chat",
        endpoints=[{"host": "127.0.0.1", "ports": [8000], "parallel": 4}],
    )
    assert report["episodes"] == 20
    assert report["corpus_activation_ceiling"] == 90_000
    assert report["corpus_request_ceiling_no_retries"] == 90_000
    assert report["corpus_request_ceiling_with_retries"] == 180_000
    first = yaml.safe_load(Path(report["configs"][0]).read_text(encoding="utf-8"))
    simulation = first["simulation"]
    assert simulation["num_timesteps"] == 30
    assert simulation["cutoff_step"] == 18
    assert simulation["activation_policy"] == "budgeted_activity"
    assert simulation["scenario_id"] == "leader_amplifier"
    assert simulation["force_all_agents_active"] is False
    chunks_path = Path(simulation["deep_persona_chunks_path"])
    assert chunks_path.parent.name == "personas"
    assert chunks_path.name == "seed_11.chunks.json"
    assert simulation["coverage_deadlines"] == [18]
    assert simulation["max_silent_steps"] == 14
    assert simulation["export_debug_artifacts"] is False
    assert simulation["export_visualizations"] is False
    assert first["inference"]["max_tokens"] == 512
    assert first["inference"]["dispatch_interval"] == 0.05
    assert first["inference"]["completion_poll_interval"] == 0.05
    assert first["model"]["cfgs"][0]["is_openai_model"] is False
    assert "seed_11.profiles.json" in report["population_commands"][0]["command"]
    assert "--tweet-pool" in report["population_commands"][0]["command"]
