import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.audit_formal_simulation_inputs import audit_plan


def test_audit_accepts_aligned_population(tmp_path, monkeypatch):
    personas = tmp_path / "personas"
    personas.mkdir()
    chunks = personas / "seed_11.chunks.json"
    chunks.write_text(json.dumps([
        {"agent_id": 0, "section": "summary", "text": "zero"},
        {"agent_id": 1, "section": "summary", "text": "one"},
    ]), encoding="utf-8")
    chunks.with_name("seed_11.personas.manifest.json").write_text(json.dumps({
        "schema_version": "hyperdecept.persona-population.v1",
        "agents": 2,
        "prototype_count": 2,
        "min_attributes": 200,
        "assignments": [{"agent_id": 0}, {"agent_id": 1}],
    }), encoding="utf-8")
    csv_path = tmp_path / "agents.csv"
    pd.DataFrame({
        "user_id": [0, 1],
        "user_char": ["zero", "one"],
        "scenario_id": ["leader_amplifier"] * 2,
        "previous_tweets": ["['seed zero']", "['seed one']"],
    }).to_csv(csv_path, index=False)
    config_path = tmp_path / "episode.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {"csv_path": str(csv_path), "db_path": str(tmp_path / "run.db")},
        "simulation": {
            "scenario_id": "leader_amplifier",
            "force_all_agents_active": False,
            "deep_persona_chunks_path": str(chunks),
        },
        "model": {"num_agents": 2, "cfgs": [{"num": 2}]},
        "inference": {
            "parallel_per_endpoint": 1,
            "max_tokens": 512,
            "server_url": [{
                "base_url": "https://api.deepseek.com/v1",
                "parallel": 64,
            }],
        },
    }), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": "hyperdecept.formal-simulation-plan.v1",
        "episodes": 1,
        "num_agents": 2,
        "configs": [str(config_path)],
    }), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    report = audit_plan(plan_path)
    assert report["status"] == "passed"
    assert report["csv_rows_checked"] == 2
    assert report["endpoint_parallel_slots"] == [64]
    assert report["api_key_present"] is True
