import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.audit_formal_simulation_inputs import audit_plan


def test_audit_accepts_aligned_population(tmp_path, monkeypatch):
    personas = tmp_path / "personas"
    personas.mkdir()
    chunks = personas / "formal_dp_personas.chunks.json"
    chunks.write_text(json.dumps([
        {"agent_id": 0, "section": "summary", "text": "zero"},
        {"agent_id": 1, "section": "summary", "text": "one"},
    ]), encoding="utf-8")
    profiles = personas / "formal_dp_personas.profiles.json"
    profiles.write_text(json.dumps([{"Profile Index": 1}, {"Profile Index": 2}]), encoding="utf-8")
    chunks_sha256 = hashlib.sha256(chunks.read_bytes()).hexdigest()
    profiles_sha256 = hashlib.sha256(profiles.read_bytes()).hexdigest()
    manifest = personas / "formal_dp_personas.personas.manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "hyperdecept.formal-dp-personas.v1",
        "artifact_kind": "shared_population_package",
        "design": "independently_generated_fixed_population_reused_across_episode_seeds",
        "agents": 2,
        "attribute_count": 200,
        "unique_content_hashes": 2,
        "profiles_file": profiles.name,
        "chunks_file": chunks.name,
        "profiles_sha256": profiles_sha256,
        "chunks_sha256": chunks_sha256,
        "records": [{"agent_id": 0}, {"agent_id": 1}],
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
            "deep_persona_manifest_path": str(manifest),
            "deep_persona_population_sha256": chunks_sha256,
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
        "shared_population": {"chunks_sha256": chunks_sha256},
    }), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    report = audit_plan(plan_path)
    assert report["status"] == "passed"
    assert report["csv_rows_checked"] == 2
    assert report["endpoint_parallel_slots"] == [64]
    assert report["api_key_present"] is True
    assert report["persona_populations"] == 1
