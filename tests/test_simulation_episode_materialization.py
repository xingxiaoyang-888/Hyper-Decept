import importlib.util
import json
from pathlib import Path
import sys
import sqlite3

import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_processing"
    / "materialize_simulation_episode.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "simulation_episode_materialization_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_future_trace_disables_next_action_supervision():
    module = _load_module()
    profiles = pd.DataFrame({"user_id": ["0", "1"]})
    targets, enabled = module._future_action_targets(
        profiles, None, None, input_is_cutoff_snapshot=False
    )
    assert not enabled
    assert list(targets.columns) == ["user_id"]


def test_future_action_is_first_observation_strictly_after_cutoff(tmp_path):
    module = _load_module()
    trace = tmp_path / "future.csv"
    pd.DataFrame({
        "user_id": ["0", "0", "0", "1"],
        "created_at": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:03Z",
        ],
        "action": ["at_cutoff", "reply", "post", "like"],
    }).to_csv(trace, index=False)
    targets, enabled = module._future_action_targets(
        pd.DataFrame({"user_id": ["0", "1", "2"]}),
        trace,
        "2026-01-01T00:00:00Z",
        input_is_cutoff_snapshot=True,
    )
    assert enabled
    assert targets.set_index("user_id")["next_action"].to_dict()["0"] == "reply"
    assert targets.set_index("user_id")["next_action"].to_dict()["1"] == "like"
    assert pd.isna(targets.set_index("user_id").loc["2", "next_action"])


def test_future_targets_require_explicit_cutoff_snapshot_assertion(tmp_path):
    module = _load_module()
    trace = tmp_path / "future.csv"
    pd.DataFrame({
        "user_id": ["0"],
        "created_at": ["2026-01-01T00:00:01Z"],
        "action": ["post"],
    }).to_csv(trace, index=False)
    with pytest.raises(ValueError, match="cutoff-snapshot"):
        module._future_action_targets(
            pd.DataFrame({"user_id": ["0"]}),
            trace,
            "2026-01-01T00:00:00Z",
            input_is_cutoff_snapshot=False,
        )


def test_formal_future_trace_must_be_strictly_after_cutoff_step(tmp_path):
    module = _load_module()
    trace = tmp_path / "future.csv"
    pd.DataFrame({
        "user_id": ["0", "1"],
        "created_at": [
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:02Z",
        ],
        "timestep": [18, 19],
        "action": ["post", "like"],
    }).to_csv(trace, index=False)
    with pytest.raises(ValueError, match="at or before cutoff_step"):
        module._future_action_targets(
            pd.DataFrame({"user_id": ["0", "1"]}),
            trace,
            "2026-01-01T00:00:00Z",
            18,
            input_is_cutoff_snapshot=True,
        )


def test_formal_future_target_includes_target_timestep(tmp_path):
    module = _load_module()
    trace = tmp_path / "future.csv"
    pd.DataFrame({
        "user_id": ["0", "0"],
        "created_at": [57, 60],
        "timestep": [19, 20],
        "action": ["post", "like"],
    }).to_csv(trace, index=False)
    targets, enabled = module._future_action_targets(
        pd.DataFrame({"user_id": ["0"]}),
        trace,
        None,
        18,
        input_is_cutoff_snapshot=True,
    )
    assert enabled
    assert targets.loc[0, "next_action"] == "post"
    assert targets.loc[0, "target_timestep"] == 19


def test_formal_activation_audit_requires_full_pre_cutoff_coverage(tmp_path):
    module = _load_module()
    audit = tmp_path / "activation.json"
    audit.write_text(json.dumps({
        "schema_version": "hyperdecept.activation-audit.v1",
        "policy": "budgeted_activity",
        "num_agents": 2,
        "time_steps": 2,
        "cutoff_step": 1,
        "steps": [
            {
                "timestep": 1,
                "budget": 1,
                "selected_count": 1,
                "selected": [{"agent_id": 0, "reason": "activity_budget"}],
            },
            {
                "timestep": 2,
                "budget": 1,
                "selected_count": 1,
                "selected": [{"agent_id": 1, "reason": "activity_budget"}],
            },
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="misses 1 agents"):
        module._validate_activation_audit(
            audit,
            num_agents=2,
            time_steps=2,
            cutoff_step=1,
        )


def test_cutoff_snapshot_rejects_future_trace_rows(tmp_path):
    module = _load_module()
    db = tmp_path / "cutoff.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE trace (user_id INTEGER, created_at TEXT, action TEXT, info TEXT)"
        )
        connection.execute(
            "INSERT INTO trace VALUES (1, '57', 'post', '{}')"
        )
    with pytest.raises(ValueError, match="after cutoff_step"):
        module._validate_cutoff_snapshot(db, 18)
