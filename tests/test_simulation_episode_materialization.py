import importlib.util
from pathlib import Path
import sys

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
