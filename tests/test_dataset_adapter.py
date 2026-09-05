import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from data_processing.dataset_adapter import DatasetCapabilities, UnifiedDatasetBundle


GRAPH_BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "Character Classification"
    / "graph_builder.py"
)


def _load_graph_builder():
    sys.path.insert(0, str(GRAPH_BUILDER_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "graph_builder_legacy_contract_test", GRAPH_BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shared_bundle_contract_preserves_capabilities_and_counts():
    capabilities = DatasetCapabilities(
        temporal=True,
        stable_post_ids=True,
        external_neighbors=True,
        ground_truth_roles=False,
        raw_text=True,
        interaction_target_ids=True,
        supervised_labels=True,
    )
    bundle = UnifiedDatasetBundle(
        dataset_kind="twibot22_raw",
        capabilities=capabilities,
        core_users=np.array([], dtype=object),
        boundary_users=np.array([], dtype=object),
        labels=np.array([], dtype=object),
        follow_edges=np.array([], dtype=object),
        actions=np.array([], dtype=object),
    )
    manifest = bundle.manifest()
    assert manifest["dataset_kind"] == "twibot22_raw"
    assert manifest["capabilities"]["temporal"] is True
    assert manifest["counts"] == {
        "core_users": 0,
        "boundary_users": 0,
        "follow_edges": 0,
        "actions": 0,
    }


def test_legacy_twibot_v5_database_is_rejected(tmp_path):
    db_path = tmp_path / "twibot_1000_v5.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE user (user_id TEXT)")
        connection.execute("CREATE TABLE follow (follower_id TEXT, followee_id TEXT)")
        connection.execute(
            "CREATE TABLE agent_actions (agent_name TEXT, action_type TEXT, content TEXT)"
        )

    graph_builder = _load_graph_builder()
    with pytest.raises(ValueError, match="Legacy TwiBot V5"):
        graph_builder.build_hetero_data(
            ["u1"], np.ones((1, 2), dtype=np.float32), str(db_path), threshold=1.1
        )
