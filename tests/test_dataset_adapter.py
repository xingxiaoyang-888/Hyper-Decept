import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_processing.dataset_adapter import (
    TwiBotStaticAdapter,
    detect_dataset_kind,
)


GRAPH_BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "Character Classification"
    / "graph_builder.py"
)


def _load_graph_builder():
    sys.path.insert(0, str(GRAPH_BUILDER_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "graph_builder_adapter_test", GRAPH_BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_twibot_pair(tmp_path):
    csv_path = tmp_path / "twibot_multimodal.csv"
    pd.DataFrame({
        "user_id": ["core_good", "core_bad"],
        "user_char": ["public bio", None],
        "followers_count": [10, 20],
        "following_count": [3, 4],
        "previous_tweets": ["hello", "automated text"],
        "user_type": ["good", "bad"],
    }).to_csv(csv_path, index=False)
    db_path = tmp_path / "twibot.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE user (user_id TEXT, user_type TEXT, "
            "followers INTEGER, following INTEGER)"
        )
        connection.execute(
            "CREATE TABLE follow (follower_id TEXT, followee_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE agent_actions "
            "(agent_name TEXT, action_type TEXT, content TEXT)"
        )
        connection.executemany(
            "INSERT INTO user VALUES (?, ?, ?, ?)",
            [
                ("core_good", "good", 10, 3),
                ("core_bad", "bad", 20, 4),
                ("boundary_1", "boundary", 0, 0),
            ],
        )
        connection.executemany(
            "INSERT INTO follow VALUES (?, ?)",
            [
                ("core_good", "boundary_1"),
                ("core_good", "boundary_1"),
                ("boundary_1", "core_bad"),
            ],
        )
        connection.executemany(
            "INSERT INTO agent_actions VALUES (?, ?, ?)",
            [
                ("core_good", "post", "original public post"),
                ("core_bad", "like", "external liked post"),
            ],
        )
    return csv_path, db_path


def test_twibot_adapter_separates_public_features_and_labels(tmp_path):
    csv_path, db_path = _make_twibot_pair(tmp_path)
    bundle = TwiBotStaticAdapter(str(db_path), str(csv_path)).load()
    assert bundle.dataset_kind == "twibot_static_v5"
    assert bundle.capabilities.temporal is False
    assert bundle.capabilities.stable_post_ids is False
    assert bundle.capabilities.external_neighbors is True
    assert set(bundle.core_users["user_id"]) == {"core_good", "core_bad"}
    assert set(bundle.boundary_users["user_id"]) == {"boundary_1"}
    assert "user_type" not in bundle.core_users.columns
    assert bundle.core_users.set_index("user_id").loc["core_good", "bio"] == "public bio"
    assert set(bundle.labels.columns) == {"user_id", "user_type", "is_bad"}


def test_twibot_adapter_preserves_multiplicity_and_marks_derived_ids(tmp_path):
    csv_path, db_path = _make_twibot_pair(tmp_path)
    bundle = TwiBotStaticAdapter(str(db_path), str(csv_path)).load()
    duplicated = bundle.follow_edges[
        (bundle.follow_edges["follower_id"] == "core_good")
        & (bundle.follow_edges["followee_id"] == "boundary_1")
    ].iloc[0]
    assert duplicated["multiplicity"] == 2
    assert len(duplicated["evidence_ids"]) == 2
    assert bundle.actions["event_time"].isna().all()
    post = bundle.actions[bundle.actions["action_type"] == "post"].iloc[0]
    liked = bundle.actions[bundle.actions["action_type"] == "like"].iloc[0]
    assert post["content_node_id"].startswith("twibot_action:")
    assert post["id_kind"] == "derived_action_rowid"
    assert liked["content_node_id"].startswith("external_text:")
    assert "without_original_post_id" in liked["id_kind"]


def test_graph_builder_keeps_boundary_topology_and_static_text(tmp_path):
    _, db_path = _make_twibot_pair(tmp_path)
    graph_builder = _load_graph_builder()
    data, reverse_users = graph_builder.build_hetero_data(
        ["core_good", "core_bad"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        str(db_path),
        threshold=1.1,
    )
    assert reverse_users == {0: "core_good", 1: "core_bad"}
    assert data["boundary_user"].num_nodes == 1
    assert data["tweet"].num_nodes == 2
    assert ("user", "follows", "boundary_user") in data.edge_types
    assert ("boundary_user", "followed_by", "user") in data.edge_types
    assert ("boundary_user", "follows", "user") in data.edge_types
    assert ("user", "posts", "tweet") in data.edge_types
    assert ("user", "likes", "tweet") in data.edge_types
    relation = data["user", "follows", "boundary_user"]
    assert relation.edge_index.shape[1] == 1
    assert torch.equal(relation.multiplicity, torch.tensor([2.0]))
    assert len(relation.evidence_ids[0]) == 2
    assert data.dataset_capabilities["temporal"] is False
    assert data["tweet"].temporal is False


def test_dataset_kind_detection_distinguishes_simulation_db(tmp_path):
    _, twibot_db = _make_twibot_pair(tmp_path)
    assert detect_dataset_kind(str(twibot_db)) == "twibot_static_v5"
    simulation_db = tmp_path / "simulation.db"
    with sqlite3.connect(simulation_db) as connection:
        connection.execute("CREATE TABLE user (user_id INTEGER)")
        connection.execute("CREATE TABLE post (post_id INTEGER, user_id INTEGER)")
    assert detect_dataset_kind(str(simulation_db)) == "simulation_event_db"
