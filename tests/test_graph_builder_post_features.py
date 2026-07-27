import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Character Classification"
    / "graph_builder.py"
)


def _load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("graph_builder_post_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_db(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE post (
                post_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                original_post_id INTEGER,
                content TEXT,
                quote_content TEXT,
                created_at TEXT,
                num_likes INTEGER,
                num_dislikes INTEGER,
                num_shares INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO post VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (10, 1, None, "Hello #topic https://example.org", None,
                 "2026-01-01T06:00:00Z", 3, 0, 1),
                (11, 2, 10, "RESHARE @user", "quoted text",
                 "2026-01-01T18:00:00Z", 5, 1, 2),
            ],
        )


def test_post_node_features_are_deterministic_and_observed(tmp_path):
    module = _load_module()
    db_path = tmp_path / "events.db"
    _make_db(db_path)
    users = ["1", "2"]
    user_features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    first, _ = module.build_hetero_data(users, user_features, str(db_path))
    second, _ = module.build_hetero_data(users, user_features, str(db_path))
    assert torch.equal(first["tweet"].x, second["tweet"].x)
    assert first["tweet"].x.shape == (2, len(module.POST_OBSERVED_FEATURES))
    assert first["tweet"].feature_source == "observed_post_features"
    assert torch.isfinite(first["tweet"].x).all()
    assert ("user", "posts", "tweet") in first.edge_types
    assert ("tweet", "authored_by", "user") in first.edge_types
    forward = first["user", "posts", "tweet"].edge_index
    reverse = first["tweet", "authored_by", "user"].edge_index
    assert torch.equal(forward.flip(0), reverse)


def test_external_semantic_embeddings_append_by_real_post_id(tmp_path):
    module = _load_module()
    db_path = tmp_path / "events.db"
    _make_db(db_path)
    embeddings = {
        "10": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "11": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
    }
    data, _ = module.build_hetero_data(
        ["1", "2"],
        np.eye(2, dtype=np.float32),
        str(db_path),
        post_embeddings=embeddings,
    )
    assert data["tweet"].x.shape[1] == len(module.POST_OBSERVED_FEATURES) + 3
    assert torch.allclose(
        data["tweet"].x[:, -3:],
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )
    assert "external_semantic_embeddings" in data["tweet"].feature_source


def test_embedding_csv_loader_rejects_labels_and_loads_numeric_columns(tmp_path):
    module = _load_module()
    path = tmp_path / "post_embeddings.csv"
    pd.DataFrame({
        "post_id": [10, 11],
        "embedding_0": [0.1, 0.2],
        "embedding_1": [0.3, 0.4],
    }).to_csv(path, index=False)
    loaded = module.load_post_embeddings(str(path))
    assert set(loaded) == {"10", "11"}
    assert np.allclose(loaded["10"], [0.1, 0.3])
