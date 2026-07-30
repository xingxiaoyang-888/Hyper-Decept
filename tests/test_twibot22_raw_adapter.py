import json
import importlib.util
import sys

import numpy as np

from data_processing.twibot22_raw_adapter import (
    TwiBot22RawAdapter,
    load_materialized_bundle,
)


def _load_graph_builder():
    path = __import__("pathlib").Path(__file__).resolve().parents[1] / "Character Classification" / "graph_builder.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("raw_graph_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_raw_fixture(root):
    (root / "label.csv").write_text(
        "id,label\nu1,human\nu2,bot\n", encoding="utf-8"
    )
    (root / "split.csv").write_text(
        "id,split\nu1,train\nu2,val\n", encoding="utf-8"
    )
    (root / "edge.csv").write_text(
        "source_id,relation,target_id\n"
        "u1,following,u3\n"
        "u3,followers,u1\n"
        "u1,post,100\n"
        "u1,retweeted,200\n"
        "u2,mentioned,u1\n",
        encoding="utf-8",
    )
    (root / "user.json").write_text(json.dumps([
        {"id": 1, "description": "core bio", "username": "core",
         "created_at": "2020-01-01T00:00:00Z",
         "public_metrics": {"followers_count": 4, "following_count": 2}},
        {"id": 2, "description": "second core bio", "username": "core2",
         "created_at": "2020-01-01T00:00:00Z",
         "public_metrics": {"followers_count": 6, "following_count": 1}},
        {"id": 3, "description": "boundary bio", "username": "boundary",
         "created_at": "2020-01-02T00:00:00Z",
         "public_metrics": {"followers_count": 5, "following_count": 3}},
    ]), encoding="utf-8")
    (root / "tweet_0.json").write_text(json.dumps([
        {"id": 100, "author_id": 1, "text": "first post",
         "created_at": "2024-01-01T00:00:00Z",
         "public_metrics": {"like_count": 3},
         "referenced_tweets": None},
        {"id": 200, "author_id": 2, "text": "retweeted content",
         "created_at": "2024-01-02T00:00:00Z",
         "public_metrics": {"retweet_count": 1},
         "referenced_tweets": [{"type": "retweeted", "id": 100}]},
    ]), encoding="utf-8")


def test_raw_adapter_preserves_stable_ids_time_and_relation_evidence(tmp_path):
    _write_raw_fixture(tmp_path)
    bundle = TwiBot22RawAdapter(str(tmp_path), ["u1", "u2"]).load()
    assert bundle.dataset_kind == "twibot22_raw"
    assert bundle.capabilities.temporal is True
    assert bundle.capabilities.stable_post_ids is True
    assert set(bundle.core_users["user_id"]) == {"u1", "u2"}
    assert set(bundle.boundary_users["user_id"]) == {"u3"}
    assert set(bundle.labels["data_split"]) == {"train", "val"}

    assert set(bundle.follow_edges["follower_id"]) == {"u1", "u3"}
    assert set(bundle.follow_edges["followee_id"]) == {"u3", "u1"}
    assert set(bundle.relations["relation"]) == {
        "following", "followers", "post", "retweeted", "mentioned"
    }
    assert len(bundle.posts) == 2
    post = bundle.posts.set_index("post_id").loc["100"]
    assert post["created_at"] == "2024-01-01T00:00:00Z"
    assert post["author_id"] == "u1"
    assert bundle.actions.iloc[0]["evidence_id"].startswith("edge:")
    assert bundle.actions["event_time"].notna().any()


def test_raw_adapter_caps_actions_without_changing_core_labels(tmp_path):
    _write_raw_fixture(tmp_path)
    bundle = TwiBot22RawAdapter(
        str(tmp_path), ["u1", "u2"], max_actions_per_user=1
    ).load()
    assert len(bundle.actions[bundle.actions["source_id"] == "u1"]) == 1
    assert set(bundle.labels["user_id"]) == {"u1", "u2"}


def test_graph_builder_accepts_raw_directory_and_keeps_temporal_tweet_metadata(tmp_path):
    _write_raw_fixture(tmp_path)
    graph_builder = _load_graph_builder()
    data, reverse = graph_builder.build_hetero_data(
        ["u1", "u2"],
        np.ones((2, 2), dtype=np.float32),
        str(tmp_path),
        threshold=1.1,
    )
    assert reverse == {0: "u1", 1: "u2"}
    assert data.dataset_kind == "twibot22_raw"
    assert data.dataset_capabilities["temporal"] is True
    assert data["tweet"].temporal is True
    assert set(data["tweet"].post_ids) == {"100", "200"}
    assert any(value is not None for value in data["tweet"].created_at)
    assert ("user", "following", "boundary_user") in data.edge_types


def test_materialized_bundle_avoids_raw_rescan_and_builds_same_graph(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw_fixture(raw)
    original = TwiBot22RawAdapter(str(raw), ["u1", "u2"]).load()
    exported = tmp_path / "exported"
    exported.mkdir()
    for name, frame in (
        ("core_users", original.core_users),
        ("boundary_users", original.boundary_users),
        ("labels", original.labels),
        ("follow_edges", original.follow_edges),
        ("actions", original.actions),
        ("relations", original.relations),
        ("posts", original.posts),
    ):
        frame.to_csv(exported / f"{name}.csv", index=False)
    (exported / "adapter_manifest.json").write_text(
        json.dumps(original.manifest()), encoding="utf-8"
    )

    materialized = load_materialized_bundle(exported)
    assert set(materialized.core_users["user_id"]) == {"u1", "u2"}
    assert materialized.follow_edges.iloc[0]["evidence_ids"]
    graph_builder = _load_graph_builder()
    data, reverse = graph_builder.build_hetero_data(
        ["u1", "u2"],
        np.ones((2, 2), dtype=np.float32),
        str(raw),
        threshold=1.1,
        twibot_bundle=materialized,
    )
    assert reverse == {0: "u1", 1: "u2"}
    assert data.dataset_kind == "twibot22_raw"
    assert set(data["tweet"].post_ids) == {"100", "200"}
    assert ("user", "following", "boundary_user") in data.edge_types
