import pandas as pd
import pytest
import torch

from data_processing.mgtab_adapter import MGTABAdapter, write_split_csv


def _write_bundle(root, *, invalid_endpoint=False):
    node_count = 60
    features = torch.full((node_count, 788), 0.25, dtype=torch.float32)
    features[:, :20] = 0.5
    features[0, 20:] = 0.0

    bot_labels = []
    stance_labels = []
    for bot in (0, 1):
        for stance in (0, 1, 2):
            bot_labels.extend([bot] * 10)
            stance_labels.extend([stance] * 10)

    sources = [
        0,       # followers
        1,       # friends
        2, 2,    # mention: one self-loop and one ordinary edge
        3, 3, 3, 5, 5,  # reply: three duplicates and two self-loop rows
        6, 6, 7,         # quoted: two duplicates and one self-loop row
        8,       # url
        10,      # hashtag
    ]
    targets = [
        1,
        2,
        2, 3,
        4, 4, 4, 5, 5,
        7, 7, 7,
        9,
        11,
    ]
    relation_types = [
        0,
        1,
        2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4,
        5,
        6,
    ]
    weights = [1.0] * 12 + [0.4, 0.6]
    edge_index = torch.tensor([sources, targets], dtype=torch.int64)
    if invalid_endpoint:
        edge_index[1, 0] = node_count

    torch.save(edge_index, root / "edge_index.pt")
    torch.save(torch.tensor(relation_types), root / "edge_type.pt")
    torch.save(torch.tensor(weights, dtype=torch.float32), root / "edge_weight.pt")
    torch.save(features, root / "features.pt")
    torch.save(torch.tensor(bot_labels), root / "labels_bot.pt")
    torch.save(torch.tensor(stance_labels), root / "labels_stance.pt")


def _edge_pairs(store):
    return list(map(tuple, store.edge_index.T.tolist()))


def test_adapter_builds_native_graph_labels_and_manifest(tmp_path):
    _write_bundle(tmp_path)
    graph, labels, manifest = MGTABAdapter(tmp_path, split_seed=42).load()

    assert graph["user"].x.shape == (60, 789)
    assert graph["user"].node_ids[:2] == ["mgtab:0", "mgtab:1"]
    assert graph["user"].tweet_embedding_available.tolist() == [False] + [True] * 59
    assert graph.dataset_name == "mgtab"
    assert set(graph.edge_types) == {
        ("user", relation, "user")
        for relation in (
            "followers", "friends", "mention", "reply", "quoted", "url", "hashtag"
        )
    }

    assert list(labels.columns) == ["user_id", "is_bad", "stance", "data_split"]
    assert labels["user_id"].is_unique
    assert set(labels["data_split"]) == {"train", "validation", "test"}
    assert labels["data_split"].value_counts().to_dict() == {
        "train": 42,
        "validation": 12,
        "test": 6,
    }

    reply = graph[("user", "reply", "user")]
    assert _edge_pairs(reply) == [(3, 4)]
    assert reply.multiplicity.tolist() == [3.0]
    assert reply.base_weight.tolist() == [1.0]
    assert not reply.temporal_available.any()

    quoted = graph[("user", "quoted", "user")]
    assert _edge_pairs(quoted) == [(6, 7)]
    assert quoted.multiplicity.tolist() == [2.0]

    url = graph[("user", "url", "user")]
    assert _edge_pairs(url) == [(8, 9), (9, 8)]
    assert torch.allclose(url.base_weight, torch.tensor([0.4, 0.4]))
    assert url.is_synthetic_reverse.tolist() == [False, True]

    self_counts = graph["user"].self_interaction_counts
    assert self_counts.shape == (60, 7)
    assert self_counts[2, 2] == 1
    assert self_counts[5, 3] == 2
    assert self_counts[7, 4] == 1

    assert manifest["dataset_name"] == "mgtab"
    assert manifest["release"] == "standard"
    assert manifest["feature_dim_original"] == 788
    assert manifest["feature_dim_model"] == 789
    assert manifest["tweet_embedding_missing_count"] == 1
    assert manifest["relations"]["reply"]["removed_self_loop_rows"] == 2
    assert manifest["relations"]["reply"]["multiplicity_sum"] == 3
    assert len(manifest["split_hash"]) == 64


def test_adapter_split_is_reproducible_and_can_be_persisted(tmp_path):
    _write_bundle(tmp_path)
    _, first, first_manifest = MGTABAdapter(tmp_path, split_seed=42).load()
    _, second, second_manifest = MGTABAdapter(tmp_path, split_seed=42).load()

    pd.testing.assert_frame_equal(first, second)
    assert first_manifest["split_hash"] == second_manifest["split_hash"]

    split_path = write_split_csv(first, tmp_path / "derived" / "split.csv")
    persisted = pd.read_csv(split_path)
    pd.testing.assert_frame_equal(
        persisted,
        first[["user_id", "data_split"]],
    )


def test_preserve_multiedges_is_available_for_ablation(tmp_path):
    _write_bundle(tmp_path)
    graph, _, manifest = MGTABAdapter(
        tmp_path, multiedge_policy="preserve_multiedges"
    ).load()

    reply = graph[("user", "reply", "user")]
    assert _edge_pairs(reply) == [(3, 4), (3, 4), (3, 4)]
    assert reply.multiplicity.tolist() == [1.0, 1.0, 1.0]
    assert manifest["multiedge_policy"] == "preserve_multiedges"


def test_adapter_rejects_invalid_edge_endpoints(tmp_path):
    _write_bundle(tmp_path, invalid_endpoint=True)
    with pytest.raises(ValueError, match="out-of-range"):
        MGTABAdapter(tmp_path).load()
