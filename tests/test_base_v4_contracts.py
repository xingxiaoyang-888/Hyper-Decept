import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import HeteroData

from data_processing.feature_contracts import (
    OBSERVABLE_17_NO_TEMPORAL,
    OBSERVABLE_18,
    resolve_feature_contract,
)
from scripts.base_models import DomainAwareEuclideanHGT, DomainAwareMLP
from scripts.train_base import _apply_graph_intervention, _build_model


def _batch():
    graph = HeteroData()
    graph["user"].x = torch.randn(5, 18)
    graph["user"].node_ids = [f"u{index}" for index in range(5)]
    graph["tweet"].x = torch.randn(3, 4)
    graph["user", "posts", "tweet"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 0, 1]], dtype=torch.long
    )
    graph["tweet", "authored_by", "user"].edge_index = torch.tensor(
        [[0, 1, 2, 0, 1], [0, 1, 2, 3, 4]], dtype=torch.long
    )
    graph["user", "posts", "tweet"].base_weight = torch.arange(5, dtype=torch.float32)
    graph["user", "posts", "tweet"].temporal_sync = torch.arange(5, dtype=torch.float32) + 10
    graph["user", "posts", "tweet"].temporal_recency = torch.arange(5, dtype=torch.float32) + 20
    graph["user", "posts", "tweet"].temporal_available = torch.ones(5, dtype=torch.float32)
    return SimpleNamespace(episode_id="sim:test:s11", graph=graph)


def _metadata():
    return _batch().graph.metadata()


def test_observable_contract_excludes_all_psychology_proxies():
    assert len(OBSERVABLE_18) == 18
    assert len(OBSERVABLE_17_NO_TEMPORAL) == 17
    assert "Temporal_Entropy" not in OBSERVABLE_17_NO_TEMPORAL
    assert set(OBSERVABLE_18) - set(OBSERVABLE_17_NO_TEMPORAL) == {"Temporal_Entropy"}
    assert resolve_feature_contract("observable18") == OBSERVABLE_18
    assert resolve_feature_contract("observable17_no_temporal") == OBSERVABLE_17_NO_TEMPORAL
    for excluded in ("full26", "psychology_proxy8", "semantic8", "behavior10"):
        with pytest.raises(ValueError, match="unknown feature contract"):
            resolve_feature_contract(excluded)
    with pytest.raises(ValueError, match="unknown feature contract"):
        resolve_feature_contract("observable_18")


def test_degree_preserving_shuffle_is_deterministic_and_preserves_marginals():
    first, second = _batch(), _batch()
    original = {
        edge_type: store.edge_index.clone()
        for edge_type, store in first.graph.edge_items()
    }
    _apply_graph_intervention(first, "degree_preserving_edge_shuffle", 7)
    _apply_graph_intervention(second, "degree_preserving_edge_shuffle", 7)
    for edge_type, store in first.graph.edge_items():
        before, after = original[edge_type], store.edge_index
        assert torch.equal(before[0].sort().values, after[0].sort().values)
        assert torch.equal(before[1].sort().values, after[1].sort().values)
        assert torch.equal(after, second.graph[edge_type].edge_index)
    assert torch.equal(
        first.graph["user", "posts", "tweet"].base_weight.sort().values,
        torch.arange(5, dtype=torch.float32).sort().values,
    )


def test_remove_temporal_information_preserves_topology_and_non_temporal_edges():
    batch = _batch()
    edge_type = ("user", "posts", "tweet")
    original_edges = batch.graph[edge_type].edge_index.clone()
    original_weights = batch.graph[edge_type].base_weight.clone()

    _apply_graph_intervention(batch, "remove_temporal_information", 7)

    assert torch.equal(batch.graph[edge_type].edge_index, original_edges)
    assert torch.equal(batch.graph[edge_type].base_weight, original_weights)
    for name in ("temporal_sync", "temporal_recency", "temporal_available"):
        assert torch.count_nonzero(batch.graph[edge_type][name]) == 0


def test_baseline_models_share_detector_output_contract():
    graph = _batch().graph
    for model in (
        DomainAwareMLP(
            hidden_dim=8, num_layers=1, metadata=graph.metadata(),
            dataset_domains=("deeppersona_oasis",), dropout=0.0,
        ),
        DomainAwareEuclideanHGT(
            hidden_dim=8, num_heads=2, num_layers=1, metadata=graph.metadata(),
            dataset_domains=("deeppersona_oasis",), dropout=0.0,
        ),
    ):
        output = model(graph, domain="synthetic", dataset_name="deeppersona_oasis")
        assert output["coordination_logits"].shape == (5,)
        assert output["coordination_class_logits"].shape == (5, 2)
        assert output["user_tangent"].shape == (5, 8)
        assert model.enable_privileged_heads is False


def test_model_factory_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown model variant"):
        _build_model("not-a-model", _metadata())
