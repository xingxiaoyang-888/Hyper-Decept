"""Non-hyperbolic detector baselines with the HyperTrace model interface."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from coordination_contract import CoordinationCheckpointMixin


EdgeType = Tuple[str, str, str]


class _DatasetFeatureAdapters(torch.nn.Module):
    def __init__(self, dataset_domains, node_types, hidden_dim: int) -> None:
        super().__init__()
        from torch_geometric.nn import Linear

        self.layers = torch.nn.ModuleDict({
            dataset: torch.nn.ModuleDict({
                node_type: torch.nn.Sequential(
                    Linear(-1, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                )
                for node_type in node_types
            })
            for dataset in dataset_domains
        })

    def forward(self, dataset: str, x_dict):
        if dataset not in self.layers:
            raise ValueError(f"model was not configured for dataset {dataset!r}")
        return {
            node_type: self.layers[dataset][node_type](features.float())
            for node_type, features in x_dict.items()
        }


class DomainAwareEuclideanHGT(CoordinationCheckpointMixin, torch.nn.Module):
    """Standard Euclidean HGT baseline with the same input adapter budget."""

    geometry_backend = "euclidean_hgt"
    enable_privileged_heads = False

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        metadata,
        dataset_domains: Sequence[str],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import HGTConv

        node_types = tuple(metadata[0])
        self.dataset_domains = tuple(dataset_domains)
        self.feature_adapters = _DatasetFeatureAdapters(
            self.dataset_domains, node_types, hidden_dim
        ).layers
        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, num_heads)
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                node_type: torch.nn.LayerNorm(hidden_dim)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])
        self.dropout = float(dropout)
        self.coordination_head = torch.nn.Linear(hidden_dim, 2)

    def forward(
        self,
        graph,
        *,
        domain: str,
        dataset_name: Optional[str] = None,
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
    ):
        del domain
        if edge_mask_dict is not None:
            raise ValueError("standard Euclidean HGT baseline does not accept edge masks")
        dataset = dataset_name or "synthetic"
        if dataset not in self.feature_adapters:
            raise ValueError(f"model was not configured for dataset {dataset!r}")
        hidden = {
            node_type: self.feature_adapters[dataset][node_type](features.float())
            for node_type, features in graph.x_dict.items()
        }
        for layer_index, conv in enumerate(self.convs):
            updated = conv(hidden, graph.edge_index_dict)
            hidden = {
                node_type: F.dropout(
                    F.gelu(self.norms[layer_index][node_type](
                        updated.get(node_type, current)
                    )),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type, current in hidden.items()
            }
        user_tangent = hidden["user"]
        class_logits = self.coordination_head(user_tangent)
        return {
            "user_tangent": user_tangent,
            "coordination_class_logits": class_logits,
            "coordination_logits": class_logits[:, 1] - class_logits[:, 0],
        }

    def geometry_metadata(self):
        return {
            "geometry_backend": self.geometry_backend,
            "decision_geometry": "euclidean_linear",
            "edge_reliability_gate": False,
            "privileged_simulation_heads": [],
        }


class DomainAwareMLP(CoordinationCheckpointMixin, torch.nn.Module):
    """Per-user no-message-passing baseline for shortcut diagnosis."""

    geometry_backend = "node_mlp"
    enable_privileged_heads = False

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        metadata,
        dataset_domains: Sequence[str],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types = tuple(metadata[0])
        self.dataset_domains = tuple(dataset_domains)
        self.feature_adapters = _DatasetFeatureAdapters(
            self.dataset_domains, node_types, hidden_dim
        ).layers
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.coordination_head = torch.nn.Linear(hidden_dim, 2)

    def forward(
        self,
        graph,
        *,
        domain: str,
        dataset_name: Optional[str] = None,
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
    ):
        del domain, edge_mask_dict
        dataset = dataset_name or "synthetic"
        if dataset not in self.feature_adapters:
            raise ValueError(f"model was not configured for dataset {dataset!r}")
        user_tangent = self.feature_adapters[dataset]["user"](
            graph["user"].x.float()
        )
        for layer in self.layers:
            user_tangent = user_tangent + layer(user_tangent)
        class_logits = self.coordination_head(user_tangent)
        return {
            "user_tangent": user_tangent,
            "coordination_class_logits": class_logits,
            "coordination_logits": class_logits[:, 1] - class_logits[:, 0],
        }

    def geometry_metadata(self):
        return {
            "geometry_backend": self.geometry_backend,
            "decision_geometry": "euclidean_linear",
            "message_passing": False,
            "privileged_simulation_heads": [],
        }
