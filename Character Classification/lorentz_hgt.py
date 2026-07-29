"""Intrinsic Lorentz heterogeneous message passing for HyperDecept.

The previous role encoder performs Euclidean HGT message passing and only
projects the final representation into a Poincare ball.  This module keeps
node states on a Lorentz manifold during relation-specific propagation.  Its
design follows the curvature-aware HT/HR operators and relation-specific
spaces used by HypHGT (Park et al., arXiv:2601.08251), while exposing edge
masks and relation gates required by HyperDecept's evidence explanations.

This is intentionally a small, dependency-light backbone.  It uses PyTorch
and ``torch_geometric.utils.softmax`` only, so the research code does not
depend on an unmaintained hyperbolic-learning package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EdgeType = Tuple[str, str, str]


def edge_type_name(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentzian inner product using the time-first convention."""
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(
        dim=-1, keepdim=True
    )


def lorentz_from_spatial(
    spatial: torch.Tensor,
    curvature: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reconstruct a valid point on the upper Lorentz hyperboloid.

    ``curvature`` is the positive magnitude ``k`` of negative curvature
    ``-k``.  Returned points satisfy ``<x, x>_L = -1/k``.
    """
    k = curvature.to(dtype=spatial.dtype, device=spatial.device).clamp_min(eps)
    time = torch.sqrt((1.0 / k) + spatial.square().sum(dim=-1, keepdim=True))
    return torch.cat([time, spatial], dim=-1)


def expmap0(
    tangent_spatial: torch.Tensor,
    curvature: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map an origin-tangent vector to the Lorentz manifold."""
    k = curvature.to(
        dtype=tangent_spatial.dtype, device=tangent_spatial.device
    ).clamp_min(eps)
    sqrt_k = torch.sqrt(k)
    norm = torch.linalg.vector_norm(tangent_spatial, dim=-1, keepdim=True)
    scaled = (sqrt_k * norm).clamp(max=15.0)
    direction_scale = torch.sinh(scaled) / (sqrt_k * norm).clamp_min(eps)
    direction_scale = torch.where(
        norm > eps, direction_scale, torch.ones_like(direction_scale)
    )
    time = torch.cosh(scaled) / sqrt_k
    return torch.cat([time, direction_scale * tangent_spatial], dim=-1)


def logmap0(
    point: torch.Tensor,
    curvature: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the spatial coordinates of the origin logarithmic map."""
    k = curvature.to(dtype=point.dtype, device=point.device).clamp_min(eps)
    alpha = (torch.sqrt(k) * point[..., :1]).clamp_min(1.0 + eps)
    denominator = torch.sqrt((alpha.square() - 1.0).clamp_min(eps))
    scale = torch.acosh(alpha) / denominator
    return scale * point[..., 1:]


def lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Geodesic distance between Lorentz points at the same curvature."""
    k = curvature.to(dtype=x.dtype, device=x.device).clamp_min(eps)
    argument = (-k * minkowski_dot(x, y)).clamp_min(1.0 + eps)
    return torch.acosh(argument) / torch.sqrt(k)


def lorentz_to_poincare(
    point: torch.Tensor,
    curvature: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert to a dimensionless unit Poincare ball representation."""
    k = curvature.to(dtype=point.dtype, device=point.device).clamp_min(eps)
    sqrt_k = torch.sqrt(k)
    return sqrt_k * point[..., 1:] / (sqrt_k * point[..., :1] + 1.0).clamp_min(eps)


def weighted_lorentz_centroid(
    points: torch.Tensor,
    weights: torch.Tensor,
    curvature: torch.Tensor,
    *,
    dim: int = -2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a positive weighted Minkowski sum back to the hyperboloid."""
    if points.ndim < 2 or points.shape[-1] < 2:
        raise ValueError("points must contain Lorentz coordinates")
    if weights.shape != points.shape[:-1]:
        raise ValueError("weights must match points without the coordinate axis")
    if torch.any(weights < 0):
        raise ValueError("Lorentz centroid weights must be non-negative")
    coordinate_dim = points.ndim - 1
    normalized_dim = dim if dim >= 0 else points.ndim + dim
    if normalized_dim < 0 or normalized_dim >= coordinate_dim:
        raise ValueError("centroid dimension must index a non-coordinate axis")

    weighted_sum = (points * weights.unsqueeze(-1)).sum(dim=normalized_dim)
    total_weight = weights.sum(dim=normalized_dim)
    minkowski_norm = minkowski_dot(weighted_sum, weighted_sum).squeeze(-1)
    valid = (total_weight > eps) & (minkowski_norm < -eps)
    scale = torch.sqrt(
        (-curvature.to(weighted_sum) * minkowski_norm).clamp_min(eps)
    )
    centroid = weighted_sum / scale.unsqueeze(-1)
    origin = torch.zeros_like(centroid)
    origin[..., 0] = torch.rsqrt(curvature.to(origin))
    return torch.where(valid.unsqueeze(-1), centroid, origin)


class LorentzPrototypeClassifier(torch.nn.Module):
    """Classify manifold points by geodesic distance to learned prototypes."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        *,
        initial_temperature: float = 1.0,
        minimum_temperature: float = 1e-3,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_classes <= 1:
            raise ValueError("prototype dimensions and class count must be valid")
        if initial_temperature <= minimum_temperature:
            raise ValueError("initial temperature must exceed its minimum")
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.minimum_temperature = float(minimum_temperature)
        self.prototype_spatial = torch.nn.Parameter(
            torch.empty(num_classes, hidden_dim)
        )
        torch.nn.init.normal_(self.prototype_spatial, mean=0.0, std=0.02)
        inverse_softplus = torch.log(
            torch.expm1(torch.tensor(initial_temperature - minimum_temperature))
        )
        self.temperature_raw = torch.nn.Parameter(inverse_softplus)
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def temperature(self) -> torch.Tensor:
        return self.minimum_temperature + F.softplus(self.temperature_raw)

    def prototypes(self, curvature: torch.Tensor) -> torch.Tensor:
        return lorentz_from_spatial(self.prototype_spatial, curvature)

    def forward(
        self,
        points: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        if points.shape[-1] != self.hidden_dim + 1:
            raise ValueError("point dimension does not match prototype dimension")
        distances = lorentz_distance(
            points.unsqueeze(-2),
            self.prototypes(curvature).unsqueeze(0),
            curvature,
        ).squeeze(-1)
        return self.bias - distances / self.temperature().to(points)


class BoundedCurvature(torch.nn.Module):
    """Stable learnable magnitude for a negative curvature."""

    def __init__(
        self,
        initial: float = 1.0,
        minimum: float = 1e-3,
        maximum: float = 10.0,
    ) -> None:
        super().__init__()
        if not minimum < initial < maximum:
            raise ValueError("curvature must satisfy minimum < initial < maximum")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        ratio = (initial - minimum) / (maximum - minimum)
        self.raw = torch.nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(ratio)))))

    def forward(self) -> torch.Tensor:
        span = self.maximum - self.minimum
        return self.minimum + span * torch.sigmoid(self.raw)


class LorentzLinear(torch.nn.Module):
    """Curvature-aware hyperbolic transformation (HT)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim + 1, output_dim)

    def forward(
        self,
        point: torch.Tensor,
        input_curvature: torch.Tensor,
        output_curvature: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.sqrt(
            input_curvature.to(point) / output_curvature.to(point)
        )
        spatial = ratio * self.linear(point)
        return lorentz_from_spatial(spatial, output_curvature)


@dataclass
class RelationAudit:
    curvature: torch.Tensor
    attention: torch.Tensor
    reliability: torch.Tensor


class LorentzRelationMessage(torch.nn.Module):
    """Relation-specific intrinsic distance attention and aggregation."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.curvature = BoundedCurvature()
        self.source_transform = LorentzLinear(hidden_dim, hidden_dim)
        self.target_transform = LorentzLinear(hidden_dim, hidden_dim)
        self.output_transform = LorentzLinear(hidden_dim, hidden_dim)
        self.distance_scale_raw = torch.nn.Parameter(torch.zeros(num_heads))
        self.attention_bias = torch.nn.Parameter(torch.zeros(num_heads))
        self.reliability_distance_scale_raw = torch.nn.Parameter(
            torch.zeros(num_heads)
        )
        self.reliability_bias = torch.nn.Parameter(torch.ones(num_heads))
        self.reliability_attribute_weights = torch.nn.Parameter(
            torch.zeros(num_heads, 4)
        )

    def _reliability_attributes(
        self,
        edge_count: int,
        reference: torch.Tensor,
        edge_attributes: Optional[Mapping[str, torch.Tensor]],
    ) -> torch.Tensor:
        defaults = {
            "base_weight": 1.0,
            "multiplicity": 1.0,
            "temporal_sync": 0.0,
            "temporal_recency": 0.0,
            "temporal_available": 0.0,
        }
        values: Dict[str, torch.Tensor] = {}
        for name, default in defaults.items():
            raw = None if edge_attributes is None else edge_attributes.get(name)
            if raw is None:
                values[name] = reference.new_full((edge_count,), default)
                continue
            if raw.ndim != 1 or raw.shape[0] != edge_count:
                raise ValueError(
                    f"edge attribute {name} must align with edge count {edge_count}"
                )
            values[name] = raw.to(device=reference.device, dtype=reference.dtype)
            if not torch.isfinite(values[name]).all():
                raise ValueError(f"edge attribute {name} must contain finite values")
        temporal_available = values["temporal_available"].clamp(0.0, 1.0)
        return torch.stack([
            torch.log1p(values["base_weight"].clamp_min(0.0)),
            torch.log1p(values["multiplicity"].clamp_min(0.0)),
            values["temporal_sync"].clamp(0.0, 1.0) * temporal_available,
            values["temporal_recency"].clamp(0.0, 1.0) * temporal_available,
        ], dim=-1)

    def forward(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        edge_index: torch.Tensor,
        common_curvature: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        edge_attributes: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, RelationAudit]:
        from torch_geometric.utils import softmax

        relation_curvature = self.curvature()
        source_relation = self.source_transform(
            source_points, common_curvature, relation_curvature
        )
        target_relation = self.target_transform(
            target_points, common_curvature, relation_curvature
        )
        source_index, target_index = edge_index
        distance = lorentz_distance(
            source_relation[source_index],
            target_relation[target_index],
            relation_curvature,
        )
        distance_scale = F.softplus(self.distance_scale_raw) + 1e-4
        logits = self.attention_bias.unsqueeze(0) - distance_scale.unsqueeze(0) * distance
        attention = softmax(
            logits, target_index, num_nodes=target_points.shape[0], dim=0
        )

        reliability_scale = F.softplus(
            self.reliability_distance_scale_raw
        ) + 1e-4
        reliability_attributes = self._reliability_attributes(
            edge_index.shape[1], distance, edge_attributes
        )
        attribute_contribution = (
            reliability_attributes.unsqueeze(1)
            * self.reliability_attribute_weights.unsqueeze(0)
        ).sum(dim=-1)
        reliability = torch.sigmoid(
            self.reliability_bias.unsqueeze(0)
            - reliability_scale.unsqueeze(0) * distance
            + attribute_contribution
        )
        attention = attention * reliability

        if edge_mask is not None:
            mask = edge_mask.to(attention).clamp(0.0, 1.0).unsqueeze(-1)
            attention = attention * mask

        source_spatial = source_relation[source_index, 1:].reshape(
            -1, self.num_heads, self.head_dim
        )
        source_head_points = lorentz_from_spatial(
            source_spatial, relation_curvature
        )
        weighted_points = source_head_points * attention.unsqueeze(-1)
        aggregate = weighted_points.new_zeros(
            (target_points.shape[0], self.num_heads, self.head_dim + 1)
        )
        aggregate.index_add_(0, target_index, weighted_points)
        confidence = attention.new_zeros(
            (target_points.shape[0], self.num_heads)
        )
        confidence.index_add_(0, target_index, attention)
        confidence = confidence.clamp(0.0, 1.0)
        head_centroids = weighted_lorentz_centroid(
            aggregate.unsqueeze(-2),
            torch.ones_like(aggregate[..., 0]).unsqueeze(-1),
            relation_curvature,
            dim=-2,
        )
        head_origin = torch.zeros_like(head_centroids)
        head_origin[..., 0] = torch.rsqrt(
            relation_curvature.to(head_origin)
        )
        head_centroids = weighted_lorentz_centroid(
            torch.stack([head_origin, head_centroids], dim=-2),
            torch.stack([1.0 - confidence, confidence], dim=-1),
            relation_curvature,
            dim=-2,
        )
        combined_spatial = head_centroids[..., 1:].reshape(
            target_points.shape[0], -1
        )
        relation_point = lorentz_from_spatial(
            combined_spatial, relation_curvature
        )
        common_point = self.output_transform(
            relation_point, relation_curvature, common_curvature
        )
        return common_point, RelationAudit(
            curvature=relation_curvature,
            attention=attention,
            reliability=reliability,
        )


class IntrinsicLorentzHGTLayer(torch.nn.Module):
    """One auditable heterogeneous Lorentz message-passing layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        metadata: Tuple[Sequence[str], Sequence[EdgeType]],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.relations = torch.nn.ModuleDict({
            edge_type_name(edge_type): LorentzRelationMessage(
                hidden_dim, num_heads
            )
            for edge_type in self.edge_types
        })
        self.relation_gates = torch.nn.ParameterDict({
            edge_type_name(edge_type): torch.nn.Parameter(torch.tensor(0.0))
            for edge_type in self.edge_types
        })
        self.fusion_gates = torch.nn.ModuleDict({
            node_type: torch.nn.Sequential(
                torch.nn.Linear(3 * hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )
            for node_type in node_types
        })
        self.norms = torch.nn.ModuleDict({
            node_type: torch.nn.LayerNorm(hidden_dim) for node_type in node_types
        })
        self.dropout = float(dropout)

    def forward(
        self,
        point_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        common_curvature: torch.Tensor,
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
        edge_attribute_dict: Optional[
            Mapping[EdgeType, Mapping[str, torch.Tensor]]
        ] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, RelationAudit]]:
        candidates: Dict[str, list] = {node_type: [] for node_type in point_dict}
        audits: Dict[str, RelationAudit] = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
            source_type, _, target_type = edge_type
            key = edge_type_name(edge_type)
            candidate, audit = self.relations[key](
                point_dict[source_type],
                point_dict[target_type],
                edge_index_dict[edge_type],
                common_curvature,
                None if edge_mask_dict is None else edge_mask_dict.get(edge_type),
                None if edge_attribute_dict is None else edge_attribute_dict.get(
                    edge_type
                ),
            )
            candidates[target_type].append((key, candidate))
            audits[key] = audit

        output: Dict[str, torch.Tensor] = {}
        for node_type, current in point_dict.items():
            typed_candidates = candidates[node_type]
            if typed_candidates:
                gate_logits = torch.stack([
                    self.relation_gates[key] for key, _ in typed_candidates
                ])
                relation_weights = torch.softmax(gate_logits, dim=0)
                candidate_points = torch.stack(
                    [candidate for _, candidate in typed_candidates], dim=1
                )
                candidate_weights = relation_weights.unsqueeze(0).expand(
                    current.shape[0], -1
                )
                relation_centroid = weighted_lorentz_centroid(
                    candidate_points,
                    candidate_weights,
                    common_curvature,
                    dim=1,
                )
                residual_spatial = current[..., 1:]
                relation_spatial = relation_centroid[..., 1:]
                residual_weight = torch.sigmoid(
                    self.fusion_gates[node_type](torch.cat([
                        residual_spatial,
                        relation_spatial,
                        torch.abs(residual_spatial - relation_spatial),
                    ], dim=-1))
                ).squeeze(-1)
                mixed = weighted_lorentz_centroid(
                    torch.stack([current, relation_centroid], dim=1),
                    torch.stack([
                        residual_weight.expand(current.shape[0]),
                        (1.0 - residual_weight).expand(current.shape[0]),
                    ], dim=1),
                    common_curvature,
                    dim=1,
                )
            else:
                mixed = current
            refined = self.norms[node_type](mixed[..., 1:])
            refined = F.gelu(refined)
            refined = F.dropout(refined, p=self.dropout, training=self.training)
            output[node_type] = lorentz_from_spatial(refined, common_curvature)
        return output, audits


class IntrinsicLorentzHGT(torch.nn.Module):
    """Relation-aware HGT whose hidden states stay on a Lorentz manifold."""

    geometry_backend = "intrinsic_lorentz"

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        metadata: Tuple[Sequence[str], Sequence[EdgeType]],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import Linear

        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.metadata = metadata
        self.common_curvature = BoundedCurvature()
        self.input_scale_raw = torch.nn.Parameter(torch.tensor(-2.0))
        self.input_projectors = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in metadata[0]
        })
        self.layers = torch.nn.ModuleList([
            IntrinsicLorentzHGTLayer(
                hidden_dim, num_heads, metadata, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self._last_audits: Dict[str, RelationAudit] = {}

    def encode_lorentz_nodes(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
        edge_attribute_dict: Optional[
            Mapping[EdgeType, Mapping[str, torch.Tensor]]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        curvature = self.common_curvature()
        scale = torch.sigmoid(self.input_scale_raw)
        points = {
            node_type: expmap0(
                scale * self.input_projectors[node_type](features), curvature
            )
            for node_type, features in x_dict.items()
        }
        all_audits: Dict[str, RelationAudit] = {}
        for layer_index, layer in enumerate(self.layers):
            points, audits = layer(
                points,
                edge_index_dict,
                curvature,
                edge_mask_dict=edge_mask_dict,
                edge_attribute_dict=edge_attribute_dict,
            )
            all_audits.update({
                f"layer_{layer_index}:{key}": value for key, value in audits.items()
            })
        self._last_audits = all_audits
        return points

    def encode_nodes(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
        edge_attribute_dict: Optional[
            Mapping[EdgeType, Mapping[str, torch.Tensor]]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        curvature = self.common_curvature()
        points = self.encode_lorentz_nodes(
            x_dict,
            edge_index_dict,
            edge_mask_dict=edge_mask_dict,
            edge_attribute_dict=edge_attribute_dict,
        )
        return {
            node_type: lorentz_to_poincare(point, curvature)
            for node_type, point in points.items()
        }

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
        edge_attribute_dict: Optional[
            Mapping[EdgeType, Mapping[str, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        return self.encode_nodes(
            x_dict,
            edge_index_dict,
            edge_mask_dict=edge_mask_dict,
            edge_attribute_dict=edge_attribute_dict,
        )["user"]

    def geometry_metadata(self) -> Dict[str, object]:
        relation_curvatures: Dict[str, float] = {}
        for layer_index, layer in enumerate(self.layers):
            for key, relation in layer.relations.items():
                relation_curvatures[f"layer_{layer_index}:{key}"] = -float(
                    relation.curvature().detach().cpu()
                )
        return {
            "geometry_backend": self.geometry_backend,
            "common_curvature": -float(
                self.common_curvature().detach().cpu()
            ),
            "relation_curvatures": relation_curvatures,
            "edge_masks_supported": True,
            "edge_reliability_gate": True,
            "edge_reliability_attributes": [
                "base_weight",
                "multiplicity",
                "temporal_sync",
                "temporal_recency",
                "temporal_available",
            ],
            "node_adaptive_self_neighbor_fusion": True,
        }
