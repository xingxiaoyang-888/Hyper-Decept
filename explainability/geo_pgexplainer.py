"""PGExplainer-style edge masks with optional hyperbolic geometry fidelity.

The implementation is deliberately model-agnostic. The caller supplies a frozen
graph model through ``model_forward(edge_mask_dict)`` and chooses how its output
is converted into a scalar prediction through ``prediction_loss``. This keeps
the explainer reusable for the current projection-head HGT and a later intrinsic
hyperbolic HGT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from .hyperbolic_geometry import (
    normalized_distance_distortion,
    poincare_distance,
    poincare_radius,
    radial_order_agreement,
    radial_order_loss,
)


EdgeType = Tuple[str, str, str]
EdgeMaskDict = Dict[EdgeType, torch.Tensor]


@dataclass
class GeoPGExplainerConfig:
    epochs: int = 100
    learning_rate: float = 0.01
    hidden_dim: int = 64
    temperature_start: float = 5.0
    temperature_end: float = 1.0
    size_coefficient: float = 0.01
    entropy_coefficient: float = 0.005
    geodesic_coefficient: float = 0.0
    radial_coefficient: float = 0.0
    radial_margin: float = 0.0
    relation_size_coefficients: Dict[str, float] = field(default_factory=dict)
    seed: int = 42


@dataclass
class GeoPGExplanation:
    edge_masks: EdgeMaskDict
    losses: Dict[str, float]
    metrics: Dict[str, float]
    mode: str


def edge_type_name(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def default_prediction_loss(
    explained_output: torch.Tensor,
    reference_output: torch.Tensor,
) -> torch.Tensor:
    """Preserve a frozen model output using mean-squared fidelity."""
    return F.mse_loss(explained_output, reference_output.detach())


def role_probability_fidelity(
    reference_probabilities: torch.Tensor,
    explained_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Mean probability agreement for a supplied role classifier."""
    if reference_probabilities.shape != explained_probabilities.shape:
        raise ValueError("role probability tensors must have equal shape")
    return 1.0 - torch.mean(
        torch.abs(reference_probabilities - explained_probabilities)
    ).clamp(0.0, 1.0)


def _edge_features(
    node_embeddings: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[EdgeType, torch.Tensor],
) -> Dict[EdgeType, torch.Tensor]:
    features: Dict[EdgeType, torch.Tensor] = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        src = node_embeddings[src_type][edge_index[0]]
        dst = node_embeddings[dst_type][edge_index[1]]
        features[edge_type] = torch.cat([src, dst, src - dst], dim=-1)
    return features


class GeoPGExplainer(torch.nn.Module):
    """Train relation-specific parameterized edge mask generators.

    With both geometry coefficients set to zero this is the ordinary
    PGExplainer-style baseline used by the falsification experiment. Positive
    coefficients enable Geo-PGExplainer.
    """

    def __init__(self, config: Optional[GeoPGExplainerConfig] = None) -> None:
        super().__init__()
        self.config = config or GeoPGExplainerConfig()
        self.mask_networks = torch.nn.ModuleDict()

    def _ensure_networks(self, features: Mapping[EdgeType, torch.Tensor]) -> None:
        for edge_type, values in features.items():
            name = edge_type_name(edge_type)
            if name in self.mask_networks:
                continue
            input_dim = int(values.shape[-1])
            self.mask_networks[name] = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.config.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_dim, 1),
            ).to(values.device)

    @staticmethod
    def _concrete_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logits + logistic_noise) / temperature)

    def _temperature(self, epoch: int) -> float:
        if self.config.epochs <= 1:
            return self.config.temperature_end
        fraction = epoch / float(self.config.epochs - 1)
        start = self.config.temperature_start
        end = self.config.temperature_end
        return start * (end / start) ** fraction

    def _build_masks(
        self,
        features: Mapping[EdgeType, torch.Tensor],
        temperature: float,
        training: bool,
    ) -> EdgeMaskDict:
        masks: EdgeMaskDict = {}
        for edge_type, values in features.items():
            logits = self.mask_networks[edge_type_name(edge_type)](values).squeeze(-1)
            masks[edge_type] = (
                self._concrete_sample(logits, temperature)
                if training
                else torch.sigmoid(logits)
            )
        return masks

    def _regularization(self, masks: Mapping[EdgeType, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = next(iter(masks.values()))
        size_loss = sample.new_tensor(0.0)
        entropy_loss = sample.new_tensor(0.0)
        for edge_type, mask in masks.items():
            coefficient = self.config.relation_size_coefficients.get(
                edge_type_name(edge_type), self.config.size_coefficient
            )
            size_loss = size_loss + float(coefficient) * mask.mean()
            probabilities = mask.clamp(1e-6, 1.0 - 1e-6)
            entropy = -probabilities * torch.log(probabilities)
            entropy -= (1.0 - probabilities) * torch.log(1.0 - probabilities)
            entropy_loss = entropy_loss + entropy.mean()
        return size_loss, self.config.entropy_coefficient * entropy_loss

    @staticmethod
    def _geometry_terms(
        reference_embedding: torch.Tensor,
        explained_embedding: torch.Tensor,
        geometry_pairs: Optional[torch.Tensor],
        radial_margin: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        reference_radius = poincare_radius(reference_embedding)
        explained_radius = poincare_radius(explained_embedding)
        radial = radial_order_loss(reference_radius, explained_radius, margin=radial_margin)

        if geometry_pairs is None:
            geometry_pairs = torch.triu_indices(
                reference_embedding.shape[0], reference_embedding.shape[0],
                offset=1, device=reference_embedding.device,
            )
        if geometry_pairs.numel() == 0:
            reference_distance = reference_radius.new_empty(0)
            explained_distance = explained_radius.new_empty(0)
        else:
            reference_distance = poincare_distance(
                reference_embedding[geometry_pairs[0]],
                reference_embedding[geometry_pairs[1]],
            )
            explained_distance = poincare_distance(
                explained_embedding[geometry_pairs[0]],
                explained_embedding[geometry_pairs[1]],
            )
        geodesic = normalized_distance_distortion(reference_distance, explained_distance)
        metrics = {
            "geodesic_distortion": geodesic.detach(),
            "radial_order_agreement": radial_order_agreement(
                reference_radius, explained_radius
            ).detach(),
        }
        return geodesic, radial, metrics

    def fit(
        self,
        node_embeddings: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        model_forward: Callable[[EdgeMaskDict], Tuple[torch.Tensor, torch.Tensor]],
        prediction_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = default_prediction_loss,
        geometry_pairs: Optional[torch.Tensor] = None,
    ) -> GeoPGExplanation:
        """Fit masks against a frozen model and return deterministic edge scores.

        ``model_forward`` must return ``(prediction_output, user_poincare_embedding)``.
        It is responsible for applying each continuous edge mask during message
        passing; this avoids any dependency on a specific PyG version here.
        """
        if not edge_index_dict:
            raise ValueError("edge_index_dict must contain at least one relation")
        torch.manual_seed(self.config.seed)
        features = _edge_features(node_embeddings, edge_index_dict)
        self._ensure_networks(features)
        reference_output, reference_embedding = model_forward(
            {edge_type: torch.ones(edge_index.shape[1], device=edge_index.device)
             for edge_type, edge_index in edge_index_dict.items()}
        )
        reference_output = reference_output.detach()
        reference_embedding = reference_embedding.detach()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        last_losses: Dict[str, float] = {}
        for epoch in range(self.config.epochs):
            self.train()
            optimizer.zero_grad()
            masks = self._build_masks(features, self._temperature(epoch), training=True)
            explained_output, explained_embedding = model_forward(masks)
            fidelity = prediction_loss(explained_output, reference_output)
            size, entropy = self._regularization(masks)
            geodesic, radial, _ = self._geometry_terms(
                reference_embedding, explained_embedding, geometry_pairs,
                self.config.radial_margin,
            )
            total = fidelity + size + entropy
            total += self.config.geodesic_coefficient * geodesic
            total += self.config.radial_coefficient * radial
            total.backward()
            optimizer.step()
            last_losses = {
                "total": float(total.detach()),
                "prediction_fidelity": float(fidelity.detach()),
                "size": float(size.detach()),
                "entropy": float(entropy.detach()),
                "geodesic": float(geodesic.detach()),
                "radial": float(radial.detach()),
            }

        self.eval()
        with torch.no_grad():
            masks = self._build_masks(features, self.config.temperature_end, training=False)
            explained_output, explained_embedding = model_forward(masks)
            final_fidelity = prediction_loss(explained_output, reference_output)
            _, _, geometry_metrics = self._geometry_terms(
                reference_embedding, explained_embedding, geometry_pairs,
                self.config.radial_margin,
            )
            metrics = {
                "prediction_fidelity_loss": float(final_fidelity),
                "selected_edge_fraction_at_0.5": float(torch.cat([
                    mask.reshape(-1) for mask in masks.values()
                ]).ge(0.5).float().mean()),
                **{key: float(value) for key, value in geometry_metrics.items()},
            }

        mode = (
            "geo_pgexplainer"
            if self.config.geodesic_coefficient > 0 or self.config.radial_coefficient > 0
            else "pgexplainer"
        )
        return GeoPGExplanation(
            edge_masks={key: value.detach().cpu() for key, value in masks.items()},
            losses=last_losses,
            metrics=metrics,
            mode=mode,
        )
