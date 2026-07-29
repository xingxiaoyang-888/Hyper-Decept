"""Sparse neural additive risk head with exact visible decomposition.

Each concept has its own shape network and only explicitly declared concept
pairs receive an interaction network.  The final logit is exactly:

    bias + sum(main concept effects) + sum(allowed interaction effects)

This preserves nonlinear capacity while keeping every decision contribution
visible and evidence-linkable.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adapters import ExplanationFragment, ExplanationRequest
from .schemas import ConceptRecord, ContributionRecord, PredictionRecord


class _ShapeFunction(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass(frozen=True)
class AdditiveDecision:
    case_id: str
    probability: float
    logit: float
    bias: float
    concept_values: Dict[str, float]
    main_contributions: Dict[str, float]
    interaction_contributions: Dict[str, float]
    reconstruction_error: float

    def to_fragment(
        self,
        evidence_map: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> ExplanationFragment:
        evidence = {
            str(name): list(map(str, values))
            for name, values in (evidence_map or {}).items()
        }
        concepts = []
        contributions = []
        for name, value in self.concept_values.items():
            contribution = self.main_contributions[name]
            ids = evidence.get(name, [])
            concepts.append(
                ConceptRecord(
                    concept_name=name,
                    concept_value=value,
                    activation=contribution,
                    evidence_ids=ids,
                    metadata={
                        "main_effect_contribution": contribution,
                        "decomposition_space": "logit",
                    },
                )
            )
            contributions.append(
                ContributionRecord(
                    feature_name=f"concept:{name}",
                    feature_value=value,
                    contribution=contribution,
                    direction=("supporting" if contribution >= 0 else "opposing"),
                    evidence_ids=ids,
                    metadata={"component_type": "main_effect"},
                )
            )
        for pair_name, contribution in self.interaction_contributions.items():
            left, right = pair_name.split("::", maxsplit=1)
            ids = list(dict.fromkeys(evidence.get(left, []) + evidence.get(right, [])))
            contributions.append(
                ContributionRecord(
                    feature_name=f"interaction:{left}×{right}",
                    feature_value=0.0,
                    contribution=contribution,
                    direction=("supporting" if contribution >= 0 else "opposing"),
                    evidence_ids=ids,
                    metadata={
                        "component_type": "allowed_interaction",
                        "concept_values": {
                            left: self.concept_values[left],
                            right: self.concept_values[right],
                        },
                    },
                )
            )
        contributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        for rank, contribution in enumerate(contributions, start=1):
            contribution.rank = rank
        all_evidence = list(
            dict.fromkeys(
                evidence_id
                for contribution in contributions
                for evidence_id in contribution.evidence_ids
            )
        )
        return ExplanationFragment(
            contributions=contributions,
            concepts=concepts,
            evidence_ids=all_evidence,
            metadata={
                "additive_decomposition": {
                    "bias": self.bias,
                    "logit": self.logit,
                    "probability": self.probability,
                    "visible_contribution_sum": (
                        self.bias
                        + sum(self.main_contributions.values())
                        + sum(self.interaction_contributions.values())
                    ),
                    "reconstruction_error": self.reconstruction_error,
                    "space": "logit",
                }
            },
        )


class SparseNeuralAdditiveRiskHead(nn.Module):
    """Nonlinear NAM with sparse main effects and allow-listed interactions."""

    def __init__(
        self,
        concept_names: Sequence[str],
        *,
        interactions: Sequence[Tuple[str, str]] = (),
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.concept_names = [str(name) for name in concept_names]
        if not self.concept_names or any(not name for name in self.concept_names):
            raise ValueError("concept_names must contain non-empty names")
        if len(set(self.concept_names)) != len(self.concept_names):
            raise ValueError("concept_names must be unique")
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self._concept_index = {
            name: index for index, name in enumerate(self.concept_names)
        }
        normalized_pairs = []
        seen_pairs = set()
        for left, right in interactions:
            left, right = str(left), str(right)
            if left not in self._concept_index or right not in self._concept_index:
                raise ValueError(f"Unknown interaction concepts: {(left, right)}")
            if left == right:
                raise ValueError("Self-interactions are not allowed")
            canonical = tuple(sorted((left, right)))
            if canonical in seen_pairs:
                raise ValueError(f"Duplicate interaction: {canonical}")
            seen_pairs.add(canonical)
            normalized_pairs.append(canonical)
        self.interactions = normalized_pairs
        self.main_effects = nn.ModuleList(
            [_ShapeFunction(1, self.hidden_dim) for _ in self.concept_names]
        )
        self.interaction_effects = nn.ModuleList(
            [_ShapeFunction(2, self.hidden_dim) for _ in self.interactions]
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("normalization_mean", torch.zeros(len(self.concept_names)))
        self.register_buffer("normalization_scale", torch.ones(len(self.concept_names)))

    def _validate_x(self, values: Any) -> torch.Tensor:
        if torch.is_tensor(values):
            tensor = values.to(dtype=torch.float32, device=self.bias.device)
        else:
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.bias.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != len(self.concept_names):
            raise ValueError(
                f"Expected shape (N, {len(self.concept_names)}), got "
                f"{tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("Concept values must be finite")
        return tensor

    def set_normalization(self, values: Any) -> None:
        tensor = self._validate_x(values).detach()
        mean = tensor.mean(dim=0)
        scale = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.normalization_mean.copy_(mean)
        self.normalization_scale.copy_(scale)

    def components(self, values: Any) -> Dict[str, torch.Tensor]:
        raw = self._validate_x(values)
        normalized = (raw - self.normalization_mean) / self.normalization_scale
        main = torch.cat(
            [
                network(normalized[:, index:index + 1])
                for index, network in enumerate(self.main_effects)
            ],
            dim=1,
        )
        interaction_parts = []
        for (left, right), network in zip(
            self.interactions, self.interaction_effects
        ):
            pair = torch.stack(
                [
                    normalized[:, self._concept_index[left]],
                    normalized[:, self._concept_index[right]],
                ],
                dim=1,
            )
            interaction_parts.append(network(pair))
        interactions = (
            torch.cat(interaction_parts, dim=1)
            if interaction_parts
            else raw.new_zeros((raw.shape[0], 0))
        )
        logits = self.bias + main.sum(dim=1) + interactions.sum(dim=1)
        return {
            "raw": raw,
            "normalized": normalized,
            "main": main,
            "interactions": interactions,
            "logits": logits,
        }

    def forward(self, values: Any) -> torch.Tensor:
        return self.components(values)["logits"]

    def fit_model(
        self,
        values: Any,
        labels: Any,
        *,
        validation_data: Optional[Tuple[Any, Any]] = None,
        epochs: int = 300,
        learning_rate: float = 3e-3,
        sparsity_strength: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 40,
        random_seed: int = 42,
    ) -> List[Dict[str, float]]:
        """Train with AdamW and output-level sparsity regularization.

        A validation set is accepted explicitly; the method never creates a
        random internal split that could violate group/time separation.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        torch.manual_seed(int(random_seed))
        np.random.seed(int(random_seed))
        x = self._validate_x(values)
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.bias.device).view(-1)
        if len(y) != len(x):
            raise ValueError("labels length must equal number of rows")
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("labels must be binary 0/1")
        self.set_normalization(x)
        validation = None
        if validation_data is not None:
            val_x = self._validate_x(validation_data[0])
            val_y = torch.as_tensor(
                validation_data[1], dtype=torch.float32, device=self.bias.device
            ).view(-1)
            if len(val_y) != len(val_x):
                raise ValueError("validation labels length mismatch")
            validation = (val_x, val_y)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        history: List[Dict[str, float]] = []
        best_state = None
        best_score = float("inf")
        stale_epochs = 0
        for epoch in range(1, int(epochs) + 1):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            parts = self.components(x)
            prediction_loss = F.binary_cross_entropy_with_logits(parts["logits"], y)
            visible = torch.cat([parts["main"], parts["interactions"]], dim=1)
            sparsity = visible.abs().mean()
            loss = prediction_loss + float(sparsity_strength) * sparsity
            loss.backward()
            optimizer.step()
            row = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "prediction_loss": float(prediction_loss.detach().cpu()),
                "sparsity_penalty": float(sparsity.detach().cpu()),
            }
            score = row["prediction_loss"]
            if validation is not None:
                self.eval()
                with torch.no_grad():
                    val_logits = self(validation[0])
                    val_loss = F.binary_cross_entropy_with_logits(
                        val_logits, validation[1]
                    )
                score = float(val_loss.cpu())
                row["validation_loss"] = score
            history.append(row)
            if score < best_score - 1e-7:
                best_score = score
                best_state = copy.deepcopy(self.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if validation is not None and stale_epochs >= int(patience):
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return history

    def explain_one(self, values: Any, case_id: str = "") -> AdditiveDecision:
        self.eval()
        with torch.no_grad():
            parts = self.components(values)
        if parts["raw"].shape[0] != 1:
            raise ValueError("explain_one requires exactly one row")
        raw = parts["raw"][0].cpu().tolist()
        main = parts["main"][0].cpu().tolist()
        interaction_values = parts["interactions"][0].cpu().tolist()
        logit = float(parts["logits"][0].cpu())
        bias = float(self.bias.detach().cpu())
        visible_sum = bias + sum(main) + sum(interaction_values)
        return AdditiveDecision(
            case_id=str(case_id),
            probability=float(torch.sigmoid(parts["logits"][0]).cpu()),
            logit=logit,
            bias=bias,
            concept_values=dict(zip(self.concept_names, map(float, raw))),
            main_contributions=dict(zip(self.concept_names, map(float, main))),
            interaction_contributions={
                f"{left}::{right}": float(value)
                for (left, right), value in zip(
                    self.interactions, interaction_values
                )
            },
            reconstruction_error=abs(logit - visible_sum),
        )

    def configuration(self) -> dict:
        return {
            "concept_names": self.concept_names,
            "interactions": [list(pair) for pair in self.interactions],
            "hidden_dim": self.hidden_dim,
        }

    def save(self, path: str) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"configuration": self.configuration(), "state_dict": self.state_dict()},
            destination,
        )
        return str(destination)

    @classmethod
    def load(cls, path: str, *, map_location: str = "cpu") -> "SparseNeuralAdditiveRiskHead":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**payload["configuration"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class AdditiveRiskAdapter:
    """Register one additive head as both predictor and explainer adapter."""

    def __init__(
        self,
        head: SparseNeuralAdditiveRiskHead,
        *,
        adapter_id: str = "sparse_neural_additive_risk",
        model_version: str = "nam-0.1",
        threshold: float = 0.5,
        positive_label: str = "coordinated_deception",
        negative_label: str = "normal",
    ) -> None:
        self.head = head
        self.adapter_id = adapter_id
        self.model_version = model_version
        self.threshold = float(threshold)
        self.positive_label = positive_label
        self.negative_label = negative_label

    def _vector(self, request: ExplanationRequest) -> List[float]:
        values = request.payload.get("concepts", request.payload.get("features"))
        if isinstance(values, Mapping):
            missing = [name for name in self.head.concept_names if name not in values]
            if missing:
                raise ValueError(f"Missing concepts: {missing}")
            return [float(values[name]) for name in self.head.concept_names]
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return list(map(float, values))
        raise ValueError("request payload must contain concepts mapping or features list")

    def predict(self, request: ExplanationRequest) -> PredictionRecord:
        decision = self.head.explain_one(self._vector(request), request.case_id)
        return PredictionRecord(
            label=(
                self.positive_label
                if decision.probability >= self.threshold else self.negative_label
            ),
            probability=decision.probability,
            threshold=self.threshold,
            metadata={
                "prediction_space": "logit",
                "additive_logit": decision.logit,
                "reconstruction_error": decision.reconstruction_error,
            },
        )

    def explain(
        self,
        request: ExplanationRequest,
        _prediction: PredictionRecord,
    ) -> ExplanationFragment:
        decision = self.head.explain_one(self._vector(request), request.case_id)
        evidence_map = request.payload.get("concept_evidence", {})
        return decision.to_fragment(evidence_map=evidence_map)
