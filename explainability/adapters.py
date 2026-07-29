"""Framework-neutral adapters and orchestration for HyperTrace explanations.

The UI and real-time trace layer consume :class:`ExplanationPacket` only.
Model-specific objects stay behind these small interfaces, allowing XGBoost,
Lorentz-HGT and future predictors/explainers to coexist without UI changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .evidence_registry import EvidenceRegistry
from .schemas import (
    ConceptRecord,
    ContributionRecord,
    CounterfactualRecord,
    EvidenceRecord,
    ExplanationPacket,
    HyperbolicRoleRecord,
    PredictionRecord,
    SubgraphRecord,
)


@dataclass(frozen=True)
class ExplanationRequest:
    """One framework-neutral explanation request."""

    case_id: str
    payload: Dict[str, Any]
    run_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise ValueError("case_id must be non-empty")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")


@dataclass
class ExplanationFragment:
    """Partial output returned by one explainer adapter."""

    contributions: List[ContributionRecord] = field(default_factory=list)
    concepts: List[ConceptRecord] = field(default_factory=list)
    hyperbolic_role: Optional[HyperbolicRoleRecord] = None
    critical_subgraph: Optional[SubgraphRecord] = None
    counterfactuals: List[CounterfactualRecord] = field(default_factory=list)
    uncertainty: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictorAdapter(Protocol):
    adapter_id: str
    model_version: str

    def predict(self, request: ExplanationRequest) -> PredictionRecord:
        ...


class ExplainerAdapter(Protocol):
    adapter_id: str

    def explain(
        self,
        request: ExplanationRequest,
        prediction: PredictionRecord,
    ) -> ExplanationFragment:
        ...


class EvidenceProvider(Protocol):
    provider_id: str

    def resolve(self, evidence_ids: Iterable[str]) -> List[EvidenceRecord]:
        ...


class AdapterRegistry:
    """Explicit registry; duplicate IDs are rejected instead of overwritten."""

    def __init__(self) -> None:
        self._predictors: Dict[str, PredictorAdapter] = {}
        self._explainers: Dict[str, ExplainerAdapter] = {}
        self._evidence_providers: Dict[str, EvidenceProvider] = {}

    @staticmethod
    def _register(target: dict, adapter_id: str, adapter: Any) -> None:
        key = str(adapter_id).strip()
        if not key:
            raise ValueError("adapter_id/provider_id must be non-empty")
        if key in target:
            raise ValueError(f"Adapter already registered: {key}")
        target[key] = adapter

    def register_predictor(self, adapter: PredictorAdapter) -> None:
        self._register(self._predictors, adapter.adapter_id, adapter)

    def register_explainer(self, adapter: ExplainerAdapter) -> None:
        self._register(self._explainers, adapter.adapter_id, adapter)

    def register_evidence_provider(self, provider: EvidenceProvider) -> None:
        self._register(
            self._evidence_providers, provider.provider_id, provider
        )

    def predictor(self, adapter_id: str) -> PredictorAdapter:
        try:
            return self._predictors[adapter_id]
        except KeyError as error:
            raise KeyError(f"Unknown predictor adapter: {adapter_id}") from error

    def explainer(self, adapter_id: str) -> ExplainerAdapter:
        try:
            return self._explainers[adapter_id]
        except KeyError as error:
            raise KeyError(f"Unknown explainer adapter: {adapter_id}") from error

    @property
    def evidence_providers(self) -> List[EvidenceProvider]:
        return list(self._evidence_providers.values())

    def capabilities(self) -> dict:
        return {
            "predictors": sorted(self._predictors),
            "explainers": sorted(self._explainers),
            "evidence_providers": sorted(self._evidence_providers),
        }


class RegistryEvidenceProvider:
    """Expose the existing EvidenceRegistry through the unified protocol."""

    def __init__(
        self,
        registry: EvidenceRegistry,
        provider_id: str = "evidence_registry",
    ) -> None:
        self.registry = registry
        self.provider_id = provider_id

    def resolve(self, evidence_ids: Iterable[str]) -> List[EvidenceRecord]:
        records = []
        seen = set()
        for evidence_id in evidence_ids:
            key = str(evidence_id)
            if key in seen:
                continue
            seen.add(key)
            record = self.registry.get(key)
            if record is not None:
                records.append(record)
        return records


class FunctionPredictorAdapter:
    """Small bridge for existing predictors without modifying their classes."""

    def __init__(
        self,
        adapter_id: str,
        model_version: str,
        predict_fn: Callable[[ExplanationRequest], PredictionRecord],
    ) -> None:
        self.adapter_id = adapter_id
        self.model_version = model_version
        self._predict_fn = predict_fn

    def predict(self, request: ExplanationRequest) -> PredictionRecord:
        prediction = self._predict_fn(request)
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("predict_fn must return PredictionRecord")
        return prediction


class FunctionExplainerAdapter:
    """Small bridge for PG, Geo-PG, SHAP or future explainers."""

    def __init__(
        self,
        adapter_id: str,
        explain_fn: Callable[
            [ExplanationRequest, PredictionRecord], ExplanationFragment
        ],
    ) -> None:
        self.adapter_id = adapter_id
        self._explain_fn = explain_fn

    def explain(
        self,
        request: ExplanationRequest,
        prediction: PredictionRecord,
    ) -> ExplanationFragment:
        fragment = self._explain_fn(request, prediction)
        if not isinstance(fragment, ExplanationFragment):
            raise TypeError("explain_fn must return ExplanationFragment")
        return fragment


class TabularPredictorAdapter:
    """Concrete adapter for fitted sklearn/XGBoost-style binary classifiers."""

    def __init__(
        self,
        fitted_model: Any,
        feature_names: Sequence[str],
        *,
        adapter_id: str = "tabular_binary_classifier",
        model_version: str = "",
        threshold: float = 0.5,
        positive_label: str = "coordinated_deception",
        negative_label: str = "normal",
    ) -> None:
        if not hasattr(fitted_model, "predict_proba"):
            raise TypeError("fitted_model must implement predict_proba")
        self.model = fitted_model
        self.feature_names = list(map(str, feature_names))
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        self.adapter_id = adapter_id
        self.model_version = model_version
        self.threshold = float(threshold)
        self.positive_label = positive_label
        self.negative_label = negative_label

    def _vector(self, request: ExplanationRequest) -> np.ndarray:
        values = request.payload.get("features")
        if isinstance(values, dict):
            missing = [name for name in self.feature_names if name not in values]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            vector = [values[name] for name in self.feature_names]
        elif isinstance(values, (list, tuple, np.ndarray)):
            vector = values
        else:
            raise ValueError("request payload must contain features mapping/list")
        array = np.asarray(vector, dtype=float).reshape(1, -1)
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {array.shape[1]}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Feature values must be finite")
        return array

    def predict(self, request: ExplanationRequest) -> PredictionRecord:
        output = np.asarray(self.model.predict_proba(self._vector(request)))
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError(f"Unexpected predict_proba shape: {output.shape}")
        if output.shape[1] == 2:
            probability = float(output[0, 1])
        elif output.shape[1] == 1:
            probability = float(output[0, 0])
        else:
            raise ValueError("Binary classifier predict_proba must have 1 or 2 columns")
        return PredictionRecord(
            label=(
                self.positive_label
                if probability >= self.threshold else self.negative_label
            ),
            probability=probability,
            threshold=self.threshold,
            metadata={"feature_names": self.feature_names},
        )


class LocalExplanationAdapter:
    """Bridge an existing LocalTreeExplainer into the unified fragment API."""

    def __init__(
        self,
        local_explainer: Any,
        *,
        provenance: Optional[Dict[str, Any]] = None,
        adapter_id: str = "local_tree_explainer",
    ) -> None:
        if not hasattr(local_explainer, "explain_one"):
            raise TypeError("local_explainer must implement explain_one(user_id)")
        self.local_explainer = local_explainer
        self.provenance = provenance or {}
        self.adapter_id = adapter_id

    @staticmethod
    def _user_id(request: ExplanationRequest) -> str:
        explicit = request.payload.get("user_id")
        if explicit is not None:
            return str(explicit)
        case_id = str(request.case_id)
        return case_id.split(":", maxsplit=1)[1] if case_id.startswith("user:") else case_id

    def explain(
        self,
        request: ExplanationRequest,
        _prediction: PredictionRecord,
    ) -> ExplanationFragment:
        user_id = self._user_id(request)
        local = self.local_explainer.explain_one(user_id)
        user_provenance = self.provenance.get(user_id, {})
        contributions = []
        for contribution in local.contributions:
            provenance = user_provenance.get(contribution.feature_name, {})
            linked = list(map(str, contribution.evidence_ids))
            if isinstance(provenance, dict):
                linked.extend(map(str, provenance.get("evidence_ids", [])))
            metadata = dict(contribution.metadata)
            if isinstance(provenance, dict):
                metadata.update({
                    "extractor": provenance.get("extractor", ""),
                    "text_indices": provenance.get("text_indices", []),
                })
            contributions.append(replace(
                contribution,
                evidence_ids=list(dict.fromkeys(linked)),
                metadata=metadata,
            ))
        return ExplanationFragment(
            contributions=contributions,
            evidence_ids=[
                evidence_id
                for contribution in contributions
                for evidence_id in contribution.evidence_ids
            ],
            metadata={
                "base_value": local.base_value,
                "reconstruction_error": local.reconstruction_error,
                "local_explanation_metadata": local.metadata,
            },
        )


class ExplanationOrchestrator:
    """Compose registered model components into one ExplanationPacket."""

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        top_k: Optional[int] = None,
        fail_fast: bool = False,
    ) -> None:
        self.registry = registry
        self.top_k = top_k
        self.fail_fast = fail_fast

    def predict(
        self,
        request: ExplanationRequest,
        predictor_id: str,
    ) -> PredictionRecord:
        prediction = self.registry.predictor(predictor_id).predict(request)
        if not 0.0 <= float(prediction.probability) <= 1.0:
            raise ValueError("Prediction probability must be in [0, 1]")
        return prediction

    @staticmethod
    def _fragment_evidence_ids(fragment: ExplanationFragment) -> List[str]:
        evidence_ids = list(map(str, fragment.evidence_ids))
        for contribution in fragment.contributions:
            evidence_ids.extend(map(str, contribution.evidence_ids))
        for concept in fragment.concepts:
            evidence_ids.extend(map(str, concept.evidence_ids))
        if fragment.critical_subgraph is not None:
            for edge in fragment.critical_subgraph.edges:
                value = edge.get("evidence_ids", edge.get("evidence_id", []))
                if isinstance(value, (str, int)):
                    evidence_ids.append(str(value))
                elif isinstance(value, list):
                    evidence_ids.extend(map(str, value))
        return evidence_ids

    def _resolve_evidence(
        self, evidence_ids: Sequence[str]
    ) -> tuple[List[EvidenceRecord], List[str]]:
        requested = list(dict.fromkeys(map(str, evidence_ids)))
        resolved: Dict[str, EvidenceRecord] = {}
        for provider in self.registry.evidence_providers:
            for record in provider.resolve(requested):
                resolved.setdefault(record.evidence_id, record)
        missing = [item for item in requested if item not in resolved]
        return list(resolved.values()), missing

    def explain_prediction(
        self,
        request: ExplanationRequest,
        prediction: PredictionRecord,
        predictor_id: str,
        explainer_ids: Sequence[str],
    ) -> ExplanationPacket:
        contributions: List[ContributionRecord] = []
        concepts: List[ConceptRecord] = []
        counterfactuals: List[CounterfactualRecord] = []
        uncertainty: List[Dict[str, Any]] = []
        warnings: List[str] = []
        evidence_ids: List[str] = []
        role: Optional[HyperbolicRoleRecord] = None
        subgraph = SubgraphRecord()
        fragment_metadata: Dict[str, Any] = {}

        used_explainers = []
        for explainer_id in explainer_ids:
            adapter = self.registry.explainer(explainer_id)
            try:
                fragment = adapter.explain(request, prediction)
            except Exception as error:
                if self.fail_fast:
                    raise
                warnings.append(
                    f"Explainer '{explainer_id}' failed: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            used_explainers.append(explainer_id)
            fragment_metadata[explainer_id] = dict(fragment.metadata)
            for contribution in fragment.contributions:
                metadata = dict(contribution.metadata)
                metadata.setdefault("explainer_id", explainer_id)
                contributions.append(replace(contribution, metadata=metadata))
            for concept in fragment.concepts:
                metadata = dict(concept.metadata)
                metadata.setdefault("explainer_id", explainer_id)
                concepts.append(replace(concept, metadata=metadata))
            counterfactuals.extend(fragment.counterfactuals)
            uncertainty.extend(fragment.uncertainty)
            warnings.extend(fragment.warnings)
            evidence_ids.extend(self._fragment_evidence_ids(fragment))
            if role is None and fragment.hyperbolic_role is not None:
                role = fragment.hyperbolic_role
            if fragment.critical_subgraph is not None:
                subgraph.nodes.extend(fragment.critical_subgraph.nodes)
                subgraph.edges.extend(fragment.critical_subgraph.edges)
                subgraph.paths.extend(fragment.critical_subgraph.paths)
                subgraph.metadata.update(fragment.critical_subgraph.metadata)

        contributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        if self.top_k is not None:
            contributions = contributions[: self.top_k]
        contributions = [
            replace(item, rank=index)
            for index, item in enumerate(contributions, start=1)
        ]
        evidence, missing = self._resolve_evidence(evidence_ids)
        if missing:
            warnings.append(
                "Unresolved evidence IDs: " + ", ".join(missing[:20])
                + (" ..." if len(missing) > 20 else "")
            )

        predictor = self.registry.predictor(predictor_id)
        return ExplanationPacket(
            schema_version="1.1",
            case_id=request.case_id,
            run_id=request.run_id,
            model_version=predictor.model_version,
            prediction=prediction,
            contributions=contributions,
            concepts=concepts,
            hyperbolic_role=role,
            critical_subgraph=SubgraphRecord(
                nodes=list(dict.fromkeys(map(str, subgraph.nodes))),
                edges=subgraph.edges,
                paths=subgraph.paths,
                metadata=subgraph.metadata,
            ),
            evidence=evidence,
            counterfactuals=counterfactuals,
            uncertainty=uncertainty,
            warnings=list(dict.fromkeys(warnings)),
            metadata={
                **request.metadata,
                "predictor_id": predictor_id,
                "explainer_ids": used_explainers,
                "fragment_metadata": fragment_metadata,
                "adapter_capabilities": self.registry.capabilities(),
            },
        )

    def explain(
        self,
        request: ExplanationRequest,
        predictor_id: str,
        explainer_ids: Sequence[str],
    ) -> ExplanationPacket:
        prediction = self.predict(request, predictor_id)
        return self.explain_prediction(
            request, prediction, predictor_id, explainer_ids
        )
