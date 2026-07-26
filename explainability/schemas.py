"""
HyperDecept-WB M1: Unified white-box dataclass schemas.

All structures are plain Python dataclasses with to_dict() and to_json()
serialisation.  No external schema library is required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_value(value: Any) -> Any:
    """Recursively convert numpy scalars and other non-JSON types to plain Python.

    Handles: dict, list, tuple, set, numpy scalar, ndarray, Path.
    """
    if isinstance(value, dict):
        return {k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _safe_value(value.tolist())
    if isinstance(value, (Path,)):
        return str(value)
    return value


def _safe_dict(data: dict) -> dict:
    """Recursively make every value in *data* JSON-safe.

    Delegates to _safe_value which handles recursion for nested structures.
    """
    return _safe_value(data)


# ---------------------------------------------------------------------------
# JSON serialisation mixin
# ---------------------------------------------------------------------------

class _JSONMixin:
    """Adds to_json() to any dataclass that also provides to_dict()."""

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Return JSON string; optionally write to *path*."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)  # type: ignore[attr-defined]
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text


# ---------------------------------------------------------------------------
# Low-level records
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord(_JSONMixin):
    """A single observable evidence row from DB or CSV."""

    evidence_id: str                    # e.g. "post:103"
    evidence_type: str                  # "post", "follow", "like", ...
    actor_id: Optional[str] = None      # user who performed the action
    target_id: Optional[str] = None     # user / post / comment targeted
    content: Optional[str] = None       # raw text (truncated if very long)
    timestamp: Optional[str] = None     # ISO-8601 or raw DB value
    source_table: Optional[str] = None  # which DB table or CSV column
    source_row_id: Optional[int] = None # row id in source
    observed: bool = True               # always True for M1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


@dataclass
class ConceptRecord(_JSONMixin):
    """Concept-bottleneck slot (reserved for M3, empty in M1)."""

    concept_name: str = ""
    concept_value: float = 0.0
    activation: Optional[float] = None
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


@dataclass
class ContributionRecord(_JSONMixin):
    """Single feature contribution from a local explanation."""

    feature_name: str
    feature_value: float
    contribution: float          # SHAP value in log-odds / margin space
    direction: str = ""          # "supporting" | "opposing"
    rank: int = 0                # 1 = largest absolute contribution
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


@dataclass
class HyperbolicRoleRecord(_JSONMixin):
    """Hyperbolic embedding role and geometry-fidelity diagnostics."""

    role_label: Optional[str] = None
    poincare_coords: List[float] = field(default_factory=list)
    role_confidence: Optional[float] = None
    poincare_radius: Optional[float] = None
    geodesic_fidelity: Optional[float] = None
    radial_order_fidelity: Optional[float] = None
    role_fidelity: Optional[float] = None
    geometry_backend: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


@dataclass
class SubgraphRecord(_JSONMixin):
    """Critical subgraph (reserved for M4, empty in M1)."""

    nodes: List[str] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


@dataclass
class CounterfactualRecord(_JSONMixin):
    """Counterfactual explanation (reserved for M4, empty in M1)."""

    original_prediction: Optional[float] = None
    counterfactual_prediction: Optional[float] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


# ---------------------------------------------------------------------------
# Prediction record
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord(_JSONMixin):
    """Classification result for one case."""

    label: str = ""                     # "coordinated_deception" | "normal"
    probability: float = 0.0
    threshold: float = 0.5
    calibrated_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _safe_dict(asdict(self))


# ---------------------------------------------------------------------------
# Top-level packet
# ---------------------------------------------------------------------------

@dataclass
class ExplanationPacket:
    """M1 white-box explanation for a single user / case."""

    schema_version: str = "1.0"
    case_id: str = ""
    run_id: str = ""
    model_version: str = ""
    prediction: PredictionRecord = field(default_factory=PredictionRecord)
    contributions: List[ContributionRecord] = field(default_factory=list)
    concepts: List[ConceptRecord] = field(default_factory=list)
    hyperbolic_role: Optional[HyperbolicRoleRecord] = None
    critical_subgraph: SubgraphRecord = field(default_factory=SubgraphRecord)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    counterfactuals: List[CounterfactualRecord] = field(default_factory=list)
    uncertainty: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        payload: dict = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "model_version": self.model_version,
            "prediction": self.prediction.to_dict(),
            "contributions": [c.to_dict() for c in self.contributions],
            "concepts": [c.to_dict() for c in self.concepts],
            "hyperbolic_role": (
                self.hyperbolic_role.to_dict() if self.hyperbolic_role else None
            ),
            "critical_subgraph": self.critical_subgraph.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "uncertainty": self.uncertainty,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
        return _safe_dict(payload)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Return JSON string; optionally write to *path*."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text
