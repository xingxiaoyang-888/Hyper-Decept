"""
HyperDecept-WB M1: ExplanationPacket builder.

Assembles a single :class:`ExplanationPacket` per user by joining:
- local SHAP contributions
- per-feature provenance (text indices, evidence ids)
- :class:`EvidenceRegistry` look-ups
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence_registry import EvidenceRegistry
from .local_explainer import LocalExplanation
from .schemas import (
    ContributionRecord,
    EvidenceRecord,
    ExplanationPacket,
    PredictionRecord,
    SubgraphRecord,
    HyperbolicRoleRecord,
)

logger = logging.getLogger(__name__)


class ExplanationPacketBuilder:
    """Builds M1 :class:`ExplanationPacket` objects.

    Parameters
    ----------
    registry : EvidenceRegistry
        Pre-populated evidence store.
    run_id : str
        Opaque identifier for this experiment run.
    model_version : str
        Human-readable model version string.
    top_k : int
        How many top (by absolute SHAP) features to include per packet.
    """

    def __init__(
        self,
        registry: EvidenceRegistry,
        run_id: str = "",
        model_version: str = "",
        top_k: int = 15,
    ) -> None:
        self.registry = registry
        self.run_id = run_id
        self.model_version = model_version
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Build one packet
    # ------------------------------------------------------------------

    def build(
        self,
        explanation: LocalExplanation,
        provenance: Optional[Dict[str, Any]] = None,
        role_info: Optional[Dict[str, Any]] = None,
        oof_probability: Optional[float] = None,
    ) -> ExplanationPacket:
        """Construct an ExplanationPacket for a single user.

        Parameters
        ----------
        explanation : LocalExplanation
            Per-user SHAP explanation from :class:`LocalTreeExplainer`.
        provenance : dict or None
            Per-feature provenance map for this user (keyed by feature_name).
        role_info : dict or None
            Optional role metadata (reserved for M2+).
        """
        user_prov: Dict[str, Any] = {}
        if provenance is not None:
            user_prov = provenance.get(str(explanation.user_id), {})

        # -- Prediction ----------------------------------------------------
        pkt_metadata: Dict[str, Any] = {
            "threshold": explanation.threshold,
            "base_value": explanation.base_value,
            "reconstruction_error": explanation.reconstruction_error,
            "top_k": self.top_k,
            "total_features": len(explanation.contributions),
            "prediction_scope": "full_fit_model",
        }
        if oof_probability is not None:
            pkt_metadata["evaluation_probability_oof"] = float(oof_probability)

        prediction = PredictionRecord(
            label=explanation.predicted_label,
            probability=explanation.probability,
            threshold=explanation.threshold,
            calibrated_confidence=None,
        )

        # -- Contributions (top-k by abs SHAP) ----------------------------
        warnings: List[str] = []
        selected: List[ContributionRecord] = []
        ev_records: Dict[str, EvidenceRecord] = {}

        for contrib in explanation.contributions[:self.top_k]:
            fname = contrib.feature_name
            prov = user_prov.get(fname, {})

            # Link evidence ids from provenance
            evidence_ids: List[str] = []
            if isinstance(prov, dict):
                eids = prov.get("evidence_ids", [])
                if isinstance(eids, list):
                    evidence_ids = [str(e) for e in eids]

            # Warn on evidence_ids that cannot be resolved in registry
            resolved_ids: List[str] = []
            for eid in evidence_ids:
                if eid in ev_records:
                    resolved_ids.append(eid)
                    continue
                ev = self.registry.get(eid)
                if ev is not None:
                    ev_records[eid] = ev
                    resolved_ids.append(eid)
                else:
                    warnings.append(
                        f"Evidence id '{eid}' for feature '{fname}' "
                        f"(rank={contrib.rank}) is not in the registry."
                    )

            # Create a copy with linked evidence
            rec = ContributionRecord(
                feature_name=contrib.feature_name,
                feature_value=contrib.feature_value,
                contribution=contrib.contribution,
                direction=contrib.direction,
                rank=contrib.rank,
                evidence_ids=resolved_ids,
                metadata={
                    "extractor": prov.get("extractor", "") if isinstance(prov, dict) else "",
                    "text_indices": prov.get("text_indices", []) if isinstance(prov, dict) else [],
                },
            )
            selected.append(rec)

            # Warn on high-contribution features without evidence at all
            if abs(contrib.contribution) >= 0.01 and not resolved_ids:
                warnings.append(
                    f"Feature '{fname}' (rank={contrib.rank}, "
                    f"contribution={contrib.contribution:.4f}) has no evidence source."
                )

        # -- Assemble packet -----------------------------------------------
        hyperbolic_role = None
        if role_info is not None:
            coords = role_info.get("poincare_coords", [])
            hyperbolic_role = HyperbolicRoleRecord(
                role_label=role_info.get("role_label", role_info.get("role")),
                poincare_coords=list(coords) if coords is not None else [],
                role_confidence=role_info.get("role_confidence"),
                poincare_radius=role_info.get("poincare_radius"),
                geodesic_fidelity=role_info.get("geodesic_fidelity"),
                radial_order_fidelity=role_info.get("radial_order_fidelity"),
                role_fidelity=role_info.get("role_fidelity"),
                geometry_backend=role_info.get("geometry_backend"),
                metadata=role_info.get("metadata", {}),
            )

        packet = ExplanationPacket(
            schema_version="1.0",
            case_id=f"user:{explanation.user_id}",
            run_id=self.run_id,
            model_version=self.model_version,
            prediction=prediction,
            contributions=selected,
            concepts=[],
            hyperbolic_role=hyperbolic_role,
            critical_subgraph=SubgraphRecord(),
            evidence=list(ev_records.values()),
            counterfactuals=[],
            uncertainty=[],
            warnings=warnings,
            metadata=pkt_metadata,
        )

        return packet

    # ------------------------------------------------------------------
    # Batch & persistence
    # ------------------------------------------------------------------

    def build_all(
        self,
        explanations: List[LocalExplanation],
        provenance: Optional[Dict[str, Any]] = None,
        role_info: Optional[Dict[str, Any]] = None,
        oof_probabilities: Optional[Dict[str, float]] = None,
    ) -> List[ExplanationPacket]:
        """Build packets for a list of local explanations."""
        oof_map = oof_probabilities or {}
        packets: List[ExplanationPacket] = []
        for exp in explanations:
            oof = oof_map.get(str(exp.user_id))
            p = self.build(exp, provenance=provenance, role_info=role_info,
                           oof_probability=oof)
            packets.append(p)
        return packets

    def save_packet(self, packet: ExplanationPacket, path: str) -> str:
        """Write a single packet as a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        packet.to_json(path=path)
        return path

    def save_packets_jsonl(
        self,
        packets: List[ExplanationPacket],
        path: str,
    ) -> str:
        """Write multiple packets as JSONL (one per line)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for pkt in packets:
                fh.write(pkt.to_json(indent=None) + "\n")
        logger.info("Explanation packets JSONL saved: %s (%d packets)", path, len(packets))
        return path
