"""
Tests for explainability.packet_builder.

Uses in-memory EvidenceRegistry and synthetic explanations --
no external models needed for the core logic.
"""

import json
import os
import tempfile

import pytest

# These are pure Python, always importable.
from explainability.evidence_registry import EvidenceRegistry
from explainability.packet_builder import ExplanationPacketBuilder
from explainability.local_explainer import LocalExplanation
from explainability.schemas import (
    ContributionRecord,
    EvidenceRecord,
    ExplanationPacket,
    PredictionRecord,
)


# ------------------------------------------------------------------ helpers

def _fake_explanation(user_id="17", prob=0.87, label="coordinated_deception"):
    """Build a minimal LocalExplanation with a few contributions."""
    contribs = [
        ContributionRecord(
            feature_name="Empathy_Gap_Mean", feature_value=0.63,
            contribution=0.45, direction="supporting", rank=1,
            evidence_ids=["text:17:4"],
        ),
        ContributionRecord(
            feature_name="Dark_Triad_Max", feature_value=0.71,
            contribution=0.32, direction="supporting", rank=2,
            evidence_ids=["text:17:7"],
        ),
        ContributionRecord(
            feature_name="Semantic_3", feature_value=1.20,
            contribution=0.18, direction="supporting", rank=3,
            evidence_ids=[],  # <-- no evidence source!
        ),
        ContributionRecord(
            feature_name="Follower_Following_Ratio", feature_value=0.05,
            contribution=-0.10, direction="opposing", rank=4,
            evidence_ids=[],
        ),
    ]
    return LocalExplanation(
        user_id=user_id,
        probability=prob,
        predicted_label=label,
        threshold=0.5,
        base_value=0.02,
        contributions=contribs,
        reconstruction_error=0.001,
    )


def _fake_registry():
    reg = EvidenceRegistry()
    # Manually insert evidence records that match the explanation
    reg._records["text:17:4"] = EvidenceRecord(
        evidence_id="text:17:4",
        evidence_type="text_synthetic",
        actor_id="17",
        content="sample tweet showing empathy gap",
        source_table="synthetic",
    )
    reg._records["text:17:7"] = EvidenceRecord(
        evidence_id="text:17:7",
        evidence_type="text_synthetic",
        actor_id="17",
        content="sample tweet showing dark triad traits",
        source_table="synthetic",
    )
    return reg


def _fake_provenance():
    return {
        "17": {
            "Empathy_Gap_Mean": {
                "value": 0.63,
                "extractor": "EmpathyGapAnalyzer",
                "evidence_ids": ["text:17:4"],
                "text_indices": [4],
                "metadata": {},
            },
            "Dark_Triad_Max": {
                "value": 0.71,
                "extractor": "DarkTriadAnalyzer",
                "evidence_ids": ["text:17:7"],
                "text_indices": [7],
                "metadata": {},
            },
        },
    }


# ------------------------------------------------------------------ tests


class TestPacketBuilderBasic:
    def test_build_packet_structure(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(
            registry=reg,
            run_id="test_run",
            model_version="v0",
            top_k=10,
        )
        exp = _fake_explanation()
        prov = _fake_provenance()

        pkt = builder.build(exp, provenance=prov)

        assert isinstance(pkt, ExplanationPacket)
        assert pkt.schema_version == "1.0"
        assert pkt.case_id == "user:17"
        assert pkt.run_id == "test_run"
        assert pkt.model_version == "v0"

    def test_prediction_in_packet(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation(prob=0.92, label="coordinated_deception")

        pkt = builder.build(exp)
        assert pkt.prediction.probability == 0.92
        assert pkt.prediction.label == "coordinated_deception"
        assert pkt.prediction.threshold == 0.5

    def test_top_k_truncation(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg, top_k=2)
        exp = _fake_explanation()  # has 4 contributions

        pkt = builder.build(exp)
        assert len(pkt.contributions) == 2

    def test_evidence_linking(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()
        prov = _fake_provenance()

        pkt = builder.build(exp, provenance=prov)

        # Empathy_Gap_Mean should have evidence linked from provenance
        emp_contrib = next(
            c for c in pkt.contributions
            if c.feature_name == "Empathy_Gap_Mean"
        )
        assert "text:17:4" in emp_contrib.evidence_ids

        # The evidence records should appear in the packet
        ev_ids_in_packet = {e.evidence_id for e in pkt.evidence}
        assert "text:17:4" in ev_ids_in_packet
        assert "text:17:7" in ev_ids_in_packet

    def test_no_evidence_warning(self):
        """High-contribution features without evidence must produce a warning."""
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()

        pkt = builder.build(exp)
        assert len(pkt.warnings) > 0
        # Semantic_3 has contribution 0.18 and no evidence
        assert any("Semantic_3" in w for w in pkt.warnings)

    def test_low_contribution_no_warning(self):
        """Features with very small abs contribution should not warn even
        without evidence."""
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = LocalExplanation(
            user_id="1",
            probability=0.5,
            predicted_label="normal",
            threshold=0.5,
            base_value=0.0,
            contributions=[
                ContributionRecord(
                    feature_name="Tiny_Feature", feature_value=0.0,
                    contribution=0.0001, direction="supporting", rank=1,
                    evidence_ids=[],
                ),
            ],
        )
        pkt = builder.build(exp)
        # Tiny_Feature < 0.01 threshold, should not warn
        assert not any("Tiny_Feature" in w for w in pkt.warnings)

    def test_empty_structures_preserved(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()

        pkt = builder.build(exp)
        assert pkt.concepts == []
        assert pkt.counterfactuals == []
        assert pkt.hyperbolic_role is None
        assert pkt.critical_subgraph.nodes == []

    def test_role_info_populates_hyperbolic_record(self):
        builder = ExplanationPacketBuilder(_fake_registry())
        pkt = builder.build(_fake_explanation(), role_info={
            "role": "Opinion Leader",
            "poincare_coords": [0.1, -0.2],
            "poincare_radius": 0.22,
            "geodesic_fidelity": 0.9,
            "radial_order_fidelity": 0.8,
            "role_fidelity": 1.0,
            "geometry_backend": "projection_head",
        })
        assert pkt.hyperbolic_role is not None
        assert pkt.hyperbolic_role.role_label == "Opinion Leader"
        assert pkt.hyperbolic_role.geodesic_fidelity == 0.9
        assert "role_info" not in pkt.metadata

    def test_json_serializable(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()
        pkt = builder.build(exp)
        text = pkt.to_json()
        parsed = json.loads(text)
        assert parsed["schema_version"] == "1.0"

    def test_save_packet_to_file(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()
        pkt = builder.build(exp)

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "packet.json")
        builder.save_packet(pkt, path)
        assert os.path.isfile(path)

        with open(path, "r") as fh:
            data = json.load(fh)
        assert data["case_id"] == "user:17"

    def test_build_all(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exps = [
            _fake_explanation(user_id="1"),
            _fake_explanation(user_id="2"),
        ]
        packets = builder.build_all(exps)
        assert len(packets) == 2
        assert packets[0].case_id == "user:1"
        assert packets[1].case_id == "user:2"

    def test_save_packets_jsonl(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exps = [_fake_explanation(user_id="1"), _fake_explanation(user_id="2")]
        packets = builder.build_all(exps)

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "packets.jsonl")
        builder.save_packets_jsonl(packets, path)
        assert os.path.isfile(path)

        with open(path, "r") as fh:
            lines = fh.readlines()
        assert len(lines) == 2


class TestPacketBuilderEdgeCases:
    def test_no_provenance(self):
        """Builder should work without provenance (no evidence linking)."""
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()
        pkt = builder.build(exp, provenance=None)
        assert isinstance(pkt, ExplanationPacket)

    def test_provenance_with_missing_user(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation(user_id="99")
        prov = {"17": {}}  # user 99 not in provenance
        pkt = builder.build(exp, provenance=prov)
        # Should still build without error, just no evidence links
        assert len(pkt.evidence) == 0

    def test_contributions_preserve_direction(self):
        reg = _fake_registry()
        builder = ExplanationPacketBuilder(reg)
        exp = _fake_explanation()
        pkt = builder.build(exp)

        supporting = [c for c in pkt.contributions if c.direction == "supporting"]
        opposing = [c for c in pkt.contributions if c.direction == "opposing"]
        assert len(supporting) > 0
        assert len(opposing) > 0
