"""
Tests for explainability.schemas -- dataclass JSON serialisation.

These tests use only the standard library; no external dependencies required.
"""

import json
import tempfile
import os

import numpy as np
import pytest

from explainability.schemas import (
    EvidenceRecord,
    ConceptRecord,
    ContributionRecord,
    HyperbolicRoleRecord,
    SubgraphRecord,
    CounterfactualRecord,
    PredictionRecord,
    ExplanationPacket,
    _safe_value,
    _safe_dict,
)


# ------------------------------------------------------------------ helpers

def _assert_json_roundtrip(obj, tmp_path):
    """Verify that obj.to_json / obj.to_dict produce valid JSON."""
    # to_dict
    payload = obj.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert isinstance(text, str)
    assert len(text) > 2

    # to_json to string
    text2 = obj.to_json(indent=2)
    assert isinstance(text2, str)

    # to_json to file
    fpath = os.path.join(tmp_path, "test.json")
    obj.to_json(path=fpath)
    with open(fpath, "r", encoding="utf-8") as fh:
        assert json.load(fh)


# ------------------------------------------------------------------ tests


class TestNumpySafeValue:
    def test_int(self):
        assert _safe_value(np.int64(42)) == 42
        assert isinstance(_safe_value(np.int64(42)), int)

    def test_float(self):
        v = _safe_value(np.float64(3.14))
        assert isinstance(v, float)
        assert abs(v - 3.14) < 1e-10

    def test_float32(self):
        v = _safe_value(np.float32(0.87))
        assert isinstance(v, float)
        assert not isinstance(v, np.floating)

    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        result = _safe_value(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        d = {"a": np.float32(1.5), "b": [np.int32(7)]}
        out = _safe_dict(d)
        assert isinstance(out["a"], float)
        assert isinstance(out["b"][0], int)

    def test_list_of_dicts_with_float32(self):
        """list -> dict -> np.float32 must be fully converted."""
        nested = [{"score": np.float32(0.87)}, {"score": np.float32(0.92)}]
        out = _safe_value(nested)
        assert isinstance(out, list)
        for item in out:
            assert isinstance(item, dict)
            assert isinstance(item["score"], float)
            assert not isinstance(item["score"], np.floating)

    def test_recursive_nested(self):
        """Deeply nested structure with tuple, set, ndarray."""
        data = {
            "a": [1, np.float32(2.5), {"b": np.int64(3)}],
            "c": (4, np.float64(5.5)),
            "d": np.array([[1, 2], [3, 4]]),
        }
        out = _safe_value(data)
        # top level
        assert isinstance(out["a"][1], float)
        assert isinstance(out["a"][2]["b"], int)
        assert isinstance(out["c"], list)  # tuple -> list
        assert isinstance(out["d"], list)  # ndarray -> list

    def test_bool(self):
        v = _safe_value(np.bool_(True))
        assert isinstance(v, bool)
        assert v is True

    def test_path(self):
        from pathlib import Path
        v = _safe_value(Path("/tmp/test"))
        assert isinstance(v, str)
        assert v.replace("\\", "/").endswith("/tmp/test")


class TestEvidenceRecord:
    def test_basic_roundtrip(self, tmp_path):
        r = EvidenceRecord(
            evidence_id="post:17",
            evidence_type="post",
            actor_id="17",
            target_id=None,
            content="hello world",
            timestamp="2025-01-01",
            source_table="post",
            source_row_id=1,
        )
        _assert_json_roundtrip(r, tmp_path)
        d = r.to_dict()
        assert d["evidence_id"] == "post:17"

    def test_defaults(self):
        r = EvidenceRecord(evidence_id="e:1", evidence_type="generic")
        assert r.observed is True
        assert r.metadata == {}
        assert r.actor_id is None


class TestContributionRecord:
    def test_roundtrip(self, tmp_path):
        c = ContributionRecord(
            feature_name="Empathy_Gap_Mean",
            feature_value=0.87,
            contribution=0.123,
            direction="supporting",
            rank=1,
            evidence_ids=["post:103", "post:107"],
        )
        _assert_json_roundtrip(c, tmp_path)
        d = c.to_dict()
        assert d["direction"] == "supporting"


class TestExplanationPacket:
    def test_minimal_packet(self, tmp_path):
        pkt = ExplanationPacket(
            schema_version="1.0",
            case_id="user:17",
            run_id="run_001",
            model_version="v1",
            prediction=PredictionRecord(
                label="coordinated_deception",
                probability=0.87,
                threshold=0.5,
            ),
        )
        _assert_json_roundtrip(pkt, tmp_path)
        d = pkt.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["case_id"] == "user:17"
        assert d["critical_subgraph"]["nodes"] == []
        assert d["critical_subgraph"]["edges"] == []
        assert d["hyperbolic_role"] is None

    def test_full_packet_structure(self, tmp_path):
        """Verify all reserved slots appear in the dict."""
        pkt = ExplanationPacket(
            case_id="user:42",
            contributions=[
                ContributionRecord(
                    feature_name="F1", feature_value=0.5,
                    contribution=0.1, direction="supporting", rank=1,
                )
            ],
            evidence=[
                EvidenceRecord(evidence_id="post:1", evidence_type="post")
            ],
            warnings=["No evidence for F2"],
        )
        d = pkt.to_dict()
        required_keys = [
            "schema_version", "case_id", "run_id", "model_version",
            "prediction", "contributions", "concepts",
            "hyperbolic_role", "critical_subgraph", "evidence",
            "counterfactuals", "uncertainty", "warnings", "metadata",
        ]
        for k in required_keys:
            assert k in d, f"Missing key: {k}"

    def test_numpy_in_contributions(self, tmp_path):
        """Contributions with numpy values must serialise."""
        c = ContributionRecord(
            feature_name="F",
            feature_value=np.float32(0.99),
            contribution=np.float64(-0.05),
            direction="opposing",
            rank=3,
        )
        pkt = ExplanationPacket(
            case_id="user:1",
            contributions=[c],
        )
        text = pkt.to_json()
        parsed = json.loads(text)
        assert isinstance(parsed["contributions"][0]["feature_value"], float)

    def test_empty_structures(self):
        """Reserved fields default to empty containers, not None."""
        pkt = ExplanationPacket()
        d = pkt.to_dict()
        assert d["contributions"] == []
        assert d["evidence"] == []
        assert d["counterfactuals"] == []
        assert d["warnings"] == []
        assert d["critical_subgraph"]["nodes"] == []
        assert d["critical_subgraph"]["paths"] == []


class TestReservedRecords:
    def test_subgraph_record(self, tmp_path):
        s = SubgraphRecord(
            nodes=["u1", "u2"],
            edges=[{"source": "u1", "target": "u2"}],
            paths=[["u1", "u2"]],
        )
        _assert_json_roundtrip(s, tmp_path)

    def test_counterfactual_record(self, tmp_path):
        c = CounterfactualRecord(
            original_prediction=0.9,
            counterfactual_prediction=0.3,
            changes=[{"feature": "F1", "from": 0.5, "to": 0.1}],
        )
        _assert_json_roundtrip(c, tmp_path)

    def test_hyperbolic_role_record(self, tmp_path):
        h = HyperbolicRoleRecord(
            role_label="core_bridge",
            poincare_coords=[0.1, -0.3],
            role_confidence=0.85,
            poincare_radius=0.32,
            geodesic_fidelity=0.91,
            radial_order_fidelity=0.88,
            role_fidelity=1.0,
            geometry_backend="projection_head",
        )
        _assert_json_roundtrip(h, tmp_path)
