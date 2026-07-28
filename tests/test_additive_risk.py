import numpy as np
import pytest

torch = pytest.importorskip("torch")

from explainability.adapters import (
    AdapterRegistry,
    ExplanationOrchestrator,
    ExplanationRequest,
    RegistryEvidenceProvider,
)
from explainability.additive_risk import (
    AdditiveRiskAdapter,
    SparseNeuralAdditiveRiskHead,
)
from explainability.evidence_registry import EvidenceRegistry
from explainability.schemas import EvidenceRecord


def _head():
    torch.manual_seed(2)
    return SparseNeuralAdditiveRiskHead(
        ["burst", "radius", "empathy_gap"],
        interactions=[("burst", "radius")],
        hidden_dim=8,
    )


def test_additive_decomposition_exactly_reconstructs_logit_and_evidence():
    head = _head()
    decision = head.explain_one([0.8, 0.2, 0.6], case_id="u1")
    assert decision.reconstruction_error < 1e-6
    fragment = decision.to_fragment({
        "burst": ["post:1"], "radius": ["follow:1"]
    })
    visible = fragment.metadata["additive_decomposition"]
    assert abs(visible["visible_contribution_sum"] - visible["logit"]) < 1e-6
    interaction = next(
        item for item in fragment.contributions
        if item.metadata["component_type"] == "allowed_interaction"
    )
    assert interaction.evidence_ids == ["post:1", "follow:1"]


def test_additive_head_trains_without_creating_an_internal_random_split():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 3)).astype("float32")
    y = (1.5 * x[:, 0] - 0.9 * x[:, 1] + 0.4 * x[:, 2] > 0).astype("float32")
    head = _head()
    history = head.fit_model(
        x, y, epochs=180, learning_rate=0.01,
        sparsity_strength=1e-4, random_seed=3,
    )
    assert history[-1]["prediction_loss"] < history[0]["prediction_loss"]
    with torch.no_grad():
        predictions = (torch.sigmoid(head(x)) >= 0.5).cpu().numpy()
    assert np.mean(predictions == y) > 0.88


def test_additive_adapter_works_through_unified_orchestrator():
    head = _head()
    adapter = AdditiveRiskAdapter(head, model_version="nam-test")
    adapters = AdapterRegistry()
    adapters.register_predictor(adapter)
    adapters.register_explainer(adapter)
    evidence = EvidenceRegistry()
    evidence._records["post:1"] = EvidenceRecord(
        evidence_id="post:1", evidence_type="post"
    )
    adapters.register_evidence_provider(RegistryEvidenceProvider(evidence))
    request = ExplanationRequest(
        case_id="user:u1",
        run_id="run-nam",
        payload={
            "concepts": {"burst": 0.8, "radius": 0.2, "empathy_gap": 0.6},
            "concept_evidence": {"burst": ["post:1"]},
        },
    )
    packet = ExplanationOrchestrator(adapters).explain(
        request, adapter.adapter_id, [adapter.adapter_id]
    )
    assert packet.model_version == "nam-test"
    assert packet.concepts
    assert packet.metadata["predictor_id"] == adapter.adapter_id
    assert packet.prediction.metadata["reconstruction_error"] < 1e-6
    assert [item.evidence_id for item in packet.evidence] == ["post:1"]


def test_additive_head_save_load_roundtrip(tmp_path):
    head = _head()
    before = head.explain_one([0.1, 0.2, 0.3]).logit
    path = tmp_path / "nam.pt"
    head.save(str(path))
    restored = SparseNeuralAdditiveRiskHead.load(str(path))
    after = restored.explain_one([0.1, 0.2, 0.3]).logit
    assert before == pytest.approx(after, abs=1e-7)
