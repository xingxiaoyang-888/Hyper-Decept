import pytest

from explainability.adapters import (
    AdapterRegistry,
    ExplanationFragment,
    ExplanationOrchestrator,
    ExplanationRequest,
    FunctionExplainerAdapter,
    FunctionPredictorAdapter,
    LocalExplanationAdapter,
    RegistryEvidenceProvider,
    TabularPredictorAdapter,
)
from explainability.evidence_registry import EvidenceRegistry
from explainability.local_explainer import LocalExplanation
from explainability.schemas import (
    ConceptRecord,
    ContributionRecord,
    EvidenceRecord,
    PredictionRecord,
    SubgraphRecord,
)


def _orchestrator(failing=False):
    registry = AdapterRegistry()
    registry.register_predictor(FunctionPredictorAdapter(
        "predictor", "model-v2",
        lambda _request: PredictionRecord(
            label="coordinated_deception", probability=0.8, threshold=0.5
        ),
    ))
    registry.register_explainer(FunctionExplainerAdapter(
        "feature",
        lambda _request, _prediction: ExplanationFragment(
            contributions=[ContributionRecord(
                feature_name="burst", feature_value=0.9,
                contribution=0.4, evidence_ids=["post:1"],
            )],
            critical_subgraph=SubgraphRecord(
                nodes=["u1", "u2"],
                edges=[{
                    "source": "u1", "target": "u2",
                    "evidence_id": "follow:1",
                }],
            ),
        ),
    ))
    registry.register_explainer(FunctionExplainerAdapter(
        "concept",
        lambda _request, _prediction: ExplanationFragment(
            concepts=[ConceptRecord(
                concept_name="coordination", concept_value=0.7,
                evidence_ids=["post:1"],
            )],
            warnings=["synthetic warning"],
        ),
    ))
    if failing:
        def fail(_request, _prediction):
            raise RuntimeError("broken explainer")
        registry.register_explainer(FunctionExplainerAdapter("broken", fail))
    evidence = EvidenceRegistry()
    evidence._records["post:1"] = EvidenceRecord(
        evidence_id="post:1", evidence_type="post", actor_id="u1"
    )
    evidence._records["follow:1"] = EvidenceRecord(
        evidence_id="follow:1", evidence_type="follow", actor_id="u1",
        target_id="u2",
    )
    registry.register_evidence_provider(RegistryEvidenceProvider(evidence))
    return ExplanationOrchestrator(registry), registry


def test_orchestrator_combines_framework_outputs_and_resolves_evidence():
    orchestrator, registry = _orchestrator()
    request = ExplanationRequest(
        case_id="user:u1", run_id="run-1", payload={"features": [0.9]}
    )
    packet = orchestrator.explain(
        request, "predictor", ["feature", "concept"]
    )
    assert packet.schema_version == "1.1"
    assert packet.model_version == "model-v2"
    assert packet.prediction.probability == 0.8
    assert packet.contributions[0].metadata["explainer_id"] == "feature"
    assert packet.concepts[0].metadata["explainer_id"] == "concept"
    assert {item.evidence_id for item in packet.evidence} == {
        "post:1", "follow:1"
    }
    assert packet.critical_subgraph.nodes == ["u1", "u2"]
    assert registry.capabilities()["predictors"] == ["predictor"]


def test_explainer_failure_is_visible_but_does_not_hide_other_results():
    orchestrator, _ = _orchestrator(failing=True)
    packet = orchestrator.explain(
        ExplanationRequest(case_id="u1", payload={}),
        "predictor",
        ["broken", "feature"],
    )
    assert len(packet.contributions) == 1
    assert any("broken explainer" in warning for warning in packet.warnings)


def test_fail_fast_and_duplicate_registration_are_enforced():
    orchestrator, registry = _orchestrator(failing=True)
    orchestrator.fail_fast = True
    with pytest.raises(RuntimeError, match="broken explainer"):
        orchestrator.explain(
            ExplanationRequest(case_id="u1", payload={}),
            "predictor", ["broken"],
        )
    with pytest.raises(ValueError, match="already registered"):
        registry.register_predictor(FunctionPredictorAdapter(
            "predictor", "other", lambda _r: PredictionRecord()
        ))


def test_invalid_probability_is_rejected():
    registry = AdapterRegistry()
    registry.register_predictor(FunctionPredictorAdapter(
        "bad", "v1", lambda _r: PredictionRecord(probability=1.2)
    ))
    with pytest.raises(ValueError, match="probability"):
        ExplanationOrchestrator(registry).predict(
            ExplanationRequest(case_id="u1", payload={}), "bad"
        )


def test_concrete_tabular_and_existing_local_explainer_bridges():
    class Model:
        def predict_proba(self, values):
            assert values.shape == (1, 2)
            return [[0.25, 0.75]]

    class LocalExplainer:
        def explain_one(self, user_id):
            assert user_id == "u1"
            return LocalExplanation(
                user_id="u1", probability=0.75,
                predicted_label="coordinated_deception", threshold=0.5,
                base_value=0.1,
                contributions=[ContributionRecord(
                    feature_name="f1", feature_value=2.0,
                    contribution=0.5,
                )],
                reconstruction_error=1e-8,
            )

    predictor = TabularPredictorAdapter(
        Model(), ["f1", "f2"], model_version="xgb-test"
    )
    explainer = LocalExplanationAdapter(
        LocalExplainer(),
        provenance={"u1": {"f1": {"evidence_ids": ["post:1"]}}},
    )
    adapters = AdapterRegistry()
    adapters.register_predictor(predictor)
    adapters.register_explainer(explainer)
    evidence = EvidenceRegistry()
    evidence._records["post:1"] = EvidenceRecord(
        evidence_id="post:1", evidence_type="post"
    )
    adapters.register_evidence_provider(RegistryEvidenceProvider(evidence))
    packet = ExplanationOrchestrator(adapters).explain(
        ExplanationRequest(
            case_id="user:u1", payload={"features": {"f1": 2, "f2": 1}}
        ),
        predictor.adapter_id,
        [explainer.adapter_id],
    )
    assert packet.prediction.probability == 0.75
    assert packet.contributions[0].evidence_ids == ["post:1"]
    assert packet.metadata["fragment_metadata"][explainer.adapter_id][
        "reconstruction_error"
    ] == pytest.approx(1e-8)
