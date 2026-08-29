from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


def load_app(tmp_path: Path, cases_path: Path | None = None):
    os.environ["HYPERTRACE_DB_PATH"] = str(tmp_path / "study.sqlite3")
    os.environ["HYPERTRACE_CASES_PATH"] = str(
        cases_path or ROOT / "data" / "study_cases.demo.json"
    )
    os.environ["STUDY_SALT"] = "test-study-salt"
    os.environ["ADMIN_TOKEN"] = "test-admin-token"
    os.environ["HYPERTRACE_DURABLE_STORAGE"] = "1"
    os.environ["HYPERTRACE_PREVIEW_MODE"] = "1"
    os.environ["HYPERTRACE_TRIAL_COUNT"] = "8"
    module_name = f"review_app_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def create_session(client: TestClient, code: str) -> str:
    response = client.post(
        "/api/session",
        json={"participant_code": code, "consent": True, "age_confirmed": True},
    )
    assert response.status_code == 200
    return response.json()["session_id"]


def test_conditions_are_balanced_and_labels_never_reach_trial_payload(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    views = set()

    for index in range(3):
        session_id = create_session(client, f"participant-{index}")
        response = client.get(f"/api/session/{session_id}/trial")
        assert response.status_code == 200
        case = response.json()["case"]
        assert case["phase"] == "initial"
        assert "ground_truth" not in case
        assert "model_recommendation" not in case
        assert "risk_percentile" not in case
        initial = client.post(
            f"/api/session/{session_id}/initial-response",
            json={
                "trial_index": case["trial_index"],
                "case_id": case["case_id"],
                "decision": "coordinated",
                "confidence": 50,
            },
        )
        assert initial.status_code == 200
        revealed = client.post(
            f"/api/session/{session_id}/reveal",
            json={"trial_index": case["trial_index"], "case_id": case["case_id"]},
        )
        assert revealed.status_code == 200
        assisted = revealed.json()["case"]
        views.add(assisted["view_mode"])
        assert assisted["phase"] == "assisted"
        assert "ground_truth" not in assisted
        if assisted["view_mode"] == "risk_only":
            assert "relation_summary" not in assisted
            assert "explanation" not in assisted
        elif assisted["view_mode"] == "standard_signals":
            assert "relation_summary" in assisted
            assert "explanation" not in assisted
        else:
            assert "relation_summary" in assisted
            assert "explanation" in assisted

    assert views == set(module.CONDITIONS)

    for index in range(100):
        order = module.trial_order(f"assignment-audit-{index}")
        cases = [module.CASE_LOOKUP[case_id] for case_id in order]
        assert sum(case["ground_truth"] == "coordinated" for case in cases) == 4
        assert sum(case["ground_truth"] == "not_coordinated" for case in cases) == 4
        assert sum(
            case["ground_truth"] != case["model_recommendation"] for case in cases
        ) == module.TARGET_MODEL_ERRORS
        operation_counts = {}
        for case in cases:
            operation_counts[case["operation"]] = operation_counts.get(case["operation"], 0) + 1
        assert sorted(operation_counts.values()) == [4, 4]


def test_complete_workflow_and_admin_export(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    session_id = create_session(client, "workflow-participant")

    for expected_index in range(module.TRIAL_COUNT):
        trial = client.get(f"/api/session/{session_id}/trial").json()["case"]
        assert trial["trial_index"] == expected_index
        assert trial["phase"] == "initial"
        initial = client.post(
            f"/api/session/{session_id}/initial-response",
            json={
                "trial_index": expected_index,
                "case_id": trial["case_id"],
                "decision": "not_coordinated",
                "confidence": 45,
            },
        )
        assert initial.status_code == 200
        resumed = client.get(f"/api/session/{session_id}/trial").json()["case"]
        assert resumed["phase"] == "ready_to_reveal"
        reveal = client.post(
            f"/api/session/{session_id}/reveal",
            json={"trial_index": expected_index, "case_id": trial["case_id"]},
        )
        assert reveal.status_code == 200
        assert reveal.json()["case"]["phase"] == "assisted"
        response = client.post(
            f"/api/session/{session_id}/response",
            json={
                "trial_index": expected_index,
                "case_id": trial["case_id"],
                "decision": "coordinated",
                "confidence": 60,
                "rationale": "test response",
            },
        )
        assert response.status_code == 200

    assert client.get(f"/api/session/{session_id}/trial").json()["complete"] is True
    complete = client.post(
        f"/api/session/{session_id}/questionnaire",
        json={
            "trust": 4,
            "clarity": 5,
            "workload": 3,
            "evidence_usefulness": 5,
            "feedback": "",
        },
    )
    assert complete.status_code == 200
    assert len(complete.json()["completion_code"]) == 10

    assert client.get("/api/admin/export").status_code == 401
    exported = client.get(
        "/api/admin/export",
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert exported.status_code == 200
    assert "participant_hash" in exported.text
    assert "workflow-participant" not in exported.text
    assert "initial_correct" in exported.text
    assert "final_correct" in exported.text
    assert "decision_changed" in exported.text
    assert "wrong_ai_rejection" in exported.text
    assert "rair" in exported.text
    assert "rsr" in exported.text


def test_enforces_initial_reveal_final_order(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    session_id = create_session(client, "ordering-participant")
    trial = client.get(f"/api/session/{session_id}/trial").json()["case"]
    identity = {"trial_index": trial["trial_index"], "case_id": trial["case_id"]}
    final_payload = {
        **identity,
        "decision": "coordinated",
        "confidence": 60,
        "rationale": "",
    }
    assert client.post(f"/api/session/{session_id}/reveal", json=identity).status_code == 409
    assert client.post(f"/api/session/{session_id}/response", json=final_payload).status_code == 409
    assert client.post(
        f"/api/session/{session_id}/initial-response",
        json={**identity, "decision": "coordinated", "confidence": 60},
    ).status_code == 200
    assert client.post(f"/api/session/{session_id}/response", json=final_payload).status_code == 409
    assert client.post(f"/api/session/{session_id}/reveal", json=identity).status_code == 200
    assert client.post(f"/api/session/{session_id}/response", json=final_payload).status_code == 200


def test_case_assignment_balances_coverage_within_each_condition(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    for index in range(60):
        create_session(client, f"coverage-participant-{index}")

    with module.STORE.connect() as db:
        rows = db.execute(
            "SELECT condition, trial_order_json FROM sessions "
            "WHERE protocol_version = ?",
            (module.PROTOCOL_VERSION,),
        ).fetchall()
    by_condition = {
        condition: {case["case_id"]: 0 for case in module.CASES}
        for condition in module.CONDITIONS
    }
    for row in rows:
        counts = by_condition[row["condition"]]
        for case_id in module.json.loads(row["trial_order_json"]):
            counts[case_id] = counts.get(case_id, 0) + 1
    for counts in by_condition.values():
        error_counts = [
            counts[case["case_id"]] for case in module.CASES
            if case["ground_truth"] != case["model_recommendation"]
        ]
        correct_counts = [
            counts[case["case_id"]] for case in module.CASES
            if case["ground_truth"] == case["model_recommendation"]
        ]
        assert max(error_counts) - min(error_counts) <= 1
        assert max(correct_counts) - min(correct_counts) <= 1


def test_rejects_missing_consent(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    response = client.post(
        "/api/session",
        json={"participant_code": "p-001", "consent": False, "age_confirmed": True},
    )
    assert response.status_code == 400


def test_hugging_face_embedding_and_admin_page_are_supported(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)

    index = client.get("/")
    assert index.status_code == 200
    assert "https://huggingface.co" in index.headers["content-security-policy"]
    assert "x-frame-options" not in index.headers
    assert client.get("/admin").status_code == 200
    assert client.get("/api/health").json()["durable_storage"] is True


def test_local_preview_switches_conditions_without_exposing_labels(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)

    for condition in module.CONDITIONS:
        response = client.get(f"/api/preview?mode={condition}&case_index=0")
        assert response.status_code == 200
        case = response.json()["case"]
        assert case["view_mode"] == condition
        assert "ground_truth" not in case
        if condition == "hypertrace_evidence":
            assert case["alternative_explanations"]
            assert "ground_truth" not in case["alternative_explanations"][0]
    with module.STORE.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_demo_versions_updates_rollback_and_judgment_revision(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)

    versions = client.get("/api/preview/versions?case_id=DEMO-A01").json()
    assert versions["current_version_id"] == "v2"
    assert len(versions["versions"]) == 2

    update = client.post(
        "/api/preview/simulate-update",
        json={"trial_index": 0, "case_id": "DEMO-A01"},
    )
    assert update.status_code == 200
    assert update.json()["current_version_id"] == "v3"
    status = client.get("/api/preview/updates?case_id=DEMO-A01&since=v2").json()
    assert status["changed"] is True
    assert status["requires_rollback"] is True

    restored = client.post(
        "/api/preview/rollback",
        json={"trial_index": 0, "case_id": "DEMO-A01", "version_id": "v2"},
    )
    assert restored.status_code == 200
    assert restored.json()["current_version_id"] == "v4"

    session_id = create_session(client, "revision-participant")
    trial = client.get(f"/api/session/{session_id}/trial").json()["case"]
    identity = {"trial_index": trial["trial_index"], "case_id": trial["case_id"]}
    assert client.post(
        f"/api/session/{session_id}/initial-response",
        json={**identity, "decision": "coordinated", "confidence": 50},
    ).status_code == 200
    assert client.post(f"/api/session/{session_id}/reveal", json=identity).status_code == 200
    final = client.post(
        f"/api/session/{session_id}/response",
        json={**identity, "decision": "coordinated", "confidence": 50, "rationale": "first"},
    )
    assert final.status_code == 200
    assert final.json()["case"]["phase"] == "submitted"
    revision = client.post(
        f"/api/session/{session_id}/response/revise",
        json={**identity, "decision": "not_coordinated", "confidence": 35, "rationale": "corrected", "reason": "review"},
    )
    assert revision.status_code == 200
    assert revision.json()["revision_no"] == 2
    assert len(revision.json()["history"]) == 2


def test_provider_neutral_evidence_ingestion_versions_new_records_and_deduplicates(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    digest = "a" * 64
    record = {
        "evidence_id": "EV-INGEST-01",
        "source_record_id": "provider-record-01",
        "timestamp": "2019-04-04T10:00:00+00:00",
        "event_type": "retweet",
        "relation_type": "amplifies",
        "text": "New normalized provider record.",
    }
    response = client.post(
        "/api/preview/evidence/ingest",
        json={
            "trial_index": 0,
            "case_id": "DEMO-A01",
            "provider": "example-platform-adapter",
            "source_bundle_hash": digest,
            "records": [record],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "new_version"
    assert response.json()["records_added"] == 1
    assert response.json()["current_version_id"] == "v3"
    duplicate = client.post(
        "/api/preview/evidence/ingest",
        json={
            "trial_index": 0,
            "case_id": "DEMO-A01",
            "provider": "example-platform-adapter",
            "source_bundle_hash": digest,
            "records": [record],
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "duplicate_only"
    assert duplicate.json()["current_version_id"] == "v3"


def test_provider_neutral_ingestion_rejects_missing_provenance_fields(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    response = client.post(
        "/api/preview/evidence/ingest",
        json={
            "trial_index": 0,
            "case_id": "DEMO-A01",
            "provider": "adapter",
            "source_bundle_hash": "b" * 64,
            "records": [{"evidence_id": "missing-fields"}],
        },
    )
    assert response.status_code == 422


def test_formal_deployment_disables_provider_ingestion_by_default(tmp_path) -> None:
    module = load_app(tmp_path, ROOT / "data" / "study_cases.private.json")
    client = TestClient(module.app)
    response = client.post(
        "/api/session/not-a-session/evidence/ingest",
        json={
            "trial_index": 0,
            "case_id": "PRIVATE-A01",
            "provider": "adapter",
            "source_bundle_hash": "c" * 64,
            "records": [{
                "evidence_id": "EV-INGEST-FORMAL-01",
                "source_record_id": "provider-record-01",
                "timestamp": "2019-04-04T10:00:00+00:00",
                "event_type": "retweet",
            }],
        },
    )
    assert response.status_code == 403


def test_formal_initial_payload_contains_only_common_case_evidence(tmp_path) -> None:
    module = load_app(tmp_path, ROOT / "data" / "study_cases.private.json")
    client = TestClient(module.app)
    assert client.get("/api/health").json()["common_evidence_ready"] is True
    session_id = create_session(client, "formal-common-evidence")
    case = client.get(f"/api/session/{session_id}/trial").json()["case"]
    assert case["phase"] == "initial"
    assert case["common_case_evidence"]
    assert "ground_truth" not in case
    assert "model_recommendation" not in case
    assert "risk_percentile" not in case
    assert "common_case_contract" not in case
    for record in case["common_case_evidence"]:
        assert set(record) == {"record_id", "timestamp", "event_type", "text"}
