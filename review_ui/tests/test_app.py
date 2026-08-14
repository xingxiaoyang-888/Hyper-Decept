from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


def load_app(tmp_path: Path):
    os.environ["HYPERTRACE_DB_PATH"] = str(tmp_path / "study.sqlite3")
    os.environ["HYPERTRACE_CASES_PATH"] = str(ROOT / "data" / "study_cases.demo.json")
    os.environ["STUDY_SALT"] = "test-study-salt"
    os.environ["ADMIN_TOKEN"] = "test-admin-token"
    os.environ["HYPERTRACE_DURABLE_STORAGE"] = "1"
    os.environ["HYPERTRACE_PREVIEW_MODE"] = "1"
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
        views.add(case["view_mode"])
        assert "ground_truth" not in case
        if case["view_mode"] == "risk_only":
            assert "relation_summary" not in case
            assert "explanation" not in case
        elif case["view_mode"] == "standard_signals":
            assert "relation_summary" in case
            assert "explanation" not in case
        else:
            assert "relation_summary" in case
            assert "explanation" in case

    assert views == set(module.CONDITIONS)


def test_complete_workflow_and_admin_export(tmp_path) -> None:
    module = load_app(tmp_path)
    client = TestClient(module.app)
    session_id = create_session(client, "workflow-participant")

    for expected_index in range(module.TRIAL_COUNT):
        trial = client.get(f"/api/session/{session_id}/trial").json()["case"]
        assert trial["trial_index"] == expected_index
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
    assert "appropriate_reliance" in exported.text


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
    with module.STORE.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
