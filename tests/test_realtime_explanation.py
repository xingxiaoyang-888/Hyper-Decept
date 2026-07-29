import sqlite3

import pytest

from explainability.adapters import ExplanationRequest
from explainability.realtime import (
    ConcurrentTraceUpdateError,
    ExplanationState,
    InvalidTransitionError,
    SQLiteTraceStore,
    TraceableExplanationService,
)
from tests.test_explanation_adapters import _orchestrator


def test_traceable_service_reaches_review_and_preserves_hash_chain(tmp_path):
    orchestrator, _ = _orchestrator()
    path = tmp_path / "trace.db"
    store = SQLiteTraceStore(str(path))
    service = TraceableExplanationService(orchestrator, store)
    request = ExplanationRequest(
        case_id="user:u1", run_id="run-1", payload={"private": "not-in-ingest-event"}
    )
    packet = service.explain(request, "predictor", ["feature", "concept"])
    assert packet.case_id == "user:u1"
    assert store.current_state("user:u1", "run-1") == ExplanationState.READY_FOR_REVIEW
    events = store.get_events("user:u1", "run-1")
    assert [event.state for event in events] == [
        "INGESTED", "PREDICTED", "EXPLAINING", "EVIDENCE_LINKED",
        "READY_FOR_REVIEW",
    ]
    assert "private" not in str(events[0].payload)
    assert store.verify_chain("user:u1", "run-1") is True
    service.record_review(
        "user:u1", "run-1", reviewer_id="reviewer-7",
        decision="bot", confidence=0.8,
    )
    service.finalize_review(
        "user:u1", "run-1", reviewer_id="reviewer-7",
        corrected=False,
    )
    assert store.current_state("user:u1", "run-1") == ExplanationState.CONFIRMED
    assert store.verify_chain("user:u1", "run-1") is True
    store.close()

    reopened = SQLiteTraceStore(str(path))
    assert reopened.current_state("user:u1", "run-1") == ExplanationState.CONFIRMED
    assert reopened.verify_chain("user:u1", "run-1") is True
    reopened.close()


def test_invalid_and_stale_transitions_are_rejected():
    store = SQLiteTraceStore()
    with pytest.raises(InvalidTransitionError, match="start"):
        store.append("u1", "r1", ExplanationState.PREDICTED)
    store.append("u1", "r1", ExplanationState.INGESTED)
    with pytest.raises(InvalidTransitionError, match="Invalid transition"):
        store.append("u1", "r1", ExplanationState.READY_FOR_REVIEW)
    with pytest.raises(ConcurrentTraceUpdateError, match="Expected sequence"):
        store.append(
            "u1", "r1", ExplanationState.PREDICTED, expected_sequence=0
        )
    store.close()


def test_sqlite_trace_rows_cannot_be_updated_or_deleted():
    store = SQLiteTraceStore()
    store.append("u1", "r1", ExplanationState.INGESTED)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._connection.execute(
            "UPDATE explanation_trace_events SET actor='tampered'"
        )
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._connection.execute("DELETE FROM explanation_trace_events")
    store.close()


def test_service_failure_is_recorded():
    orchestrator, _ = _orchestrator(failing=True)
    orchestrator.fail_fast = True
    store = SQLiteTraceStore()
    service = TraceableExplanationService(orchestrator, store)
    with pytest.raises(RuntimeError, match="broken explainer"):
        service.explain(
            ExplanationRequest(case_id="u1", run_id="r1", payload={}),
            "predictor", ["broken"],
        )
    assert store.current_state("u1", "r1") == ExplanationState.FAILED
    assert store.verify_chain("u1", "r1") is True
    store.close()
