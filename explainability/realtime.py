"""Append-only real-time explanation state and audit history.

This module is transport-neutral: FastAPI/WebSocket/SSE layers can stream the
same persisted events without changing model code.  Every transition is
hash-chained so an explanation remains auditable after later model updates or
changes in deceptive behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from .adapters import ExplanationOrchestrator, ExplanationRequest
from .schemas import ExplanationPacket, PredictionRecord, _safe_value


class ExplanationState(str, Enum):
    INGESTED = "INGESTED"
    PREDICTED = "PREDICTED"
    EXPLAINING = "EXPLAINING"
    EVIDENCE_LINKED = "EVIDENCE_LINKED"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEWED = "REVIEWED"
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS = {
    ExplanationState.INGESTED: {
        ExplanationState.PREDICTED,
        ExplanationState.FAILED,
    },
    ExplanationState.PREDICTED: {
        ExplanationState.EXPLAINING,
        ExplanationState.FAILED,
    },
    ExplanationState.EXPLAINING: {
        ExplanationState.EVIDENCE_LINKED,
        ExplanationState.FAILED,
    },
    ExplanationState.EVIDENCE_LINKED: {
        ExplanationState.READY_FOR_REVIEW,
        ExplanationState.FAILED,
    },
    ExplanationState.READY_FOR_REVIEW: {
        ExplanationState.REVIEWED,
        ExplanationState.FAILED,
    },
    ExplanationState.REVIEWED: {
        ExplanationState.CONFIRMED,
        ExplanationState.CORRECTED,
    },
    ExplanationState.CONFIRMED: {ExplanationState.ARCHIVED},
    ExplanationState.CORRECTED: {ExplanationState.ARCHIVED},
    ExplanationState.FAILED: {ExplanationState.ARCHIVED},
    ExplanationState.ARCHIVED: set(),
}


class InvalidTransitionError(ValueError):
    pass


class ConcurrentTraceUpdateError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _safe_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


def payload_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    case_id: str
    run_id: str
    sequence: int
    previous_state: Optional[str]
    state: str
    event_type: str
    actor: str
    created_at: str
    payload: Dict[str, Any]
    previous_hash: str
    event_hash: str

    def to_dict(self) -> dict:
        return _safe_value(asdict(self))


def _event_hash_payload(event: TraceEvent) -> dict:
    data = event.to_dict()
    data.pop("event_hash", None)
    return data


def _calculate_event_hash(event: TraceEvent) -> str:
    return payload_digest(_event_hash_payload(event))


class SQLiteTraceStore:
    """SQLite append-only event store with optimistic sequence checks."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS explanation_trace_events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    previous_state TEXT,
                    state TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    UNIQUE(case_id, run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_case_run
                    ON explanation_trace_events(case_id, run_id, sequence);
                CREATE TRIGGER IF NOT EXISTS trace_events_no_update
                BEFORE UPDATE ON explanation_trace_events
                BEGIN
                    SELECT RAISE(ABORT, 'explanation trace is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS trace_events_no_delete
                BEFORE DELETE ON explanation_trace_events
                BEGIN
                    SELECT RAISE(ABORT, 'explanation trace is append-only');
                END;
                """
            )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TraceEvent:
        return TraceEvent(
            event_id=row["event_id"],
            case_id=row["case_id"],
            run_id=row["run_id"],
            sequence=int(row["sequence"]),
            previous_state=row["previous_state"],
            state=row["state"],
            event_type=row["event_type"],
            actor=row["actor"],
            created_at=row["created_at"],
            payload=json.loads(row["payload_json"]),
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
        )

    def get_events(
        self,
        case_id: str,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> List[TraceEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM explanation_trace_events
                WHERE case_id = ? AND run_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (case_id, run_id, int(after_sequence)),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def current_event(self, case_id: str, run_id: str) -> Optional[TraceEvent]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM explanation_trace_events
                WHERE case_id = ? AND run_id = ?
                ORDER BY sequence DESC LIMIT 1
                """,
                (case_id, run_id),
            ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def current_state(
        self, case_id: str, run_id: str
    ) -> Optional[ExplanationState]:
        event = self.current_event(case_id, run_id)
        return ExplanationState(event.state) if event is not None else None

    def append(
        self,
        case_id: str,
        run_id: str,
        state: ExplanationState,
        *,
        payload: Optional[Dict[str, Any]] = None,
        actor: str = "system",
        event_type: str = "state_transition",
        expected_sequence: Optional[int] = None,
    ) -> TraceEvent:
        state = ExplanationState(state)
        safe_payload = _safe_value(payload or {})
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    """
                    SELECT * FROM explanation_trace_events
                    WHERE case_id = ? AND run_id = ?
                    ORDER BY sequence DESC LIMIT 1
                    """,
                    (case_id, run_id),
                ).fetchone()
                previous = self._row_to_event(row) if row is not None else None
                previous_sequence = previous.sequence if previous else 0
                if (
                    expected_sequence is not None
                    and int(expected_sequence) != previous_sequence
                ):
                    raise ConcurrentTraceUpdateError(
                        f"Expected sequence {expected_sequence}, "
                        f"found {previous_sequence}"
                    )
                if previous is None:
                    if state is not ExplanationState.INGESTED:
                        raise InvalidTransitionError(
                            "A trace must start in INGESTED"
                        )
                    previous_state = None
                    previous_hash = ""
                else:
                    previous_state = ExplanationState(previous.state)
                    if state not in ALLOWED_TRANSITIONS[previous_state]:
                        raise InvalidTransitionError(
                            f"Invalid transition: {previous_state.value} -> "
                            f"{state.value}"
                        )
                    previous_hash = previous.event_hash
                event = TraceEvent(
                    event_id=str(uuid4()),
                    case_id=str(case_id),
                    run_id=str(run_id),
                    sequence=previous_sequence + 1,
                    previous_state=(
                        previous_state.value if previous_state else None
                    ),
                    state=state.value,
                    event_type=str(event_type),
                    actor=str(actor),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    payload=safe_payload,
                    previous_hash=previous_hash,
                    event_hash="",
                )
                event = TraceEvent(
                    **{**event.to_dict(), "event_hash": _calculate_event_hash(event)}
                )
                cursor.execute(
                    """
                    INSERT INTO explanation_trace_events (
                        event_id, case_id, run_id, sequence, previous_state,
                        state, event_type, actor, created_at, payload_json,
                        previous_hash, event_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id, event.case_id, event.run_id,
                        event.sequence, event.previous_state, event.state,
                        event.event_type, event.actor, event.created_at,
                        _canonical_json(event.payload), event.previous_hash,
                        event.event_hash,
                    ),
                )
                self._connection.commit()
                return event
            except Exception:
                self._connection.rollback()
                raise

    def verify_chain(self, case_id: str, run_id: str) -> bool:
        previous_hash = ""
        previous_state = None
        for expected_sequence, event in enumerate(
            self.get_events(case_id, run_id), start=1
        ):
            if event.sequence != expected_sequence:
                return False
            if event.previous_hash != previous_hash:
                return False
            if event.previous_state != previous_state:
                return False
            if _calculate_event_hash(event) != event.event_hash:
                return False
            previous_hash = event.event_hash
            previous_state = event.state
        return True

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteTraceStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class TraceableExplanationService:
    """Run one explanation while persisting every externally visible stage."""

    def __init__(
        self,
        orchestrator: ExplanationOrchestrator,
        trace_store: SQLiteTraceStore,
    ) -> None:
        self.orchestrator = orchestrator
        self.trace_store = trace_store

    def explain(
        self,
        request: ExplanationRequest,
        predictor_id: str,
        explainer_ids: Sequence[str],
        *,
        actor: str = "system",
    ) -> ExplanationPacket:
        if self.trace_store.current_event(request.case_id, request.run_id):
            raise ValueError(
                "Trace already exists for this case_id/run_id; use a new run_id"
            )
        self.trace_store.append(
            request.case_id,
            request.run_id,
            ExplanationState.INGESTED,
            payload={
                "request_payload_sha256": payload_digest(request.payload),
                "request_metadata": request.metadata,
            },
            actor=actor,
            event_type="case_ingested",
        )
        try:
            prediction = self.orchestrator.predict(request, predictor_id)
            self.trace_store.append(
                request.case_id,
                request.run_id,
                ExplanationState.PREDICTED,
                payload={
                    "predictor_id": predictor_id,
                    "prediction": prediction.to_dict(),
                },
                actor=actor,
                event_type="prediction_ready",
            )
            self.trace_store.append(
                request.case_id,
                request.run_id,
                ExplanationState.EXPLAINING,
                payload={"requested_explainers": list(explainer_ids)},
                actor=actor,
                event_type="explanation_started",
            )
            packet = self.orchestrator.explain_prediction(
                request, prediction, predictor_id, explainer_ids
            )
            self.trace_store.append(
                request.case_id,
                request.run_id,
                ExplanationState.EVIDENCE_LINKED,
                payload={
                    "evidence_ids": [item.evidence_id for item in packet.evidence],
                    "evidence_count": len(packet.evidence),
                    "unresolved_warning_count": sum(
                        "Unresolved evidence IDs" in item
                        for item in packet.warnings
                    ),
                },
                actor=actor,
                event_type="evidence_linked",
            )
            self.trace_store.append(
                request.case_id,
                request.run_id,
                ExplanationState.READY_FOR_REVIEW,
                payload={
                    "packet_sha256": payload_digest(packet.to_dict()),
                    "packet": packet.to_dict(),
                },
                actor=actor,
                event_type="explanation_ready",
            )
            return packet
        except Exception as error:
            current = self.trace_store.current_state(request.case_id, request.run_id)
            if current is not None and ExplanationState.FAILED in ALLOWED_TRANSITIONS[current]:
                self.trace_store.append(
                    request.case_id,
                    request.run_id,
                    ExplanationState.FAILED,
                    payload={
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                    actor=actor,
                    event_type="explanation_failed",
                )
            raise

    def record_review(
        self,
        case_id: str,
        run_id: str,
        *,
        reviewer_id: str,
        decision: str,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> TraceEvent:
        return self.trace_store.append(
            case_id,
            run_id,
            ExplanationState.REVIEWED,
            payload={
                "decision": decision,
                "confidence": confidence,
                "notes": notes,
            },
            actor=reviewer_id,
            event_type="human_review_recorded",
        )

    def finalize_review(
        self,
        case_id: str,
        run_id: str,
        *,
        reviewer_id: str,
        corrected: bool,
        reason: str = "",
    ) -> TraceEvent:
        state = (
            ExplanationState.CORRECTED
            if corrected else ExplanationState.CONFIRMED
        )
        return self.trace_store.append(
            case_id,
            run_id,
            state,
            payload={"reason": reason},
            actor=reviewer_id,
            event_type=(
                "human_decision_corrected" if corrected
                else "human_decision_confirmed"
            ),
        )
