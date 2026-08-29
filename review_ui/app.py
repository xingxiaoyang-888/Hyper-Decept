from __future__ import annotations

import csv
import hashlib
import hmac
import io
import itertools
import json
import os
import random
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("HYPERTRACE_DATA_DIR", ROOT / "data")).resolve()
PRIVATE_CASES = DATA_DIR / "study_cases.private.json"
DEMO_CASES = DATA_DIR / "study_cases.demo.json"
CASES_PATH = Path(os.getenv("HYPERTRACE_CASES_PATH", PRIVATE_CASES)).resolve()
if not CASES_PATH.is_file():
    CASES_PATH = DEMO_CASES

DEFAULT_DB = Path("/data/hypertrace_study.sqlite3")
if not DEFAULT_DB.parent.exists() or not os.access(DEFAULT_DB.parent, os.W_OK):
    DEFAULT_DB = DATA_DIR / "hypertrace_study.sqlite3"
DB_PATH = Path(os.getenv("HYPERTRACE_DB_PATH", DEFAULT_DB)).resolve()
STUDY_SALT = os.getenv("STUDY_SALT", "development-only-change-before-study")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DURABLE_STORAGE = os.getenv("HYPERTRACE_DURABLE_STORAGE", "0").lower() in {
    "1", "true", "yes"
}
PREVIEW_MODE = os.getenv("HYPERTRACE_PREVIEW_MODE", "0").lower() in {
    "1", "true", "yes"
}
DEMO_MODE = CASES_PATH == DEMO_CASES
ALLOW_JUDGMENT_REVISION = os.getenv(
    "HYPERTRACE_ALLOW_JUDGMENT_REVISION", "1" if DEMO_MODE else "0"
).lower() in {"1", "true", "yes"}
CONSENT_VERSION = "hypertrace-chi-consent-v2"
PROTOCOL_VERSION = "three-stage-v2"
CONDITIONS = ("risk_only", "standard_signals", "hypertrace_evidence")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or len(cases) < 6:
        raise RuntimeError("study case package must contain at least six cases")
    identifiers: set[str] = set()
    for case in cases:
        required = {
            "case_id",
            "operation",
            "ground_truth",
            "model_recommendation",
            "risk_percentile",
            "risk_stratum",
        }
        missing = sorted(required - set(case))
        if missing:
            raise RuntimeError(f"case package entry missing fields: {missing}")
        if case["case_id"] in identifiers:
            raise RuntimeError(f"duplicate case_id: {case['case_id']}")
        if case["ground_truth"] not in {"coordinated", "not_coordinated"}:
            raise RuntimeError("unsupported ground-truth class")
        if path == PRIVATE_CASES or path.name == PRIVATE_CASES.name:
            common_evidence = case.get("common_case_evidence")
            contract = case.get("common_case_contract", {})
            if not common_evidence:
                raise RuntimeError(
                    f"formal case {case['case_id']} has no common case evidence"
                )
            if any(
                contract.get(field) is not False
                for field in (
                    "ground_truth_consumed",
                    "model_output_consumed",
                    "explainer_ranking_consumed",
                )
            ):
                raise RuntimeError(
                    f"formal case {case['case_id']} violates the common-evidence boundary"
                )
        identifiers.add(case["case_id"])
    return cases


CASES = load_cases(CASES_PATH)
CASE_LOOKUP = {case["case_id"]: case for case in CASES}
TRIAL_COUNT = min(int(os.getenv("HYPERTRACE_TRIAL_COUNT", "8")), len(CASES))
TARGET_MODEL_ERRORS = 3
if TRIAL_COUNT % 2:
    raise RuntimeError("HYPERTRACE_TRIAL_COUNT must be even")


class SessionRequest(BaseModel):
    participant_code: str = Field(min_length=3, max_length=96)
    consent: bool
    age_confirmed: bool


class TrialIdentity(BaseModel):
    trial_index: int = Field(ge=0)
    case_id: str = Field(min_length=3, max_length=64)


class InitialResponse(TrialIdentity):
    decision: Literal["coordinated", "not_coordinated"]
    confidence: int = Field(ge=0, le=100)


class TrialResponse(InitialResponse):
    decision: Literal["coordinated", "not_coordinated"]
    confidence: int = Field(ge=0, le=100)
    rationale: str = Field(default="", max_length=1000)


class QuestionnaireResponse(BaseModel):
    trust: int = Field(ge=1, le=7)
    clarity: int = Field(ge=1, le=7)
    workload: int = Field(ge=1, le=7)
    evidence_usefulness: int = Field(ge=1, le=7)
    feedback: str = Field(default="", max_length=2000)


class RevisionRequest(TrialIdentity):
    decision: Literal["coordinated", "not_coordinated"]
    confidence: int = Field(ge=0, le=100)
    rationale: str = Field(default="", max_length=1000)
    reason: str = Field(default="Reviewer requested a correction", max_length=500)


class VersionRequest(TrialIdentity):
    version_id: str = Field(min_length=1, max_length=80)
    reason: str = Field(default="Reviewer restored an earlier evidence state", max_length=500)


class UpdateRequest(TrialIdentity):
    reason: str = Field(default="Demonstration update", max_length=500)


class StudyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    participant_hash TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    consent_version TEXT NOT NULL,
                    protocol_version TEXT NOT NULL DEFAULT 'three-stage-v2',
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    trial_order_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exposures (
                    session_id TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, trial_index),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS initial_responses (
                    session_id TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    submitted_at TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    PRIMARY KEY (session_id, trial_index),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS assistance_exposures (
                    session_id TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    revealed_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, trial_index),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS responses (
                    session_id TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    rationale TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    PRIMARY KEY (session_id, trial_index),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS questionnaires (
                    session_id TEXT PRIMARY KEY,
                    trust INTEGER NOT NULL,
                    clarity INTEGER NOT NULL,
                    workload INTEGER NOT NULL,
                    evidence_usefulness INTEGER NOT NULL,
                    feedback TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS evidence_versions (
                    scope_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    parent_version_id TEXT,
                    created_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    invalidates_current INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (scope_id, case_id, version_id)
                );
                CREATE TABLE IF NOT EXISTS judgment_revisions (
                    session_id TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    revision_no INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    rationale TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    revised_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, trial_index, revision_no),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                """
            )
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "protocol_version" not in columns:
                db.execute(
                    "ALTER TABLE sessions ADD COLUMN protocol_version TEXT "
                    "NOT NULL DEFAULT 'single-stage-v1'"
                )

    def assign_condition(self, db: sqlite3.Connection) -> str:
        counts = {
            condition: db.execute(
                "SELECT COUNT(*) FROM sessions WHERE condition = ? AND protocol_version = ?",
                (condition, PROTOCOL_VERSION),
            ).fetchone()[0]
            for condition in CONDITIONS
        }
        minimum = min(counts.values())
        choices = [condition for condition, count in counts.items() if count == minimum]
        return random.SystemRandom().choice(choices)


STORE = StudyStore(DB_PATH)


def participant_hash(code: str) -> str:
    normalized = " ".join(code.strip().lower().split())
    return hmac.new(
        STUDY_SALT.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def assigned_case_counts(db: sqlite3.Connection, condition: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    rows = db.execute(
        "SELECT trial_order_json FROM sessions "
        "WHERE condition = ? AND protocol_version = ?",
        (condition, PROTOCOL_VERSION),
    ).fetchall()
    for row in rows:
        counts.update(json.loads(row["trial_order_json"]))
    return counts


def valid_trial_sets() -> list[tuple[dict, ...]]:
    errors = [
        case for case in CASES
        if case["ground_truth"] != case["model_recommendation"]
    ]
    correct = [
        case for case in CASES
        if case["ground_truth"] == case["model_recommendation"]
    ]
    if len(errors) < TARGET_MODEL_ERRORS:
        raise RuntimeError("case package cannot support the model-error exposure gate")

    per_class = TRIAL_COUNT // 2
    operations = sorted({case["operation"] for case in CASES})
    operation_floor = TRIAL_COUNT // len(operations)
    operation_ceiling = operation_floor + int(TRIAL_COUNT % len(operations) > 0)
    candidates: list[tuple[dict, ...]] = []
    for error_set in itertools.combinations(errors, TARGET_MODEL_ERRORS):
        for correct_set in itertools.combinations(correct, TRIAL_COUNT - TARGET_MODEL_ERRORS):
            selected = error_set + correct_set
            labels = Counter(case["ground_truth"] for case in selected)
            if any(labels[label] != per_class for label in ("coordinated", "not_coordinated")):
                continue
            operation_counts = Counter(case["operation"] for case in selected)
            if any(
                operation_counts[operation] < operation_floor
                or operation_counts[operation] > operation_ceiling
                for operation in operations
            ):
                continue
            candidates.append(selected)
    if not candidates:
        raise RuntimeError("case package cannot support the configured balanced trial design")
    return candidates


VALID_TRIAL_SETS = valid_trial_sets()


def trial_order(
    session_id: str,
    condition: str | None = None,
    case_counts: Counter[str] | None = None,
) -> list[str]:
    seed = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    counts = case_counts or Counter()
    scored = []
    for selected in VALID_TRIAL_SETS:
        usage = [counts[case["case_id"]] for case in selected]
        score = (sum((value + 1) ** 2 for value in usage), max(usage, default=0))
        scored.append((score, selected))
    best_score = min(score for score, _ in scored)
    best = [selected for score, selected in scored if score == best_score]
    selected = list(rng.choice(best))
    rng.shuffle(selected)
    return [case["case_id"] for case in selected]


def common_case(case: dict, trial_index: int, phase: str) -> dict:
    summary = case.get("common_case_summary", {})
    return {
        "case_id": case["case_id"],
        "operation": case["operation"],
        "operation_label": case.get("operation_label", case["operation"].upper()),
        "snapshot_scope": case.get("snapshot_scope", "full_release_snapshot"),
        "trial_index": trial_index,
        "trial_count": TRIAL_COUNT,
        "phase": phase,
        "case_context": {
            "record_count": summary.get("source_activity_count"),
            "displayed_record_count": summary.get("displayed_record_count"),
            "observation_start": summary.get("first_event_at"),
            "observation_end": summary.get("last_event_at"),
            "event_type_counts": summary.get("event_type_counts", {}),
        },
        "common_case_evidence": case.get("common_case_evidence", []),
    }


def _version_scope(scope_id: str | None) -> str:
    return scope_id or "preview"


def ensure_evidence_versions(db: sqlite3.Connection, case: dict, scope_id: str) -> None:
    """Seed two deterministic snapshots for old case bundles, then preserve all later revisions."""
    exists = db.execute(
        "SELECT 1 FROM evidence_versions WHERE scope_id = ? AND case_id = ? LIMIT 1",
        (scope_id, case["case_id"]),
    ).fetchone()
    if exists:
        return
    evidence = list(case.get("evidence", []))
    baseline_count = max(0, len(evidence) // 2)
    now = utc_now()
    db.executemany(
        "INSERT INTO evidence_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (scope_id, case["case_id"], "v1", None, now, "Initial evidence snapshot", json.dumps(evidence[:baseline_count]), 0),
            (scope_id, case["case_id"], "v2", "v1", now, "Current constrained evidence reconstruction", json.dumps(evidence), 0),
        ],
    )


def evidence_versions(db: sqlite3.Connection, case: dict, scope_id: str) -> list[dict]:
    ensure_evidence_versions(db, case, scope_id)
    rows = db.execute(
        "SELECT version_id, parent_version_id, created_at, reason, evidence_json, invalidates_current "
        "FROM evidence_versions WHERE scope_id = ? AND case_id = ? ORDER BY rowid",
        (scope_id, case["case_id"]),
    ).fetchall()
    return [
        {
            "version_id": row["version_id"],
            "parent_version_id": row["parent_version_id"],
            "created_at": row["created_at"],
            "reason": row["reason"],
            "evidence": json.loads(row["evidence_json"]),
            "invalidates_current": bool(row["invalidates_current"]),
        }
        for row in rows
    ]


def attach_version_state(case_payload: dict, case: dict, scope_id: str) -> dict:
    with STORE.lock, STORE.connect() as db:
        versions = evidence_versions(db, case, scope_id)
    if versions:
        current = versions[-1]
        case_payload["evidence"] = current["evidence"]
        case_payload["evidence_versions"] = versions
        case_payload["current_version_id"] = current["version_id"]
    return case_payload


def public_case(case: dict, condition: str, trial_index: int, scope_id: str | None = None) -> dict:
    common = {
        **common_case(case, trial_index, "assisted"),
        "risk_percentile": case["risk_percentile"],
        "risk_stratum": case["risk_stratum"],
        "model_recommendation": case["model_recommendation"],
        "view_mode": condition,
    }
    if condition in {"standard_signals", "hypertrace_evidence"}:
        common["relation_summary"] = case.get("relation_summary", {})
        common["temporal_summary"] = case.get("temporal_summary", {})
    if condition == "hypertrace_evidence":
        common["explanation"] = case.get("explanation", {})
        common["evidence"] = case.get("evidence", [])
        common["audit"] = case.get("audit", {})
        attach_version_state(common, case, _version_scope(scope_id))
    return common


def session_row(session_id: str) -> sqlite3.Row:
    with STORE.connect() as db:
        row = db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown study session")
    return row


def admin_guard(authorization: str | None = Header(default=None)) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="admin export is not configured")
    expected = f"Bearer {ADMIN_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid admin token")


app = FastAPI(title="HyperTrace Review Study", docs_url=None, redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' https://unpkg.com; "
        "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
        "font-src 'self'; frame-ancestors 'self' https://huggingface.co "
        "https://*.huggingface.co; base-uri 'self'"
    )
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/assets/app.js")
def javascript() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="application/javascript")


@app.get("/assets/styles.css")
def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "styles.css", media_type="text/css")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(ROOT / "admin.html", headers={"Cache-Control": "no-store"})


@app.get("/assets/admin.js")
def admin_javascript() -> FileResponse:
    return FileResponse(ROOT / "admin.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "case_count": len(CASES),
        "trial_count": TRIAL_COUNT,
        "demo_data": CASES_PATH == DEMO_CASES,
        "durable_storage": DURABLE_STORAGE,
        "storage_path": str(DB_PATH.parent),
        "preview_mode": PREVIEW_MODE,
        "common_evidence_ready": all(
            bool(case.get("common_case_evidence")) for case in CASES
        ),
    }


@app.get("/api/study")
def study_metadata() -> dict:
    return {
        "consent_version": CONSENT_VERSION,
        "trial_count": TRIAL_COUNT,
        "data_policy": "No names, IP addresses, or raw participant codes are stored.",
    }


@app.get("/api/preview")
def preview_case(mode: str, case_index: int = 0) -> dict:
    if not PREVIEW_MODE:
        raise HTTPException(status_code=404, detail="preview mode is disabled")
    if mode not in CONDITIONS:
        raise HTTPException(status_code=400, detail="unknown interface condition")
    if case_index < 0 or case_index >= len(CASES):
        raise HTTPException(status_code=400, detail="preview case index out of range")
    return {
        "case": public_case(CASES[case_index], mode, case_index, "preview"),
        "preview": True,
        "available_cases": len(CASES),
    }


def version_response(case_id: str, scope_id: str) -> dict:
    case = CASE_LOOKUP.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case")
    with STORE.lock, STORE.connect() as db:
        versions = evidence_versions(db, case, scope_id)
    return {"case_id": case_id, "versions": versions, "current_version_id": versions[-1]["version_id"]}


@app.get("/api/preview/versions")
def preview_versions(case_id: str) -> dict:
    if not PREVIEW_MODE:
        raise HTTPException(status_code=404, detail="preview mode is disabled")
    return version_response(case_id, "preview")


def rollback_version(case_id: str, scope_id: str, version_id: str, reason: str) -> dict:
    case = CASE_LOOKUP.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case")
    with STORE.lock, STORE.connect() as db:
        versions = evidence_versions(db, case, scope_id)
        selected = next((item for item in versions if item["version_id"] == version_id), None)
        current = versions[-1]
        if selected is None:
            raise HTTPException(status_code=404, detail="unknown evidence version")
        if selected["version_id"] == current["version_id"]:
            raise HTTPException(status_code=409, detail="version is already current")
        next_no = len(versions) + 1
        new_id = f"v{next_no}"
        db.execute(
            "INSERT INTO evidence_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (scope_id, case_id, new_id, current["version_id"], utc_now(), reason or f"Restored from {version_id}", json.dumps(selected["evidence"]), 0),
        )
    return version_response(case_id, scope_id)


@app.post("/api/preview/rollback")
def preview_rollback(payload: VersionRequest) -> dict:
    if not PREVIEW_MODE:
        raise HTTPException(status_code=404, detail="preview mode is disabled")
    return rollback_version(payload.case_id, "preview", payload.version_id, payload.reason)


@app.get("/api/preview/updates")
def preview_updates(case_id: str, since: str = "") -> dict:
    if not PREVIEW_MODE:
        raise HTTPException(status_code=404, detail="preview mode is disabled")
    data = version_response(case_id, "preview")
    current = data["versions"][-1]
    return {"case_id": case_id, "current_version_id": current["version_id"], "changed": bool(since and since != current["version_id"]), "requires_rollback": current["invalidates_current"]}


@app.post("/api/preview/simulate-update")
def preview_simulate_update(payload: UpdateRequest) -> dict:
    if not PREVIEW_MODE or not DEMO_MODE:
        raise HTTPException(status_code=404, detail="demonstration updates are disabled")
    case = CASE_LOOKUP.get(payload.case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="unknown case")
    with STORE.lock, STORE.connect() as db:
        versions = evidence_versions(db, case, "preview")
        current = versions[-1]
        evidence = list(current["evidence"])
        evidence.append({
            "evidence_id": f"EV-LIVE-{len(versions) + 1:02d}",
            "timestamp": utc_now(),
            "event_type": "new_record",
            "text": "New evidence record received in the demonstration stream.",
        })
        new_id = f"v{len(versions) + 1}"
        db.execute(
            "INSERT INTO evidence_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("preview", payload.case_id, new_id, current["version_id"], utc_now(), payload.reason, json.dumps(evidence), 1),
        )
    return version_response(payload.case_id, "preview")


@app.post("/api/session")
def create_session(payload: SessionRequest) -> dict:
    if not payload.consent or not payload.age_confirmed:
        raise HTTPException(status_code=400, detail="consent and age confirmation required")
    session_id = str(uuid.uuid4())
    with STORE.lock, STORE.connect() as db:
        existing = db.execute(
            "SELECT session_id FROM sessions WHERE participant_hash = ? AND completed_at IS NULL "
            "AND protocol_version = ? ORDER BY created_at DESC LIMIT 1",
            (participant_hash(payload.participant_code), PROTOCOL_VERSION),
        ).fetchone()
        if existing:
            return {"session_id": existing["session_id"], "resumed": True}
        condition = STORE.assign_condition(db)
        order = trial_order(
            session_id,
            condition,
            assigned_case_counts(db, condition),
        )
        db.execute(
            """
            INSERT INTO sessions (
                session_id, participant_hash, condition, consent_version,
                protocol_version, created_at, completed_at, trial_order_json
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                session_id,
                participant_hash(payload.participant_code),
                condition,
                CONSENT_VERSION,
                PROTOCOL_VERSION,
                utc_now(),
                json.dumps(order),
            ),
        )
    return {"session_id": session_id, "resumed": False}


@app.get("/api/session/{session_id}/trial")
def get_trial(session_id: str) -> dict:
    row = session_row(session_id)
    order = json.loads(row["trial_order_json"])
    with STORE.lock, STORE.connect() as db:
        completed = {
            int(value[0])
            for value in db.execute(
                "SELECT trial_index FROM responses WHERE session_id = ?", (session_id,)
            ).fetchall()
        }
        next_index = next((index for index in range(len(order)) if index not in completed), None)
        if next_index is None:
            return {"complete": True, "questionnaire_required": True}
        case_id = order[next_index]
        db.execute(
            "INSERT OR IGNORE INTO exposures VALUES (?, ?, ?, ?)",
            (session_id, next_index, case_id, utc_now()),
        )
        initial = db.execute(
            "SELECT decision, confidence FROM initial_responses "
            "WHERE session_id = ? AND trial_index = ?",
            (session_id, next_index),
        ).fetchone()
        assistance = db.execute(
            "SELECT revealed_at FROM assistance_exposures "
            "WHERE session_id = ? AND trial_index = ?",
            (session_id, next_index),
        ).fetchone()
    if initial is None:
        case_payload = common_case(CASE_LOOKUP[case_id], next_index, "initial")
    elif assistance is None:
        case_payload = common_case(CASE_LOOKUP[case_id], next_index, "ready_to_reveal")
        case_payload["initial_response"] = dict(initial)
    else:
        case_payload = public_case(CASE_LOOKUP[case_id], row["condition"], next_index, f"session:{session_id}")
        case_payload["initial_response"] = dict(initial)
    return {
        "complete": False,
        "case": case_payload,
    }


def validate_trial(row: sqlite3.Row, payload: TrialIdentity) -> list[str]:
    order = json.loads(row["trial_order_json"])
    if payload.trial_index >= len(order) or order[payload.trial_index] != payload.case_id:
        raise HTTPException(status_code=409, detail="response does not match assigned trial")
    return order


@app.post("/api/session/{session_id}/initial-response")
def record_initial_response(session_id: str, payload: InitialResponse) -> dict:
    row = session_row(session_id)
    validate_trial(row, payload)
    with STORE.lock, STORE.connect() as db:
        exposure = db.execute(
            "SELECT opened_at FROM exposures WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()
        if exposure is None:
            raise HTTPException(status_code=409, detail="trial was not opened")
        if db.execute(
            "SELECT 1 FROM responses WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone():
            raise HTTPException(status_code=409, detail="final response already submitted")
        opened = datetime.fromisoformat(exposure["opened_at"])
        latency_ms = max(
            0, int((datetime.now(timezone.utc) - opened).total_seconds() * 1000)
        )
        try:
            db.execute(
                "INSERT INTO initial_responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    payload.trial_index,
                    payload.case_id,
                    payload.decision,
                    payload.confidence,
                    utc_now(),
                    latency_ms,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="initial response already submitted") from error
    return {"accepted": True, "phase": "ready_to_reveal"}


@app.post("/api/session/{session_id}/reveal")
def reveal_assistance(session_id: str, payload: TrialIdentity) -> dict:
    row = session_row(session_id)
    validate_trial(row, payload)
    with STORE.lock, STORE.connect() as db:
        initial = db.execute(
            "SELECT decision, confidence FROM initial_responses "
            "WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()
        if initial is None:
            raise HTTPException(status_code=409, detail="initial response is required")
        if db.execute(
            "SELECT 1 FROM responses WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone():
            raise HTTPException(status_code=409, detail="final response already submitted")
        db.execute(
            "INSERT OR IGNORE INTO assistance_exposures VALUES (?, ?, ?, ?)",
            (session_id, payload.trial_index, payload.case_id, utc_now()),
        )
    case_payload = public_case(
        CASE_LOOKUP[payload.case_id], row["condition"], payload.trial_index, f"session:{session_id}"
    )
    case_payload["initial_response"] = dict(initial)
    return {"case": case_payload}


@app.post("/api/session/{session_id}/response")
def record_response(session_id: str, payload: TrialResponse) -> dict:
    row = session_row(session_id)
    order = validate_trial(row, payload)
    with STORE.lock, STORE.connect() as db:
        initial = db.execute(
            "SELECT 1 FROM initial_responses WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()
        if initial is None:
            raise HTTPException(status_code=409, detail="initial response is required")
        assistance = db.execute(
            "SELECT revealed_at FROM assistance_exposures "
            "WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()
        if assistance is None:
            raise HTTPException(status_code=409, detail="model assistance has not been revealed")
        revealed = datetime.fromisoformat(assistance["revealed_at"])
        latency_ms = max(
            0, int((datetime.now(timezone.utc) - revealed).total_seconds() * 1000)
        )
        try:
            db.execute(
                "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    payload.trial_index,
                    payload.case_id,
                    payload.decision,
                    payload.confidence,
                    payload.rationale.strip(),
                    utc_now(),
                    latency_ms,
                ),
            )
            db.execute(
                "INSERT INTO judgment_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, payload.trial_index, 1, payload.decision, payload.confidence,
                 payload.rationale.strip(), "Initial final submission", utc_now()),
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="trial already submitted") from error
        completed = db.execute(
            "SELECT COUNT(*) FROM responses WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    if ALLOW_JUDGMENT_REVISION:
        case_payload = public_case(
            CASE_LOOKUP[payload.case_id], row["condition"], payload.trial_index, f"session:{session_id}"
        )
        case_payload["phase"] = "submitted"
        case_payload["final_response"] = {
            "decision": payload.decision,
            "confidence": payload.confidence,
            "rationale": payload.rationale.strip(),
        }
        return {"accepted": True, "completed_trials": completed, "trial_count": len(order), "case": case_payload}
    return {"accepted": True, "completed_trials": completed, "trial_count": len(order)}


@app.get("/api/session/{session_id}/versions")
def session_versions(session_id: str, case_id: str) -> dict:
    row = session_row(session_id)
    if case_id not in json.loads(row["trial_order_json"]):
        raise HTTPException(status_code=403, detail="case is not assigned to this session")
    return version_response(case_id, f"session:{session_id}")


@app.post("/api/session/{session_id}/evidence/rollback")
def session_rollback(session_id: str, payload: VersionRequest) -> dict:
    row = session_row(session_id)
    if payload.case_id not in json.loads(row["trial_order_json"]):
        raise HTTPException(status_code=403, detail="case is not assigned to this session")
    return rollback_version(payload.case_id, f"session:{session_id}", payload.version_id, payload.reason)


@app.get("/api/session/{session_id}/updates")
def session_updates(session_id: str, case_id: str, since: str = "") -> dict:
    row = session_row(session_id)
    if case_id not in json.loads(row["trial_order_json"]):
        raise HTTPException(status_code=403, detail="case is not assigned to this session")
    data = version_response(case_id, f"session:{session_id}")
    current = data["versions"][-1]
    return {"case_id": case_id, "current_version_id": current["version_id"], "changed": bool(since and since != current["version_id"]), "requires_rollback": current["invalidates_current"]}


@app.post("/api/session/{session_id}/response/revise")
def revise_response(session_id: str, payload: RevisionRequest) -> dict:
    if not ALLOW_JUDGMENT_REVISION:
        raise HTTPException(status_code=403, detail="judgment revision is disabled for this deployment")
    row = session_row(session_id)
    validate_trial(row, payload)
    with STORE.lock, STORE.connect() as db:
        existing = db.execute(
            "SELECT 1 FROM responses WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=409, detail="final response has not been submitted")
        latest = db.execute(
            "SELECT COALESCE(MAX(revision_no), 0) FROM judgment_revisions WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()[0]
        revision_no = int(latest) + 1
        now = utc_now()
        db.execute(
            "UPDATE responses SET decision = ?, confidence = ?, rationale = ?, submitted_at = ? WHERE session_id = ? AND trial_index = ?",
            (payload.decision, payload.confidence, payload.rationale.strip(), now, session_id, payload.trial_index),
        )
        db.execute(
            "INSERT INTO judgment_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, payload.trial_index, revision_no, payload.decision, payload.confidence, payload.rationale.strip(), payload.reason.strip(), now),
        )
        history = [dict(item) for item in db.execute(
            "SELECT revision_no, decision, confidence, rationale, reason, revised_at FROM judgment_revisions WHERE session_id = ? AND trial_index = ? ORDER BY revision_no",
            (session_id, payload.trial_index),
        ).fetchall()]
    return {"accepted": True, "revision_no": revision_no, "history": history}


@app.post("/api/session/{session_id}/questionnaire")
def record_questionnaire(session_id: str, payload: QuestionnaireResponse) -> dict:
    row = session_row(session_id)
    order = json.loads(row["trial_order_json"])
    with STORE.lock, STORE.connect() as db:
        response_count = db.execute(
            "SELECT COUNT(*) FROM responses WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        initial_count = db.execute(
            "SELECT COUNT(*) FROM initial_responses WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        reveal_count = db.execute(
            "SELECT COUNT(*) FROM assistance_exposures WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        if response_count != len(order) or initial_count != len(order) or reveal_count != len(order):
            raise HTTPException(status_code=409, detail="all trials must be completed first")
        db.execute(
            "INSERT OR REPLACE INTO questionnaires VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                payload.trust,
                payload.clarity,
                payload.workload,
                payload.evidence_usefulness,
                payload.feedback.strip(),
                utc_now(),
            ),
        )
        db.execute(
            "UPDATE sessions SET completed_at = ? WHERE session_id = ?",
            (utc_now(), session_id),
        )
    completion_code = hmac.new(
        STUDY_SALT.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:10].upper()
    return {"complete": True, "completion_code": completion_code}


def response_records() -> list[dict]:
    with STORE.connect() as db:
        rows = db.execute(
            """
            SELECT s.session_id, s.participant_hash, s.condition, s.protocol_version,
                   s.created_at, s.completed_at, r.trial_index, r.case_id,
                   i.decision AS initial_decision,
                   i.confidence AS initial_confidence,
                   i.submitted_at AS initial_submitted_at,
                   i.latency_ms AS initial_latency_ms,
                   a.revealed_at AS assistance_revealed_at,
                   r.decision AS final_decision,
                   r.confidence AS final_confidence,
                   r.rationale, r.submitted_at AS final_submitted_at,
                   r.latency_ms AS assisted_latency_ms
            FROM sessions s
            JOIN responses r ON s.session_id = r.session_id
            LEFT JOIN initial_responses i
              ON i.session_id = r.session_id AND i.trial_index = r.trial_index
            LEFT JOIN assistance_exposures a
              ON a.session_id = r.session_id AND a.trial_index = r.trial_index
            ORDER BY s.created_at, r.trial_index
            """
        ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        case = CASE_LOOKUP[record["case_id"]]
        record["ground_truth"] = case["ground_truth"]
        record["model_recommendation"] = case["model_recommendation"]
        protocol_complete = (
            record["protocol_version"] == PROTOCOL_VERSION
            and record["initial_decision"] is not None
            and record["assistance_revealed_at"] is not None
        )
        record["protocol_complete"] = int(protocol_complete)
        record["initial_correct"] = (
            int(record["initial_decision"] == case["ground_truth"])
            if record["initial_decision"] is not None
            else None
        )
        record["final_correct"] = int(
            record["final_decision"] == case["ground_truth"]
        )
        record["decision_changed"] = (
            int(record["initial_decision"] != record["final_decision"])
            if record["initial_decision"] is not None
            else None
        )
        record["confidence_change"] = (
            record["final_confidence"] - record["initial_confidence"]
            if record["initial_confidence"] is not None
            else None
        )
        record["reveal_latency_ms"] = (
            max(
                0,
                int(
                    (
                        datetime.fromisoformat(record["assistance_revealed_at"])
                        - datetime.fromisoformat(record["initial_submitted_at"])
                    ).total_seconds()
                    * 1000
                ),
            )
            if record["assistance_revealed_at"] is not None
            and record["initial_submitted_at"] is not None
            else None
        )
        record["total_latency_ms"] = (
            record["initial_latency_ms"]
            + record["reveal_latency_ms"]
            + record["assisted_latency_ms"]
            if record["initial_latency_ms"] is not None
            and record["reveal_latency_ms"] is not None
            else None
        )
        model_correct = case["model_recommendation"] == case["ground_truth"]
        record["model_correct"] = int(model_correct)
        record["initial_accepted_model"] = (
            int(record["initial_decision"] == case["model_recommendation"])
            if record["initial_decision"] is not None
            else None
        )
        record["final_accepted_model"] = int(
            record["final_decision"] == case["model_recommendation"]
        )
        record["correct_ai_acceptance"] = (
            record["final_accepted_model"] if model_correct else None
        )
        record["wrong_ai_rejection"] = (
            int(not record["final_accepted_model"]) if not model_correct else None
        )
        rair_eligible = bool(
            protocol_complete and model_correct and record["initial_correct"] == 0
        )
        rsr_eligible = bool(
            protocol_complete and not model_correct and record["initial_correct"] == 1
        )
        record["rair_eligible"] = int(rair_eligible)
        record["rair"] = record["final_correct"] if rair_eligible else None
        record["rsr_eligible"] = int(rsr_eligible)
        record["rsr"] = record["final_correct"] if rsr_eligible else None
        record["accuracy_gain"] = (
            record["final_correct"] - record["initial_correct"]
            if record["initial_correct"] is not None
            else None
        )
        records.append(record)
    return records


def mean_defined(records: list[dict], field: str) -> float | None:
    values = [record[field] for record in records if record.get(field) is not None]
    return sum(values) / len(values) if values else None


@app.get("/api/admin/summary", dependencies=[Depends(admin_guard)])
def admin_summary() -> dict:
    records = response_records()
    by_condition = {}
    for condition in CONDITIONS:
        selected = [
            record for record in records
            if record["condition"] == condition and record["protocol_complete"]
        ]
        by_condition[condition] = {
            "responses": len(selected),
            "initial_accuracy": mean_defined(selected, "initial_correct"),
            "final_accuracy": mean_defined(selected, "final_correct"),
            "correct_ai_acceptance": mean_defined(selected, "correct_ai_acceptance"),
            "wrong_ai_rejection": mean_defined(selected, "wrong_ai_rejection"),
            "rair": mean_defined(selected, "rair"),
            "rsr": mean_defined(selected, "rsr"),
        }
    with STORE.connect() as db:
        sessions = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        completed = db.execute(
            "SELECT COUNT(*) FROM sessions WHERE completed_at IS NOT NULL"
        ).fetchone()[0]
        analyzable = db.execute(
            "SELECT COUNT(*) FROM sessions WHERE completed_at IS NOT NULL "
            "AND protocol_version = ?",
            (PROTOCOL_VERSION,),
        ).fetchone()[0]
    return {
        "sessions": sessions,
        "completed_sessions": completed,
        "analyzable_sessions": analyzable,
        "conditions": by_condition,
    }


@app.get("/api/admin/export", dependencies=[Depends(admin_guard)])
def admin_export() -> Response:
    records = response_records()
    if not records:
        return PlainTextResponse("", media_type="text/csv")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return Response(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=hypertrace_responses.csv"},
    )


@app.exception_handler(HTTPException)
async def http_error(_: Request, error: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
