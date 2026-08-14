from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import random
import sqlite3
import threading
import uuid
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
CONSENT_VERSION = "hypertrace-chi-consent-v1"
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
        identifiers.add(case["case_id"])
    return cases


CASES = load_cases(CASES_PATH)
CASE_LOOKUP = {case["case_id"]: case for case in CASES}
TRIAL_COUNT = min(int(os.getenv("HYPERTRACE_TRIAL_COUNT", "10")), len(CASES))


class SessionRequest(BaseModel):
    participant_code: str = Field(min_length=3, max_length=96)
    consent: bool
    age_confirmed: bool


class TrialResponse(BaseModel):
    trial_index: int = Field(ge=0)
    case_id: str = Field(min_length=3, max_length=64)
    decision: Literal["coordinated", "not_coordinated"]
    confidence: int = Field(ge=0, le=100)
    rationale: str = Field(default="", max_length=1000)


class QuestionnaireResponse(BaseModel):
    trust: int = Field(ge=1, le=7)
    clarity: int = Field(ge=1, le=7)
    workload: int = Field(ge=1, le=7)
    evidence_usefulness: int = Field(ge=1, le=7)
    feedback: str = Field(default="", max_length=2000)


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
                """
            )

    def assign_condition(self, db: sqlite3.Connection) -> str:
        counts = {
            condition: db.execute(
                "SELECT COUNT(*) FROM sessions WHERE condition = ?", (condition,)
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


def trial_order(session_id: str) -> list[str]:
    seed = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    def operation_balanced(values: list[dict], count: int) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for case in values:
            groups.setdefault(case["operation"], []).append(case)
        for group in groups.values():
            rng.shuffle(group)
        selected: list[dict] = []
        while len(selected) < count and any(groups.values()):
            for operation in sorted(groups):
                if groups[operation] and len(selected) < count:
                    selected.append(groups[operation].pop())
        return selected

    coordinated = [case for case in CASES if case["ground_truth"] == "coordinated"]
    not_coordinated = [
        case for case in CASES if case["ground_truth"] == "not_coordinated"
    ]
    per_class = TRIAL_COUNT // 2
    if len(coordinated) < per_class or len(not_coordinated) < per_class:
        raise RuntimeError("case package cannot support a balanced study trial set")
    selected = operation_balanced(coordinated, per_class)
    selected.extend(operation_balanced(not_coordinated, per_class))
    rng.shuffle(selected)
    return [case["case_id"] for case in selected]


def public_case(case: dict, condition: str, trial_index: int) -> dict:
    common = {
        "case_id": case["case_id"],
        "operation": case["operation"],
        "operation_label": case.get("operation_label", case["operation"].upper()),
        "risk_percentile": case["risk_percentile"],
        "risk_stratum": case["risk_stratum"],
        "model_recommendation": case["model_recommendation"],
        "snapshot_scope": case.get("snapshot_scope", "full_release_snapshot"),
        "trial_index": trial_index,
        "trial_count": TRIAL_COUNT,
        "view_mode": condition,
    }
    if condition in {"standard_signals", "hypertrace_evidence"}:
        common["relation_summary"] = case.get("relation_summary", {})
        common["temporal_summary"] = case.get("temporal_summary", {})
    if condition == "hypertrace_evidence":
        common["explanation"] = case.get("explanation", {})
        common["evidence"] = case.get("evidence", [])
        common["audit"] = case.get("audit", {})
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
        "case": public_case(CASES[case_index], mode, case_index),
        "preview": True,
        "available_cases": len(CASES),
    }


@app.post("/api/session")
def create_session(payload: SessionRequest) -> dict:
    if not payload.consent or not payload.age_confirmed:
        raise HTTPException(status_code=400, detail="consent and age confirmation required")
    session_id = str(uuid.uuid4())
    order = trial_order(session_id)
    with STORE.lock, STORE.connect() as db:
        existing = db.execute(
            "SELECT session_id FROM sessions WHERE participant_hash = ? AND completed_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (participant_hash(payload.participant_code),),
        ).fetchone()
        if existing:
            return {"session_id": existing["session_id"], "resumed": True}
        condition = STORE.assign_condition(db)
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (
                session_id,
                participant_hash(payload.participant_code),
                condition,
                CONSENT_VERSION,
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
    return {
        "complete": False,
        "case": public_case(CASE_LOOKUP[case_id], row["condition"], next_index),
    }


@app.post("/api/session/{session_id}/response")
def record_response(session_id: str, payload: TrialResponse) -> dict:
    row = session_row(session_id)
    order = json.loads(row["trial_order_json"])
    if payload.trial_index >= len(order) or order[payload.trial_index] != payload.case_id:
        raise HTTPException(status_code=409, detail="response does not match assigned trial")
    with STORE.lock, STORE.connect() as db:
        exposure = db.execute(
            "SELECT opened_at FROM exposures WHERE session_id = ? AND trial_index = ?",
            (session_id, payload.trial_index),
        ).fetchone()
        if exposure is None:
            raise HTTPException(status_code=409, detail="trial was not opened")
        opened = datetime.fromisoformat(exposure["opened_at"])
        latency_ms = max(
            0, int((datetime.now(timezone.utc) - opened).total_seconds() * 1000)
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
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="trial already submitted") from error
        completed = db.execute(
            "SELECT COUNT(*) FROM responses WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    return {"accepted": True, "completed_trials": completed, "trial_count": len(order)}


@app.post("/api/session/{session_id}/questionnaire")
def record_questionnaire(session_id: str, payload: QuestionnaireResponse) -> dict:
    row = session_row(session_id)
    order = json.loads(row["trial_order_json"])
    with STORE.lock, STORE.connect() as db:
        response_count = db.execute(
            "SELECT COUNT(*) FROM responses WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        if response_count != len(order):
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
            SELECT s.session_id, s.participant_hash, s.condition, s.created_at,
                   s.completed_at, r.trial_index, r.case_id, r.decision,
                   r.confidence, r.rationale, r.submitted_at, r.latency_ms
            FROM sessions s JOIN responses r ON s.session_id = r.session_id
            ORDER BY s.created_at, r.trial_index
            """
        ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        case = CASE_LOOKUP[record["case_id"]]
        record["ground_truth"] = case["ground_truth"]
        record["model_recommendation"] = case["model_recommendation"]
        record["correct"] = int(record["decision"] == case["ground_truth"])
        model_correct = case["model_recommendation"] == case["ground_truth"]
        record["model_correct"] = int(model_correct)
        record["accepted_model"] = int(
            record["decision"] == case["model_recommendation"]
        )
        record["appropriate_reliance"] = int(
            (model_correct and record["accepted_model"])
            or (not model_correct and not record["accepted_model"])
        )
        records.append(record)
    return records


@app.get("/api/admin/summary", dependencies=[Depends(admin_guard)])
def admin_summary() -> dict:
    records = response_records()
    by_condition = {}
    for condition in CONDITIONS:
        selected = [record for record in records if record["condition"] == condition]
        by_condition[condition] = {
            "responses": len(selected),
            "accuracy": (
                sum(record["correct"] for record in selected) / len(selected)
                if selected
                else None
            ),
            "appropriate_reliance": (
                sum(record["appropriate_reliance"] for record in selected) / len(selected)
                if selected
                else None
            ),
        }
    with STORE.connect() as db:
        sessions = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        completed = db.execute(
            "SELECT COUNT(*) FROM sessions WHERE completed_at IS NOT NULL"
        ).fetchone()[0]
    return {"sessions": sessions, "completed_sessions": completed, "conditions": by_condition}


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
