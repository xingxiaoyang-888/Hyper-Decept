"""Run a synthetic, non-human pilot through the same HTTP study contract.

This output is for UI/metric validation only and must never be reported as
human-subject evidence in a CHI paper.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASE = "http://127.0.0.1:8766"
CASE_FILE = DATA / "study_cases.private.json"
OUT_JSON = DATA / "synthetic_experts_pilot.json"
OUT_CSV = DATA / "synthetic_experts_pilot.csv"
OUT_REPORT = DATA / "synthetic_experts_pilot_report.md"
TARGET_PARTICIPANTS = 20
TRIALS_PER_PARTICIPANT = 10


def request(method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def main() -> None:
    pilot_db = DATA / "hypertrace_synthetic_pilot.sqlite3"
    env = os.environ.copy()
    env.update({
        "HYPERTRACE_CASES_PATH": str(CASE_FILE),
        "HYPERTRACE_DB_PATH": str(pilot_db),
        "STUDY_SALT": "synthetic-pilot-salt",
        "ADMIN_TOKEN": "synthetic-pilot-admin",
        "HYPERTRACE_DURABLE_STORAGE": "1",
        "HYPERTRACE_PREVIEW_MODE": "0",
        "HYPERTRACE_TRIAL_COUNT": str(TRIALS_PER_PARTICIPANT),
    })
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8766"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(30):
            try:
                request("GET", "/api/health")
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("synthetic pilot server did not start")

        _run_pilot()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def _run_pilot() -> None:
    package = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in package["cases"]}
    records = _load_existing_records()
    _archive_original_fifteen(records)
    completed_participants = {row["participant"] for row in records}

    # The synthetic experts differ only in stable calibration/noise. The
    # condition effects are deliberately modest and are not empirical claims.
    accept_correct = {"risk_only": 0.82, "standard_signals": 0.88, "hypertrace_evidence": 0.93}
    accept_wrong = {"risk_only": 0.72, "standard_signals": 0.45, "hypertrace_evidence": 0.22}
    base_latency = {"risk_only": 7.0, "standard_signals": 10.0, "hypertrace_evidence": 14.0}

    for expert_index in range(1, TARGET_PARTICIPANTS + 1):
        participant = f"RE-{expert_index:02d}"
        if participant in completed_participants:
            continue
        rng = random.Random(20260815 + expert_index * 1009)
        session = request("POST", "/api/session", {
            "participant_code": participant,
            "consent": True,
            "age_confirmed": True,
        })["session_id"]
        condition = None
        for _ in range(TRIALS_PER_PARTICIPANT):
            trial = request("GET", f"/api/session/{session}/trial")
            case = trial["case"]
            condition = case["view_mode"]
            truth = cases[case["case_id"]]["ground_truth"]
            model = case["model_recommendation"]
            model_correct = model == truth
            p_accept = accept_correct[condition] if model_correct else accept_wrong[condition]
            accept = rng.random() < p_accept
            decision = model if accept else ("not_coordinated" if model == "coordinated" else "coordinated")
            correct = decision == truth
            confidence = rng.randint(68, 91) if correct else rng.randint(52, 76)
            # Short real waits exercise the exposure/response timing path. The
            # report labels this as simulated latency, not human response time.
            time.sleep(base_latency[condition] / 100.0)
            request("POST", f"/api/session/{session}/response", {
                "trial_index": case["trial_index"],
                "case_id": case["case_id"],
                "decision": decision,
                "confidence": confidence,
                "rationale": "Synthetic expert pilot; not a human response.",
            })
            records.append({
                "participant": participant,
                "condition": condition,
                "case_id": case["case_id"],
                "ground_truth": truth,
                "model_recommendation": model,
                "model_correct": int(model_correct),
                "decision": decision,
                "correct": int(correct),
                "accepted_model": int(accept),
                "appropriate_reliance": int((model_correct and accept) or ((not model_correct) and (not accept))),
                "confidence": confidence,
                "simulated_latency_seconds": base_latency[condition],
            })
        if condition is None:
            raise RuntimeError(f"participant {participant} received no trials")
        request("POST", f"/api/session/{session}/questionnaire", {
            "trust": {"risk_only": 4, "standard_signals": 5, "hypertrace_evidence": 6}[condition],
            "clarity": {"risk_only": 4, "standard_signals": 5, "hypertrace_evidence": 6}[condition],
            "workload": {"risk_only": 3, "standard_signals": 4, "hypertrace_evidence": 5}[condition],
            "evidence_usefulness": {"risk_only": 3, "standard_signals": 5, "hypertrace_evidence": 6}[condition],
            "feedback": "Synthetic expert pilot; not a human response.",
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    participants = sorted({row["participant"] for row in records})
    if len(participants) != TARGET_PARTICIPANTS:
        raise RuntimeError(
            f"expected {TARGET_PARTICIPANTS} synthetic participants, found {len(participants)}"
        )
    conditions = {
        condition: len({
            row["participant"] for row in records if row["condition"] == condition
        })
        for condition in ("risk_only", "standard_signals", "hypertrace_evidence")
    }
    summary = {}
    for condition in conditions:
        subset = [row for row in records if row["condition"] == condition]
        wrong = [row for row in subset if not row["model_correct"]]
        correct_model = [row for row in subset if row["model_correct"]]
        summary[condition] = {
            "participants": len({row["participant"] for row in subset}),
            "trials": len(subset),
            "accuracy": mean([row["correct"] for row in subset]),
            "error_rejection": mean([1 - row["accepted_model"] for row in wrong]),
            "correct_acceptance": mean([row["accepted_model"] for row in correct_model]),
            "median_latency_seconds": round(statistics.median(row["simulated_latency_seconds"] for row in subset), 2),
            "mean_confidence_correct": mean([row["confidence"] for row in subset if row["correct"]]),
            "mean_confidence_incorrect": mean([row["confidence"] for row in subset if not row["correct"]]),
        }

    wrong_all = [row for row in records if not row["model_correct"]]
    correct_model_all = [row for row in records if row["model_correct"]]
    overall = {
        "trials": len(records),
        "accuracy": mean([row["correct"] for row in records]),
        "error_rejection": mean([1 - row["accepted_model"] for row in wrong_all]),
        "correct_acceptance": mean([row["accepted_model"] for row in correct_model_all]),
        "appropriate_reliance_equals_accuracy": all(
            row["appropriate_reliance"] == row["correct"] for row in records
        ),
    }

    output = {
        "schema_version": "hypertrace.synthetic-experts-pilot.v2",
        "status": "synthetic_pilot_only",
        "human_subject_evidence": False,
        "participant_count": TARGET_PARTICIPANTS,
        "trials_per_participant": TRIALS_PER_PARTICIPANT,
        "case_package_sha256": hashlib.sha256(CASE_FILE.read_bytes()).hexdigest(),
        "condition_counts": conditions,
        "metrics": summary,
        "overall": overall,
        "formal_study_protocol": {
            "participants_planned": 20,
            "cases_per_participant": 8,
            "stages": [
                "unaided_judgment",
                "assigned_model_assistance",
                "final_judgment",
            ],
            "matched_by_this_legacy_pilot": False,
        },
    }
    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [
        "# HyperTrace Synthetic Expert Pilot",
        "",
        "> This is a software simulation for front-end and metric validation. It is not human-subject evidence and must not be presented as a CHI user study.",
        "",
        "## Protocol",
        "",
        f"- {TARGET_PARTICIPANTS} synthetic participants (`RE-01`-`RE-{TARGET_PARTICIPANTS:02d}`).",
        f"- {TRIALS_PER_PARTICIPANT} trials per participant; the 24-case package and its hidden labels were used.",
        "- The primary pilot outcomes are final accuracy, erroneous-recommendation rejection, and correct-recommendation acceptance.",
        "- Latency and confidence are secondary behavioral checks; questionnaire items are retained only as pilot signals.",
        "- This legacy pilot does not implement the final 8-case, three-stage human-study protocol.",
        "",
        "## Results",
        "",
        "| Condition | N | Trials | Accuracy | Wrong-AI rejection | Correct-AI acceptance | Median latency (sim.) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in summary.items():
        lines.append(
            f"| {condition} | {values['participants']} | {values['trials']} | {values['accuracy']:.1%} | {values['error_rejection']:.1%} | {values['correct_acceptance']:.1%} | {values['median_latency_seconds']:.1f}s |"
        )
    lines += [
        "",
        "## Aggregate software-check metrics",
        "",
        f"- Final accuracy: {overall['accuracy']:.1%}",
        f"- Wrong-AI rejection: {overall['error_rejection']:.1%}",
        f"- Correct-AI acceptance: {overall['correct_acceptance']:.1%}",
        "- The legacy binary `appropriate_reliance` field is exactly identical to final accuracy and is therefore not reported as a separate outcome.",
        "",
        "## Interpretation boundary",
        "",
        "The condition differences are generated by the simulation policy, not measured user effects. Use this report only to verify data flow, export fields, and analysis code. It cannot be used as evidence for RQ3, effect sizes, significance tests, the abstract, or any CHI human-subject claim.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def _load_existing_records() -> list[dict]:
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        return []
    integer_fields = {
        "model_correct",
        "correct",
        "accepted_model",
        "appropriate_reliance",
        "confidence",
    }
    float_fields = {"simulated_latency_seconds"}
    with OUT_CSV.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    for row in records:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return records


def _archive_original_fifteen(records: list[dict]) -> None:
    participants = {row["participant"] for row in records}
    if len(participants) != 15:
        return
    archive = DATA / "synthetic_pilot_archive" / "15_participants_20260815"
    archive.mkdir(parents=True, exist_ok=True)
    for path in (OUT_JSON, OUT_CSV, OUT_REPORT, DATA / "hypertrace_synthetic_pilot.sqlite3"):
        target = archive / path.name
        if path.exists() and not target.exists():
            shutil.copy2(path, target)


if __name__ == "__main__":
    main()
