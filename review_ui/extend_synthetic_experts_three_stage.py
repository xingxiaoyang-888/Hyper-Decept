"""Extend the synthetic three-stage reviewer template from 20 to 30 participants.

This preserves the original 160 trial records and appends ten deterministic
synthetic participants.  The resulting artifacts are analysis-template data,
not human-subject observations.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from run_synthetic_experts_three_stage import (
    CONDITIONS,
    SEED,
    fit_binary_mixed_model,
    fit_secondary_association,
    fit_time_model,
    cluster_bootstrap_difference,
    cluster_bootstrap_median_ratio,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CASE_FILE = DATA / "study_cases.private.json"
OLD_CSV = DATA / "synthetic_experts_three_stage.csv"
OUT_CSV = DATA / "synthetic_experts_three_stage_30.csv"
OUT_JSON = DATA / "synthetic_experts_three_stage_30.json"
OUT_REPORT = DATA / "synthetic_experts_three_stage_30_report.md"
TRIALS_PER_PARTICIPANT = 8
NEW_CONDITIONS = (
    "risk_only", "risk_only", "risk_only", "risk_only",
    "standard_signals", "standard_signals", "standard_signals",
    "hypertrace_evidence", "hypertrace_evidence", "hypertrace_evidence",
)


def opposite(decision: str) -> str:
    return "not_coordinated" if decision == "coordinated" else "coordinated"


def choose_cases(cases: dict[str, dict], rng: random.Random) -> list[dict]:
    """Select 5 correct and 3 incorrect recommendations with 4/4 labels."""
    rows = list(cases.values())
    correct = [c for c in rows if c["ground_truth"] == c["model_recommendation"]]
    incorrect = [c for c in rows if c["ground_truth"] != c["model_recommendation"]]
    for _ in range(10000):
        selected = rng.sample(correct, 5) + rng.sample(incorrect, 3)
        if sum(c["ground_truth"] == "coordinated" for c in selected) != 4:
            continue
        if len({c["operation"] for c in selected}) < 2:
            continue
        rng.shuffle(selected)
        return selected
    raise RuntimeError("could not construct a balanced eight-case block")


def generate_participant(index: int, condition: str, cases: dict[str, dict]) -> tuple[list[dict], int]:
    rng = random.Random(SEED + index * 1009)
    participant = f"SIM-{index:02d}"
    skill = rng.gauss(0.0, 0.055)
    participant_time_factor = math.exp(rng.gauss(0.0, 0.10))
    rows: list[dict] = []
    for case in choose_cases(cases, rng):
        case_noise = (int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF - 0.5) * 0.18
        initial_probability = min(0.88, max(0.50, 0.70 + skill + case_noise))
        initial_correct = rng.random() < initial_probability
        truth = case["ground_truth"]
        model = case["model_recommendation"]
        initial_decision = truth if initial_correct else opposite(truth)
        initial_confidence = int(round(min(94, max(45, 61 + 15 * initial_correct + rng.gauss(0, 7)))))
        model_correct = model == truth
        if model_correct and initial_correct:
            final_probability = {"risk_only": .96, "standard_signals": .97, "hypertrace_evidence": .99}[condition]
        elif model_correct and not initial_correct:
            final_probability = {"risk_only": .72, "standard_signals": .82, "hypertrace_evidence": .94}[condition]
        elif not model_correct and initial_correct:
            final_probability = {"risk_only": .36, "standard_signals": .58, "hypertrace_evidence": .88}[condition]
        else:
            final_probability = {"risk_only": .08, "standard_signals": .12, "hypertrace_evidence": .25}[condition]
        final_correct = rng.random() < final_probability
        final_decision = truth if final_correct else opposite(truth)
        bonus = {"risk_only": 1, "standard_signals": 3, "hypertrace_evidence": 5}[condition]
        final_confidence = int(round(min(97, max(45, 62 + 15 * final_correct + bonus + rng.gauss(0, 6)))))
        median_time = {"risk_only": 48.0, "standard_signals": 72.0, "hypertrace_evidence": 104.0}[condition]
        case_factor = 1.0 + (int(hashlib.sha256(case["case_id"].encode()).hexdigest()[8:16], 16) / 0xFFFFFFFF - 0.5) * 0.20
        review_time = median_time * participant_time_factor * case_factor * math.exp(rng.gauss(0.0, 0.18))
        explanation = case.get("explanation", {})
        rair_eligible = model_correct and not initial_correct
        rsr_eligible = (not model_correct) and initial_correct
        rows.append({
            "participant": participant, "condition": condition,
            "case_id": case["case_id"], "operation": case["operation"],
            "ground_truth": truth, "model_recommendation": model,
            "model_correct": int(model_correct), "initial_decision": initial_decision,
            "initial_correct": int(initial_correct), "initial_confidence": initial_confidence,
            "final_decision": final_decision, "final_correct": int(final_correct),
            "final_confidence": final_confidence,
            "confidence_change": final_confidence - initial_confidence,
            "final_accepted_model": int(final_decision == model),
            "correct_ai_acceptance": int(final_decision == model) if model_correct else None,
            "wrong_ai_rejection": int(final_decision != model) if not model_correct else None,
            "rair_eligible": int(rair_eligible), "rair": int(final_correct) if rair_eligible else None,
            "rsr_eligible": int(rsr_eligible), "rsr": int(final_correct) if rsr_eligible else None,
            "review_time_seconds": round(review_time, 3),
            "geometry_fidelity": explanation.get("geometry_fidelity"),
            "sufficiency_error": explanation.get("sufficiency_error"),
        })
    workload_center = {"risk_only": 3.0, "standard_signals": 3.8, "hypertrace_evidence": 4.6}[condition]
    workload = int(round(min(7, max(1, workload_center + rng.gauss(0, 0.65)))))
    for row in rows:
        row["workload"] = workload
    return rows, workload


def participant_points(frame: pd.DataFrame) -> list[dict]:
    result = []
    for (condition, participant), group in frame.groupby(["condition", "participant"], sort=True):
        result.append({
            "condition": condition, "participant": participant,
            "initial_accuracy": float(group.initial_correct.mean()),
            "final_accuracy": float(group.final_correct.mean()),
            "wrong_ai_rejection": float(group.wrong_ai_rejection.dropna().mean()) if group.wrong_ai_rejection.notna().any() else None,
            "confidence_change": float(group.confidence_change.mean()),
            "median_latency": float(group.review_time_seconds.median()),
            "workload": float(group.workload.iloc[0]),
        })
    return result


def summarize_all(frame: pd.DataFrame) -> dict:
    summary = {}
    for condition in CONDITIONS:
        g = frame[frame.condition == condition]
        correct = g[g.model_correct == 1]
        wrong = g[g.model_correct == 0]
        rair = g[g.rair_eligible == 1]
        rsr = g[g.rsr_eligible == 1]
        summary[condition] = {
            "participants": int(g.participant.nunique()), "trials": len(g),
            "initial_correct_n": int(g.initial_correct.sum()), "final_correct_n": int(g.final_correct.sum()),
            "initial_accuracy": float(g.initial_correct.mean()), "final_accuracy": float(g.final_correct.mean()),
            "improvement": float(g.final_correct.mean() - g.initial_correct.mean()),
            "correct_recommendation_trials": len(correct), "incorrect_recommendation_trials": len(wrong),
            "correct_ai_acceptance": float(correct.final_accepted_model.mean()),
            "incorrect_ai_rejection": float((wrong.final_accepted_model == 0).mean()),
            "rair": float(rair.rair.mean()) if len(rair) else None, "rair_n": len(rair),
            "rsr": float(rsr.rsr.mean()) if len(rsr) else None, "rsr_n": len(rsr),
            "latency_median": float(g.review_time_seconds.median()),
            "latency_iqr": [float(g.review_time_seconds.quantile(.25)), float(g.review_time_seconds.quantile(.75))],
            "mental_effort_mean": float(g.groupby("participant").workload.first().mean()),
            "mental_effort_sd": float(g.groupby("participant").workload.first().std(ddof=1)),
            "confidence_change_mean": float(g.confidence_change.mean()),
            "confidence_change_sd": float(g.confidence_change.std(ddof=1)),
        }
    diffs = {}
    rejection = {}
    ratios = {}
    condition_bootstrap = {}
    records = frame.to_dict("records")
    for control, offset in (("risk_only", 1), ("standard_signals", 2)):
        d, ci = cluster_bootstrap_difference(records, "final_correct", "hypertrace_evidence", control, SEED + offset)
        diffs[control] = {"difference": d, "ci": ci}
        d, ci = cluster_bootstrap_difference(records, "wrong_ai_rejection", "hypertrace_evidence", control, SEED + 10 + offset)
        rejection[control] = {"difference": d, "ci": ci}
        ratio, ci = cluster_bootstrap_median_ratio(records, "hypertrace_evidence", control, SEED + 20 + offset)
        ratios[control] = {"ratio": ratio, "ci": ci}
    for condition in CONDITIONS:
        subset = [row for row in records if row["condition"] == condition]
        participants = sorted({row["participant"] for row in subset})
        grouped = {p: [row for row in subset if row["participant"] == p] for p in participants}
        rng = random.Random(SEED + 100 + CONDITIONS.index(condition))
        draws = {key: [] for key in ("initial_accuracy", "final_accuracy", "incorrect_rejection")}
        for _ in range(10000):
            sampled = [rng.choice(participants) for _ in participants]
            rows = [row for p in sampled for row in grouped[p]]
            draws["initial_accuracy"].append(float(np.mean([row["initial_correct"] for row in rows])))
            draws["final_accuracy"].append(float(np.mean([row["final_correct"] for row in rows])))
            wrong = [row["wrong_ai_rejection"] for row in rows if row["wrong_ai_rejection"] is not None and not pd.isna(row["wrong_ai_rejection"])]
            draws["incorrect_rejection"].append(float(np.mean(wrong)))
        condition_bootstrap[condition] = {key: {"ci": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]} for key, values in draws.items()}
    binary = fit_binary_mixed_model(frame)
    time = fit_time_model(frame)
    geometry = fit_secondary_association(frame, "geometry_fidelity", .01)
    sufficiency = fit_secondary_association(frame, "sufficiency_error", .001)
    return {"by_condition": summary, "condition_bootstrap_ci": condition_bootstrap, "final_accuracy_differences": diffs,
            "incorrect_rejection_differences": rejection, "latency_ratios": ratios,
            "binary_model": binary, "time_model": time,
            "secondary_associations": {"geometry": geometry, "sufficiency": sufficiency},
            "participant_points": participant_points(frame)}


def main() -> None:
    old = pd.read_csv(OLD_CSV)
    if len(old) != 160 or old.participant.nunique() != 20:
        raise RuntimeError("expected the existing 20-participant, 160-trial source CSV")
    package = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in package["cases"]}
    additions = []
    for offset, condition in enumerate(NEW_CONDITIONS, start=21):
        rows, _ = generate_participant(offset, condition, cases)
        additions.extend(rows)
    combined = pd.concat([old, pd.DataFrame(additions)], ignore_index=True)
    if len(combined) != 240 or combined.participant.nunique() != 30:
        raise RuntimeError("combined data must contain 30 participants and 240 trials")
    counts = combined.groupby("condition").participant.nunique().to_dict()
    trial_counts = combined.groupby("condition").size().to_dict()
    if counts != {"risk_only": 10, "standard_signals": 10, "hypertrace_evidence": 10}:
        raise RuntimeError(f"unexpected participant allocation: {counts}")
    if trial_counts != {"risk_only": 80, "standard_signals": 80, "hypertrace_evidence": 80}:
        raise RuntimeError(f"unexpected trial allocation: {trial_counts}")
    combined.to_csv(OUT_CSV, index=False)
    results = summarize_all(combined)
    pd.DataFrame(results["participant_points"]).to_csv(DATA / "synthetic_experts_three_stage_30_participant_points.csv", index=False)
    error_rows = []
    for condition in CONDITIONS:
        for metric, estimate_key in (("initial_accuracy", "initial_accuracy"), ("final_accuracy", "final_accuracy"), ("incorrect_rejection", "incorrect_ai_rejection")):
            error_rows.append({"condition": condition, "metric": metric, "estimate": results["by_condition"][condition][estimate_key], "ci95_low": results["condition_bootstrap_ci"][condition][metric]["ci"][0], "ci95_high": results["condition_bootstrap_ci"][condition][metric]["ci"][1], "bootstrap_draws": 10000})
    pd.DataFrame(error_rows).to_csv(DATA / "synthetic_experts_three_stage_30_error_bars.csv", index=False)
    payload = {"schema_version": "hypertrace.synthetic-experts.three-stage.v4",
               "status": "synthetic_template_validation_only",
               "human_subject_evidence": False, "participant_count": 30,
               "trials_per_participant": 8, "trial_count": 240,
               "source_existing_csv": str(OLD_CSV), "results": results}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# HyperTrace Three-Stage Synthetic Reviewer Check (30 participants)", "",
             "> Synthetic template-validation data only; not human-subject findings.", "",
             "All values below were recomputed from the combined 240 trial-level records; no old percentages were averaged.", "",
             "## Descriptive results", "",
             "| Condition | Participants | Reviews | Initial | Final | Improvement | Correct-AI acceptance (50) | Incorrect-AI rejection (30) | RAIR (n) | RSR (n) | Latency median (IQR) | Workload mean (SD) | Confidence change (SD) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        x = results["by_condition"][c]
        lines.append(f"| {c} | {x['participants']} | {x['trials']} | {x['initial_accuracy']:.1%} | {x['final_accuracy']:.1%} | {x['improvement']:+.1%} | {x['correct_ai_acceptance']:.1%} | {x['incorrect_ai_rejection']:.1%} | {x['rair']:.1%} ({x['rair_n']}) | {x['rsr']:.1%} ({x['rsr_n']}) | {x['latency_median']:.1f}s ({x['latency_iqr'][0]:.1f}-{x['latency_iqr'][1]:.1f}) | {x['mental_effort_mean']:.2f} ({x['mental_effort_sd']:.2f}) | {x['confidence_change_mean']:+.2f} ({x['confidence_change_sd']:.2f}) |")
    lines += ["", "## Inferential results", "", "```json", json.dumps({k: results[k] for k in ('condition_bootstrap_ci','final_accuracy_differences','incorrect_rejection_differences','latency_ratios','binary_model','time_model','secondary_associations')}, indent=2), "```", "", "## Participant-level data", "", "`synthetic_experts_three_stage_30_participant_points.csv` contains all participant-level points; `synthetic_experts_three_stage_30_error_bars.csv` contains 10,000-draw participant-cluster bootstrap intervals for plotted accuracy and rejection metrics.", "", "## Provenance", "", "The first 160 trial values are retained from the prior 20-participant CSV; rows 161-240 are the ten appended synthetic participants."]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "report": str(OUT_REPORT), "participants": 30, "reviews": 240, "allocation": counts}, indent=2))


if __name__ == "__main__":
    main()
