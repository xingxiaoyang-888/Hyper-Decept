"""Exercise the formal three-stage study protocol with synthetic reviewers.

The generated records are for analysis/template validation only. They are not
human-subject observations and must never be reported as empirical findings.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASE = "http://127.0.0.1:8766"
CASE_FILE = DATA / "study_cases.private.json"
DB_FILE = DATA / "hypertrace_synthetic_three_stage.sqlite3"
OUT_JSON = DATA / "synthetic_experts_three_stage.json"
OUT_CSV = DATA / "synthetic_experts_three_stage.csv"
OUT_REPORT = DATA / "synthetic_experts_three_stage_report.md"
TARGET_PARTICIPANTS = 20
TRIALS_PER_PARTICIPANT = 8
CONDITIONS = ("risk_only", "standard_signals", "hypertrace_evidence")
SEED = 20260827


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


def opposite(decision: str) -> str:
    return "not_coordinated" if decision == "coordinated" else "coordinated"


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def mean_defined(rows: list[dict], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and not pd.isna(row[field])
    ]
    return statistics.mean(values) if values else math.nan


def cluster_bootstrap_difference(
    records: list[dict], field: str, treatment: str, control: str, seed: int
) -> tuple[float, tuple[float, float]]:
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if (
            row["condition"] in {treatment, control}
            and row.get(field) is not None
            and not pd.isna(row[field])
        ):
            grouped[row["condition"]][row["participant"]].append(row)

    observed = mean_defined(
        [row for row in records if row["condition"] == treatment], field
    ) - mean_defined([row for row in records if row["condition"] == control], field)
    draws = []
    for _ in range(10000):
        condition_means = {}
        for condition in (treatment, control):
            participants = sorted(grouped[condition])
            sampled = [rng.choice(participants) for _ in participants]
            sampled_rows = [row for participant in sampled for row in grouped[condition][participant]]
            condition_means[condition] = mean_defined(sampled_rows, field)
        draws.append(condition_means[treatment] - condition_means[control])
    return observed, (quantile(draws, 0.025), quantile(draws, 0.975))


def cluster_bootstrap_median_ratio(
    records: list[dict], treatment: str, control: str, seed: int
) -> tuple[float, tuple[float, float]]:
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["condition"] in {treatment, control}:
            grouped[row["condition"]][row["participant"]].append(row)
    observed = statistics.median(
        row["review_time_seconds"] for row in records if row["condition"] == treatment
    ) / statistics.median(
        row["review_time_seconds"] for row in records if row["condition"] == control
    )
    draws = []
    for _ in range(10000):
        medians = {}
        for condition in (treatment, control):
            participants = sorted(grouped[condition])
            sampled = [rng.choice(participants) for _ in participants]
            values = [
                row["review_time_seconds"]
                for participant in sampled
                for row in grouped[condition][participant]
            ]
            medians[condition] = statistics.median(values)
        draws.append(medians[treatment] / medians[control])
    return observed, (quantile(draws, 0.025), quantile(draws, 0.975))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def fit_binary_mixed_model(frame: pd.DataFrame) -> dict:
    formula = (
        "final_correct ~ C(condition, Treatment(reference='risk_only')) "
        "* model_correct + initial_correct + C(operation)"
    )
    result = smf.glm(
        formula,
        data=frame,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["participant"], "use_correction": True},
    )
    names = list(result.params.index)
    params = np.asarray(result.params)
    covariance = np.asarray(result.cov_params())
    indices = [index for index, name in enumerate(names) if ":model_correct" in name]
    beta = params[indices]
    cov = covariance[np.ix_(indices, indices)]
    chi2 = float(beta.T @ np.linalg.pinv(cov) @ beta)
    omnibus_p = float(stats.chi2.sf(chi2, len(indices)))

    contrasts = {}
    interaction_indices = {}
    for condition, token in (
        ("standard_signals", "standard_signals"),
        ("hypertrace_evidence", "hypertrace_evidence"),
    ):
        index = next(
            i for i, name in enumerate(names)
            if token in name and ":model_correct" in name
        )
        interaction_indices[condition] = index
        estimate = params[index]
        standard_error = math.sqrt(max(covariance[index, index], 0.0))
        contrasts[condition] = {
            "aor": math.exp(estimate),
            "ci": (
                math.exp(estimate - 1.96 * standard_error),
                math.exp(estimate + 1.96 * standard_error),
            ),
        }
    ht_index = interaction_indices["hypertrace_evidence"]
    standard_index = interaction_indices["standard_signals"]
    estimate = params[ht_index] - params[standard_index]
    variance = (
        covariance[ht_index, ht_index]
        + covariance[standard_index, standard_index]
        - 2 * covariance[ht_index, standard_index]
    )
    standard_error = math.sqrt(max(variance, 0.0))
    contrasts["hypertrace_vs_standard"] = {
        "aor": math.exp(estimate),
        "ci": (
            math.exp(estimate - 1.96 * standard_error),
            math.exp(estimate + 1.96 * standard_error),
        ),
    }
    return {"chi2": chi2, "p": omnibus_p, "contrasts": contrasts}


def fit_time_model(frame: pd.DataFrame) -> dict:
    fit_frame = frame.copy()
    fit_frame["log_time"] = np.log(fit_frame["review_time_seconds"])
    model = MixedLM.from_formula(
        "log_time ~ C(condition, Treatment(reference='risk_only'))",
        groups="participant",
        vc_formula={"case": "0 + C(case_id)"},
        data=fit_frame,
    )
    result = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    output = {}
    for condition, token in (
        ("standard_signals", "standard_signals"),
        ("hypertrace_evidence", "hypertrace_evidence"),
    ):
        name = next(name for name in result.fe_params.index if token in name)
        estimate = float(result.fe_params[name])
        standard_error = float(result.bse_fe[name])
        output[condition] = {
            "ratio": math.exp(estimate),
            "ci": (
                math.exp(estimate - 1.96 * standard_error),
                math.exp(estimate + 1.96 * standard_error),
            ),
            "p": float(2 * stats.norm.sf(abs(estimate / standard_error))),
        }
    standard_name = next(name for name in result.fe_params.index if "standard_signals" in name)
    hypertrace_name = next(name for name in result.fe_params.index if "hypertrace_evidence" in name)
    estimate = float(result.fe_params[hypertrace_name] - result.fe_params[standard_name])
    covariance = result.cov_params()
    variance = (
        covariance.loc[hypertrace_name, hypertrace_name]
        + covariance.loc[standard_name, standard_name]
        - 2 * covariance.loc[hypertrace_name, standard_name]
    )
    standard_error = math.sqrt(max(float(variance), 0.0))
    output["hypertrace_vs_standard"] = {
        "ratio": math.exp(estimate),
        "ci": (
            math.exp(estimate - 1.96 * standard_error),
            math.exp(estimate + 1.96 * standard_error),
        ),
        "p": float(2 * stats.norm.sf(abs(estimate / standard_error))),
    }
    return output


def fit_secondary_association(frame: pd.DataFrame, field: str, scale: float) -> dict:
    selected = frame[frame["condition"] == "hypertrace_evidence"].copy()
    selected["metric_scaled"] = (selected[field] - selected[field].mean()) / scale
    result = smf.glm(
        "final_correct ~ initial_correct + model_correct + metric_scaled + C(operation)",
        data=selected,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": selected["participant"], "use_correction": True},
    )
    names = list(result.params.index)
    index = names.index("metric_scaled")
    estimate = float(result.params.iloc[index])
    covariance = np.asarray(result.cov_params())
    standard_error = math.sqrt(max(covariance[index, index], 0.0))
    z_value = estimate / standard_error
    return {
        "aor": math.exp(estimate),
        "ci": (
            math.exp(estimate - 1.96 * standard_error),
            math.exp(estimate + 1.96 * standard_error),
        ),
        "p": float(2 * stats.norm.sf(abs(z_value))),
    }


def run_protocol(cases: dict[str, dict]) -> tuple[list[dict], dict[str, int]]:
    if DB_FILE.exists():
        DB_FILE.unlink()
    env = os.environ.copy()
    env.update(
        {
            "HYPERTRACE_CASES_PATH": str(CASE_FILE),
            "HYPERTRACE_DB_PATH": str(DB_FILE),
            "STUDY_SALT": "synthetic-three-stage-salt",
            "ADMIN_TOKEN": "synthetic-three-stage-admin",
            "HYPERTRACE_DURABLE_STORAGE": "1",
            "HYPERTRACE_PREVIEW_MODE": "0",
            "HYPERTRACE_TRIAL_COUNT": str(TRIALS_PER_PARTICIPANT),
        }
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8766"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    records: list[dict] = []
    workloads: dict[str, int] = {}
    try:
        for _ in range(50):
            try:
                request("GET", "/api/health")
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("synthetic three-stage server did not start")

        for expert_index in range(1, TARGET_PARTICIPANTS + 1):
            participant = f"SIM-{expert_index:02d}"
            rng = random.Random(SEED + expert_index * 1009)
            skill = rng.gauss(0.0, 0.055)
            session = request(
                "POST",
                "/api/session",
                {"participant_code": participant, "consent": True, "age_confirmed": True},
            )["session_id"]
            condition = None
            participant_time_factor = math.exp(rng.gauss(0.0, 0.10))
            for _ in range(TRIALS_PER_PARTICIPANT):
                trial = request("GET", f"/api/session/{session}/trial")
                public_case = trial["case"]
                case = cases[public_case["case_id"]]
                condition = request_condition = None
                # Condition is intentionally unavailable before the initial response.
                case_noise = (int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF - 0.5) * 0.18
                initial_probability = min(0.88, max(0.50, 0.70 + skill + case_noise))
                initial_correct = rng.random() < initial_probability
                truth = case["ground_truth"]
                initial_decision = truth if initial_correct else opposite(truth)
                initial_confidence = int(round(min(94, max(45, 61 + 15 * initial_correct + rng.gauss(0, 7)))))
                request(
                    "POST",
                    f"/api/session/{session}/initial-response",
                    {
                        "trial_index": public_case["trial_index"],
                        "case_id": case["case_id"],
                        "decision": initial_decision,
                        "confidence": initial_confidence,
                    },
                )
                assisted = request(
                    "POST",
                    f"/api/session/{session}/reveal",
                    {"trial_index": public_case["trial_index"], "case_id": case["case_id"]},
                )["case"]
                condition = request_condition = assisted["view_mode"]
                model = case["model_recommendation"]
                model_correct = model == truth
                if model_correct and initial_correct:
                    final_correct_probability = {
                        "risk_only": 0.96,
                        "standard_signals": 0.97,
                        "hypertrace_evidence": 0.99,
                    }[request_condition]
                elif model_correct and not initial_correct:
                    final_correct_probability = {
                        "risk_only": 0.72,
                        "standard_signals": 0.82,
                        "hypertrace_evidence": 0.94,
                    }[request_condition]
                elif not model_correct and initial_correct:
                    final_correct_probability = {
                        "risk_only": 0.36,
                        "standard_signals": 0.58,
                        "hypertrace_evidence": 0.88,
                    }[request_condition]
                else:
                    final_correct_probability = {
                        "risk_only": 0.08,
                        "standard_signals": 0.12,
                        "hypertrace_evidence": 0.25,
                    }[request_condition]
                final_correct = rng.random() < final_correct_probability
                final_decision = truth if final_correct else opposite(truth)
                confidence_bonus = {"risk_only": 1, "standard_signals": 3, "hypertrace_evidence": 5}[request_condition]
                final_confidence = int(round(min(97, max(45, 62 + 15 * final_correct + confidence_bonus + rng.gauss(0, 6)))))
                median_time = {"risk_only": 48.0, "standard_signals": 72.0, "hypertrace_evidence": 104.0}[request_condition]
                case_time_factor = 1.0 + (int(hashlib.sha256(case["case_id"].encode()).hexdigest()[8:16], 16) / 0xFFFFFFFF - 0.5) * 0.20
                review_time = median_time * participant_time_factor * case_time_factor * math.exp(rng.gauss(0.0, 0.18))
                request(
                    "POST",
                    f"/api/session/{session}/response",
                    {
                        "trial_index": public_case["trial_index"],
                        "case_id": case["case_id"],
                        "decision": final_decision,
                        "confidence": final_confidence,
                        "rationale": "Synthetic three-stage protocol check; not a human response.",
                    },
                )
                explanation = case.get("explanation", {})
                rair_eligible = model_correct and not initial_correct
                rsr_eligible = (not model_correct) and initial_correct
                records.append(
                    {
                        "participant": participant,
                        "condition": request_condition,
                        "case_id": case["case_id"],
                        "operation": case["operation"],
                        "ground_truth": truth,
                        "model_recommendation": model,
                        "model_correct": int(model_correct),
                        "initial_decision": initial_decision,
                        "initial_correct": int(initial_correct),
                        "initial_confidence": initial_confidence,
                        "final_decision": final_decision,
                        "final_correct": int(final_correct),
                        "final_confidence": final_confidence,
                        "confidence_change": final_confidence - initial_confidence,
                        "final_accepted_model": int(final_decision == model),
                        "correct_ai_acceptance": int(final_decision == model) if model_correct else None,
                        "wrong_ai_rejection": int(final_decision != model) if not model_correct else None,
                        "rair_eligible": int(rair_eligible),
                        "rair": int(final_correct) if rair_eligible else None,
                        "rsr_eligible": int(rsr_eligible),
                        "rsr": int(final_correct) if rsr_eligible else None,
                        "review_time_seconds": round(review_time, 3),
                        "geometry_fidelity": explanation.get("geometry_fidelity"),
                        "sufficiency_error": explanation.get("sufficiency_error"),
                    }
                )
            if condition is None:
                raise RuntimeError(f"participant {participant} received no trials")
            workload_center = {"risk_only": 3.0, "standard_signals": 3.8, "hypertrace_evidence": 4.6}[condition]
            workload = int(round(min(7, max(1, workload_center + rng.gauss(0, 0.65)))))
            workloads[participant] = workload
            request(
                "POST",
                f"/api/session/{session}/questionnaire",
                {
                    "trust": {"risk_only": 4, "standard_signals": 5, "hypertrace_evidence": 5}[condition],
                    "clarity": {"risk_only": 4, "standard_signals": 5, "hypertrace_evidence": 6}[condition],
                    "workload": workload,
                    "evidence_usefulness": {"risk_only": 3, "standard_signals": 5, "hypertrace_evidence": 6}[condition],
                    "feedback": "Synthetic three-stage protocol check; not a human response.",
                },
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
    return records, workloads


def summarize(records: list[dict], workloads: dict[str, int]) -> dict:
    frame = pd.DataFrame(records)
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        participants = sorted({row["participant"] for row in rows})
        times = [row["review_time_seconds"] for row in rows]
        workload_values = [workloads[participant] for participant in participants]
        summary[condition] = {
            "participants": len(participants),
            "trials": len(rows),
            "initial_accuracy": mean_defined(rows, "initial_correct"),
            "final_accuracy": mean_defined(rows, "final_correct"),
            "correct_ai_acceptance": mean_defined(rows, "correct_ai_acceptance"),
            "wrong_ai_rejection": mean_defined(rows, "wrong_ai_rejection"),
            "rair": mean_defined(rows, "rair"),
            "rair_n": sum(row["rair_eligible"] for row in rows),
            "rsr": mean_defined(rows, "rsr"),
            "rsr_n": sum(row["rsr_eligible"] for row in rows),
            "median_time": statistics.median(times),
            "time_iqr": (quantile(times, 0.25), quantile(times, 0.75)),
            "workload_mean": statistics.mean(workload_values),
            "workload_sd": statistics.stdev(workload_values),
            "confidence_change": mean_defined(rows, "confidence_change"),
        }

    final_differences = {}
    rejection_differences = {}
    for control, offset in (("risk_only", 1), ("standard_signals", 2)):
        diff, ci = cluster_bootstrap_difference(
            records, "final_correct", "hypertrace_evidence", control, SEED + offset
        )
        final_differences[control] = {"difference": diff, "ci": ci}
        diff, ci = cluster_bootstrap_difference(
            records, "wrong_ai_rejection", "hypertrace_evidence", control, SEED + 10 + offset
        )
        rejection_differences[control] = {"difference": diff, "ci": ci}

    raw_p = []
    for control in ("risk_only", "standard_signals"):
        participant_rates = defaultdict(list)
        for row in records:
            if (
                row["condition"] in {"hypertrace_evidence", control}
                and row["wrong_ai_rejection"] is not None
                and not pd.isna(row["wrong_ai_rejection"])
            ):
                participant_rates[(row["condition"], row["participant"])].append(row["wrong_ai_rejection"])
        ht = [statistics.mean(values) for (condition, _), values in participant_rates.items() if condition == "hypertrace_evidence"]
        comparison = [statistics.mean(values) for (condition, _), values in participant_rates.items() if condition == control]
        raw_p.append(float(stats.mannwhitneyu(ht, comparison, alternative="two-sided").pvalue))
    adjusted = holm_adjust(raw_p)
    for control, p_value in zip(("risk_only", "standard_signals"), adjusted):
        rejection_differences[control]["holm_p"] = p_value

    time_ratios = {}
    for control, offset in (("risk_only", 1), ("standard_signals", 2)):
        ratio, ci = cluster_bootstrap_median_ratio(
            records, "hypertrace_evidence", control, SEED + 20 + offset
        )
        time_ratios[control] = {"ratio": ratio, "bootstrap_ci": ci}

    mixed_binary = fit_binary_mixed_model(frame)
    mixed_time = fit_time_model(frame)
    geometry = fit_secondary_association(frame, "geometry_fidelity", 0.01)
    sufficiency = fit_secondary_association(frame, "sufficiency_error", 0.001)
    return {
        "by_condition": summary,
        "final_accuracy_differences": final_differences,
        "wrong_rejection_differences": rejection_differences,
        "median_time_ratios": time_ratios,
        "binary_mixed_model": mixed_binary,
        "time_mixed_model": mixed_time,
        "secondary_associations": {"geometry": geometry, "sufficiency": sufficiency},
    }


def write_outputs(records: list[dict], workloads: dict[str, int], results: dict) -> None:
    for row in records:
        row["workload"] = workloads[row["participant"]]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "schema_version": "hypertrace.synthetic-experts.three-stage.v3",
        "status": "synthetic_template_validation_only",
        "human_subject_evidence": False,
        "participant_count": TARGET_PARTICIPANTS,
        "trials_per_participant": TRIALS_PER_PARTICIPANT,
        "trial_count": len(records),
        "case_package_sha256": hashlib.sha256(CASE_FILE.read_bytes()).hexdigest(),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# HyperTrace Three-Stage Synthetic Reviewer Check",
        "",
        "> Synthetic template-validation data only. These values are generated by the policy encoded in `run_synthetic_experts_three_stage.py`; they are not human-subject findings.",
        "",
        "## Protocol",
        "",
        f"- {TARGET_PARTICIPANTS} synthetic reviewers, {TRIALS_PER_PARTICIPANT} cases each, {len(records)} reviews total.",
        "- Initial model-blind judgment, assistance reveal, and final judgment were recorded for every trial.",
        "- Every assigned block contained five correct and three incorrect model recommendations.",
        "",
        "## Descriptive results",
        "",
        "| Condition | N | Trials | Initial accuracy | Final accuracy | Correct-AI acceptance | Wrong-AI rejection | RAIR | RSR | Median time (IQR) | Workload mean (SD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        value = results["by_condition"][condition]
        lines.append(
            f"| {condition} | {value['participants']} | {value['trials']} | {value['initial_accuracy']:.1%} | {value['final_accuracy']:.1%} | {value['correct_ai_acceptance']:.1%} | {value['wrong_ai_rejection']:.1%} | {value['rair']:.1%} (n={value['rair_n']}) | {value['rsr']:.1%} (n={value['rsr_n']}) | {value['median_time']:.1f}s ({value['time_iqr'][0]:.1f}-{value['time_iqr'][1]:.1f}) | {value['workload_mean']:.2f} ({value['workload_sd']:.2f}) |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The apparent condition effects and all inferential statistics are consequences of the simulation policy. They validate the analysis and manuscript fields only and must be replaced with genuine study estimates before submission.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    package = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in package["cases"]}
    records, workloads = run_protocol(cases)
    if len(records) != TARGET_PARTICIPANTS * TRIALS_PER_PARTICIPANT:
        raise RuntimeError(f"expected 160 records, found {len(records)}")
    results = summarize(records, workloads)
    write_outputs(records, workloads, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
