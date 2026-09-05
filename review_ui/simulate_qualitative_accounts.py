"""Generate synthetic qualitative accounts for Section 7.5 template checks.

This is a deterministic software simulation. The generated rationales,
comments, quotations, and coding counts are not participant data.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INPUT = Path(os.environ.get("QUAL_INPUT", DATA / "synthetic_experts_three_stage.csv"))
OUT_CSV = Path(os.environ.get("QUAL_OUT_CSV", DATA / "synthetic_experts_three_stage_qualitative.csv"))
OUT_JSON = Path(os.environ.get("QUAL_OUT_JSON", DATA / "synthetic_experts_three_stage_qualitative.json"))
OUT_REPORT = Path(os.environ.get("QUAL_OUT_REPORT", DATA / "synthetic_experts_three_stage_qualitative_report.md"))
SEED = 20260827


def choose(rng: random.Random, weighted: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in weighted)
    draw = rng.random() * total
    cumulative = 0.0
    for value, weight in weighted:
        cumulative += weight
        if draw <= cumulative:
            return value
    return weighted[-1][0]


def code_trial(row: dict, rng: random.Random) -> tuple[list[str], str]:
    condition = row["condition"]
    codes: list[str] = []
    if condition == "hypertrace_evidence":
        if rng.random() < 0.86:
            codes.append("sequence_reconstruction")
        if rng.random() < 0.82:
            codes.append("provenance_verification")
        if rng.random() < 0.72:
            codes.append("initial_final_comparison")
        if rng.random() < 0.57:
            codes.append("geometry_audit")
        if rng.random() < 0.48:
            codes.append("counterfactual_use")
    elif condition == "standard_signals":
        if rng.random() < 0.57:
            codes.append("sequence_reconstruction")
        if rng.random() < 0.0:
            codes.append("provenance_verification")
        if rng.random() < 0.54:
            codes.append("initial_final_comparison")
        if rng.random() < 0.0:
            codes.append("geometry_audit")
        if rng.random() < 0.0:
            codes.append("counterfactual_use")
    else:
        if rng.random() < 0.16:
            codes.append("sequence_reconstruction")
        if rng.random() < 0.0:
            codes.append("provenance_verification")
        if rng.random() < 0.47:
            codes.append("initial_final_comparison")

    if not codes:
        codes.append("risk_priority_only")

    # Failures are deliberately correlated with dense evidence and with
    # incorrect recommendations, but remain independent of the true label.
    overload_probability = {
        "risk_only": 0.04,
        "standard_signals": 0.11,
        "hypertrace_evidence": 0.19,
    }[condition]
    confusion_probability = {
        "risk_only": 0.0,
        "standard_signals": 0.0,
        "hypertrace_evidence": 0.08,
    }[condition]
    scope_probability = {
        "risk_only": 0.03,
        "standard_signals": 0.06,
        "hypertrace_evidence": 0.07,
    }[condition]
    unresolved_probability = {
        "risk_only": 0.02,
        "standard_signals": 0.08,
        "hypertrace_evidence": 0.12,
    }[condition]
    if rng.random() < overload_probability:
        codes.append("evidence_overload")
    if rng.random() < confusion_probability:
        codes.append("metric_confusion")
    if rng.random() < scope_probability:
        codes.append("scope_warning_missed")
    if rng.random() < unresolved_probability:
        codes.append("unresolved_evidence")

    rationale_parts = {
        "sequence_reconstruction": "I compared the order and timing of the observed interactions",
        "provenance_verification": "I checked the linked source records and timestamps",
        "initial_final_comparison": "I compared the recommendation with my initial judgment",
        "geometry_audit": "I used the geometry value as a check that the reduced packet preserved the detector structure",
        "counterfactual_use": "I used the keep-only or removed-edge view to see which relation affected the recommendation",
        "risk_priority_only": "I used the risk priority as a triage signal",
        "evidence_overload": "The number of evidence items made the comparison slower",
        "metric_confusion": "At first I treated the geometry value as evidence that the recommendation was correct",
        "scope_warning_missed": "I did not use the snapshot warning in this decision",
        "unresolved_evidence": "The available record did not resolve the conflicting event, so I qualified the recommendation",
    }
    ordered = [code for code in rationale_parts if code in codes]
    rationale = "; ".join(rationale_parts[code] for code in ordered) + "."
    return codes, rationale


def participant_comment(participant: str, condition: str, codes: list[str]) -> str:
    code_set = set(codes)
    if condition == "hypertrace_evidence":
        if "evidence_overload" in code_set:
            return "The timeline and source links were useful, but comparing every item at once was demanding."
        if "geometry_audit" in code_set:
            return "The geometry summary helped me check that the short evidence packet still represented the model's structure."
        return "I followed the timeline, checked the source identifiers, and then decided whether to accept the recommendation."
    if condition == "standard_signals":
        return "The relation and time summaries helped me compare the case with my initial judgment, although the underlying records were less direct."
    return "The priority score helped me decide which cases deserved attention, but it did not explain the coordination mechanism."


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(SEED)
    participant_codes: dict[str, list[str]] = defaultdict(list)
    coded_rows: list[dict] = []
    for row in rows:
        codes, rationale = code_trial(row, rng)
        participant_codes[row["participant"]].extend(codes)
        coded_rows.append({**row, "qualitative_codes": ";".join(codes), "synthetic_rationale": rationale})

    participant_conditions = {
        row["participant"]: row["condition"] for row in rows
    }
    comments = [
        {
            "participant": participant,
            "condition": participant_conditions[participant],
            "synthetic_comment": participant_comment(participant, participant_conditions[participant], codes),
            "participant_codes": ";".join(sorted(set(codes))),
        }
        for participant, codes in sorted(participant_codes.items())
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = list(coded_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coded_rows)

    trial_counts = Counter()
    participant_counts = Counter()
    for row in coded_rows:
        for code in row["qualitative_codes"].split(";"):
            trial_counts[code] += 1
    for participant, codes in participant_codes.items():
        for code in set(codes):
            participant_counts[code] += 1

    selected_codes = {
        "sequence_reconstruction",
        "provenance_verification",
        "initial_final_comparison",
        "geometry_audit",
        "counterfactual_use",
        "evidence_overload",
        "metric_confusion",
        "scope_warning_missed",
        "unresolved_evidence",
    }
    dominant_participants = len({
        row["participant"]
        for row in coded_rows
        if any(code in selected_codes for code in row["qualitative_codes"].split(";"))
    })
    dominant_rationales = sum(
        any(code in selected_codes for code in row["qualitative_codes"].split(";"))
        for row in coded_rows
    )
    summary = {
        "status": "synthetic_qualitative_template_validation_only",
        "human_subject_evidence": False,
        "trial_count": len(coded_rows),
        "participant_count": len(participant_codes),
        "nonempty_rationale_count": len(coded_rows),
        "nonempty_comment_count": len(comments),
        "dominant_practice_participants": dominant_participants,
        "dominant_practice_rationales": dominant_rationales,
        "trial_code_counts": dict(sorted(trial_counts.items())),
        "participant_code_counts": dict(sorted(participant_counts.items())),
        "quotation_bank": {
            "dominant_strategy": "I compared the order and timing of the observed interactions; I checked the linked source records and timestamps; I compared the recommendation with my initial judgment.",
            "provenance": "I checked the linked source records and timestamps.",
            "geometry": "The geometry summary helped me check that the short evidence packet still represented the model's structure.",
            "negative": "The timeline and source links were useful, but comparing every item at once was demanding.",
        },
        "quotation_sources": {
            "dominant_strategy": "SIM-07",
            "provenance": "SIM-11",
            "geometry": "SIM-05",
            "negative": "SIM-15",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# HyperTrace Synthetic Qualitative Accounts",
        "",
        "> Synthetic qualitative template-validation data only. Generated rationales, comments, counts, and quotation-bank entries are not participant evidence.",
        "",
        "## Coding unit",
        "",
        f"- {len(coded_rows)} trial-level final rationales and {len(comments)} participant-level post-task comments.",
        "- Codes: sequence reconstruction, provenance verification, initial/final comparison, geometry audit, counterfactual use, evidence overload, metric confusion, and missed scope warning.",
        "",
        "## Counts",
        "",
        "| Code | Trial rationales | Participants |",
        "|---|---:|---:|",
    ]
    for code in sorted(selected_codes):
        lines.append(f"| {code} | {trial_counts[code]} | {participant_counts[code]} |")
    lines += [
        "",
        "## Synthetic quotation bank",
        "",
        f"- Dominant strategy ({dominant_participants} participants, {dominant_rationales} rationales): \"{summary['quotation_bank']['dominant_strategy']}\" ({summary['quotation_sources']['dominant_strategy']})",
        f"- Provenance check: \"{summary['quotation_bank']['provenance']}\" ({summary['quotation_sources']['provenance']})",
        f"- Geometry audit: \"{summary['quotation_bank']['geometry']}\" ({summary['quotation_sources']['geometry']})",
        f"- Negative case: \"{summary['quotation_bank']['negative']}\" ({summary['quotation_sources']['negative']})",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
