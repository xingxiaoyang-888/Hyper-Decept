"""Build a de-identified private review-study package from frozen evidence runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


OPERATIONS = ("honduras", "uae")
HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,32}")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LONG_NUMBER = re.compile(r"\b\d{7,}\b")
RAW_IDENTIFIER = re.compile(
    r"(?<![\d.])\d{15,}(?![\d.])|(?:user|tweet):\d{5,}", re.IGNORECASE
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alias(prefix: str, value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest.upper()}"


def sanitize_text(value: str) -> str:
    text = URL.sub("[link]", value or "")
    text = HANDLE.sub("@account", text)
    text = LONG_NUMBER.sub("[number]", text)
    text = " ".join(text.split())
    return text[:360]


def load_formal_cases(root: Path) -> dict[str, list[dict]]:
    result = {}
    for operation in OPERATIONS:
        path = root / operation / "cases.jsonl"
        result[operation] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(result[operation]) != 12:
            raise ValueError(f"{operation} formal run must contain 12 cases")
    return result


def selected_user_ids(cases: dict[str, list[dict]]) -> dict[str, set[str]]:
    return {
        operation: {case["case_id"].removeprefix("user:") for case in rows}
        for operation, rows in cases.items()
    }


def load_labels(external_root: Path, wanted: dict[str, set[str]]) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    for operation in OPERATIONS:
        with (external_root / operation / "labels.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                node_id = str(row["node_id"])
                if node_id in wanted[operation]:
                    labels[(operation, node_id)] = (
                        "coordinated" if str(row["is_bad"]).lower() in {"1", "true"}
                        else "not_coordinated"
                    )
    expected = sum(len(values) for values in wanted.values())
    if len(labels) != expected:
        raise ValueError(f"resolved {len(labels)} of {expected} formal case labels")
    return labels


def load_evidence(
    external_root: Path,
    formal_cases: dict[str, list[dict]],
) -> dict[tuple[str, str], dict]:
    wanted = {
        operation: {
            evidence_id
            for case in rows
            for evidence_id in case["provenance"]["unique_evidence_ids"]
        }
        for operation, rows in formal_cases.items()
    }
    evidence: dict[tuple[str, str], dict] = {}
    for operation in OPERATIONS:
        with (external_root / operation / "events.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                evidence_id = str(row["evidence_id"])
                if evidence_id in wanted[operation]:
                    evidence[(operation, evidence_id)] = {
                        "timestamp": row.get("timestamp"),
                        "event_type": row.get("event_type") or "event",
                        "text": sanitize_text(row.get("text") or ""),
                    }
    return evidence


def build_case(
    operation: str,
    index: int,
    case: dict,
    labels: dict[tuple[str, str], str],
    evidence_lookup: dict[tuple[str, str], dict],
    salt: str,
) -> dict:
    user_id = case["case_id"].removeprefix("user:")
    evidence_rows = []
    for raw_id in case["provenance"]["unique_evidence_ids"][:8]:
        record = evidence_lookup.get((operation, raw_id))
        if record is None:
            continue
        evidence_rows.append({
            "evidence_id": alias("EV", raw_id, salt),
            **record,
        })
    timestamps = sorted(
        record["timestamp"] for record in evidence_rows if record.get("timestamp")
    )
    event_counts = Counter(record["event_type"] for record in evidence_rows)
    prediction = case["prediction"]
    selection = case["selection"]
    return {
        "case_id": f"HT-{operation[0].upper()}-{index + 1:02d}",
        "operation": operation,
        "operation_label": f"Operation {operation[0].upper()}",
        "ground_truth": labels[(operation, user_id)],
        "model_recommendation": (
            "coordinated"
            if prediction["predicted_class"] == "coordination"
            else "not_coordinated"
        ),
        "risk_percentile": prediction["frozen_consensus_percentile"],
        "risk_stratum": case["risk_stratum"],
        "snapshot_scope": case["temporal"]["scope"],
        "relation_summary": selection["relation_counts"],
        "temporal_summary": {
            "first_event_at": timestamps[0] if timestamps else None,
            "last_event_at": timestamps[-1] if timestamps else None,
            "event_type_counts": dict(sorted(event_counts.items())),
            "evidence_rows": case["temporal"]["evidence_rows"],
        },
        "explanation": {
            "sparsity": selection["sparsity"],
            "selected_units": selection["selected_evidence_units"],
            "candidate_units": selection["candidate_evidence_units"],
            "sufficiency_error": prediction["sufficiency_percentile_error"],
            "geometry_fidelity": prediction["geometry_fidelity"],
            "checkpoint_agreement": selection["checkpoint_topk_jaccard"],
            "prototype_vote_agreement": prediction["prototype_vote_agreement"],
            "full_percentile": prediction["recomputed_full_percentile"],
            "keep_only_percentile": prediction["keep_only_percentile"],
        },
        "audit": {
            "provenance_coverage": case["provenance"]["coverage"],
            "timestamp_coverage": case["temporal"]["timestamp_parse_coverage"],
            "labels_consumed_during_selection": case["labels_consumed"],
        },
        "evidence": evidence_rows,
    }


def run(args: argparse.Namespace) -> dict:
    formal_root = args.formal_root.resolve()
    external_root = args.external_root.resolve()
    formal_cases = load_formal_cases(formal_root)
    wanted = selected_user_ids(formal_cases)
    labels = load_labels(external_root, wanted)
    evidence = load_evidence(external_root, formal_cases)
    cases = [
        build_case(operation, index, case, labels, evidence, args.study_salt)
        for operation in OPERATIONS
        for index, case in enumerate(formal_cases[operation])
    ]
    payload = {
        "schema_version": "hypertrace.review-study-cases.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_protocol": "formal relation-stratified constrained evidence evaluation",
        "contains_raw_identifiers": False,
        "private_label_boundary": "ground_truth is retained by the API and never emitted in trial payloads",
        "cases": cases,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if RAW_IDENTIFIER.search(serialized):
        raise ValueError("de-identification audit found a raw platform identifier")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    audit = {
        "schema_version": "hypertrace.review-study-case-audit.v1",
        "status": "passed",
        "case_count": len(cases),
        "operation_counts": dict(Counter(case["operation"] for case in cases)),
        "ground_truth_counts": dict(Counter(case["ground_truth"] for case in cases)),
        "recommendation_counts": dict(Counter(case["model_recommendation"] for case in cases)),
        "raw_identifier_scan_passed": True,
        "missing_evidence_case_count": sum(not case["evidence"] for case in cases),
        "output_sha256": sha256(args.output),
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-salt", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
