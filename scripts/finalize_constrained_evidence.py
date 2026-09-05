"""Validate formal constrained-evidence runs and emit paper-ready summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


METRICS = (
    ("comprehensiveness", lambda case: case["prediction"]["comprehensiveness"]),
    (
        "sufficiency_percentile_error",
        lambda case: case["prediction"]["sufficiency_percentile_error"],
    ),
    ("sparsity", lambda case: case["selection"]["sparsity"]),
    ("geometry_fidelity", lambda case: case["prediction"]["geometry_fidelity"]),
    (
        "prototype_vote_agreement",
        lambda case: case["prediction"]["prototype_vote_agreement"],
    ),
    ("provenance_coverage", lambda case: case["provenance"]["coverage"]),
    (
        "timestamp_parse_coverage",
        lambda case: case["temporal"]["timestamp_parse_coverage"],
    ),
    (
        "explanation_stability",
        lambda case: case["selection"]["checkpoint_topk_jaccard"],
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _load_run(operation: str, directory: Path, expected_cases: int) -> dict:
    audit_path = directory / "audit.json"
    cases_path = directory / "cases.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    if audit.get("status") != "passed":
        errors.append("audit status is not passed")
    if audit.get("operation") != operation:
        errors.append("operation does not match requested operation")
    if audit.get("case_count") != expected_cases or len(cases) != expected_cases:
        errors.append(f"expected exactly {expected_cases} cases")
    if audit.get("checkpoint_count") != 15:
        errors.append("formal protocol requires exactly 15 checkpoints")
    if audit.get("labels_consumed") is not False:
        errors.append("labels_consumed must be false")
    if audit.get("target_label_values_consumed") is not False:
        errors.append("target_label_values_consumed must be false")
    expected_hash = audit.get("artifacts", {}).get("cases_sha256")
    if expected_hash != _sha256(cases_path):
        errors.append("cases.jsonl hash does not match audit")
    for index, case in enumerate(cases):
        if case.get("labels_consumed") is not False:
            errors.append(f"case {index} consumed labels")
        if case.get("target_label_values_consumed") is not False:
            errors.append(f"case {index} consumed target label values")
        if case.get("operation") != operation:
            errors.append(f"case {index} operation mismatch")
    if errors:
        raise ValueError(f"{operation} formal evidence run invalid: {errors}")
    return {
        "audit_path": audit_path,
        "cases_path": cases_path,
        "audit": audit,
        "cases": cases,
    }


def _rows(operation: str, cases: list[dict]) -> list[dict]:
    rows = []
    groups = [("all", cases)] + [
        (stratum, [case for case in cases if case["risk_stratum"] == stratum])
        for stratum in ("high", "medium", "low")
    ]
    for stratum, selected in groups:
        row: dict[str, str | int | float] = {
            "operation": operation,
            "risk_stratum": stratum,
            "n": len(selected),
            "candidate_evidence_units_mean": statistics.fmean(
                case["selection"]["candidate_evidence_units"] for case in selected
            ),
            "selected_evidence_units_mean": statistics.fmean(
                case["selection"]["selected_evidence_units"] for case in selected
            ),
        }
        for name, getter in METRICS:
            row[name] = statistics.fmean(float(getter(case)) for case in selected)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Operation | Stratum | n | Candidate units | Selected units | Sparsity | Suff. error | Comp. | Geometry | Vote | Provenance | Time | Stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {operation} | {risk_stratum} | {n} | {candidate_evidence_units_mean:.2f} | "
            "{selected_evidence_units_mean:.2f} | {sparsity:.4f} | "
            "{sufficiency_percentile_error:.4f} | {comprehensiveness:.4f} | "
            "{geometry_fidelity:.4f} | {prototype_vote_agreement:.4f} | "
            "{provenance_coverage:.4f} | {timestamp_parse_coverage:.4f} | "
            "{explanation_stability:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "honduras": _load_run("honduras", args.honduras_dir.resolve(), args.case_count),
        "uae": _load_run("uae", args.uae_dir.resolve(), args.case_count),
    }
    rows = [
        row
        for operation, run_data in runs.items()
        for row in _rows(operation, run_data["cases"])
    ]
    csv_path = output_dir / "constrained_evidence_paper_table.csv"
    markdown_path = output_dir / "constrained_evidence_paper_table.md"
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)

    summary = {
        "schema_version": "hypertrace.constrained-evidence-formal-summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "protocol": {
            "case_count_per_operation": args.case_count,
            "checkpoint_count": 15,
            "selection": "label-blind high/mid/low consensus strata",
            "evidence_search": "relation-stratified attention-reliability prefixes",
        },
        "operations": {
            operation: {
                "audit_sha256": _sha256(run_data["audit_path"]),
                "cases_sha256": _sha256(run_data["cases_path"]),
                "metrics": {
                    name: _summary([
                        float(getter(case)) for case in run_data["cases"]
                    ])
                    for name, getter in METRICS
                },
            }
            for operation, run_data in runs.items()
        },
    }
    summary_path = output_dir / "constrained_evidence_formal_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {
        path.name: _sha256(path)
        for path in (summary_path, csv_path, markdown_path)
    }
    checksum_path = output_dir / "constrained_evidence_sha256.txt"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--honduras-dir", type=Path, required=True)
    parser.add_argument("--uae-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=12)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
