"""Add label-blind, model-blind common activity records to study cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .build_study_cases import alias, sanitize_text
except ImportError:
    from build_study_cases import alias, sanitize_text


SOURCE_FILES = {
    "honduras": (
        "honduras-bad-anonymized.jsonl",
        "honduras-good-anonymized.jsonl",
    ),
    "uae": (
        "uae-bad-anonymized.jsonl",
        "uae-good-anonymized.jsonl",
    ),
}
COMMON_RECORD_LIMIT = 8


def identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or None


def iso_time(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(
            int(value) / 1000, tz=timezone.utc
        ).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def event_type(row: dict[str, Any]) -> str:
    if identifier(row.get("retweet_tweetid")):
        return "retweet"
    if identifier(row.get("in_reply_to_tweetid")):
        return "reply"
    if identifier(row.get("quoted_tweet_tweetid")):
        return "quote"
    return "post"


def load_case_mapping(formal_root: Path, package: dict) -> dict[str, tuple[str, str]]:
    private_by_operation: dict[str, list[dict]] = defaultdict(list)
    for case in package["cases"]:
        private_by_operation[case["operation"]].append(case)

    mapping = {}
    for operation in SOURCE_FILES:
        formal_path = formal_root / operation / "cases.jsonl"
        formal = [
            json.loads(line)
            for line in formal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        private = private_by_operation[operation]
        if len(formal) != len(private):
            raise ValueError(f"case count mismatch for {operation}")
        for index, (formal_case, private_case) in enumerate(zip(formal, private)):
            expected = f"HT-{operation[0].upper()}-{index + 1:02d}"
            if private_case["case_id"] != expected:
                raise ValueError(f"unexpected private case order: {private_case['case_id']}")
            user_id = str(formal_case["case_id"]).removeprefix("user:")
            mapping[user_id] = (operation, private_case["case_id"])
    return mapping


def scan_source(
    source_root: Path,
    mapping: dict[str, tuple[str, str]],
) -> dict[str, list[dict]]:
    wanted_by_operation: dict[str, set[str]] = defaultdict(set)
    for user_id, (operation, _) in mapping.items():
        wanted_by_operation[operation].add(user_id)

    records: dict[str, list[dict]] = defaultdict(list)
    for operation, filenames in SOURCE_FILES.items():
        wanted = wanted_by_operation[operation]
        pattern = re.compile(b"(?:" + b"|".join(
            re.escape(value.encode("ascii")) for value in sorted(wanted)
        ) + b")")
        for filename in filenames:
            path = source_root / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not pattern.search(line):
                        continue
                    row = json.loads(line)
                    user_id = identifier(row.get("userid"))
                    if user_id not in wanted:
                        continue
                    records[user_id].append({
                        "raw_event_id": identifier(row.get("tweetid")),
                        "timestamp": iso_time(row.get("tweet_time")),
                        "timestamp_ms": int(row.get("tweet_time") or 0),
                        "event_type": event_type(row),
                        "text": sanitize_text(str(row.get("tweet_text") or "")),
                        "source_file": filename,
                        "source_line": line_number,
                    })
    missing = sorted(set(mapping) - set(records))
    if missing:
        raise ValueError(f"no source activity found for {len(missing)} cases")
    return records


def time_stratified(records: list[dict], limit: int) -> list[dict]:
    ordered = sorted(
        records,
        key=lambda row: (row["timestamp_ms"], row["raw_event_id"] or ""),
    )
    if len(ordered) <= limit:
        return ordered
    positions = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[position] for position in positions]


def enrich(
    package_path: Path,
    formal_root: Path,
    source_root: Path,
    output: Path,
    salt: str,
) -> dict:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    mapping = load_case_mapping(formal_root, package)
    source_records = scan_source(source_root, mapping)
    case_lookup = {case["case_id"]: case for case in package["cases"]}
    for user_id, (_, case_id) in mapping.items():
        all_records = source_records[user_id]
        selected = time_stratified(all_records, COMMON_RECORD_LIMIT)
        common = []
        for record in selected:
            raw_event_id = record["raw_event_id"] or (
                f"{record['source_file']}:{record['source_line']}"
            )
            common.append({
                "record_id": alias("CR", raw_event_id, salt),
                "timestamp": record["timestamp"],
                "event_type": record["event_type"],
                "text": record["text"],
            })
        case = case_lookup[case_id]
        case["common_case_evidence"] = common
        valid_times = [
            record["timestamp"] for record in all_records if record["timestamp"]
        ]
        case["common_case_summary"] = {
            "source_activity_count": len(all_records),
            "displayed_record_count": len(common),
            "first_event_at": min(valid_times) if valid_times else None,
            "last_event_at": max(valid_times) if valid_times else None,
            "event_type_counts": dict(sorted(Counter(
                record["event_type"] for record in all_records
            ).items())),
        }
        case["common_case_contract"] = {
            "selection": "chronological_quantile_sample",
            "record_limit": COMMON_RECORD_LIMIT,
            "source_activity_count": len(all_records),
            "ground_truth_consumed": False,
            "model_output_consumed": False,
            "explainer_ranking_consumed": False,
        }

    package["schema_version"] = "hypertrace.review-study-cases.v2"
    package["common_case_evidence_contract"] = {
        "selection": "up_to_8_chronological_quantile_records_from_reviewed_account",
        "shared_across_conditions": True,
        "ground_truth_consumed": False,
        "model_output_consumed": False,
        "explainer_ranking_consumed": False,
    }
    serialized = json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized, encoding="utf-8")
    audit = {
        "schema_version": "hypertrace.common-case-evidence-audit.v1",
        "case_count": len(package["cases"]),
        "all_cases_populated": all(
            bool(case.get("common_case_evidence")) for case in package["cases"]
        ),
        "record_counts": dict(Counter(
            len(case["common_case_evidence"]) for case in package["cases"]
        )),
        "ground_truth_consumed": False,
        "model_output_consumed": False,
        "explainer_ranking_consumed": False,
        "raw_identifiers_emitted": False,
        "output_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    output.with_suffix(".common-evidence.audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    audit = enrich(
        args.package.resolve(),
        args.formal_root.resolve(),
        args.source_root.resolve(),
        args.output.resolve(),
        args.salt,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
