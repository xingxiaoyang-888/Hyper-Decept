"""Audit the public Honduras/UAE information-operation JSONL release.

The report contains aggregate schema and provenance statistics only. It never
copies tweet text or user identifiers into the audit artifact.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "hypertrace.honduras-uae-source-audit.v1"
SOURCE_DOI = "10.5281/zenodo.13912659"
EXPECTED_FILES = {
    "honduras-bad-anonymized.jsonl": {
        "country": "honduras",
        "cohort": "malicious_io",
        "good_value": 0,
        "bytes": 86_954_301,
        "md5": "c6ab84ebd78ba7df78d0266d013b471e",
    },
    "honduras-good-anonymized.jsonl": {
        "country": "honduras",
        "cohort": "genuine_control",
        "good_value": 1,
        "bytes": 758_558_125,
        "md5": "da75759d86993d2c12b1946e59761bb4",
    },
    "uae-bad-anonymized.jsonl": {
        "country": "uae",
        "cohort": "malicious_io",
        "good_value": 0,
        "bytes": 371_545_972,
        "md5": "c216b78629796fd38c08a455625c7d88",
    },
    "uae-good-anonymized.jsonl": {
        "country": "uae",
        "cohort": "genuine_control",
        "good_value": 1,
        "bytes": 3_018_950_993,
        "md5": "ce0f005ad71c13b44f2da064f3332de9",
    },
}
REQUIRED_FIELDS = (
    "tweetid",
    "userid",
    "tweet_text",
    "tweet_time",
    "good",
)
RELATION_FIELDS = {
    "reply": "in_reply_to_userid",
    "retweet": "retweet_userid",
}


def _digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _listed_identifiers(value: Any) -> tuple[list[str], bool]:
    if value is None or value == "":
        return [], True
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return [], True
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [], False
    if not isinstance(parsed, (list, tuple, set)):
        return [], False
    return [item for item in (_identifier(v) for v in parsed) if item], True


def _timestamp_millis(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number <= 0:
        return None
    return number


def _iso_millis(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _coverage(count: int, total: int) -> float:
    return count / total if total else 0.0


def audit_file(path: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], set[str], set[str]]:
    users: set[str] = set()
    tweet_ids: set[str] = set()
    endpoint_counts = {name: Counter() for name in RELATION_FIELDS}
    mention_endpoint_counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    relation_events: Counter[str] = Counter()
    rows = 0
    parse_errors = 0
    duplicate_tweets = 0
    nonempty_text = 0
    valid_time = 0
    invalid_time = 0
    min_time: int | None = None
    max_time: int | None = None
    mention_parse_errors = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parse_errors += 1
                continue
            if not isinstance(row, dict):
                parse_errors += 1
                continue
            rows += 1
            for field in REQUIRED_FIELDS:
                if row.get(field) is None:
                    missing[field] += 1
            user_id = _identifier(row.get("userid"))
            if user_id is not None:
                users.add(user_id)
            tweet_id = _identifier(row.get("tweetid"))
            if tweet_id is not None:
                if tweet_id in tweet_ids:
                    duplicate_tweets += 1
                tweet_ids.add(tweet_id)
            labels[str(row.get("good"))] += 1
            if str(row.get("tweet_text") or "").strip():
                nonempty_text += 1
            timestamp = _timestamp_millis(row.get("tweet_time"))
            if timestamp is None:
                invalid_time += 1
            else:
                valid_time += 1
                min_time = timestamp if min_time is None else min(min_time, timestamp)
                max_time = timestamp if max_time is None else max(max_time, timestamp)

            for relation, field in RELATION_FIELDS.items():
                endpoint = _identifier(row.get(field))
                if endpoint is not None:
                    relation_events[relation] += 1
                    endpoint_counts[relation][endpoint] += 1
            if _identifier(row.get("quoted_tweet_tweetid")) is not None:
                relation_events["quote"] += 1
            mentions, valid_mentions = _listed_identifiers(row.get("user_mentions"))
            if not valid_mentions:
                mention_parse_errors += 1
            if mentions:
                relation_events["mention"] += len(mentions)
                mention_endpoint_counts.update(mentions)

    endpoint_counts["mention"] = mention_endpoint_counts
    internal_relation_events = {
        relation: sum(count for endpoint, count in counts.items() if endpoint in users)
        for relation, counts in endpoint_counts.items()
    }
    expected_label = str(contract["good_value"])
    unexpected_labels = {
        key: value for key, value in labels.items() if key != expected_label
    }
    size = path.stat().st_size
    md5 = _digest(path)
    errors = []
    if size != contract["bytes"]:
        errors.append("official_size_mismatch")
    if md5 != contract["md5"]:
        errors.append("official_md5_mismatch")
    if parse_errors:
        errors.append("jsonl_parse_errors")
    if unexpected_labels:
        errors.append("unexpected_good_label")
    if missing:
        errors.append("missing_required_fields")
    if valid_time == 0:
        errors.append("no_valid_timestamps")
    if nonempty_text == 0:
        errors.append("no_nonempty_text")

    report = {
        "path": path.name,
        "country": contract["country"],
        "cohort": contract["cohort"],
        "official_integrity": {
            "expected_bytes": contract["bytes"],
            "actual_bytes": size,
            "expected_md5": contract["md5"],
            "actual_md5": md5,
            "passed": size == contract["bytes"] and md5 == contract["md5"],
        },
        "rows": rows,
        "jsonl_parse_errors": parse_errors,
        "unique_users": len(users),
        "unique_tweets": len(tweet_ids),
        "duplicate_tweet_ids": duplicate_tweets,
        "good_value_distribution": dict(sorted(labels.items())),
        "expected_good_value": contract["good_value"],
        "required_field_missing": dict(sorted(missing.items())),
        "text": {
            "nonempty_rows": nonempty_text,
            "coverage": _coverage(nonempty_text, rows),
        },
        "time": {
            "valid_rows": valid_time,
            "invalid_rows": invalid_time,
            "coverage": _coverage(valid_time, rows),
            "minimum_epoch_ms": min_time,
            "maximum_epoch_ms": max_time,
            "minimum_utc": _iso_millis(min_time),
            "maximum_utc": _iso_millis(max_time),
        },
        "relations": {
            relation: {
                "events": count,
                "internal_endpoint_events": internal_relation_events.get(relation, 0),
                "internal_endpoint_fraction": _coverage(
                    internal_relation_events.get(relation, 0), count
                ),
            }
            for relation, count in sorted(relation_events.items())
        },
        "mention_parse_errors": mention_parse_errors,
        "errors": errors,
        "passed": not errors,
    }
    return report, users, tweet_ids


def _pairwise_overlaps(values: dict[str, set[str]]) -> list[dict[str, Any]]:
    names = sorted(values)
    result = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = len(values[left] & values[right])
            if overlap or left.split("-", 1)[0] == right.split("-", 1)[0]:
                result.append({"left": left, "right": right, "count": overlap})
    return result


def audit_root(root: Path, *, require_all: bool) -> dict[str, Any]:
    file_reports = []
    user_sets: dict[str, set[str]] = {}
    tweet_sets: dict[str, set[str]] = {}
    missing_files = []
    for filename, contract in EXPECTED_FILES.items():
        path = root / filename
        if not path.is_file():
            missing_files.append(filename)
            continue
        report, users, tweets = audit_file(path, contract)
        cohort_id = f"{contract['country']}-{contract['cohort']}"
        file_reports.append(report)
        user_sets[cohort_id] = users
        tweet_sets[cohort_id] = tweets

    relation_events = sum(
        relation["events"]
        for report in file_reports
        for relation in report["relations"].values()
    )
    time_rows = sum(report["time"]["valid_rows"] for report in file_reports)
    text_rows = sum(report["text"]["nonempty_rows"] for report in file_reports)
    complete = not missing_files
    passed = bool(file_reports) and all(report["passed"] for report in file_reports)
    if require_all:
        passed = passed and complete
    status = "passed" if passed and complete else ("partial_pass" if passed else "failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "doi": SOURCE_DOI,
            "license": "CC-BY-4.0",
            "release_title": "Twitter dataset about Information Operations in Honduras and UAE",
        },
        "status": status,
        "require_all": require_all,
        "missing_files": missing_files,
        "files": file_reports,
        "overlap": {
            "users": _pairwise_overlaps(user_sets),
            "tweets": _pairwise_overlaps(tweet_sets),
        },
        "capabilities": {
            "observed_text": text_rows > 0,
            "observed_timestamps": time_rows > 0,
            "observed_user_relations": relation_events > 0,
            "explicit_information_operation_cohorts": True,
            "explicit_campaign_ids": False,
            "explicit_cib_membership": False,
            "llm_origin_labels": False,
        },
        "allowed_uses": {
            "real_graph_self_supervised_warm_start": relation_events > 0 and time_rows > 0,
            "frozen_real_io_external_evaluation": complete and passed,
            "campaign_disjoint_supervised_training": False,
        },
        "semantic_constraints": [
            "The bad/good cohorts support real information-operation evaluation, not generic bot detection.",
            "The release does not establish LLM origin or campaign-level CIB membership.",
            "Derived reply, retweet, quote, and mention edges must retain their source tweet IDs and timestamps.",
        ],
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = audit_root(args.root, require_all=args.require_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "files": len(report["files"]),
        "missing_files": report["missing_files"],
        "capabilities": report["capabilities"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    if args.require_all and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
