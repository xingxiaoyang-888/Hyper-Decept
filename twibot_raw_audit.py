"""Read-only audit utility for an original TwiBot-22 directory.

The report contains file/schema/count information only.  It deliberately does
not copy profile descriptions, tweet text, user names, or other raw values, so
it is small enough to share before the full dataset is transferred.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


CHUNK_SIZE = 1024 * 1024
EXPECTED_CSV = ("edge.csv", "label.csv", "split.csv")


def normalize_id(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def load_core_ids(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Core-ID file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            id_column = next(
                (name for name in ("user_id", "id") if name in fieldnames),
                None,
            )
            if id_column is None:
                raise ValueError("Core-ID CSV must contain user_id or id")
            return {
                value
                for row in reader
                if (value := normalize_id(row.get(id_column))) is not None
            }
    with path.open("r", encoding="utf-8-sig") as handle:
        return {
            value
            for line in handle
            if (value := normalize_id(line)) is not None
        }


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _iter_json_array(handle) -> Iterator[object]:
    decoder = json.JSONDecoder()
    buffer = ""
    index = 0
    eof = False
    started = False
    while True:
        # Compact only after a sizeable portion has been consumed. Slicing the
        # buffer after every record makes multi-gigabyte arrays effectively
        # quadratic because 1-2 MiB would be copied for each JSON object.
        if index >= CHUNK_SIZE:
            buffer = buffer[index:]
            index = 0
        if not eof and len(buffer) - index < CHUNK_SIZE:
            chunk = handle.read(CHUNK_SIZE)
            if chunk:
                buffer += chunk
            else:
                eof = True
        while index < len(buffer) and buffer[index].isspace():
            index += 1
        if not started:
            if index >= len(buffer):
                if eof:
                    return
                buffer = ""
                continue
            if buffer[index] != "[":
                raise ValueError("Expected a top-level JSON array")
            started = True
            index += 1
        while index < len(buffer) and (
            buffer[index].isspace() or buffer[index] == ","
        ):
            index += 1
        if index < len(buffer) and buffer[index] == "]":
            return
        if index >= len(buffer):
            if eof:
                raise ValueError("Unterminated JSON array")
            continue
        try:
            value, end = decoder.raw_decode(buffer, index)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = handle.read(CHUNK_SIZE)
            if chunk:
                buffer += chunk
            else:
                eof = True
            continue
        yield value
        index = end


def iter_json_records(path: Path) -> Iterator[object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            yield from _iter_json_array(handle)
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from error


def inspect_csv(path: Path, core_ids: set[str], with_sha256: bool) -> dict:
    row_count = 0
    distributions: Dict[str, Counter] = {
        "relation": Counter(),
        "label": Counter(),
        "split": Counter(),
    }
    core_source_rows = 0
    core_target_rows = 0
    incident_relations = Counter()
    incident_noncore_ids: set[str] = set()
    observed_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        id_column = next(
            (name for name in ("id", "user_id") if name in columns), None
        )
        for row in reader:
            row_count += 1
            for column, counter in distributions.items():
                value = row.get(column)
                if value not in (None, ""):
                    counter[str(value)] += 1
            if id_column is not None:
                value = normalize_id(row.get(id_column))
                if value is not None:
                    observed_ids.add(value)
            if core_ids and {"source_id", "target_id"}.issubset(columns):
                source = normalize_id(row.get("source_id"))
                target = normalize_id(row.get("target_id"))
                relation = str(row.get("relation") or "<missing>")
                if source in core_ids:
                    core_source_rows += 1
                    incident_relations[relation] += 1
                    if target is not None and target not in core_ids:
                        incident_noncore_ids.add(target)
                if target in core_ids:
                    core_target_rows += 1
                    incident_relations[relation] += 1
                    if source is not None and source not in core_ids:
                        incident_noncore_ids.add(source)
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "columns": columns,
        "row_count": row_count,
        "distributions": {
            key: dict(sorted(counter.items()))
            for key, counter in distributions.items()
            if counter
        },
    }
    if core_ids:
        result["core_coverage"] = {
            "matched_id_rows": len(core_ids.intersection(observed_ids)),
            "core_as_source_rows": core_source_rows,
            "core_as_target_rows": core_target_rows,
            "incident_noncore_ids": len(incident_noncore_ids),
            "incident_relation_counts": dict(sorted(incident_relations.items())),
        }
    if with_sha256:
        result["sha256"] = sha256_file(path)
    return result


def inspect_json(
    path: Path,
    core_ids: set[str],
    with_sha256: bool,
    max_records: Optional[int],
) -> dict:
    record_count = 0
    key_types: Dict[str, Counter] = {}
    nested_keys: Dict[str, set[str]] = {}
    id_count = 0
    core_id_matches: set[str] = set()
    author_id_count = 0
    core_author_matches = 0
    created_at_count = 0
    text_count = 0
    referenced_tweets_count = 0
    truncated = False
    for record in iter_json_records(path):
        if max_records is not None and record_count >= max_records:
            truncated = True
            break
        record_count += 1
        if not isinstance(record, dict):
            key_types.setdefault("<record>", Counter())[_type_name(record)] += 1
            continue
        for key, value in record.items():
            key_types.setdefault(str(key), Counter())[_type_name(value)] += 1
            if isinstance(value, dict):
                nested_keys.setdefault(str(key), set()).update(map(str, value.keys()))
        record_id = normalize_id(record.get("id"))
        if record_id is not None:
            id_count += 1
            if record_id in core_ids:
                core_id_matches.add(record_id)
        author_id = normalize_id(record.get("author_id"))
        if author_id is not None:
            author_id_count += 1
            if author_id in core_ids:
                core_author_matches += 1
        if record.get("created_at") not in (None, ""):
            created_at_count += 1
        if record.get("text") not in (None, ""):
            text_count += 1
        if record.get("referenced_tweets"):
            referenced_tweets_count += 1
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "record_count": record_count,
        "truncated": truncated,
        "schema": {
            key: {
                "types": dict(sorted(counter.items())),
                **(
                    {"nested_keys": sorted(nested_keys[key])}
                    if key in nested_keys else {}
                ),
            }
            for key, counter in sorted(key_types.items())
        },
        "field_coverage": {
            "id_non_null": id_count,
            "author_id_non_null": author_id_count,
            "created_at_non_null": created_at_count,
            "text_non_null": text_count,
            "referenced_tweets_nonempty": referenced_tweets_count,
        },
    }
    if core_ids:
        result["core_coverage"] = {
            "record_id_matches": len(core_id_matches),
            "records_authored_by_core": core_author_matches,
        }
    if with_sha256:
        result["sha256"] = sha256_file(path)
    return result


def _find_files(root: Path, predicate) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and predicate(path)),
        key=lambda path: str(path.relative_to(root)).lower(),
    )


def audit_twibot_directory(
    root: Path,
    core_ids: Optional[Iterable[str]] = None,
    with_sha256: bool = False,
    max_json_records: Optional[int] = None,
) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"TwiBot directory does not exist: {root}")
    normalized_core = {
        value
        for item in (core_ids or [])
        if (value := normalize_id(item)) is not None
    }
    csv_files = _find_files(
        root, lambda path: path.name.lower() in EXPECTED_CSV
    )
    json_files = _find_files(
        root,
        lambda path: path.name.lower() == "user.json"
        or path.name.lower() in {"list.json", "hashtag.json"}
        or (
            path.name.lower().startswith("tweet_")
            and path.suffix.lower() == ".json"
        ),
    )
    documentation = _find_files(
        root,
        lambda path: "readme" in path.name.lower()
        or "license" in path.name.lower(),
    )
    present_names = {path.name.lower() for path in csv_files + json_files}
    missing = [name for name in EXPECTED_CSV if name not in present_names]
    if "user.json" not in present_names:
        missing.append("user.json")
    if not any(name.startswith("tweet_") for name in present_names):
        missing.append("tweet_*.json")
    warnings = []
    if missing:
        warnings.append(f"Missing expected TwiBot-22 files: {', '.join(missing)}")
    if not documentation:
        warnings.append(
            "No README/LICENSE found; relation semantics and redistribution terms "
            "must be obtained separately."
        )
    elif not any("license" in path.name.lower() for path in documentation):
        warnings.append(
            "README found but no LICENSE file is present; obtain the official "
            "license before transferring or redistributing the dataset."
        )

    csv_reports = []
    for path in csv_files:
        print(f"[audit] CSV start: {path.name}", flush=True)
        csv_reports.append(inspect_csv(path, normalized_core, with_sha256))
        print(f"[audit] CSV done:  {path.name}", flush=True)

    json_reports = []
    for path in json_files:
        print(f"[audit] JSON start: {path.name}", flush=True)
        json_reports.append(
            inspect_json(
                path,
                normalized_core,
                with_sha256,
                max_json_records,
            )
        )
        print(f"[audit] JSON done:  {path.name}", flush=True)

    return {
        "audit_version": 1,
        "dataset_root": str(root),
        "detected_kind": "twibot22_raw" if not missing else "incomplete_or_unknown",
        "privacy": "No raw text, descriptions, names, or example values are included.",
        "core_id_count": len(normalized_core),
        "csv_files": csv_reports,
        "json_files": json_reports,
        "documentation_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                **({"sha256": sha256_file(path)} if with_sha256 else {}),
            }
            for path in documentation
        ],
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a privacy-preserving audit report for TwiBot-22 raw files."
    )
    parser.add_argument("--twibot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--core-ids",
        type=Path,
        help="Optional newline list or CSV containing the selected 1000 user IDs.",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash every inspected file; accurate but slow for tweet shards.",
    )
    parser.add_argument(
        "--max-json-records",
        type=int,
        default=None,
        help="Optional fast-audit limit per JSON file. Omit for exact counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_ids = load_core_ids(args.core_ids)
    report = audit_twibot_directory(
        args.twibot_dir,
        core_ids=core_ids,
        with_sha256=args.sha256,
        max_json_records=args.max_json_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
