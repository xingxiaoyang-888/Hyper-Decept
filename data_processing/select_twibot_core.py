"""Select a reproducible TwiBot-22 core stratified by official split and label."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


SCHEMA_VERSION = "hyperdecept.twibot22-core-selection.v1"
ALGORITHM = "proportional_split_label_stratified_sha256_rank_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, value_column: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", value_column}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} must contain id and {value_column}")
        for row in reader:
            user_id = str(row.get("id") or "").strip()
            value = str(row.get(value_column) or "").strip().lower()
            if not user_id or not value:
                continue
            if user_id in values and values[user_id] != value:
                raise ValueError(f"conflicting {value_column} for {user_id}")
            values[user_id] = value
    return values


def _allocate(stratum_sizes: dict[tuple[str, str], int], count: int) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    total = sum(stratum_sizes.values())
    if count > total:
        raise ValueError(f"requested {count} core users, only {total} are eligible")
    raw = {key: count * size / total for key, size in stratum_sizes.items()}
    allocation = {key: min(stratum_sizes[key], int(value)) for key, value in raw.items()}
    remaining = count - sum(allocation.values())
    order = sorted(
        stratum_sizes,
        key=lambda key: (-(raw[key] - int(raw[key])), key),
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < stratum_sizes[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate requested core size")
    return allocation


def _rank(user_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{user_id}".encode("utf-8")).digest()


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def select_core_users(
    *,
    label_path: Path,
    split_path: Path,
    count: int,
    seed: int,
) -> tuple[list[str], dict]:
    labels = _read_mapping(label_path, "label")
    splits = _read_mapping(split_path, "split")
    unknown_labels = sorted(set(labels.values()) - {"bot", "human"})
    if unknown_labels:
        raise ValueError(f"unsupported labels: {unknown_labels}")
    eligible_ids = sorted(set(labels).intersection(splits))
    if not eligible_ids:
        raise ValueError("label.csv and split.csv have no overlapping IDs")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for user_id in eligible_ids:
        strata[(splits[user_id], labels[user_id])].append(user_id)
    sizes = {key: len(values) for key, values in strata.items()}
    allocation = _allocate(sizes, count)
    selected_by_stratum = {
        key: sorted(values, key=lambda value: (_rank(value, seed), value))[:allocation[key]]
        for key, values in strata.items()
    }
    selected = [
        user_id
        for key in sorted(selected_by_stratum)
        for user_id in selected_by_stratum[key]
    ]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("selection count or uniqueness invariant failed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "seed": seed,
        "requested_core_count": count,
        "selected_core_count": len(selected),
        "eligible_count": len(eligible_ids),
        "label_only_ids": len(set(labels) - set(splits)),
        "split_only_ids": len(set(splits) - set(labels)),
        "source": {
            "label_csv": str(label_path.resolve()),
            "label_sha256": _sha256(label_path),
            "split_csv": str(split_path.resolve()),
            "split_sha256": _sha256(split_path),
        },
        "strata": {
            f"{split_name}|{label}": {
                "eligible": sizes[(split_name, label)],
                "selected": allocation[(split_name, label)],
            }
            for split_name, label in sorted(sizes)
        },
        "selected_distribution": dict(sorted(Counter(
            f"{splits[user_id]}|{labels[user_id]}" for user_id in selected
        ).items())),
    }
    return selected, report


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twibot-dir", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    root = args.twibot_dir.expanduser().resolve()
    selected, report = select_core_users(
        label_path=root / "label.csv",
        split_path=root / "split.csv",
        count=args.count,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    _write_lines(output, selected)
    report["output"] = {
        "core_ids": str(output),
        "core_ids_sha256": _sha256(output),
    }
    report["code_commit"] = _git_commit(Path.cwd())
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "core_ids": str(output),
        "audit": str(audit_output),
        "selected": len(selected),
        "strata": report["strata"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
