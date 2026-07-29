"""Recover the DeepPersona summary input from an existing agent CSV.

The simulation consumes the same summary through the ``user_char`` column.
This utility makes that canonical data reusable by ``profile_chunker.py`` when
the large/private original profile JSON has not been transferred.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            summary = str(row.get("user_char") or "").strip()
            if not summary:
                raise ValueError(f"Row {row_number} has an empty user_char")
            raw_id = str(row.get("user_id") or row_number - 1).strip()
            profiles.append(
                {
                    "Profile Index": int(raw_id) + 1,
                    "Summary": summary,
                }
            )

    if not profiles:
        raise ValueError(f"No profiles found in {args.csv}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Recovered {len(profiles)} profiles -> {args.output}")


if __name__ == "__main__":
    main()
