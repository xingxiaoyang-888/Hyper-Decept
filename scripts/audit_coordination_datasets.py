"""Create reproducible, read-only audits for real coordination datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.coordination_adapter import load_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=(
        "crypto-campaign", "uk2019-coordinated-behavior", "fake-accounts-activity"
    ))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = load_dataset(args.dataset, args.root)
    files = [p for p in args.root.rglob("*") if p.is_file() and ".git" not in p.parts]
    report = bundle.manifest()
    report["files"] = [
        {"path": str(p.relative_to(args.root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(files)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
