"""Build an auditable seed-text pool from the upstream framework files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "hyperdecept.framework-native-twitter-seed.v1"
SOURCE_REPOSITORY = "https://github.com/renqibing/MultiAgent4Collusion"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_texts(paths: Iterable[Path], label: str) -> tuple[list[str], list[dict]]:
    values: list[str] = []
    sources: list[dict] = []
    for value in paths:
        path = value.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"{label} source must be a JSON list: {path}")
        cleaned = []
        for item in payload:
            if not isinstance(item, str):
                raise TypeError(f"{label} source contains non-string text: {path}")
            text = item.strip()
            if text:
                cleaned.append(text)
        values.extend(cleaned)
        sources.append({
            "path": str(path),
            "label": label,
            "source_sha256": _sha256(path),
            "source_items": len(payload),
            "nonempty_items": len(cleaned),
        })
    # Preserve upstream file order while removing exact duplicate texts.
    unique = list(dict.fromkeys(values))
    return unique, sources


def build_pool(*, good: list[Path], bad: list[Path], output: Path, manifest: Path | None = None) -> dict:
    if not good or not bad:
        raise ValueError("at least one good and one bad source are required")
    good_texts, good_sources = _load_texts(good, "good")
    bad_texts, bad_sources = _load_texts(bad, "bad")
    if not good_texts or not bad_texts:
        raise ValueError("native good and bad pools must both be non-empty")
    output = output.expanduser().resolve()
    manifest = (manifest or output.with_name(output.stem + ".manifest.json")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"good": good_texts, "bad": bad_texts}
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_repository": SOURCE_REPOSITORY,
        "source_ref": "master",
        "source_files": good_sources + bad_sources,
        "deduplication": "exact_text_preserving_first_occurrence",
        "counts": {"good": len(good_texts), "bad": len(bad_texts)},
        "pool_file": output.name,
        "pool_sha256": _sha256(output),
        "use_scope": "native upstream Twitter seed text for DeepPersona-injected OASIS simulations",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "manifest_path": str(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good", nargs="+", required=True, type=Path)
    parser.add_argument("--bad", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_pool(
        good=args.good,
        bad=args.bad,
        output=args.output,
        manifest=args.manifest,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
