"""Prepare deterministic DeepPersona profiles for MultiAgent4Collusion.

The source is the official ``THzva/deeppersona_dataset`` repository.  We use
the world profile set and prefer the richer 200--350 attribute configurations.
The selected profiles are re-indexed to 1..N because the simulation uses
zero-based agent IDs after chunking.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path


DEFAULT_REPO_ID = "THzva/deeppersona_dataset"
DEFAULT_REPO_FILE = "profiles_examples/profile_world_4.1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--min-attributes", type=int, default=200)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "deeppersonal_agents.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "deeppersonal_agents_manifest.json",
    )
    return parser.parse_args()


def source_path(local_source: Path | None) -> Path:
    if local_source is not None:
        return local_source.expanduser().resolve()
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DEFAULT_REPO_ID,
            repo_type="dataset",
            filename=DEFAULT_REPO_FILE,
        )
    )


def attribute_count(source_key: str) -> int:
    match = re.search(r"Count_(\d+)", source_key)
    return int(match.group(1)) if match else 0


def main() -> None:
    args = parse_args()
    input_path = source_path(args.source)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Expected the official DeepPersona file to contain a JSON object")

    eligible = [
        (key, value)
        for key, value in raw.items()
        if attribute_count(key) >= args.min_attributes
    ]
    if len(eligible) < args.count:
        raise ValueError(
            f"Only {len(eligible)} profiles satisfy min_attributes="
            f"{args.min_attributes}; requested {args.count}"
        )

    selected = []
    selected_sources = []
    for agent_id, (key, profile) in enumerate(eligible[: args.count]):
        prepared = deepcopy(profile)
        prepared["Profile Index"] = agent_id + 1
        selected.append(prepared)
        selected_sources.append(
            {
                "agent_id": agent_id,
                "source_key": key,
                "source_profile_index": profile.get("Profile Index"),
                "attribute_configuration": attribute_count(key),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.manifest.write_text(
        json.dumps(
            {
                "source_repository": DEFAULT_REPO_ID,
                "source_file": DEFAULT_REPO_FILE,
                "local_source": str(input_path),
                "selection_rule": {
                    "count": args.count,
                    "min_attributes": args.min_attributes,
                    "order": "official JSON insertion order",
                },
                "profiles": selected_sources,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(selected)} profiles: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
