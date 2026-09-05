"""Instantiate auditable formal populations from DeepPersona prototypes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

try:
    from deeppersona_ai.profile_chunker import chunk_profile
except ModuleNotFoundError:
    from profile_chunker import chunk_profile


DEFAULT_REPO_ID = "THzva/deeppersona_dataset"
DEFAULT_REPO_FILE = "profiles_examples/profile_world_4.1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(local_source: Path | None) -> Path:
    if local_source is not None:
        return local_source.expanduser().resolve()
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=DEFAULT_REPO_ID,
        repo_type="dataset",
        filename=DEFAULT_REPO_FILE,
    ))


def _attribute_count(source_key: str) -> int:
    match = re.search(r"Count_(\d+)", source_key)
    return int(match.group(1)) if match else 0


def instantiate_population(
    prototypes: list[tuple[str, dict]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if count <= 0:
        raise ValueError("count must be positive")
    usable = [
        (source_key, profile)
        for source_key, profile in prototypes
        if isinstance(profile, dict)
        and isinstance(profile.get("Summary"), str)
        and profile["Summary"].strip()
    ]
    if not usable:
        raise ValueError("no prototypes contain a non-empty Summary")

    rng = random.Random(seed)
    assignments: list[tuple[str, dict]] = []
    while len(assignments) < count:
        cycle = list(usable)
        rng.shuffle(cycle)
        assignments.extend(cycle)
    assignments = assignments[:count]

    profiles = []
    chunks = []
    provenance = []
    for agent_id, (source_key, prototype) in enumerate(assignments):
        profile = deepcopy(prototype)
        profile["Profile Index"] = agent_id + 1
        profiles.append(profile)
        chunks.extend(chunk_profile(profile, agent_id))
        provenance.append({
            "agent_id": agent_id,
            "prototype_id": source_key,
            "prototype_profile_index": prototype.get("Profile Index"),
        })
    return profiles, chunks, provenance


def prepare_seed(
    *,
    source: Path,
    output_dir: Path,
    count: int,
    seed: int,
    min_attributes: int = 200,
) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DeepPersona source must contain a JSON object")
    prototypes = [
        item for item in payload.items()
        if _attribute_count(item[0]) >= min_attributes
    ]
    if not prototypes:
        raise ValueError(
            f"no prototypes satisfy min_attributes={min_attributes}"
        )
    profiles, chunks, provenance = instantiate_population(
        prototypes,
        count=count,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / f"seed_{seed}.profiles.json"
    chunks_path = output_dir / f"seed_{seed}.chunks.json"
    manifest_path = output_dir / f"seed_{seed}.personas.manifest.json"
    profiles_path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prototype_counts: dict[str, int] = {}
    for assignment in provenance:
        prototype_id = assignment["prototype_id"]
        prototype_counts[prototype_id] = prototype_counts.get(prototype_id, 0) + 1
    manifest = {
        "schema_version": "hyperdecept.persona-population.v1",
        "source_repository": DEFAULT_REPO_ID,
        "source_file": DEFAULT_REPO_FILE,
        "source_sha256": _sha256(source),
        "seed": seed,
        "agents": count,
        "prototype_count": len(prototypes),
        "min_attributes": min_attributes,
        "design": "balanced_seeded_prototype_instantiation",
        "independence_claim": (
            "Agents are simulation instances, not independently collected "
            "human personas. Role, graph, activity, and scenario variation are "
            "assigned downstream."
        ),
        "profiles_path": str(profiles_path.resolve()),
        "chunks_path": str(chunks_path.resolve()),
        "profiles_sha256": _sha256(profiles_path),
        "chunks_sha256": _sha256(chunks_path),
        "prototype_counts": prototype_counts,
        "assignments": provenance,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--min-attributes", type=int, default=200)
    parser.add_argument("--seeds", default="11,22,33,44")
    args = parser.parse_args()
    source = _source_path(args.source)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must contain unique integer values")
    reports = [
        prepare_seed(
            source=source,
            output_dir=args.output_dir.expanduser().resolve(),
            count=args.count,
            seed=seed,
            min_attributes=args.min_attributes,
        )
        for seed in seeds
    ]
    print(json.dumps({
        "status": "passed",
        "source": str(source),
        "seeds": seeds,
        "agents_per_seed": args.count,
        "prototype_count": reports[0]["prototype_count"],
        "min_attributes": args.min_attributes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
