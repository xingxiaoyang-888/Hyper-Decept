import json

from deeppersona_ai.prepare_formal_personas import (
    instantiate_population,
    prepare_seed,
)


def _prototypes(count=3):
    return [
        (
            f"prototype-{index}",
            {
                "Profile Index": index + 10,
                "Summary": f"persona summary {index}",
                "Core Values, Beliefs, and Philosophy": {
                    "priority": f"value {index}"
                },
            },
        )
        for index in range(count)
    ]


def test_instances_are_balanced_aligned_and_reproducible():
    first = instantiate_population(_prototypes(), count=8, seed=11)
    second = instantiate_population(_prototypes(), count=8, seed=11)
    profiles, chunks, provenance = first
    assert first == second
    assert [profile["Profile Index"] for profile in profiles] == list(range(1, 9))
    assert {chunk["agent_id"] for chunk in chunks} == set(range(8))
    counts = {}
    for row in provenance:
        counts[row["prototype_id"]] = counts.get(row["prototype_id"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_seed_manifest_discloses_prototype_instantiation(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(dict(_prototypes()), ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = prepare_seed(
        source=source,
        output_dir=tmp_path / "personas",
        count=7,
        seed=22,
        min_attributes=0,
    )
    assert manifest["agents"] == 7
    assert manifest["prototype_count"] == 3
    assert "not independently collected" in manifest["independence_claim"]
    assert len(manifest["assignments"]) == 7
    chunks = json.loads(
        (tmp_path / "personas/seed_22.chunks.json").read_text(encoding="utf-8")
    )
    assert {row["agent_id"] for row in chunks} == set(range(7))
