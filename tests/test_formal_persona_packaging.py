import hashlib
import json

import pytest

from deeppersona_ai.package_formal_personas import package_population


def _profile(index):
    return {
        "Profile Index": index,
        "Demographic Information": {
            f"field_{value}": f"profile_{index}_value_{value}"
            for value in range(200)
        },
        "Summary": " ".join([f"grounded{index}"] * 80),
    }


def _inputs(tmp_path, profiles):
    source = tmp_path / "formal_dp_personas.json"
    source.write_text(json.dumps(profiles), encoding="utf-8")
    report = tmp_path / "generation_report.json"
    report.write_text(json.dumps({
        "schema_version": "hyperdecept.formal-dp-personas.v1",
        "generator": "DeepPersona test engine",
        "design": "independently_generated_fixed_population_reused_across_episode_seeds",
        "requested_count": len(profiles),
        "total_valid_profiles": len(profiles),
        "attribute_count": 200,
        "consolidated_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return source, report


def test_package_population_attests_unique_shared_population(tmp_path):
    source, report = _inputs(tmp_path, [_profile(1), _profile(2)])
    manifest = package_population(
        source=source,
        generation_report=report,
        output_dir=tmp_path / "package",
        count=2,
        attribute_count=200,
    )
    assert manifest["agents"] == 2
    assert manifest["unique_content_hashes"] == 2
    assert len(manifest["records"]) == 2
    assert (tmp_path / "package/formal_dp_personas.chunks.json").is_file()


def test_package_population_rejects_duplicated_content(tmp_path):
    first = _profile(1)
    duplicate = {**first, "Profile Index": 2}
    source, report = _inputs(tmp_path, [first, duplicate])
    with pytest.raises(ValueError, match="duplicate generated persona content"):
        package_population(
            source=source,
            generation_report=report,
            output_dir=tmp_path / "package",
            count=2,
            attribute_count=200,
        )
