import json

from deeppersona_ai.generate_formal_dp_personas import (
    _expected_values,
    generate_population,
    validate_profile,
)


def _profile(index=1):
    return {
        "Profile Index": index,
        "Demographic Information": {
            f"field_{value}": f"value_{value}" for value in range(150)
        },
        "Summary": " ".join(["grounded"] * 80),
    }


def test_validate_profile_enforces_depth_and_summary():
    assert validate_profile(_profile(), minimum_leaves=150)["valid"] is True
    shallow = _profile()
    shallow["Demographic Information"] = {"age": 20}
    quality = validate_profile(shallow, minimum_leaves=150)
    assert quality["valid"] is False
    assert "nonempty_leaves_below_150" in quality["errors"]


def test_expected_values_accepts_nested_and_dotted_json():
    paths = [
        "Demographic Information.Location.timezone",
        "Education and Learning.skills.Technical Skills",
    ]
    payload = {
        "Demographic Information": {
            "Location": {"timezone": "UTC+8"},
        },
        "Education and Learning.skills.Technical Skills": "Python",
    }
    assert _expected_values(payload, paths) == {
        paths[0]: "UTC+8",
        paths[1]: "Python",
    }


def test_population_resume_skips_existing_valid_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    root = tmp_path / "formal"
    profiles = root / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "agent_00000.json").write_text(
        json.dumps(_profile(), ensure_ascii=False), encoding="utf-8"
    )
    report = generate_population(
        output_root=root,
        count=1,
        workers=1,
        seed=7,
        minimum_leaves=150,
    )
    assert report["total_valid_profiles"] == 1
    assert report["generated_this_run"] == 0
    assert report["failed_this_run"] == 0
