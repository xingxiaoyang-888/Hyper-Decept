import json
from pathlib import Path

import pytest

from scripts.frozen_manifest import resolve_frozen_checkpoint
from scripts.run_frozen_external_io import _sha256, _warm_start_provenance


def _checkpoint(root: Path) -> Path:
    path = root / "adaptive_evasion" / "seed_7" / "best_checkpoint.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"checkpoint")
    return path.resolve()


def test_resolves_manifest_relative_v2_path(tmp_path: Path) -> None:
    manifest = tmp_path / "freeze" / "manifest.json"
    manifest.parent.mkdir()
    expected = _checkpoint(manifest.parent)
    entry = {
        "frozen_checkpoint": "adaptive_evasion/seed_7/best_checkpoint.pt",
        "held_out_scenario": "adaptive_evasion",
        "seed": 7,
    }

    assert resolve_frozen_checkpoint(manifest, entry) == expected


def test_relocates_stale_v1_absolute_path(tmp_path: Path) -> None:
    manifest = tmp_path / "freeze" / "manifest.json"
    manifest.parent.mkdir()
    expected = _checkpoint(manifest.parent)
    entry = {
        "frozen_checkpoint": (
            "/old/server/frozen/adaptive_evasion/seed_7/best_checkpoint.pt"
        ),
        "held_out_scenario": "adaptive_evasion",
        "seed": 7,
    }

    assert resolve_frozen_checkpoint(manifest, entry) == expected


def test_rejects_relative_path_outside_manifest_root(tmp_path: Path) -> None:
    manifest = tmp_path / "freeze" / "manifest.json"
    manifest.parent.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError):
        resolve_frozen_checkpoint(
            manifest,
            {"frozen_checkpoint": "../outside.pt"},
        )


def test_freeze_manifest_supplies_missing_checkpoint_source(tmp_path: Path) -> None:
    manifest = tmp_path / "freeze" / "manifest.json"
    manifest.parent.mkdir()
    checkpoint = _checkpoint(manifest.parent)
    entry = {
        "frozen_checkpoint": "adaptive_evasion/seed_7/best_checkpoint.pt",
        "held_out_scenario": "adaptive_evasion",
        "seed": 7,
        "sha256": _sha256(checkpoint),
        "warm_start": {"source_operation": "honduras"},
    }
    manifest.write_text(json.dumps({"checkpoints": [entry]}), encoding="utf-8")

    warm_start, provenance, manifest_hash = _warm_start_provenance(
        checkpoint,
        {"warm_start_report": None},
        manifest,
    )

    assert warm_start["source_operation"] == "honduras"
    assert provenance == "freeze_manifest"
    assert manifest_hash == _sha256(manifest)
