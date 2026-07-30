from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from data_processing.p2_smoke_report import (
    build_p2_smoke_report,
    write_p2_smoke_report,
)


def _sha(value: str) -> str:
    return value * 64


def test_p2_report_is_portable_hashed_and_path_free(tmp_path):
    secret_root = tmp_path / "Users" / "alexmason" / "Hyper-Decept"
    secret_root.mkdir(parents=True)
    artifact = secret_root / "labels.csv"
    artifact.write_text("user_id,label\nu1,1\n", encoding="utf-8")
    adapter = secret_root / "adapter_manifest.json"
    adapter.write_text(json.dumps({
        "artifacts": {
            "labels": {
                "path": str(artifact),
                "rows": 1,
                "bytes": artifact.stat().st_size,
                "sha256": _sha("a"),
            }
        }
    }), encoding="utf-8")
    manifest = secret_root / "episode.manifest.json"
    manifest.write_text(json.dumps({
        "episode_id": "twibot-smoke",
        "dataset_name": "twibot22",
        "domain": "real",
        "purpose": "real_primary",
        "partition": "shared",
        "split_level": "node",
        "status": "ready",
        "num_agents": 1000,
        "source_sha256": _sha("b"),
        "artifacts": {
            "labels_csv": str(artifact),
            "adapter_manifest_json": str(adapter),
        },
    }), encoding="utf-8")
    checkpoint = secret_root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    started = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
    report = build_p2_smoke_report(
        summary={
            "status": "passed",
            "checkpoint": str(checkpoint),
            "twibot_users": 1000,
            "twibot_train_labels": 700,
        },
        manifest_paths=[manifest],
        started_at=started,
        finished_at=started + timedelta(seconds=5),
        provenance={
            "git_commit": _sha("c"),
            "git_branch": "main",
            "git_dirty_at_start": False,
        },
        environment={"python": "3.11", "gpu": None},
    )
    serialized = json.dumps(report)
    assert "/Users/" not in serialized
    assert "alexmason" not in serialized
    assert report["datasets"][0]["adapter_artifacts"]["labels"]["rows"] == 1
    assert report["datasets"][0]["observed_users"] == 1000
    assert len(report["checkpoint_integrity"]["sha256"]) == 64
    assert report["timing"]["duration_seconds"] == 5.0

    json_path, markdown_path = write_p2_smoke_report(report, tmp_path / "reports")
    assert json_path.name == "20260730T080000Z_cccccccc.json"
    assert markdown_path.is_file()
    assert "Smoke/interface metrics are not paper results" in markdown_path.read_text()
    assert "| twibot22 | real | 1000 |" in markdown_path.read_text()


def test_p2_report_rejects_naive_datetimes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    try:
        build_p2_smoke_report(
            summary={"checkpoint": str(checkpoint)},
            manifest_paths=[manifest],
            started_at=datetime(2026, 7, 30),
            finished_at=datetime(2026, 7, 30),
            provenance={},
            environment={},
        )
    except ValueError as error:
        assert "timezone-aware" in str(error)
    else:
        raise AssertionError("naive datetimes must be rejected")
