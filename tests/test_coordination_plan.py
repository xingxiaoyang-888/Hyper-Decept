import pandas as pd
import shutil

from data_processing.coordination_plan import (
    CoordinationCorpusPlan,
    CorpusEntry,
    campaign_disjoint_split,
)


def test_campaign_disjoint_split_is_deterministic_and_complete():
    campaigns = [f"campaign-{index}" for index in range(100)]
    first = campaign_disjoint_split(campaigns, seed=42)
    second = campaign_disjoint_split(reversed(campaigns), seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["data_split"]) == {"train", "validation", "test"}
    assert first["campaign_id"].is_unique


def test_campaign_disjoint_split_changes_with_seed():
    campaigns = [f"campaign-{index}" for index in range(100)]
    first = campaign_disjoint_split(campaigns, seed=42)
    second = campaign_disjoint_split(campaigns, seed=43)
    assert not first["data_split"].equals(second["data_split"])


def test_coordination_plan_roundtrip_is_relocatable(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    audit = root / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    plan = CoordinationCorpusPlan("test", (
        CorpusEntry("crypto-campaign", "real_campaign_adaptation", str(root),
                    str(audit), supervision_tasks=("campaign",)),
    ))
    path = plan.write(tmp_path / "plans" / "plan.json")
    restored = CoordinationCorpusPlan.read(path)
    assert restored.entries[0].raw_root == str(root.resolve())
    assert restored.audit(require_files=True)["valid"]

    moved = tmp_path / "server-bundle"
    shutil.copytree(tmp_path / "plans", moved / "plans")
    shutil.copytree(root, moved / "data")
    moved_plan = CoordinationCorpusPlan.read(moved / "plans" / "plan.json")
    assert moved_plan.audit(require_files=True)["valid"]
