from pathlib import Path

import pytest

from data_processing.episode_manifest import (
    DatasetPlan,
    EpisodeManifest,
    audit_plan_artifacts,
    audit_episode_splits,
    build_recommended_dataset_plan,
    leave_one_scenario_out_assignments,
    training_protocol_assignments,
)


SCENARIOS = (
    "leader_amplifier",
    "bridge_infiltration",
    "synchronized_boosting",
    "persona_drift",
    "adaptive_evasion",
)


def _recommended(tmp_path):
    return build_recommended_dataset_plan(
        simulation_root=tmp_path / "simulation",
        twibot_root=tmp_path / "twibot22",
        mgtab_root=tmp_path / "mgtab",
        scenarios=SCENARIOS,
    )


def test_recommended_plan_uses_incomplete_factorial_design(tmp_path):
    plan = _recommended(tmp_path)
    summary = plan.summary()
    assert summary["episodes"] == 40
    assert summary["real_episodes"] == 2
    assert summary["synthetic_episodes"] == 38
    assert summary["synthetic_agent_instances"] == 79_000
    assert summary["scale_sizes"] == [500, 1000, 5000]
    artifact_report = audit_plan_artifacts(plan)
    assert artifact_report.valid
    assert artifact_report.warnings


def test_plan_round_trip_is_stable(tmp_path):
    plan = _recommended(tmp_path)
    path = plan.write(tmp_path / "plan.json")
    restored = DatasetPlan.read(path)
    assert restored.to_dict() == plan.to_dict()


def test_complete_blueprint_includes_optional_external_benchmarks(tmp_path):
    plan = build_recommended_dataset_plan(
        simulation_root=tmp_path / "simulation",
        twibot_root=tmp_path / "twibot22",
        mgtab_root=tmp_path / "mgtab",
        fox8_root=tmp_path / "fox8-23",
        botsim_root=tmp_path / "botsim-24",
        scenarios=SCENARIOS,
    )
    assert plan.summary()["episodes"] == 42
    datasets = {episode.dataset_name for episode in plan.episodes}
    assert {"twibot22", "mgtab", "fox8-23", "botsim-24"}.issubset(datasets)
    botsim = next(
        episode for episode in plan.episodes if episode.dataset_name == "botsim-24"
    )
    assert botsim.domain == "synthetic"
    assert botsim.identity_scope == "dataset"


def test_real_training_protocols_separate_holdout_from_multisource_use(tmp_path):
    plan = _recommended(tmp_path)
    p1 = training_protocol_assignments(
        plan,
        protocol_id="P1_external_holdout",
        held_out_scenario="adaptive_evasion",
    )
    p2 = training_protocol_assignments(
        plan,
        protocol_id="P2_multisource_real",
        held_out_scenario="adaptive_evasion",
    )
    mgtab_id = next(
        episode.episode_id for episode in plan.episodes
        if episode.dataset_name == "mgtab"
    )
    mgtab = next(
        episode for episode in plan.episodes
        if episode.dataset_name == "mgtab"
    )
    assert mgtab.capabilities["raw_text"] is False
    assert mgtab.capabilities["precomputed_text_embeddings"] is True
    assert mgtab.capabilities["external_neighbors"] is False
    assert p1[mgtab_id] == "test"
    assert p2[mgtab_id] == "shared"


def test_leave_one_scenario_out_keeps_main_scenario_disjoint(tmp_path):
    plan = _recommended(tmp_path)
    assignments = leave_one_scenario_out_assignments(
        plan, "adaptive_evasion", validation_seed=44
    )
    report = audit_episode_splits(
        plan,
        assignments,
        user_ids_by_episode={},
        campaign_ids_by_episode={},
    )
    assert report.valid
    main = [
        episode for episode in plan.episodes
        if episode.purpose == "simulation_main"
    ]
    assert all(
        assignments[episode.episode_id] == "test"
        for episode in main if episode.scenario_id == "adaptive_evasion"
    )
    assert all(
        assignments[episode.episode_id] == "validation"
        for episode in main
        if episode.scenario_id != "adaptive_evasion"
        and episode.simulation_seed == 44
    )


def test_audit_rejects_scenario_leakage(tmp_path):
    plan = _recommended(tmp_path)
    assignments = leave_one_scenario_out_assignments(plan, "adaptive_evasion")
    leaked = next(
        episode for episode in plan.episodes
        if episode.purpose == "simulation_main"
        and episode.scenario_id == "leader_amplifier"
        and assignments[episode.episode_id] == "train"
    )
    assignments[leaked.episode_id] = "test"
    report = audit_episode_splits(
        plan,
        assignments,
        user_ids_by_episode={},
        campaign_ids_by_episode={},
    )
    assert not report.valid
    assert any("scenarios cross train/test" in error for error in report.errors)


def test_audit_rejects_real_identity_overlap_across_splits(tmp_path):
    common = dict(
        dataset_name="realset",
        domain="real",
        purpose="real_primary",
        partition="shared",
        split_level="node",
        identity_scope="dataset",
        label_provenance={"bot": "annotated"},
    )
    plan = DatasetPlan(
        plan_id="overlap",
        episodes=(
            EpisodeManifest(
                episode_id="real-train",
                source_path=str(tmp_path / "train.db"),
                **common,
            ),
            EpisodeManifest(
                episode_id="real-test",
                source_path=str(tmp_path / "test.db"),
                **common,
            ),
        ),
    )
    report = audit_episode_splits(
        plan,
        {"real-train": "train", "real-test": "test"},
        user_ids_by_episode={
            "real-train": {"u1", "u2"},
            "real-test": {"u2", "u3"},
        },
        campaign_ids_by_episode={},
        require_scenario_holdout=False,
    )
    assert not report.valid
    assert any("user 'u2'" in error for error in report.errors)


def test_synthetic_local_user_ids_are_episode_scoped(tmp_path):
    common = dict(
        dataset_name="simulation",
        domain="synthetic",
        purpose="simulation_main",
        partition="pool",
        split_level="scenario",
        identity_scope="episode",
        num_agents=2,
        label_provenance={"bot": "generated"},
    )
    plan = DatasetPlan(
        plan_id="local-identities",
        episodes=(
            EpisodeManifest(
                episode_id="sim-a",
                source_path=str(tmp_path / "a.db"),
                scenario_id="a",
                simulation_seed=1,
                **common,
            ),
            EpisodeManifest(
                episode_id="sim-b",
                source_path=str(tmp_path / "b.db"),
                scenario_id="b",
                simulation_seed=2,
                **common,
            ),
        ),
    )
    report = audit_episode_splits(
        plan,
        {"sim-a": "train", "sim-b": "test"},
        user_ids_by_episode={"sim-a": {"0", "1"}, "sim-b": {"0", "1"}},
        campaign_ids_by_episode={},
    )
    assert report.valid


def test_ready_episode_requires_content_hash(tmp_path):
    with pytest.raises(ValueError, match="source_sha256"):
        EpisodeManifest(
            episode_id="ready-without-hash",
            dataset_name="realset",
            domain="real",
            purpose="real_primary",
            partition="shared",
            split_level="node",
            source_path=str(tmp_path / "data.db"),
            identity_scope="dataset",
            status="ready",
        )
