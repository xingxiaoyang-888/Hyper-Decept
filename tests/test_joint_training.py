import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from data_processing.episode_manifest import EpisodeManifest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "Character Classification"
    / "joint_training.py"
)


def _load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("joint_training_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _graph(feature_dim, include_tweet=False):
    graph = HeteroData()
    graph["user"].x = torch.randn(4, feature_dim)
    graph["user"].node_ids = ["u0", "u1", "u2", "u3"]
    graph["user", "follows", "user"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
    )
    if include_tweet:
        graph["tweet"].x = torch.randn(3, feature_dim + 2)
        graph["user", "posts", "tweet"].edge_index = torch.tensor(
            [[0, 1, 3], [0, 1, 2]], dtype=torch.long
        )
        graph["tweet", "authored_by", "user"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 3]], dtype=torch.long
        )
    return graph


def _real_labels():
    return pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        "is_bad": [0, 0, 1, 1],
        # These columns must not become real ground-truth task masks.
        "role": ["leader", "member", "leader", "member"],
        "campaign_id": ["a", "a", "b", "b"],
        "next_action": ["post", "reply", "post", "reply"],
    })


def _synthetic_labels():
    return pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        "is_bad": [0, 0, 1, 1],
        "role": ["member", "member", "leader", "amplifier"],
        "campaign_id": ["organic", "organic", "campaign-x", "campaign-x"],
        "next_action": ["post", "reply", "post", "reply"],
    })


def _batches(module):
    real_graph = _graph(4)
    synthetic_graph = _graph(6, include_tweet=True)
    real = module.build_episode_batch(
        real_graph,
        episode_id="real:toy",
        domain="real",
        labels_frame=_real_labels(),
    )
    synthetic = module.build_episode_batch(
        synthetic_graph,
        episode_id="sim:toy",
        domain="synthetic",
        labels_frame=_synthetic_labels(),
        role_vocabulary={"member": 0, "leader": 1, "amplifier": 2},
        action_vocabulary={"post": 0, "reply": 1},
    )
    return real, synthetic


def test_target_alignment_keeps_privileged_labels_synthetic_only():
    module = _load_module()
    real, synthetic = _batches(module)
    assert real.bot_mask.all()
    assert not real.role_mask.any()
    assert not real.campaign_mask.any()
    assert not real.temporal_action_mask.any()
    assert synthetic.role_mask.all()
    assert synthetic.campaign_mask.all()
    assert synthetic.temporal_action_mask.all()
    assert set(synthetic.campaign_targets.tolist()) == {0, 1}


def test_metadata_union_supports_different_episode_schemas():
    module = _load_module()
    real, synthetic = _batches(module)
    metadata = module.merge_heterogeneous_metadata([real.graph, synthetic.graph])
    assert metadata[0] == ("tweet", "user")
    assert ("user", "follows", "user") in metadata[1]
    assert ("user", "posts", "tweet") in metadata[1]


def test_joint_model_accepts_domain_specific_feature_dimensions():
    module = _load_module()
    real, synthetic = _batches(module)
    metadata = module.merge_heterogeneous_metadata([real.graph, synthetic.graph])
    model = module.DomainAwareLorentzHGT(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        metadata=metadata,
        num_roles=3,
        num_temporal_actions=2,
        dropout=0.0,
    )
    real_output = model(real.graph, domain="real")
    synthetic_output = model(synthetic.graph, domain="synthetic")
    assert real_output["bot_logits"].shape == (4,)
    assert "role_logits" not in real_output
    assert synthetic_output["role_logits"].shape == (4, 3)
    assert synthetic_output["temporal_action_logits"].shape == (4, 2)
    assert synthetic_output["campaign_embedding"].shape == (4, 16)


def test_joint_train_step_routes_all_available_losses_and_backpropagates():
    torch.manual_seed(7)
    module = _load_module()
    real, synthetic = _batches(module)
    metadata = module.merge_heterogeneous_metadata([real.graph, synthetic.graph])
    model = module.DomainAwareLorentzHGT(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        metadata=metadata,
        num_roles=3,
        num_temporal_actions=2,
        dropout=0.0,
    )
    # Materialize lazy domain adapters before creating the optimizer.
    model(real.graph, domain="real")
    model(synthetic.graph, domain="synthetic")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = module.DomainAlternatingTrainer(model, optimizer)
    metrics = trainer.train_step(real, synthetic)
    expected = {
        "loss_total",
        "loss_alignment",
        "loss_real_bot",
        "loss_real_relation",
        "loss_synthetic_bot",
        "loss_synthetic_role",
        "loss_synthetic_campaign",
        "loss_synthetic_temporal_action",
        "loss_synthetic_relation",
        "gradient_norm",
    }
    assert expected.issubset(metrics)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["gradient_norm"] > 0


def test_real_episode_rejects_privileged_ground_truth_masks():
    module = _load_module()
    graph = _graph(4)
    values = torch.zeros(4)
    mask = torch.zeros(4, dtype=torch.bool)
    privileged = torch.ones(4, dtype=torch.bool)
    try:
        module.EpisodeBatch(
            episode_id="bad-real",
            domain="real",
            graph=graph,
            bot_targets=values,
            bot_mask=torch.ones(4, dtype=torch.bool),
            role_targets=torch.zeros(4, dtype=torch.long),
            role_mask=privileged,
            campaign_targets=torch.zeros(4, dtype=torch.long),
            campaign_mask=mask,
            temporal_action_targets=torch.zeros(4, dtype=torch.long),
            temporal_action_mask=mask,
        )
    except ValueError as error:
        assert "privileged" in str(error)
    else:
        raise AssertionError("real privileged labels should be rejected")


def test_manifest_loader_reuses_graph_builder_and_explicit_feature_contract(tmp_path):
    module = _load_module()
    db_path = tmp_path / "episode.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE user (user_id TEXT)")
        connection.execute(
            "CREATE TABLE post (post_id TEXT, user_id TEXT, content TEXT)"
        )
    feature_values = {
        name: [float(index + offset) for index in range(4)]
        for offset, name in enumerate(module.DEFAULT_FEATURE_COLUMNS)
    }
    features_path = tmp_path / "episode.features.csv"
    pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        **feature_values,
    }).to_csv(features_path, index=False)
    labels_path = tmp_path / "episode.labels.csv"
    pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        "user_type": ["good", "good", "bad_leader", "bad_member"],
    }).to_csv(labels_path, index=False)
    targets_path = tmp_path / "episode.event_targets.csv"
    pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        "campaign_id": ["organic", "organic", "x", "x"],
        "next_action": ["post", "reply", "post", "reply"],
    }).to_csv(targets_path, index=False)
    manifest = EpisodeManifest(
        episode_id="sim:loader",
        dataset_name="simulation",
        domain="synthetic",
        purpose="simulation_main",
        partition="pool",
        split_level="scenario",
        source_path=str(db_path),
        identity_scope="episode",
        scenario_id="loader",
        simulation_seed=1,
        num_agents=4,
        artifacts={
            "features_csv": str(features_path),
            "labels_csv": str(labels_path),
            "event_targets_csv": str(targets_path),
        },
    )
    batch = module.load_episode_batch_from_manifest(
        manifest,
        role_vocabulary={"organic": 0, "leader": 1, "member": 2},
        action_vocabulary={"post": 0, "reply": 1},
        similarity_threshold=1.1,
    )
    assert batch.graph["user"].node_ids == ["u0", "u1", "u2", "u3"]
    assert batch.bot_targets.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert batch.role_mask.all()
    assert batch.temporal_action_mask.all()


def test_neighbor_sample_masks_context_users_out_of_supervised_losses():
    module = _load_module()
    _, parent = _batches(module)
    sample = HeteroData()
    sample["user"].x = parent.graph["user"].x[[2, 0, 3]]
    sample["user"].n_id = torch.tensor([2, 0, 3])
    sample["user"].batch_size = 2
    sample["user", "follows", "user"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]], dtype=torch.long
    )
    sliced = module.episode_batch_from_neighbor_sample(parent, sample)
    assert sliced.graph["user"].node_ids == ["u2", "u0", "u3"]
    assert sliced.bot_targets.tolist() == [1.0, 0.0, 1.0]
    assert sliced.bot_mask.tolist() == [True, True, False]
    assert sliced.role_mask.tolist() == [True, True, False]


def test_manifest_loader_applies_real_node_split_without_graph_leakage(tmp_path):
    module = _load_module()
    db_path = tmp_path / "real.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE user (user_id TEXT)")
        connection.execute(
            "CREATE TABLE post (post_id TEXT, user_id TEXT, content TEXT)"
        )
    feature_values = {
        name: [float(index + offset) for index in range(4)]
        for offset, name in enumerate(module.DEFAULT_FEATURE_COLUMNS)
    }
    features_path = tmp_path / "real.features.csv"
    pd.DataFrame({
        "user_id": ["u0", "u1", "u2", "u3"],
        **feature_values,
    }).to_csv(features_path, index=False)
    labels_path = tmp_path / "real.labels.csv"
    pd.DataFrame({
        "id": ["u0", "u1", "u2", "u3"],
        "label": ["human", "bot", "human", "bot"],
        "role": ["not-ground-truth"] * 4,
    }).to_csv(labels_path, index=False)
    splits_path = tmp_path / "real.splits.csv"
    pd.DataFrame({
        "id": ["u0", "u1", "u2", "u3"],
        "split": ["train", "train", "val", "test"],
    }).to_csv(splits_path, index=False)
    manifest = EpisodeManifest(
        episode_id="real:loader",
        dataset_name="real",
        domain="real",
        purpose="real_primary",
        partition="shared",
        split_level="node",
        source_path=str(db_path),
        identity_scope="dataset",
        artifacts={
            "features_csv": str(features_path),
            "labels_csv": str(labels_path),
            "splits_csv": str(splits_path),
        },
    )
    train = module.load_episode_batch_from_manifest(
        manifest, node_split="train", similarity_threshold=1.1
    )
    validation = module.load_episode_batch_from_manifest(
        manifest, node_split="val", similarity_threshold=1.1
    )
    assert train.bot_mask.tolist() == [True, True, False, False]
    assert validation.bot_mask.tolist() == [False, False, True, False]
    assert not train.role_mask.any()


def test_evaluation_and_checkpoint_record_calibration_and_geometry(tmp_path):
    module = _load_module()
    real, synthetic = _batches(module)
    metadata = module.merge_heterogeneous_metadata([real.graph, synthetic.graph])
    model = module.DomainAwareLorentzHGT(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        metadata=metadata,
        num_roles=3,
        num_temporal_actions=2,
        dropout=0.0,
    )
    model(real.graph, domain="real")
    model(synthetic.graph, domain="synthetic")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = module.evaluate_bot_batch(model, real)
    assert {"auroc", "auprc", "f1", "brier", "ece"}.issubset(metrics)
    checkpoint = module.save_joint_checkpoint(
        tmp_path / "model.pt",
        model=model,
        optimizer=optimizer,
        loss_config=module.JointLossConfig(),
        epoch=2,
        plan_id="plan-v1",
        fold_id="holdout-adaptive-evasion",
        metrics=metrics,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["schema_version"] == "hyperdecept.joint-checkpoint.v1"
    assert payload["plan_id"] == "plan-v1"
    assert payload["geometry"]["geometry_backend"] == (
        "domain_aware_intrinsic_lorentz"
    )
