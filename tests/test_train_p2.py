import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from data_processing.episode_manifest import EpisodeManifest, DatasetPlan


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "train_p2.py"
    spec = importlib.util.spec_from_file_location("train_p2_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _real_episode():
    return EpisodeManifest(
        episode_id="real:twibot22:test",
        dataset_name="twibot22",
        domain="real",
        purpose="real_primary",
        partition="shared",
        split_level="node",
        source_path="unused",
        identity_scope="dataset",
        artifacts={"features_csv": "unused", "labels_csv": "unused"},
    )


def _synthetic_episode(seed):
    return EpisodeManifest(
        episode_id=f"sim:scenario:n4:s{seed}:main",
        dataset_name="deeppersona_oasis",
        domain="synthetic",
        purpose="simulation_main",
        partition="pool",
        split_level="scenario",
        source_path="unused",
        identity_scope="episode",
        scenario_id=f"scenario_{seed}",
        simulation_seed=seed,
        num_agents=4,
        artifacts={
            "features_csv": "unused",
            "labels_csv": "unused",
        },
    )


def test_load_protocol_batches_respects_real_node_and_synthetic_episode_splits(
    monkeypatch,
):
    module = _load_runner()
    plan = DatasetPlan(
        plan_id="runner-test",
        episodes=(_real_episode(), _synthetic_episode(1), _synthetic_episode(2)),
    )
    assignments = {
        plan.episodes[0].episode_id: "shared",
        plan.episodes[1].episode_id: "train",
        plan.episodes[2].episode_id: "validation",
    }
    calls = []

    def fake_loader(manifest, **kwargs):
        calls.append((manifest.episode_id, kwargs["node_split"]))
        return SimpleNamespace(
            domain=manifest.domain,
            dataset_name=manifest.dataset_name,
            bot_mask=torch.tensor([True]),
        )

    monkeypatch.setattr(module, "load_episode_batch_from_manifest", fake_loader)
    train = module.load_protocol_batches(
        plan,
        assignments,
        split="train",
        domain="synthetic",
        role_vocabulary={},
        action_vocabulary={},
    )
    validation = module.load_protocol_batches(
        plan,
        assignments,
        split="validation",
        role_vocabulary={},
        action_vocabulary={},
    )

    assert len(train) == 1
    assert len(validation) == 2
    assert calls == [
        ("sim:scenario:n4:s1:main", None),
        ("real:twibot22:test", "validation"),
        ("sim:scenario:n4:s2:main", None),
    ]


def test_load_protocol_batches_filters_real_domain(monkeypatch):
    module = _load_runner()
    plan = DatasetPlan(
        plan_id="runner-domain-test",
        episodes=(_real_episode(), _synthetic_episode(1)),
    )
    assignments = {
        plan.episodes[0].episode_id: "shared",
        plan.episodes[1].episode_id: "train",
    }
    calls = []

    def fake_loader(manifest, **kwargs):
        calls.append(manifest.episode_id)
        return SimpleNamespace(
            domain=manifest.domain,
            dataset_name=manifest.dataset_name,
            bot_mask=torch.tensor([True]),
        )

    monkeypatch.setattr(module, "load_episode_batch_from_manifest", fake_loader)
    batches = module.load_protocol_batches(
        plan,
        assignments,
        split="train",
        domain="real",
        role_vocabulary={},
        action_vocabulary={},
    )

    assert len(batches) == 1
    assert calls == ["real:twibot22:test"]


def test_load_protocol_batches_includes_external_real_test_split(monkeypatch):
    module = _load_runner()
    base = _real_episode()
    external = base.__class__(**{
        **base.to_dict(),
        "episode_id": "real:external:test",
        "dataset_name": "external",
        "purpose": "real_temporal_external",
        "partition": "external_test",
    })
    plan = DatasetPlan(plan_id="runner-external-test", episodes=(external,))
    assignments = {external.episode_id: "test"}
    calls = []

    def fake_loader(manifest, **kwargs):
        calls.append(kwargs["node_split"])
        return SimpleNamespace(
            domain="real",
            dataset_name=manifest.dataset_name,
            bot_mask=torch.tensor([True]),
        )

    monkeypatch.setattr(module, "load_episode_batch_from_manifest", fake_loader)
    batches = module.load_protocol_batches(
        plan,
        assignments,
        split="test",
        domain="real",
        role_vocabulary={},
        action_vocabulary={},
    )

    assert len(batches) == 1
    assert calls == ["test"]


def test_mean_metrics_ignores_nonfinite_values():
    module = _load_runner()
    result = module._mean_metrics([
        {"auprc": 0.5, "auroc": float("nan")},
        {"auprc": 0.7, "auroc": 0.8},
    ])
    assert result == {"auprc": 0.6, "auroc": 0.8}


def test_action_vocabulary_uses_only_training_episodes(tmp_path):
    module = _load_runner()
    import pandas as pd

    train_path = tmp_path / "train.events.csv"
    test_path = tmp_path / "test.events.csv"
    pd.DataFrame({"next_action": ["post", "reply"]}).to_csv(train_path, index=False)
    pd.DataFrame({"next_action": ["unseen_test_action"]}).to_csv(test_path, index=False)
    train = _synthetic_episode(1)
    test = _synthetic_episode(2)
    train = train.__class__(**{
        **train.to_dict(),
        "artifacts": {"features_csv": "unused", "labels_csv": "unused", "event_targets_csv": str(train_path)},
    })
    test = test.__class__(**{
        **test.to_dict(),
        "artifacts": {"features_csv": "unused", "labels_csv": "unused", "event_targets_csv": str(test_path)},
    })
    plan = DatasetPlan(plan_id="vocab", episodes=(train, test))
    assignments = {train.episode_id: "train", test.episode_id: "test"}

    assert module._action_vocabulary(plan, assignments) == {"post": 0, "reply": 1}
