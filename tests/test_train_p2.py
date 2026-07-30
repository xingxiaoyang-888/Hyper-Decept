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

    assert len(train) == 2
    assert len(validation) == 2
    assert calls == [
        ("real:twibot22:test", "train"),
        ("sim:scenario:n4:s1:main", None),
        ("real:twibot22:test", "validation"),
        ("sim:scenario:n4:s2:main", None),
    ]


def test_mean_metrics_ignores_nonfinite_values():
    module = _load_runner()
    result = module._mean_metrics([
        {"auprc": 0.5, "auroc": float("nan")},
        {"auprc": 0.7, "auroc": 0.8},
    ])
    assert result == {"auprc": 0.6, "auroc": 0.8}
