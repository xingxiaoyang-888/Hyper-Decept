from types import SimpleNamespace

import pytest
import torch

from scripts.train_base import (
    JointLossConfig,
    _restore_rng_state,
    _positive_class_weight,
    compute_episode_losses,
)


def test_atomic_torch_save_replaces_complete_checkpoint(tmp_path) -> None:
    from scripts.train_base import _atomic_torch_save

    target = tmp_path / "last_checkpoint.pt"
    _atomic_torch_save({"epoch": 1}, target)
    _atomic_torch_save({"epoch": 2}, target)
    assert torch.load(target, map_location="cpu", weights_only=False)["epoch"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def _batch(targets):
    values = torch.tensor(targets, dtype=torch.float32)
    return SimpleNamespace(
        coordination_targets=values,
        coordination_mask=torch.ones(values.shape[0], dtype=torch.bool),
    )


def test_positive_class_weight_uses_training_prevalence():
    assert _positive_class_weight([_batch([0, 0, 0, 1]), _batch([0, 1])]) == 2.0


def test_positive_class_weight_requires_both_classes():
    with pytest.raises(ValueError, match="both coordination classes"):
        _positive_class_weight([_batch([0, 0])])


def test_coordination_only_objective_skips_privileged_targets():
    batch = SimpleNamespace(
        domain="synthetic",
        coordination_mask=torch.tensor([True, True]),
        coordination_targets=torch.tensor([0.0, 1.0]),
        role_mask=torch.tensor([True, True]),
        role_targets=torch.tensor([0, 1]),
        campaign_mask=torch.tensor([True, True]),
        campaign_targets=torch.tensor([0, 1]),
        temporal_action_mask=torch.tensor([True, True]),
        temporal_action_targets=torch.tensor([0, 1]),
    )
    output = {
        "user_tangent": torch.ones((2, 2), requires_grad=True),
        "coordination_logits": torch.tensor([0.0, 0.0], requires_grad=True),
    }
    losses = compute_episode_losses(
        SimpleNamespace(),
        output,
        batch,
        JointLossConfig(privileged_weight=0.0, alignment_weight=0.0),
    )
    assert set(losses) == {"detection", "total"}


def test_rng_restore_accepts_serialized_cpu_state():
    state = {
        "python": __import__("random").getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "torch": torch.get_rng_state(),
    }
    _restore_rng_state(state)
