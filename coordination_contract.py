"""Shared runtime and checkpoint compatibility contract for coordination models."""

from __future__ import annotations

from typing import Mapping

import torch


def migrate_legacy_coordination_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map pre-migration ``bot_head`` checkpoint keys to the coordination head.

    Legacy names are accepted only at this serialization boundary. Runtime
    batches and model outputs use the coordination task contract exclusively.
    """
    migrated = state_dict.copy()
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        migrated._metadata = metadata.copy()
    for old_key in tuple(migrated):
        if old_key == "bot_head" or old_key.startswith("bot_head."):
            new_key = old_key.replace("bot_head", "coordination_head", 1)
        elif ".bot_head." in old_key:
            new_key = old_key.replace(".bot_head.", ".coordination_head.", 1)
        else:
            continue
        if new_key in migrated:
            raise ValueError(
                "checkpoint contains conflicting legacy and coordination keys: "
                f"{old_key!r} and {new_key!r}"
            )
        migrated[new_key] = migrated.pop(old_key)
    return migrated


class CoordinationCheckpointMixin:
    """Load legacy detector heads without exposing legacy runtime aliases."""

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        migrated = migrate_legacy_coordination_state_dict(state_dict)
        if assign:
            return super().load_state_dict(migrated, strict=strict, assign=True)
        # ``assign`` was added after the oldest supported PyTorch release.
        return super().load_state_dict(migrated, strict=strict)
