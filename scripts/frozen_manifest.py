"""Portable path handling for frozen checkpoint manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_frozen_checkpoint(
    manifest_path: Path, entry: Mapping[str, Any]
) -> Path:
    """Resolve v1 absolute or v2 manifest-relative checkpoint paths safely."""
    manifest_path = Path(manifest_path).resolve()
    manifest_root = manifest_path.parent
    declared = Path(str(entry["frozen_checkpoint"]))

    if declared.is_absolute() and declared.is_file():
        return declared.resolve()

    candidates: list[Path] = []
    if not declared.is_absolute():
        candidates.append(manifest_root / declared)

    scenario = entry.get("held_out_scenario")
    seed = entry.get("seed")
    if scenario is not None and seed is not None:
        candidates.append(
            manifest_root
            / str(scenario)
            / f"seed_{int(seed)}"
            / "best_checkpoint.pt"
        )

    # Relocated v1 manifests retain an obsolete absolute prefix. Its final
    # scenario/seed/file components are still a deterministic local locator.
    if declared.is_absolute() and len(declared.parts) >= 3:
        candidates.append(manifest_root.joinpath(*declared.parts[-3:]))

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if not _within(manifest_root, resolved):
            continue
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        "unable to resolve frozen checkpoint relative to manifest "
        f"{manifest_path}; declared={declared}; checked={checked}"
    )
