"""Shared contracts for heterogeneous HyperTrace dataset adapters.

Concrete dataset loaders live in dataset-specific modules.  Keeping only the
shared capability and bundle contracts here prevents a legacy export format
from being selected implicitly by a generic adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetCapabilities:
    temporal: bool
    stable_post_ids: bool
    external_neighbors: bool
    ground_truth_roles: bool
    raw_text: bool
    interaction_target_ids: bool
    supervised_labels: bool

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class UnifiedDatasetBundle:
    dataset_kind: str
    capabilities: DatasetCapabilities
    core_users: pd.DataFrame
    boundary_users: pd.DataFrame
    labels: pd.DataFrame
    follow_edges: pd.DataFrame
    actions: pd.DataFrame
    warnings: List[str] = field(default_factory=list)

    def manifest(self) -> dict:
        manifest = {
            "dataset_kind": self.dataset_kind,
            "capabilities": self.capabilities.to_dict(),
            "counts": {
                "core_users": int(len(self.core_users)),
                "boundary_users": int(len(self.boundary_users)),
                "follow_edges": int(len(self.follow_edges)),
                "actions": int(len(self.actions)),
            },
            "warnings": list(self.warnings),
        }
        extra = getattr(self, "manifest_extra", None)
        if isinstance(extra, dict):
            manifest["extra"] = extra
        return manifest


def normalize_id(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    return text
