"""Deterministic adapter for the standard MGTAB tensor release.

The standard release contains anonymous account nodes, two supervised label
vectors and seven user-to-user relation types.  This module keeps that native
contract intact: it does not manufacture Twitter IDs, raw text, timestamps,
campaigns or tactical roles.

``MGTABAdapter.load`` returns the three artifacts consumed by HyperDecept:

``graph``
    A :class:`torch_geometric.data.HeteroData` graph with one ``user`` node
    type and seven named relation stores.
``labels_frame``
    Node-aligned bot/stance labels and a deterministic transductive split.
``adapter_manifest``
    A JSON-serialisable record of every transformation applied by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import pandas as pd
import torch

from data_processing.mgtab_raw_audit import (
    BOT_LABELS,
    RELATION_TYPES,
    REQUIRED_FILES,
    STANCE_LABELS,
)


EdgeType = Tuple[str, str, str]
VALID_MULTIEDGE_POLICIES = {"coalesce_with_count", "preserve_multiedges"}
UNDIRECTED_RELATIONS = {"url", "hashtag"}


@dataclass(frozen=True)
class MGTABAdapterConfig:
    """Explicit policies used to translate the released multigraph."""

    split_seed: int = 42
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    multiedge_policy: str = "coalesce_with_count"
    add_reverse_undirected_edges: bool = True
    exclude_self_loops_from_message_passing: bool = True

    def __post_init__(self) -> None:
        if self.multiedge_policy not in VALID_MULTIEDGE_POLICIES:
            raise ValueError(
                "multiedge_policy must be coalesce_with_count or "
                "preserve_multiedges"
            )
        if len(self.split_ratios) != 3 or any(
            ratio <= 0.0 for ratio in self.split_ratios
        ):
            raise ValueError("split_ratios must contain three positive values")
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError("split_ratios must sum to one")


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value)!r}")
    return value.detach().cpu()


def _split_digest(frame: pd.DataFrame) -> str:
    canonical = frame[["user_id", "data_split"]].sort_values(
        "user_id", kind="stable"
    )
    payload = "".join(
        f"{row.user_id},{row.data_split}\n"
        for row in canonical.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_split_csv(labels_frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist the already-decided split without modifying raw MGTAB files."""
    required = {"user_id", "data_split"}
    missing = required.difference(labels_frame.columns)
    if missing:
        raise ValueError(f"labels_frame is missing split columns: {sorted(missing)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels_frame[["user_id", "data_split"]].to_csv(target, index=False)
    return target


class MGTABAdapter:
    """Load and transform the official MGTAB standard tensor bundle."""

    def __init__(
        self,
        root: str | Path,
        *,
        split_seed: int = 42,
        split_ratios: Sequence[float] = (0.7, 0.2, 0.1),
        multiedge_policy: str = "coalesce_with_count",
        add_reverse_undirected_edges: bool = True,
        exclude_self_loops_from_message_passing: bool = True,
    ) -> None:
        self.root = Path(root)
        self.config = MGTABAdapterConfig(
            split_seed=int(split_seed),
            split_ratios=tuple(float(value) for value in split_ratios),
            multiedge_policy=str(multiedge_policy),
            add_reverse_undirected_edges=bool(add_reverse_undirected_edges),
            exclude_self_loops_from_message_passing=bool(
                exclude_self_loops_from_message_passing
            ),
        )

    def _load_bundle(self) -> Dict[str, torch.Tensor]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"MGTAB directory does not exist: {self.root}")
        missing = [name for name in REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"MGTAB bundle is missing files: {missing}")
        bundle = {
            name: _load_tensor(self.root / name) for name in REQUIRED_FILES
        }
        self._validate_bundle(bundle)
        return bundle

    @staticmethod
    def _validate_bundle(bundle: Mapping[str, torch.Tensor]) -> None:
        features = bundle["features.pt"]
        edge_index = bundle["edge_index.pt"]
        edge_type = bundle["edge_type.pt"]
        edge_weight = bundle["edge_weight.pt"]
        labels_bot = bundle["labels_bot.pt"]
        labels_stance = bundle["labels_stance.pt"]

        if features.ndim != 2 or features.shape[1] != 788:
            raise ValueError("MGTAB features must have shape [num_nodes, 788]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("MGTAB features must be finite floating-point values")
        node_count = int(features.shape[0])
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("MGTAB edge_index must have shape [2, num_edges]")
        edge_count = int(edge_index.shape[1])
        if edge_type.shape != (edge_count,) or edge_weight.shape != (edge_count,):
            raise ValueError("edge_type and edge_weight must align with edge_index")
        if edge_index.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise ValueError("MGTAB edge_index must use an integer dtype")
        if edge_count and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count
        ):
            raise ValueError("MGTAB edge_index contains an out-of-range node")
        if edge_type.numel() and not set(edge_type.to(torch.int64).tolist()).issubset(
            RELATION_TYPES
        ):
            raise ValueError("MGTAB edge_type contains an unknown relation ID")
        if not edge_weight.is_floating_point() or not torch.isfinite(edge_weight).all():
            raise ValueError("MGTAB edge weights must be finite floating-point values")
        if torch.any(edge_weight <= 0):
            raise ValueError("MGTAB edge weights must be positive")
        if labels_bot.shape != (node_count,) or labels_stance.shape != (node_count,):
            raise ValueError("MGTAB label vectors must align with feature rows")
        if not set(labels_bot.to(torch.int64).tolist()).issubset(BOT_LABELS):
            raise ValueError("MGTAB bot labels must be 0 or 1")
        if not set(labels_stance.to(torch.int64).tolist()).issubset(STANCE_LABELS):
            raise ValueError("MGTAB stance labels must be 0, 1 or 2")

    def _make_labels(
        self,
        labels_bot: torch.Tensor,
        labels_stance: torch.Tensor,
    ) -> pd.DataFrame:
        from sklearn.model_selection import train_test_split

        node_count = int(labels_bot.numel())
        frame = pd.DataFrame({
            "user_id": [f"mgtab:{index}" for index in range(node_count)],
            "is_bad": labels_bot.to(torch.int64).numpy(),
            "stance": labels_stance.to(torch.int64).numpy(),
        })
        joint = frame["is_bad"].astype(str) + "|" + frame["stance"].astype(str)
        joint_counts = joint.value_counts()
        if len(joint_counts) < 2 or int(joint_counts.min()) < 3:
            raise ValueError(
                "bot x stance stratification requires at least three nodes in "
                "each observed joint class"
            )

        train_ratio, validation_ratio, test_ratio = self.config.split_ratios
        indices = frame.index.to_numpy()
        train_indices, holdout_indices = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=self.config.split_seed,
            shuffle=True,
            stratify=joint.to_numpy(),
        )
        holdout_joint = joint.iloc[holdout_indices].to_numpy()
        relative_test_ratio = test_ratio / (validation_ratio + test_ratio)
        validation_indices, test_indices = train_test_split(
            holdout_indices,
            test_size=relative_test_ratio,
            random_state=self.config.split_seed,
            shuffle=True,
            stratify=holdout_joint,
        )
        frame["data_split"] = ""
        frame.loc[train_indices, "data_split"] = "train"
        frame.loc[validation_indices, "data_split"] = "validation"
        frame.loc[test_indices, "data_split"] = "test"
        if (frame["data_split"] == "").any():
            raise RuntimeError("MGTAB split generation left nodes unassigned")
        return frame

    @staticmethod
    def _coalesce_relation(
        source: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        node_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coalesce pairs while proving that duplicate base weights agree."""
        if source.numel() == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long)
            empty_float = torch.empty((0,), dtype=torch.float32)
            return empty_index, empty_float, empty_float, empty_float
        keys = source.to(torch.int64) * node_count + target.to(torch.int64)
        order = torch.argsort(keys, stable=True)
        sorted_keys = keys[order]
        sorted_weight = weight[order].to(torch.float32)
        unique_keys, counts = torch.unique_consecutive(
            sorted_keys, return_counts=True
        )
        starts = torch.cumsum(counts, dim=0) - counts
        base_weight = sorted_weight[starts]
        expanded_base = torch.repeat_interleave(base_weight, counts)
        if not torch.allclose(sorted_weight, expanded_base, rtol=1e-6, atol=1e-7):
            raise ValueError(
                "duplicate MGTAB edges have inconsistent base weights; "
                "coalescing would be ambiguous"
            )
        edge_index = torch.stack(
            [unique_keys // node_count, unique_keys % node_count], dim=0
        ).to(torch.long)
        multiplicity = counts.to(torch.float32)
        synthetic_reverse = torch.zeros_like(multiplicity, dtype=torch.bool)
        return edge_index, base_weight, multiplicity, synthetic_reverse

    def _prepare_relation(
        self,
        relation_name: str,
        source: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        node_count: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], torch.Tensor]:
        source = source.to(torch.long)
        target = target.to(torch.long)
        weight = weight.to(torch.float32)
        raw_rows = int(source.numel())
        self_mask = source == target
        self_counts = torch.zeros(node_count, dtype=torch.float32)
        if torch.any(self_mask):
            self_counts.index_add_(
                0, source[self_mask], torch.ones(int(self_mask.sum()))
            )
        removed_self_loop_rows = int(self_mask.sum())
        if self.config.exclude_self_loops_from_message_passing:
            keep = ~self_mask
            source, target, weight = source[keep], target[keep], weight[keep]

        if (
            self.config.multiedge_policy == "coalesce_with_count"
            and relation_name in {"reply", "quoted"}
        ):
            edge_index, base_weight, multiplicity, synthetic_reverse = (
                self._coalesce_relation(source, target, weight, node_count)
            )
        else:
            edge_index = torch.stack([source, target], dim=0)
            base_weight = weight
            multiplicity = torch.ones(source.numel(), dtype=torch.float32)
            synthetic_reverse = torch.zeros(source.numel(), dtype=torch.bool)

        forward_edge_count = int(edge_index.shape[1])
        if (
            self.config.add_reverse_undirected_edges
            and relation_name in UNDIRECTED_RELATIONS
            and forward_edge_count
        ):
            reverse_index = edge_index.flip(0)
            edge_index = torch.cat([edge_index, reverse_index], dim=1)
            base_weight = torch.cat([base_weight, base_weight], dim=0)
            multiplicity = torch.cat([multiplicity, multiplicity], dim=0)
            synthetic_reverse = torch.cat([
                synthetic_reverse,
                torch.ones(forward_edge_count, dtype=torch.bool),
            ])

        edge_data = {
            "edge_index": edge_index.contiguous(),
            "base_weight": base_weight.contiguous(),
            "edge_weight": base_weight.contiguous(),
            "multiplicity": multiplicity.contiguous(),
            "temporal_sync": torch.zeros(edge_index.shape[1], dtype=torch.float32),
            "temporal_recency": torch.zeros(edge_index.shape[1], dtype=torch.float32),
            "temporal_available": torch.zeros(edge_index.shape[1], dtype=torch.bool),
            "is_synthetic_reverse": synthetic_reverse.contiguous(),
        }
        relation_manifest = {
            "raw_edge_rows": raw_rows,
            "removed_self_loop_rows": removed_self_loop_rows,
            "forward_edges_after_policy": forward_edge_count,
            "model_edge_count": int(edge_index.shape[1]),
            "added_reverse_edges": int(synthetic_reverse.sum()),
            "multiplicity_sum": int(multiplicity[~synthetic_reverse].sum()),
            "base_weight_min": float(base_weight.min()) if base_weight.numel() else None,
            "base_weight_max": float(base_weight.max()) if base_weight.numel() else None,
        }
        return edge_data, relation_manifest, self_counts

    def load(self):
        """Return ``(graph, labels_frame, adapter_manifest)``."""
        from torch_geometric.data import HeteroData

        bundle = self._load_bundle()
        features = bundle["features.pt"].to(torch.float32)
        labels_bot = bundle["labels_bot.pt"].to(torch.int64)
        labels_stance = bundle["labels_stance.pt"].to(torch.int64)
        edge_index = bundle["edge_index.pt"].to(torch.int64)
        edge_type = bundle["edge_type.pt"].to(torch.int64)
        edge_weight = bundle["edge_weight.pt"].to(torch.float32)
        node_count = int(features.shape[0])

        tweet_embedding_available = torch.any(features[:, 20:] != 0, dim=1)
        model_features = torch.cat([
            features,
            tweet_embedding_available.to(torch.float32).unsqueeze(1),
        ], dim=1)

        graph = HeteroData()
        graph["user"].x = model_features
        graph["user"].node_ids = [
            f"mgtab:{index}" for index in range(node_count)
        ]
        graph["user"].tweet_embedding_available = tweet_embedding_available
        graph["user"].feature_source = (
            "mgtab_standard:20_profile+768_mean_labse+1_availability"
        )

        relation_manifest: Dict[str, Dict[str, Any]] = {}
        self_interaction_counts = torch.zeros(
            (node_count, len(RELATION_TYPES)), dtype=torch.float32
        )
        for relation_id, relation_name in RELATION_TYPES.items():
            mask = edge_type == relation_id
            edge_data, relation_record, self_counts = self._prepare_relation(
                relation_name,
                edge_index[0, mask],
                edge_index[1, mask],
                edge_weight[mask],
                node_count,
            )
            store = graph[("user", relation_name, "user")]
            for name, value in edge_data.items():
                setattr(store, name, value)
            relation_manifest[relation_name] = relation_record
            self_interaction_counts[:, relation_id] = self_counts

        graph["user"].self_interaction_counts = self_interaction_counts
        graph["user"].self_interaction_relation_order = [
            RELATION_TYPES[index] for index in sorted(RELATION_TYPES)
        ]
        graph.dataset_name = "mgtab"
        graph.dataset_kind = "mgtab_standard"
        graph.dataset_capabilities = {
            "bot_labels": True,
            "stance_labels": True,
            "multi_relation_graph": True,
            "precomputed_text_embeddings": True,
            "raw_text": False,
            "timestamps": False,
            "original_user_ids": False,
            "campaign_labels": False,
            "ground_truth_roles": False,
            "external_neighbors": False,
        }

        labels_frame = self._make_labels(labels_bot, labels_stance)
        split_counts = {
            key: int(value)
            for key, value in labels_frame["data_split"].value_counts().items()
        }
        manifest: Dict[str, Any] = {
            "schema_version": "hyperdecept.mgtab-adapter.v1",
            "dataset_name": "mgtab",
            "release": "standard",
            "source_path": str(self.root.resolve()),
            "node_count": node_count,
            "feature_dim_original": int(features.shape[1]),
            "feature_dim_model": int(model_features.shape[1]),
            "tweet_embedding_missing_count": int(
                (~tweet_embedding_available).sum()
            ),
            "node_id_policy": "anonymous_index_prefixed_mgtab",
            "original_ids_available": False,
            "raw_text_available": False,
            "timestamps_available": False,
            "split_strategy": "stratified_bot_x_stance_transductive_node_classification",
            "split_seed": self.config.split_seed,
            "split_ratios": {
                "train": self.config.split_ratios[0],
                "validation": self.config.split_ratios[1],
                "test": self.config.split_ratios[2],
            },
            "split_counts": split_counts,
            "split_hash": _split_digest(labels_frame),
            "multiedge_policy": self.config.multiedge_policy,
            "undirected_policy": (
                "explicit_reverse_edges"
                if self.config.add_reverse_undirected_edges else "released_direction_only"
            ),
            "self_loop_policy": (
                "exclude_from_message_passing_keep_sidecar_counts"
                if self.config.exclude_self_loops_from_message_passing
                else "preserve_in_message_passing"
            ),
            "edge_attribute_contract": [
                "base_weight",
                "multiplicity",
                "temporal_sync",
                "temporal_recency",
                "temporal_available",
                "is_synthetic_reverse",
            ],
            "relations": relation_manifest,
            "warnings": [
                "MGTAB standard contains anonymous node indices, not original Twitter IDs.",
                "Stance is retained for stratification and evaluation; it is not a tactical role label.",
                "Edge attributes are preserved by the adapter but require explicit model support to affect attention.",
            ],
        }
        graph.adapter_manifest = manifest
        return graph, labels_frame, manifest


__all__ = [
    "MGTABAdapter",
    "MGTABAdapterConfig",
    "VALID_MULTIEDGE_POLICIES",
    "write_split_csv",
]
