"""Domain-aware episodic joint training for HyperDecept.

This module reuses :class:`IntrinsicLorentzHGT` as the shared encoder while
keeping real-world bot supervision separate from privileged simulation labels.
It deliberately does not concatenate real and generated rows into one table:
each graph remains an auditable episode and the optimizer alternates domains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from lorentz_hgt import (
    IntrinsicLorentzHGT,
    LorentzPrototypeClassifier,
    logmap0,
    lorentz_distance,
)


EdgeType = Tuple[str, str, str]
VALID_DOMAINS = {"real", "synthetic"}
DEFAULT_FEATURE_COLUMNS = (
    "Semantic_0", "Semantic_1", "Semantic_2", "Semantic_3",
    "Semantic_4", "Semantic_5", "Semantic_6", "Semantic_7",
    "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
    "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
    "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
    "Empathy_Gap_Mean", "Empathy_Gap_Max", "Dark_Triad_Mean",
    "Dark_Triad_Max", "Contagion_Mean", "Contagion_Max",
    "Volatility_Mean", "Volatility_Max",
)


def merge_heterogeneous_metadata(graphs: Iterable[object]):
    """Return deterministic union metadata for independently built episodes."""
    node_types = set()
    edge_types = set()
    found = False
    for graph in graphs:
        found = True
        graph_nodes, graph_edges = graph.metadata()
        node_types.update(graph_nodes)
        edge_types.update(tuple(edge_type) for edge_type in graph_edges)
    if not found or "user" not in node_types:
        raise ValueError("at least one graph with a user node type is required")
    return tuple(sorted(node_types)), tuple(sorted(edge_types))


@dataclass
class EpisodeBatch:
    """One graph episode plus node-aligned, availability-masked targets."""

    episode_id: str
    domain: str
    graph: object
    bot_targets: torch.Tensor
    bot_mask: torch.Tensor
    role_targets: torch.Tensor
    role_mask: torch.Tensor
    campaign_targets: torch.Tensor
    campaign_mask: torch.Tensor
    temporal_action_targets: torch.Tensor
    temporal_action_mask: torch.Tensor
    dataset_name: str = "unspecified"

    def __post_init__(self) -> None:
        if self.domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}")
        if not str(self.episode_id).strip():
            raise ValueError("episode_id must not be empty")
        if not str(self.dataset_name).strip():
            raise ValueError("dataset_name must not be empty")
        if "user" not in self.graph.node_types:
            raise ValueError("episode graph must contain user nodes")
        num_users = int(self.graph["user"].num_nodes)
        tensors = {
            "bot_targets": self.bot_targets,
            "bot_mask": self.bot_mask,
            "role_targets": self.role_targets,
            "role_mask": self.role_mask,
            "campaign_targets": self.campaign_targets,
            "campaign_mask": self.campaign_mask,
            "temporal_action_targets": self.temporal_action_targets,
            "temporal_action_mask": self.temporal_action_mask,
        }
        for name, value in tensors.items():
            if value.ndim != 1 or value.shape[0] != num_users:
                raise ValueError(f"{name} must have one value per user node")
        for name in (
            "bot_mask", "role_mask", "campaign_mask", "temporal_action_mask"
        ):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        if self.domain == "real" and (
            torch.any(self.role_mask)
            or torch.any(self.campaign_mask)
            or torch.any(self.temporal_action_mask)
        ):
            raise ValueError(
                "privileged role/campaign/action targets are disabled for real episodes"
            )

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        self.graph = self.graph.to(device)
        for name in (
            "bot_targets",
            "bot_mask",
            "role_targets",
            "role_mask",
            "campaign_targets",
            "campaign_mask",
            "temporal_action_targets",
            "temporal_action_mask",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


def _normalise_identifier(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    return text


def _numeric_or_vocab(value, vocabulary: Optional[Mapping[str, int]]) -> Optional[int]:
    normalized = _normalise_identifier(value)
    if normalized is None:
        return None
    if vocabulary is not None:
        return vocabulary.get(normalized)
    try:
        number = float(normalized)
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def build_episode_batch(
    graph,
    *,
    episode_id: str,
    domain: str,
    labels_frame,
    role_vocabulary: Optional[Mapping[str, int]] = None,
    action_vocabulary: Optional[Mapping[str, int]] = None,
    user_id_column: str = "user_id",
    bot_column: str = "is_bad",
    role_column: str = "role",
    campaign_column: str = "campaign_id",
    action_column: str = "next_action",
    dataset_name: Optional[str] = None,
) -> EpisodeBatch:
    """Align a label frame to graph user order without inventing missing labels."""
    if domain not in VALID_DOMAINS:
        raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}")
    node_ids = getattr(graph["user"], "node_ids", None)
    if node_ids is None:
        raise ValueError("graph['user'].node_ids is required for target alignment")
    if user_id_column not in labels_frame.columns:
        raise ValueError(f"labels_frame is missing {user_id_column}")
    labels = labels_frame.copy()
    labels[user_id_column] = labels[user_id_column].map(_normalise_identifier)
    labels = labels.dropna(subset=[user_id_column])
    if labels[user_id_column].duplicated().any():
        duplicates = labels.loc[
            labels[user_id_column].duplicated(keep=False), user_id_column
        ].tolist()
        raise ValueError(f"duplicate label user IDs: {sorted(set(duplicates))}")
    lookup = labels.set_index(user_id_column).to_dict("index")
    num_users = len(node_ids)

    bot_targets = torch.zeros(num_users, dtype=torch.float)
    bot_mask = torch.zeros(num_users, dtype=torch.bool)
    role_targets = torch.full((num_users,), -1, dtype=torch.long)
    role_mask = torch.zeros(num_users, dtype=torch.bool)
    campaign_targets = torch.full((num_users,), -1, dtype=torch.long)
    campaign_mask = torch.zeros(num_users, dtype=torch.bool)
    temporal_targets = torch.full((num_users,), -1, dtype=torch.long)
    temporal_mask = torch.zeros(num_users, dtype=torch.bool)

    raw_campaigns = []
    for node_id in node_ids:
        row = lookup.get(_normalise_identifier(node_id))
        value = None if row is None else _normalise_identifier(row.get(campaign_column))
        if value is not None:
            raw_campaigns.append(value)
    campaign_vocabulary = {
        value: index for index, value in enumerate(sorted(set(raw_campaigns)))
    }

    for index, node_id in enumerate(node_ids):
        row = lookup.get(_normalise_identifier(node_id))
        if row is None:
            continue
        bot_value = _normalise_identifier(row.get(bot_column))
        if bot_value is not None:
            try:
                parsed_bot = float(bot_value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid binary target for user {node_id}: {bot_value}"
                ) from exc
            if parsed_bot not in {0.0, 1.0}:
                raise ValueError(f"binary target must be 0 or 1 for user {node_id}")
            bot_targets[index] = parsed_bot
            bot_mask[index] = True

        if domain == "synthetic":
            role_value = _numeric_or_vocab(row.get(role_column), role_vocabulary)
            if role_value is not None:
                role_targets[index] = role_value
                role_mask[index] = True
            campaign_value = _normalise_identifier(row.get(campaign_column))
            if campaign_value is not None:
                campaign_targets[index] = campaign_vocabulary[campaign_value]
                campaign_mask[index] = True
            action_value = _numeric_or_vocab(row.get(action_column), action_vocabulary)
            if action_value is not None:
                temporal_targets[index] = action_value
                temporal_mask[index] = True

    return EpisodeBatch(
        episode_id=episode_id,
        domain=domain,
        graph=graph,
        bot_targets=bot_targets,
        bot_mask=bot_mask,
        role_targets=role_targets,
        role_mask=role_mask,
        campaign_targets=campaign_targets,
        campaign_mask=campaign_mask,
        temporal_action_targets=temporal_targets,
        temporal_action_mask=temporal_mask,
        dataset_name=dataset_name or domain,
    )


def _normalize_label_contract(frame):
    """Normalize public dataset label names without creating new ground truth."""
    result = frame.copy()
    if "user_id" not in result.columns and "id" in result.columns:
        result = result.rename(columns={"id": "user_id"})
    if "user_id" not in result.columns:
        raise ValueError("labels artifact must contain user_id or id")
    result["user_id"] = result["user_id"].map(_normalise_identifier)
    if result["user_id"].isna().any() or result["user_id"].duplicated().any():
        raise ValueError("label user_id values must be non-null and unique")
    if "is_bad" not in result.columns:
        source = None
        for candidate in ("label", "user_type"):
            if candidate in result.columns:
                source = candidate
                break
        if source is None:
            raise ValueError("labels artifact must contain is_bad, label, or user_type")
        normalized = result[source].fillna("").astype(str).str.lower()
        known = normalized.isin({
            "0", "1", "bot", "human", "bad", "good",
            "bad_leader", "bad_member",
        }) | normalized.str.contains("bot|bad", regex=True)
        if not known.all():
            unknown = sorted(set(normalized[~known]))
            raise ValueError(f"unsupported bot labels: {unknown}")
        result["is_bad"] = normalized.str.contains("bot|bad", regex=True).astype(int)
    if "role" not in result.columns and "user_type" in result.columns:
        # DeepPersona user_type is generator provenance, so exposing its
        # explicit suffix is permitted only for synthetic EpisodeBatch objects.
        user_type = result["user_type"].fillna("").astype(str).str.lower()
        result["role"] = user_type.map({
            "bad_leader": "leader",
            "bad_member": "member",
            "good": "organic",
        })
    return result


def load_episode_batch_from_manifest(
    manifest,
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    role_vocabulary: Optional[Mapping[str, int]] = None,
    action_vocabulary: Optional[Mapping[str, int]] = None,
    node_split: Optional[str] = None,
    similarity_threshold: float = 0.7,
) -> EpisodeBatch:
    """Load one planned episode through existing feature and graph builders.

    Event-target artifacts must already represent one causal cutoff/window.  A
    many-row-per-user event file is rejected instead of silently selecting a
    future action and leaking temporal information.
    """
    import pandas as pd
    from graph_builder import build_hetero_data, load_post_embeddings

    artifacts = dict(getattr(manifest, "artifacts", {}))

    # MGTAB is an anonymous tensor bundle, not a 26-column CSV/SQLite export.
    # Dispatch before the generic path so its directory is not mistaken for a
    # raw TwiBot-22 directory by ``build_hetero_data``.
    if manifest.dataset_name == "mgtab":
        from data_processing.mgtab_adapter import MGTABAdapter

        generator_metadata = dict(
            getattr(manifest, "generator_metadata", {}) or {}
        )
        try:
            split_seed = int(generator_metadata.get("split_seed", 42))
        except (TypeError, ValueError) as exc:
            raise ValueError("MGTAB split_seed must be an integer") from exc
        multiedge_policy = generator_metadata.get(
            "multiedge_policy", "coalesce_with_count"
        )
        graph, labels, adapter_manifest = MGTABAdapter(
            manifest.source_path,
            split_seed=split_seed,
            multiedge_policy=multiedge_policy,
        ).load()

        # A persisted split takes precedence over the deterministic generated
        # split.  Require full, one-to-one coverage so a stale split cannot
        # silently move or drop anonymous nodes.
        splits_path = artifacts.get("splits_csv")
        if splits_path and Path(splits_path).exists():
            splits = pd.read_csv(splits_path, low_memory=False)
            if "user_id" not in splits.columns and "id" in splits.columns:
                splits = splits.rename(columns={"id": "user_id"})
            split_column = (
                "data_split" if "data_split" in splits.columns else "split"
            )
            if "user_id" not in splits.columns or split_column not in splits.columns:
                raise ValueError(
                    "MGTAB split artifact must contain user_id/id and "
                    "split/data_split"
                )
            splits = splits[["user_id", split_column]].copy()
            splits["user_id"] = splits["user_id"].map(_normalise_identifier)
            splits["data_split"] = (
                splits[split_column].fillna("").astype(str).str.lower()
                .replace({"val": "validation", "dev": "validation"})
            )
            if splits["user_id"].isna().any() or splits["user_id"].duplicated().any():
                raise ValueError("MGTAB split user IDs must be non-null and unique")
            if not set(splits["data_split"]).issubset(
                {"train", "validation", "test"}
            ):
                raise ValueError("MGTAB split contains an unsupported partition")
            expected_ids = set(labels["user_id"])
            split_ids = set(splits["user_id"])
            if split_ids != expected_ids:
                raise ValueError(
                    "MGTAB split must cover every anonymous node exactly once"
                )
            labels = labels.drop(columns=["data_split"]).merge(
                splits[["user_id", "data_split"]],
                on="user_id",
                how="left",
                validate="one_to_one",
            )
            adapter_manifest["split_source"] = "declared_csv"
            adapter_manifest["split_file"] = str(Path(splits_path))
            adapter_manifest["split_file_sha256"] = hashlib.sha256(
                Path(splits_path).read_bytes()
            ).hexdigest()
            adapter_manifest["split_counts"] = {
                key: int(value)
                for key, value in labels["data_split"].value_counts().items()
            }
        else:
            adapter_manifest["split_source"] = "adapter_generated"

        if node_split is not None:
            requested_split = str(node_split).lower()
            requested_split = {
                "val": "validation", "dev": "validation"
            }.get(requested_split, requested_split)
            if requested_split not in {"train", "validation", "test"}:
                raise ValueError("node_split must be train, validation, or test")
            labels = labels[labels["data_split"] == requested_split].copy()

        graph.adapter_manifest = adapter_manifest
        return build_episode_batch(
            graph,
            episode_id=manifest.episode_id,
            domain=manifest.domain,
            labels_frame=labels,
            role_vocabulary=role_vocabulary,
            action_vocabulary=action_vocabulary,
            dataset_name=manifest.dataset_name,
        )

    missing = sorted({"features_csv", "labels_csv"}.difference(artifacts))
    if missing:
        raise ValueError(f"episode manifest missing artifacts: {missing}")
    features_path = artifacts["features_csv"]
    labels_path = artifacts["labels_csv"]
    for name, path in (("source", manifest.source_path),
                       ("features_csv", features_path),
                       ("labels_csv", labels_path)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    features = pd.read_csv(features_path, low_memory=False)
    if "user_id" not in features.columns:
        raise ValueError("features artifact must contain user_id")
    features = features.copy()
    features["user_id"] = features["user_id"].map(_normalise_identifier)
    if features["user_id"].isna().any() or features["user_id"].duplicated().any():
        raise ValueError("feature user_id values must be non-null and unique")
    missing_features = sorted(set(feature_columns).difference(features.columns))
    if missing_features:
        raise ValueError(f"features artifact missing columns: {missing_features}")
    numeric_features = features[list(feature_columns)].apply(
        pd.to_numeric, errors="raise"
    ).fillna(0.0)

    labels = _normalize_label_contract(pd.read_csv(labels_path, low_memory=False))
    splits_path = artifacts.get("splits_csv")
    if splits_path:
        if not Path(splits_path).exists():
            raise FileNotFoundError(f"splits_csv does not exist: {splits_path}")
        splits = pd.read_csv(splits_path, low_memory=False)
        if "user_id" not in splits.columns and "id" in splits.columns:
            splits = splits.rename(columns={"id": "user_id"})
        split_column = "data_split" if "data_split" in splits.columns else "split"
        if "user_id" not in splits.columns or split_column not in splits.columns:
            raise ValueError("splits artifact must contain user_id/id and split/data_split")
        splits = splits[["user_id", split_column]].copy()
        splits["user_id"] = splits["user_id"].map(_normalise_identifier)
        splits["data_split"] = (
            splits[split_column].fillna("").astype(str).str.lower()
            .replace({"val": "validation", "dev": "validation"})
        )
        if splits["user_id"].isna().any() or splits["user_id"].duplicated().any():
            raise ValueError("split user IDs must be non-null and unique")
        if "data_split" in labels.columns:
            labels = labels.drop(columns=["data_split"])
        labels = labels.merge(
            splits[["user_id", "data_split"]],
            on="user_id",
            how="left",
            validate="one_to_one",
        )
    if node_split is not None:
        requested_split = str(node_split).lower()
        requested_split = {
            "val": "validation", "dev": "validation"
        }.get(requested_split, requested_split)
        if "data_split" not in labels.columns:
            raise ValueError("node_split requires a declared splits_csv artifact")
        if requested_split not in {"train", "validation", "test"}:
            raise ValueError("node_split must be train, validation, or test")
        labels = labels[labels["data_split"] == requested_split].copy()
    event_targets_path = artifacts.get("event_targets_csv")
    if event_targets_path and Path(event_targets_path).exists():
        event_targets = pd.read_csv(event_targets_path, low_memory=False)
        if "user_id" not in event_targets.columns:
            raise ValueError("event target artifact must contain user_id")
        event_targets = event_targets.copy()
        event_targets["user_id"] = event_targets["user_id"].map(
            _normalise_identifier
        )
        if event_targets["user_id"].duplicated().any():
            raise ValueError(
                "event target artifact must be pre-windowed to one row per user"
            )
        extra_columns = [
            column for column in ("campaign_id", "next_action", "attack_phase")
            if column in event_targets.columns and column not in labels.columns
        ]
        labels = labels.merge(
            event_targets[["user_id", *extra_columns]],
            on="user_id",
            how="left",
            validate="one_to_one",
        )

    post_embeddings = None
    post_embeddings_path = artifacts.get("post_embeddings")
    if post_embeddings_path:
        post_embeddings = load_post_embeddings(post_embeddings_path)
    graph, _ = build_hetero_data(
        features["user_id"].tolist(),
        numeric_features.to_numpy(dtype="float32"),
        str(manifest.source_path),
        threshold=similarity_threshold,
        post_embeddings=post_embeddings,
    )
    return build_episode_batch(
        graph,
        episode_id=manifest.episode_id,
        domain=manifest.domain,
        labels_frame=labels,
        role_vocabulary=role_vocabulary,
        action_vocabulary=action_vocabulary,
        dataset_name=manifest.dataset_name,
    )


def episode_batch_from_neighbor_sample(
    parent: EpisodeBatch,
    sampled_graph,
) -> EpisodeBatch:
    """Align parent targets to a PyG NeighborLoader subgraph.

    Only the leading seed users contribute supervised losses; sampled context
    users remain available for message passing and relation reconstruction.
    """
    n_id = getattr(sampled_graph["user"], "n_id", None)
    batch_size = getattr(sampled_graph["user"], "batch_size", None)
    if n_id is None or batch_size is None:
        raise ValueError("sampled graph must expose user.n_id and user.batch_size")
    n_id = n_id.to(dtype=torch.long, device="cpu")
    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size > n_id.numel():
        raise ValueError("invalid sampled user batch_size")
    parent_node_ids = getattr(parent.graph["user"], "node_ids", None)
    if parent_node_ids is not None:
        sampled_graph["user"].node_ids = [
            str(parent_node_ids[index]) for index in n_id.tolist()
        ]

    def slice_values(name: str, *, seed_only: bool = False) -> torch.Tensor:
        value = getattr(parent, name).detach().cpu()[n_id].clone()
        if seed_only:
            value[batch_size:] = False
        return value

    return EpisodeBatch(
        episode_id=parent.episode_id,
        domain=parent.domain,
        graph=sampled_graph,
        bot_targets=slice_values("bot_targets"),
        bot_mask=slice_values("bot_mask", seed_only=True),
        role_targets=slice_values("role_targets"),
        role_mask=slice_values("role_mask", seed_only=True),
        campaign_targets=slice_values("campaign_targets"),
        campaign_mask=slice_values("campaign_mask", seed_only=True),
        temporal_action_targets=slice_values("temporal_action_targets"),
        temporal_action_mask=slice_values(
            "temporal_action_mask", seed_only=True
        ),
        dataset_name=parent.dataset_name,
    )


class DomainAwareLorentzHGT(torch.nn.Module):
    """Shared Lorentz encoder with domain-specific inputs and task heads."""

    geometry_backend = "domain_aware_intrinsic_lorentz"

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        metadata,
        num_roles: int,
        num_temporal_actions: int,
        campaign_dim: int = 16,
        domains: Sequence[str] = ("real", "synthetic"),
        dataset_domains: Sequence[str] = (
            "twibot22", "mgtab", "simulation", "real", "synthetic",
        ),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        domain_values = tuple(domains)
        if not domain_values or not set(domain_values).issubset(VALID_DOMAINS):
            raise ValueError(f"domains must be drawn from {sorted(VALID_DOMAINS)}")
        dataset_values = tuple(dict.fromkeys(str(value) for value in dataset_domains))
        if not dataset_values or any(not value.strip() for value in dataset_values):
            raise ValueError("dataset_domains must contain non-empty identifiers")
        if num_roles <= 0 or num_temporal_actions <= 0 or campaign_dim <= 0:
            raise ValueError("task dimensions must be positive")
        from torch_geometric.nn import Linear

        node_types = tuple(metadata[0])
        self.domains = domain_values
        self.dataset_domains = dataset_values
        self.feature_adapters = torch.nn.ModuleDict({
            dataset: torch.nn.ModuleDict({
                node_type: torch.nn.Sequential(
                    Linear(-1, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                )
                for node_type in node_types
            })
            for dataset in dataset_values
        })
        self.encoder = IntrinsicLorentzHGT(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            metadata=metadata,
            dropout=dropout,
        )
        self.bot_head = LorentzPrototypeClassifier(hidden_dim, 2)
        self.role_head = LorentzPrototypeClassifier(hidden_dim, num_roles)
        self.campaign_projection = torch.nn.Linear(hidden_dim, campaign_dim)
        self.temporal_action_head = torch.nn.Linear(
            hidden_dim, num_temporal_actions
        )

    def forward(
        self,
        graph,
        *,
        domain: str,
        dataset_name: Optional[str] = None,
        edge_mask_dict: Optional[Mapping[EdgeType, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if domain not in self.domains:
            raise ValueError(f"model was not configured for domain {domain!r}")
        dataset = dataset_name or domain
        if dataset not in self.feature_adapters:
            raise ValueError(f"model was not configured for dataset {dataset!r}")
        adapted = {
            node_type: self.feature_adapters[dataset][node_type](features.float())
            for node_type, features in graph.x_dict.items()
        }
        lorentz_nodes = self.encoder.encode_lorentz_nodes(
            adapted,
            graph.edge_index_dict,
            edge_mask_dict=edge_mask_dict,
        )
        curvature = self.encoder.common_curvature()
        user_lorentz = lorentz_nodes["user"]
        user_tangent = logmap0(user_lorentz, curvature)
        bot_class_logits = self.bot_head(user_lorentz, curvature)
        result = {
            "user_lorentz": user_lorentz,
            "user_tangent": user_tangent,
            "bot_class_logits": bot_class_logits,
            "bot_logits": bot_class_logits[:, 1] - bot_class_logits[:, 0],
        }
        if domain == "synthetic":
            result.update({
                "role_logits": self.role_head(user_lorentz, curvature),
                "campaign_embedding": self.campaign_projection(user_tangent),
                "temporal_action_logits": self.temporal_action_head(user_tangent),
            })
        return result

    def geometry_metadata(self) -> Dict[str, object]:
        metadata = self.encoder.geometry_metadata()
        metadata.update({
            "geometry_backend": self.geometry_backend,
            "domains": list(self.domains),
            "dataset_domains": list(self.dataset_domains),
            "dataset_specific_input_adapters": True,
            "domain_specific_bot_heads": False,
            "decision_geometry": "lorentz_distance_prototypes",
            "bot_prototypes": self.bot_head.num_classes,
            "role_prototypes": self.role_head.num_classes,
            "privileged_simulation_heads": ["role", "campaign", "next_action"],
        })
        return metadata


@dataclass(frozen=True)
class JointLossConfig:
    detection_weight: float = 1.0
    privileged_weight: float = 0.3
    alignment_weight: float = 0.05
    privileged_role_share: float = 1.0
    privileged_campaign_share: float = 1.0
    privileged_action_share: float = 1.0
    privileged_warmup_epochs: int = 1
    privileged_decay_epochs: int = 20
    privileged_final_scale: float = 0.1
    alignment_method: str = "hyperbolic_supcon"
    alignment_temperature: float = 0.2
    campaign_temperature: float = 0.2
    gradient_clip_norm: float = 5.0
    real_positive_class_weight: Optional[float] = None
    synthetic_positive_class_weight: Optional[float] = None

    def __post_init__(self) -> None:
        values = (
            self.detection_weight,
            self.privileged_weight,
            self.alignment_weight,
            self.privileged_role_share,
            self.privileged_campaign_share,
            self.privileged_action_share,
        )
        if any(value < 0 for value in values):
            raise ValueError("loss weights must be non-negative")
        if self.campaign_temperature <= 0 or self.alignment_temperature <= 0:
            raise ValueError("alignment and campaign temperatures must be positive")
        if self.alignment_method not in {"hyperbolic_supcon", "coral"}:
            raise ValueError("alignment_method must be hyperbolic_supcon or coral")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient clip norm must be positive")
        if self.privileged_warmup_epochs < 0 or self.privileged_decay_epochs <= 0:
            raise ValueError("privileged schedule epochs must be valid")
        if not 0.0 <= self.privileged_final_scale <= 1.0:
            raise ValueError("privileged_final_scale must be in [0, 1]")
        for name, value in (
            ("real_positive_class_weight", self.real_positive_class_weight),
            ("synthetic_positive_class_weight", self.synthetic_positive_class_weight),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def privileged_scale(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch < self.privileged_warmup_epochs:
            return 1.0
        progress = min(
            1.0,
            (epoch - self.privileged_warmup_epochs)
            / self.privileged_decay_epochs,
        )
        return 1.0 - progress * (1.0 - self.privileged_final_scale)


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _campaign_loss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    valid = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if valid.numel() < 2:
        return _zero_like(embedding)
    values = F.normalize(embedding[valid], dim=-1)
    labels = targets[valid]
    left, right = torch.triu_indices(
        valid.numel(), valid.numel(), offset=1, device=embedding.device
    )
    if left.numel() == 0:
        return _zero_like(embedding)
    logits = (values[left] * values[right]).sum(dim=-1) / temperature
    pair_targets = (labels[left] == labels[right]).to(logits.dtype)
    # An episode with only one campaign or all singleton campaigns provides no
    # discriminative pairwise supervision.
    if not torch.any(pair_targets == 1) or not torch.any(pair_targets == 0):
        return _zero_like(embedding)
    return F.binary_cross_entropy_with_logits(logits, pair_targets)


def coral_alignment_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """CORAL loss in the shared origin-tangent space."""
    if first.shape[0] < 2 or second.shape[0] < 2:
        return _zero_like(first) + _zero_like(second)
    if first.shape[1] != second.shape[1]:
        raise ValueError("aligned representations must have equal dimensions")
    first_centered = first - first.mean(dim=0, keepdim=True)
    second_centered = second - second.mean(dim=0, keepdim=True)
    first_covariance = first_centered.T @ first_centered / (first.shape[0] - 1)
    second_covariance = second_centered.T @ second_centered / (second.shape[0] - 1)
    dimension = first.shape[1]
    mean_loss = F.mse_loss(first.mean(dim=0), second.mean(dim=0))
    covariance_loss = (
        (first_covariance - second_covariance).square().sum()
        / (4.0 * dimension * dimension)
    )
    return mean_loss + covariance_loss


def conditional_domain_alignment_loss(
    real_output: Mapping[str, torch.Tensor],
    real_batch: EpisodeBatch,
    synthetic_output: Mapping[str, torch.Tensor],
    synthetic_batch: EpisodeBatch,
) -> torch.Tensor:
    """Align domains within bot/human classes to avoid indiscriminate collapse."""
    losses = []
    for label in (0.0, 1.0):
        real_selection = real_batch.bot_mask & (real_batch.bot_targets == label)
        synthetic_selection = (
            synthetic_batch.bot_mask & (synthetic_batch.bot_targets == label)
        )
        if real_selection.sum() >= 2 and synthetic_selection.sum() >= 2:
            losses.append(coral_alignment_loss(
                real_output["user_tangent"][real_selection],
                synthetic_output["user_tangent"][synthetic_selection],
            ))
    if not losses:
        return _zero_like(real_output["user_tangent"]) + _zero_like(
            synthetic_output["user_tangent"]
        )
    return torch.stack(losses).mean()


def hyperbolic_supervised_alignment_loss(
    model: DomainAwareLorentzHGT,
    real_output: Mapping[str, torch.Tensor],
    real_batch: EpisodeBatch,
    synthetic_output: Mapping[str, torch.Tensor],
    synthetic_batch: EpisodeBatch,
    *,
    temperature: float,
) -> torch.Tensor:
    """Cross-domain supervised contrastive loss using Lorentz distances."""
    real_indices = torch.nonzero(real_batch.bot_mask, as_tuple=False).reshape(-1)
    synthetic_indices = torch.nonzero(
        synthetic_batch.bot_mask, as_tuple=False
    ).reshape(-1)
    if real_indices.numel() < 2 or synthetic_indices.numel() < 2:
        return _zero_like(real_output["user_lorentz"]) + _zero_like(
            synthetic_output["user_lorentz"]
        )
    real_points = real_output["user_lorentz"][real_indices]
    synthetic_points = synthetic_output["user_lorentz"][synthetic_indices]
    real_labels = real_batch.bot_targets[real_indices].to(dtype=torch.long)
    synthetic_labels = synthetic_batch.bot_targets[synthetic_indices].to(
        dtype=torch.long
    )
    curvature = model.encoder.common_curvature()
    distances = lorentz_distance(
        real_points[:, None, :],
        synthetic_points[None, :, :],
        curvature,
    ).squeeze(-1)
    logits = -distances / temperature

    def directional(
        scores: torch.Tensor,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        positive = source_labels[:, None] == target_labels[None, :]
        negative = ~positive
        valid = positive.any(dim=1) & negative.any(dim=1)
        if not torch.any(valid):
            return _zero_like(scores)
        selected_scores = scores[valid]
        selected_positive = positive[valid]
        numerator = torch.logsumexp(
            selected_scores.masked_fill(~selected_positive, float("-inf")),
            dim=1,
        )
        denominator = torch.logsumexp(selected_scores, dim=1)
        return (denominator - numerator).mean()

    forward = directional(logits, real_labels, synthetic_labels)
    backward = directional(logits.T, synthetic_labels, real_labels)
    return 0.5 * (forward + backward)


def compute_episode_losses(
    model: DomainAwareLorentzHGT,
    output: Mapping[str, torch.Tensor],
    batch: EpisodeBatch,
    config: JointLossConfig,
    *,
    epoch: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute detection and one bundled privileged-simulation objective."""
    losses: Dict[str, torch.Tensor] = {}
    reference = output["user_tangent"]
    if torch.any(batch.bot_mask):
        configured_positive_weight = (
            config.real_positive_class_weight
            if batch.domain == "real"
            else config.synthetic_positive_class_weight
        )
        positive_weight = (
            None
            if configured_positive_weight is None
            else output["bot_logits"].new_tensor(configured_positive_weight)
        )
        bot_loss = F.binary_cross_entropy_with_logits(
            output["bot_logits"][batch.bot_mask],
            batch.bot_targets[batch.bot_mask],
            pos_weight=positive_weight,
        )
    else:
        bot_loss = _zero_like(reference)
    losses["detection"] = config.detection_weight * bot_loss

    if batch.domain == "synthetic":
        privileged_parts = []
        if torch.any(batch.role_mask):
            privileged_parts.append(
                config.privileged_role_share * F.cross_entropy(
                output["role_logits"][batch.role_mask],
                batch.role_targets[batch.role_mask],
                )
            )
        if torch.any(batch.campaign_mask):
            privileged_parts.append(
                config.privileged_campaign_share * _campaign_loss(
                    output["campaign_embedding"],
                    batch.campaign_targets,
                    batch.campaign_mask,
                    config.campaign_temperature,
                )
            )
        if torch.any(batch.temporal_action_mask):
            privileged_parts.append(
                config.privileged_action_share * F.cross_entropy(
                output["temporal_action_logits"][batch.temporal_action_mask],
                batch.temporal_action_targets[batch.temporal_action_mask],
                )
            )
        if privileged_parts:
            normalized = torch.stack(privileged_parts).mean()
            losses["privileged"] = (
                config.privileged_weight
                * config.privileged_scale(epoch)
                * normalized
            )
        else:
            losses["privileged"] = _zero_like(reference)
    losses["total"] = sum(losses.values(), _zero_like(reference))
    return losses


class DomainAlternatingTrainer:
    """Train paired real/synthetic episodes with conditional domain alignment."""

    def __init__(
        self,
        model: DomainAwareLorentzHGT,
        optimizer: torch.optim.Optimizer,
        *,
        config: Optional[JointLossConfig] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config or JointLossConfig()
        self.device = torch.device(device)
        self.epoch = 0

    def train_step(
        self,
        real_batch: EpisodeBatch,
        synthetic_batch: EpisodeBatch,
    ) -> Dict[str, float]:
        if real_batch.domain != "real" or synthetic_batch.domain != "synthetic":
            raise ValueError("train_step requires one real and one synthetic episode")
        real_batch.to(self.device)
        synthetic_batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        real_output = self.model(
            real_batch.graph,
            domain="real",
            dataset_name=real_batch.dataset_name,
        )
        synthetic_output = self.model(
            synthetic_batch.graph,
            domain="synthetic",
            dataset_name=synthetic_batch.dataset_name,
        )
        real_losses = compute_episode_losses(
            self.model, real_output, real_batch, self.config, epoch=self.epoch
        )
        synthetic_losses = compute_episode_losses(
            self.model, synthetic_output, synthetic_batch, self.config,
            epoch=self.epoch,
        )
        if self.config.alignment_method == "hyperbolic_supcon":
            raw_alignment = hyperbolic_supervised_alignment_loss(
                self.model,
                real_output,
                real_batch,
                synthetic_output,
                synthetic_batch,
                temperature=self.config.alignment_temperature,
            )
        else:
            raw_alignment = conditional_domain_alignment_loss(
                real_output,
                real_batch,
                synthetic_output,
                synthetic_batch,
            )
        alignment = self.config.alignment_weight * raw_alignment
        total = real_losses["total"] + synthetic_losses["total"] + alignment
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        metrics = {
            "loss_total": float(total.detach().cpu()),
            "loss_alignment": float(alignment.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
        }
        metrics.update({
            f"loss_real_{name}": float(value.detach().cpu())
            for name, value in real_losses.items() if name != "total"
        })
        metrics.update({
            f"loss_synthetic_{name}": float(value.detach().cpu())
            for name, value in synthetic_losses.items() if name != "total"
        })
        return metrics

    def train_epoch(
        self,
        real_batches: Sequence[EpisodeBatch],
        synthetic_batches: Sequence[EpisodeBatch],
        *,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        if not real_batches or not synthetic_batches:
            raise ValueError("both real and synthetic episode sequences are required")
        steps = max(len(real_batches), len(synthetic_batches))
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError("max_steps must be positive")
            steps = min(steps, max_steps)
        real_iterator = cycle(real_batches)
        synthetic_iterator = cycle(synthetic_batches)
        totals: Dict[str, float] = {}
        for _ in range(steps):
            values = self.train_step(next(real_iterator), next(synthetic_iterator))
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
        self.epoch += 1
        return {name: value / steps for name, value in totals.items()}


@torch.no_grad()
def evaluate_bot_batch(
    model: DomainAwareLorentzHGT,
    batch: EpisodeBatch,
    *,
    device: torch.device | str = "cpu",
    calibration_bins: int = 10,
) -> Dict[str, float]:
    """Report discrimination and calibration on explicitly masked users."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    if calibration_bins <= 1:
        raise ValueError("calibration_bins must exceed one")
    batch.to(device)
    model.eval()
    output = model(
        batch.graph,
        domain=batch.domain,
        dataset_name=batch.dataset_name,
    )
    mask = batch.bot_mask
    if not torch.any(mask):
        raise ValueError("evaluation batch contains no bot labels")
    probabilities = torch.sigmoid(output["bot_logits"][mask]).cpu()
    targets = batch.bot_targets[mask].to(dtype=torch.long).cpu()
    predictions = (probabilities >= 0.5).to(dtype=torch.long)
    target_array = targets.numpy()
    probability_array = probabilities.numpy()
    prediction_array = predictions.numpy()
    both_classes = len(set(target_array.tolist())) == 2

    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, calibration_bins + 1)
    for index in range(calibration_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selection = (
            (probabilities >= lower)
            & (probabilities < upper if index + 1 < calibration_bins else probabilities <= upper)
        )
        if not torch.any(selection):
            continue
        confidence = probabilities[selection].mean()
        accuracy = targets[selection].to(probabilities.dtype).mean()
        ece += float(selection.float().mean() * torch.abs(confidence - accuracy))

    return {
        "auroc": float(roc_auc_score(target_array, probability_array))
        if both_classes else float("nan"),
        "auprc": float(average_precision_score(target_array, probability_array))
        if both_classes else float("nan"),
        "f1": float(f1_score(target_array, prediction_array, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_array, prediction_array)
        ),
        "brier": float(torch.mean((probabilities - targets.float()).square())),
        "ece": ece,
        "count": float(targets.numel()),
    }


def save_joint_checkpoint(
    path: str | Path,
    *,
    model: DomainAwareLorentzHGT,
    optimizer: torch.optim.Optimizer,
    loss_config: JointLossConfig,
    epoch: int,
    plan_id: str,
    fold_id: str,
    metrics: Optional[Mapping[str, float]] = None,
) -> Path:
    """Save model state together with the exact data-plan and geometry identity."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Dataset-specific lazy adapters that were not exercised in this fold are
    # still ``UninitializedParameter`` objects.  Serialising those objects
    # breaks PyTorch's default safe ``weights_only=True`` loader.  Omit only
    # those untouched keys and record them explicitly; every trained tensor is
    # preserved and a future loader can materialise the adapter before loading.
    from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

    state = model.state_dict()
    uninitialized_types = (UninitializedParameter, UninitializedBuffer)
    omitted_keys = sorted(
        name for name, value in state.items()
        if isinstance(value, uninitialized_types)
    )
    serializable_state = {
        name: value for name, value in state.items()
        if name not in omitted_keys
    }
    torch.save({
        "schema_version": "hyperdecept.joint-checkpoint.v1",
        "model_state": serializable_state,
        "uninitialized_model_state_keys": omitted_keys,
        "optimizer_state": optimizer.state_dict(),
        "loss_config": asdict(loss_config),
        "epoch": int(epoch),
        "plan_id": str(plan_id),
        "fold_id": str(fold_id),
        "metrics": dict(metrics or {}),
        "geometry": model.geometry_metadata(),
    }, target)
    return target


def make_user_neighbor_loader(
    graph,
    *,
    input_mask: Optional[torch.Tensor] = None,
    num_neighbors: Sequence[int] = (20, 10),
    batch_size: int = 512,
    shuffle: bool = True,
):
    """Create a heterogeneous user-seed loader for server-scale episodes."""
    from torch_geometric.loader import NeighborLoader

    if batch_size <= 0 or not num_neighbors or any(value <= 0 for value in num_neighbors):
        raise ValueError("batch_size and num_neighbors must be positive")
    if input_mask is not None:
        if input_mask.dtype != torch.bool:
            raise ValueError("input_mask must be boolean")
        if input_mask.shape != (graph["user"].num_nodes,):
            raise ValueError("input_mask must align with user nodes")
    input_nodes = "user" if input_mask is None else ("user", input_mask)
    return NeighborLoader(
        graph,
        input_nodes=input_nodes,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=shuffle,
    )
