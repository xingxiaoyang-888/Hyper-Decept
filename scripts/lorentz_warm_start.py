"""Schema-validated warm-start utilities for HyperTrace Base training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import hashlib

import torch


TRANSFER_SCHEMA = "hypertrace.uk-global-warm-start.v1"
ALLOWED_KEYS = ("common_curvature.raw", "input_scale_raw")
REAL_TRANSFER_SCHEMA = "hypertrace.real-operation-warm-start.v1"
REAL_PRETRAIN_SCHEMA = "hypertrace.real-operation-pretrain.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_uk_global_warm_start(encoder: torch.nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    """Transfer only schema-agnostic Lorentz scalars from a UK checkpoint.

    Relation-specific message-passing weights are intentionally excluded because
    UK2019 has a single user-user relation while synthetic episodes are typed
    user/tweet graphs. The returned report is stored with every run.
    """
    source_path = Path(checkpoint).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source_path, map_location="cpu")
    if payload.get("schema_version") != "hypertrace.lorentz-pretrain.v1":
        raise ValueError("unsupported UK Lorentz checkpoint schema")
    source = payload.get("encoder")
    if not isinstance(source, Mapping):
        raise ValueError("UK checkpoint does not contain encoder state")
    transferred = []
    for key in ALLOWED_KEYS:
        if key not in source:
            raise ValueError(f"UK checkpoint missing transferable key: {key}")
        value = source[key]
        target = encoder.common_curvature.raw if key == "common_curvature.raw" else encoder.input_scale_raw
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(value.shape)} != {tuple(target.shape)}")
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        transferred.append(key)
    return {
        "schema_version": TRANSFER_SCHEMA,
        "source_checkpoint": str(source_path),
        "source_schema": payload["schema_version"],
        "transferred_keys": transferred,
        "excluded_relation_specific_parameters": True,
        "excluded_reason": "UK2019 user-coordination-user schema differs from synthetic user/tweet typed graph",
    }


def _edge_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


def _real_transfer_allowed(
    key: str,
    *,
    common_node_types: set[str],
    common_edge_keys: set[str],
) -> bool:
    if key in ALLOWED_KEYS:
        return True
    if key.startswith("input_projectors."):
        parts = key.split(".")
        return len(parts) > 1 and parts[1] in common_node_types
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "layers":
        return False
    family, name = parts[2], parts[3]
    if family in {"relations", "relation_gates"}:
        return name in common_edge_keys
    if family in {"fusion_gates", "norms"}:
        return name in common_node_types
    return False


def apply_real_schema_warm_start(
    encoder: torch.nn.Module,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Transfer only trained, schema-aligned real-operation encoder weights.

    The checkpoint must declare which node and edge types were actually
    observed during self-supervision. Merely instantiated but unobserved
    relation modules are excluded from transfer.
    """
    source_path = Path(checkpoint).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source_path, map_location="cpu")
    if payload.get("schema_version") != REAL_PRETRAIN_SCHEMA:
        raise ValueError("unsupported real-operation checkpoint schema")
    source = payload.get("encoder")
    if not isinstance(source, Mapping):
        raise ValueError("real-operation checkpoint does not contain encoder state")
    observed_nodes = {str(value) for value in payload.get("observed_node_types", [])}
    observed_edges = {
        tuple(map(str, value)) for value in payload.get("observed_edge_types", [])
    }
    if not observed_nodes or not observed_edges:
        raise ValueError("checkpoint must declare observed node and edge types")
    target_nodes = set(map(str, encoder.metadata[0]))
    target_edges = {tuple(map(str, value)) for value in encoder.metadata[1]}
    common_nodes = observed_nodes & target_nodes
    common_edges = observed_edges & target_edges
    if not common_nodes or not common_edges:
        raise ValueError("real and target graph schemas have no transferable overlap")
    common_edge_keys = {_edge_key(value) for value in common_edges}
    target_state = encoder.state_dict()
    transferred = []
    excluded_shape = []
    for key, target in target_state.items():
        if not _real_transfer_allowed(
            key,
            common_node_types=common_nodes,
            common_edge_keys=common_edge_keys,
        ):
            continue
        value = source.get(key)
        if value is None:
            continue
        if tuple(value.shape) != tuple(target.shape):
            excluded_shape.append(key)
            continue
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        transferred.append(key)
    if not transferred:
        raise ValueError("no schema-aligned parameters were transferred")
    return {
        "schema_version": REAL_TRANSFER_SCHEMA,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_schema": payload["schema_version"],
        "source_operation": payload.get("source_operation"),
        "source_cutoff": payload.get("source_cutoff"),
        "transferred_keys": transferred,
        "transferred_parameter_tensors": len(transferred),
        "common_node_types": sorted(common_nodes),
        "common_edge_types": [list(value) for value in sorted(common_edges)],
        "excluded_shape_mismatch": excluded_shape,
        "excluded_unobserved_or_incompatible_parameters": True,
    }
