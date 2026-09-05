"""Run a frozen HyperTrace Base checkpoint on an audited real-operation bundle.

This is an external, read-only evaluation.  The official bad/good cohorts are
reported as information-operation membership; no checkpoint update, threshold
fit, or synthetic label reinterpretation occurs here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (str(ROOT), str(CHARACTER_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from joint_training import (  # noqa: E402
    DomainAwareLorentzHGT,
    evaluate_coordination_batch,
)
from scripts.external_bundle_episode import load_external_bundle_episode  # noqa: E402
from scripts.frozen_manifest import resolve_frozen_checkpoint  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_edge_types(state: dict) -> tuple[tuple[str, str, str], ...]:
    found = set()
    for key in state:
        match = re.match(r"encoder\.layers\.\d+\.relations\.([^.]*)\.", key)
        if not match:
            continue
        encoded = match.group(1)
        parts = encoded.split("__")
        if len(parts) != 3:
            raise ValueError(f"invalid encoded checkpoint relation: {encoded}")
        found.add(tuple(parts))
    if not found:
        raise ValueError("checkpoint contains no Lorentz relation modules")
    return tuple(sorted(found))


def _build_model(checkpoint: dict, metadata):
    contract = checkpoint.get("run_contract") or {}
    if contract.get("feature_contract") != "observable18":
        raise ValueError("frozen external runner requires observable18 checkpoint")
    if contract.get("model_variant") != "lorentz_hgt":
        raise ValueError("frozen external runner requires Lorentz-HGT checkpoint")
    model = DomainAwareLorentzHGT(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        metadata=metadata,
        num_roles=1,
        num_temporal_actions=1,
        campaign_dim=64,
        dataset_domains=("deeppersona_oasis", "honduras_uae_io"),
        dropout=0.1,
        enable_privileged_heads=False,
    )
    # Initialize lazy Linear(-1, ...) adapters for both node types before
    # loading the checkpoint.  The external adapter has the same observable18
    # user width and the observed 14-dimensional tweet feature contract.
    for dataset, dims in {
        "deeppersona_oasis": {"user": 18, "tweet": 14},
        "honduras_uae_io": {"user": 18, "tweet": 14},
    }.items():
        for node_type, width in dims.items():
            model.feature_adapters[dataset][node_type](
                torch.zeros((1, width), dtype=torch.float32)
            )
    state = checkpoint["model"]
    # The external bundle uses the same observable18/observed-tweet contracts
    # as the frozen synthetic input.  Reuse the trained input adapters rather
    # than introducing an untrained random domain adapter at test time.
    for node_type in ("user", "tweet"):
        for name in ("0.weight", "0.bias", "1.weight", "1.bias", "2.weight", "2.bias"):
            source = f"feature_adapters.deeppersona_oasis.{node_type}.{name}"
            target = f"feature_adapters.honduras_uae_io.{node_type}.{name}"
            if source in state:
                state[target] = state[source].clone()
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("feature_adapters.honduras_uae_io.")]
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected[:5]}")
    # The external adapter is intentionally initialized independently; all
    # Lorentz encoder and synthetic adapter parameters come from the frozen
    # Base checkpoint.
    missing = [key for key in missing if not key.startswith("feature_adapters.honduras_uae_io.")]
    if missing:
        raise ValueError(f"missing frozen checkpoint keys: {missing[:5]}")
    return model


def _warm_start_provenance(
    checkpoint_path: Path,
    checkpoint: dict,
    freeze_manifest_path: Path | None,
) -> tuple[dict, str, str | None]:
    checkpoint_warm_start = dict(checkpoint.get("warm_start_report") or {})
    checkpoint_source = checkpoint_warm_start.get("source_operation")
    if freeze_manifest_path is None:
        return checkpoint_warm_start, "checkpoint", None

    freeze_manifest_path = Path(freeze_manifest_path)
    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    matches = [
        entry for entry in freeze.get("checkpoints", [])
        if resolve_frozen_checkpoint(freeze_manifest_path, entry)
        == checkpoint_path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(
            "checkpoint must have exactly one entry in the freeze manifest"
        )
    entry = matches[0]
    if _sha256(checkpoint_path) != entry.get("sha256"):
        raise ValueError("checkpoint hash does not match freeze manifest")
    manifest_warm_start = dict(entry.get("warm_start") or {})
    manifest_source = manifest_warm_start.get("source_operation")
    if checkpoint_source and manifest_source and checkpoint_source != manifest_source:
        raise ValueError("checkpoint and freeze manifest source operations disagree")
    merged = {**manifest_warm_start, **checkpoint_warm_start}
    source = checkpoint_source or manifest_source
    if source is not None:
        merged["source_operation"] = source
    provenance = "checkpoint" if checkpoint_source else "freeze_manifest"
    return merged, provenance, _sha256(freeze_manifest_path)


def run(
    checkpoint_path: Path,
    bundle_path: Path,
    output_path: Path,
    device: str,
    freeze_manifest_path: Path | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    allowed_schemas = {
        "hypertrace.base-best.v2",
        "hypertrace.base-resume.v2",
    }
    if checkpoint.get("schema_version") not in allowed_schemas:
        raise ValueError("checkpoint is not a supported frozen base checkpoint")
    edges = _checkpoint_edge_types(checkpoint["model"])
    batch = load_external_bundle_episode(bundle_path)
    # Use the checkpoint's exact relation vocabulary.  The adapter supplies
    # empty stores for relations absent from the external source.
    metadata = (tuple(sorted(batch.graph.node_types)), edges)
    model = _build_model(checkpoint, metadata)
    target_device = torch.device(device)
    model.to(target_device)
    metrics = evaluate_coordination_batch(
        model, batch.to(target_device), device=target_device
    )
    warm_start, source_provenance, freeze_manifest_sha256 = (
        _warm_start_provenance(
            checkpoint_path,
            checkpoint,
            freeze_manifest_path,
        )
    )
    source_operation = warm_start.get("source_operation")
    target_operation = str(batch.episode_id)
    operation_disjoint = bool(source_operation and source_operation != target_operation)
    result = {
        "schema_version": "hypertrace.operation-disjoint-external-io.v1",
        "status": "passed",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "bundle": str(bundle_path.resolve()),
        "bundle_manifest_sha256": _sha256(bundle_path / "manifest.json"),
        "model_selection": "lorentz_warm_started_observable18",
        "geometry": "intrinsic_lorentz",
        "feature_contract": "observable18_partial",
        "evaluation_scope": "real_information_operation_external_evaluation",
        "label_semantics": "information_operation_membership",
        "not_generic_bot_metrics": True,
        "not_campaign_level_cib_metrics": True,
        "threshold": 0.5,
        "metrics": metrics,
        "graph_node_types": list(batch.graph.node_types),
        "graph_edge_types": [list(edge) for edge in batch.graph.edge_types],
        "unsupported_source_relations": list(getattr(batch.graph, "unsupported_source_relations", [])),
        "training_labels_consumed": False,
        "checkpoint_updated": False,
        "checkpoint_schema": checkpoint.get("schema_version"),
        "warm_start": warm_start,
        "source_operation": source_operation,
        "source_operation_provenance": source_provenance,
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "target_operation": target_operation,
        "operation_disjoint": operation_disjoint,
        "evaluation_protocol": (
            "strict_unseen_operation"
            if operation_disjoint
            else "same_operation_unlabeled_warm_start"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-manifest", type=Path, default=None)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.bundle,
        args.output,
        args.device,
        freeze_manifest_path=args.freeze_manifest,
    )


if __name__ == "__main__":
    main()
