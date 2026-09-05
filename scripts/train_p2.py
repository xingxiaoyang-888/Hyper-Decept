"""Run the auditable P2 multi-source Lorentz-HGT training protocol.

The runner consumes only a DatasetPlan and the artifact paths declared by its
episode manifests. It keeps real node splits separate from synthetic episode
splits and writes one checkpoint/metrics bundle for a reproducible fold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (str(ROOT), str(CHARACTER_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from data_processing.episode_manifest import (  # noqa: E402
    DatasetPlan,
    EpisodeManifest,
    audit_episode_splits,
    audit_plan_artifacts,
    training_protocol_assignments,
)
from joint_training import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    DomainAlternatingTrainer,
    DomainAwareLorentzHGT,
    EpisodeBatch,
    JointLossConfig,
    evaluate_coordination_batch,
    load_episode_batch_from_manifest,
    merge_heterogeneous_metadata,
    save_joint_checkpoint,
)


ROLE_VOCABULARY = {
    "organic": 0,
    "independent_adversary": 1,
    "leader": 2,
    "member": 3,
}


def set_seed(seed: int) -> None:
    """Set all local RNGs used by the runner and model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Keep repeated paper runs comparable across CPU/GPU workers. The caller
    # can still choose a different seed for an independent replicate.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _action_vocabulary(
    plan: DatasetPlan,
    assignments: Mapping[str, str],
) -> dict[str, int]:
    """Build action classes from training episodes only.

    Validation/test action values remain masked when they are unseen, avoiding
    using held-out labels to determine the model output space.
    """
    import pandas as pd

    actions: set[str] = set()
    for episode in plan.episodes:
        if episode.domain != "synthetic" or assignments[episode.episode_id] != "train":
            continue
        path = episode.artifacts.get("event_targets_csv")
        if not path:
            continue
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(
                f"event_targets_csv does not exist for {episode.episode_id}: {path}"
            )
        columns = pd.read_csv(target, nrows=0).columns
        if "next_action" not in columns:
            continue
        frame = pd.read_csv(target, usecols=["next_action"])
        if "next_action" in frame.columns:
            actions.update(
                value.strip()
                for value in frame["next_action"].dropna().astype(str)
                if value.strip()
            )
    return {value: index for index, value in enumerate(sorted(actions))}


def _split_for_episode(episode: EpisodeManifest, assignment: str) -> str | None:
    """Map plan assignment to a node split only for real datasets."""
    if episode.domain == "real":
        if assignment in {"shared", "train", "validation", "test"}:
            return "train" if assignment == "shared" else assignment
        return None
    return None


def load_protocol_batches(
    plan: DatasetPlan,
    assignments: Mapping[str, str],
    *,
    split: str,
    domain: str | None = None,
    role_vocabulary: Mapping[str, int],
    action_vocabulary: Mapping[str, int],
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    similarity_threshold: float = 0.7,
    graph_cache_dir: Path | None = None,
) -> list[EpisodeBatch]:
    """Load batches assigned to one protocol split without directory scanning."""
    if domain not in {None, "real", "synthetic"}:
        raise ValueError("domain must be None, real, or synthetic")
    batches: list[EpisodeBatch] = []
    for episode in plan.episodes:
        if domain is not None and episode.domain != domain:
            continue
        assignment = assignments[episode.episode_id]
        if episode.domain == "real":
            # Shared real datasets expose their declared node partitions.
            # External real datasets are assigned to test as whole datasets,
            # but still use their declared test node partition when present.
            if assignment not in {"shared", "test"}:
                continue
            if assignment == "test" and split != "test":
                continue
            node_split = "test" if assignment == "test" else split
        elif assignment != split:
            continue
        else:
            node_split = None
        batch = load_episode_batch_from_manifest(
            episode,
            feature_columns=feature_columns,
            role_vocabulary=role_vocabulary,
            action_vocabulary=action_vocabulary,
            node_split=node_split,
            similarity_threshold=similarity_threshold,
            graph_cache_dir=graph_cache_dir,
        )
        if torch.any(batch.coordination_mask):
            batches.append(batch)
    if not batches:
        raise ValueError(f"no labelled batches available for protocol split: {split}")
    return batches


def _mean_metrics(metrics: Iterable[Mapping[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in metrics:
        for key, value in row.items():
            value = float(value)
            if np.isfinite(value):
                totals[key] += value
                counts[key] += 1
    return {
        key: totals[key] / counts[key]
        for key in sorted(totals)
        if counts[key]
    }


def _evaluate(
    model: DomainAwareLorentzHGT,
    batches: Sequence[EpisodeBatch],
    device: torch.device,
) -> dict[str, float]:
    rows = [
        evaluate_coordination_batch(model, batch, device=device)
        for batch in batches
    ]
    return _mean_metrics(rows)


def _evaluate_details(
    model: DomainAwareLorentzHGT,
    batches: Sequence[EpisodeBatch],
    device: torch.device,
) -> dict:
    """Return macro, sample-weighted, and per-episode evaluation metrics."""
    by_episode = {
        batch.episode_id: evaluate_coordination_batch(model, batch, device=device)
        for batch in batches
    }
    macro = _mean_metrics(by_episode.values())
    weighted_totals: dict[str, float] = defaultdict(float)
    weighted_counts: dict[str, float] = defaultdict(float)
    for row in by_episode.values():
        weight = float(row.get("count", 0.0))
        for key, value in row.items():
            if key == "count":
                continue
            value = float(value)
            if key == "count" or not np.isfinite(value) or weight <= 0:
                continue
            weighted_totals[key] += value * weight
            weighted_counts[key] += weight
    weighted = {
        key: weighted_totals[key] / weighted_counts[key]
        for key in sorted(weighted_totals)
        if weighted_counts[key]
    }
    return {
        "macro": macro,
        "sample_weighted_episode_mean": weighted,
        "by_episode": by_episode,
    }


def _run_contract(
    *,
    plan_id: str,
    plan_digest: str,
    protocol_id: str,
    held_out_scenario: str,
    seed: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    similarity_threshold: float,
    max_steps: int | None,
    loss_config: JointLossConfig,
) -> tuple[dict, str]:
    contract = {
        "schema_version": "hyperdecept.p2-resume-contract.v1",
        "plan_id": str(plan_id),
        "plan_digest": str(plan_digest),
        "protocol_id": str(protocol_id),
        "held_out_scenario": str(held_out_scenario),
        "seed": int(seed),
        "hidden_dim": int(hidden_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "similarity_threshold": float(similarity_threshold),
        "max_steps": max_steps,
        "loss_config": asdict(loss_config),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return contract, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_torch_save(value: object, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


def _atomic_json_write(value: Mapping[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(value: Mapping[str, object]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and value.get("torch_cuda"):
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _trusted_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_resume_checkpoint(
    target: Path,
    *,
    model: DomainAwareLorentzHGT,
    optimizer: torch.optim.Optimizer,
    trainer: DomainAlternatingTrainer,
    run_contract: Mapping[str, object],
    run_signature: str,
    completed_epochs: int,
    active_epoch: int | None,
    completed_steps_in_epoch: int,
    epoch_totals: Mapping[str, float],
    history: Sequence[Mapping[str, object]],
    best_epoch: int,
    best_validation_score: float,
    best_model_state,
    best_optimizer_state,
) -> None:
    payload = {
        "schema_version": "hyperdecept.p2-resume-checkpoint.v1",
        "run_contract": dict(run_contract),
        "run_signature": run_signature,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "trainer_epoch": int(trainer.epoch),
        "completed_epochs": int(completed_epochs),
        "active_epoch": active_epoch,
        "completed_steps_in_epoch": int(completed_steps_in_epoch),
        "epoch_totals": dict(epoch_totals),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_validation_score),
        "best_model_state": best_model_state,
        "best_optimizer_state": best_optimizer_state,
        "rng_state": _rng_state(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_torch_save(payload, target)
    _atomic_json_write({
        "schema_version": "hyperdecept.p2-training-progress.v1",
        "status": "running",
        "run_signature": run_signature,
        "completed_epochs": int(completed_epochs),
        "active_epoch": active_epoch,
        "completed_steps_in_epoch": int(completed_steps_in_epoch),
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_validation_score),
        "checkpoint": str(target),
        "saved_at_utc": payload["saved_at_utc"],
    }, target.parent / "progress.json")


def _load_resume_checkpoint(
    target: Path,
    *,
    expected_signature: str,
) -> dict:
    if not target.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {target}")
    payload = _trusted_torch_load(target)
    if payload.get("schema_version") != "hyperdecept.p2-resume-checkpoint.v1":
        raise ValueError("unsupported P2 resume checkpoint schema")
    if payload.get("run_signature") != expected_signature:
        raise ValueError(
            "resume checkpoint configuration does not match the requested run"
        )
    return payload


def run_training(
    *,
    plan: DatasetPlan,
    protocol_id: str,
    held_out_scenario: str,
    output_dir: Path,
    seed: int = 7,
    epochs: int = 10,
    device: str = "cpu",
    hidden_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    similarity_threshold: float = 0.7,
    max_steps: int | None = None,
    graph_cache_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 1,
) -> dict:
    """Train one declared P2 fold and persist auditable outputs."""
    if protocol_id != "P2_multisource_real":
        raise ValueError(
            "train_p2 currently implements P2_multisource_real only; "
            "P1 external holdout requires an explicit external-adapter "
            "pretraining protocol and must not use randomly initialized "
            "test-only input adapters"
        )
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_dir = (
        graph_cache_dir.expanduser().resolve()
        if graph_cache_dir is not None
        else output_dir.parent / "graph_cache"
    )
    started_at = datetime.now(timezone.utc)
    set_seed(seed)
    torch_device = torch.device(device)
    assignments = training_protocol_assignments(
        plan,
        protocol_id=protocol_id,
        held_out_scenario=held_out_scenario,
    )
    split_audit = audit_episode_splits(plan, assignments)
    split_audit.raise_for_errors()
    artifact_audit = audit_plan_artifacts(plan, require_files=True)
    artifact_audit.raise_for_errors()
    action_vocabulary = _action_vocabulary(plan, assignments)
    train_real = load_protocol_batches(
        plan, assignments, split="train", domain="real",
        role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
        graph_cache_dir=graph_cache_dir,
    )
    train_synthetic = load_protocol_batches(
        plan, assignments, split="train", domain="synthetic",
        role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
        graph_cache_dir=graph_cache_dir,
    )
    validation = load_protocol_batches(
        plan, assignments, split="validation", role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
        graph_cache_dir=graph_cache_dir,
    )
    test = load_protocol_batches(
        plan, assignments, split="test", role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
        graph_cache_dir=graph_cache_dir,
    )
    all_batches = [*train_real, *train_synthetic, *validation, *test]
    # Build the architecture from training graphs only. Test-only edge/node
    # types must not alter the model after the fold is fixed.
    metadata = merge_heterogeneous_metadata(
        batch.graph for batch in [*train_real, *train_synthetic]
    )
    train_dataset_names = {
        batch.dataset_name for batch in [*train_real, *train_synthetic]
    }
    unseen_eval_datasets = sorted({
        batch.dataset_name for batch in [*validation, *test]
        if batch.dataset_name not in train_dataset_names
    })
    if unseen_eval_datasets:
        raise ValueError(
            "evaluation contains datasets without a trained input adapter: "
            f"{unseen_eval_datasets}; add an explicit adapter-training protocol "
            "before reporting external-dataset metrics"
        )
    train_node_types, train_edge_types = metadata
    train_node_types = set(train_node_types)
    train_edge_types = {tuple(value) for value in train_edge_types}
    for batch in [*validation, *test]:
        node_types, edge_types = batch.graph.metadata()
        if not set(node_types).issubset(train_node_types) or not {
            tuple(value) for value in edge_types
        }.issubset(train_edge_types):
            raise ValueError(
                f"evaluation graph schema is not covered by training schema: "
                f"{batch.episode_id}"
            )
    dataset_domains = tuple(sorted(train_dataset_names))
    model = DomainAwareLorentzHGT(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        metadata=metadata,
        num_roles=len(ROLE_VOCABULARY),
        num_temporal_actions=max(1, len(action_vocabulary)),
        campaign_dim=hidden_dim,
        dataset_domains=dataset_domains,
        dropout=dropout,
    )
    # Materialize only the lazy input adapters from one feature row.  A full
    # CPU model forward over every training graph is both unnecessary and
    # prohibitively expensive for the cached TwiBot graph.
    for batch in [*train_real, *train_synthetic]:
        adapters = model.feature_adapters[batch.dataset_name]
        for node_type, features in batch.graph.x_dict.items():
            if features.shape[0] == 0:
                raise ValueError(
                    f"cannot initialize {batch.dataset_name}/{node_type} "
                    "adapter from an empty node table"
                )
            adapters[node_type](features[:1].float())
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_config = JointLossConfig()
    trainer = DomainAlternatingTrainer(
        model,
        optimizer,
        config=loss_config,
        device=torch_device,
    )
    run_contract, run_signature = _run_contract(
        plan_id=plan.plan_id,
        plan_digest=hashlib.sha256(
            json.dumps(
                plan.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        protocol_id=protocol_id,
        held_out_scenario=held_out_scenario,
        seed=seed,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        similarity_threshold=similarity_threshold,
        max_steps=max_steps,
        loss_config=loss_config,
    )
    resume_path = output_dir / "last_checkpoint.pt"
    history: list[dict] = []
    best_epoch = 0
    best_validation_score = float("-inf")
    best_model_state = None
    best_optimizer_state = None
    completed_epochs = 0
    active_epoch = None
    completed_steps_in_epoch = 0
    epoch_totals: dict[str, float] = {}
    if resume:
        resume_state = _load_resume_checkpoint(
            resume_path,
            expected_signature=run_signature,
        )
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        trainer.epoch = int(resume_state["trainer_epoch"])
        completed_epochs = int(resume_state["completed_epochs"])
        active_epoch = resume_state.get("active_epoch")
        completed_steps_in_epoch = int(
            resume_state.get("completed_steps_in_epoch", 0)
        )
        epoch_totals = {
            str(name): float(value)
            for name, value in dict(resume_state.get("epoch_totals") or {}).items()
        }
        history = list(resume_state.get("history") or [])
        best_epoch = int(resume_state.get("best_epoch", 0))
        best_validation_score = float(
            resume_state.get("best_validation_score", float("-inf"))
        )
        best_model_state = resume_state.get("best_model_state")
        best_optimizer_state = resume_state.get("best_optimizer_state")
        _restore_rng_state(resume_state["rng_state"])
    if completed_epochs > epochs:
        raise ValueError("resume checkpoint is beyond the requested epoch count")

    for epoch in range(completed_epochs, epochs):
        resume_this_epoch = active_epoch == epoch
        start_step = completed_steps_in_epoch if resume_this_epoch else 0
        running_totals = epoch_totals if resume_this_epoch else {}

        def save_step_checkpoint(
            completed_steps: int,
            total_steps: int,
            totals: Mapping[str, float],
        ) -> None:
            if (
                completed_steps % checkpoint_every_steps != 0
                and completed_steps != total_steps
            ):
                return
            _save_resume_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                trainer=trainer,
                run_contract=run_contract,
                run_signature=run_signature,
                completed_epochs=epoch,
                active_epoch=epoch,
                completed_steps_in_epoch=completed_steps,
                epoch_totals=totals,
                history=history,
                best_epoch=best_epoch,
                best_validation_score=best_validation_score,
                best_model_state=best_model_state,
                best_optimizer_state=best_optimizer_state,
            )

        train_metrics = trainer.train_epoch(
            train_real,
            train_synthetic,
            max_steps=max_steps,
            start_step=start_step,
            initial_totals=running_totals,
            step_callback=save_step_checkpoint,
        )
        validation_details = _evaluate_details(model, validation, torch_device)
        validation_metrics = validation_details["macro"]
        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_weighted": validation_details["sample_weighted_episode_mean"],
            "validation_by_episode": validation_details["by_episode"],
        })
        validation_score = validation_metrics.get("auprc", float("nan"))
        if not np.isfinite(validation_score):
            validation_score = validation_metrics.get("balanced_accuracy", float("nan"))
        if np.isfinite(validation_score) and validation_score > best_validation_score:
            best_epoch = epoch + 1
            best_validation_score = float(validation_score)
            best_model_state = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
        completed_epochs = epoch + 1
        active_epoch = None
        completed_steps_in_epoch = 0
        epoch_totals = {}
        _save_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            run_contract=run_contract,
            run_signature=run_signature,
            completed_epochs=completed_epochs,
            active_epoch=None,
            completed_steps_in_epoch=0,
            epoch_totals={},
            history=history,
            best_epoch=best_epoch,
            best_validation_score=best_validation_score,
            best_model_state=best_model_state,
            best_optimizer_state=best_optimizer_state,
        )
    if best_model_state is None:
        raise RuntimeError("validation produced no finite model-selection metric")
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    test_details = _evaluate_details(model, test, torch_device)
    test_metrics = test_details["macro"]
    checkpoint = save_joint_checkpoint(
        output_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        loss_config=trainer.config,
        epoch=best_epoch,
        plan_id=plan.plan_id,
        fold_id=held_out_scenario,
        metrics=test_metrics,
    )
    result = {
        "schema_version": "hyperdecept.p2-training-run.v1",
        "status": "passed",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan.plan_id,
        "protocol_id": protocol_id,
        "held_out_scenario": held_out_scenario,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selection_metric": "validation_auprc_or_balanced_accuracy",
        "best_validation_score": best_validation_score,
        "device": str(torch_device),
        "resumed": bool(resume),
        "run_signature": run_signature,
        "graph_cache_dir": str(graph_cache_dir),
        "dataset_domains": list(dataset_domains),
        "train_dataset_domains": sorted(train_dataset_names),
        "action_vocabulary": action_vocabulary,
        "train_real_batches": len(train_real),
        "train_synthetic_batches": len(train_synthetic),
        "validation_batches": len(validation),
        "test_batches": len(test),
        "history": history,
        "test": test_metrics,
        "test_weighted": test_details["sample_weighted_episode_mean"],
        "test_by_episode": test_details["by_episode"],
        "checkpoint": str(checkpoint),
        "resume_checkpoint": str(resume_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(json.dumps({
        "protocol_id": protocol_id,
        "held_out_scenario": held_out_scenario,
        "seed": seed,
        "epochs": epochs,
        "device": str(torch_device),
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "graph_cache_dir": str(graph_cache_dir),
        "checkpoint_every_steps": checkpoint_every_steps,
        "run_signature": run_signature,
        "loss_config": asdict(trainer.config),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan.write(output_dir / "data_plan.json")
    _atomic_json_write({
        "schema_version": "hyperdecept.p2-training-progress.v1",
        "status": "completed",
        "run_signature": run_signature,
        "completed_epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_score": best_validation_score,
        "checkpoint": str(checkpoint),
        "resume_checkpoint": str(resume_path),
        "finished_at_utc": result["finished_at_utc"],
    }, output_dir / "progress.json")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", default="P2_multisource_real")
    parser.add_argument("--held-out-scenario", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--graph-cache-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from OUTPUT_DIR/last_checkpoint.pt",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = DatasetPlan.read(args.plan)
    result = run_training(
        plan=plan,
        protocol_id=args.protocol,
        held_out_scenario=args.held_out_scenario,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        similarity_threshold=args.similarity_threshold,
        max_steps=args.max_steps,
        graph_cache_dir=args.graph_cache_dir,
        resume=args.resume,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
