"""Run the auditable P2 multi-source Lorentz-HGT training protocol.

The runner consumes only a DatasetPlan and the artifact paths declared by its
episode manifests. It keeps real node splits separate from synthetic episode
splits and writes one checkpoint/metrics bundle for a reproducible fold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
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
    training_protocol_assignments,
)
from joint_training import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    DomainAlternatingTrainer,
    DomainAwareLorentzHGT,
    EpisodeBatch,
    JointLossConfig,
    evaluate_bot_batch,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _action_vocabulary(plan: DatasetPlan) -> dict[str, int]:
    """Build one deterministic action vocabulary from declared artifacts."""
    import pandas as pd

    actions: set[str] = set()
    for episode in plan.episodes:
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
    role_vocabulary: Mapping[str, int],
    action_vocabulary: Mapping[str, int],
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    similarity_threshold: float = 0.7,
) -> list[EpisodeBatch]:
    """Load batches assigned to one protocol split without directory scanning."""
    batches: list[EpisodeBatch] = []
    for episode in plan.episodes:
        assignment = assignments[episode.episode_id]
        if episode.domain == "real":
            # ``shared`` real episodes expose their declared train/val/test
            # node partitions; synthetic assignments refer to whole episodes.
            if assignment != "shared":
                continue
            node_split = split
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
        )
        if torch.any(batch.bot_mask):
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
    rows = [evaluate_bot_batch(model, batch, device=device) for batch in batches]
    return _mean_metrics(rows)


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
) -> dict:
    """Train one declared P2 fold and persist auditable outputs."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
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
    action_vocabulary = _action_vocabulary(plan)
    train_real = load_protocol_batches(
        plan, assignments, split="train", role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
    )
    train_synthetic = load_protocol_batches(
        plan, assignments, split="train", role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
    )
    validation = load_protocol_batches(
        plan, assignments, split="validation", role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
    )
    test = load_protocol_batches(
        plan, assignments, split="test", role_vocabulary=ROLE_VOCABULARY,
        action_vocabulary=action_vocabulary,
        similarity_threshold=similarity_threshold,
    )
    all_batches = [*train_real, *train_synthetic, *validation, *test]
    metadata = merge_heterogeneous_metadata(batch.graph for batch in all_batches)
    dataset_domains = tuple(sorted({batch.dataset_name for batch in all_batches}))
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
    ).to(torch_device)
    for batch in all_batches:
        model(batch.graph, domain=batch.domain, dataset_name=batch.dataset_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = DomainAlternatingTrainer(
        model,
        optimizer,
        config=JointLossConfig(),
        device=torch_device,
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(
            train_real, train_synthetic, max_steps=max_steps
        )
        validation_metrics = _evaluate(model, validation, torch_device)
        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        })
    test_metrics = _evaluate(model, test, torch_device)
    checkpoint = save_joint_checkpoint(
        output_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        loss_config=trainer.config,
        epoch=trainer.epoch,
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
        "device": str(torch_device),
        "dataset_domains": list(dataset_domains),
        "action_vocabulary": action_vocabulary,
        "train_real_batches": len(train_real),
        "train_synthetic_batches": len(train_synthetic),
        "validation_batches": len(validation),
        "test_batches": len(test),
        "history": history,
        "test": test_metrics,
        "checkpoint": str(checkpoint),
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
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan.write(output_dir / "data_plan.json")
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
