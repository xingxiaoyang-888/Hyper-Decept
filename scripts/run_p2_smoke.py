"""Run one auditable P2 forward/backward/checkpoint smoke step.

This is an interface test, not an experiment runner and not a source of paper
metrics.  It deliberately performs exactly one paired MGTAB/simulation update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (str(ROOT), str(CHARACTER_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from data_processing.episode_manifest import EpisodeManifest  # noqa: E402
from joint_training import (  # noqa: E402
    DomainAlternatingTrainer,
    DomainAwareLorentzHGT,
    JointLossConfig,
    evaluate_bot_batch,
    load_episode_batch_from_manifest,
    merge_heterogeneous_metadata,
    save_joint_checkpoint,
)


def _read_manifest(path: Path) -> EpisodeManifest:
    return EpisodeManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgtab-manifest", required=True, type=Path)
    parser.add_argument("--simulation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mgtab_path = args.mgtab_manifest.expanduser().resolve()
    simulation_path = args.simulation_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mgtab_manifest = _read_manifest(mgtab_path)
    simulation_manifest = _read_manifest(simulation_path)

    event_targets = pd.read_csv(
        simulation_manifest.artifacts["event_targets_csv"], low_memory=False
    )
    actions = sorted(
        value for value in event_targets.get("next_action", pd.Series(dtype=str))
        .dropna().astype(str).unique() if value.strip()
    )
    if not actions:
        actions = ["unavailable"]
    action_vocabulary = {value: index for index, value in enumerate(actions)}
    role_vocabulary = {
        "organic": 0,
        "independent_adversary": 1,
        "leader": 2,
        "member": 3,
    }

    real_batch = load_episode_batch_from_manifest(
        mgtab_manifest,
        node_split="train",
    )
    synthetic_batch = load_episode_batch_from_manifest(
        simulation_manifest,
        role_vocabulary=role_vocabulary,
        action_vocabulary=action_vocabulary,
        similarity_threshold=0.8,
    )
    metadata = merge_heterogeneous_metadata([
        real_batch.graph, synthetic_batch.graph
    ])
    model = DomainAwareLorentzHGT(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        metadata=metadata,
        num_roles=len(role_vocabulary),
        num_temporal_actions=len(action_vocabulary),
        campaign_dim=8,
        dataset_domains=("mgtab", "deeppersona_oasis"),
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_config = JointLossConfig(alignment_method="coral")
    trainer = DomainAlternatingTrainer(
        model, optimizer, config=loss_config, device=args.device
    )
    train_metrics = trainer.train_step(real_batch, synthetic_batch)
    validation_metrics = evaluate_bot_batch(
        model, real_batch, device=args.device
    )
    checkpoint_path = save_joint_checkpoint(
        output_dir / "p2_smoke_checkpoint.pt",
        model=model,
        optimizer=optimizer,
        loss_config=loss_config,
        epoch=1,
        plan_id="p2_smoke_data_2026_07_29",
        fold_id="mgtab_train_plus_simulation_smoke",
        metrics=train_metrics,
    )
    summary = {
        "status": "passed",
        "warning": "Smoke/interface metrics are not paper results.",
        "mgtab_manifest": str(mgtab_path),
        "simulation_manifest": str(simulation_path),
        "real_users": int(real_batch.graph["user"].num_nodes),
        "real_train_labels": int(real_batch.bot_mask.sum()),
        "synthetic_users": int(synthetic_batch.graph["user"].num_nodes),
        "synthetic_bot_labels": int(synthetic_batch.bot_mask.sum()),
        "synthetic_role_labels": int(synthetic_batch.role_mask.sum()),
        "synthetic_campaign_labels": int(synthetic_batch.campaign_mask.sum()),
        "synthetic_action_labels": int(synthetic_batch.temporal_action_mask.sum()),
        "action_vocabulary": action_vocabulary,
        "train_metrics_smoke_only": train_metrics,
        "validation_metrics_smoke_only": validation_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }
    summary_path = output_dir / "p2_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
