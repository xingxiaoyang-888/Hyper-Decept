"""Run one auditable P2 forward/backward/checkpoint smoke step.

This is an interface test, not an experiment runner and not a source of paper
metrics. It performs one TwiBot/simulation step and one MGTAB/simulation step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from data_processing.p2_smoke_report import (  # noqa: E402
    build_p2_smoke_report,
    collect_git_provenance,
    collect_runtime_environment,
    write_p2_smoke_report,
)
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
    return EpisodeManifest.read(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twibot-manifest", required=True, type=Path)
    parser.add_argument("--mgtab-manifest", required=True, type=Path)
    parser.add_argument("--simulation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/p2_smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    provenance = collect_git_provenance(ROOT)
    twibot_path = args.twibot_manifest.expanduser().resolve()
    mgtab_path = args.mgtab_manifest.expanduser().resolve()
    simulation_path = args.simulation_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    twibot_manifest = _read_manifest(twibot_path)
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

    twibot_train = load_episode_batch_from_manifest(
        twibot_manifest,
        node_split="train",
    )
    twibot_validation = load_episode_batch_from_manifest(
        twibot_manifest,
        node_split="validation",
    )
    mgtab_train = load_episode_batch_from_manifest(
        mgtab_manifest,
        node_split="train",
    )
    mgtab_validation = load_episode_batch_from_manifest(
        mgtab_manifest,
        node_split="validation",
    )
    synthetic_batch = load_episode_batch_from_manifest(
        simulation_manifest,
        role_vocabulary=role_vocabulary,
        action_vocabulary=action_vocabulary,
        similarity_threshold=0.8,
    )
    metadata = merge_heterogeneous_metadata([
        twibot_train.graph, mgtab_train.graph, synthetic_batch.graph
    ])
    model = DomainAwareLorentzHGT(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        metadata=metadata,
        num_roles=len(role_vocabulary),
        num_temporal_actions=len(action_vocabulary),
        campaign_dim=8,
        dataset_domains=("twibot22", "mgtab", "deeppersona_oasis"),
        dropout=0.0,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    twibot_train.to(device)
    mgtab_train.to(device)
    synthetic_batch.to(device)
    # Materialize every lazy dataset adapter before constructing the optimizer.
    model(twibot_train.graph, domain="real", dataset_name="twibot22")
    model(mgtab_train.graph, domain="real", dataset_name="mgtab")
    model(
        synthetic_batch.graph,
        domain="synthetic",
        dataset_name="deeppersona_oasis",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_config = JointLossConfig(alignment_method="coral")
    trainer = DomainAlternatingTrainer(
        model, optimizer, config=loss_config, device=device
    )
    twibot_train_metrics = trainer.train_step(twibot_train, synthetic_batch)
    mgtab_train_metrics = trainer.train_step(mgtab_train, synthetic_batch)
    validation_metrics = {
        "twibot22": evaluate_bot_batch(
            model, twibot_validation, device=args.device
        ),
        "mgtab": evaluate_bot_batch(
            model, mgtab_validation, device=args.device
        ),
    }
    checkpoint_path = save_joint_checkpoint(
        output_dir / "p2_smoke_checkpoint.pt",
        model=model,
        optimizer=optimizer,
        loss_config=loss_config,
        epoch=1,
        plan_id="p2_smoke_data_2026_07_29",
        fold_id="twibot_mgtab_plus_simulation_smoke",
        metrics={
            "twibot_loss_total": twibot_train_metrics["loss_total"],
            "mgtab_loss_total": mgtab_train_metrics["loss_total"],
        },
    )
    summary = {
        "status": "passed",
        "warning": "Smoke/interface metrics are not paper results.",
        "twibot_manifest": str(twibot_path),
        "mgtab_manifest": str(mgtab_path),
        "simulation_manifest": str(simulation_path),
        "twibot_users": int(twibot_train.graph["user"].num_nodes),
        "twibot_train_labels": int(twibot_train.bot_mask.sum()),
        "twibot_validation_labels": int(twibot_validation.bot_mask.sum()),
        "mgtab_users": int(mgtab_train.graph["user"].num_nodes),
        "mgtab_train_labels": int(mgtab_train.bot_mask.sum()),
        "mgtab_validation_labels": int(mgtab_validation.bot_mask.sum()),
        "synthetic_users": int(synthetic_batch.graph["user"].num_nodes),
        "synthetic_bot_labels": int(synthetic_batch.bot_mask.sum()),
        "synthetic_role_labels": int(synthetic_batch.role_mask.sum()),
        "synthetic_campaign_labels": int(synthetic_batch.campaign_mask.sum()),
        "synthetic_action_labels": int(synthetic_batch.temporal_action_mask.sum()),
        "action_vocabulary": action_vocabulary,
        "train_metrics_smoke_only": {
            "twibot22": twibot_train_metrics,
            "mgtab": mgtab_train_metrics,
        },
        "validation_metrics_smoke_only": validation_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }
    summary_path = output_dir / "p2_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    finished_at = datetime.now(timezone.utc)
    report = build_p2_smoke_report(
        summary=summary,
        manifest_paths=(twibot_path, mgtab_path, simulation_path),
        started_at=started_at,
        finished_at=finished_at,
        provenance=provenance,
        environment=collect_runtime_environment(device),
    )
    report_paths = write_p2_smoke_report(
        report, args.report_dir.expanduser().resolve()
    )
    summary["portable_audit_reports"] = [str(path) for path in report_paths]
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
