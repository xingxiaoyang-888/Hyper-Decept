"""Train HyperTrace-Base on audited DeepPersona/OASIS episodes only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from itertools import cycle
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (str(ROOT), str(CHARACTER_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from data_processing.episode_manifest import (  # noqa: E402
    DatasetPlan,
    audit_episode_splits,
    audit_plan_artifacts,
    leave_one_scenario_out_assignments,
)
from data_processing.feature_contracts import (  # noqa: E402
    FEATURE_CONTRACTS,
    resolve_feature_contract,
)
from joint_training import (  # noqa: E402
    DomainAwareLorentzHGT,
    EpisodeBatch,
    JointLossConfig,
    compute_episode_losses,
    evaluate_coordination_batch,
    load_episode_batch_from_manifest,
    merge_heterogeneous_metadata,
)
from scripts.base_models import DomainAwareEuclideanHGT, DomainAwareMLP  # noqa: E402
from scripts.lorentz_warm_start import (  # noqa: E402
    apply_real_schema_warm_start,
    apply_uk_global_warm_start,
)


MODEL_VARIANTS = ("lorentz_hgt", "euclidean_hgt", "mlp")
GRAPH_INTERVENTIONS = (
    "original",
    "degree_preserving_edge_shuffle",
    "remove_temporal_information",
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write checkpoints atomically so interruption cannot corrupt ``path``."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        for index, value in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(), index)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _intervention_seed(episode_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{episode_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _apply_graph_intervention(batch: EpisodeBatch, intervention: str, seed: int) -> None:
    if intervention == "original":
        return
    if intervention == "remove_temporal_information":
        batch.graph = batch.graph.clone()
        for _, store in batch.graph.edge_items():
            for name in ("temporal_sync", "temporal_recency", "temporal_available"):
                value = store.get(name)
                if torch.is_tensor(value):
                    store[name] = torch.zeros_like(value)
        return
    if intervention != "degree_preserving_edge_shuffle":
        raise ValueError(f"unsupported graph intervention: {intervention}")
    batch.graph = batch.graph.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_intervention_seed(batch.episode_id, seed))
    for edge_type in batch.graph.edge_types:
        edge_index = batch.graph[edge_type].edge_index
        if edge_index.shape[1] < 2:
            continue
        permutation = torch.randperm(edge_index.shape[1], generator=generator)
        shuffled = edge_index.clone()
        shuffled[1] = edge_index[1, permutation]
        batch.graph[edge_type].edge_index = shuffled
        store = batch.graph[edge_type]
        for name, value in list(store.items()):
            if name == "edge_index" or not torch.is_tensor(value):
                continue
            if value.ndim > 0 and value.shape[0] == edge_index.shape[1]:
                store[name] = value[permutation.to(value.device)]


def _batches(
    plan,
    assignments,
    split,
    *,
    cache_dir,
    feature_columns,
    graph_intervention,
    intervention_seed,
):
    result = []
    for episode in plan.episodes:
        if episode.domain != "synthetic" or assignments[episode.episode_id] != split:
            continue
        batch = load_episode_batch_from_manifest(
            episode,
            feature_columns=feature_columns,
            role_vocabulary=None,
            action_vocabulary=None,
            graph_cache_dir=cache_dir,
        )
        if torch.any(batch.coordination_mask):
            _apply_graph_intervention(
                batch, graph_intervention, intervention_seed
            )
            result.append(batch)
    if not result:
        raise ValueError(f"no synthetic batches for split={split}")
    return result


def _mean(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def _positive_class_weight(batches: list[EpisodeBatch]) -> float:
    positives = 0
    negatives = 0
    for batch in batches:
        targets = batch.coordination_targets[batch.coordination_mask]
        positives += int(torch.count_nonzero(targets == 1).item())
        negatives += int(torch.count_nonzero(targets == 0).item())
    if positives == 0 or negatives == 0:
        raise ValueError("Base training requires both coordination classes")
    return negatives / positives


def _build_model(variant: str, metadata):
    common = {
        "hidden_dim": 64,
        "num_layers": 2,
        "metadata": metadata,
        "dataset_domains": ("deeppersona_oasis",),
        "dropout": 0.1,
    }
    if variant == "lorentz_hgt":
        return DomainAwareLorentzHGT(
            **common,
            num_heads=4,
            num_roles=1,
            num_temporal_actions=1,
            campaign_dim=64,
            enable_privileged_heads=False,
        )
    if variant == "euclidean_hgt":
        return DomainAwareEuclideanHGT(**common, num_heads=4)
    if variant == "mlp":
        return DomainAwareMLP(**common)
    raise ValueError(f"unknown model variant {variant!r}")


def run_training(*, plan: DatasetPlan, output_dir: Path, held_out_scenario: str,
                 seed: int, epochs: int, max_steps: int | None, device: str,
                 graph_cache_dir: Path, uk_init: Path | None = None,
                 real_init: Path | None = None,
                 resume: bool = False, feature_contract: str = "observable18",
                 model_variant: str = "lorentz_hgt",
                 graph_intervention: str = "original") -> dict:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}")
    if graph_intervention not in GRAPH_INTERVENTIONS:
        raise ValueError(
            f"graph_intervention must be one of {GRAPH_INTERVENTIONS}"
        )
    if (uk_init is not None or real_init is not None) and model_variant != "lorentz_hgt":
        raise ValueError("Lorentz warm-starts are valid only for lorentz_hgt")
    if uk_init is not None and real_init is not None:
        raise ValueError("choose at most one of --uk-init and --real-init")
    feature_columns = resolve_feature_contract(feature_contract)
    run_contract = {
        "schema_version": "hypertrace.base-run-contract.v1",
        "plan_id": plan.plan_id,
        "held_out_scenario": held_out_scenario,
        "seed": int(seed),
        "feature_contract": feature_contract,
        "feature_columns": list(feature_columns),
        "model_variant": model_variant,
        "graph_intervention": graph_intervention,
        "real_init": str(real_init.resolve()) if real_init is not None else None,
    }
    _seed(seed)
    assignments = leave_one_scenario_out_assignments(plan, held_out_scenario)
    audit_episode_splits(plan, assignments).raise_for_errors()
    audit_plan_artifacts(plan, require_files=True).raise_for_errors()
    batch_options = {
        "cache_dir": graph_cache_dir,
        "feature_columns": feature_columns,
        "graph_intervention": graph_intervention,
        "intervention_seed": seed,
    }
    train = _batches(plan, assignments, "train", **batch_options)
    validation = _batches(plan, assignments, "validation", **batch_options)
    test = _batches(plan, assignments, "test", **batch_options)
    metadata = merge_heterogeneous_metadata(batch.graph for batch in train)
    model = _build_model(model_variant, metadata)
    for batch in train:
        for node_type, features in batch.graph.x_dict.items():
            model.feature_adapters[batch.dataset_name][node_type](features[:1].float())
    target_device = torch.device(device)
    model.to(target_device)
    # Materialize PyG ``Linear(-1, ...)`` parameters before any schema-aware
    # warm-start transfer.  The real-operation checkpoint can only be applied
    # to initialized tensors; this dry forward is label-free and uses one
    # training graph solely to establish the target node/relation shapes.
    with torch.no_grad():
        model(
            train[0].graph.to(target_device),
            domain="synthetic",
            dataset_name=train[0].dataset_name,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Base is a coordination detector, not a privileged multi-task learner.
    # Synthetic role/campaign/action fields remain available for offline
    # evaluation, but they must not influence detector gradients.
    positive_class_weight = _positive_class_weight(train)
    config = JointLossConfig(
        privileged_weight=0.0,
        alignment_weight=0.0,
        synthetic_positive_class_weight=positive_class_weight,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "last_checkpoint.pt"
    start_epoch = 0
    history = []
    best_validation_auprc = float("-inf")
    best_epoch = 0
    best_model_state = None
    warm_start_report = {
        "schema_version": "hypertrace.warm-start.none.v1",
        "enabled": False,
        "transferred_keys": [],
    }
    if uk_init is not None and not resume:
        warm_start_report = apply_uk_global_warm_start(model.encoder, uk_init)
        warm_start_report["enabled"] = True
    if real_init is not None and not resume:
        warm_start_report = apply_real_schema_warm_start(model.encoder, real_init)
        warm_start_report["enabled"] = True
    if resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=target_device, weights_only=False)
        if state.get("run_contract") != run_contract:
            raise ValueError("Base checkpoint run contract does not match request")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))
        best_validation_auprc = float(state.get("best_validation_auprc", float("-inf")))
        best_epoch = int(state.get("best_epoch", 0))
        best_model_state = state.get("best_model")
        warm_start_report = dict(state.get("warm_start_report") or warm_start_report)
        _restore_rng_state(state.get("rng_state"))
    for epoch in range(start_epoch, epochs):
        model.train()
        totals = []
        batches = cycle(train)
        steps = min(max_steps, len(train)) if max_steps is not None else len(train)
        for _ in range(steps):
            batch = next(batches).to(target_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.graph, domain="synthetic", dataset_name=batch.dataset_name)
            losses = compute_episode_losses(model, output, batch, config, epoch=epoch)
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            totals.append({"loss_total": float(losses["total"].detach().cpu()), "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu())})
        validation_metrics = _mean([
            evaluate_coordination_batch(
                model, batch.to(target_device), device=target_device
            )
            for batch in validation
        ])
        history.append({"epoch": epoch + 1, "train": _mean(totals), "validation": validation_metrics})
        current_auprc = validation_metrics.get("auprc", float("nan"))
        if np.isfinite(current_auprc) and current_auprc > best_validation_auprc:
            best_validation_auprc = float(current_auprc)
            best_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())
        _atomic_torch_save(
            {"schema_version": "hypertrace.base-resume.v2", "plan_id": plan.plan_id,
             "held_out_scenario": held_out_scenario, "epoch": epoch + 1,
             "run_contract": run_contract,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "warm_start_report": warm_start_report,
             "best_validation_auprc": best_validation_auprc,
             "best_epoch": best_epoch, "best_model": best_model_state,
             "rng_state": _rng_state(), "history": history},
            checkpoint,
        )
        if best_model_state is not None:
            _atomic_torch_save(
                {"schema_version": "hypertrace.base-best.v2", "plan_id": plan.plan_id,
                 "held_out_scenario": held_out_scenario, "epoch": best_epoch,
                 "run_contract": run_contract,
                 "model": best_model_state, "validation_auprc": best_validation_auprc,
                 "warm_start_report": warm_start_report},
                output_dir / "best_checkpoint.pt",
            )
    if best_model_state is None:
        best_model_state = deepcopy(model.state_dict())
        best_epoch = start_epoch
    model.load_state_dict(best_model_state)
    test_metrics = _mean([
        evaluate_coordination_batch(
            model, batch.to(target_device), device=target_device
        )
        for batch in test
    ])
    result = {"schema_version": "hypertrace.base-training.v2", "status": "passed",
              "plan_id": plan.plan_id, "held_out_scenario": held_out_scenario,
              "seed": seed, "epochs": epochs, "resumed": start_epoch > 0,
              "run_contract": run_contract,
              "model_variant": model_variant,
              "geometry": model.geometry_metadata(),
              "feature_contract": feature_contract,
              "feature_columns": list(feature_columns),
              "graph_intervention": graph_intervention,
              "label_semantics": "generated_adversarial_account_membership",
              "training_objective": "class_balanced_coordination_only",
              "privileged_heads_used": False,
              "positive_class_weight": positive_class_weight,
              "best_epoch": best_epoch,
              "best_validation_auprc": best_validation_auprc,
              "warm_start": warm_start_report,
              "train_batches": len(train), "validation_batches": len(validation),
              "test_batches": len(test), "action_vocabulary": {},
              "history": history, "test": test_metrics,
              "checkpoint": str(checkpoint.resolve())}
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--held-out-scenario", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limit optimizer steps per epoch for preflight; omit for all train batches.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-cache-dir", required=True, type=Path)
    parser.add_argument("--uk-init", type=Path, default=None)
    parser.add_argument(
        "--real-init", type=Path, default=None,
        help="Schema-validated, label-free real-operation Lorentz warm-start checkpoint.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--feature-contract",
        choices=sorted(FEATURE_CONTRACTS),
        default="observable18",
    )
    parser.add_argument(
        "--model-variant", choices=MODEL_VARIANTS, default="lorentz_hgt"
    )
    parser.add_argument(
        "--graph-intervention",
        choices=GRAPH_INTERVENTIONS,
        default="original",
    )
    parser.add_argument(
        "--intraop-threads", type=int, default=4,
        help="PyTorch CPU threads per training worker.",
    )
    parser.add_argument(
        "--interop-threads", type=int, default=1,
        help="PyTorch inter-op threads per training worker.",
    )
    args = parser.parse_args()
    if args.intraop_threads < 1 or args.interop_threads < 1:
        parser.error("thread counts must be positive integers")
    torch.set_num_threads(args.intraop_threads)
    torch.set_num_interop_threads(args.interop_threads)
    plan = DatasetPlan.read(args.plan)
    values = vars(args).copy()
    values.pop("plan", None)
    values.pop("intraop_threads", None)
    values.pop("interop_threads", None)
    print(json.dumps(run_training(**values, plan=plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
