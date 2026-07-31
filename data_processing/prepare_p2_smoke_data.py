"""Materialize auditable real-data smoke bundles for P2.

This command is deliberately limited to data preparation.  It does not train a
model and it never mutates the licensed raw datasets.  TwiBot-22 is exported
through :class:`TwiBot22RawAdapter`; MGTAB is exported through
:class:`MGTABAdapter` using its deterministic node split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_processing.mgtab_adapter import MGTABAdapter, write_split_csv
from data_processing.episode_manifest import (
    DatasetPlan,
    EpisodeManifest,
    audit_plan_artifacts,
)
from data_processing.twibot22_raw_adapter import (
    TwiBot22RawAdapter,
    load_core_ids,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_files(paths: Iterable[Path]) -> str:
    """Hash an ordered set of files, including their names and boundaries."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_frames(
    output_dir: Path,
    frames: Iterable[tuple[str, pd.DataFrame]],
    *,
    relative_paths: bool = False,
) -> dict:
    artifacts = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames:
        path = output_dir / f"{name}.csv"
        # QUOTE_ALL is deliberate.  TwiBot text contains bare carriage returns;
        # pandas' minimal quoting only considers the configured line terminator
        # and can otherwise emit a CSV that standard readers split mid-record.
        frame.to_csv(path, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
        artifacts[name] = {
            "path": path.name if relative_paths else str(path.resolve()),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return artifacts


def prepare_twibot(args: argparse.Namespace) -> Path:
    root = args.twibot_dir.expanduser().resolve()
    core_path = args.core_ids.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    core_ids = load_core_ids(str(core_path))
    if args.expected_core_count is not None and len(core_ids) != args.expected_core_count:
        raise ValueError(
            f"expected {args.expected_core_count} core IDs, found {len(core_ids)}"
        )

    bundle = TwiBot22RawAdapter(
        str(root),
        core_ids,
        edge_chunksize=args.edge_chunksize,
    ).load()
    frames = [
        ("core_users", bundle.core_users),
        ("boundary_users", bundle.boundary_users),
        ("labels", bundle.labels),
        ("follow_edges", bundle.follow_edges),
        ("actions", bundle.actions),
        ("relations", bundle.relations),
        ("posts", bundle.posts),
    ]
    formal = args.contract_mode == "formal"
    artifacts = _write_frames(output_dir, frames, relative_paths=formal)
    copied_core_ids = output_dir / "core_ids.txt"
    copied_core_ids.write_text("".join(f"{value}\n" for value in core_ids), encoding="utf-8")
    artifacts["core_ids"] = {
        "path": copied_core_ids.name if formal else str(copied_core_ids),
        "rows": len(core_ids),
        "bytes": copied_core_ids.stat().st_size,
        "sha256": _sha256(copied_core_ids),
    }

    manifest = bundle.manifest()
    manifest.update({
        "schema_version": (
            "hyperdecept.twibot22-materialized.v1"
            if formal else "hyperdecept.twibot22-smoke.v1"
        ),
        "path_contract": (
            "hyperdecept.manifest-relative.v1" if formal else "absolute.v1"
        ),
        "status": "ready",
        "source_path": "." if formal else str(root),
        "raw_source_name": root.name,
        "core_ids_source": "core_ids.txt" if formal else str(core_path),
        "core_id_count": len(core_ids),
        "core_ids_sha256": _sha256(core_path),
        "artifacts": artifacts,
        "source_files": {
            path.name: {"bytes": path.stat().st_size}
            for path in sorted(root.iterdir())
            if path.is_file()
        },
    })
    manifest_path = output_dir / "adapter_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "counts": manifest["counts"],
        "extra": manifest.get("extra", {}),
    }, ensure_ascii=False, indent=2))
    return manifest_path


def prepare_mgtab(args: argparse.Namespace) -> Path:
    root = args.mgtab_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    graph, labels, manifest = MGTABAdapter(
        root,
        split_seed=args.split_seed,
        multiedge_policy=args.multiedge_policy,
    ).load()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = write_split_csv(labels, output_dir / f"split_seed{args.split_seed}.csv")
    labels_path = output_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    manifest.update({
        "status": "ready",
        "artifacts": {
            "splits_csv": {
                "path": str(split_path.resolve()),
                "rows": int(len(labels)),
                "sha256": _sha256(split_path),
            },
            "labels_csv": {
                "path": str(labels_path.resolve()),
                "rows": int(len(labels)),
                "sha256": _sha256(labels_path),
            },
        },
        "graph": {
            "node_types": list(graph.node_types),
            "edge_types": [list(value) for value in graph.edge_types],
        },
    })
    manifest_path = output_dir / f"adapter_manifest_seed{args.split_seed}.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "node_count": manifest["node_count"],
        "feature_dim_model": manifest["feature_dim_model"],
        "split_counts": manifest["split_counts"],
    }, ensure_ascii=False, indent=2))
    return manifest_path


def prepare_twibot_features(args: argparse.Namespace) -> Path:
    """Build smoke-only 26d features from a materialized raw-adapter bundle."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    root = args.bundle_dir.expanduser().resolve()
    core = pd.read_csv(root / "core_users.csv", low_memory=False)
    actions = pd.read_csv(root / "actions.csv", low_memory=False)
    posts = pd.read_csv(root / "posts.csv", low_memory=False)
    core["user_id"] = core["user_id"].astype(str)
    if core["user_id"].duplicated().any():
        raise ValueError("core_users.csv contains duplicate user IDs")
    if len(core) != args.expected_core_count:
        raise ValueError(
            f"expected {args.expected_core_count} core users, found {len(core)}"
        )
    if not actions.empty:
        actions["actor_id"] = actions["actor_id"].astype(str)
    if not posts.empty:
        posts["author_id"] = posts["author_id"].astype(str)
        posts["content"] = posts["content"].fillna("").astype(str)

    post_text = (
        posts.groupby("author_id", sort=False)["content"]
        .apply(lambda values: "\n".join(value for value in values if value))
        .to_dict()
        if not posts.empty else {}
    )
    texts = [
        "\n".join(filter(None, [str(row.bio or ""), post_text.get(row.user_id, "")]))
        for row in core.fillna("").itertuples(index=False)
    ]
    model = SentenceTransformer(
        args.embedding_model,
        cache_folder=str(args.model_cache.expanduser().resolve()),
        local_files_only=True,
    )
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    semantic = PCA(n_components=8, random_state=42).fit_transform(embeddings)

    counts = (
        actions.groupby(["actor_id", "action_type"]).size().unstack(fill_value=0)
        if not actions.empty else pd.DataFrame()
    )
    total_actions = counts.sum(axis=1) if not counts.empty else pd.Series(dtype=float)
    created = pd.to_datetime(actions.get("event_time"), errors="coerce", utc=True)
    temporal_entropy = {}
    if not actions.empty and created is not None:
        timed = actions.assign(_hour=created.dt.hour).dropna(subset=["_hour"])
        for user_id, group in timed.groupby("actor_id"):
            probabilities = group["_hour"].value_counts(normalize=True).to_numpy()
            temporal_entropy[str(user_id)] = float(
                -sum(value * math.log(value + 1e-12) for value in probabilities)
            )

    patterns = {
        "URL_Ratio": r"https?://|www\.",
        "Mention_Ratio": r"@[A-Za-z0-9_]",
        "Hashtag_Ratio": r"#[A-Za-z0-9_]",
        "Media_Ratio": r"pic\.twitter\.com|\.(?:jpg|jpeg|png|gif|mp4)\b",
    }
    text_ratios = {name: {} for name in patterns}
    if not posts.empty:
        for user_id, group in posts.groupby("author_id"):
            denominator = max(1, len(group))
            for name, pattern in patterns.items():
                text_ratios[name][str(user_id)] = float(
                    group["content"].str.contains(pattern, case=False, regex=True).sum()
                    / denominator
                )

    feature = pd.DataFrame({
        f"Semantic_{index}": semantic[:, index] for index in range(8)
    })
    feature["Follower_Following_Ratio"] = (
        pd.to_numeric(core["followers"], errors="coerce").fillna(0).to_numpy()
        / (pd.to_numeric(core["following"], errors="coerce").fillna(0).to_numpy() + 1.0)
    )
    feature["Action_Frequency"] = core["user_id"].map(total_actions).fillna(0).to_numpy()
    for column, action_name in (
        ("Like_Ratio", "like"),
        ("Retweet_Ratio", "retweet"),
        ("Reply_Ratio", "reply"),
    ):
        action_count = counts.get(action_name, pd.Series(dtype=float))
        numerator = core["user_id"].map(action_count).fillna(0)
        denominator = core["user_id"].map(total_actions).fillna(0).clip(lower=1)
        feature[column] = (numerator / denominator).to_numpy()
    feature["Temporal_Entropy"] = core["user_id"].map(temporal_entropy).fillna(0).to_numpy()
    for name in patterns:
        feature[name] = core["user_id"].map(text_ratios[name]).fillna(0).to_numpy()
    for name in (
        "Empathy_Gap_Mean", "Empathy_Gap_Max", "Dark_Triad_Mean",
        "Dark_Triad_Max", "Contagion_Mean", "Contagion_Max",
        "Volatility_Mean", "Volatility_Max",
    ):
        feature[name] = 0.0
    feature["user_id"] = core["user_id"].to_numpy()

    target = root / "node_features_26d.csv"
    feature.to_csv(target, index=False)
    manifest_path = root / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["features_csv"] = {
        "path": str(target),
        "rows": len(feature),
        "columns": list(feature.columns),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }
    manifest["feature_contract"] = {
        "semantic": f"{args.embedding_model}+PCA8_fit_on_smoke_bundle",
        "behavioral": "observed_raw_relations_posts_and_timestamps",
        "psychology": "unavailable_zero_placeholder_smoke_only",
        "smoke_only": True,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "features": str(target),
        "shape": list(feature.shape),
        "sha256": _sha256(target),
    }, ensure_ascii=False, indent=2))
    return target


def prepare_plan(args: argparse.Namespace) -> Path:
    """Register the prepared real and simulated bundles in one audited plan."""
    twibot_root = args.twibot_dir.expanduser().resolve()
    twibot_bundle = args.twibot_bundle.expanduser().resolve()
    mgtab_root = args.mgtab_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    twibot_adapter_manifest = twibot_bundle / "adapter_manifest.json"
    twibot_labels = twibot_bundle / "labels.csv"
    twibot_features = twibot_bundle / "node_features_26d.csv"
    twibot_core_ids = twibot_bundle / "core_ids.txt"
    mgtab_derived = mgtab_root / "derived"
    mgtab_tensor_names = (
        "edge_index.pt", "edge_type.pt", "edge_weight.pt", "features.pt",
        "labels_bot.pt", "labels_stance.pt",
    )
    declared_inputs = (
        twibot_root, twibot_adapter_manifest, twibot_labels, twibot_features,
        twibot_core_ids, mgtab_root, *(mgtab_root / name for name in mgtab_tensor_names),
        mgtab_derived / "split_seed42.csv",
        mgtab_derived / "adapter_manifest_seed42.json",
    )
    missing = [str(path) for path in declared_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"prepared inputs are missing: {missing}")

    real_episodes = [
        EpisodeManifest(
            episode_id="real:twibot22:core1000:smoke",
            dataset_name="twibot22",
            domain="real",
            purpose="real_primary",
            partition="shared",
            split_level="node",
            source_path=str(twibot_root),
            identity_scope="dataset",
            label_provenance={"bot": "annotated"},
            capabilities={
                "temporal": True,
                "raw_text": True,
                "external_neighbors": True,
                "ground_truth_roles": False,
            },
            artifacts={
                "core_ids": str(twibot_core_ids),
                "features_csv": str(twibot_features),
                "labels_csv": str(twibot_labels),
                "splits_csv": str(twibot_labels),
                "adapter_manifest_json": str(twibot_adapter_manifest),
            },
            generator_metadata={
                "smoke_only": "true",
                "core_users": "1000",
                "adapter": "twibot22_raw_adapter",
                "source_digest_scope": "adapter_manifest",
            },
            source_sha256=_sha256(twibot_adapter_manifest),
            status="ready",
        ),
        EpisodeManifest(
            episode_id="real:mgtab:full:seed42:smoke",
            dataset_name="mgtab",
            domain="real",
            purpose="real_external",
            partition="shared",
            split_level="node",
            source_path=str(mgtab_root),
            identity_scope="dataset",
            label_provenance={"bot": "annotated"},
            capabilities={
                "temporal": False,
                "raw_text": False,
                "precomputed_text_embeddings": True,
                "ground_truth_roles": False,
            },
            artifacts={
                "edge_index_pt": str(mgtab_root / "edge_index.pt"),
                "edge_type_pt": str(mgtab_root / "edge_type.pt"),
                "edge_weight_pt": str(mgtab_root / "edge_weight.pt"),
                "features_pt": str(mgtab_root / "features.pt"),
                "labels_bot_pt": str(mgtab_root / "labels_bot.pt"),
                "labels_stance_pt": str(mgtab_root / "labels_stance.pt"),
                "splits_csv": str(mgtab_derived / "split_seed42.csv"),
                "adapter_manifest_json": str(
                    mgtab_derived / "adapter_manifest_seed42.json"
                ),
            },
            generator_metadata={
                "smoke_only": "true",
                "split_seed": "42",
                "multiedge_policy": "coalesce_with_count",
                "source_digest_scope": "six_official_pt_files",
            },
            source_sha256=_sha256_files(
                mgtab_root / name for name in mgtab_tensor_names
            ),
            status="ready",
        ),
    ]
    simulation_episodes = []
    for manifest_path in (args.leader_manifest, args.independent_manifest):
        path = manifest_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        simulation_episodes.append(EpisodeManifest.read(path))

    plan = DatasetPlan(
        plan_id="p2-smoke-data-2026-07-29",
        episodes=tuple([*real_episodes, *simulation_episodes]),
        strategy={
            "scope": "pre-training interface smoke only",
            "real_smoke": "TwiBot-22 core1000 and full standard MGTAB",
            "simulation_smoke": "2 scenarios x 1 seed x 500 agents",
            "paper_metrics_allowed": False,
        },
    )
    report = audit_plan_artifacts(plan, require_files=True)
    report.raise_for_errors()
    plan.write(output_path)
    real_manifest_paths = {}
    for episode in real_episodes:
        name = f"{episode.dataset_name}.episode.manifest.json"
        path = output_path.parent / name
        episode.write(path)
        real_manifest_paths[episode.dataset_name] = str(path)
    print(json.dumps({
        "plan": str(output_path),
        "real_manifests": real_manifest_paths,
        "summary": plan.summary(),
        "artifact_audit_valid": report.valid,
        "warnings": list(report.warnings),
    }, ensure_ascii=False, indent=2))
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    twibot = subparsers.add_parser("twibot", help="prepare the raw TwiBot-22 bundle")
    twibot.add_argument("--twibot-dir", required=True, type=Path)
    twibot.add_argument("--core-ids", required=True, type=Path)
    twibot.add_argument("--output-dir", required=True, type=Path)
    twibot.add_argument("--expected-core-count", type=int, default=1000)
    twibot.add_argument("--edge-chunksize", type=int, default=250_000)
    twibot.add_argument(
        "--contract-mode", choices=("smoke", "formal"), default="smoke",
        help="formal emits portable paths and a non-smoke schema contract",
    )
    twibot.set_defaults(handler=prepare_twibot)

    mgtab = subparsers.add_parser("mgtab", help="prepare the full MGTAB bundle")
    mgtab.add_argument("--mgtab-dir", required=True, type=Path)
    mgtab.add_argument("--output-dir", required=True, type=Path)
    mgtab.add_argument("--split-seed", type=int, default=42)
    mgtab.add_argument(
        "--multiedge-policy",
        choices=("coalesce_with_count", "preserve_multiedges"),
        default="coalesce_with_count",
    )
    mgtab.set_defaults(handler=prepare_mgtab)

    features = subparsers.add_parser(
        "twibot-features", help="build smoke-only 26d features from a raw bundle"
    )
    features.add_argument("--bundle-dir", required=True, type=Path)
    features.add_argument("--expected-core-count", type=int, default=1000)
    features.add_argument("--embedding-model", default="all-mpnet-base-v2")
    features.add_argument(
        "--model-cache", type=Path,
        default=Path(".runtime/huggingface/hub"),
    )
    features.add_argument("--batch-size", type=int, default=32)
    features.set_defaults(handler=prepare_twibot_features)

    plan = subparsers.add_parser(
        "plan", help="write one audited P2 smoke DatasetPlan"
    )
    plan.add_argument("--twibot-dir", required=True, type=Path)
    plan.add_argument("--twibot-bundle", required=True, type=Path)
    plan.add_argument("--mgtab-dir", required=True, type=Path)
    plan.add_argument("--leader-manifest", required=True, type=Path)
    plan.add_argument("--independent-manifest", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.set_defaults(handler=prepare_plan)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
