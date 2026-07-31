"""Audited formal 26-dimensional feature extraction for materialized TwiBot-22.

This module intentionally does not reuse the smoke-only feature command.  It
uses the repository's four psychological-language engines, preserves the
official split, fits unsupervised transforms on train users only, and writes
evidence/provenance sidecars alongside the 26 model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SEMANTIC_COLUMNS = [f"Semantic_{index}" for index in range(8)]
BEHAVIOR_COLUMNS = [
    "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
    "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
    "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
]
PSYCHOLOGY_COLUMNS = [
    "Empathy_Gap_Mean", "Empathy_Gap_Max", "Dark_Triad_Mean",
    "Dark_Triad_Max", "Contagion_Mean", "Contagion_Max",
    "Volatility_Mean", "Volatility_Max",
]
FEATURE_COLUMNS = SEMANTIC_COLUMNS + BEHAVIOR_COLUMNS + PSYCHOLOGY_COLUMNS
csv.field_size_limit(sys.maxsize)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_frames(bundle_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = pd.read_csv(bundle_dir / "core_users.csv", low_memory=False)
    labels = pd.read_csv(bundle_dir / "labels.csv", low_memory=False)
    posts = pd.read_csv(bundle_dir / "posts.csv", engine="python")
    actions = pd.read_csv(bundle_dir / "actions.csv", engine="python")
    for frame, column in ((core, "user_id"), (labels, "user_id")):
        frame[column] = frame[column].astype(str)
        if frame[column].duplicated().any():
            raise ValueError(f"duplicate {column} values in materialized bundle")
    posts["author_id"] = posts["author_id"].astype(str)
    posts["post_id"] = posts["post_id"].astype(str)
    posts["content"] = posts["content"].fillna("").astype(str)
    actions["actor_id"] = actions["actor_id"].astype(str)
    merged = core[["user_id"]].merge(
        labels[["user_id", "data_split", "label"]], on="user_id", how="left",
        validate="one_to_one",
    )
    if merged[["data_split", "label"]].isna().any().any():
        raise ValueError("some core users have no official split or label")
    return core, labels, posts, actions


def select_posts(
    posts: pd.DataFrame,
    core_ids: Iterable[str],
    max_tweets_per_user: int | None,
) -> pd.DataFrame:
    """Choose deterministic recent evidence, then restore chronological order."""
    selected = posts[posts["author_id"].isin(set(map(str, core_ids)))].copy()
    selected["_time"] = pd.to_datetime(selected["created_at"], errors="coerce", utc=True)
    selected = selected.sort_values(
        ["author_id", "_time", "post_id"], na_position="first", kind="mergesort",
    )
    if max_tweets_per_user is not None:
        if max_tweets_per_user < 1:
            raise ValueError("max_tweets_per_user must be positive")
        selected = selected.groupby("author_id", sort=False, group_keys=False).tail(
            max_tweets_per_user
        )
        selected = selected.sort_values(
            ["author_id", "_time", "post_id"], na_position="first", kind="mergesort",
        )
    return selected.drop(columns=["_time"])


def build_behavior_features(
    core: pd.DataFrame,
    actions: pd.DataFrame,
    posts: pd.DataFrame,
) -> pd.DataFrame:
    counts = actions.groupby(["actor_id", "action_type"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    times = pd.to_datetime(actions.get("event_time"), errors="coerce", utc=True)
    timed = actions.assign(_hour=times.dt.hour).dropna(subset=["_hour"])
    entropy: dict[str, float] = {}
    for user_id, group in timed.groupby("actor_id"):
        probabilities = group["_hour"].value_counts(normalize=True).to_numpy()
        entropy[str(user_id)] = float(-sum(p * math.log(p + 1e-12) for p in probabilities))

    patterns = {
        "URL_Ratio": r"https?://|www\.",
        "Mention_Ratio": r"@[A-Za-z0-9_]",
        "Hashtag_Ratio": r"#[A-Za-z0-9_]",
        "Media_Ratio": r"pic\.twitter\.com|\.(?:jpg|jpeg|png|gif|mp4)\b",
    }
    ratios: dict[str, dict[str, float]] = {name: {} for name in patterns}
    for user_id, group in posts.groupby("author_id"):
        denominator = max(1, len(group))
        for name, pattern in patterns.items():
            ratios[name][str(user_id)] = float(
                group["content"].str.contains(pattern, case=False, regex=True).sum()
                / denominator
            )

    result = pd.DataFrame(index=core.index)
    followers = pd.to_numeric(core["followers"], errors="coerce").fillna(0)
    following = pd.to_numeric(core["following"], errors="coerce").fillna(0)
    result["Follower_Following_Ratio"] = followers / (following + 1.0)
    result["Action_Frequency"] = core["user_id"].map(totals).fillna(0).to_numpy()
    for column, action_name in (
        ("Like_Ratio", "like"), ("Retweet_Ratio", "retweet"), ("Reply_Ratio", "reply")
    ):
        numerator = core["user_id"].map(counts.get(action_name, pd.Series(dtype=float))).fillna(0)
        denominator = core["user_id"].map(totals).fillna(0).clip(lower=1)
        result[column] = (numerator / denominator).to_numpy()
    result["Temporal_Entropy"] = core["user_id"].map(entropy).fillna(0).to_numpy()
    for name in patterns:
        result[name] = core["user_id"].map(ratios[name]).fillna(0).to_numpy()
    return result[BEHAVIOR_COLUMNS]


def _group_texts(core: pd.DataFrame, posts: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    text_by_user: dict[str, list[str]] = {}
    ids_by_user: dict[str, list[str]] = {}
    for user_id, group in posts.groupby("author_id", sort=False):
        valid = group[group["content"].str.strip().ne("")]
        text_by_user[str(user_id)] = valid["content"].tolist()
        ids_by_user[str(user_id)] = valid["post_id"].tolist()
    return (
        [text_by_user.get(uid, []) for uid in core["user_id"]],
        [ids_by_user.get(uid, []) for uid in core["user_id"]],
    )


def build_semantic_features(
    core: pd.DataFrame,
    text_groups: list[list[str]],
    embedding_model: Path,
    train_mask: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(embedding_model), device="cuda", local_files_only=True)
    bios = core["bio"].fillna("").astype(str).tolist()
    bio_embeddings = np.asarray(model.encode(
        bios, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
    ))
    flat_texts = [text for group in text_groups for text in group]
    tweet_embeddings = np.zeros_like(bio_embeddings)
    if flat_texts:
        flat_embeddings = np.asarray(model.encode(
            flat_texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        ))
        cursor = 0
        for index, group in enumerate(text_groups):
            if group:
                tweet_embeddings[index] = flat_embeddings[cursor:cursor + len(group)].mean(axis=0)
                cursor += len(group)
    combined = np.hstack([bio_embeddings, tweet_embeddings])
    pca = PCA(n_components=8, random_state=42)
    transformed = pca.fit(combined[train_mask]).transform(combined)
    return transformed, {
        "input_dim": int(combined.shape[1]),
        "fit_scope": "official_train_core_users_only",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),
    }


def _top_evidence(results: list[dict[str, Any]], post_ids: list[str], key: str) -> list[dict[str, Any]]:
    ranked = sorted(
        ((index, float(value.get(key, 0.0))) for index, value in enumerate(results)),
        key=lambda item: item[1], reverse=True,
    )[:5]
    return [
        {"post_id": post_ids[index], "evidence_id": f"post:{post_ids[index]}", "score": score}
        for index, score in ranked if score > 0 and index < len(post_ids)
    ]


def build_psychology_features(
    text_groups: list[list[str]],
    post_id_groups: list[list[str]],
    emotion_model: Path,
    fluency_model: Path,
    nli_model: Path,
    embedding_model: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import torch
    from emotional_analysis.contagion_analyzer import ContagionAnalyzer
    from emotional_analysis.dark_triad_analyzer import DarkTriadAnalyzer
    from emotional_analysis.empathy_gap_analyzer import EmpathyGapAnalyzer
    from emotional_analysis.volatility_analyzer import EmotionVolatilityAnalyzer

    device = torch.device("cuda")
    empathy = EmpathyGapAnalyzer.reload(
        device=device, emotion_model_name=str(emotion_model),
        ppl_model_name=str(fluency_model), enable_fluency=True,
    )
    dark = DarkTriadAnalyzer.reload(device=device, nli_model_name=str(nli_model))
    contagion = ContagionAnalyzer.reload(device=device, model_name=str(embedding_model))
    volatility = EmotionVolatilityAnalyzer.reload(
        device=device, emotion_model_name=str(emotion_model),
    )
    if dark.model is None or contagion.embedder is None or volatility.model is None:
        raise RuntimeError("one or more formal psychology models failed to load")
    if not empathy.has_dependency_parser:
        raise RuntimeError("formal empathy extraction requires a spaCy dependency parser")
    if not empathy.high_arousal_negative_labels:
        raise RuntimeError("emotion model exposes none of the required negative-emotion labels")

    flat = [text for group in text_groups for text in group]
    empathy_flat = empathy.analyze_batch(flat, batch_size=32)
    dark_flat = dark.analyze_batch(flat, batch_size=32)
    contagion_flat = contagion.analyze_batch(flat, batch_size=64)
    volatility_users = volatility.evaluate_agents_batch(text_groups, batch_size=64)

    rows: list[list[float]] = []
    evidence: list[dict[str, Any]] = []
    cursor = 0
    for index, (texts, post_ids) in enumerate(zip(text_groups, post_id_groups)):
        stop = cursor + len(texts)
        emp = empathy_flat[cursor:stop]
        dt = dark_flat[cursor:stop]
        cont = contagion_flat[cursor:stop]
        emp_values = [float(value.get("Empathy_Gap", 0.0)) for value in emp]
        dt_values = [float(value.get("Dark_Triad_Index", 0.0)) for value in dt]
        cont_values = [float(value.get("Max_Payload_Alignment", 0.0)) for value in cont]
        vol = volatility_users[index]
        rows.append([
            float(np.mean(emp_values)) if emp_values else 0.0,
            float(np.max(emp_values)) if emp_values else 0.0,
            float(np.mean(dt_values)) if dt_values else 0.0,
            float(np.max(dt_values)) if dt_values else 0.0,
            float(np.mean(cont_values)) if cont_values else 0.0,
            float(np.max(cont_values)) if cont_values else 0.0,
            float(vol.get("Agent_Mean_Volatility", 0.0)),
            float(vol.get("Agent_Max_Volatility", 0.0)),
        ])
        evidence.append({
            "empathy_gap": _top_evidence(emp, post_ids, "Empathy_Gap"),
            "dark_triad": _top_evidence(dt, post_ids, "Dark_Triad_Index"),
            "contagion": _top_evidence(cont, post_ids, "Frictionless_Contagion_Score"),
            "volatility_source_post_ids": post_ids,
        })
        cursor = stop
    return np.asarray(rows, dtype=float), evidence


def _column_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        column: {
            "min": float(frame[column].min()), "max": float(frame[column].max()),
            "mean": float(frame[column].mean()), "std": float(frame[column].std(ddof=0)),
            "zero_fraction": float(frame[column].eq(0).mean()),
            "missing_fraction": float(frame[column].isna().mean()),
        }
        for column in frame.columns if column != "user_id"
    }


def run(args: argparse.Namespace) -> Path:
    started = datetime.now(timezone.utc)
    root = args.bundle_dir.resolve()
    output = args.output.resolve()
    core, labels, posts, actions = load_frames(root)
    if args.expected_core_count is not None and len(core) != args.expected_core_count:
        raise ValueError(f"expected {args.expected_core_count} core users, found {len(core)}")
    if args.pilot_users is not None:
        merged = core[["user_id"]].merge(labels, on="user_id", validate="one_to_one")
        if args.pilot_users < 1 or args.pilot_users > len(merged):
            raise ValueError("pilot_users must be within the core population")
        strata = list(merged.groupby(["data_split", "label"], sort=True))
        exact = [args.pilot_users * len(group) / len(merged) for _, group in strata]
        allocations = [math.floor(value) for value in exact]
        for index in sorted(
            range(len(strata)), key=lambda idx: exact[idx] - allocations[idx], reverse=True
        )[: args.pilot_users - sum(allocations)]:
            allocations[index] += 1
        selected_ids: list[str] = []
        for (_, group), take in zip(strata, allocations):
            ordered = group.assign(
                _key=group["user_id"].map(
                    lambda uid: hashlib.sha256(f"42:{uid}".encode()).hexdigest()
                )
            ).sort_values("_key")
            selected_ids.extend(ordered.head(take)["user_id"].tolist())
        selected_set = set(selected_ids)
        core = core[core["user_id"].isin(selected_set)].copy().reset_index(drop=True)

    all_core_posts = posts[posts["author_id"].isin(set(core["user_id"]))].copy()
    selected_posts = select_posts(posts, core["user_id"], args.max_tweets_per_user)
    text_groups, post_id_groups = _group_texts(core, selected_posts)
    label_order = core[["user_id"]].merge(
        labels[["user_id", "data_split", "label"]], on="user_id", validate="one_to_one"
    )
    train_mask = label_order["data_split"].eq("train").to_numpy()
    if train_mask.sum() < 9:
        raise ValueError("at least 9 train users are required for PCA8")

    semantic, pca_state = build_semantic_features(
        core, text_groups, args.embedding_model.resolve(), train_mask, args.batch_size,
    )
    behavior_raw = build_behavior_features(core, actions, all_core_posts)
    psychology_raw, evidence = build_psychology_features(
        text_groups, post_id_groups, args.emotion_model.resolve(),
        args.fluency_model.resolve(), args.nli_model.resolve(),
        args.embedding_model.resolve(),
    )
    raw_nonsemantic = np.hstack([behavior_raw.to_numpy(float), psychology_raw])
    scaler = StandardScaler().fit(raw_nonsemantic[train_mask])
    normalized = scaler.transform(raw_nonsemantic)
    features = pd.DataFrame(
        np.hstack([semantic, normalized]), columns=FEATURE_COLUMNS,
    )
    features.insert(0, "user_id", core["user_id"].to_numpy())
    raw_features = pd.DataFrame(
        np.hstack([semantic, raw_nonsemantic]), columns=FEATURE_COLUMNS,
    )
    raw_features.insert(0, "user_id", core["user_id"].to_numpy())
    if not np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("formal features contain non-finite values")
    zero_psych = [column for column in PSYCHOLOGY_COLUMNS if raw_features[column].eq(0).all()]
    if args.enforce_nonzero_psychology and zero_psych:
        raise ValueError(f"all-zero formal psychology columns: {zero_psych}")

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
    evidence_output = output.with_name("psychology_evidence.jsonl")
    manifest_output = output.with_name("formal_feature_manifest.json")
    features.to_csv(output, index=False)
    raw_features.to_csv(raw_output, index=False)
    with evidence_output.open("w", encoding="utf-8") as handle:
        for user_id, item in zip(core["user_id"], evidence):
            handle.write(json.dumps({"user_id": user_id, **item}, ensure_ascii=False) + "\n")

    model_paths = {
        "semantic_and_contagion": args.embedding_model.resolve(),
        "emotion_and_volatility": args.emotion_model.resolve(),
        "fluency": args.fluency_model.resolve(),
        "dark_triad_nli": args.nli_model.resolve(),
    }
    manifest = {
        "schema_version": "hyperdecept.twibot22-formal-features.v1",
        "status": "passed" if not zero_psych else "warning",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "extractor_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "bundle_dir": str(root),
        "core_users": len(core),
        "official_split_counts": label_order["data_split"].value_counts().to_dict(),
        "max_tweets_per_user": args.max_tweets_per_user,
        "selected_posts": len(selected_posts),
        "users_with_text": sum(bool(group) for group in text_groups),
        "text_coverage": sum(bool(group) for group in text_groups) / len(core),
        "post_count_quantiles": pd.Series([len(group) for group in text_groups]).quantile(
            [0, .25, .5, .75, .9, .95, .99, 1]
        ).to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "psychology_semantics": "observable_language_behavior_proxies_not_personality_diagnoses",
        "all_zero_psychology_columns": zero_psych,
        "pca": pca_state,
        "standard_scaler": {
            "fit_scope": "official_train_core_users_only",
            "columns": BEHAVIOR_COLUMNS + PSYCHOLOGY_COLUMNS,
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        },
        "models": {
            name: {"path": str(path), "path_exists": path.exists()}
            for name, path in model_paths.items()
        },
        "artifacts": {},
        "raw_column_statistics": _column_statistics(raw_features),
        "normalized_column_statistics": _column_statistics(features),
    }
    if args.model_cache_manifest is not None:
        cache_manifest = args.model_cache_manifest.resolve()
        manifest["model_cache_manifest"] = {
            "path": str(cache_manifest),
            "bytes": cache_manifest.stat().st_size,
            "sha256": _sha256(cache_manifest),
        }
    for name, path in {
        "features": output, "raw_features": raw_output, "evidence": evidence_output,
    }.items():
        manifest["artifacts"][name] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--emotion-model", type=Path, required=True)
    parser.add_argument("--fluency-model", type=Path, required=True)
    parser.add_argument("--nli-model", type=Path, required=True)
    parser.add_argument("--model-cache-manifest", type=Path)
    parser.add_argument("--expected-core-count", type=int)
    parser.add_argument("--max-tweets-per-user", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pilot-users", type=int)
    parser.add_argument("--enforce-nonzero-psychology", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
