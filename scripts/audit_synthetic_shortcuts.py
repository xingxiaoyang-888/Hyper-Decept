#!/usr/bin/env python3
"""Audit synthetic LOSO data for feature shortcuts and cross-fold duplicates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.episode_manifest import (  # noqa: E402
    DatasetPlan,
    audit_episode_splits,
    leave_one_scenario_out_assignments,
)
from data_processing.feature_contracts import (  # noqa: E402
    FEATURE_CONTRACTS,
    FULL_26,
    OBSERVABLE_18,
)


FORBIDDEN_INPUT_TOKENS = (
    "is_bad",
    "user_type",
    "role",
    "campaign",
    "next_action",
    "personality",
    "hidden",
    "label",
)
GROUPS = (
    "observable18",
    "semantic8",
    "behavior10",
    "psychology_proxy8",
    "full26",
)


def _load_episode(episode) -> pd.DataFrame:
    features_path = Path(episode.artifacts["features_csv"])
    labels_path = Path(episode.artifacts["labels_csv"])
    features = pd.read_csv(features_path, low_memory=False)
    labels = pd.read_csv(labels_path, low_memory=False)
    if "user_id" not in features or "user_id" not in labels or "is_bad" not in labels:
        raise ValueError(f"invalid feature/label contract for {episode.episode_id}")
    suspicious = sorted(
        column
        for column in features.columns
        if column != "user_id"
        and any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)
    )
    if suspicious:
        raise ValueError(
            f"forbidden label/provenance columns in features for "
            f"{episode.episode_id}: {suspicious}"
        )
    missing = sorted(set(FULL_26).difference(features.columns))
    if missing:
        raise ValueError(f"{episode.episode_id} missing feature columns: {missing}")
    frame = features[["user_id", *FULL_26]].merge(
        labels[["user_id", "is_bad"]],
        on="user_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != int(episode.num_agents):
        raise ValueError(
            f"{episode.episode_id} aligned rows={len(frame)}, expected {episode.num_agents}"
        )
    frame[list(FULL_26)] = frame[list(FULL_26)].apply(
        pd.to_numeric, errors="raise"
    )
    values = frame[list(FULL_26)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{episode.episode_id} contains non-finite feature values")
    labels_array = pd.to_numeric(frame["is_bad"], errors="raise").to_numpy()
    if set(np.unique(labels_array)).difference({0, 1}):
        raise ValueError(f"{episode.episode_id} contains non-binary labels")
    frame["is_bad"] = labels_array.astype(np.int64)
    frame["episode_id"] = episode.episode_id
    return frame


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auprc": float(average_precision_score(targets, probabilities)),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
    }


def _fit_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
    )
    model.fit(train[list(columns)], train["is_bad"])
    probabilities = model.predict_proba(test[list(columns)])[:, 1]
    return _metrics(test["is_bad"].to_numpy(), probabilities)


def _overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float | int]:
    train_hashes = set(
        pd.util.hash_pandas_object(train[list(OBSERVABLE_18)], index=False).tolist()
    )
    test_hashes = pd.util.hash_pandas_object(
        test[list(OBSERVABLE_18)], index=False
    ).tolist()
    count = sum(value in train_hashes for value in test_hashes)
    return {
        "exact_observable18_test_rows_seen_in_train": int(count),
        "test_rows": int(len(test)),
        "fraction": float(count / len(test)),
    }


def audit(plan: DatasetPlan) -> dict:
    episodes = [
        episode
        for episode in plan.episodes
        if episode.domain == "synthetic" and episode.purpose == "simulation_main"
    ]
    frames = {episode.episode_id: _load_episode(episode) for episode in episodes}
    scenarios = sorted({str(episode.scenario_id) for episode in episodes})
    folds = []
    for scenario in scenarios:
        assignments = leave_one_scenario_out_assignments(plan, scenario)
        split_audit = audit_episode_splits(plan, assignments)
        split_audit.raise_for_errors()
        train = pd.concat([
            frames[episode.episode_id]
            for episode in episodes
            if assignments[episode.episode_id] == "train"
        ], ignore_index=True)
        test = pd.concat([
            frames[episode.episode_id]
            for episode in episodes
            if assignments[episode.episode_id] == "test"
        ], ignore_index=True)
        group_scores = {
            name: _fit_score(train, test, FEATURE_CONTRACTS[name])
            for name in GROUPS
        }
        univariate = {
            column: _fit_score(train, test, (column,))
            for column in FULL_26
        }
        folds.append({
            "held_out_scenario": scenario,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_positive_fraction": float(train["is_bad"].mean()),
            "test_positive_fraction": float(test["is_bad"].mean()),
            "exact_feature_overlap": _overlap(train, test),
            "group_scores": group_scores,
            "univariate_scores": univariate,
        })

    macro_groups = {
        group: {
            metric: float(np.mean([
                fold["group_scores"][group][metric] for fold in folds
            ]))
            for metric in ("auprc", "auroc", "f1", "balanced_accuracy")
        }
        for group in GROUPS
    }
    macro_univariate_auprc = {
        column: float(np.mean([
            fold["univariate_scores"][column]["auprc"] for fold in folds
        ]))
        for column in FULL_26
    }
    ranked = sorted(
        macro_univariate_auprc.items(), key=lambda item: item[1], reverse=True
    )
    warnings = []
    if ranked and ranked[0][1] >= 0.90:
        warnings.append(
            "A single feature reaches macro AUPRC >= 0.90; investigate generator leakage."
        )
    overlap_fraction = float(np.mean([
        fold["exact_feature_overlap"]["fraction"] for fold in folds
    ]))
    if overlap_fraction > 0:
        warnings.append(
            "Exact observable18 vectors cross train/test episodes; episode-scoped IDs "
            "do not eliminate profile-level duplication."
        )
    return {
        "schema_version": "hypertrace.synthetic-shortcut-audit.v1",
        "status": "warning" if warnings else "passed",
        "plan_id": plan.plan_id,
        "episode_count": len(episodes),
        "scenario_count": len(scenarios),
        "macro_group_scores": macro_groups,
        "ranked_univariate_auprc": [
            {"feature": feature, "auprc": value} for feature, value in ranked
        ],
        "macro_exact_observable18_overlap_fraction": overlap_fraction,
        "folds": folds,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(DatasetPlan.read(args.plan))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
