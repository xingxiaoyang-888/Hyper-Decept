"""Train Hyper-Decept once and score an independent simulation database.

Unlike ``new_main_classifier.py``, this entry point never refits PCA, scaling,
or XGBoost on the evaluation database.  It is intended for paired adversarial
experiments where the detector must remain fixed while the attack changes.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import build_text_fields, label_columns, load_label_frame, resolve_repo_path
from graph_builder import (
    add_knn_edges_to_graph,
    build_cosine_edges,
    compute_graph_features,
    get_original_graph,
)
from new_feature_extractor import MultimodalExtractor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INTRINSIC_NAMES = (
    [f"Semantic_{i}" for i in range(8)]
    + [
        "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
        "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy", "URL_Ratio",
        "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio", "Empathy_Gap_Mean",
        "Empathy_Gap_Max", "Dark_Triad_Mean", "Dark_Triad_Max",
        "Contagion_Mean", "Contagion_Max", "Volatility_Mean", "Volatility_Max",
    ]
)


def _raw_modalities(db_path, csv_path, psychology_mode, cache_dir):
    labels = load_label_frame(csv_path, db_path)
    labels["user_id"] = labels["user_id"].astype(str)
    bios, tweets, _ = build_text_fields(labels)
    user_ids = labels["user_id"].tolist()

    extractor = MultimodalExtractor(
        psychology_mode=psychology_mode,
        cache_dir=str(cache_dir),
    )
    extracted_ids, behavior = extractor.extract_behavior_features(db_path, user_ids)
    semantic = extractor._encode_text_dual_stream(bios, tweets)
    psychology = extractor._extract_llm_native_psychology(tweets)
    if list(map(str, extracted_ids)) != user_ids:
        raise ValueError("Feature extractor changed user order; refusing unsafe alignment.")
    return labels, user_ids, semantic, behavior, psychology


def _fit_intrinsic(semantic, behavior, psychology):
    semantic_pca = PCA(n_components=min(8, semantic.shape[0], semantic.shape[1]), random_state=42)
    behavior_scaler = StandardScaler()
    semantic_8 = semantic_pca.fit_transform(semantic)
    if semantic_8.shape[1] < 8:
        semantic_8 = np.pad(semantic_8, ((0, 0), (0, 8 - semantic_8.shape[1])))
    behavior_psych = behavior_scaler.fit_transform(np.hstack([behavior, psychology]))
    return np.hstack([semantic_8, behavior_psych]), semantic_pca, behavior_scaler


def _transform_intrinsic(semantic, behavior, psychology, semantic_pca, behavior_scaler):
    semantic_8 = semantic_pca.transform(semantic)
    if semantic_8.shape[1] < 8:
        semantic_8 = np.pad(semantic_8, ((0, 0), (0, 8 - semantic_8.shape[1])))
    behavior_psych = behavior_scaler.transform(np.hstack([behavior, psychology]))
    return np.hstack([semantic_8, behavior_psych])


def _final_frame(labels, user_ids, intrinsic, db_path, similarity_threshold):
    original_graph = get_original_graph(labels, db_path)
    similarity_edges = build_cosine_edges(intrinsic, threshold=similarity_threshold)
    graph = add_knn_edges_to_graph(original_graph.copy(), similarity_edges, user_ids)
    graph_features = compute_graph_features(graph, user_ids).reset_index(names="user_id")

    intrinsic_frame = pd.DataFrame(intrinsic, columns=INTRINSIC_NAMES)
    intrinsic_frame["user_id"] = user_ids
    final = intrinsic_frame.merge(graph_features, on="user_id", how="left")
    final = final.merge(labels[label_columns(labels)], on="user_id", how="inner")
    metadata = [c for c in ["user_id", "user_type", "name", "username", "is_bad"] if c in final]
    features = [c for c in final.columns if c not in metadata]
    final[features] = final[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return final, features, graph


def _xgb_kwargs(y):
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    return dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=negatives / max(positives, 1),
    )


def run(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_cache"

    train = _raw_modalities(
        args.train_db, args.train_csv, args.psychology_mode, cache_dir / "train"
    )
    test = _raw_modalities(
        args.test_db, args.test_csv, args.psychology_mode, cache_dir / "test"
    )

    train_intrinsic, semantic_pca, behavior_scaler = _fit_intrinsic(*train[2:])
    test_intrinsic = _transform_intrinsic(*test[2:], semantic_pca, behavior_scaler)
    train_frame, feature_names, _ = _final_frame(
        train[0], train[1], train_intrinsic, args.train_db, args.sim_threshold
    )
    test_frame, test_feature_names, test_graph = _final_frame(
        test[0], test[1], test_intrinsic, args.test_db, args.sim_threshold
    )
    if feature_names != test_feature_names:
        raise ValueError("Train/test feature schemas differ.")

    x_train = train_frame[feature_names].to_numpy(float)
    y_train = train_frame["is_bad"].to_numpy(int)
    x_test = test_frame[feature_names].to_numpy(float)
    y_test = test_frame["is_bad"].to_numpy(int)
    model = xgb.XGBClassifier(**_xgb_kwargs(y_train))
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= args.threshold).astype(int)

    print("\nIndependent fixed-detector test")
    print(classification_report(y_test, prediction, target_names=["Human", "Bot"]))
    auc = roc_auc_score(y_test, probability) if len(np.unique(y_test)) == 2 else None
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    result_columns = [c for c in ["user_id", "user_type", "name", "username", "is_bad"] if c in test_frame]
    results = test_frame[result_columns].copy()
    results["y_pred"] = prediction
    results["prob_bot"] = probability
    results.to_csv(output_dir / "independent_test_results.csv", index=False)
    train_frame.to_csv(output_dir / "train_features.csv", index=False)
    test_frame.to_csv(output_dir / "test_features.csv", index=False)
    pd.DataFrame(test_graph.edges(), columns=["source", "target"]).to_csv(
        output_dir / "test_graph_edges.csv", index=False
    )

    matrix = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Fixed Hyper-Decept: Independent Test")
    plt.tight_layout()
    plt.savefig(output_dir / "independent_confusion_matrix.png", dpi=300)
    plt.close()

    bundle = {
        "semantic_pca": semantic_pca,
        "behavior_scaler": behavior_scaler,
        "model": model,
        "feature_names": feature_names,
        "psychology_mode": args.psychology_mode,
        "sim_threshold": args.sim_threshold,
        "decision_threshold": args.threshold,
    }
    joblib.dump(bundle, output_dir / "fixed_detector.joblib")

    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "train_db": str(Path(args.train_db).resolve()),
        "test_db": str(Path(args.test_db).resolve()),
        "psychology_mode": args.psychology_mode,
        "threshold": args.threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((tp + tn) / max(matrix.sum(), 1)),
        "bot_precision": float(tp / max(tp + fp, 1)),
        "bot_recall": float(tp / max(tp + fn, 1)),
        "attack_escape_rate": float(fn / max(tp + fn, 1)),
        "roc_auc": float(auc) if auc is not None else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved fixed detector and independent results to %s", output_dir)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train once, then independently score attack traces.")
    parser.add_argument("--train-db", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-db", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--psychology-mode", choices=["full", "fast", "off"], default="off")
    parser.add_argument("--sim-threshold", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
