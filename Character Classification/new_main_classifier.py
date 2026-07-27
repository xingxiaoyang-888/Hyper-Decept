"""
Script 1: build the enhanced heterogeneous graph and run binary bot detection.

The model logic is unchanged from the original script:
1. MultimodalExtractor builds semantic, behavior, and psychology features.
2. Features are reduced to 26 dimensions.
3. Cosine-similarity edges are merged with original follow edges.
4. Graph statistics are appended to the feature matrix.
5. XGBoost is evaluated with cross-validation and fitted on the full data.
6. Results, graph edges, node features, and visual reports are saved.
"""

import argparse
import json
import os
import sys
import warnings
import logging

# Ensure the project root and Character Classification dir are importable
# regardless of the working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict

from config import (
    DATASET_CHOICES,
    PROJECT_ROOT,
    build_text_fields,
    configure_utf8_streams,
    label_columns,
    load_label_frame,
    make_experiment_dir,
    resolve_dataset_paths,
    write_manifest,
)
from graph_builder import (
    add_knn_edges_to_graph,
    build_26dim_features,
    build_cosine_edges,
    compute_graph_features,
    get_original_graph,
)
from data_processing.dataset_adapter import detect_dataset_kind
from new_feature_extractor import MultimodalExtractor as _Extractor
from visualizer import CognitiveVisualizer

# -- white-box explainability (optional) -----------------------------------
try:
    from explainability.evidence_registry import EvidenceRegistry
    from explainability.local_explainer import LocalTreeExplainer
    from explainability.packet_builder import ExplanationPacketBuilder
except ImportError as _wb_import_err:
    EvidenceRegistry = None       # type: ignore[assignment]
    LocalTreeExplainer = None     # type: ignore[assignment]
    ExplanationPacketBuilder = None  # type: ignore[assignment]
    _wb_import_err = None

try:
    import shap
except ImportError:
    shap = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
configure_utf8_streams()

DB_FILE, CSV_FILE = resolve_dataset_paths()
SAVE_DIR = str(PROJECT_ROOT / "new_result" / "hyper_newtest")
SIM_THRESHOLD = 0.7

FEAT_NAMES_26 = (
    [f"Semantic_{i}" for i in range(8)] +
    [
        "Follower_Following_Ratio", "Action_Frequency", "Like_Ratio",
        "Retweet_Ratio", "Reply_Ratio", "Temporal_Entropy",
        "URL_Ratio", "Mention_Ratio", "Hashtag_Ratio", "Media_Ratio",
        "Empathy_Gap_Mean", "Empathy_Gap_Max",
        "Dark_Triad_Mean", "Dark_Triad_Max",
        "Contagion_Mean", "Contagion_Max",
        "Volatility_Mean", "Volatility_Max",
    ]
)


def _build_xgb_kwargs(y):
    bad_count = int(np.sum(y))
    good_count = int(len(y) - bad_count)
    kwargs = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=good_count / max(bad_count, 1),
    )
    if torch.cuda.is_available():
        try:
            xgb.XGBClassifier(device="cuda", **kwargs)
            kwargs["device"] = "cuda"
            logger.info("  XGBoost using CUDA")
        except TypeError:
            kwargs["tree_method"] = "gpu_hist"
            logger.info("  XGBoost using gpu_hist")
    return kwargs


def _cross_val_predict(model, X, y):
    n_samples = len(y)
    bad_count = int(np.sum(y))
    good_count = int(n_samples - bad_count)

    if bad_count == 0 or good_count == 0:
        raise ValueError("Single-class dataset, cannot do binary classification.")

    if n_samples <= 20:
        logger.info("LOOCV mode...")
        loo = LeaveOneOut()
        y_pred = np.zeros(n_samples)
        y_proba = np.zeros(n_samples)
        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            if len(np.unique(y_tr)) < 2:
                y_pred[test_idx] = y_tr[0]
                y_proba[test_idx] = float(y_tr[0])
            else:
                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)
                y_pred[test_idx] = fold_model.predict(X_te)[0]
                y_proba[test_idx] = fold_model.predict_proba(X_te)[0, 1]
        return y_pred, y_proba

    n_splits = min(5, bad_count, good_count)
    logger.info("%s-Fold CV + fold-internal SMOTE...", n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    use_smote = SMOTE is not None and ImbPipeline is not None and min(bad_count, good_count) > 5
    if use_smote:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("xgb", model),
        ])
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv)
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return y_pred, y_proba


def run_classifier(
    db_file=DB_FILE,
    csv_file=CSV_FILE,
    save_dir=SAVE_DIR,
    sim_threshold=SIM_THRESHOLD,
    psychology_mode="full",
    max_tweets_per_user=None,
    run_visualizer=True,
    whitebox=False,
    whitebox_top_k=15,
):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Script 1: enhanced heterogeneous graph + binary classification")
    print("=" * 60)
    logger.info("DB: %s", db_file)
    logger.info("CSV: %s", csv_file)
    logger.info("Output: %s", save_dir)

    logger.info("Loading CSV...")
    df_labels = load_label_frame(csv_file, db_file)
    df_labels["user_id"] = df_labels["user_id"].astype(str)
    logger.info("  %s samples, %s bots", len(df_labels), int(df_labels["is_bad"].sum()))

    bios, tweets_joined, _ = build_text_fields(df_labels)
    user_ids_master = df_labels["user_id"].astype(str).tolist()

    logger.info("Initializing extractor...")
    extractor = _Extractor(
        psychology_mode=psychology_mode,
        max_tweets_per_user=max_tweets_per_user,
        cache_dir=os.path.join(save_dir, "feature_cache"),
    )
    _fuse_result = extractor.fuse_multimodal_features(
        db_file,
        bios,
        tweets_list=tweets_joined,
        user_ids_master=user_ids_master,
        return_provenance=whitebox,
    )
    if whitebox:
        user_ids, fused_matrix, provenance = _fuse_result
    else:
        user_ids, fused_matrix = _fuse_result
        provenance = None
    user_ids = [str(uid) for uid in user_ids]
    logger.info("  Fused matrix: %s", fused_matrix.shape)

    logger.info("Reducing to 26-dim features...")
    features_26 = build_26dim_features(fused_matrix, n_semantic=8)
    logger.info("  26-dim shape: %s", features_26.shape)

    logger.info("Building enhanced graph (cosine threshold=%s)...", sim_threshold)
    G_orig = get_original_graph(df_labels, db_file)
    logger.info("  Original graph: %s nodes, %s edges", G_orig.number_of_nodes(), G_orig.number_of_edges())

    cosine_edges = build_cosine_edges(features_26, threshold=sim_threshold)
    logger.info("  Cosine similarity edges: %s", len(cosine_edges))

    G = G_orig.copy()
    G = add_knn_edges_to_graph(G, cosine_edges, user_ids)
    logger.info("  Enhanced graph: %s nodes, %s edges", G.number_of_nodes(), G.number_of_edges())

    logger.info("Computing graph features...")
    df_graph_feats = compute_graph_features(G, user_ids)
    logger.info("  Graph features: %s", df_graph_feats.shape)

    df_feat = pd.DataFrame(features_26, columns=FEAT_NAMES_26)
    df_feat["user_id"] = user_ids
    df_feat["user_id"] = df_feat["user_id"].astype(str)
    df_graph_feats["user_id"] = df_graph_feats.index.astype(str)

    df_final = pd.merge(df_feat, df_graph_feats, on="user_id", how="left")
    df_final = pd.merge(df_final, df_labels[label_columns(df_labels)], on="user_id", how="inner")
    if df_final.empty:
        raise ValueError("No rows after merging extracted features with CSV labels.")

    label_cols = [c for c in ["user_id", "user_type", "name", "username", "is_bad"] if c in df_final.columns]
    feature_cols = [c for c in df_final.columns if c not in label_cols]

    X_df = df_final[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X_df.to_numpy(dtype=float)
    y = df_final["is_bad"].to_numpy(dtype=int)
    logger.info("  Final feature matrix: %s (%s features)", X.shape, len(feature_cols))

    edge_path = os.path.join(save_dir, "enhanced_graph_edges.csv")
    pd.DataFrame(list(G.edges()), columns=["source", "target"]).to_csv(edge_path, index=False)
    logger.info("  Enhanced graph edges saved: %s", edge_path)

    node_feat_path = os.path.join(save_dir, "node_features.csv")
    df_final.to_csv(node_feat_path, index=False, encoding="utf-8")
    logger.info("  Node features saved: %s", node_feat_path)

    print("\n" + "-" * 50)
    print("Classification")
    print("-" * 50)

    xgb_kwargs = _build_xgb_kwargs(y)
    model = xgb.XGBClassifier(**xgb_kwargs)
    y_pred, y_proba = _cross_val_predict(model, X, y)
    y_pred = y_pred.astype(int)

    print("\n" + "=" * 45)
    print("Classification Report")
    print("=" * 45)
    print(classification_report(y, y_pred, target_names=["Human", "Bot"]))
    try:
        auc = roc_auc_score(y, y_proba)
        logger.info("ROC-AUC: %.4f", auc)
    except Exception:
        pass

    final_model = xgb.XGBClassifier(**xgb_kwargs)
    final_model.fit(X, y)

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Human", "Bot"],
        yticklabels=["Human", "Bot"],
        annot_kws={"size": 14},
    )
    plt.title("Bot Detection Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    if shap is not None:
        try:
            logger.info("SHAP explainer running...")
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X)

            plt.figure(figsize=(12, 9))
            shap.summary_plot(
                shap_values,
                pd.DataFrame(X, columns=feature_cols),
                plot_type="dot",
                max_display=15,
                show=False,
            )
            plt.title("SHAP Feature Attribution (Enhanced Graph)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as _shap_exc:
            logger.warning("Global SHAP summary plot failed (non-fatal): %s", _shap_exc)

    # Full-fit model probabilities (for ExplanationPacket)
    prob_bot_full_fit = final_model.predict_proba(X)[:, 1]

    df_result = df_final[[c for c in ["user_id", "user_type", "name", "username", "is_bad"] if c in df_final.columns]].copy()
    df_result["y_pred"] = y_pred
    df_result["prob_bot"] = y_proba            # backward-compat (OOF)
    df_result["prob_bot_oof"] = y_proba        # explicit OOF
    df_result["prob_bot_full_fit"] = prob_bot_full_fit
    for col in feature_cols:
        df_result[col] = df_final[col].values
    result_path = os.path.join(save_dir, "classification_results.csv")
    df_result.to_csv(result_path, index=False, encoding="utf-8")
    logger.info("Results saved: %s", result_path)

    if run_visualizer:
        logger.info("Generating CognitiveVisualizer reports...")
        role_labels = df_final["user_type"].astype(str).to_numpy()
        visualizer = CognitiveVisualizer(X, y, role_labels, feature_cols, save_dir=save_dir)
        visualizer.generate_all_reports(trained_xgb_model=final_model)

    # ==================================================================
    # White-box explainability (M1)
    # ==================================================================
    whitebox_paths: dict = {}
    if whitebox:
        logger.info("=" * 50)
        logger.info("White-box explainability (M1) enabled")
        logger.info("=" * 50)

        try:
            if EvidenceRegistry is None or LocalTreeExplainer is None:
                raise ImportError(
                    "explainability module not importable; check your PYTHONPATH"
                )

            # -- 1. EvidenceRegistry ----------------------------------------
            logger.info("Building EvidenceRegistry ...")
            registry = EvidenceRegistry()
            n_db = registry.register_db(db_file)
            n_csv = 0
            if os.path.isfile(csv_file):
                observable_aliases = (
                    {"user_char": "bio"}
                    if detect_dataset_kind(db_file) == "twibot_static_v5"
                    else None
                )
                n_csv = registry.register_csv(
                    csv_file, observable_aliases=observable_aliases
                )

            # Register per-user text evidence using the SAME splitter the
            # feature extractor uses (split_tweet_pool + max_tweets_per_user).
            from config import split_tweet_pool as _split_tweets
            n_text = 0
            final_user_ids = df_final["user_id"].astype(str).tolist()
            tweets_by_user = dict(zip(user_ids_master, tweets_joined))
            _max_t = max_tweets_per_user
            for uid in final_user_ids:
                tweet_str = tweets_by_user.get(uid, "")
                texts = _split_tweets(tweet_str, min_len=5)
                if _max_t is not None:
                    texts = texts[:_max_t]
                if texts:
                    n_text += registry.register_texts(uid, texts)
            logger.info("  Registered %d DB + %d CSV + %d text evidence rows",
                        n_db, n_csv, n_text)

            # -- 2. Save model artifacts ------------------------------------
            model_dir = os.path.join(save_dir, "model")
            os.makedirs(model_dir, exist_ok=True)

            model_json_path = os.path.join(model_dir, "xgboost_model.json")
            final_model.save_model(model_json_path)
            whitebox_paths["model_json"] = model_json_path
            logger.info("  XGBoost model saved: %s", model_json_path)

            feat_names_path = os.path.join(model_dir, "feature_names.json")
            with open(feat_names_path, "w", encoding="utf-8") as fh:
                json.dump(feature_cols, fh, ensure_ascii=False, indent=2)
            whitebox_paths["feature_names"] = feat_names_path

            meta_path = os.path.join(model_dir, "model_metadata.json")
            metadata = {
                "model_type": "XGBoost",
                "feature_names": feature_cols,
                "n_features": len(feature_cols),
                "threshold": 0.5,
                "dataset": os.path.basename(db_file),
                "random_seed": 42,
                "created_at": pd.Timestamp.now().isoformat(),
                "schema_version": "1.0",
                "xgb_params": {k: v for k, v in xgb_kwargs.items()
                               if not callable(v)},
            }
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)
            whitebox_paths["model_metadata"] = meta_path
            logger.info("  Model metadata saved: %s", meta_path)

            # -- 3. Local explanations --------------------------------------
            expl_dir = os.path.join(save_dir, "explanations")
            os.makedirs(expl_dir, exist_ok=True)

            # Use df_final["user_id"] (post-merge, aligned row-for-row with X).
            explainer_user_ids = df_final["user_id"].astype(str).tolist()
            explainer = LocalTreeExplainer(
                fitted_model=final_model,
                X=X,
                feature_names=feature_cols,
                user_ids=explainer_user_ids,
                threshold=0.5,
            )

            csv_path = os.path.join(expl_dir, "local_contributions.csv")
            explainer.save_csv(csv_path)
            whitebox_paths["local_contributions_csv"] = csv_path

            jsonl_path = os.path.join(expl_dir, "local_explanations.jsonl")
            explainer.save_jsonl(jsonl_path)
            whitebox_paths["local_explanations_jsonl"] = jsonl_path

            # -- 4. Explanation packets -------------------------------------
            packets_dir = os.path.join(expl_dir, "packets")
            os.makedirs(packets_dir, exist_ok=True)

            run_id_str = os.path.basename(save_dir.rstrip(os.sep))
            builder = ExplanationPacketBuilder(
                registry=registry,
                run_id=run_id_str,
                model_version=run_id_str,
                top_k=whitebox_top_k,
            )

            all_explanations = explainer.explain_all()

            # Build OOF probability map for packet metadata.
            oof_map: dict = {}
            for i, uid in enumerate(explainer_user_ids):
                if i < len(y_proba):
                    oof_map[str(uid)] = float(y_proba[i])

            packets = builder.build_all(
                all_explanations,
                provenance=provenance,
                oof_probabilities=oof_map,
            )

            # Per-user JSON files
            for pkt in packets:
                pkt_path = os.path.join(packets_dir, f"{pkt.case_id.replace(':', '_')}.json")
                builder.save_packet(pkt, pkt_path)

            # Combined JSONL
            packets_jsonl = os.path.join(expl_dir, "explanation_packets.jsonl")
            builder.save_packets_jsonl(packets, packets_jsonl)
            whitebox_paths["packets_jsonl"] = packets_jsonl
            whitebox_paths["packets_dir"] = packets_dir

            logger.info("  White-box artifacts saved under: %s", expl_dir)

        except Exception as exc:
            logger.warning(
                "White-box explainability failed: %s. "
                "Original classification results are NOT affected.",
                exc,
            )
            whitebox_paths["_whitebox_error"] = str(exc)

    # ==================================================================
    # Done
    # ==================================================================
    print(f"\n{'=' * 50}")
    print(f"  Script 1 done. Outputs in: {save_dir}")
    print("  Enhanced graph edges -> Script 2 & 3")
    print(f"{'=' * 50}")

    result: dict = {
        "save_dir": save_dir,
        "edge_path": edge_path,
        "node_feature_path": node_feat_path,
        "result_path": result_path,
    }
    if whitebox:
        result["whitebox_paths"] = whitebox_paths
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyper_newtest binary classifier on a supported dataset.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--db", dest="db_file", default=None)
    parser.add_argument("--csv", dest="csv_file", default=None)
    parser.add_argument("--save-dir", dest="save_dir", default=SAVE_DIR)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--sim-threshold", dest="sim_threshold", type=float, default=SIM_THRESHOLD)
    parser.add_argument(
        "--psychology-mode",
        choices=["full", "fast", "off"],
        default=os.getenv("AFG_PSYCHOLOGY_MODE", "full"),
    )
    parser.add_argument("--max-tweets-per-user", type=int, default=None)
    parser.add_argument("--no-visualizer", action="store_true")
    parser.add_argument("--whitebox", action="store_true",
                        help="Enable M1 white-box explainability outputs.")
    parser.add_argument("--whitebox-top-k", type=int, default=15,
                        help="Top-K feature contributions per packet (default 15).")
    return parser.parse_args()


def main():
    args = parse_args()
    db_file, csv_file = resolve_dataset_paths(args.db_file, args.csv_file, args.dataset)
    run_save_dir = make_experiment_dir(
        args.save_dir,
        db_path=db_file,
        csv_path=csv_file,
        dataset=args.dataset,
        run_name=args.run_name,
        prefix="classifier",
    )
    write_manifest(
        run_save_dir,
        script="new_main_classifier.py",
        db_path=db_file,
        csv_path=csv_file,
        dataset=args.dataset or "auto",
        sim_threshold=args.sim_threshold,
        psychology_mode=args.psychology_mode,
        max_tweets_per_user=args.max_tweets_per_user,
        visualizer=not args.no_visualizer,
    )
    return run_classifier(
        db_file=db_file,
        csv_file=csv_file,
        save_dir=run_save_dir,
        sim_threshold=args.sim_threshold,
        psychology_mode=args.psychology_mode,
        max_tweets_per_user=args.max_tweets_per_user,
        run_visualizer=not args.no_visualizer,
        whitebox=args.whitebox,
        whitebox_top_k=args.whitebox_top_k,
    )


if __name__ == "__main__":
    main()
