"""
Script 2: unsupervised bot gang community detection.

Input files are produced by Script 1:
- enhanced_graph_edges.csv
- classification_results.csv
"""

import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import PROJECT_ROOT, configure_utf8_streams


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
configure_utf8_streams()

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "new_result" / "hyper_newtest")
EDGE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "enhanced_graph_edges.csv")
RESULT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "classification_results.csv")
SAVE_DIR = DEFAULT_OUTPUT_DIR

ENGINE_COLS = [
    "Empathy_Gap_Mean", "Empathy_Gap_Max",
    "Dark_Triad_Mean", "Dark_Triad_Max",
    "Contagion_Mean", "Contagion_Max",
    "Volatility_Mean", "Volatility_Max",
]


def run_gang_detection(edge_path=EDGE_PATH, result_path=RESULT_PATH, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Script 2: bot gang community detection")
    print("=" * 60)

    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Please run Script 1 first: {edge_path}")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Please run Script 1 first: {result_path}")

    logger.info("Loading enhanced graph edges...")
    df_edges = pd.read_csv(edge_path)
    G = nx.Graph()
    G.add_edges_from([(str(u), str(v)) for u, v in df_edges[["source", "target"]].values])
    logger.info("  Graph: %s nodes, %s edges", G.number_of_nodes(), G.number_of_edges())

    logger.info("Loading classification results...")
    df_result = pd.read_csv(result_path)
    logger.info("  Total: %s nodes, bots: %s", len(df_result), int(df_result["is_bad"].sum()))

    bot_ids = set(df_result[df_result["is_bad"] == 1]["user_id"].astype(str))
    bot_nodes = [node for node in G.nodes() if node in bot_ids]
    G_bot = G.subgraph(bot_nodes).copy()
    logger.info("  Bot subgraph: %s nodes, %s edges", G_bot.number_of_nodes(), G_bot.number_of_edges())

    if G_bot.number_of_nodes() < 3:
        logger.warning("Too few bot nodes for community detection.")
        return None

    G_bot.remove_nodes_from(list(nx.isolates(G_bot)))
    logger.info("  After removing isolates: %s nodes", G_bot.number_of_nodes())

    if G_bot.number_of_nodes() < 3:
        logger.warning("Too few bot nodes after removing isolates.")
        return None

    logger.info("Running community detection...")
    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(G_bot, seed=42)
        method = "Louvain"
    except (ImportError, AttributeError):
        from networkx.algorithms.community import greedy_modularity_communities

        communities = greedy_modularity_communities(G_bot)
        method = "Greedy Modularity"

    logger.info("  Method: %s, found %s communities", method, len(communities))

    gang_id_map = {}
    for gid, members in enumerate(communities):
        for node in members:
            gang_id_map[node] = gid

    df_bot = df_result[df_result["user_id"].astype(str).isin(bot_nodes)].copy()
    df_bot["gang_id"] = df_bot["user_id"].astype(str).map(gang_id_map).fillna(-1).astype(int)

    gang_counts = df_bot["gang_id"].value_counts().sort_index()
    logger.info("Gang distribution: %s", gang_counts.to_dict())

    psycho_cols = [col for col in ENGINE_COLS if col in df_bot.columns]
    if psycho_cols:
        print("\n" + "=" * 60)
        print("  Gang Psychological Profiles (mean)")
        print("=" * 60)
        profile = df_bot[df_bot["gang_id"] >= 0].groupby("gang_id")[psycho_cols].mean()
        print(profile.round(4).to_string())

        profile_path = os.path.join(save_dir, "gang_profiles.csv")
        profile.to_csv(profile_path)
        logger.info("Gang profiles saved: %s", profile_path)

    gang_cols = ["user_id", "user_type", "is_bad", "gang_id"]
    if "name" in df_bot.columns:
        gang_cols.insert(1, "name")
    if "username" in df_bot.columns:
        gang_cols.insert(2, "username")

    out = df_bot[[col for col in gang_cols if col in df_bot.columns]].sort_values(["gang_id", "user_id"])
    gang_path = os.path.join(save_dir, "gang_results.csv")
    out.to_csv(gang_path, index=False, encoding="utf-8")
    logger.info("Gang assignments saved: %s", gang_path)

    gang_edge_path = os.path.join(save_dir, "bot_gang_edges.csv")
    pd.DataFrame(list(G_bot.edges()), columns=["source", "target"]).to_csv(gang_edge_path, index=False)
    logger.info("Bot subgraph edges saved: %s", gang_edge_path)

    if len(psycho_cols) >= 3:
        X_bot = df_bot[df_bot["gang_id"] >= 0][psycho_cols].values.astype(float)
        y_gang = df_bot[df_bot["gang_id"] >= 0]["gang_id"].values

        if X_bot.shape[0] >= 3:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_bot)

            plt.figure(figsize=(9, 7))
            scatter = plt.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=y_gang,
                cmap="tab10",
                s=70,
                alpha=0.7,
                edgecolors="k",
                linewidth=0.5,
            )
            plt.colorbar(scatter, label="Gang ID")
            plt.title(f"Bot Gang Detection ({method}, {len(communities)} communities)", fontsize=13, fontweight="bold")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "gang_scatter.png"), dpi=300)
            plt.close()
            logger.info("Gang scatter saved.")

    print(f"\n{'=' * 50}")
    print(f"  Script 2 done. {len(communities)} gangs detected.")
    print(f"  Outputs in: {save_dir}")
    print(f"{'=' * 50}")

    return {
        "gang_results": gang_path,
        "gang_edges": gang_edge_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run bot gang detection from Script 1 outputs.")
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--edges", default=None)
    parser.add_argument("--results", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    edge_path = args.edges or os.path.join(args.save_dir, "enhanced_graph_edges.csv")
    result_path = args.results or os.path.join(args.save_dir, "classification_results.csv")
    return run_gang_detection(edge_path=edge_path, result_path=result_path, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
