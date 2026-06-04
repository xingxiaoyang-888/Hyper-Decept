"""
new_role_assigner.py [Script 3]

HGT Heterogeneous Graph + Poincaré Hyperbolic Projection + DPMM Role Discovery

Pipeline:

1. Read 26-dimensional features + augmented graph from Script 1 output

2. Construct PyG heterogeneous graph (user/tweet nodes, followers/posts/retweets/similar/likes/comments edges)

3. Learn structured embeddings end-to-end using HGT

4. Preserve hierarchical radiation using Poincaré spherical projection

5. Automatically infer the number of roles using DPMM nonparametric clustering

6. Map role semantics using hyperbolic radius + visualization

Dependency: node_features.csv from Script 1 output

Reference: detection_module/hetero_hyperrole_classifier.py

Usage:
python -m detection_module.hyper_newtest.new_role_assigner --save-dir <script1_output_dir>
"""

import os
import sys
import argparse
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DATASET_CHOICES,
    PROJECT_ROOT,
    configure_utf8_streams,
    resolve_dataset_paths,
)
from graph_builder import build_hetero_data
from visualizer import CognitiveVisualizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
configure_utf8_streams()

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "new_result" / "hyper_newtest")
DB_PATH, _CSV_PATH = resolve_dataset_paths()
NODE_FEAT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "node_features.csv")
SAVE_DIR = DEFAULT_OUTPUT_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
EPOCHS = 200
SIM_THRESHOLD = 0.7
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NEG_SAMPLES = 5   
MARGIN = 3.0     


FEAT_26_COLS = [
    'Semantic_0', 'Semantic_1', 'Semantic_2', 'Semantic_3',
    'Semantic_4', 'Semantic_5', 'Semantic_6', 'Semantic_7',
    'Follower_Following_Ratio', 'Action_Frequency', 'Like_Ratio',
    'Retweet_Ratio', 'Reply_Ratio', 'Temporal_Entropy',
    'URL_Ratio', 'Mention_Ratio', 'Hashtag_Ratio', 'Media_Ratio',
    'Empathy_Gap_Mean', 'Empathy_Gap_Max',
    'Dark_Triad_Mean', 'Dark_Triad_Max',
    'Contagion_Mean', 'Contagion_Max',
    'Volatility_Mean', 'Volatility_Max',
]



class HyperRoleHGNN(torch.nn.Module):


    def __init__(self, hidden_dim, num_heads, num_layers, metadata):
        super().__init__()
        from torch_geometric.nn import HGTConv, Linear

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_dim, hidden_dim, metadata, num_heads)
            self.convs.append(conv)

        self.out_lin = Linear(hidden_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        x_aligned = {
            nt: self.lin_dict[nt](x).relu_() for nt, x in x_dict.items()
        }
        for conv in self.convs:
            x_aligned = conv(x_aligned, edge_index_dict)
        user_emb = self.out_lin(x_aligned["user"])
        hyp_emb = poincare_proj(user_emb)
        return hyp_emb




def poincare_proj(x):
  
    norm = torch.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp_min(norm, 1e-6)  # 防止除0
    return x / (1.0 + norm)

def poincare_distance(u, v, eps=1e-5):
   
    u_norm_sq = torch.sum(u ** 2, dim=-1, keepdim=True)
    v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True)
    diff = u - v
    diff_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)
    numerator = 2 * diff_sq
    denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
    arg = 1 + numerator / (denominator + eps)
    arg = torch.clamp(arg, min=1.0 + eps)
    return torch.acosh(arg)




def train_hgt(data, epochs=EPOCHS):
    """
    Poincaré distance loss + negative sampling is used to train the HGT.

Simultaneously, follows + similar edges are used as positive samples (to bring them closer),

unrelated node pairs are negatively sampled (to push them apart) to prevent embedding collapse.
    """
    logger.info(f"Training HGT (device={DEVICE})...")
    data = data.to(DEVICE)

    model = HyperRoleHGNN(
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, metadata=data.metadata()
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

  
    pos_edges = []
    for etype in [("user", "follows", "user"), ("user", "similar", "user")]:
        if etype in data.edge_types:
            ei = data[etype].edge_index.cpu().numpy()
            pos_edges.append(ei)
            logger.info(f"  Positive signal: {etype} ({ei.shape[1]} edges)")

    if not pos_edges:
        raise RuntimeError("No user-user edge type available for training")

    pos_edges = np.concatenate(pos_edges, axis=1)
   
    pos_edges = np.unique(pos_edges, axis=1)
    logger.info(f"  Total unique positive edges: {pos_edges.shape[1]}")

    n_users = data["user"].x.shape[0]


    deg = np.zeros(n_users, dtype=np.float32)
    for etype in [("user", "follows", "user"), ("user", "similar", "user")]:
        if etype in data.edge_types:
            ei = data[etype].edge_index.cpu().numpy()
            for idx in range(ei.shape[1]):
                deg[ei[0, idx]] += 1
                deg[ei[1, idx]] += 1
    deg_pow = deg ** 0.75
    neg_dist = deg_pow / (deg_pow.sum() + 1e-10)

    n_pos = pos_edges.shape[1]
    pos_t = torch.tensor(pos_edges, dtype=torch.long, device=DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # The distance is then calculated after projecting onto the Poincaré sphere (gradient flow through the unprojected user_emb).
        hyp = model(data.x_dict, data.edge_index_dict)

        # Positive Sample Loss: Closing the Poincaré Distance
        u = hyp[pos_t[0]]
        v = hyp[pos_t[1]]
        pos_dist = poincare_distance(u, v)
        pos_loss = torch.nn.functional.softplus(pos_dist).mean()

        #Negative sampling loss: Hinge loss, no longer penalizes distances exceeding MARGIN.
        neg_i = np.random.choice(n_users, size=(n_pos, NEG_SAMPLES), p=neg_dist)
        neg_j = np.random.choice(n_users, size=(n_pos, NEG_SAMPLES), p=neg_dist)
        neg_i_t = torch.tensor(neg_i, dtype=torch.long, device=DEVICE)
        neg_j_t = torch.tensor(neg_j, dtype=torch.long, device=DEVICE)

        u_neg = hyp[neg_i_t]
        v_neg = hyp[neg_j_t]
        neg_dist_val = poincare_distance(u_neg, v_neg)
        neg_loss = torch.relu(MARGIN - neg_dist_val).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"    Epoch {epoch+1:03d} | loss={loss.item():.4f} "
                f"| pos={pos_loss.item():.4f} | neg={neg_loss.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        final_hyp = model(data.x_dict, data.edge_index_dict).cpu().numpy()
    return model, final_hyp



def dpmm_role_discovery(hyp_emb, rev_user_map):
    """
   The Dirichlet process hybrid model automatically infers the number of roles.

   Role semantics: Smaller hyperbolic radius → Closer to the hierarchy center → Leader / Hub
    """
    logger.info("DPMM role discovery...")
    dpgmm = BayesianGaussianMixture(
        n_components=8,
        weight_concentration_prior_type="dirichlet_process",
        max_iter=500,
        random_state=SEED,
    )
    clusters = dpgmm.fit_predict(hyp_emb)
    n_active = len(np.unique(clusters))
    logger.info(f"  DPMM Valid Character Count: {n_active}")

    radii = np.linalg.norm(hyp_emb, axis=1)

    df = pd.DataFrame({
        "user_idx": np.arange(len(hyp_emb)),
        "user_id": [rev_user_map[i] for i in range(len(hyp_emb))],
        "cluster": clusters,
        "poincare_radius": radii,
    })

   
    cluster_radius_mean = df.groupby("cluster")["poincare_radius"].mean().sort_values()
    role_names = [
        "Opinion Leader",
        "Information Bridge",
        "Amplifier",
        "Community Builder",
        "Peripheral",
        "Fringe Node",
        "Boundary Node",
        "Outsider",
    ]
    role_map = {}
    for idx, (cid, _) in enumerate(cluster_radius_mean.items()):
        role_map[cid] = role_names[idx] if idx < len(role_names) else f"Role_{idx}"
    df["role"] = df["cluster"].map(role_map)

    return df, hyp_emb, dpgmm


def plot_poincare_disk(emb, df):
   
    logger.info("Plotting Poincaré disk...")
    pca = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(emb)
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    proj = proj / np.maximum(norms, 1.0)

    df["x"], df["y"] = proj[:, 0], proj[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, color="gray", lw=2, ls="--"))
    palette = sns.color_palette("husl", df["role"].nunique())
    sns.scatterplot(
        data=df, x="x", y="y", hue="role", palette=palette,
        s=50, alpha=0.8, edgecolor="w"
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title("Poincaré Disk of HyperRole Embeddings", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "poincare_disk.png"), dpi=300)
    plt.close()


def plot_radius_boxplot(df):
  
    logger.info("Plotting radius distribution...")
    order = [
        "Opinion Leader", "Information Bridge", "Amplifier",
        "Community Builder", "Peripheral",
        "Fringe Node", "Boundary Node", "Outsider",
    ]
    order = [r for r in order if r in df["role"].unique()]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="role", y="poincare_radius", order=order, palette="viridis")
    plt.title("Hyperbolic Radius by Role (smaller = more central)", fontsize=14)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "radius_distribution.png"), dpi=300)
    plt.close()


def plot_dpmm_weights(dpgmm):
   
    logger.info("Plotting DPMM weights...")
    weights = np.sort(dpgmm.weights_)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(weights)), weights, color="teal", alpha=0.7)
    plt.axvline(
        x=(dpgmm.weights_ > 0.01).sum() - 0.5,
        color="red", linestyle="--", label="Effective K"
    )
    plt.title("DPMM Dirichlet Process Cluster Weights", fontsize=14)
    plt.xlabel("Component index")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "dpmm_weights.png"), dpi=300)
    plt.close()


def generate_tactical_role_visualizer(node_feat_path, df_roles, save_dir):
    """Use discovered HyperRole labels for CognitiveVisualizer reports."""
    if not os.path.exists(node_feat_path):
        logger.warning("Skip tactical-role visualizer; missing node features: %s", node_feat_path)
        return

    try:
        df_feat = pd.read_csv(node_feat_path)
        df_feat["user_id"] = df_feat["user_id"].astype(str)
        df_roles = df_roles.copy()
        df_roles["user_id"] = df_roles["user_id"].astype(str)

        role_col = "role" if "role" in df_roles.columns else "Tactical_Role"
        if role_col not in df_roles.columns:
            logger.warning("Skip tactical-role visualizer; role column not found.")
            return

        merged = df_feat.merge(df_roles[["user_id", role_col]], on="user_id", how="inner")
        if merged.empty:
            logger.warning("Skip tactical-role visualizer; role merge produced zero rows.")
            return

        label_cols = {"user_id", "user_type", "name", "username", "is_bad", role_col}
        feature_cols = [c for c in merged.columns if c not in label_cols]
        X_df = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_true = (
            merged["is_bad"].fillna(0).astype(int).to_numpy()
            if "is_bad" in merged.columns
            else np.zeros(len(merged), dtype=int)
        )
        y_role = merged[role_col].astype(str).to_numpy()

        logger.info("Generating CognitiveVisualizer reports with discovered tactical roles...")
        visualizer = CognitiveVisualizer(
            X_df.to_numpy(dtype=float),
            y_true,
            y_role,
            feature_cols,
            save_dir=save_dir,
        )
        visualizer.generate_all_reports(trained_xgb_model=None)
    except Exception as exc:
        logger.warning("Tactical-role visualizer failed and was skipped: %s", exc)


def main(db_path=DB_PATH, node_feat_path=NODE_FEAT_PATH, save_dir=SAVE_DIR, epochs=EPOCHS):
    global SAVE_DIR
    SAVE_DIR = save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Script 3: HGT + Poincaré + DPMM 角色发现")
    print("=" * 60)

    if not os.path.exists(node_feat_path):
        raise FileNotFoundError(f"请先运行 Script 1: {node_feat_path}")

    logger.info("Loading 26-dim features...")
    df_feat = pd.read_csv(node_feat_path)
    df_feat["user_id"] = df_feat["user_id"].astype(str)
    user_ids = df_feat["user_id"].tolist()

    avail_cols = [c for c in FEAT_26_COLS if c in df_feat.columns]
    if len(avail_cols) < 26:
        logger.warning(f"Only {len(avail_cols)}/26 feature columns available")
    features_26 = df_feat[avail_cols].values.astype(np.float32)
    logger.info(f"  {len(user_ids)} users, {features_26.shape[1]}-dim features")

    logger.info("Building heterogeneous graph...")
    try:
        data, rev_user_map = build_hetero_data(
            user_ids, features_26, db_path, threshold=SIM_THRESHOLD
        )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise
    logger.info(f"  Edge types: {data.edge_types}")

   
    _, hyp_emb = train_hgt(data, epochs=epochs)
    if hyp_emb is None:
        raise RuntimeError("HGT training failed.")

   
    df_roles, hyp_emb_arr, dpgmm_model = dpmm_role_discovery(hyp_emb, rev_user_map)
    result_path = os.path.join(save_dir, "classification_results.csv")
    if os.path.exists(result_path):
        df_result = pd.read_csv(result_path)
        df_result["user_id"] = df_result["user_id"].astype(str)
        df_roles = df_roles.merge(
            df_result[["user_id", "user_type", "is_bad"]],
            on="user_id", how="left"
        )
        logger.info("  Merged classification labels.")

   
    role_path = os.path.join(save_dir, "role_assignments.csv")
    df_roles.to_csv(role_path, index=False, encoding="utf-8")
    logger.info(f"Role assignments saved: {role_path}")

   
    print("\n" + "=" * 60)
    print("  Role Distribution")
    print("=" * 60)
    print(df_roles["role"].value_counts().to_string())

    
    if "user_type" in df_roles.columns:
        print("\n  Role × user_type cross-tab:")
        print(pd.crosstab(df_roles["role"], df_roles["user_type"]).to_string())

    
    plot_poincare_disk(hyp_emb_arr, df_roles)
    plot_radius_boxplot(df_roles)
    plot_dpmm_weights(dpgmm_model)
    generate_tactical_role_visualizer(node_feat_path, df_roles, save_dir)

    print(f"\n{'=' * 50}")
    print(f"  Script 3 done. Outputs in: {save_dir}")
    print(f"{'=' * 50}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run HGT + Poincare + DPMM role assignment.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--db", dest="db_file", default=None)
    parser.add_argument("--save-dir", dest="save_dir", default=SAVE_DIR)
    parser.add_argument("--node-features", dest="node_features", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_file, _csv_file = resolve_dataset_paths(args.db_file, None, args.dataset)
    node_features = args.node_features or os.path.join(args.save_dir, "node_features.csv")
    main(db_path=db_file, node_feat_path=node_features, save_dir=args.save_dir, epochs=args.epochs)
