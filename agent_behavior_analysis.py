"""
agent_behavior_analysis.py

读取仿真 DB + CSV，提取四维行为特征，输出可视化和数值报表。

用法:
  cd Multi-agent-fraud-game-detection
  python agent_behavior_analysis.py

输出:
  behavior_analysis/
    ├── feature_table.csv          ← 每个 agent 一行，全部数值特征
    ├── action_counts.png          ← 各类行为柱状图
    ├── pca_behavior.png           ← PCA 降维可视化
    ├── network_graph.png          ← 关注有向图（按 type 着色）
    └── feature_importance.png     ← 区分度最高的特征排名
"""

import sqlite3
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ========== 配置 ==========
DB_PATH = "data/test_72.db"
CSV_PATH = "data/72agent_deeppersonal.csv"
OUTPUT_DIR = "behavior_analysis"
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Agent Behavior Analysis")
print("=" * 60)

# ── 1. 读 CSV 获取 user_type ──────────────────────────────────
df_csv = pd.read_csv(CSV_PATH)
user_type_map = dict(zip(df_csv["user_id"], df_csv["user_type"]))
print(f"\n[1] CSV loaded: {len(df_csv)} agents")
print(f"    user_type distribution: {df_csv['user_type'].value_counts().to_dict()}")

# ── 2. 读 DB ──────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 2a. User 表
cur.execute("SELECT user_id, agent_id, user_name FROM user ORDER BY user_id")
user_rows = cur.fetchall()
user_ids = [r[0] for r in user_rows]
print(f"\n[2] DB loaded: {len(user_rows)} users")

# 2b. Follow 表 → 有向图
cur.execute("SELECT follower_id, followee_id FROM follow")
follow_edges = cur.fetchall()
print(f"    follows: {len(follow_edges)} edges")

# 2c. Like 表
cur.execute("SELECT user_id, post_id FROM like")
like_rows = cur.fetchall()
print(f"    likes: {len(like_rows)}")

# 2d. Post 表
cur.execute("SELECT post_id, user_id, content FROM post")
post_rows = cur.fetchall()
post_author_map = {p[0]: p[1] for p in post_rows}  # post_id → author_id
print(f"    posts: {len(post_rows)}")

# 2e. Comment 表
cur.execute("SELECT comment_id, post_id, user_id FROM comment")
comment_rows = cur.fetchall()
print(f"    comments: {len(comment_rows)}")

# 2f. Trace 表（提取动作类型）
cur.execute("SELECT user_id, action FROM trace")
trace_rows = cur.fetchall()
trace_action_counts = defaultdict(Counter)
for uid, action in trace_rows:
    trace_action_counts[uid][action] += 1
print(f"    traces: {len(trace_rows)}")

conn.close()

# ── 3. 构建特征表 ────────────────────────────────────────────

# 3a. 关注网络图特征
G = nx.DiGraph()
G.add_nodes_from(user_ids)
G.add_edges_from(follow_edges)

print("\n[3] Computing features...")
pagerank = nx.pagerank(G, alpha=0.85)
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)

features = []
for uid in user_ids:
    user_type = user_type_map.get(uid, "unknown")

    # 基础活动量
    post_count = sum(1 for p in post_rows if p[1] == uid)
    comment_count = sum(1 for c in comment_rows if c[2] == uid)

    # 收到的点赞数（别人点了你的帖子）
    received_likes = sum(1 for l_uid, l_pid in like_rows
                         if post_author_map.get(l_pid) == uid)
    # 给出的点赞数
    given_likes = sum(1 for l_uid, l_pid in like_rows if l_uid == uid)

    # 从 trace 里统计动作
    tac = trace_action_counts.get(uid, Counter())

    # 有向图特征
    out_deg = G.out_degree(uid) if uid in G else 0
    in_deg = G.in_degree(uid) if uid in G else 0

    features.append({
        "agent_id": uid,
        "user_type": user_type,

        # 活动量
        "post_count": post_count,
        "comment_count": comment_count,
        "given_likes": given_likes,
        "received_likes": received_likes,

        # Trace 动作
        "trace_refresh": tac.get("refresh", 0),
        "trace_like_post": tac.get("like_post", 0),
        "trace_repost": tac.get("repost", 0),
        "trace_create_comment": tac.get("create_comment", 0),
        "trace_follow": tac.get("follow", 0),

        # 有向图
        "out_degree": out_deg,
        "in_degree": in_deg,
        "pagerank": round(pagerank.get(uid, 0), 6),
        "betweenness": round(betweenness.get(uid, 0), 6),
        "clustering": round(clustering.get(uid, 0), 6),

        # 比例
        "out_in_ratio": round(out_deg / max(in_deg, 1), 4),
    })

df_feat = pd.DataFrame(features)
df_feat.to_csv(os.path.join(OUTPUT_DIR, "feature_table.csv"), index=False)
print(f"    Feature table: {df_feat.shape[0]} rows, {df_feat.shape[1]} cols")
print(f"    Saved to {OUTPUT_DIR}/feature_table.csv")

# ── 4. 数值摘要 ──────────────────────────────────────────────
print("\n[4] Numerical summary by user_type:")
numeric_cols = [c for c in df_feat.columns if c not in ("agent_id", "user_type")]
summary = df_feat.groupby("user_type")[numeric_cols].mean().round(3)
print(summary.to_string())

# ── 5. 可视化 ────────────────────────────────────────────────
print("\n[5] Generating visualizations...")

COLOR_MAP = {"good": "#2196F3", "bad": "#F44336",
             "bad_leader": "#FF9800", "bad_member": "#9C27B0"}
TYPE_ORDER = ["good", "bad", "bad_leader", "bad_member"]

# 5a. 行为柱状图
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
plot_cols = ["post_count", "comment_count", "given_likes",
             "received_likes", "out_degree", "in_degree"]
plot_titles = ["Posts per Agent", "Comments per Agent",
               "Likes Given", "Likes Received",
               "Following (out-degree)", "Followers (in-degree)"]

for ax, col, title in zip(axes.flat, plot_cols, plot_titles):
    for i, utype in enumerate(TYPE_ORDER):
        subset = df_feat[df_feat["user_type"] == utype][col]
        if len(subset) == 0:
            continue
        color = COLOR_MAP.get(utype, "#888")
        bp = ax.boxplot(subset, positions=[i], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color="black"))
    ax.set_xticks(range(len(TYPE_ORDER)))
    ax.set_xticklabels(TYPE_ORDER, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Count")

plt.suptitle("Behavioral Features by Agent Type", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "action_counts.png"), dpi=120)
plt.close()
print("    action_counts.png")

# 5b. PCA 降维
feat_for_pca = [c for c in numeric_cols if c not in ("out_in_ratio",)]
X = df_feat[feat_for_pca].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
for utype in TYPE_ORDER:
    mask = df_feat["user_type"] == utype
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLOR_MAP.get(utype, "#888"), label=utype,
               alpha=0.7, s=40, edgecolors="black", linewidth=0.3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA: Agent Behavior Embedding")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_behavior.png"), dpi=120)
plt.close()
print("    pca_behavior.png")

# 5c. 关注网络图（取子集防止太密）
fig, ax = plt.subplots(figsize=(14, 12))
# 取入度 top-30 的节点
in_deg_sorted = sorted(G.in_degree(), key=lambda x: -x[1])
top_nodes = [n for n, d in in_deg_sorted[:40]]
subG = G.subgraph(top_nodes)
pos = nx.spring_layout(subG, k=0.5, iterations=50)
node_colors = [COLOR_MAP.get(user_type_map.get(n, "unknown"), "#888")
               for n in subG.nodes()]
node_sizes = [300 + 10 * subG.out_degree(n) for n in subG.nodes()]
nx.draw(subG, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edge_color="#ccc", width=0.5, with_labels=False, arrows=True,
        arrowsize=10)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLOR_MAP[t], label=t) for t in TYPE_ORDER
                   if t in df_feat["user_type"].values]
ax.legend(handles=legend_elements, loc="upper right")
ax.set_title("Follow Network (top-40 by followers)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "network_graph.png"), dpi=120)
plt.close()
print("    network_graph.png")

# 5d. 特征区分度排名（用 ANOVA F-score）
from sklearn.feature_selection import f_classif
y = df_feat["user_type"].astype("category").cat.codes
f_scores, p_values = f_classif(X_scaled, y)
fi_df = pd.DataFrame({"feature": feat_for_pca, "f_score": f_scores,
                       "p_value": p_values})
fi_df = fi_df.sort_values("f_score", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(fi_df)), fi_df["f_score"].values, color="#4CAF50", alpha=0.7)
ax.set_yticks(range(len(fi_df)))
ax.set_yticklabels(fi_df["feature"].values)
ax.invert_yaxis()
ax.set_xlabel("ANOVA F-score")
ax.set_title("Feature Discriminative Power (higher = better separates types)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
plt.close()
print("    feature_importance.png")

print(f"\n[OK] All outputs saved to {OUTPUT_DIR}/")
print("Files:")
for f in Path(OUTPUT_DIR).iterdir():
    print(f"  {f.name}")
