"""
graph_builder.py
Shared tool: A complete pipeline for building augmented heterogeneous graphs.

Responsibilities: Extract 26-dimensional features → Construct edges using cosine similarity → Merge original edges → Calculate graph features

+ Construct HGT heterogeneous graphs (user/tweet nodes, various directed/undirected edge types)

Does not depend on hetero_hyperrole_classifier; can directly read original graphs from CSV/DB.
"""

import os
import ast
import logging
import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def build_original_graph_from_csv(df):
    
    G = nx.Graph()
    user_ids = df['user_id'].astype(str).tolist()
    G.add_nodes_from(user_ids)

    for _, row in df.iterrows():
        uid = str(row['user_id'])
        try:
            follow_list = ast.literal_eval(str(row.get('following_list', '[]')))
            for target in follow_list:
                target = str(target)
                if target in G:
                    G.add_edge(uid, target)
        except Exception:
            pass

    return G


def build_original_graph_from_db(db_path):
  
    G = nx.Graph()
    if not os.path.exists(db_path):
        return G
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='follow'")
        if cursor.fetchone():
            df = pd.read_sql_query("SELECT follower_id, followee_id AS following_id FROM follow", conn)
            for _, row in df.iterrows():
                follower = _normalize_db_id(row['follower_id'])
                followee = _normalize_db_id(row['following_id'])
                if follower is not None and followee is not None:
                    G.add_edge(follower, followee)
        conn.close()
    except Exception:
        pass
    return G


def get_original_graph(df, db_path):
    
    G = build_original_graph_from_csv(df)
    if G.number_of_edges() == 0:
        G = build_original_graph_from_db(db_path)
    return G


def build_26dim_features(fused_matrix, n_semantic=8):
   
    total_dim = fused_matrix.shape[1]
    semantic_raw = fused_matrix[:, :-18]  

    if semantic_raw.shape[1] > n_semantic:
        pca = PCA(n_components=n_semantic, random_state=42)
        semantic_reduced = pca.fit_transform(semantic_raw)
    else:
        semantic_reduced = semantic_raw

    behavior = fused_matrix[:, -18:-8] 
    psycho = fused_matrix[:, -8:]       
    return np.hstack([semantic_reduced, behavior, psycho])


def build_knn_edges(features, k=10):
    
    n = features.shape[0]
    n_neighbors = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn.fit(features)
    _, indices = nn.kneighbors(features)

    edges = set()
    for i in range(n):
        for j in indices[i, 1:]:  
            if i != j:
                edges.add((min(int(i), int(j)), max(int(i), int(j))))
    return list(edges)


def add_knn_edges_to_graph(G, knn_edges, node_list):
    
    for i, j in knn_edges:
        G.add_edge(node_list[i], node_list[j])
    return G


def compute_graph_features(G, node_list):
   
    features = {}

    deg = dict(G.degree())
    features['graph_degree'] = [deg.get(n, 0) for n in node_list]

    deg_cent = nx.degree_centrality(G)
    features['graph_deg_centrality'] = [deg_cent.get(n, 0) for n in node_list]

    clustering = nx.clustering(G)
    features['graph_clustering'] = [clustering.get(n, 0) for n in node_list]

    try:
        pr = nx.pagerank(G)
    except Exception:
        pr = {n: 0 for n in node_list}
    features['graph_pagerank'] = [pr.get(n, 0) for n in node_list]

    try:
        ev = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
    except Exception:
        ev = {n: 0 for n in node_list}
    features['graph_eigenvector'] = [ev.get(n, 0) for n in node_list]

    try:
        kc = nx.core_number(G)
    except Exception:
        kc = {n: 0 for n in node_list}
    features['graph_k_core'] = [kc.get(n, 0) for n in node_list]

    return pd.DataFrame(features, index=node_list)



def build_cosine_edges(features, threshold=0.7):
    """
   Edges are constructed based on cosine similarity of 26-dimensional features; similarity > threshold → undirected edges.

    Features: (N, D) numpy array

    Returns: list of (i, j) tuples, 0-indexed, automatically deduplicated and skipping self-loops.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    n = features.shape[0]
    sim_matrix = cosine_similarity(features)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                edges.add((int(i), int(j)))
    return list(edges)


def _normalize_db_id(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def build_hetero_data(user_ids, features_26, db_path, threshold=0.7):
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError("torch_geometric is required: pip install torch_geometric")

    data = HeteroData()
    user_map = {str(uid): i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    data['user'].x = torch.tensor(features_26.astype(np.float32))
    cos_edges = build_cosine_edges(features_26, threshold)
    if cos_edges:
        src = [e[0] for e in cos_edges]
        dst = [e[1] for e in cos_edges]
        data['user', 'similar', 'user'].edge_index = torch.tensor(
            [src + dst, dst + src], dtype=torch.long
        )
        logger.info(f"  [similar] {len(cos_edges)} undirected edges → {len(src)*2} directed edges")

    if not os.path.exists(db_path):
        logger.warning(f"  DB does not exist: {db_path}, only containing cosine similarity edges")
        return data, {v: k for k, v in user_map.items()}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {t[0] for t in cursor.fetchall()}

    if 'follow' in tables:
        try:
            df_f = pd.read_sql_query(
                "SELECT follower_id, followee_id FROM follow", conn
            )
            src, dst = [], []
            for _, r in df_f.iterrows():
                s = _normalize_db_id(r['follower_id'])
                d = _normalize_db_id(r['followee_id'])
                if s in user_map and d in user_map:
                    src.append(user_map[s])
                    dst.append(user_map[d])
            if src:
                data['user', 'follows', 'user'].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )
                logger.info(f"  [follows] {len(src)} directed edges")
        except Exception as e:
            logger.warning(f"  [follows] Failed to load: {e}")

    tweet_map = {}
    if 'post' in tables:
        try:
            df_posts = pd.read_sql_query(
                "SELECT post_id, user_id, original_post_id FROM post", conn
            )
            tweet_ids = [
                pid for pid in (_normalize_db_id(v) for v in df_posts['post_id'])
                if pid is not None
            ]
            tweet_ids = list(dict.fromkeys(tweet_ids))
            tweet_map = {tid: i for i, tid in enumerate(tweet_ids)}
            n_tweets = len(tweet_map)
            data['tweet'].x = torch.randn(
                n_tweets, features_26.shape[1], dtype=torch.float
            )
            logger.info(f"  tweet nodes: {n_tweets}")

            src, dst = [], []
            for _, r in df_posts.iterrows():
                uid = _normalize_db_id(r['user_id'])
                pid = _normalize_db_id(r['post_id'])
                if uid in user_map and pid in tweet_map:
                    src.append(user_map[uid])
                    dst.append(tweet_map[pid])
            if src:
                data['user', 'posts', 'tweet'].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )
                logger.info(f"  [posts] {len(src)} directed edges")

            df_rt = df_posts[df_posts['original_post_id'].notna()]
            if len(df_rt) > 0:
                rt_src, rt_dst = [], []
                for _, r in df_rt.iterrows():
                    uid = _normalize_db_id(r['user_id'])
                    orig_pid = _normalize_db_id(r['original_post_id'])
                    if uid in user_map and orig_pid in tweet_map:
                        rt_src.append(user_map[uid])
                        rt_dst.append(tweet_map[orig_pid])
                if rt_src:
                    data['user', 'retweets', 'tweet'].edge_index = torch.tensor(
                        [rt_src, rt_dst], dtype=torch.long
                    )
                    logger.info(f"  [retweets] {len(rt_src)} directed edges")
        except Exception as e:
            logger.warning(f"  tweet/post construction failed: {e}")

    if 'like' in tables and tweet_map:
        try:
            df_likes = pd.read_sql_query(
                "SELECT user_id, post_id FROM \"like\"", conn
            )
            src, dst = [], []
            for _, r in df_likes.iterrows():
                uid = _normalize_db_id(r['user_id'])
                pid = _normalize_db_id(r['post_id'])
                if uid in user_map and pid in tweet_map:
                    src.append(user_map[uid])
                    dst.append(tweet_map[pid])
            if src:
                data['user', 'likes', 'tweet'].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )
                logger.info(f"  [likes] {len(src)} directed edges")
        except Exception as e:
            logger.warning(f"  [likes] Failed to load: {e}")

    if 'comment' in tables and tweet_map:
        try:
            df_com = pd.read_sql_query(
                "SELECT user_id, post_id FROM comment", conn
            )
            src, dst = [], []
            for _, r in df_com.iterrows():
                uid = _normalize_db_id(r['user_id'])
                pid = _normalize_db_id(r['post_id'])
                if uid in user_map and pid in tweet_map:
                    src.append(user_map[uid])
                    dst.append(tweet_map[pid])
            if src:
                data['user', 'comments', 'tweet'].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )
                logger.info(f"  [comments] {len(src)} directed edges")
        except Exception as e:
            logger.warning(f"  [comments] Failed to load: {e}")

    conn.close()
    rev_user_map = {v: k for k, v in user_map.items()}
    logger.info(
        f"  Heterogeneous graph: user={n_users}, tweet={len(tweet_map)}, "
        f"number of edge types={len(data.edge_types)}"
    )
    return data, rev_user_map