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
import re
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


POST_OBSERVED_FEATURES = [
    "log_char_count",
    "log_token_count",
    "uppercase_ratio",
    "punctuation_ratio",
    "log_url_count",
    "log_mention_count",
    "log_hashtag_count",
    "has_quote",
    "is_reshare",
    "log_like_count",
    "log_dislike_count",
    "log_share_count",
    "hour_sin",
    "hour_cos",
]


def build_observed_post_features(df_posts):
    """Build deterministic, evidence-derived post features.

    These features are a traceable fallback when no external semantic encoder
    output is supplied.  They deliberately replace the former random Tweet
    node initialization; they are not presented as a semantic representation.
    """
    rows = []
    for _, row in df_posts.iterrows():
        content = str(row.get("content", "") or "")
        quote = str(row.get("quote_content", "") or "")
        characters = max(len(content), 0)
        tokens = re.findall(r"\S+", content)
        letters = [char for char in content if char.isalpha()]
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
            if letters else 0.0
        )
        punctuation_ratio = (
            sum(not char.isalnum() and not char.isspace() for char in content)
            / max(characters, 1)
        )
        timestamp = pd.to_datetime(row.get("created_at"), errors="coerce", utc=True)
        hour = float(timestamp.hour) if not pd.isna(timestamp) else 0.0
        angle = 2.0 * np.pi * hour / 24.0
        rows.append([
            np.log1p(characters),
            np.log1p(len(tokens)),
            uppercase_ratio,
            punctuation_ratio,
            np.log1p(len(re.findall(r"https?://|www\.", content, re.I))),
            np.log1p(content.count("@")),
            np.log1p(content.count("#")),
            float(bool(quote.strip())),
            float(_normalize_db_id(row.get("original_post_id")) is not None),
            np.log1p(max(float(row.get("num_likes", 0) or 0), 0.0)),
            np.log1p(max(float(row.get("num_dislikes", 0) or 0), 0.0)),
            np.log1p(max(float(row.get("num_shares", 0) or 0), 0.0)),
            np.sin(angle),
            np.cos(angle),
        ])
    return np.asarray(rows, dtype=np.float32)


def load_post_embeddings(path):
    """Load a transparent post-id to embedding mapping from NPZ or CSV.

    NPZ files must contain ``post_ids`` and a two-dimensional ``embeddings``
    array. CSV files must contain ``post_id`` plus numeric embedding columns.
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Post embedding file does not exist: {path}")
    extension = os.path.splitext(path)[1].lower()
    if extension == ".npz":
        archive = np.load(path, allow_pickle=False)
        if "post_ids" not in archive or "embeddings" not in archive:
            raise ValueError("NPZ must contain post_ids and embeddings arrays")
        post_ids = archive["post_ids"]
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    elif extension == ".csv":
        frame = pd.read_csv(path)
        if "post_id" not in frame.columns:
            raise ValueError("Post embedding CSV must contain a post_id column")
        embedding_columns = [column for column in frame.columns if column != "post_id"]
        if not embedding_columns:
            raise ValueError("Post embedding CSV has no embedding columns")
        embeddings = frame[embedding_columns].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float32)
        post_ids = frame["post_id"].to_numpy()
    else:
        raise ValueError("Post embeddings must use .npz or .csv format")
    if embeddings.ndim != 2 or len(post_ids) != embeddings.shape[0]:
        raise ValueError("Post ids and embedding rows must have matching lengths")
    return {
        normalized: embeddings[index]
        for index, value in enumerate(post_ids)
        if (normalized := _normalize_db_id(value)) is not None
    }


def _set_relation_store(
    data,
    edge_type,
    sources,
    targets,
    evidence_ids,
    multiplicities=None,
):
    """Attach one relation and its auditable evidence alignment."""
    import torch

    if not sources:
        return
    store = data[edge_type]
    store.edge_index = torch.tensor([sources, targets], dtype=torch.long)
    store.evidence_ids = list(evidence_ids)
    if multiplicities is not None:
        store.multiplicity = torch.tensor(multiplicities, dtype=torch.float)


def _build_twibot_boundary_features(bundle):
    """Combine observed profile counts with sampled open-neighborhood degree."""
    boundary = bundle.boundary_users.copy()
    edges = bundle.follow_edges
    incoming = edges.groupby("followee_id")["multiplicity"].sum()
    outgoing = edges.groupby("follower_id")["multiplicity"].sum()
    sampled_in = boundary["user_id"].map(incoming).fillna(0.0).to_numpy(float)
    sampled_out = boundary["user_id"].map(outgoing).fillna(0.0).to_numpy(float)
    followers = pd.to_numeric(boundary["followers"], errors="coerce").fillna(0.0)
    following = pd.to_numeric(boundary["following"], errors="coerce").fillna(0.0)
    observed_profile = ((followers > 0) | (following > 0)).astype(float)
    degree_total = sampled_in + sampled_out
    return np.column_stack([
        np.log1p(followers.to_numpy(float)),
        np.log1p(following.to_numpy(float)),
        observed_profile.to_numpy(float),
        np.log1p(sampled_in),
        np.log1p(sampled_out),
        sampled_out / np.maximum(degree_total, 1.0),
    ]).astype(np.float32)


def _add_twibot_static_bundle(data, bundle, user_map, post_embeddings=None):
    """Add TwiBot boundary nodes, real follow edges and static text actions."""
    import torch

    boundary_ids = bundle.boundary_users["user_id"].astype(str).tolist()
    boundary_map = {user_id: index for index, user_id in enumerate(boundary_ids)}
    data["boundary_user"].x = torch.tensor(
        _build_twibot_boundary_features(bundle), dtype=torch.float
    )
    data["boundary_user"].node_ids = boundary_ids
    data["boundary_user"].feature_source = (
        "public_profile_counts+sampled_follow_degree"
    )

    relation_buckets = {}

    def add_follow_relation(source_id, target_id, row, inverse=False):
        source_core = source_id in user_map
        target_core = target_id in user_map
        source_boundary = source_id in boundary_map
        target_boundary = target_id in boundary_map
        if not (source_core or source_boundary) or not (target_core or target_boundary):
            return
        source_type = "user" if source_core else "boundary_user"
        target_type = "user" if target_core else "boundary_user"
        source_index = user_map[source_id] if source_core else boundary_map[source_id]
        target_index = user_map[target_id] if target_core else boundary_map[target_id]
        if bundle.dataset_kind == "twibot22_raw":
            raw_relation = str(getattr(row, "relation", "following"))
            relation = f"rev_{raw_relation}" if inverse else raw_relation
        else:
            relation = "followed_by" if inverse else "follows"
        edge_type = (source_type, relation, target_type)
        bucket = relation_buckets.setdefault(
            edge_type, {"src": [], "dst": [], "evidence": [], "multiplicity": []}
        )
        bucket["src"].append(source_index)
        bucket["dst"].append(target_index)
        bucket["evidence"].append(list(row.evidence_ids))
        bucket["multiplicity"].append(float(row.multiplicity))

    for row in bundle.follow_edges.itertuples(index=False):
        source_id = str(row.follower_id)
        target_id = str(row.followee_id)
        add_follow_relation(source_id, target_id, row, inverse=False)
        add_follow_relation(target_id, source_id, row, inverse=True)
    for edge_type, bucket in relation_buckets.items():
        _set_relation_store(
            data,
            edge_type,
            bucket["src"],
            bucket["dst"],
            bucket["evidence"],
            bucket["multiplicity"],
        )

    data.dataset_kind = bundle.dataset_kind
    data.dataset_capabilities = bundle.capabilities.to_dict()
    data.dataset_warnings = list(bundle.warnings)
    actions = bundle.actions.copy()
    if actions.empty:
        return
    content_nodes = actions.drop_duplicates("content_node_id", keep="first").copy()
    content_ids = content_nodes["content_node_id"].astype(str).tolist()
    content_map = {content_id: index for index, content_id in enumerate(content_ids)}
    observed_frame = pd.DataFrame({
        "content": content_nodes["content"].fillna("").astype(str),
        "quote_content": "",
        "original_post_id": None,
        "created_at": None,
        "num_likes": 0,
        "num_dislikes": 0,
        "num_shares": 0,
    })
    feature_parts = [build_observed_post_features(observed_frame)]
    feature_source = "observed_static_text_features"
    if post_embeddings is not None:
        dimensions = {
            np.asarray(value).reshape(-1).shape[0]
            for value in post_embeddings.values()
        }
        if len(dimensions) != 1:
            raise ValueError("All post embeddings must have the same dimension")
        embedding_dim = next(iter(dimensions)) if dimensions else 0
        semantic = np.zeros((len(content_ids), embedding_dim), dtype=np.float32)
        for content_id, index in content_map.items():
            if content_id in post_embeddings:
                semantic[index] = np.asarray(
                    post_embeddings[content_id], dtype=np.float32
                ).reshape(-1)
        feature_parts.append(semantic)
        feature_source += "+external_semantic_embeddings"
    data["tweet"].x = torch.tensor(
        np.concatenate(feature_parts, axis=1), dtype=torch.float
    )
    data["tweet"].node_ids = content_ids
    data["tweet"].feature_source = feature_source
    data["tweet"].temporal = False
    data["tweet"].original_post_ids_available = False

    action_relations = {
        "post": ("posts", "authored_by"),
        "retweet": ("retweets", "retweeted_by"),
        "reply": ("replies", "replied_by"),
        "like": ("likes", "liked_by"),
    }
    action_buckets = {}
    for row in actions.itertuples(index=False):
        actor_id = str(row.actor_id)
        if actor_id in user_map:
            actor_type, actor_index = "user", user_map[actor_id]
        elif actor_id in boundary_map:
            actor_type, actor_index = "boundary_user", boundary_map[actor_id]
        else:
            continue
        relation_pair = action_relations.get(str(row.action_type).lower())
        if relation_pair is None:
            continue
        tweet_index = content_map[str(row.content_node_id)]
        forward_type = (actor_type, relation_pair[0], "tweet")
        reverse_type = ("tweet", relation_pair[1], actor_type)
        for edge_type, source_index, target_index in (
            (forward_type, actor_index, tweet_index),
            (reverse_type, tweet_index, actor_index),
        ):
            bucket = action_buckets.setdefault(
                edge_type, {"src": [], "dst": [], "evidence": []}
            )
            bucket["src"].append(source_index)
            bucket["dst"].append(target_index)
            bucket["evidence"].append(str(row.evidence_id))
    for edge_type, bucket in action_buckets.items():
        _set_relation_store(
            data, edge_type, bucket["src"], bucket["dst"], bucket["evidence"]
        )


def _add_twibot_raw_bundle(data, bundle, user_map, post_embeddings=None):
    """Materialize a raw TwiBot-22 bundle while preserving temporal metadata."""
    _add_twibot_static_bundle(
        data, bundle, user_map, post_embeddings=post_embeddings
    )
    posts = getattr(bundle, "posts", None)
    if posts is None or posts.empty or "tweet" not in data.node_types:
        data.dataset_capabilities = bundle.capabilities.to_dict()
        data.dataset_warnings = list(bundle.warnings)
        return
    import torch

    content_ids = data["tweet"].node_ids
    post_map = {
        f"tweet:{str(row.post_id)}": row
        for row in posts.itertuples(index=False)
    }
    observed_rows = []
    for content_id in content_ids:
        row = post_map.get(str(content_id))
        if row is None:
            observed_rows.append({
                "content": "", "quote_content": "", "original_post_id": None,
                "created_at": None, "num_likes": 0, "num_dislikes": 0,
                "num_shares": 0,
            })
        else:
            observed_rows.append({
                "content": row.content,
                "quote_content": "",
                "original_post_id": row.original_post_id,
                "created_at": row.created_at,
                "num_likes": row.num_likes,
                "num_dislikes": 0,
                "num_shares": row.num_retweets,
            })
    feature_parts = [build_observed_post_features(pd.DataFrame(observed_rows))]
    feature_source = "observed_raw_tweet_features"
    if post_embeddings is not None:
        dimensions = {
            np.asarray(value).reshape(-1).shape[0]
            for value in post_embeddings.values()
        }
        if len(dimensions) != 1:
            raise ValueError("All post embeddings must have the same dimension")
        embedding_dim = next(iter(dimensions)) if dimensions else 0
        semantic = np.zeros((len(content_ids), embedding_dim), dtype=np.float32)
        for index, content_id in enumerate(content_ids):
            post_id = str(content_id).removeprefix("tweet:")
            if post_id in post_embeddings:
                semantic[index] = np.asarray(
                    post_embeddings[post_id], dtype=np.float32
                ).reshape(-1)
        feature_parts.append(semantic)
        feature_source += "+external_semantic_embeddings"
    data["tweet"].x = torch.tensor(
        np.concatenate(feature_parts, axis=1), dtype=torch.float
    )
    data["tweet"].feature_source = feature_source
    data["tweet"].temporal = True
    data["tweet"].original_post_ids_available = True
    data["tweet"].created_at = [
        getattr(post_map.get(str(content_id)), "created_at", None)
        if post_map.get(str(content_id)) is not None else None
        for content_id in content_ids
    ]
    data["tweet"].post_ids = [
        str(content_id).removeprefix("tweet:") for content_id in content_ids
    ]
    data.dataset_capabilities = bundle.capabilities.to_dict()
    data.dataset_warnings = list(bundle.warnings)



def build_hetero_data(
    user_ids,
    features_26,
    db_path,
    threshold=0.7,
    post_embeddings=None,
    twibot_bundle=None,
):
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError("torch_geometric is required: pip install torch_geometric")

    data = HeteroData()
    user_map = {str(uid): i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    data['user'].x = torch.tensor(features_26.astype(np.float32))
    # Preserve the exact feature-row identity order.  Joint training aligns
    # labels and evidence through this explicit list rather than assuming a
    # DataFrame, SQLite table, and PyG graph happen to share row order.
    data['user'].node_ids = [str(uid) for uid in user_ids]
    cos_edges = build_cosine_edges(features_26, threshold)
    if cos_edges:
        src = [e[0] for e in cos_edges]
        dst = [e[1] for e in cos_edges]
        data['user', 'similar', 'user'].edge_index = torch.tensor(
            [src + dst, dst + src], dtype=torch.long
        )
        logger.info(f"  [similar] {len(cos_edges)} undirected edges → {len(src)*2} directed edges")

    if twibot_bundle is not None:
        bundle_core_ids = set(twibot_bundle.core_users["user_id"].astype(str))
        if bundle_core_ids != set(user_map):
            raise ValueError(
                "materialized TwiBot bundle core IDs do not match feature rows"
            )
        _add_twibot_raw_bundle(
            data, twibot_bundle, user_map, post_embeddings=post_embeddings
        )
        rev_user_map = {value: key for key, value in user_map.items()}
        logger.info(
            "  Materialized TwiBot-22 graph: core=%d, boundary=%d, text=%d, edge_types=%d",
            len(user_ids),
            data["boundary_user"].num_nodes if "boundary_user" in data.node_types else 0,
            data["tweet"].num_nodes if "tweet" in data.node_types else 0,
            len(data.edge_types),
        )
        return data, rev_user_map

    if os.path.isdir(db_path):
        from data_processing.twibot22_raw_adapter import TwiBot22RawAdapter

        bundle = TwiBot22RawAdapter(
            twibot_dir=db_path,
            core_user_ids=user_ids,
        ).load()
        _add_twibot_raw_bundle(
            data, bundle, user_map, post_embeddings=post_embeddings
        )
        # The complete raw relation table remains attached to the adapter
        # bundle for later entity-store expansion. The current graph path
        # materializes user-user and action-to-tweet stores with provenance.
        rev_user_map = {value: key for key, value in user_map.items()}
        logger.info(
            "  TwiBot-22 raw graph: core=%d, boundary=%d, text=%d, edge_types=%d",
            len(user_ids),
            data["boundary_user"].num_nodes if "boundary_user" in data.node_types else 0,
            data["tweet"].num_nodes if "tweet" in data.node_types else 0,
            len(data.edge_types),
        )
        return data, rev_user_map

    if not os.path.exists(db_path):
        logger.warning(f"  DB does not exist: {db_path}, only containing cosine similarity edges")
        return data, {v: k for k, v in user_map.items()}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {t[0] for t in cursor.fetchall()}

    if {"user", "follow", "agent_actions"}.issubset(tables) and "post" not in tables:
        conn.close()
        from data_processing.dataset_adapter import TwiBotStaticAdapter

        bundle = TwiBotStaticAdapter(
            db_path=db_path,
            core_user_ids=user_ids,
        ).load()
        _add_twibot_static_bundle(
            data, bundle, user_map, post_embeddings=post_embeddings
        )
        rev_user_map = {value: key for key, value in user_map.items()}
        logger.info(
            "  TwiBot static graph: core=%d, boundary=%d, text=%d, edge_types=%d",
            len(user_ids),
            data["boundary_user"].num_nodes,
            data["tweet"].num_nodes if "tweet" in data.node_types else 0,
            len(data.edge_types),
        )
        return data, rev_user_map

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
            cursor.execute("PRAGMA table_info(post)")
            post_columns = {row[1] for row in cursor.fetchall()}
            desired_columns = [
                "post_id", "user_id", "original_post_id", "content",
                "quote_content", "created_at", "num_likes", "num_dislikes",
                "num_shares",
            ]
            selected_columns = [
                column for column in desired_columns if column in post_columns
            ]
            if "post_id" not in selected_columns or "user_id" not in selected_columns:
                raise ValueError("post table must contain post_id and user_id")
            df_posts = pd.read_sql_query(
                f"SELECT {', '.join(selected_columns)} FROM post", conn
            )
            for column in desired_columns:
                if column not in df_posts.columns:
                    df_posts[column] = None if column in {
                        "original_post_id", "quote_content", "created_at"
                    } else ("" if column == "content" else 0)
            tweet_ids = [
                pid for pid in (_normalize_db_id(v) for v in df_posts['post_id'])
                if pid is not None
            ]
            tweet_ids = list(dict.fromkeys(tweet_ids))
            tweet_map = {tid: i for i, tid in enumerate(tweet_ids)}
            n_tweets = len(tweet_map)
            first_rows = df_posts.assign(
                _normalized_post_id=df_posts["post_id"].map(_normalize_db_id)
            ).drop_duplicates("_normalized_post_id").set_index("_normalized_post_id")
            ordered_posts = first_rows.loc[tweet_ids].reset_index(drop=True)
            observed_features = build_observed_post_features(ordered_posts)
            feature_parts = [observed_features]
            feature_source = "observed_post_features"
            if post_embeddings is not None:
                dimensions = {
                    np.asarray(value).reshape(-1).shape[0]
                    for value in post_embeddings.values()
                }
                if len(dimensions) != 1:
                    raise ValueError("All post embeddings must have the same dimension")
                embedding_dim = next(iter(dimensions)) if dimensions else 0
                semantic = np.zeros((n_tweets, embedding_dim), dtype=np.float32)
                for post_id, index in tweet_map.items():
                    if post_id in post_embeddings:
                        semantic[index] = np.asarray(
                            post_embeddings[post_id], dtype=np.float32
                        ).reshape(-1)
                feature_parts.append(semantic)
                feature_source = "observed_post_features+external_semantic_embeddings"
            tweet_features = np.concatenate(feature_parts, axis=1)
            data['tweet'].x = torch.tensor(tweet_features, dtype=torch.float)
            data['tweet'].feature_source = feature_source
            data['tweet'].observed_feature_names = list(POST_OBSERVED_FEATURES)
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
                data['tweet', 'authored_by', 'user'].edge_index = torch.tensor(
                    [dst, src], dtype=torch.long
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
                    data['tweet', 'retweeted_by', 'user'].edge_index = torch.tensor(
                        [rt_dst, rt_src], dtype=torch.long
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
                data['tweet', 'liked_by', 'user'].edge_index = torch.tensor(
                    [dst, src], dtype=torch.long
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
                data['tweet', 'commented_by', 'user'].edge_index = torch.tensor(
                    [dst, src], dtype=torch.long
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
