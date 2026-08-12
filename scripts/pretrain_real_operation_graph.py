"""Self-supervise Lorentz-HGT on one real information-operation graph.

No bad/good labels enter this stage. The bundle is split at a declared time
cutoff: only pre-cutoff events and edges train the encoder, while masked-feature
and future-edge validation select the best checkpoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (ROOT, CHARACTER_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from lorentz_hgt import (  # noqa: E402
    IntrinsicLorentzHGT,
    logmap0,
    lorentz_distance,
)
from data_processing.feature_contracts import OBSERVABLE_18  # noqa: E402
from scripts.lorentz_warm_start import REAL_PRETRAIN_SCHEMA  # noqa: E402

CACHE_SCHEMA = "hypertrace.real-operation-graph-cache.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _cutoff(events: pd.DataFrame, quantile: float) -> pd.Timestamp:
    if not 0.0 < quantile < 1.0:
        raise ValueError("cutoff quantile must be between zero and one")
    timestamps = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    if not timestamps.notna().all():
        raise ValueError("all training events must have parseable timestamps")
    return timestamps.quantile(quantile)


def _cutoff_safe_node_features(
    bundle: Path,
    events: pd.DataFrame,
    event_times: pd.Series,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute node features using only events available at the cutoff."""
    pre_events = events[event_times <= cutoff].copy()
    pre_events["actor_id"] = pre_events["actor_id"].astype(str)
    pre_events["event_id"] = pre_events["event_id"].astype(str)
    if pre_events.empty:
        raise ValueError("no events are available at the pretraining cutoff")
    tweet_features_path = bundle / "tweet_features.csv"
    if tweet_features_path.is_file():
        all_tweet_features = pd.read_csv(tweet_features_path, low_memory=False)
    else:
        # Backward-compatible read-only fallback for the v1 external bundle.
        # The older materializer retained text/timestamp/event type in
        # events.csv but did not persist tweet_features.csv.  Derive only
        # source-observed lexical/time features from pre-cutoff events; no
        # labels, future rows, or semantic embeddings are introduced.
        # The legacy bundle has no materialized tweet_features.csv.  Keep this
        # fallback source-observed and cutoff-safe, but vectorize it: Honduras
        # contains >1M events and a Python row/regex loop is unnecessarily slow.
        text = pre_events.get("text", pd.Series("", index=pre_events.index))
        text = text.fillna("").astype(str)
        chars = text.str.len().astype("float64")
        token_count = text.str.count(r"\S+").astype("float64")
        letters = text.str.count(r"[A-Za-z]").astype("float64")
        uppercase = text.str.count(r"[A-Z]").astype("float64")
        punctuation = text.str.count(r"[^\w\s]").astype("float64")
        urls = text.str.count(r"https?://|www\.", flags=re.IGNORECASE).astype("float64")
        mentions = text.str.count(r"@[A-Za-z0-9_]+").astype("float64")
        hashtags = text.str.count(r"#[A-Za-z0-9_]+").astype("float64")
        event_type = pre_events.get(
            "event_type", pd.Series("post", index=pre_events.index)
        ).fillna("post").astype(str)
        timestamps = pd.to_datetime(pre_events["timestamp"], errors="coerce", utc=True)
        hour = timestamps.dt.hour.fillna(0).astype("float64")
        angle = 2.0 * np.pi * hour / 24.0
        all_tweet_features = pd.DataFrame({
            "tweet_id": pre_events["event_id"].astype(str).to_numpy(),
            "log_char_count": np.log1p(chars.to_numpy()),
            "log_token_count": np.log1p(token_count.to_numpy()),
            "uppercase_ratio": (uppercase / letters.clip(lower=1)).to_numpy(),
            "punctuation_ratio": (punctuation / chars.clip(lower=1)).to_numpy(),
            "log_url_count": np.log1p(urls.to_numpy()),
            "log_mention_count": np.log1p(mentions.to_numpy()),
            "log_hashtag_count": np.log1p(hashtags.to_numpy()),
            "has_quote": (event_type == "quote").astype("float32").to_numpy(),
            "is_reshare": (event_type == "retweet").astype("float32").to_numpy(),
            "log_like_count": np.zeros(len(pre_events), dtype="float32"),
            "log_dislike_count": np.zeros(len(pre_events), dtype="float32"),
            "log_share_count": (event_type == "retweet").astype("float32").to_numpy(),
            "hour_sin": np.sin(angle.to_numpy()),
            "hour_cos": np.cos(angle.to_numpy()),
        }).drop_duplicates("tweet_id", keep="first")
    if "tweet_id" not in all_tweet_features:
        raise ValueError("tweet features lack tweet_id")
    all_tweet_features["tweet_id"] = all_tweet_features["tweet_id"].astype(str)
    tweets = all_tweet_features[
        all_tweet_features["tweet_id"].isin(set(pre_events["event_id"]))
    ].drop_duplicates("tweet_id", keep="first")
    if len(tweets) != pre_events["event_id"].nunique():
        raise ValueError("pre-cutoff events and tweet features are not one-to-one")
    joined = pre_events[["event_id", "actor_id", "event_type"]].merge(
        tweets,
        left_on="event_id",
        right_on="tweet_id",
        how="left",
        # Multiple event rows can reference one observed tweet (for example,
        # repeated/duplicated source records); tweet features remain unique.
        validate="many_to_one",
    )
    grouped = joined.groupby("actor_id", sort=True)
    users = pd.DataFrame({"user_id": sorted(pre_events["actor_id"].unique())})
    users = users.set_index("user_id")
    for column in OBSERVABLE_18:
        users[column] = 0.0
    users["Action_Frequency"] = grouped.size().astype(float)
    event_counts = pd.crosstab(joined["actor_id"], joined["event_type"])
    for column, event_type in (
        ("Retweet_Ratio", "retweet"),
        ("Reply_Ratio", "reply"),
    ):
        numerator = event_counts.get(
            event_type, pd.Series(0.0, index=event_counts.index)
        )
        users[column] = numerator / event_counts.sum(axis=1)
    users["Like_Ratio"] = grouped["log_like_count"].apply(
        lambda values: float(np.expm1(values.to_numpy(dtype=float)).mean())
    )
    for column, source in (
        ("URL_Ratio", "log_url_count"),
        ("Mention_Ratio", "log_mention_count"),
        ("Hashtag_Ratio", "log_hashtag_count"),
    ):
        users[column] = grouped[source].apply(
            lambda values: float((values.to_numpy(dtype=float) > 0).mean())
        )
    pre_events["hour"] = event_times[event_times <= cutoff].dt.hour.to_numpy()
    hour_counts = pd.crosstab(pre_events["actor_id"], pre_events["hour"])
    probabilities = hour_counts.div(hour_counts.sum(axis=1), axis=0)
    users["Temporal_Entropy"] = -(
        probabilities * np.log(probabilities.where(probabilities > 0, 1.0))
    ).sum(axis=1)
    # Follower/following and media are intentionally unavailable here because
    # the materialized aggregate does not provide a cutoff-safe profile value.
    users = users.fillna(0.0).reset_index()
    return users, tweets


def load_training_graph(
    bundle: Path,
    *,
    cutoff_quantile: float,
    device: torch.device,
    max_train_edges: int | None,
    max_validation_edges: int,
    seed: int,
) -> tuple[HeteroData, dict[tuple[str, str, str], torch.Tensor], dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("label_semantics") != "information_operation_membership":
        raise ValueError("expected an audited information-operation bundle")
    events = pd.read_csv(bundle / "events.csv", low_memory=False)
    edges = pd.read_csv(bundle / "edges.csv", low_memory=False)
    cutoff = _cutoff(events, cutoff_quantile)
    event_times = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    users, tweets = _cutoff_safe_node_features(
        bundle, events, event_times, cutoff
    )
    edge_times = pd.to_datetime(edges["timestamp"], errors="coerce", utc=True)
    if not edge_times.notna().all():
        raise ValueError("all graph edges must have parseable timestamps")
    before = edges[edge_times <= cutoff].copy()
    future = edges[edge_times > cutoff].copy()
    user_ids = users["user_id"].astype(str).tolist()
    tweet_ids = tweets["tweet_id"].astype(str).tolist()
    maps = {
        "user": {value: index for index, value in enumerate(user_ids)},
        "tweet": {value: index for index, value in enumerate(tweet_ids)},
    }
    user_x = users.drop(columns="user_id").to_numpy(dtype="float32")
    tweet_x = tweets.drop(columns="tweet_id").to_numpy(dtype="float32")
    # Normalization is fitted only on the pretraining operation.
    normalized = {}
    normalization = {}
    for name, values in (("user", user_x), ("tweet", tweet_x)):
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        normalized[name] = (values - mean) / std
        normalization[name] = {"mean": mean.tolist(), "std": std.tolist()}

    def mapped(frame: pd.DataFrame) -> dict[tuple[str, str, str], torch.Tensor]:
        buckets: dict[tuple[str, str, str], list[torch.Tensor]] = {}
        # Group by typed relation first, then use pandas vectorized mapping;
        # the prior row-wise itertuples path was prohibitively slow on UAE.
        for (source_type, edge_name, target_type), group in frame.groupby(
            ["source_type", "edge_type", "target_type"], sort=False
        ):
            source_map = maps.get(str(source_type), {})
            target_map = maps.get(str(target_type), {})
            src = group["source_id"].astype(str).map(source_map)
            dst = group["target_id"].astype(str).map(target_map)
            valid = src.notna() & dst.notna()
            if not valid.any():
                continue
            pairs = torch.from_numpy(np.vstack([
                src.loc[valid].to_numpy(dtype=np.int64),
                dst.loc[valid].to_numpy(dtype=np.int64),
            ])).to(torch.long)
            edge_type = (str(source_type), str(edge_name), str(target_type))
            buckets.setdefault(edge_type, []).append(pairs)
        result = {}
        generator = torch.Generator().manual_seed(seed)
        for edge_type, pairs in buckets.items():
            values = torch.cat(pairs, dim=1).contiguous()
            if max_train_edges is not None and values.shape[1] > max_train_edges:
                keep = torch.randperm(values.shape[1], generator=generator)[
                    :max_train_edges
                ]
                values = values[:, keep]
            result[edge_type] = values.to(device)
        return result

    train_edges = mapped(before)
    validation_edges = mapped(future)
    validation_edges = {
        key: value[:, :max_validation_edges]
        for key, value in validation_edges.items()
        if value.shape[1]
    }
    if not train_edges:
        raise ValueError("pre-cutoff graph has no mappable edges")
    graph = HeteroData()
    graph["user"].x = torch.tensor(normalized["user"], device=device)
    graph["tweet"].x = torch.tensor(normalized["tweet"], device=device)
    observed_user_columns = {
        "Action_Frequency", "Like_Ratio", "Retweet_Ratio", "Reply_Ratio",
        "Temporal_Entropy", "URL_Ratio", "Mention_Ratio", "Hashtag_Ratio",
    }
    graph["user"].feature_available = torch.tensor(
        [[column in observed_user_columns for column in users.columns.drop("user_id")]]
        * len(users),
        dtype=torch.bool,
        device=device,
    )
    graph["tweet"].feature_available = torch.ones_like(
        graph["tweet"].x, dtype=torch.bool
    )
    for edge_type, edge_index in train_edges.items():
        graph[edge_type].edge_index = edge_index
    audit = {
        "bundle": str(bundle.resolve()),
        "bundle_manifest_sha256": _sha256(bundle / "manifest.json"),
        "source_operation": manifest.get("operation_id"),
        "source_cutoff": cutoff.isoformat(),
        "cutoff_quantile": cutoff_quantile,
        "observed_node_types": list(graph.node_types),
        "observed_edge_types": [list(value) for value in graph.edge_types],
        "train_edge_counts": {
            "__".join(key): int(value.shape[1]) for key, value in train_edges.items()
        },
        "future_edge_counts": {
            "__".join(key): int(value.shape[1])
            for key, value in validation_edges.items()
        },
        "normalization": normalization,
        "cutoff_safe_feature_recomputation": True,
        "labels_consumed": False,
    }
    return graph, validation_edges, audit


def save_graph_cache(
    cache_dir: Path,
    *,
    graph: HeteroData,
    future_edges: dict[tuple[str, str, str], torch.Tensor],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Persist one cutoff-safe graph so later runs avoid rereading raw CSVs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cache_dir / "graph.pt"
    future_path = cache_dir / "future_edges.pt"
    audit_path = cache_dir / "audit.json"
    normalization_path = cache_dir / "normalization.json"
    graph_cpu = graph.cpu()
    torch.save(graph_cpu, graph_path)
    torch.save({tuple(key): value.cpu() for key, value in future_edges.items()}, future_path)
    audit_payload = {
        "schema_version": CACHE_SCHEMA,
        **audit,
        "cache_graph": str(graph_path.resolve()),
        "cache_future_edges": str(future_path.resolve()),
        "labels_consumed": False,
    }
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    normalization_path.write_text(json.dumps(audit.get("normalization", {}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit_payload


def load_graph_cache(cache_dir: Path, *, device: torch.device):
    """Load and validate a previously materialized cutoff-safe graph."""
    cache_dir = Path(cache_dir)
    audit_path = cache_dir / "audit.json"
    graph_path = cache_dir / "graph.pt"
    future_path = cache_dir / "future_edges.pt"
    if not all(path.is_file() for path in (audit_path, graph_path, future_path)):
        raise FileNotFoundError(f"incomplete real-operation graph cache: {cache_dir}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unsupported real-operation graph cache schema")
    if audit.get("labels_consumed") is not False:
        raise ValueError("real-operation cache must declare labels_consumed=false")
    graph = torch.load(graph_path, map_location=device, weights_only=False)
    future_edges = torch.load(future_path, map_location=device, weights_only=False)
    if "user" not in graph.node_types or "tweet" not in graph.node_types:
        raise ValueError("cached graph must contain user and tweet nodes")
    return graph, future_edges, audit


def _negative_pairs(
    edge_index: torch.Tensor,
    source_count: int,
    target_count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    count = edge_index.shape[1]
    return torch.stack([
        torch.randint(source_count, (count,), generator=generator, device=edge_index.device),
        torch.randint(target_count, (count,), generator=generator, device=edge_index.device),
    ])


def _link_loss(
    points: dict[str, torch.Tensor],
    edge_dict: dict[tuple[str, str, str], torch.Tensor],
    curvature: torch.Tensor,
    *,
    generator: torch.Generator,
    max_edges: int,
) -> torch.Tensor:
    losses = []
    for edge_type, values in edge_dict.items():
        if not values.shape[1]:
            continue
        source_type, _, target_type = edge_type
        keep = torch.randperm(
            values.shape[1], generator=generator, device=values.device
        )[:max_edges]
        positive = values[:, keep]
        negative = _negative_pairs(
            positive,
            points[source_type].shape[0],
            points[target_type].shape[0],
            generator=generator,
        )
        positive_score = -lorentz_distance(
            points[source_type][positive[0]],
            points[target_type][positive[1]],
            curvature,
        ).squeeze(-1)
        negative_score = -lorentz_distance(
            points[source_type][negative[0]],
            points[target_type][negative[1]],
            curvature,
        ).squeeze(-1)
        losses.append(F.binary_cross_entropy_with_logits(
            torch.cat([positive_score, negative_score]),
            torch.cat([
                torch.ones_like(positive_score), torch.zeros_like(negative_score)
            ]),
        ))
    return torch.stack(losses).mean() if losses else curvature.sum() * 0.0


def run_pretraining(
    *,
    bundle: Path,
    output_dir: Path,
    device: str = "cuda:0",
    epochs: int = 20,
    seed: int = 2026,
    cutoff_quantile: float = 0.8,
    hidden_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    mask_rate: float = 0.15,
    max_train_edges: int | None = 200_000,
    max_validation_edges: int = 20_000,
    max_loss_edges: int = 2_048,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    _seed(seed)
    torch_device = torch.device(device)
    if cache_dir is not None:
        graph, future_edges, audit = load_graph_cache(cache_dir, device=torch_device)
    else:
        graph, future_edges, audit = load_training_graph(
            bundle,
            cutoff_quantile=cutoff_quantile,
            device=torch_device,
            max_train_edges=max_train_edges,
            max_validation_edges=max_validation_edges,
            seed=seed,
        )
    encoder = IntrinsicLorentzHGT(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        metadata=graph.metadata(),
        dropout=0.1,
    ).to(torch_device)
    decoders = torch.nn.ModuleDict({
        node_type: torch.nn.Linear(hidden_dim, graph[node_type].x.shape[1])
        for node_type in graph.node_types
    }).to(torch_device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoders.parameters()], lr=1e-3
    )
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation = float("inf")
    best_state = None
    for epoch in range(epochs):
        encoder.train()
        decoders.train()
        masked_inputs = {}
        masks = {}
        for node_type in graph.node_types:
            values = graph[node_type].x
            mask = graph[node_type].feature_available & (torch.rand(
                values.shape, generator=generator, device=torch_device
            ) < mask_rate)
            masks[node_type] = mask
            masked_inputs[node_type] = values.masked_fill(mask, 0.0)
        points = encoder.encode_lorentz_nodes(
            masked_inputs, graph.edge_index_dict
        )
        curvature = encoder.common_curvature()
        reconstruction_losses = []
        for node_type, values in points.items():
            prediction = decoders[node_type](logmap0(values, curvature))
            if masks[node_type].any():
                reconstruction_losses.append(F.mse_loss(
                    prediction[masks[node_type]], graph[node_type].x[masks[node_type]]
                ))
        feature_loss = torch.stack(reconstruction_losses).mean()
        train_link_loss = _link_loss(
            points,
            graph.edge_index_dict,
            curvature,
            generator=generator,
            max_edges=max_loss_edges,
        )
        loss = feature_loss + 0.1 * train_link_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*encoder.parameters(), *decoders.parameters()], 5.0
        )
        optimizer.step()

        encoder.eval()
        with torch.no_grad():
            validation_points = encoder.encode_lorentz_nodes(
                graph.x_dict, graph.edge_index_dict
            )
            validation_link_loss = _link_loss(
                validation_points,
                future_edges,
                encoder.common_curvature(),
                generator=generator,
                max_edges=max_validation_edges,
            )
        validation = float(validation_link_loss.detach().cpu())
        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.detach().cpu()),
            "feature_loss": float(feature_loss.detach().cpu()),
            "train_link_loss": float(train_link_loss.detach().cpu()),
            "future_link_loss": validation,
        })
        if validation < best_validation:
            best_validation = validation
            best_state = deepcopy(encoder.state_dict())
        payload = {
            "schema_version": REAL_PRETRAIN_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_operation": audit["source_operation"],
            "source_cutoff": audit["source_cutoff"],
            "observed_node_types": audit["observed_node_types"],
            "observed_edge_types": audit["observed_edge_types"],
            "labels_consumed": False,
            "encoder": encoder.state_dict(),
            "best_encoder": best_state,
            "decoders": decoders.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_validation_future_link_loss": best_validation,
            "history": history,
            "data_audit": audit,
        }
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if best_state is not None:
            best_payload = dict(payload)
            best_payload["encoder"] = best_state
            torch.save(best_payload, output_dir / "best_checkpoint.pt")
    result = {
        "schema_version": "hypertrace.real-operation-pretrain-result.v1",
        "status": "passed",
        "source_operation": audit["source_operation"],
        "source_cutoff": audit["source_cutoff"],
        "labels_consumed": False,
        "epochs": epochs,
        "best_validation_future_link_loss": best_validation,
        "history": history,
        "data_audit": audit,
        "best_checkpoint": str((output_dir / "best_checkpoint.pt").resolve()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cutoff-quantile", type=float, default=0.8)
    parser.add_argument("--max-train-edges", type=int, default=200_000)
    parser.add_argument("--max-validation-edges", type=int, default=20_000)
    parser.add_argument("--max-loss-edges", type=int, default=2_048)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_pretraining(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
