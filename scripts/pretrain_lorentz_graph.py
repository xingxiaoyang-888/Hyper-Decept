"""Self-supervised Lorentz-HGT pretraining on one real graph bundle.

This stage never consumes bot/CIB labels. It reconstructs masked observed
features and distinguishes observed graph edges from sampled non-edges. The
checkpoint is an encoder initialization for the later synthetic supervised
stage, not a detection result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (ROOT, CHARACTER_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from lorentz_hgt import IntrinsicLorentzHGT, lorentz_distance, logmap0  # noqa: E402


def _load_bundle(bundle_root: Path, device: torch.device, max_edges: int | None):
    import pandas as pd

    features = pd.read_csv(bundle_root / "features_26d.csv")
    nodes = pd.read_csv(bundle_root / "nodes.csv")
    if "user_id" not in features or "node_id" not in nodes:
        raise ValueError("bundle must contain user_id features and node_id nodes")
    node_ids = features["user_id"].astype(str).tolist()
    node_map = {value: index for index, value in enumerate(node_ids)}
    feature_names = features.columns.drop("user_id").tolist()
    raw_x = features[feature_names].to_numpy("float32")
    availability_path = bundle_root / "feature_availability.csv"
    if availability_path.is_file():
        availability_frame = pd.read_csv(availability_path)
        if "node_id" not in availability_frame:
            raise ValueError("feature availability sidecar must contain node_id")
        availability_frame["node_id"] = availability_frame["node_id"].astype(str)
        availability_frame = availability_frame.set_index("node_id").reindex(node_ids)
        missing_columns = sorted(set(feature_names) - set(availability_frame.columns))
        if missing_columns:
            raise ValueError(f"feature availability missing columns: {missing_columns}")
        availability = availability_frame[feature_names].fillna(False).to_numpy(bool)
    else:
        availability = np.ones_like(raw_x, dtype=bool)
    mean = np.zeros((1, raw_x.shape[1]), dtype="float32")
    std = np.ones((1, raw_x.shape[1]), dtype="float32")
    for index in range(raw_x.shape[1]):
        observed = raw_x[availability[:, index], index]
        if observed.size:
            mean[0, index] = observed.mean()
            observed_std = observed.std()
            std[0, index] = observed_std if observed_std >= 1e-6 else 1.0
    normalized = (raw_x - mean) / std
    normalized[~availability] = 0.0
    x = torch.tensor(normalized, device=device)
    available = torch.tensor(availability, device=device)
    edges = pd.read_csv(bundle_root / "edges.csv")
    if {"source_id", "target_id"}.issubset(edges.columns):
        source = edges["source_id"].astype(str).map(node_map)
        target = edges["target_id"].astype(str).map(node_map)
        valid = source.notna() & target.notna()
        pairs = torch.from_numpy(
            np.stack([
                source[valid].to_numpy("int64"), target[valid].to_numpy("int64")
            ])
        ).to(
            device,
        )
    else:
        pairs = torch.empty((2, 0), dtype=torch.long, device=device)
    if max_edges is not None and pairs.shape[1] > max_edges:
        generator = torch.Generator(device=device).manual_seed(42)
        keep = torch.randperm(pairs.shape[1], generator=generator, device=device)[:max_edges]
        pairs = pairs[:, keep]
    graph = HeteroData()
    graph["user"].x = x
    graph["user", "coordination", "user"].edge_index = pairs
    return graph, pairs, available, mean.astype("float32"), std.astype("float32")


def run_pretraining(
    *, bundle_root: Path, output_dir: Path, device: str = "cuda",
    hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
    epochs: int = 1, max_steps: int | None = 1,
    mask_rate: float = 0.15, learning_rate: float = 1e-3,
    resume: bool = False, max_edges: int | None = 4096,
) -> dict:
    if not 0.0 < mask_rate < 1.0:
        raise ValueError("mask_rate must be between zero and one")
    torch_device = torch.device(device)
    graph, positive, available, feature_mean, feature_std = _load_bundle(
        bundle_root, torch_device, max_edges
    )
    metadata = graph.metadata()
    encoder = IntrinsicLorentzHGT(
        hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
        metadata=metadata, dropout=0.1,
    ).to(torch_device)
    decoder = torch.nn.Linear(hidden_dim, 26).to(torch_device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()], lr=learning_rate
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last_checkpoint.pt"
    start_epoch = 0
    steps_done = 0
    history = []
    generator = torch.Generator(device=torch_device).manual_seed(2026)
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        steps_done = int(state["steps"])
        history = list(state.get("history", []))
        if state.get("generator_state") is not None:
            generator.set_state(state["generator_state"].cpu())

    edge_set = {(int(a), int(b)) for a, b in positive.t().tolist()}
    for epoch in range(start_epoch, epochs):
        encoder.train(); decoder.train()
        masked = available & (
            torch.rand(graph["user"].x.shape, generator=generator, device=torch_device)
            < mask_rate
        )
        masked_x = graph["user"].x.masked_fill(masked, 0.0)
        points = encoder.encode_lorentz_nodes(
            {"user": masked_x}, graph.edge_index_dict
        )["user"]
        tangent = logmap0(points, encoder.common_curvature())
        reconstruction = decoder(tangent)
        feature_loss = F.mse_loss(reconstruction[masked], graph["user"].x[masked]) if masked.any() else tangent.sum() * 0

        if positive.shape[1]:
            count = min(positive.shape[1], 1024)
            keep = torch.randperm(positive.shape[1], generator=generator, device=torch_device)[:count]
            pos = positive[:, keep]
            neg_src = pos[0]
            neg_dst = torch.randint(graph["user"].num_nodes, (count,), generator=generator, device=torch_device)
            for index in range(count):
                if (int(neg_src[index]), int(neg_dst[index])) in edge_set:
                    neg_dst[index] = (neg_dst[index] + 1) % graph["user"].num_nodes
            curvature = encoder.common_curvature()
            pos_score = -lorentz_distance(points[pos[0]], points[pos[1]], curvature).squeeze(-1)
            neg_score = -lorentz_distance(points[neg_src], points[neg_dst], curvature).squeeze(-1)
            link_loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_score, neg_score]),
                torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
            )
        else:
            link_loss = feature_loss * 0
        loss = feature_loss + 0.1 * link_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 5.0)
        optimizer.step()
        steps_done += 1
        history.append({"epoch": epoch + 1, "step": steps_done,
                        "loss": float(loss.detach().cpu()),
                        "feature_loss": float(feature_loss.detach().cpu()),
                        "link_loss": float(link_loss.detach().cpu())})
        torch.save({"schema_version": "hypertrace.lorentz-pretrain.v1",
                    "epoch": epoch + 1, "steps": steps_done,
                    "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(), "history": history,
                    "generator_state": generator.get_state(),
                    "bundle_root": str(bundle_root.resolve()),
                    "feature_mean": feature_mean.tolist(),
                    "feature_std": feature_std.tolist(),
                    "edge_count": int(positive.shape[1]),
                    "available_feature_values": int(available.sum().item())}, checkpoint_path)
        if max_steps is not None and steps_done >= max_steps:
            break
    result = {"status": "passed", "resumed": resume and start_epoch > 0,
              "epochs": len(history), "steps": steps_done,
              "history": history, "checkpoint": str(checkpoint_path),
              "device": str(torch_device), "edge_count": int(positive.shape[1]),
              "available_feature_values": int(available.sum().item())}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-edges", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run_pretraining(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
