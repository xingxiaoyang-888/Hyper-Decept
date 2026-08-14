"""Generate label-blind geometry, temporal, provenance and deletion evidence packets."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "Character Classification"
for value in (ROOT, CHARACTER_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from lorentz_hgt import lorentz_distance, lorentz_to_poincare
from scripts.external_bundle_episode import load_external_bundle_episode
from scripts.frozen_manifest import resolve_frozen_checkpoint
from scripts.run_frozen_external_io import _build_model, _checkpoint_edge_types, _sha256


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert inference output even when it is derived from trainable parameters."""
    return tensor.detach().cpu().numpy()


def _percentile(scores: np.ndarray, ids: np.ndarray, index: int) -> float:
    value, uid = scores[index], ids[index]
    position = int(np.count_nonzero(scores < value)) + int(np.count_nonzero((scores == value) & (ids <= uid)))
    return position / max(len(scores), 1)

def _incident_mask(graph, user_index: int) -> dict:
    masks = {}
    for edge_type, edge_index in graph.edge_index_dict.items():
        keep = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
        source_type, _, target_type = edge_type
        incident = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        if source_type == "user": incident |= edge_index[0] == user_index
        if target_type == "user": incident |= edge_index[1] == user_index
        keep[incident] = 0.0
        masks[edge_type] = keep
    return masks

def _source_evidence(bundle: Path, user_ids: list[str]) -> tuple[dict, dict]:
    selected = set(user_ids)
    events = []
    event_counts = {uid: {} for uid in user_ids}
    first_last = {uid: [None, None] for uid in user_ids}
    for chunk in pd.read_csv(bundle / "events.csv", chunksize=200_000, low_memory=False):
        actor = chunk["actor_id"].astype(str)
        hit = chunk.loc[actor.isin(selected)].copy()
        if hit.empty: continue
        hit["actor_id"] = hit["actor_id"].astype(str)
        hit["timestamp_dt"] = pd.to_datetime(hit["timestamp"], errors="coerce", utc=True)
        for uid, group in hit.groupby("actor_id", sort=False):
            counts = group["event_type"].astype(str).value_counts()
            for name, count in counts.items(): event_counts[uid][name] = event_counts[uid].get(name, 0) + int(count)
            valid = group["timestamp_dt"].dropna()
            if not valid.empty:
                lo, hi = valid.min().isoformat(), valid.max().isoformat()
                first_last[uid][0] = min(x for x in (first_last[uid][0], lo) if x is not None)
                first_last[uid][1] = max(x for x in (first_last[uid][1], hi) if x is not None)
        events.append(hit)
    event_frame = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    edge_frames = []
    relation_counts = {uid: {} for uid in user_ids}
    for chunk in pd.read_csv(bundle / "edges.csv", chunksize=250_000, low_memory=False):
        source_ids, target_ids = chunk["source_id"].astype(str), chunk["target_id"].astype(str)
        source_hit = (chunk["source_type"].astype(str) == "user") & source_ids.isin(selected)
        target_hit = (chunk["target_type"].astype(str) == "user") & target_ids.isin(selected)
        hit = chunk.loc[source_hit | target_hit].copy()
        if hit.empty: continue
        hit["source_id"] = hit["source_id"].astype(str); hit["target_id"] = hit["target_id"].astype(str)
        for uid in user_ids:
            owned = hit.loc[((hit["source_type"] == "user") & (hit["source_id"] == uid)) | ((hit["target_type"] == "user") & (hit["target_id"] == uid))]
            for name, count in owned["edge_type"].astype(str).value_counts().items(): relation_counts[uid][name] = relation_counts[uid].get(name, 0) + int(count)
        edge_frames.append(hit)
    edge_frame = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    packets = {}
    for uid in user_ids:
        ev = event_frame.loc[event_frame.get("actor_id", pd.Series(dtype=str)) == uid].copy() if not event_frame.empty else pd.DataFrame()
        if not ev.empty: ev = ev.sort_values("timestamp_dt", ascending=False).head(10)
        ed = edge_frame.loc[((edge_frame["source_type"] == "user") & (edge_frame["source_id"] == uid)) | ((edge_frame["target_type"] == "user") & (edge_frame["target_id"] == uid))].copy() if not edge_frame.empty else pd.DataFrame()
        if not ed.empty: ed = ed.sort_values("timestamp", ascending=False).head(20)
        packets[uid] = {
            "temporal_summary": {"event_type_counts": event_counts[uid], "first_event_at": first_last[uid][0], "last_event_at": first_last[uid][1], "decision_time_scope": "full_release_snapshot"},
            "relation_summary": relation_counts[uid],
            "evidence": [{k: (None if pd.isna(v) else v) for k, v in row.items() if k != "timestamp_dt"} for row in ev.to_dict("records")],
            "critical_edges": [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in ed.to_dict("records")],
        }
    return packets, {"events_rows_retained": len(event_frame), "edges_rows_retained": len(edge_frame)}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--consensus-scores", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-n", type=int, default=3)
    args = p.parse_args()
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8")); entries = freeze["checkpoints"]
    consensus_frame = pd.read_csv(args.consensus_scores)
    consensus_frame["user_id"] = consensus_frame["user_id"].astype(str)
    if consensus_frame["user_id"].duplicated().any():
        raise ValueError("consensus reference universe contains duplicate user IDs")
    reference_ids = consensus_frame["user_id"].to_numpy(dtype=str)
    scores_frame = consensus_frame.sort_values("consensus_rank").head(args.top_n).copy()
    selected_ids = scores_frame["user_id"].astype(str).tolist()
    score_lookup = scores_frame.set_index("user_id").to_dict("index")
    batch = load_external_bundle_episode(args.bundle, load_labels=False)
    if torch.any(batch.bot_mask): raise ValueError("label-blind explanation loader unexpectedly exposed labels")
    device = torch.device(args.device); batch.to(device)
    all_ids = np.asarray(batch.graph["user"].node_ids, dtype=str)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("external bundle contains duplicate user IDs")
    index_lookup = {uid: i for i, uid in enumerate(all_ids)}
    missing_reference_ids = sorted(set(reference_ids) - set(index_lookup))
    if missing_reference_ids:
        raise ValueError(
            f"consensus reference universe has {len(missing_reference_ids)} IDs absent from bundle"
        )
    reference_indices = np.asarray([index_lookup[uid] for uid in reference_ids], dtype=np.int64)
    reference_lookup = {uid: i for i, uid in enumerate(reference_ids)}
    selected_indices = [index_lookup[uid] for uid in selected_ids]
    geometry: dict[str, list[dict[str, float]]] = {uid: [] for uid in selected_ids}
    counterfactual: dict[str, list[dict[str, float]]] = {uid: [] for uid in selected_ids}
    metadata = None
    for entry in entries:
        path = resolve_frozen_checkpoint(args.freeze_manifest, entry)
        if _sha256(path) != entry["sha256"]: raise ValueError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if metadata is None: metadata = (tuple(sorted(batch.graph.node_types)), _checkpoint_edge_types(checkpoint["model"]))
        model = _build_model(checkpoint, metadata).to(device).eval()
        with torch.no_grad(): baseline = model(batch.graph, domain=batch.domain, dataset_name=batch.dataset_name)
        probabilities = torch.sigmoid(baseline["bot_logits"]).cpu().numpy()
        curvature = model.encoder.common_curvature(); points = baseline["user_lorentz"][selected_indices]
        poincare = lorentz_to_poincare(points, curvature)
        prototypes = model.bot_head.prototypes(curvature)
        distances = _to_numpy(
            lorentz_distance(
                points.unsqueeze(1), prototypes.unsqueeze(0), curvature
            ).squeeze(-1)
        )
        for local, (uid, global_index) in enumerate(zip(selected_ids, selected_indices)):
            base_pct = _percentile(
                probabilities[reference_indices], reference_ids, reference_lookup[uid]
            )
            geometry[uid].append({"poincare_radius": float(torch.linalg.vector_norm(poincare[local]).cpu()), "distance_to_normal": float(distances[local, 0]), "distance_to_coordination": float(distances[local, 1]), "geodesic_margin": float(distances[local, 0] - distances[local, 1]), "risk_probability": float(probabilities[global_index])})
            mask = _incident_mask(batch.graph, global_index)
            with torch.no_grad(): removed = model(batch.graph, domain=batch.domain, dataset_name=batch.dataset_name, edge_mask_dict=mask)
            removed_prob = torch.sigmoid(removed["bot_logits"]).cpu().numpy()
            counterfactual[uid].append({"baseline_percentile": base_pct, "removed_percentile": _percentile(removed_prob[reference_indices], reference_ids, reference_lookup[uid]), "baseline_probability": float(probabilities[global_index]), "removed_probability": float(removed_prob[global_index])})
            del removed, removed_prob, mask
        del model, baseline, probabilities, checkpoint
        if device.type == "cuda": torch.cuda.empty_cache()
    source_packets, source_audit = _source_evidence(args.bundle, selected_ids)
    packets = []
    for uid in selected_ids:
        geo, cf = geometry[uid], counterfactual[uid]
        recomputed_percentile = float(np.mean([x["baseline_percentile"] for x in cf]))
        frozen_percentile = float(score_lookup[uid]["consensus_percentile"])
        percentile_match_error = abs(recomputed_percentile - frozen_percentile)
        match_tolerance = max(1e-6, 2.0 / max(len(reference_ids), 1))
        if percentile_match_error > match_tolerance:
            raise ValueError(
                f"baseline consensus mismatch for {uid}: frozen={frozen_percentile}, "
                f"recomputed={recomputed_percentile}, tolerance={match_tolerance}"
            )
        packet = {
            "schema_version": "hypertrace.consensus-evidence-packet.v1", "case_id": f"user:{uid}",
            "operation": args.bundle.name, "labels_consumed": False,
            "prediction": {"consensus_percentile": frozen_percentile, "consensus_rank": int(score_lookup[uid]["consensus_rank"]), "checkpoint_rank_std": float(score_lookup[uid]["checkpoint_rank_std"]), "recomputed_baseline_percentile": recomputed_percentile, "baseline_percentile_match_error": percentile_match_error, "decision": "review_candidate", "probability_calibrated": False},
            "geometry": {"checkpoint_count": len(geo), "poincare_radius_mean": float(np.mean([x["poincare_radius"] for x in geo])), "poincare_radius_std": float(np.std([x["poincare_radius"] for x in geo])), "geodesic_margin_mean": float(np.mean([x["geodesic_margin"] for x in geo])), "geodesic_margin_std": float(np.std([x["geodesic_margin"] for x in geo])), "coordination_vote_fraction": float(np.mean([x["geodesic_margin"] > 0 for x in geo])), "coordinate_averaging_disabled": True},
            "counterfactual": {"intervention": "remove all incident learned-relation edges for this account", "consensus_percentile_before": float(np.mean([x["baseline_percentile"] for x in cf])), "consensus_percentile_after": float(np.mean([x["removed_percentile"] for x in cf])), "consensus_percentile_change": float(np.mean([x["removed_percentile"] - x["baseline_percentile"] for x in cf])), "mean_probability_change": float(np.mean([x["removed_probability"] - x["baseline_probability"] for x in cf])), "checkpoint_count": len(cf)},
            **source_packets[uid],
            "reference_universe": {"source": "consensus_scores.user_id", "account_count": len(reference_ids), "target_label_values_consumed": False, "eligibility_membership_consumed": True},
            "provenance": {"bundle_manifest": str((args.bundle / "manifest.json").resolve()), "bundle_manifest_sha256": _sha256(args.bundle / "manifest.json"), "freeze_manifest": str(args.freeze_manifest.resolve()), "freeze_manifest_sha256": _sha256(args.freeze_manifest), "consensus_scores": str(args.consensus_scores.resolve()), "consensus_scores_sha256": _sha256(args.consensus_scores)},
            "warnings": ["Target labels were not loaded.", "Evidence covers the full release snapshot, not a historical online decision cutoff.", "Poincare coordinates are not averaged across independently trained model coordinate frames."],
        }
        packets.append(packet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "packets.jsonl"
    with packet_path.open("w", encoding="utf-8") as f:
        for packet in packets: f.write(json.dumps(packet, ensure_ascii=False) + "\n")
    manifest = {"schema_version": "hypertrace.consensus-evidence-preflight.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "packet_count": len(packets), "labels_consumed": False, "target_label_values_consumed": False, "eligibility_membership_consumed": True, "reference_universe_accounts": len(reference_ids), "top_n": args.top_n, "packets": str(packet_path.resolve()), "packets_sha256": _sha256(packet_path), "source_audit": source_audit}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
