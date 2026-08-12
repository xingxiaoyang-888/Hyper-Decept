"""Freeze formal v5 best checkpoints with immutable provenance and hashes."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sorted(args.source.glob("*/seed_*/best_checkpoint.pt")):
        relative = source.relative_to(args.source)
        target = args.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o444)
        result = json.loads((source.parent / "metrics.json").read_text(encoding="utf-8"))
        entries.append({
            "held_out_scenario": result["held_out_scenario"], "seed": result["seed"],
            "source_checkpoint": str(source.resolve()), "frozen_checkpoint": str(target.resolve()),
            "sha256": sha256(target), "best_epoch": result["best_epoch"],
            "best_validation_auprc": result["best_validation_auprc"], "warm_start": result["warm_start"],
            "training_objective": result["training_objective"], "privileged_heads_used": result["privileged_heads_used"],
        })
    if len(entries) != 15:
        raise SystemExit(f"expected 15 formal checkpoints, found {len(entries)}")
    manifest = {
        "schema_version": "hypertrace.v5-frozen-checkpoint-manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(args.source.resolve()),
        "freeze_policy": "best_checkpoint_only; immutable 0444 copies; no parameter updates during external evaluation",
        "training_labels": "synthetic coordination membership only", "real_operation_labels_consumed": False,
        "operation_evaluation": "Honduras and UAE are read-only external bundles", "checkpoints": entries,
    }
    (args.destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "count": len(entries), "manifest": str((args.destination / "manifest.json").resolve())}, indent=2))

if __name__ == "__main__":
    main()
