"""Summarize formal v5 LOSO runs without touching checkpoints."""
from __future__ import annotations
import glob
import json
import os
import statistics

paths = sorted(glob.glob("runtime/p2_formal_package/output/base_formal_v5_honduras/*/*/metrics.json"))
rows = [(os.path.relpath(path), json.loads(open(path, encoding="utf-8").read())) for path in paths]
bad = []
for path, result in rows:
    warm = result.get("warm_start", {})
    if (
        result.get("status") != "passed"
        or not warm.get("enabled")
        or result.get("training_objective") != "class_balanced_coordination_only"
        or result.get("privileged_heads_used")
    ):
        bad.append(path)
print(json.dumps({
    "count": len(rows),
    "bad_contracts": bad,
    "mean_test_auprc": statistics.mean(r["test"]["auprc"] for _, r in rows),
    "mean_test_auroc": statistics.mean(r["test"]["auroc"] for _, r in rows),
    "mean_test_f1": statistics.mean(r["test"]["f1"] for _, r in rows),
    "runs": [
        {
            "path": path,
            "held_out_scenario": r["held_out_scenario"],
            "seed": r["seed"],
            "epochs": r["epochs"],
            "best_epoch": r["best_epoch"],
            "best_validation_auprc": r["best_validation_auprc"],
            "test": r["test"],
            "warm_start_source_operation": r["warm_start"].get("source_operation"),
            "privileged_heads_used": r["privileged_heads_used"],
            "training_objective": r["training_objective"],
        }
        for path, r in rows
    ],
}, ensure_ascii=False, indent=2))
