#!/usr/bin/env python3
"""Audit the reproducible HyperTrace v4 ablation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SCENARIOS = (
    "adaptive_evasion", "bridge_infiltration", "leader_amplifier",
    "persona_drift", "synchronized_boosting",
)
SEEDS = (7, 17, 27)
VARIANTS = (
    "mlp_observable18", "euclidean_hgt_observable18",
    "lorentz_random_observable18", "lorentz_uk_observable18",
    "lorentz_uk_observable18_edge_shuffle",
)
METRICS = ("auprc", "auroc", "f1", "balanced_accuracy", "brier", "ece")


def _finite(value, where, errors):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            errors.append(f"non-finite {where}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{where}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{where}[{index}]", errors)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values):
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    # Two-sided Student-t critical value with n-1 degrees of freedom.
    t = {15: 2.1447866879}[n]
    margin = t * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std,
            "ci95_low": mean - margin, "ci95_high": mean + margin}


def audit(root: Path, output: Path):
    errors = []
    records = []
    hashes = {}
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                fold = root / variant / scenario / f"seed_{seed}"
                metrics_path = fold / "metrics.json"
                best_path = fold / "best_checkpoint.pt"
                last_path = fold / "last_checkpoint.pt"
                if not all(path.is_file() for path in (metrics_path, best_path, last_path)):
                    errors.append(f"missing artifacts: {fold}")
                    continue
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                _finite(payload, f"{variant}/{scenario}/{seed}", errors)
                contract = payload.get("run_contract") or {}
                expected = {
                    "status": "passed", "model_variant": (
                        "mlp" if variant.startswith("mlp_") else
                        "euclidean_hgt" if variant.startswith("euclidean_") else "lorentz_hgt"
                    ),
                    "feature_contract": "full26" if variant.endswith("full26") else "observable18",
                    "held_out_scenario": scenario, "seed": seed,
                    "training_objective": "class_balanced_coordination_only",
                    "privileged_heads_used": False,
                }
                for key, value in expected.items():
                    if payload.get(key) != value:
                        errors.append(f"{variant}/{scenario}/{seed}: {key}={payload.get(key)!r}, expected {value!r}")
                if len(payload.get("history", [])) != 50:
                    errors.append(f"{variant}/{scenario}/{seed}: history length != 50")
                test = payload.get("test") or {}
                for metric in METRICS:
                    if metric not in test:
                        errors.append(f"{variant}/{scenario}/{seed}: missing test.{metric}")
                records.append({"variant": variant, "scenario": scenario, "seed": seed,
                                **{metric: float(test.get(metric, float("nan"))) for metric in METRICS}})
                for path in (metrics_path, best_path, last_path):
                    hashes[path.relative_to(root).as_posix()] = _sha256(path)
    summaries = {}
    for variant in VARIANTS:
        group = [record for record in records if record["variant"] == variant]
        if len(group) != 15:
            continue
        summaries[variant] = {metric: _summary([record[metric] for record in group]) for metric in METRICS}
    result = {"schema_version": "hypertrace.v4-matrix-audit.v1",
              "status": "passed" if not errors else "failed",
              "root": str(root.resolve()), "fold_count": len(records),
              "expected_fold_count": len(VARIANTS) * len(SCENARIOS) * len(SEEDS),
              "summaries": summaries, "errors": errors, "files": dict(sorted(hashes.items()))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.root, args.output)["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
