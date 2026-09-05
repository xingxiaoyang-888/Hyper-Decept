"""Combine operation-level explanation controls with stratified bootstraps."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("hypertrace", "static_topk", "random_edge", "degree_matched")
METRICS = (
    "sufficiency_percentile_error",
    "geometry_fidelity",
    "prototype_vote_agreement",
    "comprehensiveness",
    "selection_jaccard_with_hypertrace",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    value = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _load(run_dir: Path) -> tuple[dict, pd.DataFrame, Path]:
    audit_path = run_dir / "audit.json"
    rows_path = run_dir / "case_replicates.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("labels_consumed") is not False:
        raise ValueError(f"unusable control audit: {audit_path}")
    frame = pd.read_json(rows_path, lines=True)
    if frame["labels_consumed"].any():
        raise ValueError(f"case rows consumed labels: {rows_path}")
    return audit, frame, rows_path


def _stratified_bootstrap(
    per_case: pd.DataFrame, metric: str, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = per_case.pivot(
        index=["operation", "case_id"], columns="method", values=metric
    )
    if tuple(sorted(pivot.columns)) != tuple(sorted(METHODS)):
        raise ValueError(f"missing methods for {metric}: {pivot.columns.tolist()}")
    operation_indices = {
        operation: np.flatnonzero(
            pivot.index.get_level_values("operation") == operation
        )
        for operation in pivot.index.get_level_values("operation").unique()
    }
    rng = np.random.default_rng(_stable_seed(seed, metric, "stratified"))
    boot_means = {method: np.empty(draws) for method in METHODS}
    for draw in range(draws):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in operation_indices.values()
        ])
        for method in METHODS:
            boot_means[method][draw] = pivot[method].to_numpy()[sampled].mean()
    metric_rows = []
    values = {method: pivot[method].to_numpy(dtype=float) for method in METHODS}
    for method in METHODS:
        metric_rows.append({
            "metric": metric,
            "method": method,
            "n_cases": len(pivot),
            "mean": values[method].mean(),
            "median": np.median(values[method]),
            "stratified_case_bootstrap_ci95_low": np.quantile(
                boot_means[method], 0.025
            ),
            "stratified_case_bootstrap_ci95_high": np.quantile(
                boot_means[method], 0.975
            ),
        })
    contrast_rows = []
    for control in METHODS[1:]:
        difference = values["hypertrace"] - values[control]
        boot_difference = boot_means["hypertrace"] - boot_means[control]
        contrast_rows.append({
            "metric": metric,
            "contrast": f"hypertrace_minus_{control}",
            "n_cases": len(pivot),
            "mean_paired_difference": difference.mean(),
            "stratified_case_bootstrap_ci95_low": np.quantile(
                boot_difference, 0.025
            ),
            "stratified_case_bootstrap_ci95_high": np.quantile(
                boot_difference, 0.975
            ),
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--honduras-dir", required=True, type=Path)
    parser.add_argument("--uae-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    frames = []
    for expected_operation, run_dir in (
        ("honduras", args.honduras_dir),
        ("uae", args.uae_dir),
    ):
        audit, frame, rows_path = _load(run_dir.resolve())
        if audit.get("operation") != expected_operation:
            raise ValueError(f"expected {expected_operation}, got {audit.get('operation')}")
        inputs.append({
            "operation": expected_operation,
            "audit": str((run_dir / "audit.json").resolve()),
            "audit_sha256": _sha256((run_dir / "audit.json").resolve()),
            "case_replicates": str(rows_path.resolve()),
            "case_replicates_sha256": _sha256(rows_path.resolve()),
        })
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    per_case = combined.groupby(
        ["operation", "case_id", "method"], as_index=False
    )[list(METRICS)].mean()

    metric_frames = []
    contrast_frames = []
    for metric in METRICS:
        metric_frame, contrast_frame = _stratified_bootstrap(
            per_case, metric, args.bootstrap_draws, args.seed
        )
        metric_frames.append(metric_frame)
        contrast_frames.append(contrast_frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    contrasts = pd.concat(contrast_frames, ignore_index=True)
    metrics_path = args.output_dir / "combined_method_metrics.csv"
    contrasts_path = args.output_dir / "combined_paired_contrasts.csv"
    per_case_path = args.output_dir / "combined_per_case_method_means.csv"
    metrics.to_csv(metrics_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    per_case.to_csv(per_case_path, index=False)

    audit = {
        "schema_version": "hypertrace.explanation-controls.combined.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "labels_consumed": False,
        "operations": ["honduras", "uae"],
        "case_count": int(per_case[["operation", "case_id"]].drop_duplicates().shape[0]),
        "bootstrap": "case resampling stratified within operation",
        "bootstrap_draws": args.bootstrap_draws,
        "inputs": inputs,
        "artifacts": {
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": _sha256(metrics_path),
            "contrasts": str(contrasts_path.resolve()),
            "contrasts_sha256": _sha256(contrasts_path),
            "per_case": str(per_case_path.resolve()),
            "per_case_sha256": _sha256(per_case_path),
        },
    }
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
