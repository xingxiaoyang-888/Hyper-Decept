"""Audit and summarize the paired CHI geometry and temporal ablations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


SCENARIOS = (
    "adaptive_evasion", "bridge_infiltration", "leader_amplifier",
    "persona_drift", "synchronized_boosting",
)
SEEDS = (7, 17, 27)
METRICS = ("auprc", "auroc", "f1", "balanced_accuracy", "brier", "ece")
VARIANTS = {
    "lorentz_observable18": {
        "model_variant": "lorentz_hgt",
        "feature_contract": "observable18",
        "graph_intervention": "original",
    },
    "lorentz_scratch_observable18": {
        "model_variant": "lorentz_hgt",
        "feature_contract": "observable18",
        "graph_intervention": "original",
        "real_init_required": False,
    },
    "euclidean_observable18": {
        "model_variant": "euclidean_hgt",
        "feature_contract": "observable18",
        "graph_intervention": "original",
        "real_init_required": False,
    },
    "lorentz_no_temporal": {
        "model_variant": "lorentz_hgt",
        "feature_contract": "observable17_no_temporal",
        "graph_intervention": "remove_temporal_information",
        "real_init_required": True,
    },
}
VARIANTS["lorentz_observable18"]["real_init_required"] = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = 2.1447866879 if len(values) == 15 else 1.96
    margin = critical * std / math.sqrt(max(len(values), 1))
    return {
        "n": len(values), "mean": mean, "std": std,
        "ci95_low": mean - margin, "ci95_high": mean + margin,
    }


def _paired_bootstrap(values: list[float], *, seed: int = 20260812) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(array.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_delta_above_zero": float(np.mean(samples > 0)),
    }


def _latex_table(summaries: dict, comparisons: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & AUPRC & AUROC & Macro-F1 & Bal. Acc. & Brier $\downarrow$ & ECE $\downarrow$ \\",
        r"\midrule",
    ]
    labels = {
        "lorentz_observable18": "Lorentz-HGT + real warm-start",
        "lorentz_scratch_observable18": "Lorentz-HGT (scratch)",
        "euclidean_observable18": "Euclidean-HGT (scratch)",
        "lorentz_no_temporal": "Lorentz-HGT, no temporal input",
    }
    for variant, label in labels.items():
        values = summaries[variant]
        cells = [f"{values[name]['mean']:.4f} $\\pm$ {values[name]['std']:.4f}" for name in METRICS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "", "% Paired deltas are stored in audit.json."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    errors: list[str] = []
    records: dict[tuple[str, str, int], dict[str, float]] = {}
    hashes: dict[str, str] = {}
    for variant, expected in VARIANTS.items():
        for scenario in SCENARIOS:
            for seed in SEEDS:
                fold = args.root / variant / scenario / f"seed_{seed}"
                paths = [fold / "metrics.json", fold / "best_checkpoint.pt", fold / "last_checkpoint.pt"]
                if not all(path.is_file() for path in paths):
                    errors.append(f"missing fold artifacts: {fold}")
                    continue
                payload = json.loads(paths[0].read_text(encoding="utf-8"))
                contract = payload.get("run_contract") or {}
                checks = {
                    "status": payload.get("status"),
                    "training_objective": payload.get("training_objective"),
                    "privileged_heads_used": payload.get("privileged_heads_used"),
                    "held_out_scenario": payload.get("held_out_scenario"),
                    "seed": payload.get("seed"),
                    "model_variant": payload.get("model_variant"),
                    "feature_contract": payload.get("feature_contract"),
                    "graph_intervention": payload.get("graph_intervention"),
                    "contract": contract,
                }
                required = {
                    "status": "passed", "training_objective": "class_balanced_coordination_only",
                    "privileged_heads_used": False, "held_out_scenario": scenario, "seed": seed,
                    **{key: value for key, value in expected.items() if key != "real_init_required"},
                }
                for key, value in required.items():
                    if checks.get(key) != value:
                        errors.append(f"{variant}/{scenario}/seed_{seed}: {key} mismatch")
                real_init = contract.get("real_init")
                if expected["real_init_required"] != bool(real_init):
                    errors.append(f"{variant}/{scenario}/seed_{seed}: real_init mismatch")
                if len(payload.get("history", [])) != args.epochs:
                    errors.append(f"{variant}/{scenario}/seed_{seed}: expected {args.epochs} epochs")
                test = payload.get("test") or {}
                if any(not _finite(test.get(metric)) for metric in METRICS):
                    errors.append(f"{variant}/{scenario}/seed_{seed}: missing or non-finite metrics")
                    continue
                records[(variant, scenario, seed)] = {metric: float(test[metric]) for metric in METRICS}
                for path in paths:
                    hashes[path.relative_to(args.root).as_posix()] = _sha256(path)
    summaries: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = [records[(variant, scenario, seed)] for scenario in SCENARIOS for seed in SEEDS if (variant, scenario, seed) in records]
        if len(rows) == 15:
            summaries[variant] = {metric: _summary([row[metric] for row in rows]) for metric in METRICS}
    comparisons: dict[str, dict] = {}
    for name, candidate in (
        ("scratch_lorentz_minus_scratch_euclidean", "euclidean_observable18"),
        ("temporal_input_contribution", "lorentz_no_temporal"),
    ):
        comparison: dict[str, dict] = {}
        for metric in METRICS:
            deltas = []
            for scenario in SCENARIOS:
                for seed in SEEDS:
                    base_variant = (
                        "lorentz_scratch_observable18"
                        if name == "scratch_lorentz_minus_scratch_euclidean"
                        else "lorentz_observable18"
                    )
                    base = records.get((base_variant, scenario, seed))
                    other = records.get((candidate, scenario, seed))
                    if base is not None and other is not None:
                        direction = -1.0 if metric in ("brier", "ece") else 1.0
                        deltas.append(direction * (base[metric] - other[metric]))
            if len(deltas) == 15:
                comparison[metric] = _paired_bootstrap(deltas)
        comparisons[name] = comparison
    warm_start_comparison: dict[str, dict] = {}
    for metric in METRICS:
        direction = -1.0 if metric in ("brier", "ece") else 1.0
        deltas = [
            direction * (
                records[("lorentz_observable18", scenario, seed)][metric]
                - records[("lorentz_scratch_observable18", scenario, seed)][metric]
            )
            for scenario in SCENARIOS for seed in SEEDS
            if ("lorentz_observable18", scenario, seed) in records
            and ("lorentz_scratch_observable18", scenario, seed) in records
        ]
        if len(deltas) == 15:
            warm_start_comparison[metric] = _paired_bootstrap(deltas)
    comparisons["real_warm_start_contribution"] = warm_start_comparison
    expected_fold_count = len(VARIANTS) * len(SCENARIOS) * len(SEEDS)
    status = "passed" if not errors and len(records) == expected_fold_count else "failed"
    result = {
        "schema_version": "hypertrace.chi-model-ablation-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "excluded_modules": ["psychological_features", "role_classification"],
        "fold_count": len(records), "expected_fold_count": expected_fold_count,
        "summaries": summaries, "paired_comparisons": comparisons,
        "errors": errors, "files": dict(sorted(hashes.items())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if status == "passed":
        (args.output_dir / "table_model_ablation.tex").write_text(_latex_table(summaries, comparisons), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
