"""Build a path-free, aggregate-only CHI result package from passed audits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_chi_model_ablation import _latex_table


def _read_passed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError(f"audit is not passed: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _top_one_percent(audit: dict) -> dict:
    row = next(
        item for item in audit["review_budget_metrics"]
        if float(item["budget_fraction"]) == 0.01
    )
    return {
        key: row[key] for key in (
            "reviewed_accounts", "true_io_hits", "precision_at_k",
            "recall_at_k", "lift_at_k", "mean_pairwise_checkpoint_jaccard",
        )
    }


def _environment() -> dict:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch

        result.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        })
    except ImportError:
        result["torch"] = None
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def _markdown(results: dict) -> str:
    ablation = results["model_ablation"]["summaries"]
    labels = {
        "lorentz_observable18": "Lorentz-HGT + real warm-start",
        "lorentz_scratch_observable18": "Lorentz-HGT (scratch)",
        "euclidean_observable18": "Euclidean-HGT (scratch)",
        "lorentz_no_temporal": "Lorentz-HGT, no temporal input",
    }
    lines = [
        "# HyperTrace CHI Aggregate Results",
        "",
        "## Strict Unseen-Operation Evaluation",
        "",
        "| Target operation | AUROC | AUPRC |",
        "|---|---:|---:|",
    ]
    for name in ("honduras", "uae"):
        metrics = results["external_evaluation"][name]["metrics"]
        lines.append(
            f"| {name.title()} | {metrics['auroc']['mean']:.4f} +/- "
            f"{metrics['auroc']['std']:.4f} | {metrics['auprc']['mean']:.4f} "
            f"+/- {metrics['auprc']['std']:.4f} |"
        )
    lines.extend([
        "", "## Label-Blind Rank Consensus at 1% Review Budget", "",
        "| Target operation | Precision | Recall | Lift | Pairwise Jaccard |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("honduras", "uae"):
        row = results["rank_consensus"][name]["top_1_percent"]
        lines.append(
            f"| {name.title()} | {row['precision_at_k']:.4f} | "
            f"{row['recall_at_k']:.4f} | {row['lift_at_k']:.2f}x | "
            f"{row['mean_pairwise_checkpoint_jaccard']:.4f} |"
        )
    lines.extend([
        "", "## Mechanism Ablation (15 paired scenario/seed folds per variant)", "",
        "| Variant | AUPRC | AUROC | Macro-F1 | Balanced accuracy | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in labels.items():
        values = ablation[key]
        cells = [f"{values[metric]['mean']:.4f} +/- {values[metric]['std']:.4f}" for metric in (
            "auprc", "auroc", "f1", "balanced_accuracy", "brier", "ece"
        )]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend([
        "", "## Interpretation Boundaries", "",
        "- The scratch Euclidean model is higher on AUPRC, AUROC, balanced accuracy, and ECE; hyperbolic geometry is not claimed to universally improve predictive accuracy.",
        "- Temporal input has a small positive AUROC contribution; its paired intervals for the other reported metrics cross zero.",
        "- The real-graph Lorentz warm-start does not change AUPRC materially; improvements in F1, balanced accuracy, and calibration are trends whose paired intervals cross zero.",
        "- Explanation packets passed provenance and label-blind preflight, but the formal explanation sample and participant study remain separate required work.",
        "- Psychological features and role classification are excluded from every result in this package.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--honduras-consensus", required=True, type=Path)
    parser.add_argument("--uae-consensus", required=True, type=Path)
    parser.add_argument("--explanation", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    external = _read_passed(args.external)
    consensus = {
        "honduras": _read_passed(args.honduras_consensus),
        "uae": _read_passed(args.uae_consensus),
    }
    explanation = _read_passed(args.explanation)
    ablation = _read_passed(args.ablation)
    if ablation.get("fold_count") != 60 or ablation.get("expected_fold_count") != 60:
        raise ValueError("model ablation must contain all 60 audited folds")

    result = {
        "schema_version": "hypertrace.chi-public-results.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "aggregate_only_no_raw_data_no_checkpoints_no_target_labels",
        "excluded_modules": ["psychological_features", "role_classification"],
        "external_evaluation": {
            name: {
                "count": external[name]["count"],
                "evaluation_protocol": external[name]["evaluation_protocol"],
                "operation_disjoint": external[name]["operation_disjoint"],
                "metrics": external[name]["metrics"],
            }
            for name in ("honduras", "uae")
        },
        "rank_consensus": {
            name: {
                "protocol_status": consensus[name]["protocol_status"],
                "checkpoint_count": consensus[name]["checkpoint_count"],
                "labels_used_for_selection_scoring_or_ordering": consensus[name][
                    "target_labels_used_for_checkpoint_selection_scoring_or_ordering"
                ],
                "top_1_percent": _top_one_percent(consensus[name]),
                "stability": consensus[name]["stability"],
            }
            for name in ("honduras", "uae")
        },
        "explanation_preflight": {
            "status": explanation["status"],
            "labels_csv_opened": explanation["labels_csv_opened"],
            "operations": [
                {key: value for key, value in operation.items() if key != "artifacts"}
                | {"artifact_sha256": {
                    "manifest": operation["artifacts"]["manifest_sha256"],
                    "packets": operation["artifacts"]["packets_sha256"],
                }}
                for operation in explanation["operations"]
            ],
        },
        "model_ablation": {
            "fold_count": ablation["fold_count"],
            "summaries": ablation["summaries"],
            "paired_comparisons": ablation["paired_comparisons"],
        },
        "source_audit_sha256": {
            "external": _sha256(args.external),
            "honduras_consensus": _sha256(args.honduras_consensus),
            "uae_consensus": _sha256(args.uae_consensus),
            "explanation": _sha256(args.explanation),
            "ablation": _sha256(args.ablation),
        },
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if "/xingxiaoyang/" in serialized or "\\xingxiaoyang\\" in serialized:
        raise ValueError("public results contain an absolute server path")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paper_results.json": serialized,
        "paper_tables.md": _markdown(result),
        "table_model_ablation.tex": _latex_table(
            ablation["summaries"], ablation["paired_comparisons"]
        ),
        "environment.json": json.dumps(_environment(), ensure_ascii=False, indent=2) + "\n",
    }
    for name, content in outputs.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")
    checksum_lines = [
        f"{_sha256(args.output_dir / name)}  {name}" for name in sorted(outputs)
    ]
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": "passed", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
