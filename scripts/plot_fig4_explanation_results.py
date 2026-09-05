"""Plot the currently frozen HyperTrace explanation results for CHI Fig. 4.

The formal archive contains 24 audited cases, of which the eight high-priority
cases have non-trivial candidate neighborhoods.  This figure uses those eight
cases only for compression and fidelity plots.  Random-edge and degree-matched
controls are intentionally not fabricated here; a third comparison panel can
be added once their frozen artifacts exist.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


NAVY = "#17324D"
INK = "#1F2933"
MUTED = "#52606D"
GRID = "#CBD5E1"
HONDURAS = "#2F6B9A"
UAE = "#D1495B"


def _load_high_cases(archive: Path) -> list[dict]:
    rows: list[dict] = []
    with tarfile.open(archive, "r:gz") as handle:
        for operation in ("honduras", "uae"):
            member_name = (
                "runtime/p2_formal_package/explanations/"
                f"constrained_evidence_relation_formal12/{operation}/cases.jsonl"
            )
            member = handle.getmember(member_name)
            stream = handle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unable to read {member_name}")
            for raw in io.TextIOWrapper(stream, encoding="utf-8"):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("risk_stratum") == "high":
                    row["operation"] = operation
                    rows.append(row)
    if len(rows) != 8:
        raise ValueError(f"Expected eight high-priority cases, found {len(rows)}")
    return rows


def _case_label(row: dict, index_by_operation: dict[str, int]) -> str:
    operation = str(row["operation"])
    index_by_operation[operation] += 1
    return f"{'H' if operation == 'honduras' else 'U'}{index_by_operation[operation]}"


def plot(rows: list[dict], output: Path) -> dict[str, float]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    operation_index = {"honduras": 0, "uae": 0}
    labels = [_case_label(row, operation_index) for row in rows]
    colors = [HONDURAS if row["operation"] == "honduras" else UAE for row in rows]
    retention = np.asarray([100 * float(row["selection"]["sparsity"]) for row in rows])
    sufficiency = np.asarray(
        [1000 * float(row["prediction"]["sufficiency_percentile_error"]) for row in rows]
    )
    geometry = np.asarray([float(row["prediction"]["geometry_fidelity"]) for row in rows])
    median_retention = float(np.median(retention))
    median_sufficiency = float(np.median(sufficiency))
    median_geometry = float(np.median(geometry))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), gridspec_kw={"wspace": 0.34})
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.31, top=0.82)
    fig.suptitle(
        "Compact evidence preserves assessment and geometry",
        fontsize=10.8,
        weight="bold",
        color=NAVY,
        y=1.02,
    )

    ax = axes[0]
    ax.scatter(retention, sufficiency, s=44, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
    for x, y, label in zip(retention, sufficiency, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=MUTED)
    ax.axvline(median_retention, color=GRID, linestyle=(0, (3, 2)), linewidth=1)
    ax.axhline(median_sufficiency, color=GRID, linestyle=(0, (3, 2)), linewidth=1)
    ax.set_title("(a) Compression and sufficiency", loc="left", color=NAVY, weight="bold")
    ax.set_xlabel("Evidence retained (% of candidate units)")
    ax.set_ylabel("Sufficiency error (×10⁻³ percentile units)")
    ax.set_xlim(left=0, right=max(11.5, float(retention.max()) * 1.12))
    ax.set_ylim(bottom=-0.25, top=max(4.25, float(sufficiency.max()) * 1.18))
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.55)
    ax.text(
        0.02,
        0.97,
        f"median: {median_retention:.2f}% retained\nerror: {median_sufficiency / 1000:.5f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID, "linewidth": 0.7},
    )

    ax = axes[1]
    positions = np.arange(len(rows))
    ax.scatter(positions, geometry, s=46, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
    for x, y, label in zip(positions, geometry, labels):
        ax.annotate(label, (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0.95, color=UAE, linestyle=(0, (3, 2)), linewidth=1, alpha=0.8)
    ax.axhline(median_geometry, color=GRID, linestyle=(0, (3, 2)), linewidth=1)
    ax.set_title("(b) Lorentz geometry fidelity", loc="left", color=NAVY, weight="bold")
    ax.set_xlabel("Audited non-trivial case")
    ax.set_ylabel("Geometry fidelity")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.95, 1.002)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.55)
    ax.text(0.03, 0.97, f"median: {median_geometry:.4f}", transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)
    ax.text(0.03, 0.03, "dashed: pre-specified 0.95 minimum", transform=ax.transAxes, fontsize=7, color=MUTED)

    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=HONDURAS, label="Honduras"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=UAE, label="UAE"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.235), ncol=2, frameon=False, fontsize=7.8)
    fig.text(
        0.5,
        0.035,
        "Eight high-priority cases with non-trivial candidate neighborhoods; audit coverage is 100% across all 24 cases.\n"
        "Random-edge and degree-matched controls are not shown because their frozen runs are not yet available.",
        ha="center",
        va="top",
        fontsize=7.4,
        color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "HyperTrace explanation fidelity",
        "Subject": "Evidence compression, sufficiency, and Lorentz geometry fidelity on eight non-trivial audited cases.",
        "Creator": "HyperTrace figure generator",
    }
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08, metadata=metadata)
    fig.savefig(
        output.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Title": metadata["Title"], "Creator": metadata["Creator"]},
    )
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {
        "case_count": float(len(rows)),
        "median_retention_percent": median_retention,
        "median_sufficiency_error": median_sufficiency / 1000,
        "median_geometry_fidelity": median_geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(".runtime-downloads/HyperTrace_constrained_evidence_formal_20260815.tar.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/generated/fig4_explanation_results"),
    )
    args = parser.parse_args()
    summary = plot(_load_high_cases(args.archive), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
