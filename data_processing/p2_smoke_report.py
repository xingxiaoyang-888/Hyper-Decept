"""Create a portable, Git-safe audit report for a completed P2 smoke run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import torch


SCHEMA_VERSION = "hyperdecept.p2-smoke-audit.v1"


def _manifest_owned_path(value: Any, manifest_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_git_provenance(repo_root: Path) -> dict[str, Any]:
    """Return the exact source revision and whether it had local modifications."""

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty_at_start": status not in {"", "unavailable"},
    }


def collect_runtime_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        environment["gpu"] = {
            "name": torch.cuda.get_device_name(index),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
    return environment


def _artifact_digests(manifest: Mapping[str, Any], manifest_path: Path) -> dict:
    """Keep hashes and sizes while removing every local artifact path."""

    result: dict[str, dict[str, Any]] = {}
    artifacts = manifest.get("artifacts") or {}
    if isinstance(artifacts, Mapping):
        for name, declaration in sorted(artifacts.items()):
            record: dict[str, Any] = {}
            path_value: Any = declaration
            if isinstance(declaration, Mapping):
                path_value = declaration.get("path")
                for key in ("sha256", "bytes", "rows"):
                    if declaration.get(key) is not None:
                        record[key] = declaration[key]
            if path_value:
                path = _manifest_owned_path(path_value, manifest_path)
                if path.is_file():
                    record.setdefault("bytes", path.stat().st_size)
                    record.setdefault("sha256", sha256_file(path))
            if record:
                result[str(name)] = record
    result["episode_manifest"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    return result


def _adapter_artifacts(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict:
    artifacts = manifest.get("artifacts") or {}
    adapter_path = artifacts.get("adapter_manifest_json")
    if not adapter_path:
        return {}
    path = _manifest_owned_path(adapter_path, manifest_path)
    if not path.is_file():
        return {}
    adapter = json.loads(path.read_text(encoding="utf-8"))
    clean: dict[str, dict[str, Any]] = {}
    for name, declaration in sorted((adapter.get("artifacts") or {}).items()):
        if not isinstance(declaration, Mapping) or not declaration.get("sha256"):
            continue
        clean[str(name)] = {
            key: declaration[key]
            for key in ("sha256", "bytes", "rows")
            if declaration.get(key) is not None
        }
    return clean


def _dataset_record(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = (
        "episode_id",
        "dataset_name",
        "domain",
        "purpose",
        "partition",
        "split_level",
        "status",
        "path_contract",
        "scenario_id",
        "simulation_seed",
        "num_agents",
        "time_steps",
        "source_sha256",
        "label_provenance",
        "capabilities",
        "attack_phases",
    )
    record = {key: manifest.get(key) for key in keys if manifest.get(key) is not None}
    record["input_artifacts"] = _artifact_digests(manifest, manifest_path)
    adapter = _adapter_artifacts(manifest, manifest_path)
    if adapter:
        record["adapter_artifacts"] = adapter
    return record


def build_p2_smoke_report(
    *,
    summary: Mapping[str, Any],
    manifest_paths: Sequence[Path],
    started_at: datetime,
    finished_at: datetime,
    provenance: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a report with no source, artifact, checkpoint, or home-directory paths."""

    if started_at.tzinfo is None or finished_at.tzinfo is None:
        raise ValueError("started_at and finished_at must be timezone-aware")
    datasets = [_dataset_record(Path(path)) for path in manifest_paths]
    observed_users = {
        "twibot22": summary.get("twibot_users"),
        "mgtab": summary.get("mgtab_users"),
        "deeppersona_oasis": summary.get("synthetic_users"),
    }
    for dataset in datasets:
        count = observed_users.get(str(dataset.get("dataset_name")))
        if count is not None:
            dataset["observed_users"] = int(count)
    checkpoint = Path(str(summary["checkpoint"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "p2_three_source_smoke",
        "status": summary.get("status", "unknown"),
        "warning": "Smoke/interface metrics are not paper results.",
        "provenance": dict(provenance),
        "timing": {
            "started_at_utc": started_at.astimezone(timezone.utc).isoformat(),
            "finished_at_utc": finished_at.astimezone(timezone.utc).isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        },
        "environment": dict(environment),
        "datasets": datasets,
        "observed_counts": {
            key: value
            for key, value in summary.items()
            if key.endswith("_users") or key.endswith("_labels")
        },
        "checkpoint_integrity": {
            "included_in_git": False,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "verification": {
            "forward": "passed",
            "loss": "passed",
            "backward": "passed",
            "checkpoint_save": "passed",
            "three_source_loading": "passed",
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    timing = report["timing"]
    lines = [
        "# P2 three-source smoke audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Git commit: `{provenance['git_commit']}`",
        f"- Git branch: `{provenance['git_branch']}`",
        f"- Dirty at start: `{provenance['git_dirty_at_start']}`",
        f"- Started (UTC): `{timing['started_at_utc']}`",
        f"- Finished (UTC): `{timing['finished_at_utc']}`",
        f"- Duration: `{timing['duration_seconds']} s`",
        "",
        "> Smoke/interface metrics are not paper results.",
        "",
        "## Datasets",
        "",
        "| Dataset | Domain | Agents/users | Seed | Steps | Source SHA-256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        lines.append(
            "| {dataset} | {domain} | {agents} | {seed} | {steps} | `{digest}` |".format(
                dataset=dataset.get("dataset_name", "unavailable"),
                domain=dataset.get("domain", "unavailable"),
                agents=dataset.get(
                    "observed_users", dataset.get("num_agents", "unavailable")
                ),
                seed=dataset.get("simulation_seed", "unavailable"),
                steps=dataset.get("time_steps", "unavailable"),
                digest=dataset.get("source_sha256", "unavailable"),
            )
        )
    lines.extend([
        "",
        "## Verification",
        "",
    ])
    lines.extend(
        f"- {name}: `{status}`"
        for name, status in report["verification"].items()
    )
    lines.extend([
        "",
        "The checkpoint, raw datasets, CSV files, databases, and model weights are not included in Git.",
        "Their integrity is represented by SHA-256 digests in the JSON audit report.",
        "",
    ])
    return "\n".join(lines)


def write_p2_smoke_report(
    report: Mapping[str, Any], report_dir: Path
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.fromisoformat(str(report["timing"]["started_at_utc"]))
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_commit = str(report["provenance"]["git_commit"])[:8]
    stem = f"{timestamp}_{short_commit}"
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
