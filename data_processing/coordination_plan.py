"""Auditable corpus plan for HyperTrace coordinated-behavior experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.coordination_adapter import load_dataset


SCHEMA_VERSION = "hypertrace.coordination-corpus-plan.v1"
PATH_CONTRACT = "hypertrace.manifest-relative.v1"
ROLES = {
    "real_campaign_adaptation",
    "real_topology_pretraining",
    "real_cib_external_evaluation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_partition(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def campaign_disjoint_split(
    campaign_ids: Iterable[str], *, seed: int = 42,
    train_fraction: float = 0.70, validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Assign whole campaigns deterministically; never split campaign rows."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below 1")
    values = sorted({str(value) for value in campaign_ids if str(value).strip()})
    if len(values) < 3:
        raise ValueError("campaign-disjoint splitting needs at least three campaigns")
    rows = []
    boundary = train_fraction + validation_fraction
    for value in values:
        score = _stable_partition(value, seed)
        split = "train" if score < train_fraction else (
            "validation" if score < boundary else "test"
        )
        rows.append({"campaign_id": value, "data_split": split, "split_seed": seed})
    frame = pd.DataFrame(rows)
    missing = {"train", "validation", "test"}.difference(frame["data_split"])
    if missing:
        raise ValueError(f"deterministic split produced empty partitions: {sorted(missing)}")
    return frame


@dataclass(frozen=True)
class CorpusEntry:
    dataset_id: str
    role: str
    raw_root: str
    audit_json: str
    materialized_root: str | None = None
    split_artifact: str | None = None
    supervision_tasks: tuple[str, ...] = ()
    llm_origin: str = "not_labelled"

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unsupported corpus role: {self.role}")
        if self.llm_origin not in {"not_labelled", "human", "llm", "mixed"}:
            raise ValueError("unsupported llm_origin")


@dataclass(frozen=True)
class CoordinationCorpusPlan:
    plan_id: str
    entries: tuple[CorpusEntry, ...]
    schema_version: str = SCHEMA_VERSION
    path_contract: str = PATH_CONTRACT

    def write(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        for entry in payload["entries"]:
            for key in ("raw_root", "audit_json", "materialized_root", "split_artifact"):
                value = entry.get(key)
                if value:
                    entry[key] = Path(os.path.relpath(
                        Path(value).resolve(), start=target.parent
                    )).as_posix()
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "CoordinationCorpusPlan":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        entries = []
        for value in payload["entries"]:
            value = dict(value)
            value["supervision_tasks"] = tuple(value.get("supervision_tasks") or ())
            for key in ("raw_root", "audit_json", "split_artifact"):
                path_value = value.get(key)
                if path_value:
                    candidate = Path(path_value)
                    if not candidate.is_absolute():
                        candidate = source.parent / candidate
                    value[key] = str(candidate.resolve())
            entries.append(CorpusEntry(**value))
        return cls(
            plan_id=payload["plan_id"], entries=tuple(entries),
            schema_version=payload["schema_version"],
            path_contract=payload["path_contract"],
        )

    def audit(self, *, require_files: bool = False) -> dict[str, Any]:
        errors, files = [], []
        for entry in self.entries:
            for name, value in (("raw_root", entry.raw_root),
                                ("audit_json", entry.audit_json),
                                ("materialized_root", entry.materialized_root),
                                ("split_artifact", entry.split_artifact)):
                if value is None:
                    continue
                path = Path(value)
                exists = path.is_dir() if name in {"raw_root", "materialized_root"} else path.is_file()
                if require_files and not exists:
                    errors.append(f"{entry.dataset_id} missing {name}: {path}")
                files.append({"dataset_id": entry.dataset_id, "name": name,
                              "path": str(path), "exists": exists})
            if entry.materialized_root:
                materialized = Path(entry.materialized_root)
                for filename in ("features_26d.csv", "feature_availability.csv",
                                 "nodes.csv", "edges.csv", "events.csv",
                                 "labels.csv", "manifest.json"):
                    path = materialized / filename
                    exists = path.is_file()
                    if require_files and not exists:
                        errors.append(f"{entry.dataset_id} missing materialized artifact: {path}")
                    files.append({"dataset_id": entry.dataset_id,
                                  "name": f"materialized/{filename}",
                                  "path": str(path), "exists": exists})
        return {"valid": not errors, "errors": errors, "files": files}


def build_plan(
    root: str | Path,
    audit_root: str | Path,
    materialized_root: str | Path,
) -> CoordinationCorpusPlan:
    root, audit_root = Path(root).resolve(), Path(audit_root).resolve()
    materialized_root = Path(materialized_root).resolve()
    crypto = load_dataset("crypto-campaign", root / "crypto_campaign")
    split_path = root / "crypto_campaign" / "derived" / "campaign_split_seed42.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_disjoint_split(crypto.events["campaign_id"], seed=42).to_csv(
        split_path, index=False
    )
    return CoordinationCorpusPlan(
        plan_id="hypertrace-coordination-main-v1",
        entries=(
            CorpusEntry("crypto-campaign", "real_campaign_adaptation",
                        str(root / "crypto_campaign"),
                        str(audit_root / "crypto_campaign_audit.json"),
                        materialized_root=str(materialized_root / "crypto_campaign"),
                        split_artifact=str(split_path),
                        supervision_tasks=("bounty_participation",)),
            CorpusEntry("uk2019-coordinated-behavior", "real_topology_pretraining",
                        str(root / "uk2019"), str(audit_root / "uk2019_audit.json"),
                        materialized_root=str(materialized_root / "uk2019"),
                        supervision_tasks=("topology_cluster_auxiliary",)),
            CorpusEntry("fake-accounts-activity", "real_cib_external_evaluation",
                        str(root / "fake_accounts"),
                        str(audit_root / "fake_accounts_audit.json"),
                        materialized_root=str(materialized_root / "fake_accounts"),
                        supervision_tasks=("cib_campaign_membership",)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.root, args.audit_root, args.materialized_root)
    report = plan.audit(require_files=args.require_files)
    output = plan.write(args.output)
    print(json.dumps({"plan": str(output), **report}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
