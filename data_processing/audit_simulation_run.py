"""Write a compact, reproducible audit manifest for a completed simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rag-debug", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-total", type=int)
    parser.add_argument("--api-success", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    db = args.db.expanduser().resolve()
    source_csv = args.csv.expanduser().resolve()
    config = args.config.expanduser().resolve()
    rag_debug = args.rag_debug.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()

    conn = sqlite3.connect(db)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    table_counts = {
        table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in tables
    }
    role_counts = dict(
        conn.execute("SELECT user_type, COUNT(*) FROM user GROUP BY user_type")
    )
    action_counts = dict(
        conn.execute("SELECT action, COUNT(*) FROM trace GROUP BY action")
    )
    db_agents = conn.execute(
        "SELECT COUNT(DISTINCT agent_id) FROM user WHERE agent_id IS NOT NULL"
    ).fetchone()[0]
    persona_nonempty = conn.execute(
        "SELECT COUNT(*) FROM user WHERE trim(coalesce(user_char,'')) <> ''"
    ).fetchone()[0]
    conn.close()

    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    rag_text = rag_debug.read_text(encoding="utf-8")
    rag_agents = [int(value) for value in re.findall(r"=== Agent (\d+)", rag_text)]
    rag_yes = len(re.findall(r"Injected: YES", rag_text))
    rag_no = len(re.findall(r"Injected: NO", rag_text))
    artifact_files = sorted(path for path in artifacts.rglob("*") if path.is_file())
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"

    manifest = {
        "status": "completed",
        "git_commit": git_commit,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "inputs": {
            "csv": str(source_csv),
            "config": str(config),
            "csv_sha256": sha256(source_csv),
            "config_sha256": sha256(config),
        },
        "database": {
            "path": str(db),
            "sha256": sha256(db),
            "integrity_check": integrity,
            "agents": db_agents,
            "personas_nonempty": persona_nonempty,
            "roles": role_counts,
            "table_counts": table_counts,
            "trace_actions": action_counts,
        },
        "csv": {"rows": len(csv_rows), "unique_agent_ids": len({r['user_id'] for r in csv_rows})},
        "deep_persona_rag": {
            "debug_path": str(rag_debug),
            "active_agent_ids": rag_agents,
            "injected_yes": rag_yes,
            "injected_no": rag_no,
        },
        "api": {
            "total_requests": args.api_total,
            "successful_requests": args.api_success,
            "failed_requests": (
                args.api_total - args.api_success
                if args.api_total is not None and args.api_success is not None
                else None
            ),
        },
        "artifacts": {
            "directory": str(artifacts),
            "file_count": len(artifact_files),
            "nonempty_file_count": sum(path.stat().st_size > 2 for path in artifact_files),
            "key_files": [
                str(path)
                for path in artifact_files
                if path.name in {
                    "all_agent_raw_data.json",
                    "detection_agent_data.pkl",
                    "task_blackboard_audit.json",
                }
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
