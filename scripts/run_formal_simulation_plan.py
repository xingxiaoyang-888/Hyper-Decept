"""Run a formal simulation plan sequentially with resumable audit state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


SCHEMA_VERSION = "hyperdecept.formal-simulation-run.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_plan(path: Path) -> dict:
    plan_path = path.expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "hyperdecept.formal-simulation-plan.v1":
        raise ValueError("unsupported formal simulation plan schema")
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("formal simulation plan must contain configs")
    if payload.get("episodes") != len(configs):
        raise ValueError("formal simulation plan episode count does not match configs")
    for config in configs:
        if not Path(config).expanduser().resolve().is_file():
            raise FileNotFoundError(config)
    return payload


def run_plan(
    *,
    plan_path: Path,
    simulation_script: Path,
    state_path: Path,
    log_dir: Path,
    python_executable: str = sys.executable,
    max_episodes: int | None = None,
    dry_run: bool = False,
) -> dict:
    plan_path = plan_path.expanduser().resolve()
    simulation_script = simulation_script.expanduser().resolve()
    state_path = state_path.expanduser().resolve()
    log_dir = log_dir.expanduser().resolve()
    if not simulation_script.is_file():
        raise FileNotFoundError(simulation_script)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    plan = _load_plan(plan_path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported simulation run state schema")
        if Path(state["plan_path"]).resolve() != plan_path:
            raise ValueError("run state belongs to a different simulation plan")
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "plan_path": str(plan_path),
            "created_at": _utc_now(),
            "episodes": {},
        }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    completed_this_run = 0
    for config_value in plan["configs"]:
        config = Path(config_value).expanduser().resolve()
        episode_id = config.stem
        previous = state["episodes"].get(episode_id, {})
        if previous.get("status") == "completed":
            continue
        if max_episodes is not None and completed_this_run >= max_episodes:
            break
        command = [
            python_executable,
            str(simulation_script),
            "--config_path",
            str(config),
        ]
        if dry_run:
            state["episodes"][episode_id] = {
                "status": "dry_run",
                "config_path": str(config),
                "command": command,
            }
            completed_this_run += 1
            continue

        log_path = log_dir / f"{episode_id}.log"
        started_at = _utc_now()
        started = time.perf_counter()
        state["episodes"][episode_id] = {
            "status": "running",
            "config_path": str(config),
            "log_path": str(log_path),
            "command": command,
            "started_at": started_at,
        }
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = round(time.perf_counter() - started, 3)
        status = "completed" if result.returncode == 0 else "failed"
        state["episodes"][episode_id].update({
            "status": status,
            "returncode": result.returncode,
            "finished_at": _utc_now(),
            "elapsed_seconds": elapsed,
        })
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"simulation episode failed: {episode_id}; inspect {log_path}"
            )
        completed_this_run += 1

    statuses = [entry["status"] for entry in state["episodes"].values()]
    state["summary"] = {
        "planned": plan["episodes"],
        "completed": statuses.count("completed"),
        "failed": statuses.count("failed"),
        "dry_run": statuses.count("dry_run"),
        "remaining": plan["episodes"] - statuses.count("completed"),
        "updated_at": _utc_now(),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--simulation-script", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    state = run_plan(
        plan_path=args.plan,
        simulation_script=args.simulation_script,
        state_path=args.state,
        log_dir=args.log_dir,
        python_executable=args.python,
        max_episodes=args.max_episodes,
        dry_run=args.dry_run,
    )
    print(json.dumps(state["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
