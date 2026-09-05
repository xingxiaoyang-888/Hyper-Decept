"""Run one formal-scale timestep and decide whether a four-hour run is safe."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ERROR_MARKERS = (
    "HTTP/1.1 429",
    "Received response exception",
    "Traceback (most recent call last)",
    " error:",
)


def calibrate(
    *,
    config_path: Path,
    simulation_script: Path,
    output_dir: Path,
    max_step_seconds: float = 18.0,
    minimum_request_ratio: float = 0.90,
    python_executable: str = sys.executable,
) -> dict:
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise EnvironmentError("DEEPSEEK_API_KEY is not set")
    config_path = config_path.expanduser().resolve()
    simulation_script = simulation_script.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    # ``.venv/bin/python`` may be a symlink whose target is the system binary;
    # make it absolute without resolving the symlink, otherwise venv packages
    # (numpy, pandas, etc.) disappear from the child process.
    python_executable = str(Path(python_executable).expanduser().absolute())
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    calibration = json.loads(json.dumps(config))
    simulation = calibration["simulation"]
    simulation["num_timesteps"] = 1
    simulation.pop("cutoff_step", None)
    simulation["coverage_deadlines"] = [
        math.ceil(1 / float(simulation["target_active_fraction"]))
    ]
    simulation["activation_audit_path"] = str(
        (output_dir / "activation_audit.json").resolve()
    )
    simulation["export_debug_artifacts"] = False
    simulation["export_visualizations"] = False
    calibration["data"]["db_path"] = str(
        (output_dir / "calibration.db").resolve()
    )
    calibration_path = output_dir / "calibration.yaml"
    calibration_path.write_text(
        yaml.safe_dump(calibration, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    log_path = output_dir / "calibration.log"
    environment = os.environ.copy()
    environment["OASIS_FORMAL_MODE"] = "1"
    environment["OASIS_LOG_FULL_PROMPTS"] = "0"
    # The upstream OASIS package is a repository-local package rather than an
    # installed distribution. Make the subprocess independent of the caller's
    # current working directory.
    simulation_root = simulation_script.parents[3]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(simulation_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            [
                python_executable,
                str(simulation_script),
                "--config_path",
                str(calibration_path),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
            cwd=simulation_root,
        )
    elapsed = time.perf_counter() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    marker_hits = [marker for marker in ERROR_MARKERS if marker in log_text]
    audit_path = output_dir / "activation_audit.json"
    if not audit_path.is_file():
        report = {
            "schema_version": "hyperdecept.formal-calibration.v1",
            "status": "failed",
            "full_run_eligible": False,
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "initialization_seconds": round(elapsed, 3),
            "step_wall_seconds": None,
            "max_step_seconds": max_step_seconds,
            "selected_count": 0,
            "inference_requests": 0,
            "request_ratio": 0.0,
            "minimum_request_ratio": minimum_request_ratio,
            "error_markers": marker_hits or ["activation_audit_missing"],
            "projected_20_episode_hours": None,
            "action": "do_not_start_full_run; inspect calibration log",
            "log": str(log_path),
        }
        (output_dir / "calibration_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    step = audit["steps"][0]
    selected = int(step["selected_count"])
    requests = int(audit["inference_requests_at_export"])
    request_ratio = requests / selected if selected else 0.0
    step_seconds = float(step["wall_seconds"])
    init_seconds = max(0.0, elapsed - step_seconds)
    projected_hours = (
        step_seconds * 30 * 20 + init_seconds * 20
    ) / 3600
    eligible = (
        result.returncode == 0
        and not marker_hits
        and step_seconds <= max_step_seconds
        and request_ratio >= minimum_request_ratio
    )
    report = {
        "schema_version": "hyperdecept.formal-calibration.v1",
        "status": "passed" if result.returncode == 0 else "failed",
        "full_run_eligible": eligible,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "initialization_seconds": round(init_seconds, 3),
        "step_wall_seconds": step_seconds,
        "max_step_seconds": max_step_seconds,
        "selected_count": selected,
        "inference_requests": requests,
        "request_ratio": request_ratio,
        "minimum_request_ratio": minimum_request_ratio,
        "error_markers": marker_hits,
        "projected_20_episode_hours": projected_hours,
        "action": (
            "run_all_20_with_210_minute_soft_deadline"
            if eligible else
            "do_not_start_full_run; inspect calibration and use resumable batches"
        ),
        "log": str(log_path),
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--simulation-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-step-seconds", type=float, default=18.0)
    parser.add_argument("--minimum-request-ratio", type=float, default=0.90)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    report = calibrate(
        config_path=args.config,
        simulation_script=args.simulation_script,
        output_dir=args.output_dir,
        max_step_seconds=args.max_step_seconds,
        minimum_request_ratio=args.minimum_request_ratio,
        python_executable=args.python,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["full_run_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
