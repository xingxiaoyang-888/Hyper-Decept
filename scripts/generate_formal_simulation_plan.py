"""Generate auditable 2000-agent/30-step configs for the formal corpus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


SCENARIOS = (
    "leader_amplifier",
    "bridge_infiltration",
    "synchronized_boosting",
    "persona_drift",
    "adaptive_evasion",
)
SIMULATION_SEEDS = (11, 22, 33, 44)


def build_configs(
    *,
    output_dir: Path,
    csv_root: Path,
    db_root: Path,
    model_type: str = "deepseek-chat",
    endpoints: list[dict],
    num_agents: int = 2000,
    time_steps: int = 30,
    cutoff_step: int = 18,
    target_active_fraction: float = 0.075,
    max_tokens: int = 512,
) -> dict:
    if not 0 < cutoff_step < time_steps:
        raise ValueError("cutoff_step must be between 1 and time_steps - 1")
    if not 0 < target_active_fraction <= 1:
        raise ValueError("target_active_fraction must be in (0, 1]")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    for scenario in SCENARIOS:
        for seed in SIMULATION_SEEDS:
            episode_id = f"{scenario}_n{num_agents}_s{seed}"
            config = {
                "data": {
                    "db_path": str(db_root / scenario / f"seed_{seed}.db"),
                    "csv_path": str(csv_root / scenario / f"seed_{seed}.csv"),
                },
                "simulation": {
                    "scenario_id": scenario,
                    "num_timesteps": time_steps,
                    "cutoff_step": cutoff_step,
                    "clock_factor": 60,
                    "recsys_type": "random",
                    "reflection": False,
                    "shared_reflection": False,
                    "detection": False,
                    "activation_scale": 1.0,
                    "force_all_agents_active": False,
                    "activation_policy": "budgeted_activity",
                    "target_active_fraction": target_active_fraction,
                    "max_silent_steps": math.ceil(1 / target_active_fraction),
                    "wake_on_pending_task": True,
                    "task_wake_limit": 32,
                    "leader_slots": 4,
                    "activation_seed": seed,
                    "coverage_deadlines": [cutoff_step],
                    "export_debug_artifacts": False,
                    "export_visualizations": False,
                },
                "model": {
                    "num_agents": num_agents,
                    "model_random_seed": seed,
                    "cfgs": [{
                        "model_type": model_type,
                        "num": num_agents,
                        "server_url": "configured_by_inference_section",
                        "model_path": "openai",
                        "is_openai_model": False,
                        "stop_tokens": [],
                        "temperature": 0.0,
                    }],
                },
                "inference": {
                    "model_type": model_type,
                    "model_path": "openai",
                    "stop_tokens": [],
                    "timeout": 300,
                    "parallel_per_endpoint": 1,
                    "max_tokens": max_tokens,
                    "server_url": endpoints,
                },
            }
            path = output_dir / f"{episode_id}.yaml"
            path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            configs.append(str(path.resolve()))

    step_budget = math.ceil(num_agents * target_active_fraction)
    manifest = {
        "schema_version": "hyperdecept.formal-simulation-plan.v1",
        "scenarios": list(SCENARIOS),
        "simulation_seeds": list(SIMULATION_SEEDS),
        "episodes": len(configs),
        "num_agents": num_agents,
        "time_steps": time_steps,
        "cutoff_step": cutoff_step,
        "target_active_fraction": target_active_fraction,
        "max_tokens": max_tokens,
        "step_request_ceiling": step_budget,
        "episode_request_ceiling": step_budget * time_steps,
        "corpus_activation_ceiling": step_budget * time_steps * len(configs),
        "corpus_request_ceiling_no_retries": step_budget * time_steps * len(configs),
        "corpus_request_ceiling_with_retries": (
            2 * step_budget * time_steps * len(configs)
        ),
        "configs": configs,
        "population_commands": [
            {
                "scenario_id": scenario,
                "simulation_seed": seed,
                "csv_path": str(csv_root / scenario / f"seed_{seed}.csv"),
                "command": (
                    "python MultiAgent4Collusion-master/generate_simulation_csv.py "
                    f"--profiles $PROFILES --num-agents {num_agents} "
                    f"--scenario {scenario} --seed {seed} "
                    "--mean-following 30 "
                    f"--output {csv_root / scenario / f'seed_{seed}.csv'}"
                ),
            }
            for scenario in SCENARIOS
            for seed in SIMULATION_SEEDS
        ],
    }
    manifest_path = output_dir / "formal_simulation_plan.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--csv-root", required=True, type=Path)
    parser.add_argument("--db-root", required=True, type=Path)
    parser.add_argument("--model-type", default="deepseek-chat")
    parser.add_argument("--host", default="api.deepseek.com")
    parser.add_argument("--ports", default="443")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--target-active-fraction", type=float, default=0.075)
    parser.add_argument("--num-agents", type=int, default=2000)
    parser.add_argument("--time-steps", type=int, default=30)
    parser.add_argument("--cutoff-step", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    endpoints = [{
        "host": args.host,
        "ports": [int(value) for value in args.ports.split(",") if value.strip()],
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "base_url": args.base_url,
    }]
    print(json.dumps(build_configs(
        output_dir=args.output_dir,
        csv_root=args.csv_root,
        db_root=args.db_root,
        model_type=args.model_type,
        endpoints=endpoints,
        num_agents=args.num_agents,
        time_steps=args.time_steps,
        cutoff_step=args.cutoff_step,
        target_active_fraction=args.target_active_fraction,
        max_tokens=args.max_tokens,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
