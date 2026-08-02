import json
from pathlib import Path

from scripts.run_formal_simulation_plan import run_plan


def _plan(tmp_path, configs):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema_version": "hyperdecept.formal-simulation-plan.v1",
        "episodes": len(configs),
        "configs": [str(config.resolve()) for config in configs],
    }), encoding="utf-8")
    return path


def test_runner_resumes_completed_episodes(tmp_path):
    configs = []
    for index in range(2):
        config = tmp_path / f"episode_{index}.yaml"
        config.write_text("simulation: {}\n", encoding="utf-8")
        configs.append(config)
    simulation = tmp_path / "simulation.py"
    simulation.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--config_path')\n"
        "a=p.parse_args()\n"
        "print(a.config_path)\n",
        encoding="utf-8",
    )
    plan = _plan(tmp_path, configs)
    state_path = tmp_path / "state.json"
    first = run_plan(
        plan_path=plan,
        simulation_script=simulation,
        state_path=state_path,
        log_dir=tmp_path / "logs",
        max_episodes=1,
    )
    assert first["summary"]["completed"] == 1
    assert first["summary"]["remaining"] == 1

    second = run_plan(
        plan_path=plan,
        simulation_script=simulation,
        state_path=state_path,
        log_dir=tmp_path / "logs",
    )
    assert second["summary"]["completed"] == 2
    assert second["summary"]["remaining"] == 0


def test_runner_dry_run_never_executes_simulation(tmp_path):
    config = tmp_path / "episode.yaml"
    config.write_text("simulation: {}\n", encoding="utf-8")
    simulation = tmp_path / "simulation.py"
    simulation.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    state = run_plan(
        plan_path=_plan(tmp_path, [config]),
        simulation_script=simulation,
        state_path=tmp_path / "state.json",
        log_dir=tmp_path / "logs",
        dry_run=True,
    )
    assert state["summary"]["dry_run"] == 1
    assert state["summary"]["completed"] == 0
