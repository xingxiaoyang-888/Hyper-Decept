"""Prepare and audit the complete 20-episode formal simulation corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from deeppersona_ai.prepare_formal_personas import _source_path, prepare_seed
from scripts.audit_formal_simulation_inputs import audit_plan
from scripts.generate_formal_simulation_plan import (
    SCENARIOS,
    SIMULATION_SEEDS,
    build_configs,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_population_adapter():
    path = ROOT / "MultiAgent4Collusion-master/generate_simulation_csv.py"
    spec = importlib.util.spec_from_file_location("formal_population_adapter", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(path)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_corpus(
    *,
    output_root: Path,
    tweet_pool_path: Path,
    source: Path | None = None,
    num_agents: int = 2000,
    min_attributes: int = 200,
    parallel: int = 64,
) -> dict:
    output_root = output_root.expanduser().resolve()
    tweet_pool_path = tweet_pool_path.expanduser().resolve()
    source_path = _source_path(source)
    personas = output_root / "personas"
    csv_root = output_root / "csv"
    db_root = output_root / "db"
    config_root = output_root / "configs"
    for path in (personas, csv_root, db_root, config_root):
        path.mkdir(parents=True, exist_ok=True)

    persona_reports = {
        seed: prepare_seed(
            source=source_path,
            output_dir=personas,
            count=num_agents,
            seed=seed,
            min_attributes=min_attributes,
        )
        for seed in SIMULATION_SEEDS
    }
    adapter = _load_population_adapter()
    tweet_pool = adapter.load_tweet_pool(tweet_pool_path)
    generated_csv = []
    for scenario in SCENARIOS:
        scenario_dir = csv_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for seed in SIMULATION_SEEDS:
            profiles = json.loads(
                Path(persona_reports[seed]["profiles_path"]).read_text(
                    encoding="utf-8"
                )
            )
            adapter.random.seed(seed)
            frame = adapter.build_agent_dataframe(
                profiles,
                tweet_pool,
                scenario=scenario,
                mean_following=30,
            )
            path = scenario_dir / f"seed_{seed}.csv"
            frame.to_csv(path, index=False, encoding="utf-8")
            generated_csv.append(str(path.resolve()))

    plan = build_configs(
        output_dir=config_root,
        csv_root=csv_root,
        db_root=db_root,
        persona_root=personas,
        tweet_pool_path=tweet_pool_path,
        endpoints=[{
            "host": "api.deepseek.com",
            "ports": [443],
            "base_url": "https://api.deepseek.com/v1",
            "parallel": parallel,
            "max_tokens": 512,
        }],
        num_agents=num_agents,
    )
    plan_path = config_root / "formal_simulation_plan.json"
    audit = audit_plan(plan_path, require_api_key=False)
    report = {
        "schema_version": "hyperdecept.formal-corpus-preparation.v1",
        "status": "passed",
        "output_root": str(output_root),
        "source": str(source_path),
        "tweet_pool": str(tweet_pool_path),
        "num_agents": num_agents,
        "persona_populations": len(persona_reports),
        "prototype_count": next(iter(persona_reports.values()))["prototype_count"],
        "csv_files": len(generated_csv),
        "episodes": plan["episodes"],
        "parallel": parallel,
        "plan": str(plan_path.resolve()),
        "audit": audit,
    }
    report_path = output_root / "formal_corpus_preparation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tweet-pool", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--num-agents", type=int, default=2000)
    parser.add_argument("--min-attributes", type=int, default=200)
    parser.add_argument("--parallel", type=int, default=64)
    args = parser.parse_args()
    report = prepare_corpus(
        output_root=args.output_root,
        tweet_pool_path=args.tweet_pool,
        source=args.source,
        num_agents=args.num_agents,
        min_attributes=args.min_attributes,
        parallel=args.parallel,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
