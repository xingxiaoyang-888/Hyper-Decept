"""Build the frozen coordination-only HyperTrace Base DatasetPlan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.episode_manifest import DatasetPlan, EpisodeManifest  # noqa: E402


MANIFEST_PATTERN = re.compile(r"^(?P<scenario>.+)_n(?P<n>\d+)_s(?P<seed>\d+)\.manifest\.json$")


def build_plan(
    *, synthetic_root: Path, uk_bundle: Path, crypto_bundle: Path,
    fake_bundle: Path, output: Path, expected_scenarios: int = 5,
    expected_seeds: int = 4,
) -> DatasetPlan:
    manifests = []
    for path in sorted(synthetic_root.glob("*_n*_s*.manifest.json")):
        match = MANIFEST_PATTERN.match(path.name)
        if match is None:
            continue
        manifests.append(EpisodeManifest.read(path))
    if len(manifests) != expected_scenarios * expected_seeds:
        raise ValueError(
            f"expected {expected_scenarios * expected_seeds} synthetic manifests, "
            f"found {len(manifests)}"
        )
    scenarios = {str(item.scenario_id) for item in manifests}
    seeds = {int(item.simulation_seed) for item in manifests}
    if len(scenarios) != expected_scenarios or len(seeds) != expected_seeds:
        raise ValueError(
            f"unexpected synthetic grid: scenarios={sorted(scenarios)}, "
            f"seeds={sorted(seeds)}"
        )
    for path in (uk_bundle, crypto_bundle, fake_bundle):
        if not path.is_dir():
            raise FileNotFoundError(path)
    strategy = {
        "base_model": "HyperTrace-Base",
        "primary_supervision": "DeepPersona/OASIS audited LLM-driven episodes",
        "topology_pretraining": "UK2019 real graph; no CIB labels claimed",
        "crypto_role": "held-out campaign-disjoint domain adapter only",
        "fake_accounts_role": "external CIB evaluation only",
        "excluded_from_base": ["twibot22", "mgtab", "crypto-campaign"],
        "synthetic_split": "leave-one-scenario-out; validation uses seed 44 among seen scenarios",
        "synthetic_root": str(synthetic_root.resolve()),
        "uk_bundle": str(uk_bundle.resolve()),
        "crypto_bundle": str(crypto_bundle.resolve()),
        "fake_bundle": str(fake_bundle.resolve()),
    }
    return DatasetPlan(
        plan_id="hypertrace_base_coordination_v1",
        episodes=tuple(manifests),
        strategy=strategy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", required=True, type=Path)
    parser.add_argument("--uk-bundle", required=True, type=Path)
    parser.add_argument("--crypto-bundle", required=True, type=Path)
    parser.add_argument("--fake-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-scenarios", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=4)
    args = parser.parse_args()
    plan = build_plan(**vars(args))
    output = plan.write(args.output)
    payload = {
        "status": "passed",
        "plan": str(output),
        "summary": plan.summary(),
        "strategy": dict(plan.strategy),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
