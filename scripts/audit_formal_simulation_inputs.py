"""Audit a complete formal simulation plan before paid server execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_agent_ids(values, count: int, label: str) -> None:
    normalized = {int(value) for value in values}
    expected = set(range(count))
    if normalized != expected:
        raise ValueError(
            f"{label} agent IDs must cover 0..{count - 1}; "
            f"missing={len(expected - normalized)} extra={len(normalized - expected)}"
        )


def audit_plan(plan_path: Path, *, require_api_key: bool = True) -> dict:
    plan_path = plan_path.expanduser().resolve()
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "hyperdecept.formal-simulation-plan.v1":
        raise ValueError("unsupported formal simulation plan schema")
    num_agents = int(plan["num_agents"])
    configs = [Path(value).expanduser().resolve() for value in plan["configs"]]
    if len(configs) != int(plan["episodes"]):
        raise ValueError("episode count does not match config count")
    if require_api_key and not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise EnvironmentError("DEEPSEEK_API_KEY is not set")

    persona_cache: dict[Path, dict] = {}
    csv_rows = 0
    endpoint_slots = set()
    for config_path in configs:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        simulation = config["simulation"]
        model = config["model"]
        inference = config["inference"]
        if int(model["num_agents"]) != num_agents:
            raise ValueError(f"model population mismatch: {config_path}")
        if sum(int(row["num"]) for row in model["cfgs"]) != num_agents:
            raise ValueError(f"model allocation mismatch: {config_path}")
        if simulation.get("force_all_agents_active") is not False:
            raise ValueError(f"formal config must use budgeted activation: {config_path}")

        chunks_path = Path(
            simulation["deep_persona_chunks_path"]
        ).expanduser().resolve()
        if chunks_path not in persona_cache:
            if not chunks_path.is_file():
                raise FileNotFoundError(chunks_path)
            chunks = _load_json(chunks_path)
            _assert_agent_ids(
                (row["agent_id"] for row in chunks),
                num_agents,
                str(chunks_path),
            )
            summary_ids = {
                int(row["agent_id"])
                for row in chunks
                if row.get("section") == "summary" and str(row.get("text", "")).strip()
            }
            _assert_agent_ids(summary_ids, num_agents, f"{chunks_path} summaries")
            manifest_path = chunks_path.with_name(
                chunks_path.name.replace(".chunks.json", ".personas.manifest.json")
            )
            manifest = _load_json(manifest_path)
            if manifest.get("schema_version") != "hyperdecept.persona-population.v1":
                raise ValueError(f"unsupported persona manifest: {manifest_path}")
            if int(manifest["agents"]) != num_agents:
                raise ValueError(f"persona population mismatch: {manifest_path}")
            if len(manifest["assignments"]) != num_agents:
                raise ValueError(f"persona assignments mismatch: {manifest_path}")
            persona_cache[chunks_path] = {
                "chunks": len(chunks),
                "prototypes": int(manifest["prototype_count"]),
                "min_attributes": int(manifest["min_attributes"]),
            }

        csv_path = Path(config["data"]["csv_path"]).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(
            csv_path,
            usecols=["user_id", "user_char", "scenario_id", "previous_tweets"],
        )
        if len(frame) != num_agents:
            raise ValueError(f"CSV population mismatch: {csv_path}")
        _assert_agent_ids(frame["user_id"], num_agents, str(csv_path))
        if frame["user_char"].fillna("").str.strip().eq("").any():
            raise ValueError(f"CSV contains empty personas: {csv_path}")
        if frame["previous_tweets"].fillna("").isin(("", "[]")).any():
            raise ValueError(f"CSV contains agents without initial posts: {csv_path}")
        if set(frame["scenario_id"]) != {simulation["scenario_id"]}:
            raise ValueError(f"CSV scenario mismatch: {csv_path}")
        csv_rows += len(frame)

        for endpoint in inference["server_url"]:
            if endpoint.get("base_url") != "https://api.deepseek.com/v1":
                raise ValueError(f"unexpected inference endpoint: {config_path}")
            slots = int(endpoint.get("parallel", inference["parallel_per_endpoint"]))
            if slots <= 0:
                raise ValueError(f"invalid endpoint parallelism: {config_path}")
            endpoint_slots.add(slots)
        if int(inference["max_tokens"]) < 512:
            raise ValueError(f"max_tokens risks truncated JSON: {config_path}")

        serialized = json.dumps(config, ensure_ascii=False)
        if "sk-" in serialized or "DEEPSEEK_API_KEY" in serialized:
            raise ValueError(f"secret-like content embedded in config: {config_path}")

    return {
        "schema_version": "hyperdecept.formal-input-audit.v1",
        "status": "passed",
        "plan": str(plan_path),
        "episodes": len(configs),
        "num_agents": num_agents,
        "csv_rows_checked": csv_rows,
        "persona_populations": len(persona_cache),
        "persona_details": {
            str(path): details for path, details in persona_cache.items()
        },
        "endpoint_parallel_slots": sorted(endpoint_slots),
        "api_key_present": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--allow-missing-api-key", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_plan(
        args.plan,
        require_api_key=not args.allow_missing_api_key,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
