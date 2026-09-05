"""Generate independent, resumable personas through the DeepPersona engine."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "Deeppersona" / "generate_user_profile"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaf_count(value) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_leaf_count(item) for item in value)
    return int(value is not None and str(value).strip() != "")


def validate_profile(profile: dict, *, minimum_leaves: int) -> dict:
    summary = profile.get("Summary")
    leaves = _leaf_count({
        key: value for key, value in profile.items()
        if key not in {"Generated At", "Profile Index", "Summary"}
    })
    errors = []
    if not isinstance(summary, str) or len(summary.split()) < 80:
        errors.append("summary_shorter_than_80_words")
    if leaves < minimum_leaves:
        errors.append(f"nonempty_leaves_below_{minimum_leaves}")
    return {
        "valid": not errors,
        "nonempty_leaves": leaves,
        "summary_words": len(summary.split()) if isinstance(summary, str) else 0,
        "errors": errors,
    }


def _set_nested(target: dict, dotted_path: str, value) -> None:
    parts = dotted_path.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _flatten_response(value, prefix: str = "") -> dict:
    """Normalize either dotted-key or nested JSON without inventing values."""
    flattened = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                flattened.update(_flatten_response(item, path))
            else:
                flattened[path] = item
    return flattened


def _expected_values(payload: dict, expected_paths: list[str]) -> dict:
    flattened = _flatten_response(payload)
    values = {}
    for expected in expected_paths:
        if expected in payload:
            values[expected] = payload[expected]
            continue
        if expected in flattened:
            values[expected] = flattened[expected]
            continue
        suffix_matches = [
            value for key, value in flattened.items()
            if key.endswith(f".{expected}")
        ]
        if len(suffix_matches) == 1:
            values[expected] = suffix_matches[0]
    return values


def _generate_dp_profile(
    *,
    agent_id: int,
    attribute_count: int,
    attribute_batch_size: int,
) -> tuple[dict, int]:
    import config
    from generate_profile import generate_final_summary
    from select_attributes import generate_user_profile, get_selected_attributes

    config.API_CALL_COUNT = 0
    anchor_error = None
    for _ in range(3):
        try:
            base_info = generate_user_profile()
            break
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            anchor_error = exc
    else:
        raise ValueError(f"DeepPersona anchor generation failed: {anchor_error}")
    selected_paths = get_selected_attributes(base_info, attribute_count)
    if len(selected_paths) != attribute_count:
        raise ValueError(
            f"selected {len(selected_paths)} attributes, expected {attribute_count}"
        )
    profile = {
        "Base Info": base_info,
        "Generated At": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Profile Index": agent_id + 1,
    }
    for start in range(0, len(selected_paths), attribute_batch_size):
        batch = selected_paths[start:start + attribute_batch_size]
        aliases = {
            f"A{offset:03d}": path for offset, path in enumerate(batch)
        }
        system_prompt = (
            "You are the attribute filling stage of DeepPersona. Return one "
            "valid JSON object whose keys exactly match every supplied short "
            "attribute ID. Values must be concise (at most 12 words), specific, "
            "mutually consistent, and grounded in the provided persona anchor. "
            "Do not omit IDs or add IDs."
        )
        user_prompt = (
            "Persona anchor:\n"
            + json.dumps(base_info, ensure_ascii=False)
            + "\n\nAttributes generated in earlier batches:\n"
            + json.dumps(
                {key: value for key, value in profile.items() if key != "Base Info"},
                ensure_ascii=False,
            )
            + "\n\nFill these attribute IDs (return the ID, not the path):\n"
            + "\n".join(
                f"- {alias}: {path}" for alias, path in aliases.items()
            )
        )
        values = {}
        missing_aliases = list(aliases)
        for batch_attempt in range(3):
            retry_prompt = user_prompt
            if batch_attempt:
                retry_prompt += (
                    "\n\nYour previous response omitted or malformed these IDs. "
                    "Return valid JSON containing every listed ID exactly once:\n"
                    + "\n".join(
                        f"- {alias}: {aliases[alias]}"
                        for alias in missing_aliases
                    )
                )
            response = config.get_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.4,
            )
            parsed = config.parse_json_response(response, {})
            if isinstance(parsed, dict):
                flattened = _flatten_response(parsed)
                for alias, path in aliases.items():
                    if alias in parsed:
                        values[path] = parsed[alias]
                    elif alias in flattened:
                        values[path] = flattened[alias]
                # Accept exact original paths if a provider ignores aliases.
                values.update(_expected_values(parsed, batch))
            missing_aliases = [
                alias for alias, path in aliases.items() if path not in values
            ]
            if not missing_aliases:
                break
        if missing_aliases:
            raise ValueError(
                "attribute batch omitted "
                f"{len(missing_aliases)} IDs: {missing_aliases[:3]}"
            )
        for path in batch:
            _set_nested(profile, path, values[path])

    profile_for_summary = {
        key: value for key, value in profile.items()
        if key not in {"Base Info", "Generated At", "Profile Index"}
    }
    profile["Summary"] = generate_final_summary(profile_for_summary, base_info)
    return profile, config.API_CALL_COUNT


def _generate_one(payload: dict) -> dict:
    agent_id = int(payload["agent_id"])
    seed = int(payload["seed"])
    output_root = Path(payload["output_root"])
    work_dir = output_root / "work" / f"agent_{agent_id:05d}"
    profile_path = output_root / "profiles" / f"agent_{agent_id:05d}.json"
    audit_path = output_root / "audits" / f"agent_{agent_id:05d}.json"
    log_path = output_root / "logs" / f"agent_{agent_id:05d}.log"
    for path in (work_dir, profile_path.parent, audit_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))

    attempts = int(payload["attempts"])
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    profile, api_calls = _generate_dp_profile(
                        agent_id=agent_id,
                        attribute_count=int(payload["attribute_count"]),
                        attribute_batch_size=int(payload["attribute_batch_size"]),
                    )
            profile["Profile Index"] = agent_id + 1
            quality = validate_profile(
                profile,
                minimum_leaves=int(payload["minimum_leaves"]),
            )
            if not quality["valid"]:
                raise ValueError(",".join(quality["errors"]))
            temporary = profile_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(profile_path)
            result = {
                "agent_id": agent_id,
                "status": "completed",
                "attempt": attempt,
                "seed": seed,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": _sha256(profile_path),
                "api_calls": api_calls,
                **quality,
            }
            audit_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(traceback.format_exc() + "\n")
    result = {
        "agent_id": agent_id,
        "status": "failed",
        "seed": seed,
        "attempt": attempts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "error": last_error,
        "log_path": str(log_path.resolve()),
    }
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def generate_population(
    *,
    output_root: Path,
    count: int,
    workers: int,
    seed: int,
    attribute_count: int = 200,
    attribute_batch_size: int = 40,
    minimum_leaves: int = 150,
    attempts: int = 2,
    max_profiles: int | None = None,
) -> dict:
    if not os.getenv("DEEPSEEK_API_KEY", "").strip() and not os.getenv(
        "OPENAI_API_KEY", ""
    ).strip():
        raise EnvironmentError("DEEPSEEK_API_KEY or OPENAI_API_KEY must be set")
    if count <= 0 or workers <= 0:
        raise ValueError("count and workers must be positive")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed = set()
    for path in (output_root / "profiles").glob("agent_*.json"):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            if validate_profile(profile, minimum_leaves=minimum_leaves)["valid"]:
                completed.add(int(path.stem.split("_")[-1]))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    pending = [value for value in range(count) if value not in completed]
    if max_profiles is not None:
        pending = pending[:max_profiles]
    payloads = [{
        "agent_id": agent_id,
        "seed": seed + agent_id * 104729,
        "output_root": str(output_root),
        "attribute_count": attribute_count,
        "attribute_batch_size": attribute_batch_size,
        "minimum_leaves": minimum_leaves,
        "attempts": attempts,
    } for agent_id in pending]

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads) or 1)) as pool:
        futures = [pool.submit(_generate_one, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    new_completed = [item for item in results if item["status"] == "completed"]
    failed = [item for item in results if item["status"] == "failed"]
    elapsed = time.perf_counter() - started
    all_profiles = sorted((output_root / "profiles").glob("agent_*.json"))
    consolidated = [json.loads(path.read_text(encoding="utf-8")) for path in all_profiles]
    consolidated_path = output_root / "formal_dp_personas.json"
    consolidated_path.write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "hyperdecept.formal-dp-personas.v1",
        "generator": "DeepPersona anchor-selection-fill-summary engine",
        "design": "independently_generated_fixed_population_reused_across_episode_seeds",
        "requested_count": count,
        "total_valid_profiles": len(consolidated),
        "generated_this_run": len(new_completed),
        "failed_this_run": len(failed),
        "workers": workers,
        "master_seed": seed,
        "attribute_count": attribute_count,
        "minimum_nonempty_leaves": minimum_leaves,
        "elapsed_seconds": round(elapsed, 3),
        "mean_seconds_per_completed_profile": (
            round(sum(item["elapsed_seconds"] for item in new_completed) / len(new_completed), 3)
            if new_completed else None
        ),
        "total_api_calls_this_run": sum(
            item.get("api_calls", 0) for item in new_completed
        ),
        "mean_api_calls_per_completed_profile": (
            round(sum(item.get("api_calls", 0) for item in new_completed) / len(new_completed), 3)
            if new_completed else None
        ),
        "wall_profiles_per_hour": (
            round(len(new_completed) * 3600 / elapsed, 3) if elapsed else None
        ),
        "consolidated_path": str(consolidated_path.resolve()),
        "consolidated_sha256": _sha256(consolidated_path),
        "failures": failed,
    }
    (output_root / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--attribute-count", type=int, default=200)
    parser.add_argument("--attribute-batch-size", type=int, default=40)
    parser.add_argument("--minimum-leaves", type=int, default=150)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-profiles", type=int)
    args = parser.parse_args()
    report = generate_population(
        output_root=args.output_root,
        count=args.count,
        workers=args.workers,
        seed=args.seed,
        attribute_count=args.attribute_count,
        attribute_batch_size=args.attribute_batch_size,
        minimum_leaves=args.minimum_leaves,
        attempts=args.attempts,
        max_profiles=args.max_profiles,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed_this_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
