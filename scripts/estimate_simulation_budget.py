"""Calibrate the formal activation budget from a server microbenchmark."""

from __future__ import annotations

import argparse
import json
import math


def estimate_budget(
    *,
    mean_request_seconds: float,
    concurrent_slots: int,
    deadline_hours: float = 48.0,
    utilization: float = 0.80,
    reserve_fraction: float = 0.15,
    num_agents: int = 2000,
    time_steps: int = 30,
    episodes: int = 20,
    min_fraction: float = 0.075,
    max_fraction: float = 0.15,
    protocol_fraction: float = 0.075,
    mean_requests_per_activation: float = 1.0,
) -> dict:
    if mean_request_seconds <= 0 or concurrent_slots <= 0:
        raise ValueError("latency and concurrent slots must be positive")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    if not 1.0 <= mean_requests_per_activation <= 2.0:
        raise ValueError("mean_requests_per_activation must be in [1, 2]")
    total_opportunities = num_agents * time_steps * episodes
    usable_seconds = deadline_hours * 3600 * (1 - reserve_fraction)
    requests_per_second = concurrent_slots * utilization / mean_request_seconds
    request_capacity = math.floor(usable_seconds * requests_per_second)
    raw_fraction = request_capacity / (
        total_opportunities * mean_requests_per_activation
    )
    recommended = min(max_fraction, max(min_fraction, raw_fraction))
    step_budget = math.floor(num_agents * recommended)
    activation_ceiling = step_budget * time_steps * episodes
    request_ceiling = math.ceil(
        activation_ceiling * mean_requests_per_activation
    )
    projected_hours = (
        request_ceiling * mean_request_seconds / (concurrent_slots * utilization) / 3600
    )
    protocol_step_budget = math.ceil(num_agents * protocol_fraction)
    protocol_activation_ceiling = protocol_step_budget * time_steps * episodes
    protocol_request_ceiling = math.ceil(
        protocol_activation_ceiling * mean_requests_per_activation
    )
    protocol_projected_hours = (
        protocol_request_ceiling
        * mean_request_seconds
        / (concurrent_slots * utilization)
        / 3600
    )
    feasible = (
        raw_fraction >= protocol_fraction
        and protocol_projected_hours <= usable_seconds / 3600
    )
    return {
        "schema_version": "hyperdecept.simulation-budget.v1",
        "deadline_hours": deadline_hours,
        "reserve_fraction": reserve_fraction,
        "usable_hours": usable_seconds / 3600,
        "mean_request_seconds": mean_request_seconds,
        "concurrent_slots": concurrent_slots,
        "utilization": utilization,
        "mean_requests_per_activation": mean_requests_per_activation,
        "request_capacity": request_capacity,
        "raw_affordable_fraction": raw_fraction,
        "recommended_active_fraction": recommended,
        "step_request_budget": step_budget,
        "corpus_activation_ceiling": activation_ceiling,
        "corpus_request_ceiling": request_ceiling,
        "projected_compute_hours": projected_hours,
        "protocol_active_fraction": protocol_fraction,
        "protocol_activation_ceiling": protocol_activation_ceiling,
        "protocol_request_ceiling": protocol_request_ceiling,
        "protocol_projected_compute_hours": protocol_projected_hours,
        "feasible": feasible,
        "action": (
            "generate formal configs with the recommended fraction"
            if feasible else
            "increase stable concurrency or reduce mean request latency before formal generation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-request-seconds", required=True, type=float)
    parser.add_argument("--concurrent-slots", required=True, type=int)
    parser.add_argument("--deadline-hours", type=float, default=48.0)
    parser.add_argument("--utilization", type=float, default=0.80)
    parser.add_argument("--reserve-fraction", type=float, default=0.15)
    parser.add_argument(
        "--mean-requests-per-activation",
        type=float,
        default=1.0,
        help="Measured inference requests per activated agent, including JSON retries.",
    )
    args = parser.parse_args()
    result = estimate_budget(
        mean_request_seconds=args.mean_request_seconds,
        concurrent_slots=args.concurrent_slots,
        deadline_hours=args.deadline_hours,
        utilization=args.utilization,
        reserve_fraction=args.reserve_fraction,
        mean_requests_per_activation=args.mean_requests_per_activation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["feasible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
