"""Budgeted, reproducible activation for large OASIS simulations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ActivationDecision:
    agent_id: int
    reason: str
    score: float


class BudgetedActivationScheduler:
    """Select a bounded active set without discarding activity or task signals."""

    def __init__(
        self,
        *,
        num_agents: int,
        target_fraction: float = 0.12,
        min_active_agents: int = 1,
        max_silent_steps: int = 8,
        seed: int = 42,
        wake_on_pending_task: bool = True,
        leader_slots: int = 0,
        coverage_deadlines: Sequence[int] = (),
    ) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if not 0.0 < target_fraction <= 1.0:
            raise ValueError("target_fraction must be in (0, 1]")
        if min_active_agents <= 0:
            raise ValueError("min_active_agents must be positive")
        if max_silent_steps <= 0:
            raise ValueError("max_silent_steps must be positive")
        self.num_agents = int(num_agents)
        self.target_fraction = float(target_fraction)
        self.min_active_agents = min(int(min_active_agents), self.num_agents)
        self.max_silent_steps = int(max_silent_steps)
        self.seed = int(seed)
        self.wake_on_pending_task = bool(wake_on_pending_task)
        if leader_slots < 0:
            raise ValueError("leader_slots must be non-negative")
        self.leader_slots = int(leader_slots)
        cleaned_deadlines = tuple(sorted({int(value) for value in coverage_deadlines}))
        if any(value <= 0 for value in cleaned_deadlines):
            raise ValueError("coverage deadlines must be positive")
        self.coverage_deadlines = cleaned_deadlines
        self.last_active_step = {agent_id: 0 for agent_id in range(self.num_agents)}
        self.seen_in_window: set[int] = set()
        self._window_index = 0

    @property
    def step_budget(self) -> int:
        return min(
            self.num_agents,
            max(self.min_active_agents, math.ceil(self.num_agents * self.target_fraction)),
        )

    def _priority(self, timestep: int, agent_id: int, activity_probability: float) -> float:
        payload = f"{self.seed}:{timestep}:{agent_id}".encode("ascii")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        uniform = (value + 1) / (2**64 + 1)
        weight = max(1e-6, min(1.0, float(activity_probability)))
        # Exponential-race sampling gives higher-activity users more chances
        # while remaining deterministic for an episode seed.
        return -math.log(uniform) / weight

    def select(
        self,
        *,
        timestep: int,
        activity_probabilities: Mapping[int, float],
        user_types: Mapping[int, str],
        force_agent_ids: Iterable[int] = (),
        pending_task_member_ids: Iterable[int] = (),
    ) -> tuple[ActivationDecision, ...]:
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        candidates = set(activity_probabilities)
        if candidates != set(range(self.num_agents)):
            raise ValueError("activity probabilities must cover every agent exactly once")

        reasons: dict[int, str] = {}
        forced = {int(value) for value in force_agent_ids}
        for agent_id in forced:
            if agent_id in candidates:
                reasons[agent_id] = "explicit_force"
        if len(reasons) > self.step_budget:
            raise ValueError("explicit force set exceeds the activation budget")

        while (
            self._window_index < len(self.coverage_deadlines)
            and timestep > self.coverage_deadlines[self._window_index]
        ):
            self._window_index += 1
            self.seen_in_window.clear()
        if self._window_index < len(self.coverage_deadlines):
            deadline = self.coverage_deadlines[self._window_index]
            unseen = candidates.difference(self.seen_in_window).difference(reasons)
            remaining_steps = deadline - timestep + 1
            required = max(
                0,
                len(unseen) - (remaining_steps - 1) * self.step_budget,
            )
            if required > self.step_budget - len(reasons):
                raise RuntimeError(
                    "activation budget cannot satisfy the configured coverage deadline"
                )
            coverage_ranked = sorted(
                (
                    self._priority(
                        timestep,
                        agent_id,
                        activity_probabilities[agent_id],
                    ),
                    agent_id,
                )
                for agent_id in unseen
            )
            for _, agent_id in coverage_ranked[:required]:
                reasons[agent_id] = "window_coverage"

        leaders = [
            agent_id for agent_id, user_type in user_types.items()
            if user_type == "bad_leader" and agent_id not in reasons
        ]
        leaders.sort(key=lambda agent_id: (
            self.last_active_step[agent_id],
            self._priority(timestep, agent_id, activity_probabilities[agent_id]),
        ))
        for agent_id in leaders[: min(self.leader_slots, self.step_budget - len(reasons))]:
            reasons[agent_id] = "leader_rotation"

        if self.wake_on_pending_task:
            for agent_id in pending_task_member_ids:
                if len(reasons) >= self.step_budget:
                    break
                if int(agent_id) in candidates:
                    reasons.setdefault(int(agent_id), "pending_task")

        silent = sorted(
            (
                timestep - self.last_active_step[agent_id],
                agent_id,
            )
            for agent_id in candidates.difference(reasons)
            if timestep - self.last_active_step[agent_id] >= self.max_silent_steps
        )
        # Fairness wake-ups remain budgeted; the stalest users are served first.
        for _, agent_id in reversed(silent):
            if len(reasons) >= self.step_budget:
                break
            reasons[agent_id] = "max_silence"

        ranked = sorted(
            (
                self._priority(timestep, agent_id, activity_probabilities[agent_id]),
                agent_id,
            )
            for agent_id in candidates.difference(reasons)
        )
        for score, agent_id in ranked:
            if len(reasons) >= self.step_budget:
                break
            reasons[agent_id] = "activity_budget"

        decisions = tuple(
            ActivationDecision(
                agent_id=agent_id,
                reason=reason,
                score=self._priority(
                    timestep,
                    agent_id,
                    activity_probabilities[agent_id],
                ),
            )
            for agent_id, reason in sorted(reasons.items())
        )
        for decision in decisions:
            self.last_active_step[decision.agent_id] = timestep
            self.seen_in_window.add(decision.agent_id)
        return decisions

    def estimate_requests(self, time_steps: int, episodes: int = 1) -> int:
        if time_steps <= 0 or episodes <= 0:
            raise ValueError("time_steps and episodes must be positive")
        return self.step_budget * int(time_steps) * int(episodes)


def pending_bad_member_ids(
    agents: Sequence[tuple[int, object]],
    *,
    has_pending_tasks: bool,
    limit: int,
    seed: int,
    timestep: int,
) -> tuple[int, ...]:
    """Choose a bounded rotating set of bad members when work is pending."""
    if not has_pending_tasks or limit <= 0:
        return ()
    members = [
        int(agent_id)
        for agent_id, agent in agents
        if getattr(getattr(agent, "user_info", None), "user_type", None) == "bad_member"
    ]
    random.Random(f"{seed}:{timestep}:pending-task").shuffle(members)
    return tuple(members[:limit])
