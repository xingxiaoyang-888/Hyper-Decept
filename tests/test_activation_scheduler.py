import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "MultiAgent4Collusion-master/oasis/social_platform/activation_scheduler.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("activation_scheduler_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_budget_is_bounded_and_reproducible():
    module = _load_module()
    kwargs = dict(
        num_agents=100,
        target_fraction=0.12,
        max_silent_steps=8,
        seed=11,
        leader_slots=2,
    )
    first = module.BudgetedActivationScheduler(**kwargs)
    second = module.BudgetedActivationScheduler(**kwargs)
    probabilities = {agent_id: 0.2 + (agent_id % 5) * 0.1 for agent_id in range(100)}
    types = {agent_id: "good" for agent_id in range(100)}
    types[98] = types[99] = "bad_leader"
    decisions_a = first.select(
        timestep=1,
        activity_probabilities=probabilities,
        user_types=types,
    )
    decisions_b = second.select(
        timestep=1,
        activity_probabilities=probabilities,
        user_types=types,
    )
    assert first.step_budget == 12
    assert len(decisions_a) == 12
    assert decisions_a == decisions_b
    assert {98, 99}.issubset({decision.agent_id for decision in decisions_a})


def test_pending_tasks_wake_members_without_exceeding_budget():
    module = _load_module()
    scheduler = module.BudgetedActivationScheduler(
        num_agents=20,
        target_fraction=0.2,
        seed=22,
    )
    decisions = scheduler.select(
        timestep=1,
        activity_probabilities={agent_id: 0.5 for agent_id in range(20)},
        user_types={agent_id: "good" for agent_id in range(20)},
        pending_task_member_ids=[17, 18, 19],
    )
    selected = {decision.agent_id: decision.reason for decision in decisions}
    assert len(decisions) == 4
    assert selected[17] == selected[18] == selected[19] == "pending_task"


def test_two_day_protocol_request_ceiling():
    module = _load_module()
    scheduler = module.BudgetedActivationScheduler(
        num_agents=2000,
        target_fraction=0.075,
    )
    assert scheduler.step_budget == 150
    assert scheduler.estimate_requests(time_steps=30, episodes=20) == 90_000


def test_observed_window_coverage_is_completed_by_cutoff():
    module = _load_module()
    scheduler = module.BudgetedActivationScheduler(
        num_agents=200,
        target_fraction=0.08,
        seed=33,
        coverage_deadlines=(18, 30),
    )
    probabilities = {agent_id: 0.5 for agent_id in range(200)}
    types = {agent_id: "good" for agent_id in range(200)}
    observed = set()
    for timestep in range(1, 19):
        observed.update(decision.agent_id for decision in scheduler.select(
            timestep=timestep,
            activity_probabilities=probabilities,
            user_types=types,
        ))
    assert observed == set(range(200))


def test_repeated_task_wakeups_cannot_break_cutoff_coverage():
    module = _load_module()
    scheduler = module.BudgetedActivationScheduler(
        num_agents=200,
        target_fraction=0.08,
        seed=44,
        leader_slots=1,
        coverage_deadlines=(18,),
    )
    probabilities = {agent_id: 0.5 for agent_id in range(200)}
    types = {agent_id: "good" for agent_id in range(200)}
    types[199] = "bad_leader"
    observed = set()
    for timestep in range(1, 19):
        observed.update(decision.agent_id for decision in scheduler.select(
            timestep=timestep,
            activity_probabilities=probabilities,
            user_types=types,
            pending_task_member_ids=range(180, 199),
        ))
    assert observed == set(range(200))
