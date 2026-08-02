import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "MultiAgent4Collusion-master"
    / "generate_simulation_csv.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "simulation_population_generation_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profiles(count):
    return [{"Summary": f"persona {index}"} for index in range(count)]


def test_formal_scenarios_produce_distinct_role_contracts():
    module = _load_module()
    module.random.seed(11)
    pool = {"good": ["organic"], "bad": ["campaign"]}
    leader = module.build_agent_dataframe(
        _profiles(200), pool, scenario="leader_amplifier", mean_following=10
    )
    synchronized = module.build_agent_dataframe(
        _profiles(200), pool, scenario="synchronized_boosting", mean_following=10
    )

    assert set(leader["scenario_id"]) == {"leader_amplifier"}
    assert leader["user_type"].value_counts()["bad_member"] == 19
    assert synchronized["user_type"].value_counts()["bad_member"] == 19
    leader_bad = leader[leader["user_type"].str.contains("bad")]
    synchronized_bad = synchronized[synchronized["user_type"].str.contains("bad")]
    assert synchronized_bad["following_count"].mean() > leader_bad["following_count"].mean()


def test_bridge_scenario_includes_independent_adversary_roles():
    module = _load_module()
    module.random.seed(22)
    frame = module.build_agent_dataframe(
        _profiles(200),
        {"good": ["organic"], "bad": ["campaign"]},
        scenario="bridge_infiltration",
        mean_following=10,
    )
    counts = frame["user_type"].value_counts()
    assert counts["bad_leader"] == 1
    assert counts["bad_member"] == 11
    assert counts["bad"] == 8


def test_formal_sparse_graph_avoids_quadratic_edge_volume():
    module = _load_module()
    module.random.seed(33)
    frame = module.build_agent_dataframe(
        _profiles(500),
        {"good": ["organic"], "bad": ["campaign"]},
        scenario="adaptive_evasion",
        mean_following=20,
    )
    assert frame["following_count"].mean() < 30
    assert frame["following_count"].sum() < 15_000


def test_synchronized_scenario_has_shared_malicious_burst_hours():
    module = _load_module()
    adjusted = module._inject_scenario_activity(
        [0.1] * 24,
        "bad_member",
        "synchronized_boosting",
    )
    assert [adjusted[hour] for hour in (3, 9, 15, 21)] == [0.95] * 4
    assert max(adjusted[hour] for hour in (0, 1, 2, 4, 5)) <= 0.35
