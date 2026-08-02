from scripts.estimate_simulation_budget import estimate_budget


def test_a10_four_slot_budget_has_room_for_formal_protocol():
    result = estimate_budget(
        mean_request_seconds=4.0,
        concurrent_slots=4,
    )
    assert result["feasible"]
    assert result["recommended_active_fraction"] >= 0.075
    assert result["protocol_activation_ceiling"] == 90_000
    assert result["protocol_request_ceiling"] == 90_000
    assert result["protocol_projected_compute_hours"] <= result["usable_hours"]


def test_budget_rejects_an_infeasible_server():
    result = estimate_budget(
        mean_request_seconds=6.0,
        concurrent_slots=1,
    )
    assert not result["feasible"]
    assert "increase stable concurrency" in result["action"]


def test_retry_amplification_is_included_in_deadline_estimate():
    result = estimate_budget(
        mean_request_seconds=4.0,
        concurrent_slots=4,
        mean_requests_per_activation=1.5,
    )
    assert result["protocol_activation_ceiling"] == 90_000
    assert result["protocol_request_ceiling"] == 135_000
    assert not result["feasible"]
