from pathlib import Path

import pandas as pd

from scripts.audit_chi_model_ablation import _paired_bootstrap, _summary
from scripts.audit_consensus_evidence_preflight import _source_ids, _walk


def test_explanation_audit_rejects_excluded_modules() -> None:
    errors = []
    _walk({"geometry": {"geodesic_margin": 0.2}, "role_label": "leader"}, "packet", errors)
    assert errors == ["excluded psychology/role key at packet.role_label"]


def test_explanation_audit_resolves_only_requested_evidence(tmp_path) -> None:
    source = tmp_path / "events.csv"
    pd.DataFrame({"evidence_id": ["e1", "e2", "e3"], "value": [1, 2, 3]}).to_csv(source, index=False)
    assert _source_ids(source, {"e2", "missing"}) == {"e2"}


def test_ablation_summary_and_bootstrap_are_deterministic() -> None:
    values = [float(index) / 100 for index in range(1, 16)]
    summary = _summary(values)
    first = _paired_bootstrap(values)
    second = _paired_bootstrap(values)
    assert summary["n"] == 15
    assert first == second
    assert first["ci95_low"] < first["mean_delta"] < first["ci95_high"]
    assert first["probability_delta_above_zero"] == 1.0


def test_final_window_runner_records_excluded_modules() -> None:
    runner = Path("scripts/run_chi_final_window.sh").read_text(encoding="utf-8")
    for excluded in ("full26", "psychology_proxy8", "role_head", "campaign_head"):
        assert excluded not in runner


def test_final_window_runner_supports_single_gpu_mode() -> None:
    runner = Path("scripts/run_chi_final_window.sh").read_text(encoding="utf-8")
    assert 'GPU_LIST="${GPU_LIST:-0}"' in runner
    assert "if ((${#GPUS[@]} >= 2)); then" in runner
    assert 'WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"' in runner
    assert 'INTRAOP_THREADS="${INTRAOP_THREADS:-4}"' in runner
    assert 'INTEROP_THREADS="${INTEROP_THREADS:-1}"' in runner
    assert "lorentz_scratch_observable18" in runner
    assert "total_workers=$((" in runner
