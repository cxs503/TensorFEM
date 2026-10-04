import json

import pytest

from tensorfem.marine_panel_execution import (
    dimensional_arc_controls, execute_panel_job, execute_panel_matrix,
    execute_panel_performance_gate,
    panel_geometry_preflight, panel_initial_equilibrium_preflight,
)
from tensorfem.marine_panel_ultimate_fe import build_panel_case


def test_dimensional_controls_reach_engineering_load_scale():
    case = build_panel_case(4)
    controls = dimensional_arc_controls(case, .02)
    assert controls["characteristic_displacement_m"] == case.model.thickness
    assert controls["characteristic_force_n"] == pytest.approx(2.5e6)
    assert controls["nominal_elastic_load_increment_n"] == pytest.approx(50_000)
    assert controls["solver_step_size"] == pytest.approx(2e-4)
    assert controls["solver_load_scale_m_per_n"] == pytest.approx(4e-9)


def test_warped_imperfect_facets_use_guarded_projection_without_flattening():
    case = build_panel_case(4)
    preflight = panel_geometry_preflight(case)
    assert preflight["passed"] is True
    assert preflight["maximum_relative_warpage"] > 1e-10


def test_residual_stress_field_is_rejected_before_arc_when_not_discretely_balanced():
    check = panel_initial_equilibrium_preflight(build_panel_case(2))
    assert check["passed"] is False
    assert check["free_dof_internal_force_norm_n"] > 1e5
    assert check["relative_free_dof_internal_force_norm"] > 1e-2


def test_matrix_persists_six_fail_closed_cells_and_resumes(tmp_path, monkeypatch):
    import tensorfem.marine_panel_execution as execution
    monkeypatch.setattr(execution, "panel_geometry_preflight", lambda case: {
        "passed": False, "first_invalid_element": 0,
        "first_error": "synthetic excessive warp",
    })
    report = execute_panel_matrix(tmp_path, steps=2)
    assert report["status"] == "blocked"
    assert report["required_cells"] == 6
    assert report["completed_cells"] == 0
    saved = json.loads((tmp_path / "matrix.json").read_text())
    assert saved["evidence_sha256"] == report["evidence_sha256"]

    replayed = execute_panel_job(4, .02, steps=2, cache_dir=tmp_path)
    assert replayed["replayed"] is True


def test_first_point_gate_advances_in_order_and_records_integrity(tmp_path, monkeypatch):
    import tensorfem.marine_panel_execution as execution
    calls = []

    def fake_job(divisions, step, **kwargs):
        calls.append((divisions, step, kwargs["steps"], kwargs["maximum_wall_seconds"]))
        return {
            "status": "executed", "accepted_points": 1,
            "maximum_relative_free_dof_equilibrium_norm": 2e-8,
            "accepted_state_sha256": "a" * 64,
            "initial_equilibrium_preflight": {"passed": True},
            "elapsed_seconds": divisions,
        }

    monkeypatch.setattr(execution, "execute_panel_job", fake_job)
    report = execute_panel_performance_gate(
        tmp_path, first_point_limits={2: 10, 4: 20}
    )
    assert report["passed"] is True
    assert calls == [(2, .02, 1, 10), (4, .02, 1, 20)]
    assert report["solver"] == "matrix_free"
    assert len(report["evidence_sha256"]) == 64
    assert json.loads((tmp_path / "performance-gate.json").read_text())["passed"]


def test_first_point_gate_stops_before_4x4_after_timeout(tmp_path, monkeypatch):
    import tensorfem.marine_panel_execution as execution
    calls = []

    def fake_job(divisions, step, **kwargs):
        calls.append(divisions)
        return {"status": "failed", "accepted_points": 0,
                "elapsed_seconds": kwargs["maximum_wall_seconds"]}

    monkeypatch.setattr(execution, "execute_panel_job", fake_job)
    report = execute_panel_performance_gate(tmp_path)
    assert report["status"] == "blocked"
    assert calls == [2]
    assert len(report["stages"]) == 1
