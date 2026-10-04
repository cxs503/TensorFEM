import json

import pytest

from tensorfem.marine_panel_execution import (
    dimensional_arc_controls, execute_panel_job, execute_panel_matrix,
    panel_geometry_preflight,
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
