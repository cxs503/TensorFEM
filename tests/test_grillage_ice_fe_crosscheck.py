"""Tests for the nonlinear-FE cross-validation of the ice-grillage ultimate.

Anchors: the FE-vs-closed-form conservatism cross-check (``T11``) over the
three combined ice-patch + lateral-pressure cases, plus the uniform-branch
yield-line anchor (``T11a``).  The closed-form ``plate_local_bending``
capacity is a *first-yield* lower bound, so the FE collapse must be **at or
above** it; the ratio is the quantified conservatism margin.
"""
from __future__ import annotations

import json

import pytest

from tensorfem.grillage_ice_fe_crosscheck import (
    build_fe_panel,
    crosscheck_plate_local_bending,
    fe_lateral_collapse,
    run_fe_crosscheck,
    yield_line_uniform_pressure,
)
from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    case_study_geometry,
)


@pytest.fixture(scope="module")
def geom():
    return case_study_geometry()


def test_build_fe_panel_geometry(geom):
    model = build_fe_panel(geom, nx=14, ny=4)
    assert model.n_elem == 14 * 4
    assert model.n_layers == 5
    assert model.thickness == pytest.approx(geom.plate_thickness)


def test_yield_line_uniform_pressure(geom):
    # p_u = 8 M_p / b^2 with M_p = sigma_y t^2/4, b = bay_spacing
    mp = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    assert yield_line_uniform_pressure(geom) == pytest.approx(
        8.0 * mp / geom.bay_spacing ** 2)


def test_fe_collapse_is_conservative_for_all_combined_cases(geom):
    """The closed-form first-yield capacity must not over-predict the FE
    collapse: the FE peak load factor is at or above 1/u_plate_eta0."""
    result = run_fe_crosscheck(geom, nx=14, ny=4, n_steps=100, dlam0=0.03)
    assert result["passed"], [c for c in result["cases"] if not c["pass_conservative"]]
    assert result["n_cases"] >= 4
    # the three combined cases carry a conservatism ratio > 1 (FE above closed-form)
    combined = [c for c in result["cases"] if "yield-line" not in c["scenario"]]
    assert len(combined) == 3
    for c in combined:
        assert c["conservatism_ratio"] > 1.0


def test_fe_collapse_above_yield_line_uniform(geom):
    """The uniform-branch FE collapse is at or above the rigid-plastic
    p_u = 8 M_p / b^2 (membrane stiffening raises the collapse)."""
    uniform = CombinedLoadCase(
        axial_stress=0.0, ice_pressure=0.0, patch_length=0.0, patch_width=0.0,
        hydrostatic_pressure=1.0e6)
    fe = fe_lateral_collapse(geom, uniform, nx=14, ny=4, n_steps=100, dlam0=0.05)
    p_u = yield_line_uniform_pressure(geom)
    assert fe["lam_peak"] * fe["combined_ref"] >= p_u - 1e-9


def test_crosscheck_rejects_bad_input(geom):
    with pytest.raises(ValueError):
        build_fe_panel(geom, nx=2, ny=2)
    with pytest.raises(ValueError):
        crosscheck_plate_local_bending(
            geom, CombinedLoadCase(axial_stress=-1.0, ice_pressure=1e6,
                                   patch_length=0.5, patch_width=0.5,
                                   hydrostatic_pressure=0.0))


def test_main_writes_artifact(tmp_path, geom):
    from tensorfem.grillage_ice_fe_crosscheck import main
    payload = main(tmp_path)
    assert (tmp_path / "fe_crosscheck.json").exists()
    data = json.loads((tmp_path / "fe_crosscheck.json").read_text())
    assert data["passed"] == payload["passed"]
