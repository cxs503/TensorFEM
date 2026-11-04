"""Tests for the finite-width plate (Howland/Heywood) verification unit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tensorfem.finite_width_hole import (
    PETERSON_HOWLAND_KTN,
    FiniteWidthPlate,
    edge_loads,
    heywood_stress_concentration,
    load_balance,
    peterson_stress_concentration,
    plate_model,
    solve_case,
    starburst_mesh,
    symmetry_dofs,
)
from tensorfem.continuum import solve_continuum

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Reference-side self-checks (the yardstick must be right before it judges code)
# ---------------------------------------------------------------------------

def test_heywood_fit_limits():
    assert heywood_stress_concentration(0.0) == pytest.approx(3.0, abs=1e-12)
    assert heywood_stress_concentration(0.5) == pytest.approx(2.125, abs=1e-12)
    values = [heywood_stress_concentration(r) for r in np.linspace(0.0, 0.9, 40)]
    assert all(a >= b for a, b in zip(values, values[1:])), "K_tn must decay with d/W"


def test_heywood_fit_reproduces_howland_table():
    assert PETERSON_HOWLAND_KTN[0.2] == pytest.approx(2.51)
    for ratio, tabulated in PETERSON_HOWLAND_KTN.items():
        assert abs(heywood_stress_concentration(ratio) - tabulated) / tabulated < 0.02


def test_peterson_lookup_is_exact_only_on_table():
    assert peterson_stress_concentration(0.2) == pytest.approx(2.51)
    assert peterson_stress_concentration(0.25) is None
    assert FiniteWidthPlate(diameter=0.3).peterson_ktn == pytest.approx(2.35)
    assert FiniteWidthPlate(diameter=0.25).peterson_ktn is None


def test_configuration_guards_are_fail_closed():
    with pytest.raises(ValueError):
        FiniteWidthPlate(width=1.0, diameter=1.0)
    with pytest.raises(ValueError):
        FiniteWidthPlate(width=1.0, diameter=1.4)
    with pytest.raises(ValueError):
        FiniteWidthPlate(width=1.0, diameter=0.4, half_length=0.3)
    with pytest.raises(ValueError):
        heywood_stress_concentration(1.0)
    with pytest.raises(ValueError):
        heywood_stress_concentration(-0.1)


def test_net_and_gross_relation():
    plate = FiniteWidthPlate(diameter=0.5)
    assert plate.ratio == pytest.approx(0.5)
    assert plate.net_section_stress == pytest.approx(2.0 * plate.tension)


# ---------------------------------------------------------------------------
# Mesh and load topology
# ---------------------------------------------------------------------------

def test_mesh_geometry_invariants():
    plate = FiniteWidthPlate(diameter=0.2, half_length=1.5)
    nodes, elements, meta = starburst_mesh(plate, n_boundary=16, n_radial=12)
    xy = np.asarray(nodes, dtype=np.float64)
    radius = np.hypot(xy[:, 0], xy[:, 1])
    hole = np.array(meta["hole_nodes"])
    assert np.allclose(radius[hole], plate.hole_radius, atol=1e-12)
    assert np.all(xy[:, 0] >= -1e-12) and np.all(xy[:, 1] >= -1e-12)
    assert np.all(xy[:, 0] <= plate.half_length + 1e-12)
    assert np.all(xy[:, 1] <= plate.half_width + 1e-12)
    # counter-clockwise, non-degenerate quads
    index = np.asarray(elements)
    x, y = xy[index, 0], xy[index, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    assert np.all(area > 0.0)
    # the load-transfer corner (L, H) must be a mesh node
    corner = np.all(np.isclose(xy, [plate.half_length, plate.half_width]), axis=1)
    assert corner.sum() == 1


def test_loaded_edge_carries_the_full_quarter_load():
    plate = FiniteWidthPlate(diameter=0.2, half_length=1.5)
    _, _, meta = starburst_mesh(plate, n_boundary=20, n_radial=10)
    forces = np.asarray(edge_loads(plate, meta), dtype=np.float64)
    total = forces[0::2].sum()
    assert total == pytest.approx(plate.tension * plate.thickness * plate.half_width, rel=1e-13)
    assert abs(forces[1::2]).max() == 0.0, "loaded end has no shear component"
    lateral = [node for node, kind in zip(range(len(meta["radii"])), meta["boundary_kind"])
               if kind == "free"]
    outer_free = [node * meta["n_columns"] + meta["n_radial"] for node in lateral]
    for node in outer_free:
        assert forces[2 * node] == 0.0 and forces[2 * node + 1] == 0.0


def test_symmetry_dofs_match_symmetry_planes():
    plate = FiniteWidthPlate(diameter=0.2)
    nodes, _, meta = starburst_mesh(plate, n_boundary=12, n_radial=8)
    xy = np.asarray(nodes, dtype=np.float64)
    fixed = symmetry_dofs(meta)
    assert fixed == sorted(set(fixed))
    for dof in fixed:
        node, component = divmod(dof, 2)
        if component == 1:
            assert abs(xy[node, 1]) < 1e-12
        else:
            assert abs(xy[node, 0]) < 1e-12


# ---------------------------------------------------------------------------
# Production solve against the literature anchor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.2, 0.5])
def test_stress_concentration_within_three_percent(ratio):
    plate = FiniteWidthPlate(diameter=ratio, half_length=1.5)
    case = solve_case(plate, n_boundary=24)
    assert 2.0 < case["ktn_fe"] < 3.0
    assert case["error_vs_heywood"] < 0.03, case["error_vs_heywood"]
    assert abs(case["ktn_fe"] - case["ktn_heywood"]) / case["ktn_heywood"] < 0.03
    if case["ktn_peterson"] is not None:
        assert case["error_vs_peterson"] < 0.03
    # gross/net bookkeeping is the only difference between the two conventions
    assert case["ktg_fe"] == pytest.approx(case["ktn_fe"] / (1.0 - ratio), rel=1e-12)


def test_peak_sits_on_the_symmetry_line():
    case = solve_case(FiniteWidthPlate(diameter=0.3, half_length=1.5), n_boundary=20)
    assert case["peak_theta_degrees"] == pytest.approx(90.0, abs=1e-9)


def test_equilibrium_is_exact_and_matches_applied_load():
    plate = FiniteWidthPlate(diameter=0.2, half_length=1.5)
    _, _, meta = starburst_mesh(plate, n_boundary=16, n_radial=16)
    result = solve_continuum(plate_model(plate, meta))
    balance = load_balance(plate, meta, result.reaction.detach().cpu().numpy())
    assert balance["relative_error"] < 1e-10
    assert balance["applied"] == pytest.approx(plate.tension * plate.thickness * plate.half_width)


def test_kirsch_limit_is_recovered_as_hole_shrinks():
    """Exact anchor: with d/W -> 0 the gross factor must return to Kirsch's 3."""
    plate = FiniteWidthPlate(diameter=0.02, half_length=1.5)
    case = solve_case(plate, n_boundary=32)
    assert abs(case["ktg_fe"] - 3.0) / 3.0 < 0.01, case["ktg_fe"]


def test_hole_edge_profiles_match_kirsch_shape():
    """Hoop stress is -T at the waist and the peak at theta = 90 (Kirsch: 1 - 2 cos 2 theta)."""
    case = solve_case(FiniteWidthPlate(diameter=0.2, half_length=1.5), n_boundary=24)
    hoop = np.array(case["hole_edge_hoop_profile_over_tension"])
    axial = np.array(case["hole_edge_axial_profile_over_tension"])
    assert hoop[0, 0] == pytest.approx(0.0)
    assert hoop[-1, 0] == pytest.approx(90.0)
    assert hoop[0, 1] == pytest.approx(-1.0, abs=0.2), hoop[0, 1]
    assert axial[0, 1] == pytest.approx(0.0, abs=0.05), axial[0, 1]
    assert hoop[-1, 1] == pytest.approx(case["ktg_fe"])


def test_mesh_refinement_approaches_the_reference_monotonically():
    plate = FiniteWidthPlate(diameter=0.4, half_length=1.5)
    series = [solve_case(plate, n_boundary=level)["ktn_fe"] for level in (12, 20, 28)]
    assert series[0] < series[1] < series[2]
    assert abs(series[-1] - plate.heywood_ktn) / plate.heywood_ktn < 0.03


# ---------------------------------------------------------------------------
# Stored evidence must stay consistent with the code
# ---------------------------------------------------------------------------

def test_stored_evidence_gates():
    path = ROOT / "benchmarks" / "finite_width_hole_reference.json"
    if not path.exists():
        pytest.skip("evidence not generated in this checkout")
    data = json.loads(path.read_text())
    assert data["headline_level"] == 40
    assert data["max_error_vs_heywood"] < 0.03
    assert data["max_error_vs_peterson"] < 0.03
    assert data["max_load_balance_error"] < 1e-10
    for row in data["headline"]:
        assert row["error_vs_heywood"] < 0.03
        extrapolated = row["richardson"]["extrapolated"]
        if extrapolated is not None:
            assert abs(extrapolated - row["ktn_heywood"]) / row["ktn_heywood"] < 0.02
    for case in data["kirsch_limit"]:
        assert case["ktg_error_vs_kirsch"] < 0.01
    for block in data["saint_venant"]:
        assert block["designated_error"] < 0.03
    for figure in data["figures"]:
        assert (ROOT / figure).exists(), figure