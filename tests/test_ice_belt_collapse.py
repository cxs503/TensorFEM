"""Regression tests for the ice-belt elastoplastic progressive-collapse surface.

Each test names the campaign criterion it guards:

* ``test_steel_law_*``            -- hardening law and in-module calibration
* ``test_ice_pressure_*``         -- IACS pressure-area law
* ``test_element_*``              -- MITC4/layered element consistency
* ``test_C1_*``                   -- elastic plate bending vs Navier (<= 1 %)
* ``test_C2_*``                   -- plastic hinge moment vs 1-D section law (<= 3 %)
* ``test_C3_*``                   -- plate-field buckling vs CSR (<= 5 %, slow lane)
* ``test_C6_*``                   -- ice-belt limit pressure vs code (<= 15 %, slow lane)
* ``test_energy_ledger``          -- external work = elastic + plastic energy (<= 2 %)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ice_belt_collapse import (
    AH36, DH36, DOF_PER_NODE, E_STEEL, IACSPressureArea, I_RX, NU_STEEL, PlateModel,
    Stiffener, analytic_moment_curvature, calibrate_user_K, csr_plate_buckling_stress,
    edge_nodes, elastic_perfectly_plastic, elastic_plate_case, euler_column_stress,
    ice_belt_case, inplane_free_bcs, line_nodes, navier_ssss_centre,
    plate_buckling_case, plastic_hinge_case, section_moment,
    solve_displacement_control, solve_load_control, structured_mesh,
    tee_section, uniform_pressure_load, _mitc4_shear_operators,
)


# --------------------------------------------------------------------------- #
# material law
# --------------------------------------------------------------------------- #
def test_steel_law_calibration_passes_through_tensile_strength():
    K = calibrate_user_K(3.55e8, 4.90e8, 0.15, 0.5)
    assert K == pytest.approx((4.90e8 - 3.55e8) / math.sqrt(0.15), rel=1e-12)
    assert float(AH36.yield_stress(0.15)) == pytest.approx(4.90e8, rel=1e-12)
    # p is floored at 1e-12 to keep the hardening slope finite
    assert float(AH36.yield_stress(0.0)) == pytest.approx(AH36.sigma_0, abs=1.0e3)


def test_steel_law_is_monotonic_and_hardening():
    p = np.linspace(0.0, 0.2, 50)
    sigma = AH36.yield_stress(p)
    assert np.all(np.diff(sigma) >= 0.0)
    assert sigma[0] == pytest.approx(AH36.sigma_0, abs=1.0e3)
    assert DH36.yield_stress(p) == pytest.approx(sigma, rel=1e-12)  # same strength


def test_plastic_work_increment_is_the_integral_of_the_law():
    grade = AH36
    p0, p1 = 0.01, 0.012
    fine = np.linspace(p0, p1, 20001)
    expected = np.trapezoid(grade.yield_stress(fine), fine)
    got = float(grade.plastic_work_increment(np.array([p0]), np.array([p1]))[0])
    assert got == pytest.approx(expected, rel=1e-6)


def test_section_law_reaches_the_plastic_moment():
    epp = elastic_perfectly_plastic()
    mp = epp.sigma_0 * 0.1 ** 2 / 4.0
    high = analytic_moment_curvature(epp, 100.0 * 2 * epp.sigma_0 / (epp.young * 0.1), 0.1)
    assert high == pytest.approx(mp, rel=1e-3)


# --------------------------------------------------------------------------- #
# ice load
# --------------------------------------------------------------------------- #
def test_ice_pressure_area_law():
    ice = IACSPressureArea(p0=2.0e6, area_ref=1.0, exponent=0.3)
    assert float(ice.pressure(1.0)) == pytest.approx(2.0e6, rel=1e-12)
    assert float(ice.pressure(4.0)) == pytest.approx(2.0e6 * 4.0 ** -0.3, rel=1e-12)
    assert float(ice.pressure(4.0)) < float(ice.pressure(1.0))
    with pytest.raises(ValueError):
        ice.pressure(0.0)
    with pytest.raises(ValueError):
        IACSPressureArea(p0=-1.0).pressure(1.0)


def test_ice_pressure_total_force_scales_with_area():
    ice = IACSPressureArea(p0=2.0e6)
    f1 = float(ice.total_force(1.0))
    f4 = float(ice.total_force(4.0))
    assert f4 / f1 == pytest.approx(4.0 ** 0.7, rel=1e-12)


# --------------------------------------------------------------------------- #
# element consistency
# --------------------------------------------------------------------------- #
def test_mitc4_shear_operator_reproduces_a_constant_shear_field():
    """A linear ``w`` field with zero rotation is a pure shear state."""
    length, width = 0.2, 0.1
    coords = np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]])
    ops = _mitc4_shear_operators(coords)
    ue = np.zeros(20)
    for i in range(4):
        ue[DOF_PER_NODE * i + 2] = 1.0e-3 * coords[i, 0]
    gamma = np.einsum("qij,j->qi", ops, ue)
    assert np.allclose(gamma[:, 0], 1.0e-3, rtol=1e-12)
    assert np.allclose(gamma[:, 1], 0.0, atol=1e-18)


def test_layer_operators_integrate_the_section_correctly():
    nodes, elems = structured_mesh(0.4, 0.2, 4, 2)
    model = PlateModel(nodes, elems, 0.02, AH36, n_layers=5)
    # sum of the layer measures x the in-plane Gauss measure = element volume
    volume = float(model.layer_weight.sum())
    assert volume == pytest.approx(0.4 * 0.2 * 0.02, rel=1e-12)
    # first moment of z must vanish for a symmetric section
    assert float(np.sum(model.layer_z * model.layer_weight)) == pytest.approx(0.0, abs=1e-18)


def test_tee_section_matches_closed_form():
    sec = tee_section(0.25, 0.01, 0.10, 0.015, 0.03)
    assert sec["area"] == pytest.approx(0.0025 + 0.0015, rel=1e-12)
    # parallel-axis combination about the centroid, recomputed independently
    a_w, a_f = 0.0025, 0.0015
    z_w, z_f = -(0.015 + 0.125), -(0.015 + 0.25 + 0.0075)
    centroid = (a_w * z_w + a_f * z_f) / (a_w + a_f)
    inertia = (0.01 * 0.25 ** 3 / 12 + a_w * (z_w - centroid) ** 2
               + 0.10 * 0.015 ** 3 / 12 + a_f * (z_f - centroid) ** 2)
    assert sec["offset"] == pytest.approx(-centroid, rel=1e-12)
    assert sec["inertia"] == pytest.approx(inertia, rel=1e-12)


def test_euler_column_stress_of_a_bare_plate_strip():
    """Zero stiffener -> the plate strip must buckle as a column."""
    span, width, thickness = 2.0, 0.3, 0.02
    out = euler_column_stress(span, width, thickness, 0.0, 0.0, 0.0, E_STEEL)
    expected = math.pi ** 2 * E_STEEL * thickness ** 2 / (12.0 * span ** 2)
    assert out["euler_stress"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# C1: elastic plate vs Navier
# --------------------------------------------------------------------------- #
def test_C1_elastic_plate_deflection_matches_navier():
    result = elastic_plate_case(16)
    assert abs(result["rel_error"]) <= 0.01, result


def test_C1_navier_reference_recovers_the_classical_coefficient():
    """Square SSSS plate: w_max D/(q a^4) must equal the classical 0.00406."""
    thickness = 0.005
    rigidity = E_STEEL * thickness ** 3 / (12.0 * (1.0 - NU_STEEL ** 2))
    coefficient = navier_ssss_centre(1.0, 1.0, thickness, E_STEEL, NU_STEEL, 1.0) * rigidity
    assert coefficient == pytest.approx(0.004062, rel=2e-3)


# --------------------------------------------------------------------------- #
# C2: plastic hinge
# --------------------------------------------------------------------------- #
def _hinge_model(grade, layers=5):
    length, width, thickness = 0.2, 0.05, 0.1
    nodes, elems = structured_mesh(length, width, 8, 4)
    x, y = nodes[:, 0], nodes[:, 1]
    lower = int(np.argmin(x + y))
    upper = int(np.argmin(x + (y.max() - y)))
    fixed = [DOF_PER_NODE * n + 0 for n in range(len(nodes))]
    fixed += [DOF_PER_NODE * lower + 2, DOF_PER_NODE * lower + 1,
              DOF_PER_NODE * upper + 2]
    model = PlateModel(nodes, elems, thickness, grade, n_layers=layers)
    model.von_karman = False
    return model, fixed, length, thickness


def test_C2_elastic_limit_of_the_layered_section():
    grade = AH36
    model, fixed, length, thickness = _hinge_model(grade)
    kappa = 0.5 * 2 * grade.sigma_0 / (grade.young * thickness)
    theta = 0.5 * kappa * length
    steps = np.linspace(0.0, theta, 11)[1:]
    prescribed = {}
    for n in edge_nodes(model, "x0"):
        prescribed[DOF_PER_NODE * int(n) + I_RX] = steps
    for n in edge_nodes(model, "x1"):
        prescribed[DOF_PER_NODE * int(n) + I_RX] = -steps
    path = solve_displacement_control(model, fixed, prescribed, steps,
                                      tol=1e-9, max_iter=20)
    assert len(path.u) == len(steps)
    moment = abs(section_moment(model, path.u[-1], path.state, path.ez, 4))
    reference = analytic_moment_curvature(grade, kappa, thickness)
    assert moment / reference - 1.0 == pytest.approx(0.0, abs=5e-3)


def test_C2_plastic_hinge_case_within_tolerance():
    result = plastic_hinge_case(ratios=(0.5, 1.0, 2.0))
    assert result["all_converged"], result["rows"]
    assert result["max_abs_error"] <= 0.03, result["rows"]
    assert result["full_plastic_moment"] == pytest.approx(3.55e8 * 0.1 ** 2 / 4.0)


# --------------------------------------------------------------------------- #
# C3: plate-field buckling vs CSR (slow lane)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_C3_plate_field_buckling_matches_csr_formula():
    result = plate_buckling_case(24, 16)
    assert abs(result["rel_error"]) <= 0.05, result
    assert result["sigma_fe"] > 0.0


def test_C3_csr_formula_reduces_to_k4_for_long_plates():
    long_plate = csr_plate_buckling_stress(10.0, 0.8, 0.01, E_STEEL, NU_STEEL)
    square = csr_plate_buckling_stress(0.8, 0.8, 0.01, E_STEEL, NU_STEEL)
    assert square / long_plate == pytest.approx(1.0, rel=5e-2)


# --------------------------------------------------------------------------- #
# C6: ice-belt limit pressure (slow lane)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_C6_ice_belt_limit_pressure_against_code_yield_line():
    result = ice_belt_case(nx=24, ny=8, steps=30)
    assert result["collapse_pressure"] is not None, result
    assert abs(result["rel_error"]) <= 0.15, {k: result[k] for k in
                                              ("collapse_pressure", "code_pressure",
                                               "rel_error", "collapse_basis")}
    assert abs(result["energy_residual"]) <= 0.02, result["energy_residual"]
    assert result["collapse_lower_bound"] <= result["collapse_pressure"] * 1.0000001


@pytest.mark.slow
def test_energy_ledger_closes_on_the_ice_belt_case():
    result = ice_belt_case(nx=16, ny=6, steps=16)
    assert abs(result["energy_residual"]) <= 0.02, result["energy_residual"]


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_bad_mesh_and_material_inputs_are_rejected():
    with pytest.raises(ValueError):
        structured_mesh(1.0, 1.0, 0, 4)
    nodes, elems = structured_mesh(0.4, 0.2, 2, 2)
    with pytest.raises(ValueError):
        PlateModel(nodes, elems, 0.01, AH36, n_layers=4)
    model = PlateModel(nodes, elems, 0.01, AH36)
    with pytest.raises(ValueError):
        model.set_imperfection(np.zeros(3))


def test_imperfection_does_not_load_the_flat_state():
    """The out-of-flatness enters as an initial slope: the flat state stays in
    equilibrium (no spurious internal force)."""
    nodes, elems = structured_mesh(0.6, 0.3, 6, 3)
    model = PlateModel(nodes, elems, 0.012, AH36)
    w0 = 0.004 * np.sin(np.pi * nodes[:, 0] / 0.6) * np.sin(np.pi * nodes[:, 1] / 0.3)
    model.set_imperfection(w0)
    ue, Baug, strain = model.layer_kinematics(np.zeros(model.ndof))
    assert np.max(np.abs(strain)) == pytest.approx(0.0, abs=1e-18)
    force = model.assembled_internal_force(ue, Baug, np.zeros((model.n_points, 3)),
                                           model.shear_strain(ue))
    assert np.max(np.abs(force)) == pytest.approx(0.0, abs=1e-18)


def test_stiffener_axial_rigidity_and_geometric_term():
    from tensorfem.ice_belt_collapse import stiffener_triplets
    nodes, elems = structured_mesh(1.0, 0.4, 10, 4)
    model = PlateModel(nodes, elems, 0.02, AH36)
    line = line_nodes(model, "x", 0.0)
    stiff = Stiffener(node_line=line, direction="x", area=0.004,
                      inertia=3e-5, offset=0.15)
    rows, cols, data = stiffener_triplets(model, stiff)
    matrix = np.zeros((model.ndof, model.ndof))
    np.add.at(matrix, (rows, cols), data)
    # sum of the axial diagonal = 2 * sum(EA/L) over the segments
    segments = line.size - 1
    axial = sum(matrix[DOF_PER_NODE * int(n), DOF_PER_NODE * int(n)]
                for n in line)
    assert axial == pytest.approx(2.0 * 0.004 * E_STEEL * segments / 0.1, rel=1e-9)
    # a rigid transverse translation of the whole line carries no stiffener energy
    w = np.zeros(model.ndof)
    w[2::DOF_PER_NODE] = 1.0e-3
    assert float(w @ matrix @ w) == pytest.approx(0.0, abs=1e-12)
    # geometric term: compression softens a column mode
    rows_g, cols_g, data_g = stiffener_triplets(model, stiff, axial_force=-1.0e5,
                                                include_elastic=False)
    assert data_g.size == 4 * segments
    geo = np.zeros((model.ndof, model.ndof))
    np.add.at(geo, (rows_g, cols_g), data_g)
    # compression leaves a rigid transverse translation energy-free ...
    assert float(w @ geo @ w) == pytest.approx(0.0, abs=1e-9)
    # ... and softens a mode with a slope (negative geometric energy)
    ramp = np.zeros(model.ndof)
    ramp[2::DOF_PER_NODE] = np.linspace(0.0, 1.0e-3, model.n_nodes)
    assert float(ramp @ geo @ ramp) < 0.0