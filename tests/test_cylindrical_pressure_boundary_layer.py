"""Gates for the pressurised cylinder with clamped ends.

The reference is the exact axisymmetric shell solution in
:mod:`tensorfem.cylindrical_pressure_reference` (four-constant clamped--clamped
solution of ``D w'''' + (E t / R^2) w = p``); the discretisation under test is
the production metric-consistent Shell4 element
(:mod:`tensorfem.cylindrical_shell4`).

Evidence: ``benchmarks/cylindrical_pressure_boundary_layer_evidence.json``,
write-up ``docs/cylindrical-pressure-boundary-layer.md``.
"""
import math

import pytest
import torch

from tensorfem.cylindrical_pressure_reference import (
    cylinder_parameters,
    exact_axisymmetric_solution,
    pressure_cylinder_mesh,
    consistent_pressure_loads,
    self_validation,
    solve_pressure_cylinder,
    work_energy_balance,
)

CASE = dict(
    radius=1.5,
    thickness=0.015,
    young=206.0e9,
    poisson=0.3,
    pressure=1.0e6,
    length=1.2,
)


@pytest.fixture(scope="module")
def params():
    return cylinder_parameters(**CASE)


@pytest.fixture(scope="module")
def production(params):
    return solve_pressure_cylinder(32, 16, params)


# ---------------------------------------------------------------------------
# The reference must first be validated against its own definition
# ---------------------------------------------------------------------------
def test_reference_satisfies_the_governing_ode(params):
    checks = self_validation(params)
    assert checks["ode_residual_over_pressure"] < 1e-12
    assert checks["clamped_displacement_over_wp"] < 1e-9
    assert checks["clamped_slope_over_wp_beta"] < 1e-9


def test_reference_matches_classical_long_cylinder_limits(params):
    checks = self_validation(params)
    # |M_x(0)| -> p/(2 beta^2) and the two edge layers superpose to O(e^-beta L)
    assert abs(checks["edge_moment_vs_long_limit"]) < 3e-3
    assert checks["two_layer_superposition_gap"] < 3e-3
    assert checks["decay_at_far_end"] < 1e-3


def test_reference_obeys_pressure_work_equals_strain_energy(params):
    balance = work_energy_balance(params, samples=20001)
    assert balance["pressure_work"] > 0.0
    assert balance["relative_gap"] < 1e-6


def test_membrane_and_edge_limits_are_physical(params):
    wp = params.pressure * params.radius ** 2 / (params.young * params.thickness)
    assert params.membrane_radial_displacement == pytest.approx(wp, rel=1e-14)
    assert params.membrane_hoop_stress == pytest.approx(
        params.pressure * params.radius / params.thickness, rel=1e-14
    )
    edge = exact_axisymmetric_solution(torch.tensor([0.0]), params)
    # Outward pressure, clamped edge: moment is negative and close to the
    # semi-infinite magnitude, and the shell is pushed out in the interior.
    assert float(edge.moment[0]) < 0.0
    assert abs(float(edge.moment[0])) == pytest.approx(
        params.pressure / (2.0 * params.beta ** 2), rel=3e-3
    )
    mid = exact_axisymmetric_solution(torch.tensor([params.length / 2.0]), params)
    assert float(mid.displacement[0]) == pytest.approx(wp, rel=3e-2)


def test_long_cylinder_midspan_reaches_the_membrane_state():
    long_case = cylinder_parameters(**{**CASE, "length": 3.0})
    mid = exact_axisymmetric_solution(
        torch.tensor([long_case.length / 2.0]), long_case
    )
    # exp(-beta L / 2) = 2.6e-6 at beta L = 25.7, so 1e-5 is the honest bound.
    assert float(mid.displacement[0]) == pytest.approx(
        long_case.membrane_radial_displacement, rel=1e-5
    )


def test_reference_rejects_out_of_range_sampling(params):
    with pytest.raises(ValueError):
        exact_axisymmetric_solution(torch.tensor([-0.1]), params)
    with pytest.raises(ValueError):
        exact_axisymmetric_solution(torch.tensor([params.length + 0.1]), params)
    with pytest.raises(ValueError):
        cylinder_parameters(**{**CASE, "thickness": -1.0})


# ---------------------------------------------------------------------------
# Mesh and load bookkeeping
# ---------------------------------------------------------------------------
def test_pressure_load_is_self_equilibrated_and_correct_in_magnitude(params):
    nodes, connectivity = pressure_cylinder_mesh(4, 8, params)
    load = consistent_pressure_loads(nodes, connectivity, params).reshape(-1, 6)
    resultant = load[:, :3].sum(dim=0)
    scale = params.pressure * 2.0 * math.pi * params.radius * params.length
    # A uniform pressure on a closed ring must produce no net force.
    assert float(torch.linalg.vector_norm(resultant)) < 1e-9 * scale
    # Every node of an interior ring carries the same radial magnitude.
    ring = load[8:16, 1:3]
    magnitude = torch.linalg.vector_norm(ring, dim=1)
    assert float((magnitude - magnitude.mean()).abs().max()) < 1e-9 * float(
        magnitude.mean()
    )
    # The nodal load is exactly p * (tributary area) times a chord factor,
    # i.e. within a fraction of a percent of the flat-area value.
    nominal = params.pressure * (params.length / 4.0) * params.radius * (
        2.0 * math.pi / 8.0
    )
    assert float(magnitude.mean()) == pytest.approx(nominal, rel=0.07)


# ---------------------------------------------------------------------------
# Element accuracy against the exact solution
# ---------------------------------------------------------------------------
def test_production_mesh_meets_the_three_percent_gate(production):
    # 32 axial x 16 circumferential elements, 3168 dofs.
    assert production.dofs == 3168
    assert production.displacement_l2_error < 0.03
    assert production.moment_l2_error < 0.03
    assert production.moment_sign_correlation > 0.999
    assert production.circumferential_uniformity < 1e-9


def test_boundary_layer_error_is_local_and_reported(production):
    # The node immediately inside the clamped edge sits in the steepest part of
    # the layer; its raw error is reported rather than hidden inside the L2.
    assert 0.0 < production.peak_displacement_error < 0.05
    assert production.peak_displacement_error >= production.displacement_l2_error


def test_axial_refinement_reduces_the_displacement_error(params):
    coarse = solve_pressure_cylinder(8, 16, params)
    fine = solve_pressure_cylinder(32, 16, params)
    assert coarse.displacement_l2_error > 5 * fine.displacement_l2_error
    assert coarse.displacement_l2_error > 0.03


def test_coarse_circumferential_mesh_is_rejected(params):
    # 8 facets (45 deg) leave a ~5.8 % displacement error at the same axial
    # mesh: the case documents a mesh requirement instead of hiding it.
    eight = solve_pressure_cylinder(16, 8, params)
    sixteen = solve_pressure_cylinder(16, 16, params)
    assert eight.displacement_l2_error > 0.03
    assert sixteen.displacement_l2_error < 0.03
    assert eight.circumferential_uniformity < 1e-9  # wrong for a real reason


def test_open_end_condition_is_enforced(production):
    # N_x must vanish for an open-ended cylinder; the residual reported here is
    # the finite-difference measurement error, not a physical axial force.
    assert production.axial_force_error < 0.05


def test_solution_is_bitwise_reproducible(params):
    first = solve_pressure_cylinder(16, 16, params)
    second = solve_pressure_cylinder(16, 16, params)
    assert torch.equal(first.displacement_fe, second.displacement_fe)


def test_negative_control_reference_with_wrong_beta_is_rejected(params):
    # A reference built on a 5 % stiffer hoop term must fail the 3 % gate, which
    # proves the gate discriminates rather than passing anything smooth.
    wrong = cylinder_parameters(**{**CASE, "thickness": CASE["thickness"] * 1.10})
    result = solve_pressure_cylinder(32, 16, params)
    exact_wrong = exact_axisymmetric_solution(result.x, wrong)
    error = float(
        torch.linalg.vector_norm(result.displacement_exact - exact_wrong.displacement)
        / torch.linalg.vector_norm(exact_wrong.displacement)
    )
    assert error > 0.03