"""Gates for the elastic-plastic thick-walled cylinder (Hill 1950 plane strain).

Two independent layers are exercised:

* the closed-form oracle itself is audited against the *physical identities* it
  must satisfy (bore traction, traction-free outer surface, exact yield in the
  plastic zone, strict elasticity in the elastic zone, equilibrium, interface
  continuity), so a typo in the formula cannot silently become the reference;
* the plane-strain axisymmetric Q4 model is checked first against the **elastic**
  Lamé solution (element quality) and then against the elastic-plastic solution,
  with the campaign quantity ``r_p`` (elastic-plastic interface radius) required
  to be within 3% of the closed form.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.j2_saturation_hardening import HardeningLaw
from tensorfem.thick_cylinder_plasticity import (
    HillPlaneStrainTube,
    annulus_mesh,
    field_errors,
    solve_plastic_tube,
)

torch.set_num_threads(2)

ELASTIC = 200.0e9
NU = 0.3


@pytest.fixture(scope="module")
def tube() -> HillPlaneStrainTube:
    return HillPlaneStrainTube(young=ELASTIC, poisson=NU)


@pytest.fixture(scope="module")
def law(tube) -> HardeningLaw:
    return HardeningLaw(tube.yield_stress)


@pytest.fixture(scope="module")
def elastic_solution(tube, law):
    pressure = 0.5 * tube.pressure_first_yield
    return solve_plastic_tube(tube, annulus_mesh(16, grading=0.1), law, pressure, steps=3)


@pytest.fixture(scope="module")
def plastic_solution(tube, law):
    pressure = 1.0 * tube.k
    return solve_plastic_tube(tube, annulus_mesh(24, grading=0.1), law, pressure, steps=6)


# --------------------------------------------------------------------------- #
# closed-form oracle audit
# --------------------------------------------------------------------------- #
def test_pressure_limits_follow_from_the_plastic_radius_formula(tube):
    a, b, k = tube.inner_radius, tube.outer_radius, tube.k
    assert tube.pressure_at_radius(a) == pytest.approx(k * (1.0 - (a / b) ** 2), rel=1e-14)
    assert tube.pressure_first_yield == pytest.approx(tube.pressure_at_radius(a), rel=1e-14)
    assert tube.pressure_limit == pytest.approx(tube.pressure_at_radius(b), rel=1e-14)
    assert tube.pressure_limit == pytest.approx(2.0 * k * math.log(b / a), rel=1e-14)
    assert tube.plastic_radius(tube.pressure_first_yield) == pytest.approx(a, rel=1e-6)
    assert tube.plastic_radius(tube.pressure_limit) == pytest.approx(b, rel=1e-6)


@pytest.mark.parametrize("ratio", [0.8, 1.0, 1.15, 1.3])
def test_oracle_satisfies_the_physical_identities(tube, ratio):
    pressure = ratio * tube.k
    c = tube.plastic_radius(pressure)
    grid = torch.linspace(tube.inner_radius, tube.outer_radius, 2001, dtype=torch.float64)
    sig_r, sig_t, sig_z, q = tube.stresses(grid, pressure)

    # bore traction and traction-free outer surface
    assert float(sig_r[0]) == pytest.approx(-pressure, rel=1e-3)
    assert abs(float(sig_r[-1])) < 1e-3 * pressure
    # von Mises yield exactly in the plastic zone, strictly below yield outside it
    plastic = grid <= c * (1.0 - 1e-12)
    elastic = grid > c * (1.0 + 1e-9)
    assert float((q[plastic] - tube.yield_stress).abs().max()) < 1e-9 * tube.yield_stress
    assert float(q[elastic].max()) <= tube.yield_stress * (1.0 + 1e-9)
    # plane strain: sigma_z = (sig_r+sig_t)/2 in the plastic zone
    assert float((sig_z[plastic] - 0.5 * (sig_r[plastic] + sig_t[plastic])).abs().max()) \
        < 1e-9 * tube.yield_stress
    # radial equilibrium d sigma_r / d r = (sigma_theta - sigma_r)/r
    dr = float(grid[1] - grid[0])
    slope = (sig_r[2:] - sig_r[:-2]) / (2.0 * dr)
    interior = slice(1, -1)
    residual = slope - (sig_t[interior] - sig_r[interior]) / grid[interior]
    # the field is piecewise smooth; its second derivative is integrable with a jump
    # at r = c, so compare away from the interface and in the relative sense
    away = (grid[interior] < c * 0.98) | (grid[interior] > c * 1.02)
    assert float(residual[away].abs().max()) < 1e-2 * pressure / (tube.outer_radius)
    # hoop and radial stress are continuous at the interface
    left = float(tube.stresses(torch.tensor(c * (1 - 1e-7), dtype=torch.float64), pressure)[1])
    right = float(tube.stresses(torch.tensor(c * (1 + 1e-7), dtype=torch.float64), pressure)[1])
    assert abs(left - right) < 1e-4 * tube.yield_stress


def test_oracle_displacement_is_continuous_and_matches_the_elastic_branch(tube):
    pressure = 0.5 * tube.pressure_first_yield
    grid = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)
    computed = tube.radial_displacement(grid, pressure)
    amp = tube.elastic_amplitude(pressure)
    lame_b = amp * tube.outer_radius ** 2
    analytic = (1.0 + tube.poisson) / tube.young * (
        (1.0 - 2.0 * tube.poisson) * amp * grid + lame_b / grid)
    assert torch.allclose(computed, analytic, rtol=1e-12)
    # elastic-plastic branch: u_r must be continuous at r = c and monotone decreasing
    plastic_pressure = 1.0 * tube.k
    c = tube.plastic_radius(plastic_pressure)
    left = float(tube.radial_displacement(torch.tensor(c * (1 - 1e-6), dtype=torch.float64),
                                          plastic_pressure))
    right = float(tube.radial_displacement(torch.tensor(c * (1 + 1e-6), dtype=torch.float64),
                                           plastic_pressure))
    assert abs(left - right) / right < 1e-5


# --------------------------------------------------------------------------- #
# element check against the elastic Lamé solution
# --------------------------------------------------------------------------- #
def test_axisymmetric_element_reproduces_the_elastic_lame_field(tube, elastic_solution):
    errors = field_errors(elastic_solution, tube, elastic_solution.pressure)
    assert float(elastic_solution.gauss_alpha.abs().max()) == 0.0
    assert errors["u_bore_rel"] <= 5.0e-3
    assert errors["sigma_theta_max_rel"] <= 3.0e-2
    assert errors["sigma_r_max_rel"] <= 5.0e-2
    # plane strain is imposed exactly by the kinematics
    assert float(elastic_solution.displacement[:, 1].abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# elastic-plastic gate
# --------------------------------------------------------------------------- #
def test_plastic_radius_within_three_percent(tube, plastic_solution):
    reference = tube.plastic_radius(plastic_solution.pressure)
    estimate = plastic_solution.plastic_radius(tube.yield_stress)
    assert estimate["plastic_fraction"] > 0.05, "the load must actually yield part of the wall"
    assert abs(estimate["lame_fit"] / reference - 1.0) <= 3.0e-2
    # the model-free bracket estimator must bracket the same answer
    assert abs(estimate["bracket"] / reference - 1.0) <= 3.0e-2


def test_yield_condition_and_incompressibility_hold_at_every_yielded_gauss_point(
        tube, plastic_solution):
    errors = field_errors(plastic_solution, tube, plastic_solution.pressure)
    assert errors["yield_residual"] <= 1e-9
    assert errors["max_q_over_sigma_y_elastic"] <= 1.0 + 1e-9
    assert abs(errors["sigma_r_bore"] + 1.0) <= 3.0e-2      # sigma_r(a) = -p
    assert abs(errors["sigma_r_outer"]) <= 2.0e-2           # sigma_r(b) = 0


def test_stress_and_displacement_fields_are_within_five_percent(tube, plastic_solution):
    errors = field_errors(plastic_solution, tube, plastic_solution.pressure)
    assert errors["u_bore_rel"] <= 1.0e-2
    assert errors["sigma_theta_max_rel"] <= 5.0e-2
    assert errors["sigma_r_max_rel"] <= 5.0e-2


def test_gauss_point_field_error_halves_when_the_mesh_is_refined(tube, law, plastic_solution):
    """The exact field has a derivative jump at r_p, so the Gauss-point stress
    error is first order in the element size; this pins that rate (no stalling)."""
    coarse = solve_plastic_tube(tube, annulus_mesh(12, grading=0.1), law,
                                plastic_solution.pressure, steps=6)
    fine = field_errors(plastic_solution, tube, plastic_solution.pressure)
    rough = field_errors(coarse, tube, coarse.pressure)
    assert rough["sigma_theta_max_rel"] > fine["sigma_theta_max_rel"]
    assert fine["sigma_theta_max_rel"] < 0.5 * rough["sigma_theta_max_rel"]


def test_global_newton_is_quadratically_convergent(plastic_solution):
    history = list(plastic_solution.residual_history)
    assert plastic_solution.iterations <= 12
    assert history[-1] <= 1e-10
    ratios = [history[i + 1] / history[i] for i in range(len(history) - 1)]
    assert min(ratios) < 0.1
    assert ratios[-1] < 1e-2


def test_hardening_shrinks_the_plastic_zone(tube, law, plastic_solution):
    hardening = HardeningLaw(tube.yield_stress, saturation=80.0e6, saturation_rate=40.0)
    pressure = plastic_solution.pressure
    solution = solve_plastic_tube(tube, annulus_mesh(16, grading=0.1), hardening, pressure,
                                  steps=6)
    perfect = plastic_solution.plastic_radius(tube.yield_stress)["lame_fit"]
    hardened = solution.plastic_radius(tube.yield_stress)["lame_fit"]
    assert hardened < perfect
    assert solution.gauss_equivalent_stress[solution.gauss_alpha > 0].min() \
        > tube.yield_stress - 1.0e-6 * tube.yield_stress


def test_perfect_plasticity_is_the_zero_saturation_limit_of_voce(tube, law, plastic_solution):
    nearly_perfect = HardeningLaw(tube.yield_stress, saturation=1.0e-3, saturation_rate=1.0)
    solution = solve_plastic_tube(tube, annulus_mesh(16, grading=0.1), nearly_perfect,
                                  plastic_solution.pressure, steps=6)
    reference = plastic_solution.plastic_radius(tube.yield_stress)["lame_fit"]
    limit = solution.plastic_radius(tube.yield_stress)["lame_fit"]
    assert abs(limit / reference - 1.0) < 1.0e-3


def test_load_history_is_path_independent_below_yield(tube, law):
    pressure = 0.5 * tube.pressure_first_yield
    one = solve_plastic_tube(tube, annulus_mesh(8), law, pressure, steps=1)
    many = solve_plastic_tube(tube, annulus_mesh(8), law, pressure, steps=6)
    assert torch.allclose(one.displacement, many.displacement, rtol=0.0, atol=1e-14)