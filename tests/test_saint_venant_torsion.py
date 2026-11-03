"""Verification of the Saint-Venant torsion path against a closed-form series.

Every assertion is against (a) a literature coefficient, (b) an independent
numerical identity of the closed form (PDE residual, boundary condition, the
definition ``J = 2 int psi dA``, the thin-strip limit), or (c) the closed form
itself for the finite-element solution.  Nothing compares against a snapshot of
this repository's own output.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.torsion_bar import (
    max_shear_stress,
    prandtl_stress_function,
    rectangular_torsion_constant,
    solve_torsion_bar,
    torsion_shear_stress,
    torsion_warping_function,
)

SIDE = 1.0
LENGTH = 4.0
YOUNG = 1.0e9
POISSON = 0.3
TORQUE = 1.0
GATE = 0.03


@pytest.fixture(scope="module")
def hex8_two():
    return solve_torsion_bar(element="hex8", segments=2, side=SIDE, length=LENGTH,
                             young=YOUNG, poisson=POISSON, torque=TORQUE)


@pytest.fixture(scope="module")
def hex8_four():
    return solve_torsion_bar(element="hex8", segments=4, side=SIDE, length=LENGTH,
                             young=YOUNG, poisson=POISSON, torque=TORQUE)


@pytest.fixture(scope="module")
def hex20_one():
    return solve_torsion_bar(element="hex20", segments=1, side=SIDE, length=LENGTH,
                             young=YOUNG, poisson=POISSON, torque=TORQUE)


@pytest.fixture(scope="module")
def hex20_two():
    return solve_torsion_bar(element="hex20", segments=2, side=SIDE, length=LENGTH,
                             young=YOUNG, poisson=POISSON, torque=TORQUE)


@pytest.fixture(scope="module")
def hex20_three():
    return solve_torsion_bar(element="hex20", segments=3, side=SIDE, length=LENGTH,
                             young=YOUNG, poisson=POISSON, torque=TORQUE)


# --------------------------------------------------------------- reference side
def test_square_torsion_constant_matches_table():
    """Timoshenko & Goodier Sec. 117: J = beta a b^3 with beta = 0.141."""
    beta = rectangular_torsion_constant(SIDE, SIDE) / SIDE**4
    assert beta == pytest.approx(0.141, rel=1e-2)
    assert beta == pytest.approx(0.1406, rel=1e-3)


def test_rectangle_torsion_constant_matches_table():
    beta = rectangular_torsion_constant(2.0 * SIDE, SIDE) / (2.0 * SIDE * SIDE**3)
    assert beta == pytest.approx(0.229, rel=1e-2)


def test_peak_shear_coefficient_matches_table():
    """Timoshenko & Goodier Sec. 117: tau_max = T/(alpha a b^2), alpha = 0.208."""
    peak, location = max_shear_stress(SIDE, SIDE)
    alpha = TORQUE / (peak * SIDE**3)
    assert alpha == pytest.approx(0.208, rel=1e-2)
    # the peak sits on a long side, at its midpoint, never in the interior
    assert abs(abs(location[0]) - SIDE / 2) < 0.01 * SIDE
    assert abs(location[1]) < 0.01 * SIDE


def test_thin_strip_limit():
    """a/b -> infinity: tau_max a b^2 / T -> 3 from J -> a b^3/3."""
    peak, _ = max_shear_stress(50.0 * SIDE, SIDE, samples=401, terms=60)
    assert peak * 50.0 * SIDE * SIDE**2 / TORQUE == pytest.approx(3.0, rel=2e-2)


def test_series_is_converged():
    coarse = rectangular_torsion_constant(SIDE, SIDE, terms=100)
    fine = rectangular_torsion_constant(SIDE, SIDE, terms=1000)
    assert abs(coarse - fine) / fine < 1e-9
    slender_coarse = rectangular_torsion_constant(50.0 * SIDE, SIDE, terms=200)
    slender_fine = rectangular_torsion_constant(50.0 * SIDE, SIDE, terms=1000)
    assert abs(slender_coarse - slender_fine) / slender_fine < 1e-9


def test_prandtl_stress_function_satisfies_governing_equation():
    """lap(psi) = -2 inside the section (second-order central differences)."""
    grid = torch.linspace(-0.4 * SIDE, 0.4 * SIDE, 121, dtype=torch.float64)
    X, Y = torch.meshgrid(grid, grid, indexing="xy")
    psi = prandtl_stress_function(X.reshape(-1), Y.reshape(-1), SIDE, SIDE).reshape(121, 121)
    h = float(grid[1] - grid[0])
    laplace = (psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2] - 4 * psi[1:-1, 1:-1]) / h**2
    assert float((laplace + 2.0).abs().max()) < 5e-3


def test_warping_function_is_harmonic():
    grid = torch.linspace(-0.4 * SIDE, 0.4 * SIDE, 121, dtype=torch.float64)
    X, Y = torch.meshgrid(grid, grid, indexing="xy")
    w = torsion_warping_function(X.reshape(-1), Y.reshape(-1), SIDE, SIDE).reshape(121, 121)
    h = float(grid[1] - grid[0])
    laplace = (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * w[1:-1, 1:-1]) / h**2
    assert float(laplace.abs().max()) < 1e-2


def test_stress_function_vanishes_on_the_boundary():
    """The lateral surface is traction free: psi = 0 there (up to round-off)."""
    line = torch.linspace(-SIDE / 2, SIDE / 2, 101, dtype=torch.float64)
    walls = torch.full_like(line, SIDE / 2)
    for x, y in ((walls, line), (-walls, line), (line, walls), (line, -walls)):
        assert float(prandtl_stress_function(x, y, SIDE, SIDE).abs().max()) < 1e-6


def test_torsion_constant_equals_twice_integral_of_stress_function():
    fine = torch.linspace(-SIDE / 2, SIDE / 2, 301, dtype=torch.float64)
    X, Y = torch.meshgrid(fine, fine, indexing="xy")
    psi = prandtl_stress_function(X.reshape(-1), Y.reshape(-1), SIDE, SIDE).reshape(301, 301)
    quadrature = 2.0 * float(torch.trapezoid(torch.trapezoid(psi, fine, dim=0), fine, dim=0))
    assert quadrature == pytest.approx(rectangular_torsion_constant(SIDE, SIDE), rel=1e-4)


def test_shear_stress_is_tangential_on_the_boundary():
    """On the stress-free face x = b/2 the normal component tau_zx must vanish."""
    ys = torch.linspace(-0.49 * SIDE, 0.49 * SIDE, 41, dtype=torch.float64)
    xs = torch.full_like(ys, 0.999 * SIDE / 2)
    tx, ty = torsion_shear_stress(xs, ys, SIDE, SIDE)
    assert float(tx.abs().max()) < 0.05 * float(torch.sqrt(tx**2 + ty**2).max())


def test_series_survives_slender_sections():
    """The cosh ratio must not overflow for a/b of 50 (naive cosh does)."""
    xs = torch.tensor([0.4, 0.2, 0.0], dtype=torch.float64)
    ys = torch.zeros(3, dtype=torch.float64)
    tx, ty = torsion_shear_stress(xs, ys, 50.0 * SIDE, SIDE, terms=60)
    assert torch.isfinite(tx).all() and torch.isfinite(ty).all()


# ------------------------------------------------------------ finite-element side
def test_applied_torque_is_exact(hex8_two, hex20_two):
    for result in (hex8_two, hex20_two):
        assert result.applied_torque == pytest.approx(TORQUE, rel=1e-12)


def test_quadratic_element_passes_the_gate(hex20_two, hex20_three):
    assert hex20_two.twist_rate_relative_error <= GATE
    assert hex20_three.twist_rate_relative_error <= GATE
    assert hex20_three.twist_rate_relative_error < hex20_two.twist_rate_relative_error


def test_measured_torsion_constant(hex20_three):
    assert hex20_three.measured_torsion_constant == pytest.approx(
        rectangular_torsion_constant(SIDE, SIDE), rel=GATE)


def test_hex20_beats_hex8_at_the_same_refinement(hex8_two, hex20_two):
    """Same element count per side, same loading: the quadratic path must win."""
    assert hex8_two.n_elements == 32 and hex20_two.n_elements == 32
    assert hex20_two.twist_rate_relative_error < 0.2 * hex8_two.twist_rate_relative_error


def test_convergence_is_monotone(hex8_two, hex8_four, hex20_one, hex20_two):
    assert hex8_four.twist_rate_relative_error < hex8_two.twist_rate_relative_error
    assert hex20_two.twist_rate_relative_error < hex20_one.twist_rate_relative_error
    assert hex8_four.twist_rate_relative_error < GATE * 2.0


def test_twist_profile_is_linear_inside_the_window(hex8_four):
    """Outside the end layers the section rotation is exactly a straight line."""
    z = hex8_four.profile_z
    theta = hex8_four.profile_theta
    inside = (z >= 0.25 * LENGTH) & (z <= 0.75 * LENGTH)
    design = torch.stack((z[inside], torch.ones(int(inside.sum()), dtype=z.dtype)), 1)
    coefficients = torch.linalg.lstsq(design, theta[inside]).solution
    residual = theta[inside] - design @ coefficients
    span = float(theta[inside].abs().max() - theta[inside].abs().min())
    assert float(residual.abs().max()) < 2e-3 * span
    assert float(coefficients[0]) == pytest.approx(hex8_four.twist_rate, rel=1e-10)


def test_warping_shape_is_recovered(hex20_three):
    assert hex20_three.warping_relative_error < GATE


def test_shear_field_inside_the_section(hex20_three):
    assert math.isfinite(hex20_three.shear_l2_relative_error)
    assert hex20_three.shear_l2_relative_error < 0.10
    # centroid sampling can never reach the true boundary peak
    assert hex20_three.shear_peak_fe < hex20_three.shear_peak_exact_global


def test_face_weights_reproduce_linear_tractions():
    """The consistent end-face weights carry exactly the prescribed torque."""
    for element in ("hex8", "hex20"):
        result = solve_torsion_bar(element=element, segments=2, side=SIDE, length=LENGTH,
                                   young=YOUNG, poisson=POISSON, torque=3.5)
        assert result.applied_torque == pytest.approx(3.5, rel=1e-12)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        solve_torsion_bar(element="tet4")
    with pytest.raises(ValueError):
        solve_torsion_bar(segments=0)
    with pytest.raises(ValueError):
        solve_torsion_bar(window=(0.75, 0.25))
    with pytest.raises(ValueError):
        rectangular_torsion_constant(0.5, 1.0)  # sides passed the wrong way round
    with pytest.raises(ValueError):
        max_shear_stress(SIDE, SIDE, samples=3)