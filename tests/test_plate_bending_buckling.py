"""Tests for linearly varying edge compression of simply supported panels."""
import math

import numpy as np
import pytest

from tensorfem.marine_plate_buckling import (
    isotropic_plate_rigidities,
    navier_uniaxial_buckling_load,
)
from tensorfem.plate_bending_buckling import (
    LITERATURE_K_PURE_BENDING,
    LITERATURE_K_PURE_BENDING_SQUARE,
    LITERATURE_K_TRIANGULAR,
    LITERATURE_K_UNIFORM,
    buckling_coefficient,
    ritz_linear_stress_buckling,
    run_linear_stress_qualification,
    sine_stress_matrices,
    stress_energy_quadrature,
    stress_ratio,
    strip_buckling_coefficient,
)

E, NU, T = 210.0e9, 0.3, 0.012


def _rigidities():
    return isotropic_plate_rigidities(E, NU, T)


def _ritz(aspect, gradient, basis=16):
    d, _, h = _rigidities()
    return ritz_linear_stress_buckling(
        length=aspect,
        width=1.0,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        stress_gradient=gradient,
        basis_x=basis,
        basis_y=basis,
    )


def _strip(aspect, gradient, nodes=200):
    d, _, h = _rigidities()
    return strip_buckling_coefficient(
        length=aspect,
        width=1.0,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        stress_gradient=gradient,
        nodes=nodes,
    )


def test_uniform_limit_reproduces_exact_navier_coefficient():
    """gradient = 0 must return the exact uniform-compression value k = 4.

    ``k = 4`` holds only for integer aspect ratios; between two integers the
    half-wave number cannot adjust continuously, so the minimum rises above 4.
    Both branches are asserted, which pins the half-wave enumeration as well.
    """
    for aspect in (1.0, 2.0, 3.0):
        mode = _ritz(aspect, 0.0)
        assert mode.converged
        assert mode.coefficient == pytest.approx(4.0, rel=1e-10)
    for aspect in (1.5, 2.5, 3.5):
        assert _ritz(aspect, 0.0).coefficient > 4.0


def test_uniform_limit_agrees_with_navier_oracle_module():
    """Independent code path: the Navier oracle of marine_plate_buckling."""
    d, _, h = _rigidities()
    for aspect in (1.0, 2.5, 4.0):
        load, m, n = navier_uniaxial_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h
        )
        mode = _ritz(aspect, 0.0)
        assert mode.multiplier == pytest.approx(load, rel=1e-10)
        assert mode.dominant_mode == (m, n)


def test_closed_form_matrices_match_energy_quadrature():
    """The width moment closed form is pinned against Gauss quadrature."""
    d, _, h = _rigidities()
    for gradient in (0.0, 0.5, 1.0, 1.5, 2.0):
        kb, kg, _ = sine_stress_matrices(
            length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=gradient,
            basis_x=10, basis_y=10,
        )
        qb, qg, _ = stress_energy_quadrature(
            length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=gradient,
            basis_x=10, basis_y=10,
        )
        assert np.allclose(kb, qb, rtol=1e-11, atol=1e-9 * abs(kb).max())
        assert np.allclose(kg, qg, rtol=1e-11, atol=1e-9 * abs(kg).max())


def test_geometric_matrix_is_not_diagonal_for_linear_stress():
    """A sign/parity slip in the width moment would leave the matrix diagonal."""
    d, _, h = _rigidities()
    _, diagonal, _ = sine_stress_matrices(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        stress_gradient=0.0, basis_x=6, basis_y=6,
    )
    _, coupled, _ = sine_stress_matrices(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        stress_gradient=2.0, basis_x=6, basis_y=6,
    )
    off_diagonal = coupled - np.diag(np.diag(coupled))
    assert abs(off_diagonal).max() > 0.0
    assert abs(diagonal - np.diag(np.diag(diagonal))).max() <= 1e-9 * abs(diagonal).max()


@pytest.mark.parametrize(
    "aspect, gradient, literature",
    [
        (2.0, 1.0, LITERATURE_K_TRIANGULAR),
        (6.0, 2.0, LITERATURE_K_PURE_BENDING),
        (1.0, 2.0, LITERATURE_K_PURE_BENDING_SQUARE),
    ],
)
def test_classical_coefficients_within_three_percent(aspect, gradient, literature):
    mode = _ritz(aspect, gradient)
    assert mode.converged
    assert abs(mode.coefficient / literature - 1.0) < 0.03


def test_uniform_classical_coefficient_is_exact():
    assert LITERATURE_K_UNIFORM == 4.0
    assert _ritz(1.0, 0.0).coefficient == pytest.approx(LITERATURE_K_UNIFORM, rel=1e-10)


@pytest.mark.parametrize("aspect", [1.0, 2.0, 6.0])
@pytest.mark.parametrize("gradient", [0.0, 1.0, 2.0])
def test_strip_finite_difference_agrees_with_ritz(aspect, gradient):
    """Second, fully independent discretisation of the same eigenvalue."""
    mode = _ritz(aspect, gradient)
    strip, _ = _strip(aspect, gradient)
    assert abs(strip / mode.coefficient - 1.0) < 1e-3


def test_strip_slope_energy_keeps_second_order_convergence():
    """Regression: omitting the half-weighted boundary slope terms made the
    strip route first order (5e-3 error at 400 nodes) and biased low."""
    reference = _ritz(2.0, 1.0).coefficient
    coarse, _ = _strip(2.0, 1.0, nodes=50)
    fine, _ = _strip(2.0, 1.0, nodes=100)
    error_coarse = abs(coarse / reference - 1.0)
    error_fine = abs(fine / reference - 1.0)
    assert error_fine < error_coarse
    assert error_fine < 5.0e-4
    assert 2.5 < error_coarse / error_fine < 6.0


def test_basis_refinement_converges_from_above():
    """The variational eigenvalue is an upper bound and decreases with basis."""
    coefficients = [_ritz(2.0, 1.0, basis=basis).coefficient for basis in (2, 4, 8, 16)]
    assert all(later <= earlier for earlier, later in zip(coefficients, coefficients[1:]))
    assert abs(coefficients[-1] / coefficients[0] - 1.0) < 0.05
    assert coefficients[-1] == pytest.approx(_ritz(2.0, 1.0, basis=20).coefficient, rel=1e-4)


def test_capacity_increases_with_degree_of_stress_variation():
    """A more concentrated compression must raise the buckling coefficient."""
    coefficients = [_ritz(4.0, gradient).coefficient for gradient in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert all(later > earlier for earlier, later in zip(coefficients, coefficients[1:]))
    # pure bending of a long panel carries several times uniform compression
    assert coefficients[-1] / coefficients[0] > 4.0


def test_mode_shape_respects_neutral_axis():
    """Pure bending must buckle with an even transverse half-wave count.

    ``N_x`` changes sign at mid-width, so the mode has to change sign there; a
    uniform or triangular field, which compresses the full width, buckles with
    ``n = 1``.
    """
    for aspect in (2.0, 6.0):
        assert _ritz(aspect, 0.0).dominant_mode[1] == 1
        assert _ritz(aspect, 1.0).dominant_mode[1] == 1
        assert _ritz(aspect, 2.0).dominant_mode[1] % 2 == 0


def test_stress_ratio_convention():
    assert stress_ratio(0.0) == pytest.approx(1.0)
    assert stress_ratio(1.0) == pytest.approx(0.0)
    assert stress_ratio(2.0) == pytest.approx(-1.0)
    assert _ritz(2.0, 2.0).psi == pytest.approx(-1.0)


def test_buckling_coefficient_roundtrip():
    mode = _ritz(2.0, 1.0)
    d, _, _ = _rigidities()
    assert buckling_coefficient(edge_load=mode.multiplier, rigidity=d, width=1.0) == pytest.approx(
        mode.coefficient, rel=1e-12
    )
    assert mode.multiplier == pytest.approx(
        mode.coefficient * math.pi**2 * d / 1.0**2, rel=1e-12
    )


@pytest.mark.parametrize("gradient", [-0.1, 3.5, math.inf, math.nan])
def test_invalid_gradient_rejected(gradient):
    with pytest.raises(ValueError):
        stress_ratio(gradient)


def test_invalid_geometry_and_sizes_rejected():
    d, _, h = _rigidities()
    with pytest.raises(ValueError):
        sine_stress_matrices(
            length=-1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=1.0,
        )
    with pytest.raises(ValueError):
        sine_stress_matrices(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=1.0, basis_x=1, basis_y=4,
        )
    with pytest.raises(ValueError):
        stress_energy_quadrature(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=1.0, points=8,
        )
    with pytest.raises(ValueError):
        strip_buckling_coefficient(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=1.0, nodes=4,
        )


def test_qualification_report_passes_and_is_fail_closed():
    report = run_linear_stress_qualification()
    assert report["schema"] == "tensorfem.plate-linear-stress-buckling/1.0"
    assert report["passed"] is True
    assert all(row["passed"] for row in report["cases"])
    assert all(row["relative_error"] < 0.03 for row in report["cases"])
    with pytest.raises(AssertionError):
        run_linear_stress_qualification(tolerance=1e-9)


def test_stiffened_panel_is_stronger_but_unverified():
    """Orthotropic smeared stiffness: self-consistency only, no tabulated anchor."""
    d, _, h = _rigidities()
    stiffened = d + E * 3.0e-4 * 0.04**2 / 0.5
    isotropic = ritz_linear_stress_buckling(
        length=4.0, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, stress_gradient=2.0,
    )
    orthotropic = ritz_linear_stress_buckling(
        length=4.0, width=1.0, rigidity_x=stiffened, rigidity_y=d,
        coupling_rigidity=h, stress_gradient=2.0,
    )
    # k is normalised by Dx, so a stiffened panel is compared on critical load.
    assert orthotropic.multiplier > isotropic.multiplier
    report = run_linear_stress_qualification()
    # No literature anchor exists for the orthotropic combination, so it must
    # never appear as a case in the accuracy table.
    assert all(row["literature_k"] is not None for row in report["cases"])