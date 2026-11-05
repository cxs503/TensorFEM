"""Buckling of a simply supported panel under a linearly varying edge stress."""

import math

import numpy as np
import pytest

from tensorfem.marine_plate_buckling import isotropic_plate_rigidities
from tensorfem.plate_bending_buckling import (
    LITERATURE_KB_PURE_BENDING_LONG,
    LITERATURE_KB_PURE_BENDING_SQUARE,
    LITERATURE_K_TRIANGULAR_LONG,
    LITERATURE_K_UNIFORM_LONG,
    bending_buckling_coefficient,
    edge_stress_ratio_profile,
    finite_difference_bending_buckling,
    ritz_stress_gradient_buckling_load,
    run_bending_buckling_qualification,
    stress_gradient_coupling_matrix,
    stress_gradient_matrices,
    stress_gradient_mode_shape,
    stress_gradient_quadrature,
)

E, NU, T = 210.0e9, 0.3, 0.012


def _rigidities():
    return isotropic_plate_rigidities(E, NU, T)


def _coefficient(edge_ratio, aspect, basis=16, rigidities=None):
    d, _, h = rigidities if rigidities is not None else _rigidities()
    mode = ritz_stress_gradient_buckling_load(
        length=aspect,
        width=1.0,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        basis_x=basis,
        basis_y=basis,
        edge_ratio=edge_ratio,
    )
    return bending_buckling_coefficient(critical_load=mode.multiplier, rigidity=d, width=1.0)


# --------------------------------------------------------------------------
# closed-form coupling integral
# --------------------------------------------------------------------------
def test_coupling_matrix_is_symmetric():
    modes = np.arange(1, 9)
    matrix = stress_gradient_coupling_matrix(modes, edge_ratio=-0.3, width=2.5)
    assert np.allclose(matrix, matrix.T, atol=1e-14)


def test_coupling_matrix_uniform_profile_is_half_width_identity():
    """``psi = 1`` must collapse to ``int sin n sin q = b/2 delta`` exactly."""
    modes = np.arange(1, 9)
    matrix = stress_gradient_coupling_matrix(modes, edge_ratio=1.0, width=1.7)
    assert np.allclose(matrix, 0.5 * 1.7 * np.eye(modes.size), atol=1e-14)


def test_pure_bending_diagonal_coupling_vanishes():
    """A pure gradient cannot couple a sine mode to itself.

    This is the reason the classical 23.9 cannot be reproduced by the one-term
    Navier solution used for uniform compression: at ``psi = -1`` the whole
    diagonal of the destabilising energy is zero.
    """
    modes = np.arange(1, 13)
    matrix = stress_gradient_coupling_matrix(modes, edge_ratio=-1.0, width=1.0)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-14)
    assert np.abs(matrix).max() > 0.0


def test_gradient_coupling_is_odd_parity_only():
    """Only ``n + q`` odd couples, so the critical mode mixes both parities."""
    modes = np.arange(1, 13)
    matrix = stress_gradient_coupling_matrix(modes, edge_ratio=-1.0, width=1.0)
    for i, n in enumerate(modes):
        for j, q in enumerate(modes):
            if (n + q) % 2 == 0:
                assert matrix[i, j] == 0.0


def test_coupling_matches_gauss_legendre_integration():
    modes = np.arange(1, 10)
    matrix = stress_gradient_coupling_matrix(modes, edge_ratio=-0.4, width=3.0)
    nodes, weights = np.polynomial.legendre.leggauss(120)
    xi = 0.5 * (nodes + 1.0)
    weight = 0.5 * weights * 3.0
    profile = 1.0 + (-0.4 - 1.0) * xi
    basis = np.sin(np.pi * np.outer(xi, modes))
    reference = basis.T @ ((weight * profile)[:, None] * basis)
    assert np.abs(matrix - reference).max() / np.abs(matrix).max() < 1e-12


# --------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------
def test_bending_matrix_is_diagonal_and_matches_analytic_rigidity():
    d, _, h = _rigidities()
    a, b = 2.0, 1.0
    basis, _, modes = stress_gradient_matrices(
        length=a,
        width=b,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        basis_x=4,
        basis_y=4,
        edge_ratio=-1.0,
    )
    assert np.allclose(basis - np.diag(np.diag(basis)), 0.0)
    for index, (m, n) in enumerate(modes):
        kx, ky = m * math.pi / a, n * math.pi / b
        expected = d * (kx**2 + ky**2) ** 2 * a * b / 4.0
        assert basis[index, index] == pytest.approx(expected, rel=1e-12)


def test_closed_form_matrices_match_quadrature():
    d, _, h = _rigidities()
    closed = stress_gradient_matrices(
        length=3.0,
        width=1.0,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        basis_x=6,
        basis_y=6,
        edge_ratio=-0.7,
    )
    quad = stress_gradient_quadrature(
        length=3.0,
        width=1.0,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        basis_x=6,
        basis_y=6,
        edge_ratio=-0.7,
    )
    for closed_matrix, quad_matrix in zip(closed[:2], quad[:2]):
        scale = np.abs(closed_matrix).max()
        assert np.abs(closed_matrix - quad_matrix).max() / scale < 1e-10


def test_quadrature_rejects_too_few_samples():
    d, _, h = _rigidities()
    with pytest.raises(ValueError):
        stress_gradient_quadrature(
            length=1.0,
            width=1.0,
            rigidity_x=d,
            rigidity_y=d,
            coupling_rigidity=h,
            basis_x=3,
            basis_y=3,
            edge_ratio=-1.0,
            samples=4,
        )


# --------------------------------------------------------------------------
# literature anchors
# --------------------------------------------------------------------------
def test_uniform_limit_recovers_classical_four():
    coefficient = _coefficient(1.0, 6.0)
    assert coefficient == pytest.approx(LITERATURE_K_UNIFORM_LONG, rel=1e-9)


def test_pure_bending_long_panel_matches_literature():
    coefficient = _coefficient(-1.0, 6.0)
    assert abs(coefficient / LITERATURE_KB_PURE_BENDING_LONG - 1.0) < 0.03


def test_pure_bending_square_matches_literature():
    coefficient = _coefficient(-1.0, 1.0)
    assert abs(coefficient / LITERATURE_KB_PURE_BENDING_SQUARE - 1.0) < 0.03


def test_triangular_compression_matches_literature():
    coefficient = _coefficient(0.0, 6.0)
    assert abs(coefficient / LITERATURE_K_TRIANGULAR_LONG - 1.0) < 0.03


def test_coefficient_is_independent_of_youngs_modulus():
    """Isotropic plate buckling is a geometric problem: no material scale in k."""
    steel = isotropic_plate_rigidities(210.0e9, 0.3, T)
    alloy = isotropic_plate_rigidities(70.0e9, 0.33, T)
    steel_k = _coefficient(-1.0, 6.0, rigidities=steel)
    alloy_k = _coefficient(-1.0, 6.0, rigidities=alloy)
    assert abs(steel_k / alloy_k - 1.0) < 0.02


def test_gradient_is_monotone_between_uniform_and_bending():
    samples = [_coefficient(ratio, 6.0) for ratio in (1.0, 0.5, 0.0, -0.5, -1.0)]
    assert all(later > earlier for earlier, later in zip(samples, samples[1:]))


def test_stiffened_panel_carries_a_higher_bending_load():
    """Smeared longitudinal stiffening raises the *load*, not necessarily k.

    ``k`` is normalised by the longitudinal rigidity, which the stiffening also
    raises, so the comparison has to be made on the critical resultant.
    """
    d, _, h = _rigidities()
    stiffened = d + E * 4.0e-4 * 0.05**2 / 0.6

    def load(rigidity_x):
        return ritz_stress_gradient_buckling_load(
            length=2.0,
            width=1.0,
            rigidity_x=rigidity_x,
            rigidity_y=d,
            coupling_rigidity=h,
            basis_x=16,
            basis_y=16,
            edge_ratio=-1.0,
        ).multiplier

    assert load(stiffened) > load(d)


# --------------------------------------------------------------------------
# independent discretisation
# --------------------------------------------------------------------------
def test_finite_difference_uniform_limit_is_classical_four():
    d, _, _ = _rigidities()
    result = finite_difference_bending_buckling(
        length=1.0, width=1.0, rigidity=d, edge_ratio=1.0, divisions=32
    )
    assert abs(result["coefficient"] / LITERATURE_K_UNIFORM_LONG - 1.0) < 0.02


def test_finite_difference_agrees_with_variational_bending_load():
    d, _, _ = _rigidities()
    result = finite_difference_bending_buckling(
        length=1.0, width=1.0, rigidity=d, edge_ratio=-1.0, divisions=32
    )
    reference = _coefficient(-1.0, 1.0)
    assert abs(result["coefficient"] / reference - 1.0) < 0.05


def test_finite_difference_refinement_improves_the_coefficient():
    d, _, _ = _rigidities()
    reference = _coefficient(-1.0, 1.0)
    coarse = finite_difference_bending_buckling(
        length=1.0, width=1.0, rigidity=d, edge_ratio=-1.0, divisions=16
    )
    fine = finite_difference_bending_buckling(
        length=1.0, width=1.0, rigidity=d, edge_ratio=-1.0, divisions=32
    )
    assert abs(fine["coefficient"] - reference) < abs(coarse["coefficient"] - reference)


def test_finite_difference_grid_is_isotropic_in_element_size():
    """Equal node counts on a long panel would under-resolve x; the API must not."""
    d, _, _ = _rigidities()
    result = finite_difference_bending_buckling(
        length=4.0, width=1.0, rigidity=d, edge_ratio=1.0, divisions=12
    )
    # 12 elements across the width and 48 along the length -> 47 x 11 unknowns
    assert result["grid"] == pytest.approx(47.0 * 11.0)


def test_finite_difference_rejects_coarse_grid():
    d, _, _ = _rigidities()
    with pytest.raises(ValueError):
        finite_difference_bending_buckling(
            length=1.0, width=1.0, rigidity=d, edge_ratio=-1.0, divisions=4
        )


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [-1.5, 1.5, float("nan"), float("inf")])
def test_edge_ratio_out_of_range_is_rejected(ratio):
    with pytest.raises(ValueError):
        edge_stress_ratio_profile(ratio)


@pytest.mark.parametrize("ratio", [True, "0.5", None])
def test_edge_ratio_wrong_type_is_rejected(ratio):
    with pytest.raises(TypeError):
        edge_stress_ratio_profile(ratio)


@pytest.mark.parametrize("basis", [0, 1, -3])
def test_basis_too_small_is_rejected(basis):
    d, _, h = _rigidities()
    with pytest.raises(ValueError):
        ritz_stress_gradient_buckling_load(
            length=1.0,
            width=1.0,
            rigidity_x=d,
            rigidity_y=d,
            coupling_rigidity=h,
            basis_x=basis,
            basis_y=4,
            edge_ratio=-1.0,
        )


def test_negative_thickness_is_rejected():
    with pytest.raises(ValueError):
        isotropic_plate_rigidities(E, NU, -0.01)
        ritz_stress_gradient_buckling_load(
            length=1.0,
            width=-1.0,
            rigidity_x=1.0,
            rigidity_y=1.0,
            coupling_rigidity=1.0,
            edge_ratio=-1.0,
        )


def test_coefficient_requires_positive_quantities():
    with pytest.raises(ValueError):
        bending_buckling_coefficient(critical_load=-1.0, rigidity=1.0, width=1.0)


# --------------------------------------------------------------------------
# mode shape and qualification report
# --------------------------------------------------------------------------
def test_mode_shape_is_normalised_and_vanishes_on_the_edges():
    d, _, h = _rigidities()
    field = stress_gradient_mode_shape(
        length=1.0,
        width=1.0,
        rigidity_x=d,
        rigidity_y=d,
        coupling_rigidity=h,
        basis_x=16,
        basis_y=16,
        edge_ratio=-1.0,
        samples=41,
    )
    w = field["w"]
    assert abs(np.abs(w).max() - 1.0) < 1e-12
    assert np.allclose(w[0, :], 0.0, atol=1e-12)
    assert np.allclose(w[-1, :], 0.0, atol=1e-12)
    assert np.allclose(w[:, 0], 0.0, atol=1e-12)
    assert np.allclose(w[:, -1], 0.0, atol=1e-12)
    assert abs(field["coefficient_k"] / LITERATURE_KB_PURE_BENDING_SQUARE - 1.0) < 0.03


def test_qualification_passes_and_reports_every_anchor():
    report = run_bending_buckling_qualification()
    assert report["passed"] is True
    assert report["schema"] == "tensorfem.plate-bending-buckling/1.0"
    names = {case["case"] for case in report["cases"]}
    assert {
        "uniform_limit",
        "pure_bending_long_panel",
        "pure_bending_square",
        "triangular_long_panel",
        "finite_difference_pure_bending",
        "finite_difference_uniform_limit",
        "closed_form_vs_quadrature",
        "basis_convergence_pure_bending",
        "gradient_monotonicity",
    } <= names
    for case in report["cases"]:
        assert case["passed"] is True


def test_qualification_is_fail_closed_for_an_impossible_tolerance():
    with pytest.raises(AssertionError):
        run_bending_buckling_qualification(tolerance=1e-6)


def test_qualification_rejects_an_invalid_tolerance():
    with pytest.raises(ValueError):
        run_bending_buckling_qualification(tolerance=1.5)
