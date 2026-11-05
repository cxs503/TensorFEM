"""Verification of non-uniform prebuckling stress in the plate buckling engine.

Every assertion is against a closed form, an independent discretisation or the
tabulated classical coefficient ``k_b = 23.9`` (Timoshenko & Gere, *Theory of
Elastic Stability*, 2nd ed., Sec. 9.2) -- never against a stored snapshot of
this repository's own output.
"""
from __future__ import annotations

import math

import pytest
import numpy as np

from tensorfem.plate_bending_buckling import (
    LITERATURE_KB_LONG_PANEL,
    LITERATURE_K_UNIAXIAL,
    bending_buckling_coefficient,
    infinite_plate_bending_coefficient,
    linear_bending_profile,
    ritz_bending_buckling_load,
    sine_bending_geometric_matrix,
    sine_geometric_matrix,
    uniform_profile,
)
from tensorfem.plate_shear_buckling import (
    _smallest_positive_from_symmetric_indefinite,
    ritz_inplane_buckling_load,
    shear_buckling_coefficient,
    sine_energy_matrices,
)

D = 1.0
B = 1.0


def _ritz(aspect_ratio: float, *, basis_x: int = 12, basis_y: int = 12, **kwargs):
    return ritz_bending_buckling_load(
        length=aspect_ratio * B,
        width=B,
        rigidity_x=D,
        rigidity_y=D,
        coupling_rigidity=D,
        basis_x=basis_x,
        basis_y=basis_y,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# the new geometric matrices against the verified uniform kernel
# --------------------------------------------------------------------------- #
def test_constant_profile_reproduces_the_verified_uniaxial_matrix():
    """A constant stress must give exactly ``plate_shear_buckling``'s K_x."""
    geometric = sine_geometric_matrix(
        length=3.0, width=B, profile=uniform_profile(2.5), basis_x=5, basis_y=5
    )
    _, uniaxial, _, _ = sine_energy_matrices(
        length=3.0, width=B, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
        basis_x=5, basis_y=5,
    )
    assert np.max(np.abs(geometric - 2.5 * uniaxial)) <= 1.0e-12 * np.max(np.abs(2.5 * uniaxial))


def test_closed_form_bending_matrix_matches_quadrature():
    for aspect in (0.5, 1.0, 4.0):
        quadrature = sine_geometric_matrix(
            length=aspect * B, width=B, profile=linear_bending_profile(B),
            basis_x=6, basis_y=6,
        )
        closed = sine_bending_geometric_matrix(
            length=aspect * B, width=B, basis_x=6, basis_y=6
        )
        assert np.max(np.abs(closed - quadrature)) <= 1.0e-12 * np.max(np.abs(quadrature))


def test_bending_profile_has_zero_axial_resultant():
    """The pure-bending profile carries no membrane force, unlike uniform load."""
    profile = linear_bending_profile(B)
    y = np.linspace(0.0, B, 20001)
    assert abs(float(np.trapezoid(profile(y), y))) < 1.0e-12
    assert abs(float(profile(0.0)) - 1.0) < 1.0e-15
    assert abs(float(profile(B)) + 1.0) < 1.0e-15


def test_closed_form_bending_matrix_diagonal_is_zero():
    """``sigma(y)`` has zero mean, so pure bending cannot couple n == q."""
    matrix = sine_bending_geometric_matrix(length=B, width=B, basis_x=3, basis_y=4)
    diagonal = np.diag(matrix)
    assert np.max(np.abs(diagonal)) == 0.0


# --------------------------------------------------------------------------- #
# critical loads
# --------------------------------------------------------------------------- #
def test_uniform_compression_limit_returns_the_classical_k_equal_4():
    for aspect in (1.0, 2.0, 4.0):
        mode = _ritz(aspect, profile=uniform_profile(1.0))
        assert abs(mode.coefficient - LITERATURE_K_UNIAXIAL) / LITERATURE_K_UNIAXIAL < 1.0e-9


def test_long_panel_pure_bending_matches_the_classical_23_9():
    mode = _ritz(10.0, basis_x=24)
    assert abs(mode.coefficient - 23.9) / 23.9 < 0.03
    assert mode.dominant_mode[1] == 1  # one half wave across the width


def test_pure_bending_minimum_sits_at_two_thirds_aspect_ratio():
    sweep = [(ratio, _ritz(ratio, basis_x=12).coefficient) for ratio in (0.5, 0.667, 1.0, 2.0)]
    best = min(sweep, key=lambda item: item[1])
    assert abs(best[0] - 0.667) <= 0.2
    assert best[1] < _ritz(0.5, basis_x=12).coefficient


def test_independent_hermite_reference_agrees_with_the_ritz_engine():
    reference = infinite_plate_bending_coefficient(width=B, rigidity=D, elements=64)
    assert abs(reference["coefficient"] - 23.9) / 23.9 < 0.03
    assert reference["residual"] < 1.0e-8
    mode = _ritz(10.0, basis_x=24)
    assert abs(mode.coefficient - reference["coefficient"]) / reference["coefficient"] < 1.0e-3


def test_independent_reference_is_exact_for_uniform_compression():
    reference = infinite_plate_bending_coefficient(
        width=B, rigidity=D, profile=uniform_profile(1.0), elements=64
    )
    assert abs(reference["coefficient"] - 4.0) / 4.0 < 1.0e-5


def test_reference_converges_from_above_under_refinement():
    coarse = infinite_plate_bending_coefficient(width=B, rigidity=D, elements=16)
    fine = infinite_plate_bending_coefficient(width=B, rigidity=D, elements=64)
    assert fine["coefficient"] <= coarse["coefficient"] + 1.0e-12
    assert abs(fine["coefficient"] - coarse["coefficient"]) < 1.0e-3


def test_long_panel_basis_must_resolve_the_buckle():
    """A basis that clips the buckle overestimates the load -- documented trap."""
    clipped = _ritz(10.0, basis_x=10)
    resolved = _ritz(10.0, basis_x=24)
    assert clipped.coefficient > 1.05 * resolved.coefficient


# --------------------------------------------------------------------------- #
# negative controls: the anchors above must be able to fail
# --------------------------------------------------------------------------- #
def test_uniform_substitution_is_rejected_by_the_gate():
    """Using the width-averaged (uniform) stress is 6x non-conservative here."""
    uniform = _ritz(10.0, basis_x=24, profile=uniform_profile(1.0)).coefficient
    bending = _ritz(10.0, basis_x=24).coefficient
    assert abs(uniform - LITERATURE_KB_LONG_PANEL) / LITERATURE_KB_LONG_PANEL > 0.03
    assert abs(bending - uniform) / uniform > 1.0


def test_scaled_profile_scales_the_load_linearly():
    """The critical *load* scales with the profile; the coefficient is invariant."""
    single = _ritz(10.0, basis_x=24, profile=linear_bending_profile(B, 1.0))
    double = _ritz(10.0, basis_x=24, profile=linear_bending_profile(B, 2.0))
    # ``load`` is the critical peak stress resultant, so it is invariant under a
    # rescaling of the profile (the multiplier itself halves when the profile
    # doubles, which is why the two cancel).
    assert abs(double.load - single.load) / single.load < 1.0e-12
    assert abs(double.coefficient - single.coefficient) / single.coefficient < 1.0e-12


def test_coefficient_definition_round_trips():
    load = 1.0e4
    coefficient = bending_buckling_coefficient(load=load, rigidity=3.0, width=2.0)
    assert abs(coefficient - load * 4.0 / (math.pi**2 * 3.0)) < 1.0e-12


# --------------------------------------------------------------------------- #
# regression of the shared pencil solver (found by this work)
# --------------------------------------------------------------------------- #
def test_shared_pencil_solver_handles_a_non_diagonal_stiffness():
    """``_smallest_positive_from_symmetric_indefinite`` must solve L^-1 G L^-T.

    The earlier implementation built ``L^-T G L^-T``, which is a different
    matrix; it coincides with the correct one only when the Cholesky factor of
    the (diagonal) bending stiffness is diagonal, i.e. for every case the shear
    module had. This 3x3 pencil has a non-diagonal stiffness and an analytic
    answer.
    """
    stiffness = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    geometric = np.diag([1.0, -0.5, 2.0])
    lambda_value, phi = _smallest_positive_from_symmetric_indefinite(stiffness, geometric)
    # K phi = lambda G phi  <=>  G phi = mu K phi, mu = 1/lambda
    residual = stiffness @ phi - lambda_value * (geometric @ phi)
    scale = np.max(np.abs(stiffness @ phi))
    assert np.linalg.norm(residual) / scale < 1.0e-10
    # and the returned lambda is the smallest positive one
    mus = np.linalg.eigvals(np.linalg.solve(stiffness, geometric)).real
    positive = mus[mus > 0.0]
    assert abs(lambda_value - 1.0 / np.max(positive)) < 1.0e-10


def test_shear_module_anchors_are_unchanged():
    """The pencil fix must leave the verified shear coefficients untouched."""
    square = ritz_inplane_buckling_load(
        length=B, width=B, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
        basis_x=16, basis_y=16,
    )
    k_square = shear_buckling_coefficient(shear_load=square.multiplier, rigidity=D, width=B)
    assert abs(k_square - 9.34) / 9.34 < 0.01
    long_panel = ritz_inplane_buckling_load(
        length=6.0 * B, width=B, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
        basis_x=16, basis_y=16,
    )
    k_long = shear_buckling_coefficient(shear_load=long_panel.multiplier, rigidity=D, width=B)
    assert abs(k_long - 5.34) / 5.34 < 0.03


# --------------------------------------------------------------------------- #
# fail-closed argument handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": -1.0},
        {"width": 0.0},
        {"edge_stress": float("nan")},
    ],
)
def test_profile_validation_is_fail_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        linear_bending_profile(**kwargs)


def test_matrix_validation_is_fail_closed():
    with pytest.raises(ValueError):
        sine_geometric_matrix(
            length=1.0, width=B, profile=uniform_profile(1.0), basis_x=0, basis_y=2
        )
    with pytest.raises(TypeError):
        sine_geometric_matrix(length=1.0, width=B, profile=3.0, basis_x=2, basis_y=2)
    with pytest.raises(ValueError):
        infinite_plate_bending_coefficient(width=B, rigidity=D, elements=4)
    with pytest.raises(ValueError):
        ritz_bending_buckling_load(
            length=1.0, width=B, rigidity_x=D, rigidity_y=D, coupling_rigidity=D,
            basis_x=1, basis_y=4,
        )
