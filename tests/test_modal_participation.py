"""Verification gates for modal participation and effective modal mass.

Reference: closed-form uniform Euler--Bernoulli eigenvectors (no finite element
enters the reference).  Every threshold below is taken from the measured
evidence in ``docs/modal-participation-evidence.json`` with margin; the failing
coarse-mesh cases are asserted as failures, not hidden.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.modal import _CLAMPED_ROOTS, _CANTILEVER_ROOTS, uniform_beam_matrices
from tensorfem.modal_participation import (
    _characteristic_root,
    beam_effective_mass_reference,
    beam_reference_frequencies,
    beam_shape_value,
    modal_participation,
    translation_influence,
)

LENGTH = 1.0
EI = 1.0
MU = 1.0

#: Literature value of the first effective-mass fraction of a uniform cantilever.
LITERATURE_CANTILEVER = (0.6131, 0.1883, 0.0647)


def _cantilever(elements: int, modes: int | None = None):
    K, M = uniform_beam_matrices(LENGTH, elements, EI, MU)
    ndof = 2 * (elements + 1)
    free = [i for i in range(ndof) if i not in (0, 1)]
    influence = translation_influence(ndof, free[0::2], dtype=torch.float64)
    return modal_participation(K, M, influence, (0, 1), modes), K, M, influence, free


# ---------------------------------------------------------------------------
# The closed-form reference itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_roots_match_tabulated_values(mode: int) -> None:
    """The pure-Python root finder reproduces the tabulated Blevins roots."""
    obtained = _characteristic_root("cantilever", mode)
    assert abs(obtained - _CANTILEVER_ROOTS[mode - 1]) / _CANTILEVER_ROOTS[mode - 1] < 1e-14
    obtained_cc = _characteristic_root("clamped_clamped", mode)
    assert abs(obtained_cc - _CLAMPED_ROOTS[mode - 1]) / _CLAMPED_ROOTS[mode - 1] < 1e-14


def test_shape_function_satisfies_boundary_conditions() -> None:
    """Both closed forms are evaluated stably, so the BCs hold to machine precision."""
    h = 1e-7
    lam = _CANTILEVER_ROOTS[0]
    assert abs(beam_shape_value("cantilever", lam, 0.0)) < 1e-15
    assert abs((beam_shape_value("cantilever", lam, h) - beam_shape_value("cantilever", lam, 0.0)) / h) < 1e-6
    # The classical normalisation of the clamped-free first mode gives phi(L) = 2.
    assert abs(beam_shape_value("cantilever", lam, 1.0) - 2.0) < 1e-9
    lam_cc = _CLAMPED_ROOTS[0]
    for xi in (0.0, 1.0):
        assert abs(beam_shape_value("clamped_clamped", lam_cc, xi)) < 1e-9
    slope_0 = (beam_shape_value("clamped_clamped", lam_cc, h) - beam_shape_value("clamped_clamped", lam_cc, 0.0)) / h
    slope_1 = (beam_shape_value("clamped_clamped", lam_cc, 1.0) - beam_shape_value("clamped_clamped", lam_cc, 1.0 - h)) / h
    # Central-difference truncation at h = 1e-7 leaves ~2e-6 on a zero slope.
    assert abs(slope_0) < 1e-4
    assert abs(slope_1) < 1e-4


def test_reference_matches_literature_first_three_modes() -> None:
    """Uniform cantilever effective-mass fractions against the quoted values."""
    fractions = beam_effective_mass_reference("cantilever", 3)
    for obtained, expected in zip(fractions.tolist(), LITERATURE_CANTILEVER):
        assert abs(obtained - expected) / expected < 5e-4


def test_reference_frequencies_match_modal_roots() -> None:
    """Dimensionless closed-form frequencies equal the tabulated ``(beta L)^2``."""
    omega = beam_reference_frequencies("cantilever", 2)
    assert abs(float(omega[0]) / _CANTILEVER_ROOTS[0] ** 2 - 1.0) < 1e-12


def test_antisymmetric_clamped_modes_carry_no_translation_mass() -> None:
    """A uniform translation cannot excite the antisymmetric clamped modes."""
    fractions = beam_effective_mass_reference("clamped_clamped", 9)
    assert float(fractions[1::2].abs().max()) < 1e-12
    assert float(fractions[0::2].sum()) > 0.9


def test_reference_series_converges_to_total_mass() -> None:
    """Ten closed-form modes carry 95.95 % / 92.30 % of the beam mass."""
    assert abs(float(beam_effective_mass_reference("cantilever", 10).sum()) - 0.9595054) < 1e-6
    assert abs(float(beam_effective_mass_reference("clamped_clamped", 10).sum()) - 0.9230332) < 1e-6


# ---------------------------------------------------------------------------
# FE side: identities that must hold exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elements", [8, 16, 32])
def test_completeness_identity_is_exact(elements: int) -> None:
    """Summing over the whole free basis reproduces ``r^T M r`` bit-for-bit."""
    summary, *_ = _cantilever(elements)
    assert summary.mode_count == 2 * elements
    assert abs(float(summary.effective_mass.sum()) - summary.total_modal_mass) / summary.total_modal_mass < 1e-12


@pytest.mark.parametrize("elements", [8, 16])
def test_modes_are_mass_orthonormal(elements: int) -> None:
    summary, *_ = _cantilever(elements)
    assert summary.mass_normalisation_error < 1e-12


def test_discrete_rigid_body_inertia_defect_is_first_order() -> None:
    """``r^T M r`` misses the continuum mass by exactly ``0.6286 / n_el``.

    The defect is the one clamped element: with zero nodal rotations the Hermite
    field of a unit translation is not constant inside that element.  Documented
    as a limitation of the *nodal* influence vector, not of the eigen-solver.
    """
    ratios = []
    for elements in (8, 16, 32, 64, 128):
        summary, *_ = _cantilever(elements)
        ratios.append((1.0 - summary.total_modal_mass / (MU * LENGTH)) * elements)
    assert max(ratios) - min(ratios) < 1e-9
    assert abs(ratios[0] - 0.62857) < 1e-4


# ---------------------------------------------------------------------------
# FE versus the closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elements", [32, 64])
def test_effective_mass_fractions_converge_to_closed_form(elements: int) -> None:
    """First three fractions: 0.094 % / 0.013 % worst error at 32 / 64 elements."""
    summary, *_ = _cantilever(elements)
    physical = summary.fraction_of(MU * LENGTH)
    reference = beam_effective_mass_reference("cantilever", 3)
    error = (physical[:3] - reference).abs() / reference
    assert float(error.max()) < 5e-3 if elements == 32 else float(error.max()) < 5e-4


def test_convergence_is_better_than_second_order() -> None:
    """Halving the element size divides the mode-1 error by at least 4."""
    errors = []
    for elements in (16, 32, 64):
        summary, *_ = _cantilever(elements)
        reference = beam_effective_mass_reference("cantilever", 1)
        errors.append(abs(float(summary.fraction_of(MU * LENGTH)[0]) - float(reference[0])) / float(reference[0]))
    assert errors[0] / errors[1] > 4.0
    assert errors[1] / errors[2] > 4.0


def test_direction_symmetry_kills_antisymmetric_participation() -> None:
    """Clamped--clamped FE model: antisymmetric modes carry 1e-21 of the inertia."""
    elements = 32
    K, M = uniform_beam_matrices(LENGTH, elements, EI, MU)
    ndof = 2 * (elements + 1)
    consolidated = (0, 1, ndof - 2, ndof - 1)
    free = [i for i in range(ndof) if i not in consolidated]
    influence = translation_influence(ndof, free[0::2], dtype=torch.float64)
    summary = modal_participation(K, M, influence, consolidated, 6)
    assert float(summary.effective_mass_fraction[1::2].abs().max()) < 1e-12
    reference = beam_effective_mass_reference("clamped_clamped", 5)
    for mode in (0, 2, 4):
        obtained = float(summary.fraction_of(MU * LENGTH)[mode])
        assert abs(obtained - float(reference[mode])) / float(reference[mode]) < 3e-2


# ---------------------------------------------------------------------------
# Truncation behaviour (why the 90 % criterion must be stated with a convention)
# ---------------------------------------------------------------------------


def test_static_tip_response_truncation_ratios() -> None:
    """One mode reproduces 97.07 % of the static tip deflection, two 99.54 %.

    The Hermite model is nodally exact for a tip load, so the ratios are
    mesh-independent; the modal truncation error is therefore a property of the
    basis, not of the mesh.
    """
    measured = []
    for elements in (8, 64):
        summary, K, M, influence, free = _cantilever(elements)
        F = torch.zeros(2 * (elements + 1), dtype=torch.float64)
        F[2 * elements] = 1.0
        coefficients = summary.modes.T @ F
        exact = LENGTH ** 3 / (3.0 * EI)
        ratios = [
            float(summary.modes[2 * elements, :n] @ (coefficients[:n] / summary.angular_frequencies[:n] ** 2)) / exact
            for n in (1, 2, 3)
        ]
        measured.append(ratios)
    assert abs(measured[0][0] - 0.970692) < 1e-5
    assert abs(measured[0][1] - 0.995412) < 1e-5
    assert abs(measured[0][2] - 0.998568) < 1e-5
    # Measured spread between the 8- and 64-element models is 1.2e-5.
    assert max(abs(a - b) for a, b in zip(measured[0], measured[1])) < 5e-5


def test_mode_count_for_reports_sufficiency_and_overflow() -> None:
    summary, *_ = _cantilever(16, modes=4)
    assert summary.mode_count_for(0.5) == 1
    assert summary.mode_count_for(0.8) <= summary.mode_count_for(0.9) <= summary.mode_count_for(0.95)
    # A truncated basis cannot reach 100 %: the helper says so instead of returning the count.
    assert summary.mode_count_for(1.0) == summary.mode_count + 1
    with pytest.raises(ValueError):
        summary.mode_count_for(0.0)


def test_free_free_polluted_rigid_body_mode_is_detected() -> None:
    """Free--free beam: the rigid-body subspace is round-off dependent.

    At 16 elements a mode of 4.76e-5 Hz survives the positivity filter and
    carries 73.5 % of the inertia instead of the 100 % of a clean translation.
    The diagnostic -- not the eigen-solver -- is what this test pins: a mode of
    negligible frequency must be flagged, and it must not be mistaken for a
    converged participation split.
    """
    elements = 16
    K, M = uniform_beam_matrices(LENGTH, elements, EI, MU)
    ndof = 2 * (elements + 1)
    influence = translation_influence(ndof, list(range(0, ndof, 2)), dtype=torch.float64)
    summary = modal_participation(K, M, influence, (), 4)
    candidates = summary.rigid_body_candidates(1e-3)
    assert candidates == (0,)
    fraction = float(summary.fraction_of(MU * LENGTH)[0])
    assert 0.7 < fraction < 0.8
    # Elastic modes must be unable to carry translational inertia.
    assert float(summary.fraction_of(MU * LENGTH)[1:].abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_influence_on_constrained_dof_is_rejected() -> None:
    K, M = uniform_beam_matrices(LENGTH, 4, EI, MU)
    influence = torch.zeros(2 * 5, dtype=torch.float64)
    influence[0] = 1.0
    with pytest.raises(ValueError, match="constrained"):
        modal_participation(K, M, influence, (0, 1), 2)


def test_invalid_arguments_are_rejected() -> None:
    K, M = uniform_beam_matrices(LENGTH, 4, EI, MU)
    influence = torch.ones(2 * 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        modal_participation(K, M, influence[:-1], (0, 1), 2)
    with pytest.raises(ValueError):
        modal_participation(K, M, influence, (0, 1), 0)
    with pytest.raises(ValueError):
        modal_participation(K, M, influence, (0, 1), 10)
    with pytest.raises(ValueError):
        translation_influence(4, (9,))
    with pytest.raises(ValueError):
        beam_effective_mass_reference("simply_supported", 3)
    with pytest.raises(ValueError):
        beam_shape_value("cantilever", _CANTILEVER_ROOTS[0], 1.5)


def test_angular_frequency_relation_is_consistent() -> None:
    summary, *_ = _cantilever(16)
    expected = summary.frequencies_hz * (2.0 * math.pi)
    assert float((summary.angular_frequencies - expected).abs().max()) < 1e-9 * float(expected.max())