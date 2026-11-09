"""Verification of the exact Timoshenko dynamic stiffness (DSM) module.

The tests are organised as three independent comparison surfaces plus the
counting/robustness properties that the solver relies on:

1. closed form -- the simply supported Timoshenko beam has an exact
   Hu--Timoshenko frequency quadratic, and the Euler--Bernoulli transcendental
   roots are recovered in the slender limit;
2. element/assembly invariants -- symmetry, the static limit against the
   classical ``Phi`` element, ODE residuals of the exact solution basis, and
   independence of the number of DSM elements (no discretisation error);
3. Wittrick--Williams counting -- ``J(omega)`` against the analytic mode count
   of the simply supported beams and against a brute-force determinant scan for
   the free--free and cantilever cases.

A displacement-based finite element cross-check (mesh refinement + Richardson
extrapolation) lives in ``artifacts/m7170_dsm/run_dsm.py`` because it costs a
few seconds; here the converged single-element DSM result is asserted to be
reproduced by 4 elements to 1e-8, which is the same statement at a fraction of
the cost.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.dsbeam import (
    BOUNDARY_CONDITIONS,
    DSMBeamSection,
    clamped_clamped_angular_frequencies,
    clamped_clamped_determinant,
    dynamic_stiffness,
    euler_bernoulli_free_free_angular_frequencies,
    mode_shape,
    natural_frequencies,
    natural_frequencies_hz,
    signature_count,
    simply_supported_timoshenko_angular_frequencies,
    wittrick_williams_count,
)


@pytest.fixture(scope="module")
def steel() -> DSMBeamSection:
    """Deep (``L/h = 25``) steel beam: shear and rotary inertia both matter."""
    return DSMBeamSection.rectangle(
        young=2.06e11, poisson=0.3, density=7850.0, width=0.2, height=0.4, length=10.0
    )


@pytest.fixture(scope="module")
def slender() -> DSMBeamSection:
    """Slender beam (``L/h = 2000``) -- Timoshenko effects below 1e-5 relative."""
    return DSMBeamSection.rectangle(
        young=2.06e11, poisson=0.3, density=7850.0, width=0.005, height=0.005, length=10.0
    )


# ------------------------------------------------------- 1. closed-form oracle
def test_simply_supported_matches_hu_timoshenko_closed_form(steel):
    """Pinned--pinned DSM frequencies equal the analytic Timoshenko values."""
    reference = simply_supported_timoshenko_angular_frequencies(steel, 4)
    computed = natural_frequencies(steel, count=4, n_elements=1, bc="pinned-pinned")
    np.testing.assert_allclose(computed, reference, rtol=1e-10)


def test_simply_supported_reference_agrees_with_repo_reference_module(steel):
    """The local closed form agrees with tensorfem.timoshenko_beam_modal's."""
    from tensorfem.timoshenko_beam_modal import (
        TimoshenkoBeamSection,
        exact_simply_supported_timoshenko_angular_frequencies,
    )

    reference = TimoshenkoBeamSection(
        young=steel.young,
        poisson=steel.poisson,
        density=steel.density,
        length=steel.length,
        width=0.2,
        height=0.4,
    )
    expected = exact_simply_supported_timoshenko_angular_frequencies(reference, 4).numpy()
    np.testing.assert_allclose(
        simply_supported_timoshenko_angular_frequencies(steel, 4), expected, rtol=1e-12
    )


def test_euler_bernoulli_limit_is_recovered_for_a_slender_beam(slender):
    """With shear/rotary inertia negligible the DSM collapses onto Euler-Bernoulli."""
    computed = natural_frequencies(slender, count=4, n_elements=1, bc="free-free")
    reference = euler_bernoulli_free_free_angular_frequencies(slender, 4)
    np.testing.assert_allclose(computed, reference, rtol=1e-4)

    base = math.sqrt(slender.flexural_rigidity / (slender.mass_per_length * slender.length**4))
    cantilever_beta = (1.875104068711961, 4.694091132974174, 7.854757438237613)
    cantilever = natural_frequencies(slender, count=3, n_elements=1, bc="clamped-free")
    np.testing.assert_allclose(cantilever, [b * b * base for b in cantilever_beta], rtol=1e-4)


def test_shear_and_rotary_inertia_lower_the_free_free_frequencies(steel):
    """Timoshenko frequencies sit below Euler--Bernoulli, increasingly with mode."""
    timoshenko = natural_frequencies(steel, count=4, n_elements=1, bc="free-free")
    euler = euler_bernoulli_free_free_angular_frequencies(steel, 4)
    assert np.all(timoshenko < euler)
    deviations = (euler - timoshenko) / euler
    assert np.all(np.diff(deviations) > 0.0)
    assert deviations[0] == pytest.approx(0.0058, abs=2e-3)
    assert deviations[3] == pytest.approx(0.0497, abs=5e-3)


# --------------------------------------------------- 2. element invariants
def test_dynamic_stiffness_is_symmetric_and_real(steel):
    for omega in (1.0, 250.0, 1000.0, 5000.0):
        matrix = dynamic_stiffness(steel, omega)
        assert matrix.shape == (4, 4)
        assert np.isfinite(matrix).all()
        scale = np.abs(matrix).max()
        assert np.abs(matrix - matrix.T).max() <= 1e-12 * scale


def test_static_limit_reproduces_the_classical_timoshenko_element(steel):
    """``K(omega -> 0)`` equals the classical ``Phi`` stiffness matrix."""
    element_length = steel.length
    phi = 12.0 * steel.flexural_rigidity / (steel.shear_rigidity * element_length**2)
    c = steel.flexural_rigidity / (element_length**3 * (1.0 + phi))
    le = element_length
    expected = c * np.array(
        [
            [12.0, 6 * le, -12.0, 6 * le],
            [6 * le, (4 + phi) * le**2, -6 * le, (2 - phi) * le**2],
            [-12.0, -6 * le, 12.0, -6 * le],
            [6 * le, (2 - phi) * le**2, -6 * le, (4 + phi) * le**2],
        ]
    )
    matrix = dynamic_stiffness(steel, 1e-3)
    np.testing.assert_allclose(matrix, expected, rtol=1e-8, atol=1e-3 * np.abs(expected).max())


def test_exact_basis_satisfies_both_timoshenko_equations(steel):
    """The closed-form general solution really solves the Timoshenko ODE system."""
    from tensorfem.dsbeam import _wave_amplitudes

    omega = 317.0
    element_length = 3.3
    lam, rotation, shear, _scale, b_param = _wave_amplitudes(
        omega**2, steel, element_length
    )
    gamma2 = steel.shear_rigidity * element_length**2 / steel.flexural_rigidity
    rotary = omega**2 * steel.rotary_inertia_per_length * element_length**2 / steel.flexural_rigidity
    amplitudes = np.array([1.0, -0.4, 0.7, 0.2])
    for xi in (0.0, 0.31, 0.77, 1.0):
        basis = np.array([math.e ** (lam[k] * xi) for k in range(4)], dtype=complex) * amplitudes
        basis_prime = basis * lam * amplitudes / amplitudes  # d/dxi of exp(lambda xi)
        second = np.array(
            [lam[k] ** 2 * math.e ** (lam[k] * xi) for k in range(4)], dtype=complex
        ) * amplitudes
        w = basis.sum()
        psi = (basis * rotation).sum()
        w2 = second.sum()
        psi2 = (second * rotation).sum()
        w1 = basis_prime.sum()
        psi1 = (basis_prime * rotation).sum()
        # (1) kappa G A (w_xx - psi_x) + rho A omega^2 w = 0  ->  W'' - psi' + B W = 0
        residual_shear = w2 - psi1 + b_param * w
        # (2) EI psi_xx + kappa G A (w_x - psi) + rho I omega^2 psi = 0
        residual_moment = psi2 + gamma2 * (w1 - psi) + rotary * psi
        scale = max(abs(w2), abs(psi2), abs(gamma2 * w1), 1.0)
        assert abs(residual_shear) <= 1e-9 * scale
        assert abs(residual_moment) <= 1e-9 * scale


def test_shear_factor_expression_avoids_euler_bernoulli_indeterminacy():
    """``lambda - rotation`` degenerates in the thin limit, ``-B/lambda`` does not."""
    from tensorfem.dsbeam import _wave_amplitudes

    # Huge shear rigidity and no rotary inertia: the Euler-Bernoulli limit.
    section = DSMBeamSection(
        young=2.06e11,
        poisson=0.3,
        density=7850.0,
        area=0.08,
        inertia=1.0666666666666667e-03,
        length=10.0,
        shear_coefficient=1.0,
        include_rotary_inertia=False,
    )
    gamma2 = section.shear_rigidity * section.length**2 / section.flexural_rigidity
    assert gamma2 > 1e3
    lam, rotation, shear, _, b_param = _wave_amplitudes(500.0**2, section, section.length)
    assert b_param > 0.0
    np.testing.assert_allclose(shear, -b_param / lam, rtol=1e-14)


def test_number_of_elements_does_not_change_the_answer(steel):
    """One exact element equals four: the answer carries no discretisation error."""
    one = natural_frequencies(steel, count=4, n_elements=1, bc="free-free")
    four = natural_frequencies(steel, count=4, n_elements=4, bc="free-free")
    np.testing.assert_allclose(four, one, rtol=1e-8)


def test_rotary_inertia_is_a_physical_effect_not_a_detail():
    """Dropping rotary inertia raises mode 2+ frequencies measurably."""
    with_rotary = DSMBeamSection.rectangle(
        young=2.06e11, poisson=0.3, density=7850.0, width=0.2, height=0.4, length=10.0
    )
    without = DSMBeamSection.rectangle(
        young=2.06e11,
        poisson=0.3,
        density=7850.0,
        width=0.2,
        height=0.4,
        length=10.0,
        include_rotary_inertia=False,
    )
    a = natural_frequencies(with_rotary, count=4, bc="pinned-pinned")
    b = natural_frequencies(without, count=4, bc="pinned-pinned")
    assert np.all(b > a)
    # The analytic reference confirms which one is right.
    np.testing.assert_allclose(
        a, simply_supported_timoshenko_angular_frequencies(with_rotary, 4), rtol=1e-10
    )
    # mode 4 of this deep beam: ~0.9 % for the pinned-pinned case (the analytic
    # reference confirms the rotary-inertia-free model is the wrong one)
    assert (b[3] - a[3]) / a[3] > 0.005


def test_mode_shape_obeys_the_boundary_conditions_and_node_count(steel):
    x, w = mode_shape(steel, 1, bc="pinned-pinned", points_per_element=200)
    assert abs(w[0]) < 1e-8 and abs(w[-1]) < 1e-8
    assert abs(w).max() == pytest.approx(1.0)
    for mode in (1, 2, 3):
        _, shape = mode_shape(steel, mode, bc="pinned-pinned", points_per_element=400)
        interior = shape[1:-1]
        sign_changes = int(np.sum(np.sign(interior[:-1]) * np.sign(interior[1:]) < 0))
        assert sign_changes == mode - 1

    x, cantilever = mode_shape(steel, 1, bc="clamped-free", points_per_element=200)
    assert abs(cantilever[0]) < 1e-8
    # a clamped end starts quadratically: the first point is O(dx^2) of the peak
    assert abs(cantilever[1]) < 1e-4


# ------------------------------------------------ 3. Wittrick--Williams counts
def _analytic_simply_supported_count(section: DSMBeamSection, omega: float) -> int:
    """Analytic mode count for a pinned--pinned beam, both spectra included."""
    rho_a = section.mass_per_length
    rho_i = section.rotary_inertia_per_length
    kga = section.shear_rigidity
    ei = section.flexural_rigidity
    total = 0
    for n in range(1, 400):
        k = n * math.pi / section.length
        a = rho_i * rho_a / kga
        b = rho_a + rho_i * k**2 + rho_a * ei * k**2 / kga
        c = ei * k**4
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:  # pragma: no cover - not reachable for these inputs
            continue
        lower = math.sqrt(2.0 * c / (b + math.sqrt(discriminant)))
        upper = math.sqrt((b + math.sqrt(discriminant)) / (2.0 * a))
        total += int(lower <= omega) + int(upper <= omega)
        if lower > omega and upper > omega:
            break
    return total


@pytest.mark.parametrize("omega", [50.0, 200.0, 500.0, 900.0])
@pytest.mark.parametrize("n_elements", [1, 3])
def test_wittrick_williams_count_matches_the_analytic_mode_count(steel, omega, n_elements):
    counted = wittrick_williams_count(
        steel, omega, n_elements=n_elements, bc="pinned-pinned", omega_ceiling=1200.0
    )
    assert counted == _analytic_simply_supported_count(steel, omega)


def _fem_angular_frequencies(
    section: DSMBeamSection, bc: str, elements: int = 64
) -> np.ndarray:
    """Converged displacement-FEM frequencies (independent of the DSM)."""
    from tensorfem.timoshenko_beam_modal import (
        TimoshenkoBeamSection,
        assemble_timoshenko_beam,
    )

    fem_section = TimoshenkoBeamSection(
        young=section.young,
        poisson=section.poisson,
        density=section.density,
        length=section.length,
        width=0.2,
        height=0.4,
    )
    assert fem_section.area == pytest.approx(section.area)
    assert fem_section.inertia == pytest.approx(section.inertia)
    stiffness, mass = assemble_timoshenko_beam(fem_section, elements)
    k = stiffness.numpy()
    m = mass.numpy()
    ndof = k.shape[0]
    fixed = {
        "free-free": [],
        "pinned-pinned": [0, ndof - 2],
        "clamped-free": [0, 1],
        "clamped-clamped": [0, 1, ndof - 2, ndof - 1],
    }[bc]
    free = [i for i in range(ndof) if i not in fixed]
    k_red = k[np.ix_(free, free)]
    m_red = m[np.ix_(free, free)]
    # Cholesky + triangular solves instead of scipy.linalg.eigh: the LAPACK
    # generalised symmetric driver segfaults on the loong64 development nodes
    # (the same platform trap recorded in the sibling vibration-chain work).
    from scipy.linalg import solve_triangular

    lower = np.linalg.cholesky(m_red)
    standard = solve_triangular(lower, k_red, lower=True)
    standard = solve_triangular(lower, standard.T, lower=True).T
    eigenvalues = np.linalg.eigvalsh(0.5 * (standard + standard.T))
    eigenvalues = eigenvalues[eigenvalues > 1e-9 * max(eigenvalues.max(), 1.0)]
    return np.sqrt(eigenvalues)


@pytest.mark.parametrize("bc", ["free-free", "clamped-free", "clamped-clamped"])
def test_wittrick_williams_count_matches_a_converged_displacement_fem(steel, bc):
    """No root is skipped: J equals the number of FEM eigenvalues below omega."""
    omega = 900.0
    fem = _fem_angular_frequencies(steel, bc)
    fem_count = int(np.sum(fem < omega))
    counted = wittrick_williams_count(
        steel, omega, n_elements=1, bc=bc, omega_ceiling=1200.0
    )
    assert counted == fem_count
    # ... and the DSM frequencies themselves match the FEM ones (FEM converges
    # from above, so the DSM values are the smaller ones)
    dsm = natural_frequencies(steel, count=fem_count, n_elements=1, bc=bc)
    np.testing.assert_allclose(dsm, fem[:fem_count], rtol=2e-3)
    assert np.all(dsm <= fem[:fem_count] * (1.0 + 1e-9))


def test_clamped_clamped_frequencies_are_the_poles_of_the_element(steel):
    """The J0 roots are poles (|K_e| blows up), not zeros, of the DSM."""
    roots = clamped_clamped_angular_frequencies(steel, 900.0, n_elements=1)
    assert roots.size >= 3
    typical = np.linalg.norm(dynamic_stiffness(steel, 100.0))
    for root in roots:
        assert abs(clamped_clamped_determinant(steel, root)) < 1e-6 * abs(
            clamped_clamped_determinant(steel, root * 1.01)
        )
        assert np.linalg.norm(dynamic_stiffness(steel, root)) > 10.0 * typical
        position = int(np.argmin(np.abs(roots - root))) + 1
        computed = natural_frequencies(
            steel, count=position, n_elements=1, bc="clamped-clamped",
            omega_ceiling=float(root) * 1.05,
        )
        assert computed[-1] == pytest.approx(root, rel=1e-6)


def test_counting_stays_valid_through_the_second_spectrum():
    """Above the shear cut-off ``c_coeff -> 0`` the count keeps working.

    The upper branch of the Hu--Timoshenko quadratic is the Timoshenko "second
    spectrum".  Root polishing is deliberately not asserted across the
    degeneracy (the two dispersion roots coalesce at ``c_coeff = 0`` and the
    basis matrix becomes ill-conditioned -- a documented limitation); what is
    asserted is that the Wittrick--Williams count steps exactly at the analytic
    second-spectrum frequency, i.e. no root is lost.
    """
    section = DSMBeamSection.rectangle(
        young=2.06e11, poisson=0.3, density=7850.0, width=1.0, height=1.0, length=5.0
    )
    k = math.pi / section.length
    rho_a = section.mass_per_length
    rho_i = section.rotary_inertia_per_length
    kga = section.shear_rigidity
    ei = section.flexural_rigidity
    a = rho_i * rho_a / kga
    b = rho_a + rho_i * k**2 + rho_a * ei * k**2 / kga
    c = ei * k**4
    upper = math.sqrt((b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a))
    shear_cutoff = math.sqrt(kga / rho_i)
    assert upper > shear_cutoff
    ceiling = 1.06 * upper
    total = _analytic_simply_supported_count(section, ceiling)
    assert _analytic_simply_supported_count(section, 0.97 * upper) == total - 1
    below = wittrick_williams_count(
        section, 0.97 * upper, bc="pinned-pinned", omega_ceiling=ceiling
    )
    above = wittrick_williams_count(
        section, 1.03 * upper, bc="pinned-pinned", omega_ceiling=ceiling
    )
    assert _analytic_simply_supported_count(section, 1.03 * upper) == total
    assert below == total - 1
    assert above == total
    # the second-spectrum eigenvalue itself is recovered to root-finding
    # accuracy -- the branch-free complex-exponential formulation needs no
    # hyperbolic/trigonometric switch
    computed = natural_frequencies(
        section, count=total, n_elements=1, bc="pinned-pinned", omega_ceiling=ceiling
    )
    assert computed.size == total
    assert computed[-1] == pytest.approx(upper, rel=1e-9)
    # the matrix itself stays real and symmetric well above the cut-off
    matrix = dynamic_stiffness(section, 2.0 * shear_cutoff)
    assert np.isfinite(matrix).all()
    assert np.abs(matrix - matrix.T).max() <= 1e-10 * np.abs(matrix).max()


def test_wittrick_williams_count_is_monotone(steel):
    values = [
        wittrick_williams_count(steel, omega, bc="pinned-pinned", omega_ceiling=1200.0)
        for omega in np.linspace(20.0, 1100.0, 40)
    ]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[0] == 0


def test_signature_count_counts_negative_eigenvalues():
    assert signature_count(np.diag([1.0, 1.0, 1.0])) == 0
    assert signature_count(np.diag([-1.0, 1.0, 1.0])) == 1
    assert signature_count(np.diag([-1.0, -1.0, 1.0])) == 2
    assert signature_count(np.zeros((0, 0))) == 0


def test_frequencies_in_hz_are_the_radian_values_over_two_pi(steel):
    rad = natural_frequencies(steel, count=3, bc="pinned-pinned")
    hz = natural_frequencies_hz(steel, count=3, bc="pinned-pinned")
    np.testing.assert_allclose(hz, rad / (2.0 * math.pi), rtol=1e-14)


# ------------------------------------------------------------------ validation
def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        DSMBeamSection.rectangle(
            young=2.06e11, poisson=0.3, density=7850.0, width=-1.0, height=0.4, length=10.0
        )
    with pytest.raises(ValueError):
        DSMBeamSection(
            young=2.06e11, poisson=0.6, density=7850.0, area=0.1, inertia=1e-3, length=10.0
        )
    section = DSMBeamSection.rectangle(
        young=2.06e11, poisson=0.3, density=7850.0, width=0.2, height=0.4, length=10.0
    )
    with pytest.raises(ValueError):
        dynamic_stiffness(section, 0.0)
    with pytest.raises(ValueError):
        natural_frequencies(section, count=0)
    with pytest.raises(ValueError):
        natural_frequencies(section, count=2, n_elements=0)
    with pytest.raises(ValueError):
        natural_frequencies(section, count=2, bc="simply-supported")
    with pytest.raises(ValueError):
        wittrick_williams_count(section, 500.0, omega_ceiling=100.0)
    with pytest.raises(ValueError):
        mode_shape(section, 0)
    assert set(BOUNDARY_CONDITIONS) == {
        "free-free",
        "pinned-pinned",
        "clamped-clamped",
        "clamped-free",
    }


def test_omega_ceiling_shortfall_is_reported(steel):
    with pytest.raises(RuntimeError):
        natural_frequencies(
            steel, count=4, n_elements=1, bc="pinned-pinned", omega_ceiling=100.0
        )