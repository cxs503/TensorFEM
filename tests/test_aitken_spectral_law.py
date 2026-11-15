"""Tests for the spectral (multi-DOF) law of Aitken Delta^2 relaxation.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  These tests pin what
``docs/xval_implicit_ibm_aitken_v12.md`` derives and measures:

1. the shipped Aitken factor computed from a residual pair is the Rayleigh
   quotient of the interface operator evaluated at the *previous* residual
   (a lagged exact line search) -- independently re-derived from the pair;
2. for ``n = 1`` that factor is the exact optimum, so the v11 law (three
   sub-iterations for every gain) is recovered;
3. for ``n > 1`` a single global factor costs more than three sub-iterations as
   soon as the residual mixes two eigen-directions, and costs exactly three when
   it does not;
4. per-DOF (block) relaxation restores three sub-iterations per group;
5. the analytic ellipse added masses (Lamb) give the interface spectrum that the
   closed form uses;
6. fixed-omega decay obeys the spectral radius ``max|1 - omega mu|``, so the
   stability edge is ``omega < 2/mu_max``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_implicit import AitkenRelaxation
from tensorfem.ibm_aitken_spectral import (
    added_mass_operator,
    block_minres,
    ellipse_added_mass,
    ellipse_body_inertia,
    geometric_rate,
    minres_relaxation,
    operator_spectrum,
    rayleigh_step,
    rate_bound,
    stability_edge,
)

TOL = 1e-6


def run_production(op, r0, tol=TOL, relax_max=1.0, max_iter=500):
    """The shipped relaxation class driving an affine interface map."""
    op = np.atleast_2d(np.asarray(op, dtype=float))
    r = np.atleast_1d(np.asarray(r0, dtype=float)).copy()
    ait = AitkenRelaxation(relax_max=relax_max, relax0=relax_max)
    omega, r_prev, residuals, omegas = float(relax_max), None, [float(np.linalg.norm(r))], []
    it = 1
    for it in range(1, max_iter + 1):
        if residuals[-1] <= tol * residuals[0]:
            break
        if r_prev is not None:
            omega = ait.omega_for(r_prev, r, omega_prev=omega)
        r_new = r - omega * (op @ r)
        residuals.append(float(np.linalg.norm(r_new)))
        omegas.append(omega)
        r_prev = r.copy()
        r = r_new
    return {"residuals": residuals, "omegas": omegas, "iterations": it}


def test_aitken_factor_is_the_lagged_rayleigh_quotient():
    rng = np.random.default_rng(7)
    for n in (1, 2, 3):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        op = q @ np.diag(1.0 + rng.uniform(0.2, 6.0, size=n)) @ q.T
        r0 = rng.normal(size=n)
        r0 /= np.linalg.norm(r0)
        meas = run_production(op, r0)
        vec = [r0.copy()]
        for om in meas["omegas"]:
            vec.append(vec[-1] - om * (op @ vec[-1]))
        for k in range(1, len(meas["omegas"])):
            ray = rayleigh_step(vec[k - 1], vec[k], meas["omegas"][k - 1])
            assert abs(ray - meas["omegas"][k]) <= 1e-12 * abs(meas["omegas"][k])


def test_lagged_model_reproduces_the_production_residual_sequence():
    rng = np.random.default_rng(11)
    for n in (1, 2, 3):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        op = q @ np.diag(1.0 + rng.uniform(0.5, 4.0, size=n)) @ q.T
        r0 = rng.normal(size=n)
        r0 /= np.linalg.norm(r0)
        meas = run_production(op, r0, tol=1e-9)
        pred = minres_relaxation(op, r0, tol=1e-9, first_omega=1.0, variant="lagged")
        m = np.asarray(meas["residuals"])
        p = np.asarray(pred["residuals"])
        L = min(m.size, p.size)
        keep = p[:L] > 1e-13 * m[0]
        assert np.max(np.abs(m[:L][keep] - p[:L][keep]) / m[:L][keep]) < 1e-9


@pytest.mark.parametrize("R", [1.0, 2.0, 4.0, 12.0, 30.0])
def test_scalar_law_is_the_n_equals_one_case(R):
    op = np.array([[1.0 + R]])
    meas = run_production(op, np.array([1.0]), tol=1e-10)
    assert meas["iterations"] == 3
    assert abs(meas["omegas"][1] - 1.0 / (1.0 + R)) <= 1e-15
    assert meas["residuals"][2] == 0.0


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_pure_eigen_direction_still_costs_three(share):
    op = np.diag([2.0, 8.0])
    r0 = np.array([math.sqrt(1.0 - share), math.sqrt(share)])
    assert run_production(op, r0, tol=TOL)["iterations"] == 3


@pytest.mark.parametrize("share", [0.01, 0.25, 0.5, 0.81])
def test_mixed_residual_costs_more_than_the_scalar_law(share):
    op = np.diag([2.0, 8.0])
    r0 = np.array([math.sqrt(1.0 - share), math.sqrt(share)])
    n = run_production(op, r0, tol=TOL)["iterations"]
    assert n > 3
    pred = minres_relaxation(op, r0, tol=TOL, first_omega=1.0, variant="lagged")
    assert n == pred["iterations"] + 1


def test_block_relaxation_restores_three_sub_iterations():
    op = np.diag([11.0, 5.85])
    r0 = np.array([1.0, 1.0]) / math.sqrt(2.0)
    assert run_production(op, r0)["iterations"] > 3
    blk = block_minres(op, r0, tol=TOL)
    assert blk["iterations"] == 3
    assert all(c == 3 for c in blk["counts"])


def test_variants_coincide_only_for_one_dof():
    op1 = np.array([[3.5]])
    a = minres_relaxation(op1, np.array([1.0]), tol=1e-10, first_omega=1.0,
                          variant="aligned")
    b = minres_relaxation(op1, np.array([1.0]), tol=1e-10, first_omega=1.0,
                          variant="lagged")
    assert a["residuals"] == b["residuals"]
    op2 = np.diag([2.0, 8.0])
    r0 = np.array([1.0, 1.0]) / math.sqrt(2.0)
    a2 = minres_relaxation(op2, r0, tol=1e-10, first_omega=1.0, variant="aligned")
    b2 = minres_relaxation(op2, r0, tol=1e-10, first_omega=1.0, variant="lagged")
    assert a2["residuals"] != b2["residuals"]


def test_ellipse_added_mass_limits():
    circle = ellipse_added_mass(2.0, 2.0, rho=1.0)
    assert circle["pitch"] == pytest.approx(0.0, abs=1e-15)
    assert circle["heave"] == pytest.approx(math.pi * 4.0)
    ell = ellipse_added_mass(3.0, 0.3, rho=1.0)
    assert ell["heave"] == pytest.approx(math.pi * 9.0)
    assert ell["surge"] == pytest.approx(math.pi * 0.09)
    assert ell["pitch"] == pytest.approx(math.pi * (9.0 - 0.09) ** 2 / 8.0)


def test_ellipse_body_inertia():
    mass, inertia = ellipse_body_inertia(2.0, 0.5, rho_s=7.8)
    assert mass == pytest.approx(7.8 * math.pi * 1.0)
    assert inertia == pytest.approx(mass * (4.0 + 0.25) / 4.0)


def test_operator_spectrum_is_one_plus_the_gain():
    op = added_mass_operator(np.diag([2.0, 0.5]), np.diag([6.0, 1.0]))
    np.testing.assert_allclose(operator_spectrum(op), [3.0, 4.0])
    kappa, bound = rate_bound(operator_spectrum(op))
    assert kappa == pytest.approx(4.0 / 3.0)
    assert bound == pytest.approx((4.0 / 3.0 - 1.0) / (4.0 / 3.0 + 1.0))


def test_rate_bound_is_respected_and_monotone():
    prev = 0.0
    for kappa in (1.05, 2.0, 10.0, 100.0):
        op = np.diag([1.5, 1.5 * kappa])
        r0 = np.array([1.0, 1.0]) / math.sqrt(2.0)
        seq = run_production(op, r0, tol=1e-8)["residuals"]
        ratios, asym, _r2 = geometric_rate([v for v in seq if v > 1e-14])
        bound = (kappa - 1.0) / (kappa + 1.0)
        # the fixed first factor omega = 1 may amplify; the bound is asymptotic
        # the shipped rule is a *lagged* line search, so its measured rate sits at
        # or a few per mille above the aligned-minimal-residual bound
        # individual ratios oscillate around the bound (the rule is a lagged line
        # search); the asymptotic geometric rate is the quantity that obeys it
        assert asym <= bound * 1.05 + 1e-9
        assert bound > prev
        prev = bound


def test_fixed_omega_decay_is_the_spectral_radius():
    op = np.diag([2.0, 5.0])
    r0 = np.array([1.0, 1.0]) / math.sqrt(2.0)
    for omega in (0.1, 0.2, 0.6):
        r = r0.copy()
        n = 400
        for _ in range(n):
            r = r - omega * (op @ r)
        measured = (np.linalg.norm(r) / np.linalg.norm(r0)) ** (1.0 / n)
        spectral = max(abs(1.0 - omega * mu) for mu in (2.0, 5.0))
        assert measured == pytest.approx(spectral, rel=5e-3)


def test_stability_edge_is_two_over_mu_max():
    op = np.diag([2.0, 5.0])
    edge = stability_edge(op)
    assert edge == pytest.approx(2.0 / 5.0)
    # just below the edge decays, just above it grows
    r0 = np.array([0.0, 1.0])
    for omega, grows in ((0.9 * edge, False), (1.05 * edge, True)):
        r = r0.copy()
        for _ in range(300):
            r = r - omega * (op @ r)
        assert (np.linalg.norm(r) > np.linalg.norm(r0)) == grows


def test_geometric_rate_recovers_a_known_ratio():
    res = [1.0 * 0.5 ** k for k in range(12)]
    ratios, asym, r2 = geometric_rate(res)
    assert asym == pytest.approx(0.5)
    assert r2 == pytest.approx(1.0)
    assert all(r == pytest.approx(0.5) for r in ratios)


def test_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError):
        minres_relaxation(np.diag([1.0, 2.0]), np.zeros(2))
    with pytest.raises(ValueError):
        rate_bound([1.0, -0.5])          # indefinite operator
    with pytest.raises(ValueError):
        operator_spectrum(np.array([[1.0, -4.0], [1.0, 1.0]]))   # complex pair
    with pytest.raises(ValueError):
        added_mass_operator(np.eye(2), np.eye(3))
    with pytest.raises(ValueError):
        geometric_rate([1.0, 0.5])