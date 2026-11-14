"""Tests for the strong added-mass regime of the implicit IBM + Aitken coupling.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  These tests pin the
quantities that ``docs/xval_implicit_ibm_aitken_v4.md`` derives and measures:

1. the interface force of the direct-forcing operator is *exactly linear* in the
   trial displacement at a frozen fluid state, so the interface fixed point has
   a constant gain ``g = -4 S / (4 m_s + k)`` with stencil mass
   ``S = -(dF/dy) beta dt^2 / (2 gamma)``;
2. the measured residual ratio of the sub-iteration loop reproduces ``g``
   (the loop is a scalar linear fixed point);
3. Aitken's Delta^2 recursion recovers the analytic optimum ``omega* = 1/(1-g)``
   from the *first* estimate of every time step -- i.e. the relaxation factor
   that was actually applied is passed into the estimate (regression: with the
   stored, carried-over factor the estimate is scaled by ``omega`` and every
   step costs one extra sub-iteration);
4. below the critical added-mass ratio ``m_a/S`` the iteration converges with
   ``omega = 1`` as well, so Aitken and fixed relaxation must agree there
   (the regime round 3 swept) -- this keeps the strong-coupling claim honest;
5. above it, ``omega = 1`` fails while Aitken stays stable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_added_mass import (
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    coupled_free_vibration,
    plate_markers,
)

BETA, GAMMA = 0.25, 0.5
BOX = 96
C = 10.0


def _gain(half_width=C, box=BOX, eps=1.0e-6, omega_lbm=1.9):
    body = SpringMountedBody(plate_markers((box / 2.0, box / 2.0), half_width),
                             mass=1.0, stiffness=0.0, y0=0.0)
    fluid = LBMFluid2D((box, box), omega=omega_lbm, rho0=1.0)
    solver = IBMImplicitFSI(body, fluid, sub_tol=0.0, max_sub_iter=2)
    rho, u = fluid.macroscopic(include_force=False)
    fp, _ = solver.interface_force(rho, u, eps)
    fm, _ = solver.interface_force(rho, u, -eps)
    dFdy = (fp - fm) / (2.0 * eps)
    return dFdy, -dFdy * BETA / (2.0 * GAMMA)


def _setup(ratio, n_steps=40, sub_tol=1e-6, max_sub_iter=40, period=300.0,
           relax_mode="aitken", fixed_omega=1.0):
    m_a = analytic_added_mass(C)
    m_s = m_a / ratio
    k = (m_s + m_a) * (2.0 * math.pi / period) ** 2
    body = SpringMountedBody(plate_markers((BOX / 2.0, BOX / 2.0), C),
                             mass=m_s, stiffness=k, y0=0.2)
    fluid = LBMFluid2D((BOX, BOX), omega=1.9, rho0=1.0)
    solver = IBMImplicitFSI(body, fluid, relax_mode=relax_mode,
                            fixed_omega=fixed_omega, sub_tol=sub_tol,
                            max_sub_iter=max_sub_iter)
    return body, fluid, solver, m_s, k


# ---------------------------------------------------------------------------
# 1. the interface map is linear with a known constant gain
# ---------------------------------------------------------------------------
def test_interface_force_is_linear_in_trial_displacement():
    body = SpringMountedBody(plate_markers((BOX / 2.0, BOX / 2.0), C),
                             mass=1.0, stiffness=0.0, y0=0.0)
    fluid = LBMFluid2D((BOX, BOX), omega=1.9, rho0=1.0)
    solver = IBMImplicitFSI(body, fluid, sub_tol=0.0, max_sub_iter=2)
    rho, u = fluid.macroscopic(include_force=False)
    f0, _ = solver.interface_force(rho, u, 0.0)
    f1, _ = solver.interface_force(rho, u, 0.01)
    f2, _ = solver.interface_force(rho, u, 0.02)
    step = f1 - f0
    assert abs(step) > 0.0
    # second difference must vanish (exact linearity of direct forcing)
    assert abs((f2 - f1) - step) / abs(step) < 1e-12


def test_stencil_mass_is_linear_in_half_width():
    """S = -dF/dy beta dt^2/(2 gamma) grows like 4c (delta-support footprint)."""
    masses = {}
    for c, box in ((10.0, 96), (16.0, 144)):
        _dFdy, mass = _gain(half_width=c, box=box)
        masses[c] = mass
    assert masses[16.0] > masses[10.0]
    # slope dS/dc ~ 4 (measured 4.0); allow 15 %
    slope = (masses[16.0] - masses[10.0]) / 6.0
    assert 3.4 < slope < 4.6
    # and the critical added-mass ratio therefore scales with c
    ratio_crit_10 = analytic_added_mass(10.0) / masses[10.0]
    ratio_crit_16 = analytic_added_mass(16.0) / masses[16.0]
    assert ratio_crit_16 > ratio_crit_10


# ---------------------------------------------------------------------------
# 2. the sub-iteration loop is the scalar fixed point y -> g y + b
# ---------------------------------------------------------------------------
def test_measured_residual_ratio_matches_predicted_gain():
    _dFdy, stencil = _gain()
    body, _fluid, solver, m_s, k = _setup(ratio=8.0, n_steps=1,
                                          sub_tol=0.0, max_sub_iter=3)
    g_pred = -4.0 * stencil / (4.0 * m_s + k)
    assert g_pred < -1.0            # strong coupling at this ratio
    solver.step()
    # with omega = 1 the first two residuals satisfy r_2 / r_1 = g
    res = solver.residuals
    assert len(res) == 1
    # recover the ratio from the recorded relax factor instead: the second
    # Aitken estimate is 1/(1-g) exactly
    omega = solver.relaxes[-1]
    assert abs(omega - 1.0 / (1.0 - g_pred)) < 1e-3 * abs(1.0 / (1.0 - g_pred))


# ---------------------------------------------------------------------------
# 3. regression: the applied factor must enter the Delta^2 estimate
# ---------------------------------------------------------------------------
def test_aitken_omega_equals_optimum_on_every_step():
    _dFdy, stencil = _gain()
    _b, _f, solver, m_s, k = _setup(ratio=8.0, n_steps=25)
    g_pred = -4.0 * stencil / (4.0 * m_s + k)
    omega_star = 1.0 / (1.0 - g_pred)
    solver.run(25)
    # every recorded Aitken factor (not just the first of a step) is the optimum
    hist = np.asarray(solver.relaxes, dtype=float)
    hist = hist[np.isfinite(hist)]          # NaN = step converged, no relaxation
    assert hist.size > 0
    assert np.max(np.abs(hist - omega_star)) / omega_star < 1e-2
    # and no step needs more than optimum + 2 sub-iterations
    assert max(solver.sub_iterations) <= 4


def test_omega_for_uses_explicit_previous_factor():
    from tensorfem.ibm_aitken_implicit import AitkenRelaxation

    a = AitkenRelaxation(relax_max=1.0, relax0=0.3)
    a.omega = 0.3                                  # stale store
    r_km1 = np.array([1.0])
    r_k = np.array([-1.0])                         # g = -1 -> omega* = 0.5
    # num/den = (1*(-2))/4 = -0.5 -> omega = -base * (-0.5)
    assert abs(a.omega_for(r_km1, r_k, omega_prev=1.0) - 0.5) < 1e-12
    # omitting the argument falls back to the stored (here: wrong) factor
    a2 = AitkenRelaxation(relax_max=1.0, relax0=0.3)
    a2.omega = 0.3
    assert abs(a2.omega_for(r_km1, r_k) - 0.15) < 1e-12


# ---------------------------------------------------------------------------
# 4./5. weak vs strong coupling: what the relaxation buys
# ---------------------------------------------------------------------------
def test_weak_coupling_aitken_and_fixed_agree():
    """Below ratio_crit (round 3's band) the relaxation must not change the answer."""
    kw = dict(n_steps=120, half_width=C, center=(BOX / 2.0, BOX / 2.0),
              lattice=(BOX, BOX), y0=0.2, sub_tol=1e-6, max_sub_iter=40)
    m_a = analytic_added_mass(C)
    m_s = m_a / 0.6
    k = (m_s + m_a) * (2.0 * math.pi / 300.0) ** 2
    a = coupled_free_vibration(mass=m_s, stiffness=k, relax_mode="aitken", **kw)
    f = coupled_free_vibration(mass=m_s, stiffness=k, relax_mode="fixed",
                               fixed_omega=1.0, **kw)
    assert a["steps_completed"] == f["steps_completed"] == 120
    assert np.allclose(a["y_history"], f["y_history"], rtol=1e-10, atol=1e-12)


def test_strong_coupling_fixed_omega_one_fails_aitken_survives():
    _dFdy, _stencil = _gain()
    _b, _f, solver_fixed, m_s, k = _setup(ratio=12.0, n_steps=40,
                                          relax_mode="fixed", fixed_omega=1.0)
    fixed_hist = solver_fixed.run(40)
    assert (not fixed_hist) or any(not h["converged"] for h in fixed_hist)

    body, _fl, solver_aitken, m_s, k = _setup(ratio=12.0, n_steps=40)
    aitken_hist = solver_aitken.run(40)
    assert len(aitken_hist) == 40
    assert not solver_aitken.diverged
    assert all(h["converged"] for h in aitken_hist)