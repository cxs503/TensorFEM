"""Validity domain of the closed-form Aitken cost law (v11) beyond one DOF.

The v11 cross-validation derived, for a *scalar* affine interface map
``y -> g y + b`` with ``R := -g``, the exact statements

    omega* = 1/(1+R),   q = 1 - omega (1+R),   n(tol) = 1 + ceil(ln(tol/|r1|)/ln|q|)

and measured them on the anchor's spring-mounted plate -- a body with **one**
transverse degree of freedom, whose interface map is scalar *by construction*
(``tensorfem.ibm_aitken_added_mass.SpringMountedBody``).  Whether the same law
describes the production *flexural* solver
(``tensorfem.ibm_aitken_implicit.ImplicitIBMFSI`` + ``CorotationalBeam2D``,
30-48 free dof) was not measured: there the interface map is a matrix.

This module supplies the missing object.  With a diagonalisable gain matrix whose
eigenvalues are ``g_i = -(1 + R_i)`` and Aitken's estimate taken from the
residual two-step ratio (which sees the dominant mode ``R_max``), the update
factor is ``omega = 1/(1+R_max)`` and the per-mode residual ratio is

    q_i = 1 - omega (1 + R_i) = (R_max - R_i) / (1 + R_max)                  (1)

so mode ``R_max`` is annihilated exactly (``q = 0``, as in the scalar law) while
every other mode decays geometrically at ``|q_i| < 1``.  The loop is therefore
*unconditionally stable* for ``R_i >= 0`` but **finitely terminating only if the
spectrum is a single point**; the cost is governed by the slowest surviving mode

    n(tol) = 1 + ceil( ln(tol/|r1|) / ln|q_slow| ),  q_slow = (R_max-R_min)/(1+R_max)   (2)

Equation (2) reduces to the v11 "three sub-iterations" law exactly when
``R_min = R_max`` -- i.e. the v11 law is the rank-one special case of (2), and its
generality depends on a property of the *testbed*, not of the algorithm.

Public API: :func:`multimode_loop`, :func:`closed_form_count`,
:func:`scalar_count`, :func:`gain_from_count`, :func:`submerged_flexural_beam`.
"""

from __future__ import annotations

import math

import numpy as np

from .ibm_aitken_implicit import (
    AitkenRelaxation,
    BeamProperties,
    CorotationalBeam2D,
    ImplicitIBMFSI,
    LBMFluid2D,
)

__all__ = [
    "multimode_loop",
    "closed_form_count",
    "scalar_count",
    "gain_from_count",
    "submerged_flexural_beam",
]


def scalar_count(r1, q, tol):
    """v11 scalar cost law ``n = 1 + ceil(ln(tol/|r1|)/ln|q|)`` (``inf`` if |q|>=1)."""
    if abs(r1) <= tol:
        return 1
    if abs(q) == 0.0:
        return 2
    if abs(q) >= 1.0:
        return math.inf
    return max(1, 1 + math.ceil(math.log(tol / abs(r1)) / math.log(abs(q)) - 1e-12))


def closed_form_count(R_min, R_max, r1, tol):
    """Multi-mode cost law (2); equals :func:`scalar_count` when ``R_min == R_max``."""
    q = (R_max - R_min) / (1.0 + R_max)
    return scalar_count(r1, q, tol)


def multimode_loop(R, tol=1e-8, mode="aitken", omega=1.0, x0=None, maxit=400,
                   rot=None, relax_max=1.0):
    """Drive the **production** ``AitkenRelaxation`` on a multi-mode affine map.

    ``R`` is the vector of added-mass ratios of the interface map,
    ``G = Q diag(-R) Q^T`` with ``Q`` a fixed orthogonal matrix (``rot``, to
    keep runs comparable).  ``x0`` is the trial state at the start of the step;
    the residual is ``r = G x - x + c`` with the fixed point at ``x* = 1``.

    Returns ``dict(n, residual_norms, omegas)``; ``n`` counts the converged
    residual evaluation, matching ``ImplicitIBMFSI.sub_iters``.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    m = R.size
    if rot is None:
        # deterministic orthogonal basis, independent of R
        q, _ = np.linalg.qr(np.arange(1.0, m * m + 1.0).reshape(m, m)
                            + np.eye(m) * m)
        rot = q
    g = -R                      # v11 convention: g = -R, so q = 1 - omega (1+R)
    G = rot @ np.diag(g) @ rot.T
    x_star = np.ones(m)
    c = x_star - G @ x_star                 # fixed point at x* = 1 by construction
    # start away from the fixed point, as the solver does at a new time step
    x = np.zeros(m) if x0 is None else np.asarray(x0, dtype=float).copy()
    ait = AitkenRelaxation(relax_max=relax_max, relax0=omega)
    r_prev, om_applied, hist, norms = None, float("nan"), [], []
    om = omega if mode == "fixed" else relax_max
    for it in range(1, maxit + 1):
        x_new = G @ x + c
        r = x_new - x
        norms.append(float(np.linalg.norm(r)))
        if norms[-1] <= tol:
            return {"n": it, "residual_norms": norms, "omegas": hist,
                    "final_err": float(np.abs(x_new - x_star).max())}
        if mode == "aitken" and it >= 2:
            om = ait.omega_for(r_prev, r, omega_prev=om_applied)
        else:
            om = relax_max if mode == "aitken" else omega
        om_applied = om
        hist.append(float(om))
        r_prev = r.copy()
        x = x + om * (x_new - x)
    return {"n": maxit, "residual_norms": norms, "omegas": hist,
            "final_err": float(np.abs(x - x_star).max())}


def gain_from_count(n, omega, r1, tol, q_sign=-1.0):
    """Interface gain ``R`` implied by a measured sub-iteration count (inverse of the
    scalar law).  Returns ``nan`` when the count carries no information."""
    if not np.isfinite(n) or n <= 3 or abs(r1) <= 0.0:
        # n <= 3 means |r| fell below tol in at most two relaxations: the
        # inversion ``R = (1-q)/omega - 1`` is then ill-conditioned (q is
        # numerically zero), so the count carries no information about R.
        return float("nan")
    q_abs = abs(tol / abs(r1)) ** (1.0 / (n - 1))
    # a count fixes |q| = |1 - omega(1+R)| only; the sign is a second unknown.
    # ``q_sign`` selects the branch: q < 0 is the over-relaxed/added-mass branch
    # (omega(1+R) > 1), q > 0 the under-relaxed one.  Reported values are only
    # meaningful together with the branch used.
    return ((1.0 - q_sign * q_abs) / omega - 1.0)


def submerged_flexural_beam(n_elements=10, amp=0.05, fluid_rho=1.0,
                            lattice=(64, 64), dt=0.1, sub_tol=1e-8,
                            max_sub_iter=200, relax_mode="aitken",
                            fixed_omega=1.0):
    """Flexural (multi-dof) beam released in an at-rest D2Q9 box.

    The secondary submerged configuration of the v5/v6 rounds (``fluid_rho``,
    ``dt`` scaled for a *thick* section), promoted here to the strong-coupling
    regime by ``fluid_rho = 1`` and ``dt = 0.1``.  Returns ``(beam, fluid,
    solver)``; ``solver.sub_iters`` is filled by :meth:`ImplicitIBMFSI.step`.
    """
    beam = CorotationalBeam2D(n_elements=n_elements, props=BeamProperties(
        E=1.0e5, A=1.0, I=1.0 / 12.0, rho=1.0, length=1.0, thickness=1.0))
    beam.set_initial_deflection(amp)
    scale = 0.45 * float(lattice[0])
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amp * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0,
                            ref_length=beam.props.length, sub_tol=sub_tol,
                            max_sub_iter=max_sub_iter, relax_mode=relax_mode,
                            fixed_omega=fixed_omega)
    return beam, fluid, solver


def beam_step_count(mode="aitken", omega=1.0, n_warm=0, **kw):
    """Sub-iteration count of one production solver step, through the public API.

    ``n_warm`` steps are run first (Aitken mode) so the factor history is the one
    a running coupled simulation would have; the measured step is then forced
    into the requested relaxation mode.
    """
    beam, fluid, solver = submerged_flexural_beam(**kw)
    if n_warm:
        solver.run(n_warm)
    solver.relax_mode = mode
    solver.fixed_omega = omega
    rec = solver.step(n_warm + 1)
    return rec