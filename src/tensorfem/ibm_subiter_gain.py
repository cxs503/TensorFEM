"""Interface-gain diagnostics for sub-iteration FSI drivers (implicit IBM family).

Why this module exists
----------------------
The anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` claims, following the
joconnor22/LIFE recipe, that the per-time-step fixed point of the immersed
boundary coupling *needs* Aitken ``Delta^2`` under-relaxation.  That claim is
only meaningful if the interface iteration actually has a gain different from
one: for a map ``u -> G(u)`` iterated with relaxation ``omega``, the residual
obeys ``r_k = (1 - omega * mu) r_{k-1}`` with ``mu`` the interface gain
(``mu = 1 + m_a/m_s`` for an added-mass dominated interface, Causin-Gerbeau-
Nobile 2005).  When ``mu -> 1`` the relaxation factor is inert -- ``omega = 1``
converges as fast as any under-relaxed choice -- and a "converged with Aitken"
record certifies nothing about the coupling.

This module supplies the three measurements that separate those cases:

:func:`gain_from_trace`
    ``mu`` from a recorded sub-iteration residual sequence, at one or two fixed
    relaxation factors.  Two factors make the estimate independent of the map's
    sign convention: ``r_1/r_0 = |1 - omega mu|`` for both ``omega`` values.
:func:`marker_local_mass`
    the fluid mass a *single* sub-iteration can actually feel,
    ``M_f = sum_i rho_i eps_i ds_i``: the interface force enters the momentum
    equation locally (spread + interpolate are exact inverses once epsilon
    conservation is on), so the iterate-dependent part of the load is
    ``M_f * du_body``.  For an incompressible body the potential-flow added mass
    ``m_a`` is *many times* larger (it is built up by the pressure field over
    many time steps), which is exactly why an IBM sub-iteration converges in one
    or two sweeps even at ``m_a/m_s >> 1``.
:func:`discrete_gain_bound`
    the gain implied by the Newmark-consistent structural response,
    ``|dG/du| = gamma * dt * M_f / (M_s + k beta dt^2)`` (sign: the interface
    force opposes the motion), i.e. the upper bound on ``mu - 1`` for any
    velocity-consistent IBM sub-iteration that keeps the fluid field frozen.
:func:`added_mass_ratio`
    ``m_a/m_s`` from an independent reference (Lamb 1932 plate:
    ``m_a = rho pi c^2`` per unit length, ``m_s`` = structural mass per unit
    length) -- what the gain *would* be if the interface were added-mass limited.

Public API: :func:`gain_from_trace`, :func:`marker_local_mass`,
:func:`discrete_gain_bound`, :func:`added_mass_ratio`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "gain_from_trace",
    "marker_local_mass",
    "discrete_gain_bound",
    "added_mass_ratio",
    "INERT_TOL",
]

#: ``mu - 1`` below this is treated as "relaxation inert" (the trace's residual
#: ratio is at round-off / re-stencilling level, not a coupling gain).
INERT_TOL = 1.0e-6


def gain_from_trace(trace, omegas=(1.0, 0.5, 0.25), index=0):
    """Interface gain ``mu`` from sub-iteration residual traces.

    ``trace`` holds one residual sequence per fixed relaxation factor in
    ``omegas`` (no relaxation is applied *inside* the sequence: the driver is run
    with ``relax_mode="fixed", fixed_omega=omega``).  For a map whose residual
    behaves as ``r_k = (1 - omega mu) r_{k-1}`` the measured ratio is
    ``rho(omega) = |1 - omega mu|``; ``mu`` is the value that fits *all* the
    measured ratios, which removes the sign ambiguity of a single-``omega``
    estimate (``rho = |1 - mu|`` cannot tell ``mu = 1 + rho`` from ``1 - rho``).

    Returns ``{"mu": float, "ratios": [...], "predicted": [...], "fit_error":
    float, "method": ...}``.  ``fit_error`` is the largest deviation between the
    measured and modelled ratio, so a large value flags a non-scalar map (the
    estimate is then an effective scalar gain, not a certified one).
    """
    traces = [list(map(float, tr)) for tr in trace]
    ratios = []
    for tr in traces:
        r = tr[index:index + 2]
        if len(r) < 2 or r[0] <= 0.0 or not np.isfinite(r[0]):
            return {"mu": float("nan"), "ratios": [], "predicted": [],
                    "fit_error": float("nan"), "method": "degenerate_trace"}
        ratios.append(r[1] / r[0])
    grid = np.concatenate([np.linspace(0.0, 3.0, 3001),
                           np.geomspace(3.001, 200.0, 600)])
    cost = np.zeros_like(grid)
    for w, rho in zip(omegas, ratios):
        cost += (np.abs(1.0 - w * grid) - rho) ** 2
    mu = float(grid[int(np.argmin(cost))])
    predicted = [float(abs(1.0 - w * mu)) for w in omegas]
    fit_error = float(max(abs(p - r) for p, r in zip(predicted, ratios))) if ratios else float("nan")
    return {"mu": mu, "ratios": ratios, "predicted": predicted,
            "fit_error": fit_error,
            "method": "model_fit_%d_omega" % len(omegas), "omegas": list(omegas)}


def marker_local_mass(ibm, rho_i):
    """``M_f = sum_i rho_i eps_i ds_i`` -- the mass one sweep can actually feel."""
    scale = np.asarray(ibm.eps_used, dtype=float) * np.asarray(ibm.ds, dtype=float)
    return float(np.dot(np.asarray(rho_i, dtype=float), scale))


def discrete_gain_bound(M_f, M_s, k=0.0, dt=1.0, beta=0.25, gamma=0.5):
    """``|dG/du|`` for a frozen-fluid IBM sweep (Newmark-consistent velocity).

    The interface force responds to the trial velocity
    ``u_body = gamma/(beta dt)(u - u_n) + ...`` as ``F = M_f u_body``, and the
    structural solve gives ``du/dF = beta dt^2 / (M_s + k beta dt^2)``; the
    product is the iterate gain, i.e. ``mu = 1 + |dG/du|``.
    """
    return abs(gamma * dt * M_f / (M_s + k * beta * dt ** 2))


def added_mass_ratio(rho_f, c, M_s):
    """Lamb flat-plate added mass ``m_a = rho_f pi c^2`` (per unit length).

    Returns ``{"m_a": ..., "ratio": m_a / M_s, "mu_predicted": 1 + ratio}``.
    ``c`` is the plate half-width, ``M_s`` the structural mass of the same
    unit-length reference.
    """
    m_a = float(rho_f) * math.pi * float(c) ** 2
    ratio = m_a / float(M_s)
    return {"m_a": m_a, "ratio": ratio, "mu_predicted": 1.0 + ratio}