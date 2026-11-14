"""Momentum-budget and constraint-residual audit for the implicit IBM coupling.

The archived cross-validation of the implicit immersed-boundary scheme
(``docs/xval_implicit_ibm_aitken_v6.md``) left two debts that no amount of force
*post-processing* can settle, because both are questions about what the coupling
actually deposits in the lattice:

1. the epsilon-conservation factor ``sum(eps_i ds_i) / sum(ds_i) ~ 2.2`` -- is it
   a missing normalisation that must be divided out of the added mass, or a
   consistent property of the operator that appears on *both* sides of the
   action/reaction pair?
2. the apparent added mass ``m_app ~ 178`` against ``rho pi c^2 = 50.27`` for the
   settled ``c = 4`` plate -- is that a physical mass, an over-driven velocity
   constraint, or a confinement effect?

This module measures the coupling from the *fluid side* instead of the reported
force:

* :func:`lattice_momentum` / :func:`population_momentum` -- the two momenta a
  Guo-forced D2Q9 scheme carries (``sum rho u`` with the half-force correction,
  and ``sum c f``, the momentum the streaming step actually transports);
* :func:`epsilon_amplification` -- the marker -> lattice force scale of the
  Pinelli/LIFE epsilon system, its round-trip error and its profile;
* :func:`one_step_force_budget` -- one collision+stream under a known marker
  force: what a Lagrange force of known magnitude really deposits;
* :func:`constrained_velocity_residual` -- how well the direct-forcing scheme
  imposes ``u(marker) = u_body`` (the constraint the whole scheme exists for);
* :func:`prescribed_plate_run` -- a prescribed-kinematics plate run ("forced
  oscillation") that records force, both momenta and the constraint residual, so
  that the force route and the momentum route to the added mass can be compared
  and the *gain* of the direct-forcing formula can be swept;
* :func:`added_mass_from_momentum` -- ``m_a = Pi / v`` from the fluid momentum,
  the route that does not use the reported interface force at all.

Everything here is read-only with respect to the solvers: no scheme is changed by
importing this module.  ``prescribed_plate_run`` takes the direct-forcing gain as
an explicit parameter so that the archived ``gain = 2`` (LIFE's formula, see
:meth:`tensorfem.ibm_aitken_implicit.IBMInterface.force`) can be compared with the
gain the discrete scheme actually needs.
"""

from __future__ import annotations

import math

import numpy as np

from .ibm_added_mass_harmonic import harmonic_projection
from .ibm_aitken_added_mass import SpringMountedBody, plate_markers
from .ibm_aitken_implicit import IBMInterface, LBMFluid2D

__all__ = [
    "lattice_momentum",
    "population_momentum",
    "epsilon_amplification",
    "one_step_force_budget",
    "constrained_velocity_residual",
    "prescribed_plate_run",
    "added_mass_from_momentum",
    "ideal_direct_forcing_gain",
    "roundtrip_operator",
]


# ---------------------------------------------------------------------------
# momenta carried by the LBM state
# ---------------------------------------------------------------------------
def lattice_momentum(fluid):
    """``sum rho u`` with the Guo half-force velocity (in lattice units)."""
    return np.array([float((fluid.rho * fluid.u[..., 0]).sum()),
                     float((fluid.rho * fluid.u[..., 1]).sum())])


def population_momentum(fluid):
    """``sum_i c_i f_i`` -- the momentum the streaming step transports.

    This is the quantity whose change over a step equals the total applied
    force (the Guo source term is built so that ``sum c src = F``); ``sum rho u``
    additionally contains the ``F/2`` half-step correction and therefore jumps by
    more than the force on the collision that declares it.
    """
    mom = (fluid.f[..., :, None] * fluid.D[None, None, :, :]).sum(axis=-2)
    return np.array([float(mom[..., 0].sum()), float(mom[..., 1].sum())])


# ---------------------------------------------------------------------------
# epsilon system as an operator property
# ---------------------------------------------------------------------------
def epsilon_amplification(interface):
    """Marker -> lattice force scale of the epsilon-conservation system.

    With ``spread(F)_s = sum_i d_i(s) eps_i ds_i F_i`` (see
    :meth:`IBMInterface.spread`), a *uniform* Lagrange force ``F`` deposits
    ``sum(eps ds) * F`` in the lattice, while the marker set's own force measure
    is ``sum(ds) * F``.  Their ratio is the "amplification" quoted in the v6
    report.  ``roundtrip_error`` is ``max_i |interp_i(spread(F = e_y)) - 1|``,
    i.e. the property the epsilon system is *designed* to enforce.
    """
    ds = np.asarray(interface.ds, dtype=float)
    eps = np.asarray(interface.eps_used, dtype=float)
    w = eps * ds
    return {
        "n_markers": int(len(ds)),
        "sum_ds": float(ds.sum()),
        "sum_eps_ds": float(w.sum()),
        "amplification": float(w.sum() / ds.sum()),
        "offset": float(w.sum() - ds.sum()),
        "eps_min": float(eps.min()),
        "eps_max": float(eps.max()),
        "eps_mean": float(eps.mean()),
        "eps_endpoint_mean": float(np.mean([eps[0], eps[-1]])),
        "eps_interior_mean": float(np.mean(eps[1:-1])) if len(eps) > 2 else float("nan"),
        "roundtrip_error": float(interface.uniform_force_transmission_error()),
    }


def roundtrip_operator(interface):
    """``B = interp o spread`` in marker space, one column per marker.

    ``spread`` deposits ``d_j(s) eps_j ds_j`` per unit marker force and
    ``interpolate`` integrates with ``d_i(s)``; for a constant unit density the
    ``y`` column of the composition is ``(A diag(eps ds))_ij``.  The epsilon
    system is built so that ``B 1 = 1`` exactly (that is what
    ``uniform_force_transmission_error`` checks); ``B`` itself is *not* the
    identity, and the off-diagonal part is what a scalar direct-forcing gain
    cannot invert -- the origin of the steady slip measured in
    :func:`constrained_velocity_residual`.
    """
    n = len(interface.positions)
    rho = np.ones(interface.shape)
    cols = []
    for j in range(n):
        f = np.zeros((n, 2))
        f[j, 1] = 1.0
        site = interface.spread(f)
        _rho_i, mom_i = interface.interpolate(rho, site)
        cols.append(mom_i[:, 1])
    return np.column_stack(cols)


def ideal_direct_forcing_gain(guo_factor=1.5):
    """Gain the one-step direct-forcing update needs for the *uniform* mode.

    After ``fluid.step(force=f_site)`` the Guo velocity is
    ``u_new = sum(c f)/rho + f_site/(2 rho)`` and ``sum(c f)`` grows by
    ``f_site`` (measured in :func:`one_step_force_budget`), so a site force
    ``f_site = B F`` with ``B`` the epsilon round-trip operator changes the Guo
    velocity by ``1.5 B F / rho``.  For the uniform mode ``B = 1`` by
    construction of the epsilon system, hence ``F = 2/3 rho (u_body - u*)``
    imposes the constraint in one step and the gain is ``2/3``.
    """
    return 1.0 / guo_factor


# ---------------------------------------------------------------------------
# one-step budget
# ---------------------------------------------------------------------------
def one_step_force_budget(fluid, site_force):
    """Collide+stream once under ``site_force`` and report what was deposited.

    Returns ``population_delta`` (``sum c f`` change -- the transported
    momentum), ``guo_delta`` (``sum rho u`` change) and ``site_force_sum`` (the
    lattice-side force that was handed in).  In the Guo scheme
    ``population_delta == site_force_sum`` to round-off and ``guo_delta`` is
    larger by the half-force correction; both are measured, neither assumed.
    """
    site_force = np.asarray(site_force, dtype=float)
    p0 = population_momentum(fluid)
    m0 = lattice_momentum(fluid)
    fluid.step(force=site_force)
    p1 = population_momentum(fluid)
    m1 = lattice_momentum(fluid)
    return {
        "population_delta": (p1 - p0).tolist(),
        "guo_delta": (m1 - m0).tolist(),
        "site_force_sum": [float(site_force[..., 0].sum()),
                           float(site_force[..., 1].sum())],
    }


def constrained_velocity_residual(interface, rho, u, u_body, scale=None):
    """How much of the velocity constraint ``u(marker) = u_body`` is missed.

    ``u_body`` is either one vector (applied to every marker) or ``(n, 2)``;
    ``scale`` is the velocity amplitude the relative error is divided by
    (defaults to the instantaneous ``max |u_body|``, which is only safe when the
    body velocity is away from zero).
    The residual is reported per component as ``max_i |u_i - u_body|`` plus the
    normalised ``rel = max|.| / max|u_body|``.  A direct-forcing scheme whose
    gain is right drives ``rel`` down; a scheme that over-drives the constraint
    leaves ``rel`` of order one, which makes any force it reports meaningless as
    a hydrodynamic load.
    """
    u_body = np.asarray(u_body, dtype=float)
    if u_body.ndim == 1:
        u_body = np.tile(u_body, (len(interface.positions), 1))
    rho_i, mom_i = interface.interpolate(rho, u)
    u_i = mom_i / rho_i[:, None]
    err = np.abs(u_i - u_body)
    # normalise by the *drive amplitude*, not by the instantaneous body velocity:
    # the latter passes through zero every quarter period and would turn a bounded
    # absolute error into an unbounded relative one.
    scale = float(np.abs(u_body).max()) if scale is None else float(scale)
    return {
        "max_err": err.max(axis=0).tolist(),
        "max_rel": float(err.max() / scale) if scale > 0.0 else 0.0,
        "rms_rel": float(np.sqrt((err ** 2).mean()) / scale) if scale > 0.0 else 0.0,
    }


# ---------------------------------------------------------------------------
# prescribed-kinematics run
# ---------------------------------------------------------------------------
def prescribed_plate_run(half_width=4.0, lattice=(64, 64), drive_period=64.0,
                         amplitude=0.2, n_steps=1600, gain=2.0,
                         correct_epsilon=True, omega_lbm=1.8, rho_f=1.0,
                         dx=1.0, center=None, skip_cycles=2.0,
                         gain_mode="scalar"):
    """Drive a rigid marker segment ``y = A sin(wt)``; record force *and* momentum.

    The scheme is :func:`tensorfem.ibm_aitken_added_mass.forced_oscillation`
    with the direct-forcing gain exposed: ``f_lag = gain * (rho_i u_body - m_i)``
    (``gain = 2`` is the archived LIFE formula).  Alongside the reported
    interface force ``F_y = -sum_i f_lag_i eps_i ds_i`` the run records

    * ``momentum`` -- ``sum c f`` (the transported fluid momentum), which needs
      no interface force to define the added mass: for a body oscillating in a
      periodic box the in-phase part of the fluid momentum is ``m_a v``;
    * ``residual_guo`` / ``residual_raw`` -- the constraint error of
      :func:`constrained_velocity_residual` evaluated after each step on the
      collision (Guo) velocity ``fluid.u`` -- the physical one -- and on the
      post-stream raw velocity ``sum c f / rho``, which lags it by
      ``f/(2 rho)``.

    Returns a record with trimmed histories plus the harmonic projections
    ``m_force`` (from the reported force), ``m_momentum`` (from the fluid
    momentum) and the residual statistics.
    """
    if center is None:
        center = (0.5 * lattice[0], 0.5 * lattice[1])
    body = SpringMountedBody(plate_markers(center, half_width), mass=1.0e9,
                             stiffness=0.0)
    fluid = LBMFluid2D(lattice, omega=omega_lbm, rho0=rho_f)
    ibm = IBMInterface(body.markers(0.0), fluid.shape, dx=dx,
                       correct_epsilon=correct_epsilon)
    w = 2.0 * math.pi / float(drive_period)
    binv = None
    if gain_mode == "operator":
        binv = np.linalg.inv(roundtrip_operator(ibm))
    elif gain_mode != "scalar":
        raise ValueError(f"unknown gain_mode {gain_mode!r}")
    forces, mom_y, resid_raw, resid_guo = [], [], [], []
    for k in range(1, int(n_steps) + 1):
        t = float(k)
        y = amplitude * math.sin(w * t)
        v = amplitude * w * math.cos(w * t)
        rho, u = fluid.macroscopic(include_force=False)
        ibm.move(body.markers(y))
        rho_i, mom_i = ibm.interpolate(rho, u)
        u_body = np.array([[0.0, v]] * len(body.positions0))
        deficit = (1.0 / dx) * rho_i[:, None] * (u_body - mom_i / rho_i[:, None])
        if gain_mode == "operator":
            # B^-1 acts on the marker index of every component separately
            f_lag = float(gain) * np.stack(
                [deficit[:, 0] @ binv.T, deficit[:, 1] @ binv.T], axis=1)
        else:
            f_lag = float(gain) * deficit
        fluid.step(force=ibm.spread(f_lag))
        wgt = ibm.eps_used * ibm.ds
        forces.append(-float(np.dot(f_lag[:, 1], wgt)))
        mom_y.append(population_momentum(fluid)[1])
        ibm.move(body.markers(y))
        # two velocities, two residuals: the post-stream *raw* velocity
        # ``sum c f / rho`` and the collision (Guo) velocity ``fluid.u`` that the
        # equilibrium of this step saw.  The physical velocity of the Guo scheme
        # is the latter; the raw one lags it by ``f/(2 rho)``.  Measuring the
        # constraint on the raw velocity understates it by up to half the drive
        # amplitude, which is exactly the trap this field pair documents.
        resid_raw.append(constrained_velocity_residual(
            ibm, fluid.rho, fluid.u, np.array([0.0, v]),
            scale=amplitude * w)["max_rel"])
        rho_s, u_s = fluid.macroscopic(include_force=False)
        resid_guo.append(constrained_velocity_residual(
            ibm, rho_s, u_s, np.array([0.0, v]),
            scale=amplitude * w)["max_rel"])
    ts = np.arange(1, int(n_steps) + 1, dtype=float)
    proj_f = harmonic_projection(ts, np.asarray(forces), w,
                                 skip_cycles=skip_cycles)
    proj_m = harmonic_projection(ts, np.asarray(mom_y), w,
                                 skip_cycles=skip_cycles)
    amp = float(amplitude) * w
    return {
        "half_width": float(half_width), "lattice": list(lattice),
        "drive_period": float(drive_period), "amplitude": float(amplitude),
        "n_steps": int(n_steps), "gain": float(gain),
        "correct_epsilon": bool(correct_epsilon),
        "n_markers": int(len(body.positions0)),
        "m_force": float(proj_f["in_phase"] / (float(amplitude) * w ** 2)),
        "m_momentum": float(proj_m["quadrature"] / amp),
        "damping_force_route": float(-proj_f["quadrature"] / amp),
        "r2_force": float(proj_f["r2"]),
        "r2_momentum": float(proj_m["r2"]),
        "residual_guo_max": float(np.max(resid_raw)),
        "residual_guo_mean": float(np.mean(resid_raw)),
        "residual_guo_last_cycle": float(np.mean(resid_raw[-int(drive_period):])),
        "residual_raw_max": float(np.max(resid_guo)),
        "residual_raw_mean": float(np.mean(resid_guo)),
        "residual_raw_last_cycle": float(np.mean(resid_guo[-int(drive_period):])),
        "momentum_last": float(mom_y[-1]),
        "force_last": float(forces[-1]),
        "force_history": [float(x) for x in forces],
        "momentum_history": [float(x) for x in mom_y],
        "residual_guo_history": [float(x) for x in resid_raw],
        "residual_raw_history": [float(x) for x in resid_guo],
    }


def added_mass_from_momentum(t, momentum, velocity, omega, skip_cycles=2.0):
    """``m_a = <Pi, cos(wt)> / <v, cos(wt)>`` -- the force-free route.

    For a body oscillating in a periodic box the fluid momentum in the drive
    direction is, to first order in the amplitude, ``Pi = m_a v``; projecting
    both ``Pi`` and the known ``v = A w cos(wt)`` on the drive phase gives the
    added mass without ever forming an interface force.
    """
    proj = harmonic_projection(t, np.asarray(momentum, dtype=float), omega,
                               skip_cycles=skip_cycles)
    vel = np.asarray(velocity, dtype=float)
    ref = float(np.abs(vel).max())
    return {
        "in_phase": float(proj["in_phase"]),
        "quadrature": float(proj["quadrature"]),
        "r2": float(proj["r2"]),
        "amplitude": ref,
        "added_mass": float(proj["quadrature"] / ref) if ref > 0.0 else float("nan"),
    }