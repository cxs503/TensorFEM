"""Grid-refinement (``dx -> 0``) measurement of the implicit-IBM added mass.

Motivation
----------
Every archived round of ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (v1-v7)
swept the body size *in lattice cells* at a fixed lattice spacing, so the one
question that separates a discretisation artefact from a formulation error was
never asked: **what happens to the measured added mass when the lattice is
refined at fixed physical geometry?**

This module asks it with explicit units.  The scaling used here is the acoustic
one, ``dt = dx``, so that

* a length ``L`` is ``L/dx`` cells,
* a time ``T`` is ``T/dx`` steps,
* a velocity has the *same* numerical value in lattice and physical units
  (``dx/dt = 1``),
* the sound speed and the Mach number of the prescribed motion are refinement
  independent, so the potential-flow (added-mass) physics is unchanged while the
  interface kernel, whose support is a *fixed number of cells*, becomes thinner
  in physical units.

The plate is prescribed (``y = A sin(omega t)``), so no structural fixed point is
involved and the added mass is read from the **fluid momentum**

    ``Pi_y(t) = sum_sites (c f)_y``,   ``m_a = <Pi_y, cos(wt)> / v_amp``,

which needs no interface force at all, plus -- as a cross-check -- from the
reported marker force scaled with the *lattice* acceleration
``a_lat = (A/dx) w_lat^2``.

``refine = 1`` reproduces the archived ``dx = 1`` configuration, so the module is
a strict extension of :func:`tensorfem.ibm_momentum_budget.prescribed_plate_run`
(the difference being that the marker spacing and the drive period are made
consistent with ``dx`` here; the legacy helper keeps ``dx`` only in the interface
operator, which silently changes the physics when ``dx != 1``).
"""

from __future__ import annotations

import math

import numpy as np

from .ibm_added_mass_harmonic import harmonic_projection
from .ibm_aitken_added_mass import SpringMountedBody, plate_markers
from .ibm_aitken_implicit import IBMInterface, LBMFluid2D
from .ibm_momentum_budget import population_momentum, roundtrip_operator

__all__ = ["dx_refinement_run", "lattice_added_mass", "KERNEL_SUPPORT_CELLS"]

#: half-width of the 3-point IBM kernel support, in cells (``dirac_delta_3pt``)
KERNEL_SUPPORT_CELLS = 1.5


def lattice_added_mass(refine=1, c_phys=4.0, box_phys=64.0, amplitude=0.2,
                       period_phys=64.0, n_cycles=4.0, skip_cycles=2.0,
                       gain=2.0, gain_mode="scalar", correct_epsilon=True,
                       omega_lbm=1.8, rho_f=1.0):
    """Prescribed-oscillation added mass at lattice refinement ``1/dx = refine``.

    Parameters are *physical* (``dx = 1`` units) and are kept fixed when
    ``refine`` changes, so a sequence of runs is a genuine grid-refinement study.
    Returns the dimensionless coefficient ``C = m_lat / (pi rho c_lat^2)`` for
    the momentum route and the force route (``C = 1`` is the thin-plate value
    ``rho pi c^2``; the discrete plate is expected to approach it as the kernel
    becomes thin, i.e. as ``refine`` grows).
    """
    dx = 1.0 / float(refine)
    box = np.atleast_1d(np.asarray(box_phys, dtype=float))
    if box.size == 1:
        box = np.array([box[0], box[0]])
    n_cell = tuple(int(round(L / dx)) for L in box)
    center = (0.5 * box[0], 0.5 * box[1])
    body = SpringMountedBody(plate_markers(center, c_phys, dx=dx),
                             mass=1.0e9, stiffness=0.0)
    fluid = LBMFluid2D(n_cell, omega=omega_lbm, rho0=rho_f)
    ibm = IBMInterface(body.markers(0.0), fluid.shape, dx=dx,
                       correct_epsilon=correct_epsilon)
    t_lat = float(period_phys) / dx          # steps per drive cycle
    w_lat = 2.0 * math.pi / t_lat            # rad per lattice time unit
    w_phys = 2.0 * math.pi / float(period_phys)
    amp = float(amplitude)
    v_amp = amp * w_phys                     # lattice == physical (dt = dx)
    n_steps = int(round(t_lat * float(n_cycles)))
    binv = None
    if gain_mode == "operator":
        binv = np.linalg.inv(roundtrip_operator(ibm))
    elif gain_mode != "scalar":
        raise ValueError(f"unknown gain_mode {gain_mode!r}")

    forces, mom_y, resid, umark = [], [], [], []
    for k in range(1, n_steps + 1):
        t = float(k)
        y = amp * math.sin(w_lat * t)
        v = v_amp * math.cos(w_lat * t)
        ibm.move(body.markers(y))
        rho, u = fluid.macroscopic(include_force=False)
        rho_i, mom_i = ibm.interpolate(rho, u)
        u_body = np.array([[0.0, v]] * len(body.positions0))
        deficit = rho_i[:, None] * u_body - mom_i          # lattice momentum
        if gain_mode == "operator":
            f_lag = float(gain) * np.stack(
                [deficit[:, 0] @ binv.T, deficit[:, 1] @ binv.T], axis=1)
        else:
            f_lag = float(gain) * deficit
        fluid.step(force=ibm.spread(f_lag))
        wgt = ibm.eps_used * ibm.ds
        forces.append(-float(np.dot(f_lag[:, 1], wgt)))
        mom_y.append(population_momentum(fluid)[1])
        # constraint audit on the collision (Guo) velocity -- the physical one.
        # ``fluid.step`` leaves ``fluid.u`` at the Guo level of the *new* state,
        # which is the level the direct-forcing gain 2 is derived for; reading
        # ``macroscopic(include_force=False)`` instead would compare the target
        # velocity with the pre-force velocity and overstate the residual by
        # O(F / 2 rho) (the trap documented in the v7 report).
        ibm.move(body.markers(y))
        ri, mi = ibm.interpolate(fluid.rho, fluid.u)
        u_mark = mi[:, 1] / ri
        resid.append(float(np.max(np.abs(u_mark - v))) / max(v_amp, 1e-30))
        umark.append(float(np.mean(u_mark)))

    ts = np.arange(1, n_steps + 1, dtype=float)
    proj_m = harmonic_projection(ts, np.asarray(mom_y), w_lat, skip_cycles=skip_cycles)
    proj_f = harmonic_projection(ts, np.asarray(forces), w_lat, skip_cycles=skip_cycles)
    proj_um = harmonic_projection(ts, np.asarray(umark), w_lat, skip_cycles=skip_cycles)
    v_achieved = float(proj_um["quadrature"])
    c_lat = float(c_phys) / dx
    a_lat = (amp / dx) * w_lat ** 2
    m_mom = float(proj_m["quadrature"]) / v_amp
    m_force = float(proj_f["in_phase"]) / a_lat
    ref = math.pi * rho_f * c_lat ** 2
    return {
        "refine": int(refine), "dx": dx, "n_cell": list(n_cell),
        "c_lat": c_lat, "c_phys": float(c_phys),
        "box_phys": [float(box[0]), float(box[1])],
        "amplitude": float(amplitude), "period_phys": float(period_phys),
        "n_markers": int(len(body.positions0)),
        "ds_median": float(np.median(ibm.ds)),
        "t_lat": t_lat, "w_lat": w_lat, "v_amp": v_amp,
        "n_steps": n_steps, "gain": float(gain), "gain_mode": gain_mode,
        "correct_epsilon": bool(correct_epsilon),
        "m_momentum_lat": m_mom, "m_force_lat": m_force,
        "analytic_lat": ref,
        "C_momentum": m_mom / ref, "C_force": m_force / ref,
        "err_momentum": m_mom / ref - 1.0, "err_force": m_force / ref - 1.0,
        "r2_momentum": float(proj_m["r2"]), "r2_force": float(proj_f["r2"]),
        "periods_in_window": float(proj_m["periods_in_window"]),
        "amp_cells": amp / dx,
        "eps_amplification": float(np.sum(ibm.eps_used * ibm.ds) / np.sum(ibm.ds)),
        "constraint_rel_max": float(np.max(resid)),
        "v_achieved": v_achieved,
        "v_achieved_ratio": v_achieved / v_amp,
        "umark_history": [float(x) for x in umark],
        "momentum_history": [float(x) for x in mom_y],
        "force_history": [float(x) for x in forces],
    }


def dx_refinement_run(**kwargs):
    """Alias kept for the benchmark script's readability."""
    return lattice_added_mass(**kwargs)