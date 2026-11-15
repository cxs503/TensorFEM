"""Prototype: Picard iteration on the *trial-stepped* fluid field (true slip coupling).

For a thin rigid plate (SpringMountedBody, analytic Lamb added mass m_a = rho pi c^2)
measure the interface-map gain mu of the iteration

    F^{k+1} = F^k + omega * ( 2 (dt/dx) rho_i U_b - m_i^{post}(F^k) - F^k )

where m_i^{post} comes from interpolating a *trial* LBM step on a copy of the
lattice.  With omega = 1 the residual ratio is |1 - mu|, so mu = 1 + ratio.
Predict mu = 1 + m_a/m_s.
"""
from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    IBMInterface, LBMFluid2D, SpringMountedBody,
)


def make(c=8.0, n_markers=None, mass=500.0, rho_f=1.0, lattice=(64, 64),
         dt=1.0, omega_bgk=1.8):
    n_markers = n_markers or int(round(2 * c)) + 1
    xs = np.linspace(-c, c, n_markers)
    pos = np.stack([xs + 0.5 * lattice[0], np.full_like(xs, 0.5 * lattice[1])],
                   axis=1)
    body = SpringMountedBody(pos, mass=mass, stiffness=3.0)
    fluid = LBMFluid2D(lattice, omega=omega_bgk, rho0=rho_f)
    ibm = IBMInterface(body.markers(), fluid.shape, dx=1.0)
    return body, fluid, ibm, c


def trial_step(fluid, ibm, F):
    """One LBM step on a copy of the lattice with interface force density F."""
    f = copy.deepcopy(fluid)
    f.step(force=ibm.spread(F))
    return f


def picard(body, fluid, ibm, c, n_iter=6, omega=1.0, dt=1.0, dx=1.0,
           rho_ref=None):
    rho, u = fluid.macroscopic(include_force=False)
    rho_ref = float(rho.mean()) if rho_ref is None else rho_ref
    y = 0.05 * c
    U_b = body.body_velocity(y, dt)
    F = np.zeros((len(body.positions0), 2))
    hist = []
    for k in range(n_iter):
        ft = trial_step(fluid, ibm, F)
        rho_i, mom_i = ibm.interpolate(ft.f.sum(axis=-1),
                                      ft.macroscopic(include_force=False)[1])
        F_raw = 2.0 * ((dt / dx) * rho_i[:, None] * U_b - mom_i)
        resid = float(np.max(np.abs(F_raw - F)))
        slip = float(np.max(np.abs(mom_i / rho_i[:, None] - U_b)))
        hist.append((resid, slip, float(np.max(np.abs(F)))))
        F = F + omega * (F_raw - F)
    return hist, rho_ref


def analytic_gain(mass, c, rho_f):
    m_a = rho_f * math.pi * c ** 2
    return 1.0 + m_a / mass, m_a


if __name__ == "__main__":
    for mass in (2000.0, 1000.0, 500.0, 250.0):
        body, fluid, ibm, c = make(c=8.0, mass=mass)
        hist, _ = picard(body, fluid, ibm, c, n_iter=6, omega=1.0)
        res = [h[0] for h in hist]
        ratios = [res[i + 1] / res[i] for i in range(len(res) - 1) if res[i]]
        mu_an, m_a = analytic_gain(mass, c, 1.0)
        print(f"mass={mass} rho_f=1 c={c}: ratio={['%.3f' % r for r in ratios]} "
              f"mu_meas={1+ (ratios[1] if len(ratios)>1 else float('nan')):.3f} "
              f"mu_an={mu_an:.3f} slip={['%.2e' % h[1] for h in hist]}")