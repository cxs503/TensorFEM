"""Prototype 2: closed added-mass iteration (structure in the loop).

State: trial displacement y_k.  Map:
  U_b = body_velocity(y_k)                       (Newmark-consistent)
  F   = 2 (dt/dx) rho_i U_b - 2 m_i^post(F_prev)  (m_i^post from a trial LBM step)
  y_new = newmark(-sum F eps ds)
Measured residual ratio with omega = 1 -> mu = 1 + ratio; analytic mu = 1 + m_a/m_s.
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


def make(c=8.0, mass=500.0, rho_f=1.0, lattice=(64, 64), dt=1.0):
    xs = np.linspace(-c, c, int(round(2 * c)) + 1)
    pos = np.stack([xs + 0.5 * lattice[0], np.full_like(xs, 0.5 * lattice[1])],
                   axis=1)
    body = SpringMountedBody(pos, mass=mass, stiffness=3.0)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=rho_f)
    ibm = IBMInterface(body.markers(), fluid.shape, dx=1.0)
    return body, fluid, ibm, c


def iterate(body, fluid, ibm, n_iter=6, omega=1.0, dt=1.0, dx=1.0,
            y_start=0.02, aitken=None):
    rho, u = fluid.macroscopic(include_force=False)
    F = np.zeros((len(body.positions0), 2))
    y_k = y_start
    hist = []
    r_prev = None
    omega_applied = omega
    for k in range(n_iter):
        U_b = body.body_velocity(y_k, dt)
        ft = copy.deepcopy(fluid)
        ft.step(force=ibm.spread(F))
        rho_i, mom_i = ibm.interpolate(ft.f.sum(axis=-1),
                                       ft.macroscopic(include_force=False)[1])
        F_raw = 2.0 * ((dt / dx) * rho_i[:, None] * U_b - mom_i)
        w = ibm.eps_used * ibm.ds
        f_body = -float(np.dot(F_raw[:, 1], w))
        y_new, _v, _a = body.newmark_step(f_body, dt)
        r_k = np.array([y_new - y_k])
        hist.append((float(abs(r_k[0])), y_k, f_body, float(np.max(np.abs(F_raw)))))
        F = F_raw  # lagged fluid response: next trial uses this forcing
        if aitken is not None and r_prev is not None:
            omega_applied = aitken.omega_for(r_prev, r_k, omega_prev=omega_applied)
        else:
            omega_applied = omega
        r_prev = r_k.copy()
        y_k = y_k + omega_applied * (y_new - y_k)
    return hist


if __name__ == "__main__":
    for mass in (2000.0, 1000.0, 500.0, 250.0):
        body, fluid, ibm, c = make(c=8.0, mass=mass)
        hist = iterate(body, fluid, ibm, n_iter=8, omega=1.0)
        res = [h[0] for h in hist]
        ratios = [(res[i + 1] / res[i]) for i in range(len(res) - 1) if res[i]]
        m_a = math.pi * c ** 2
        print(f"mass={mass:6.0f} mu_an={1 + m_a/mass:5.3f} "
              f"ratios={['%.3f' % r for r in ratios]}")
        print("     res:", " ".join("%.3e" % r for r in res))