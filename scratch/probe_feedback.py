"""Probe 2: does the interface iterate actually see the trial force?

Legacy loop evaluates the IBM force from the *frozen* pre-force fluid state, so
the only feedback is through the body's own velocity.  This probe adds the
missing piece (linear fluid response to the trial force) and measures the
interface-map gain of the resulting iteration.

    u_i^{eff} = u_i^pre + c * interp(spread(F^{k-1})) / rho_i
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_implicit import ImplicitIBMFSI  # noqa: E402
from probe_gain import build  # noqa: E402


def probe(feedback=1.0, fluid_rho=1.0, rho_s=1.0, n_iter=5, dt=2.0e-3,
          n_elements=16, lattice=(96, 72), amplitude=0.05):
    beam, fluid = build(fluid_rho=fluid_rho, rho_s=rho_s, dt=dt,
                        n_elements=n_elements, lattice=lattice,
                        amplitude=amplitude)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0,
                            ref_length=beam.props.length, sub_tol=0.0,
                            max_sub_iter=n_iter, relax_mode="fixed",
                            fixed_omega=1.0)
    rho, u = fluid.macroscopic(include_force=False)
    rho_i0, _ = solver.ibm.interpolate(rho, u)
    beam.u = beam.predict(1)
    u_km1 = beam.u.copy()
    f_prev = None
    res_hist, slip = [], []
    for it in range(1, n_iter + 1):
        solver.ibm.move(beam.markers())
        rho_i, mom_i = solver.ibm.interpolate(rho, u)
        if f_prev is not None:
            # linear fluid response to the trial force: direct forcing drives the
            # marker velocity towards U_b, the induced correction scales as F/rho
            corr = solver.ibm.interpolate(np.ones_like(rho),
                                          feedback * f_prev / rho[..., None])[1]
            mom_i = mom_i + corr
        u_body = beam.node_velocities()
        f_lag = solver.ibm.force(rho_i, mom_i, u_body, dt=dt)
        f_prev = solver.ibm.spread(f_lag)
        load = beam.apply_marker_loads(-f_lag)
        u_new, _ok, _nit, _nres = beam.newmark_solve(load, dt)
        # achieved slip at the markers (what strong coupling wants to vanish)
        rho_i2, mom_i2 = solver.ibm.interpolate(rho, u + feedback * f_prev
                                               / rho[..., None])
        slip.append(float(np.max(np.linalg.norm(mom_i2 / rho_i2[:, None]
                                                - beam.node_velocities(), axis=1))))
        r_k = u_new - u_km1
        res, _s = solver.interface_residual(r_k)
        res_hist.append(res)
        u_km1 = u_new.copy()
        beam.u = u_new.copy()
    return res_hist, slip


if __name__ == "__main__":
    for fb in (0.0, 1.0):
        for fluid_rho, rho_s in [(0.2, 1.0), (1.0, 1.0), (1.0, 0.2)]:
            rh, sl = probe(feedback=fb, fluid_rho=fluid_rho, rho_s=rho_s)
            ratios = [rh[i + 1] / rh[i] if rh[i] else float("nan")
                      for i in range(len(rh) - 1)]
            print(f"fb={fb} rho_f={fluid_rho} rho_s={rho_s}")
            print("   res  :", " ".join("%.3e" % x for x in rh))
            print("   ratio:", " ".join("%.4f" % x for x in ratios))
            print("   slip :", " ".join("%.3e" % x for x in sl))