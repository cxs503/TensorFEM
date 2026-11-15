"""Probe: interface-map gain of the beam driver, and the fixed-point spectrum.

Measures, for the production `ImplicitIBMFSI` (beam) driver:
  * residual sequence with NO relaxation (omega = 1) over 4 sub-iterations
    -> interface gain mu = 1 + r_{k+1}/r_k  (for a linear 1-D map r_{k+1}=(1-omega*mu) r_k)
  * Aitken omega* actually produced
  * physics: wet/dry frequency ratio -> R_phys = (f_vac/f_wet)^2 - 1
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_implicit import (  # noqa: E402
    BeamProperties, CorotationalBeam2D, IBMInterface, ImplicitIBMFSI, LBMFluid2D,
    _section, _estimate_period,
)


def build(fluid_rho=1.0, lattice=(96, 72), n_elements=16, dt=2.0e-3,
          amplitude=0.05, rho_s=1.0):
    beam = CorotationalBeam2D(n_elements=n_elements, props=_section())
    beam.props = BeamProperties(E=beam.props.E, A=beam.props.A, I=beam.props.I,
                                rho=rho_s, length=beam.props.length,
                                thickness=beam.props.thickness)
    scale = 0.45 * float(lattice[0])
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amplitude * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    return beam, fluid


def probe_subiter(fluid_rho=1.0, rho_s=1.0, n_iter=4, **kw):
    beam, fluid = build(fluid_rho=fluid_rho, rho_s=rho_s, **kw)
    solver = ImplicitIBMFSI(beam, fluid, dt=kw.get("dt", 2.0e-3), dx=1.0,
                            ref_length=beam.props.length, sub_tol=0.0,
                            max_sub_iter=n_iter, relax_mode="fixed",
                            fixed_omega=1.0)
    rho, u = fluid.macroscopic(include_force=False)
    beam.u = beam.predict(1)
    u_km1 = beam.u.copy()
    res_hist, vnorm = [], []
    for it in range(1, n_iter + 1):
        solver.ibm.move(beam.markers())
        rho_i, mom_i = solver.ibm.interpolate(rho, u)
        f_lag = solver.ibm.force(rho_i, mom_i, beam.node_velocities(),
                                 dt=solver.dt)
        load = beam.apply_marker_loads(-f_lag)
        u_new, _ok, _nit, _nres = beam.newmark_solve(load, solver.dt)
        r_k = u_new - u_km1
        res, _split = solver.interface_residual(r_k)
        res_hist.append(res)
        vnorm.append(float(np.linalg.norm(beam.node_velocities())))
        u_km1 = u_new.copy()
        beam.u = u_new.copy()
    return res_hist, vnorm


def probe_physics(fluid_rho=1.0, rho_s=1.0, n_steps=300, **kw):
    """wet/dry frequency ratio -> R_phys (no relaxation-free tricks, aitken on)."""
    beam, fluid = build(fluid_rho=fluid_rho, rho_s=rho_s, **kw)
    dt = kw.get("dt", 2.0e-3)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0,
                            ref_length=beam.props.length, sub_tol=1e-6,
                            max_sub_iter=30, relax_mode="aitken")
    hist = solver.run(n_steps, t_start=1)
    tips = [h["tip"][1] for h in hist]
    return tips, _estimate_period(tips, dt), [h["sub_iterations"] for h in hist], [
        h["omega"] for h in hist]


def probe_dry(n_steps=300, rho_s=1.0, **kw):
    beam, _ = build(fluid_rho=0.0, rho_s=rho_s, **kw)
    dt = kw.get("dt", 2.0e-3)
    tips = []
    for k in range(1, n_steps + 1):
        beam.u = beam.predict(k)
        beam.newmark_solve(np.zeros(beam.n_dof), dt)
        beam.advance_history()
        tips.append(beam.tip()[1])
    return tips, _estimate_period(tips, dt)


if __name__ == "__main__":
    for fluid_rho, rho_s in [(0.2, 1.0), (1.0, 1.0), (1.0, 0.2)]:
        rh, vn = probe_subiter(fluid_rho=fluid_rho, rho_s=rho_s)
        ratios = [rh[i + 1] / rh[i] if rh[i] else float("nan")
                  for i in range(len(rh) - 1)]
        print(f"rho_f={fluid_rho} rho_s={rho_s} res={['%.3e' % x for x in rh]}")
        print(f"   ratios={['%.4f' % x for x in ratios]} vnorm={['%.3e' % x for x in vn]}")