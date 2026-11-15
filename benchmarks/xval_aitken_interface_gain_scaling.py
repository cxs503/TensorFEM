"""XVAL v14 -- why the Aitken sub-iteration is inert in the implicit-IBM beam driver.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  v12 reported the beam's
interface gain as ``mu ~ 1`` and left the multi-DOF law ``WITHHELD``; v13 traced
that to the frozen fluid field.  This round measures the *scaling* of the gain in
the time step (the one free parameter nobody had varied), falsifies the natural
"velocity-parity" hypothesis, and certifies the resulting statement with the full
interface Jacobian of the shipped sub-iteration map.

Stages
  S1  interface gain ``mu(dt)`` in both velocity modes (predictor / iterate)
  S2  comparison with the closed-form frozen-fluid bound of v13
  S3  full Jacobian ``dG/du`` of the shipped map, its spectrum, and the v12
      lagged-minres prediction of the sub-iteration count vs the measured one
  S4  figures

Single core, ~2 min.  Usage::

    PYTHONPATH=src python benchmarks/xval_aitken_interface_gain_scaling.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfem.ibm_aitken_implicit import (
    BeamProperties,
    CorotationalBeam2D,
    ImplicitIBMFSI,
    LBMFluid2D,
)
from tensorfem.ibm_subiter_gain import (
    discrete_gain_bound,
    gain_from_trace,
    marker_local_mass,
)
from tensorfem.ibm_aitken_spectral import minres_relaxation, operator_spectrum

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_aitken_interface_gain_scaling_artifacts")
LATTICE = (96, 72)
N_ELEMENTS = 10
AMPLITUDE = 0.02
DT_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 1.0e-2, 5.0e-3, 2.0e-3)


# ---------------------------------------------------------------------------
# geometry: the v6 / certificate submerged cantilever (fixed section, no rescale)
# ---------------------------------------------------------------------------
def submerged_beam(rho_s=1.0, fluid_rho=1.0, n_elements=N_ELEMENTS,
                   amplitude=AMPLITUDE, lattice=LATTICE):
    """Beam scaled into the lattice box, released from a static deflection.

    Same construction as :func:`tests` helper of the v6 record: the section is
    carried into lattice units (``A*scale``, ``I*scale^3``) so that the markers
    are ~1 lattice unit apart *and* the beam is ``0.45 * nx`` long.
    """
    props = BeamProperties(E=1.0e5, A=1.0, I=1.0 / 12.0, rho=rho_s,
                           length=1.0, thickness=1.0)
    scale = 0.45 * float(lattice[0])
    beam = CorotationalBeam2D(n_elements=n_elements, props=props)
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=props.E / scale, A=props.A * scale,
                                I=props.I * scale ** 3, rho=rho_s,
                                length=props.length * scale,
                                thickness=props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amplitude * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    return beam, fluid


def make_solver(rho_s=1.0, dt=2.0e-3, mode="predictor", omega=1.0,
                relax_mode="fixed", nsub=5, coupling_scale=1.0, fluid_rho=1.0):
    beam, fluid = submerged_beam(rho_s=rho_s, fluid_rho=fluid_rho)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0, sub_tol=0.0,
                            max_sub_iter=nsub, relax_mode=relax_mode,
                            fixed_omega=omega, velocity_mode=mode,
                            coupling_scale=coupling_scale)
    return solver


def nodal_mass(beam):
    """Mean translational nodal mass of the structure (the ``M_s`` of v13)."""
    diag = np.diag(beam.M)
    return float(np.mean(diag[1::3][1:]))


# ---------------------------------------------------------------------------
# S1 -- interface gain vs the time step
# ---------------------------------------------------------------------------
def gain_stage(dts=DT_GRID, sections=(0.2, 1.0), modes=("predictor", "iterate")):
    rows = []
    for rho_s in sections:
        for mode in modes:
            for dt in dts:
                traces = []
                for w in (1.0, 0.5):
                    solver = make_solver(rho_s=rho_s, dt=dt, mode=mode, omega=w)
                    solver.step(1)
                    traces.append(solver.residual_traces[-1])
                g = gain_from_trace(traces, omegas=(1.0, 0.5))
                rows.append({"rho_s": rho_s, "velocity_mode": mode, "dt": dt,
                             "mu": g["mu"], "gain": g["mu"] - 1.0,
                             "fit_error": g["fit_error"],
                             "r0": traces[0][0],
                             "first_ratio": (traces[0][1] / traces[0][0]
                                             if traces[0][0] else float("nan"))})
    # log-log slope of the gain in dt, per (section, mode)
    slopes = {}
    for rho_s in sections:
        for mode in modes:
            sel = [r for r in rows if r["rho_s"] == rho_s
                   and r["velocity_mode"] == mode and r["gain"] > 0.0]
            if len(sel) < 2:
                continue
            x = np.log([r["dt"] for r in sel])
            y = np.log([r["gain"] for r in sel])
            slopes["rho_s=%g/%s" % (rho_s, mode)] = float(np.polyfit(x, y, 1)[0])
    return {"rows": rows, "loglog_slope": slopes}


# ---------------------------------------------------------------------------
# S2 -- closed-form (v13) frozen-fluid bound at the same parameters
# ---------------------------------------------------------------------------
def bound_stage(dts=(1.0, 0.1, 1.0e-2, 2.0e-3), sections=(0.2, 1.0)):
    rows = []
    for rho_s in sections:
        solver = make_solver(rho_s=rho_s, dt=2.0e-3)
        beam, fluid = solver.beam, solver.fluid
        rho, u = fluid.macroscopic(include_force=False)
        solver.ibm.move(beam.markers())
        rho_i, _ = solver.ibm.interpolate(rho, u)
        M_f = marker_local_mass(solver.ibm, rho_i)
        M_s = nodal_mass(beam)
        for dt in dts:
            rows.append({"rho_s": rho_s, "dt": dt, "M_f": M_f, "M_s": M_s,
                         "M_f_over_M_s": M_f / M_s,
                         "bound": discrete_gain_bound(M_f, M_s, dt=dt)})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# S3 -- the shipped map's Jacobian, its spectrum, and the Aitken prediction
# ---------------------------------------------------------------------------
def map_evaluate(solver, u_state, rho, u):
    """One evaluation of the shipped sub-iteration map ``u -> G(u)``.

    Replicates the six lines of ``ImplicitIBMFSI.step`` that build the map, using
    the public interface only (no source instrumentation, no reimplementation of
    the force law or of Newmark).
    """
    beam = solver.beam
    beam.u = np.array(u_state, dtype=float)
    solver.ibm.move(beam.markers())
    rho_i, mom_i = solver.ibm.interpolate(rho, u)
    f_lag = solver.ibm.force(rho_i, mom_i, solver.marker_velocity(), dt=solver.dt)
    load = solver.coupling_scale * beam.apply_marker_loads(-f_lag)
    u_new, _ok, _nit, _res = beam.newmark_solve(load, solver.dt)
    return np.array(u_new, dtype=float)


def jacobian(solver, u_state, rel_step=1.0e-6):
    rho, u = solver.fluid.macroscopic(include_force=False)
    free = np.asarray(solver.beam.free, dtype=int)
    h = rel_step * max(1.0, float(np.linalg.norm(u_state[free])))
    J = np.zeros((free.size, free.size))
    for j, dof in enumerate(free):
        up = u_state.copy()
        um = u_state.copy()
        up[dof] += h
        um[dof] -= h
        gp = map_evaluate(solver, up, rho, u)
        gm = map_evaluate(solver, um, rho, u)
        J[:, j] = (gp[free] - gm[free]) / (2.0 * h)
    # restore the beam state the sweep left behind
    solver.beam.u = np.zeros_like(u_state)
    solver.beam.u[:] = u_state
    return J


def jacobian_stage(dts=(1.0, 2.0e-3), rho_s=1.0, rel_tol=1.0e-3):
    rows = []
    for dt in dts:
        solver = make_solver(rho_s=rho_s, dt=dt, mode="predictor", omega=1.0,
                             nsub=60, relax_mode="fixed")
        # first iterate of a step = the Newmark predictor (identical in both modes)
        solver.beam.u = solver.beam.predict(1)
        u_state = solver.beam.u.copy()
        J = jacobian(solver, u_state)
        M = np.eye(J.shape[0]) - J
        spec = operator_spectrum(M)
        # v12 lagged-minres model driven by the *measured* operator
        solver2 = make_solver(rho_s=rho_s, dt=dt, mode="predictor", omega=1.0,
                              nsub=60, relax_mode="fixed")
        r0 = None
        rec = solver2.step(1)
        # first residual of that step: r_0 = G(u_pred) - u_pred
        rho, u = solver2.fluid.macroscopic(include_force=False)
        s3 = make_solver(rho_s=rho_s, dt=dt, mode="predictor", omega=1.0, nsub=60)
        u_pred = s3.beam.predict(1)
        g = map_evaluate(s3, u_pred, *s3.fluid.macroscopic(include_force=False))
        r0 = g[np.asarray(s3.beam.free, dtype=int)] - u_pred[np.asarray(s3.beam.free, dtype=int)]
        pred = minres_relaxation(M, r0, tol=rel_tol, first_omega=1.0)
        # production Aitken count at the same relative tolerance
        solver4 = make_solver(rho_s=rho_s, dt=dt, mode="predictor", omega=1.0,
                              nsub=60, relax_mode="aitken")
        solver4.sub_tol = rel_tol * float(np.linalg.norm(r0))
        rec4 = solver4.step(1)
        rows.append({
            "rho_s": rho_s, "dt": dt, "n_free": int(J.shape[0]),
            "jac_norm2": float(np.linalg.norm(J, 2)),
            "jac_norm_max": float(np.abs(J).max()),
            "spec_abs_max": float(np.abs(spec).max()),
            "spec_abs_min": float(np.abs(spec).min()),
            "kappa_M": float(np.abs(spec).max() / max(np.abs(spec).min(), 1e-300)),
            "r0_norm": float(np.linalg.norm(r0)),
            "count_predicted": int(pred["iterations"]),
            "count_measured": int(rec4["sub_iterations"]),
            "omega_measured": float(rec4["omega"]),
            "residual_measured": float(rec4["residual"]),
            "model_residual_1": float(np.linalg.norm(pred["residuals"][0])
                                      / np.linalg.norm(r0)),
            "measured_ratio_1": (float(solver4.residual_traces[-1][1]
                                       / solver4.residual_traces[-1][0])
                                 if len(solver4.residual_traces[-1]) > 1
                                 else float("nan")),
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# S4 -- figures
# ---------------------------------------------------------------------------
def figure_gain(rows, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for rho_s, mk in ((0.2, "o"), (1.0, "s")):
        for mode, ls in (("predictor", "-"), ("iterate", "--")):
            sel = [r for r in rows if r["rho_s"] == rho_s
                   and r["velocity_mode"] == mode and r["gain"] > 0.0]
            if not sel:
                continue
            ax[0].loglog([r["dt"] for r in sel], [r["gain"] for r in sel],
                         mk + ls, label=r"$\rho_s$=%g, %s" % (rho_s, mode))
    dts = np.array([2.0e-3, 1.0])
    ax[0].loglog(dts, 4.0e-2 * dts, "k:", label=r"slope 1 ($\propto dt$)")
    ax[0].loglog(dts, 4.0e-2 * dts ** 2, "k-.", label=r"slope 2 ($\propto dt^2$)")
    ax[0].set_xlabel("time step $dt$")
    ax[0].set_ylabel(r"interface gain $\mu-1$")
    ax[0].set_title("frozen-fluid interface gain collapses with $dt$")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].grid(alpha=0.3, which="both")

    solver = make_solver(rho_s=1.0, dt=2.0e-3, mode="predictor", omega=1.0)
    solver.step(1)
    tr = solver.residual_traces[-1]
    ax[1].semilogy(range(len(tr)), np.maximum(tr, 1e-20), "o-")
    ax[1].set_xlabel("sub-iteration $k$")
    ax[1].set_ylabel(r"interface residual $|r_k|$")
    ax[1].set_title("shipped loop: $dt$ = 2e-3 (residual dies in 1 sweep)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_jacobian(jac, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for row in jac["rows"]:
        sel = row
        ax[0].bar(str(row["dt"]), row["spec_abs_max"], width=0.5,
                  label=r"$\rho(I-J)$, $dt$=%g" % row["dt"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\max|\mathrm{eig}(I-J)|$")
    ax[0].set_title("measured interface operator of the shipped map")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar([str(r["dt"]) for r in jac["rows"]],
              [r["count_measured"] for r in jac["rows"]], width=0.35,
              label="measured (production Aitken)")
    ax[1].bar([str(r["dt"]) for r in jac["rows"]],
              [r["count_predicted"] for r in jac["rows"]], width=0.2,
              color="C1", alpha=0.8, label="v12 lagged-minres model")
    ax[1].set_ylabel("sub-iterations at $10^{-3}$ relative tolerance")
    ax[1].set_title("model vs shipped loop")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    gain = gain_stage()
    bound = bound_stage()
    jac = jacobian_stage()
    figure_gain(gain["rows"], os.path.join(OUT, "xval_aitken_v14_gain.png"))
    figure_jacobian(jac, os.path.join(OUT, "xval_aitken_v14_jacobian.png"))
    payload = {
        "anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
        "version": 14,
        "lattice": list(LATTICE), "n_elements": N_ELEMENTS,
        "amplitude": AMPLITUDE,
        "gain_stage": gain, "bound_stage": bound, "jacobian_stage": jac,
        "figures": ["xval_aitken_v14_gain.png", "xval_aitken_v14_jacobian.png"],
    }
    with open(os.path.join(OUT, "xval_aitken_interface_gain_scaling.json"), "w") as fh:
        json.dump(payload, fh, indent=1)

    print("dt      rho_s mode       mu        fit      r0")
    for r in gain["rows"]:
        print("%7.1e %5.1f %-9s %9.6f %8.1e %8.1e"
              % (r["dt"], r["rho_s"], r["velocity_mode"], r["mu"],
                 r["fit_error"], r["r0"]))
    print("log-log slopes:", gain["loglog_slope"])
    print("bounds:", [(round(r["dt"], 4), round(r["bound"], 6)) for r in bound["rows"]])
    for r in jac["rows"]:
        print("JAC dt=%g  |J|2=%.3e  rho(I-J)=%.6f  kappa=%.3g  "
              "count pred/meas = %d/%d" % (r["dt"], r["jac_norm2"],
                                           r["spec_abs_max"], r["kappa_M"],
                                           r["count_predicted"],
                                           r["count_measured"]))
    print("artifacts ->", OUT)


if __name__ == "__main__":
    main()