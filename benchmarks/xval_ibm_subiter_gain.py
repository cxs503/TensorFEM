#!/usr/bin/env python3
"""XVAL v13 -- does the implicit-IBM sub-iteration actually iterate?

Anchor: ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (joconnor22/LIFE recipe).
The v1-v12 rounds of this anchor validated the IBM operators, the Aitken
recursion on synthetic fixed points and the spectral law of the *model* map,
and v12 closed with a **negative** row: the multi-DOF production driver showed
interface gain ``mu ~ 1`` (``R < 1e-4``), so ``WITHHELD``.  This round settles
*why*, and turns the one-off observation into a measurement over the whole
driver family plus a reusable gate.

Stages
------
``gain``  interface gain ``mu`` of the production beam driver
          (:class:`tensorfem.ibm_aitken_implicit.ImplicitIBMFSI`) read off the
          sub-iteration residual trace at two fixed relaxation factors, over a
          sweep of ``rho_f``, ``rho_s`` and marker resolution.  ``omega = 1`` is
          then as good as any under-relaxed choice iff ``mu - 1 < INERT_TOL``.
``bound`` what the gain *should* be in each case: the frozen-fluid discrete
          bound ``|dG/du|`` (:func:`tensorfem.ibm_subiter_gain.discrete_gain_bound`,
          from the marker-local fluid mass ``M_f``) and the added-mass ratio
          ``m_a/m_s`` with Lamb's ``m_a = rho pi c^2``.
``relax`` the answer's independence of the relaxation: Aitken, fixed
          ``omega = 1`` and fixed ``omega = 0.05`` over the same physics -- tip
          histories, sub-iteration counts and applied ``omega``.
``fig``   measured gain vs the added-mass prediction, plus the residual traces.

    python3 benchmarks/xval_ibm_subiter_gain.py --stage all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.ibm_aitken_implicit import (  # noqa: E402
    BeamProperties, CorotationalBeam2D, ImplicitIBMFSI, LBMFluid2D, _section,
    _estimate_period,
)
from tensorfem.ibm_subiter_gain import (  # noqa: E402
    INERT_TOL, added_mass_ratio, discrete_gain_bound, gain_from_trace,
    marker_local_mass,
)

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_ibm_subiter_gain_artifacts")
DEFAULT = dict(lattice=(96, 72), n_elements=16, dt=2.0e-3, amplitude=0.05)


def build(fluid_rho=1.0, rho_s=1.0, lattice=DEFAULT["lattice"],
          n_elements=DEFAULT["n_elements"], dt=DEFAULT["dt"],
          amplitude=DEFAULT["amplitude"]):
    """Beam scaled into the lattice box, exactly as the v6 submerged case."""
    sec = _section()
    beam = CorotationalBeam2D(n_elements=n_elements,
                             props=BeamProperties(E=sec.E, A=sec.A, I=sec.I,
                                                  rho=rho_s, length=sec.length,
                                                  thickness=sec.thickness))
    scale = 0.45 * float(lattice[0])
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amplitude * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    return beam, fluid, scale


def measure_gain(fluid_rho=1.0, rho_s=1.0, omegas=(1.0, 0.5, 0.25), n_iter=3, **kw):
    """Gain of the production driver: residual traces at fixed relaxation."""
    traces, extras = [], []
    for w in omegas:
        beam, fluid, scale = build(fluid_rho=fluid_rho, rho_s=rho_s, **kw)
        solver = ImplicitIBMFSI(beam, fluid, dt=kw.get("dt", DEFAULT["dt"]), dx=1.0,
                                ref_length=beam.props.length, sub_tol=0.0,
                                max_sub_iter=n_iter, relax_mode="fixed",
                                fixed_omega=w)
        solver.step(1)          # sub_tol = 0 -> the full n_iter sweeps run
        traces.append(solver.residual_traces[-1])
        rho, u = solver.fluid.macroscopic(include_force=False)
        rho_i, _ = solver.ibm.interpolate(rho, u)
        extras.append({"M_f": marker_local_mass(solver.ibm, rho_i),
                       "M_s": float(beam.M.sum()),
                       "c": 0.5 * float(beam.props.thickness),
                       "ref_length": float(beam.props.length)})
    est = gain_from_trace(traces, omegas=omegas)
    est.update(extras[0])
    est["traces"] = [list(map(float, t)) for t in traces]
    est["omegas"] = list(omegas)
    return est


def measure_physics(fluid_rho=1.0, rho_s=1.0, n_steps=160, relax_mode="aitken",
                    fixed_omega=1.0, **kw):
    """Coupled (or dry) free vibration under a chosen relaxation scheme."""
    beam, fluid, scale = build(fluid_rho=fluid_rho, rho_s=rho_s, **kw)
    dt = kw.get("dt", DEFAULT["dt"])
    if fluid_rho <= 0.0:
        tips = []
        for k in range(1, n_steps + 1):
            beam.u = beam.predict(k)
            beam.newmark_solve(np.zeros(beam.n_dof), dt)
            beam.advance_history()
            tips.append(beam.tip()[1])
        return {"tips": tips, "period": _estimate_period(tips, dt),
                "sub_iterations": [], "omegas": [], "maxit_hits": 0,
                "diverged": False, "reason": ""}
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0,
                            ref_length=beam.props.length, sub_tol=1e-6,
                            max_sub_iter=30, relax_mode=relax_mode,
                            fixed_omega=fixed_omega)
    hist = solver.run(n_steps, t_start=1)
    tips = [h["tip"][1] for h in hist]
    return {"tips": tips, "period": _estimate_period(tips, dt),
            "sub_iterations": [h["sub_iterations"] for h in hist],
            "omegas": [h["omega"] for h in hist],
            "maxit_hits": solver.n_maxit_hits,
            "diverged": bool(getattr(solver, "diverged", False)),
            "reason": getattr(solver, "diverged_reason", "")}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_gain():
    rows = []
    for rho_f in (0.05, 0.2, 1.0, 2.0):
        for rho_s in (0.2, 1.0, 5.0):
            g = measure_gain(fluid_rho=rho_f, rho_s=rho_s)
            ref = added_mass_ratio(rho_f, g["c"], g["M_s"])
            bnd = discrete_gain_bound(g["M_f"], g["M_s"], k=0.0,
                                      dt=DEFAULT["dt"])
            rows.append({
                "rho_f": rho_f, "rho_s": rho_s, "mu_measured": g["mu"],
                "mu_minus_1": g["mu"] - 1.0, "inert": bool(abs(g["mu"] - 1.0) < INERT_TOL),
                "M_f": g["M_f"], "M_s": g["M_s"], "c": g["c"],
                "discrete_bound": bnd, "m_a_lamb": ref["m_a"],
                "added_mass_ratio": ref["ratio"], "mu_added_mass": ref["mu_predicted"],
                "traces": g["traces"], "omegas": g["omegas"],
                "method": g["method"], "fit_error": g["fit_error"],
                "ratios": g["ratios"],
            })
    for rho_f, rho_s in [(1.0, 1.0)]:
        for lattice, n_el in [((64, 48), 12), ((128, 96), 20)]:
            g = measure_gain(fluid_rho=rho_f, rho_s=rho_s, lattice=lattice,
                             n_elements=n_el)
            ref = added_mass_ratio(rho_f, g["c"], g["M_s"])
            rows.append({
                "rho_f": rho_f, "rho_s": rho_s, "mu_measured": g["mu"],
                "mu_minus_1": g["mu"] - 1.0, "inert": bool(abs(g["mu"] - 1.0) < INERT_TOL),
                "M_f": g["M_f"], "M_s": g["M_s"], "c": g["c"],
                "discrete_bound": discrete_gain_bound(g["M_f"], g["M_s"],
                                                      dt=DEFAULT["dt"]),
                "m_a_lamb": ref["m_a"], "added_mass_ratio": ref["ratio"],
                "mu_added_mass": ref["mu_predicted"], "traces": g["traces"],
                "omegas": g["omegas"], "method": g["method"],
                "fit_error": g["fit_error"], "ratios": g["ratios"],
                "lattice": list(lattice), "n_elements": n_el,
            })
    return rows


def stage_relax():
    out = {}
    for tag, mode, w in [("aitken", "aitken", 1.0), ("fixed1", "fixed", 1.0),
                         ("fixed005", "fixed", 0.05)]:
        r = measure_physics(fluid_rho=1.0, rho_s=1.0, n_steps=160,
                            relax_mode=mode, fixed_omega=w)
        out[tag] = {"period": r["period"],
                    "frequency": 1.0 / r["period"] if r["period"] == r["period"] else float("nan"),
                    "sub_iterations": r["sub_iterations"],
                    "omegas": r["omegas"], "maxit_hits": r["maxit_hits"],
                    "diverged": r["diverged"], "reason": r["reason"],
                    "tips": r["tips"]}
    base = np.asarray(out["aitken"]["tips"], dtype=float)
    for tag in ("fixed1", "fixed005"):
        other = np.asarray(out[tag]["tips"], dtype=float)
        n = min(len(base), len(other))
        out[tag]["dev_vs_aitken"] = float(np.max(np.abs(base[:n] - other[:n])))
        denom = max(float(np.max(np.abs(base[:n]))), 1e-300)
        out[tag]["rel_dev_vs_aitken"] = out[tag]["dev_vs_aitken"] / denom
    return out


def stage_physics_gain():
    """Documented added-mass ratio from the wet/dry frequency shift (v6 chain)."""
    rows = []
    for rho_f in (0.2, 1.0):
        dry = measure_physics(fluid_rho=0.0, n_steps=160)
        wet = measure_physics(fluid_rho=rho_f, n_steps=160)
        if not (dry["period"] == dry["period"] and wet["period"] == wet["period"]):
            rows.append({"rho_f": rho_f, "note": "period estimate degenerate"})
            continue
        r = (dry["period"] / wet["period"]) ** 2 - 1.0
        rows.append({"rho_f": rho_f, "f_dry": 1.0 / dry["period"],
                     "f_wet": 1.0 / wet["period"],
                     "R_from_period": r, "mu_from_period": 1.0 + r,
                     "sub_iterations_mean": float(np.mean(wet["sub_iterations"]))
                     if wet["sub_iterations"] else float("nan"),
                     "omega_mean": float(np.nanmean(wet["omegas"]))
                     if wet["omegas"] else float("nan")})
    return rows


def stage_fig(rows, relax, phys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ART, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    pred = [r["mu_added_mass"] - 1.0 for r in rows]
    meas = [max(r["mu_minus_1"], 1e-16) for r in rows]
    ax.scatter(pred, meas, s=42, label="shipped beam driver")
    bnd = [r["discrete_bound"] for r in rows]
    ax.scatter(pred, [max(b, 1e-16) for b in bnd], marker="^", s=36,
               label="frozen-fluid discrete bound")
    ax.axhline(INERT_TOL, color="k", ls="--", lw=1,
               label=f"inert tol {INERT_TOL:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"added-mass prediction $\mu-1 = m_a/m_s$")
    ax.set_ylabel(r"measured $\mu-1$")
    ax.set_title("sub-iteration gain: measured vs added-mass prediction")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for r in rows[:4]:
        ax.semilogy(range(len(r["traces"][0])), r["traces"][0], "o-",
                    label=f"rho_f={r['rho_f']}, rho_s={r['rho_s']}")
    ax.set_xlabel("sub-iteration")
    ax.set_ylabel("interface residual (dimensionless)")
    ax.set_title(r"residual trace at $\omega=1$ (no relaxation)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    t = np.arange(len(relax["aitken"]["tips"])) * DEFAULT["dt"]
    for tag, lab in [("aitken", "Aitken"), ("fixed1", r"fixed $\omega=1$"),
                     ("fixed005", r"fixed $\omega=0.05$")]:
        ax.plot(t, relax[tag]["tips"], lw=1.4, label=lab)
    ax.set_xlabel("time")
    ax.set_ylabel("tip deflection")
    ax.set_title("coupled answer vs relaxation scheme")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(ART, "xval_ibm_subiter_gain.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gain", "relax", "physics", "fig", "all"])
    a = ap.parse_args(argv)
    os.makedirs(ART, exist_ok=True)
    res, path = {}, {}
    if a.stage in ("gain", "all"):
        res["gain"] = stage_gain()
    if a.stage in ("relax", "all"):
        res["relax"] = stage_relax()
    if a.stage in ("physics", "all"):
        res["physics"] = stage_physics_gain()
    if a.stage == "all":
        path["fig"] = stage_fig(res["gain"], res["relax"], res["physics"])
        with open(os.path.join(ART, "summary.json"), "w") as fh:
            json.dump({"gain": res["gain"], "physics": res["physics"],
                       "relax": {k: {kk: vv for kk, vv in v.items() if kk != "tips"}
                                 for k, v in res["relax"].items()},
                       "figure": path["fig"]}, fh, indent=2, default=float)
    if "gain" in res:
        print("rho_f rho_s  mu_meas  mu-1      inert  bound     mu_am  fit_err")
        for r in res["gain"]:
            print(f"{r['rho_f']:5.2f} {r['rho_s']:5.2f}  {r['mu_measured']:.6f}  "
                  f"{r['mu_minus_1']:9.2e}  {str(r['inert']):5s}  "
                  f"{r['discrete_bound']:8.2e}  {r['mu_added_mass']:.3f}  "
                  f"{r['fit_error']:.1e}")
    if "relax" in res:
        for tag in ("aitken", "fixed1", "fixed005"):
            r = res["relax"][tag]
            print(f"{tag:9s} f={r['frequency']:.6f} sub_it={r['sub_iterations'][:6]} "
                  f"maxit={r['maxit_hits']} dev_vs_aitken={r.get('rel_dev_vs_aitken', 0.0):.2e}")
    if "physics" in res:
        for r in res["physics"]:
            print("physics:", {k: (round(v, 5) if isinstance(v, float) else v)
                               for k, v in r.items()})
    if path:
        print("figure:", path["fig"])
    return res


if __name__ == "__main__":
    main()