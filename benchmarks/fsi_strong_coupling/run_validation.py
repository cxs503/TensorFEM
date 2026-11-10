"""Validation driver for the TensorFEM x TensorLBM strong-coupling FSI face.

Cases
-----
``cylinder``  Turek-Hron 2D-2 benchmark: rigid cylinder in the channel at
              Re = 100 (fluid-solver anchor; literature CD = 3.22-3.24,
              CL amplitude ~ 1.0, St = 0.295-0.305).
``structure`` Euler-Bernoulli cantilever: analytic eigenfrequencies and
              Newmark energy conservation, plus the modal-truncation error.
``interface`` Momentum-conserving load projection (force resultant).
``fsi2``      Full two-way coupled Turek-Hron FSI-2 run: tip displacement,
              force coefficients, sub-iteration convergence and the energy
              ledger.

Usage
-----
    python benchmarks/fsi_strong_coupling/run_validation.py --case all
    python benchmarks/fsi_strong_coupling/run_validation.py --case fsi2 \
        --cells 20 --u-lat 0.05 --duration 3.0

Every case writes a JSON file (numbers only, no hand-copied values) into
``artifacts/fsi_strong_coupling/`` and CSV time histories where applicable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.fsi_lbm_strong_coupling import (  # noqa: E402
    CantileverBeamFE,
    FSI2Parameters,
    ModalCantileverBeam,
    StrongCouplingFSI2,
    _dominant_frequency,
)
from tensorfem.lbm_channel2d import ChannelConfig, ChannelLBM  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "fsi_strong_coupling")
os.makedirs(OUT, exist_ok=True)

# Literature anchors used for the verdicts (see docs/reports/).
REF_2D2 = {"CD": 3.23, "CL_amp": 1.00, "St": 0.300,
           "CD_range": (3.22, 3.24), "St_range": (0.295, 0.305)}


def _write(name: str, payload: dict) -> str:
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Case 1: rigid cylinder, Turek-Hron 2D-2
# ---------------------------------------------------------------------------

def run_cylinder(cells: int = 20, u_lat: float = 0.1, n_cycles: float = 25.0,
                 n_steps_override: int | None = None) -> dict:
    p = FSI2Parameters()
    dx = p.D / cells
    dt = u_lat * dx / p.U_mean
    nu_lat = p.nu_f * dt / dx ** 2
    nx = int(round(p.W / dx))
    ny = int(round(p.H / dx))
    cfg = ChannelConfig(nx=nx, ny=ny, u_mean=u_lat, nu=nu_lat,
                        cylinder_center=(p.cx / dx, p.cy / dx),
                        cylinder_radius=0.5 * p.D / dx,
                        perturbation=0.02 * u_lat)
    solver = ChannelLBM(cfg)
    t_cycle = p.D / (0.3 * p.U_mean)          # rough shedding period [s]
    n_steps = int(max(4000, n_cycles * t_cycle / dt))
    if n_steps_override:
        n_steps = int(n_steps_override)
    hist = np.zeros((n_steps, 2))
    t0 = time.time()
    for k in range(n_steps):
        solver.step()
        hist[k] = solver.momentum_exchange_force()
    runtime = time.time() - t0
    q = 0.5 * p.rho_f * p.U_mean ** 2 * p.D
    cd = 2.0 * hist[:, 0] / (u_lat ** 2 * cells)      # lattice coefficient
    cl = 2.0 * hist[:, 1] / (u_lat ** 2 * cells)
    half = n_steps // 2
    cd_m = float(cd[half:].mean())
    cl_amp = float(0.5 * (cl[half:].max() - cl[half:].min()))
    f = _dominant_frequency(np.arange(n_steps) * dt, cl[half:])
    st = f * p.D / p.U_mean
    return {
        "case": "cylinder_2d2",
        "cells_per_D": cells,
        "u_lat": u_lat,
        "dt": dt,
        "nu_lat": nu_lat,
        "n_steps": n_steps,
        "physical_time": n_steps * dt,
        "runtime_s": runtime,
        "ms_per_step": 1000.0 * runtime / n_steps,
        "CD_mean": cd_m,
        "CL_amp": cl_amp,
        "St": float(st),
        "shedding_frequency_Hz": float(f),
        "reference": REF_2D2,
        "error_CD": abs(cd_m - REF_2D2["CD"]) / REF_2D2["CD"],
        "error_CL": abs(cl_amp - REF_2D2["CL_amp"]) / REF_2D2["CL_amp"],
        "error_St": abs(st - REF_2D2["St"]) / REF_2D2["St"],
        "force_history": {"t": (np.arange(n_steps) * dt).tolist(),
                          "CD": cd.tolist(), "CL": cl.tolist()},
    }


# ---------------------------------------------------------------------------
# Case 2: structure verification
# ---------------------------------------------------------------------------

def run_structure() -> dict:
    p = FSI2Parameters()
    beam = CantileverBeamFE(p.L, 14, p.EI, p.mass_per_length)
    dt = 2.5e-4
    beam.initialise_effective(dt)
    # analytic first frequency
    beta1 = 1.875104068711961
    f_analytic = beta1 ** 2 * math.sqrt(p.EI / p.mass_per_length) / p.L ** 2 / (2 * math.pi)
    # excite the first mode (smooth initial condition -> the energy check is
    # not polluted by unresolved high-frequency content)
    tip_dof = 2 * beam.n_elem
    mb = ModalCantileverBeam(p.L, beam.n_elem, p.EI, p.mass_per_length, n_modes=4)
    q0 = np.zeros_like(beam.q)
    q0[beam.free] = mb.Phi @ np.array([0.01, 0.0, 0.0, 0.0])
    beam.set_state(q0, np.zeros_like(q0), np.zeros_like(q0))
    e0 = beam.kinetic_energy() + beam.strain_energy()
    hist = []
    n_steps = int(3.0 / dt)
    for k in range(n_steps):
        beam.step(np.zeros_like(beam.q))
        if k % 20 == 0:
            hist.append((k * dt, float(beam.q[tip_dof])))
    e1 = beam.kinetic_energy() + beam.strain_energy()
    hist = np.array(hist)
    f_num = _dominant_frequency(hist[:, 0], hist[:, 1])
    # modal truncation error: static tip deflection under a uniform load
    F = np.zeros_like(beam.q)
    for e in range(beam.n_elem):
        F[2 * e] += 1.0 * beam.le / 2
        F[2 * e + 2] += 1.0 * beam.le / 2
    static_full = np.linalg.solve(beam.Kff, F[beam.free])
    errs = {}
    for n_modes in (3, 5, 8, 12):
        mb = ModalCantileverBeam(p.L, 14, p.EI, p.mass_per_length, n_modes=n_modes)
        eta = np.linalg.solve(np.diag(mb.omega2), mb.Phi.T @ F[mb.free])
        q_free = mb.Phi @ eta
        errs[str(n_modes)] = float(abs(q_free[-2] - static_full[-2]) / abs(static_full[-2]))
    return {
        "case": "structure",
        "analytic_f1_Hz": float(f_analytic),
        "numerical_f1_Hz": float(f_num),
        "frequency_error": float(abs(f_num - f_analytic) / f_analytic),
        "energy_initial_J_per_m": float(e0),
        "energy_final_J_per_m": float(e1),
        "energy_drift": float(abs(e1 - e0) / e0),
        "static_tip_modal_error": errs,
    }


# ---------------------------------------------------------------------------
# Case 3: interface conservation
# ---------------------------------------------------------------------------

def run_interface() -> dict:
    p = FSI2Parameters()
    fsi = StrongCouplingFSI2(p, cells_per_D=20, u_lat=0.05)
    itf = fsi.interface
    # arbitrary smooth load field, zero outside the beam columns
    rng = np.random.default_rng(0)
    fmap = np.zeros((2, fsi.ny, fsi.nx))
    fmap[1][:, itf.cols] = rng.normal(size=(fsi.ny, itf.cols.size)) * 0.5
    b, C, n_links, full_sum = itf.exchange_terms(
        fsi.beam.q, np.zeros_like(fmap[1]), fmap, 1.0)
    # the projection uses the shape functions whose sum is one:
    f_column = fmap[1][:, itf.cols].sum(axis=0)
    total_column = float(f_column.sum())
    total_nodes = float(b.sum())
    return {
        "case": "interface",
        "loaded_columns": int(itf.cols.size),
        "loaded_cells": int(np.count_nonzero(fmap[1][:, itf.cols])),
        "resultant_columns": total_column,
        "resultant_nodes_full_translation": full_sum,
        "resultant_nodes_free": total_nodes,
        "force_conservation_error": float(abs(full_sum - total_column)
                                          / max(abs(total_column), 1e-30)),
        "clamp_reaction_share": float(abs(total_column - b[0::2].sum())
                                      / max(abs(total_column), 1e-30)),
        "n_links": n_links,
    }


# ---------------------------------------------------------------------------
# Case 4: coupled FSI-2
# ---------------------------------------------------------------------------

def run_fsi2(cells: int = 20, u_lat: float = 0.05, duration: float = 2.0,
             n_modes: int = 5, record_every: int = 2,
             max_sub: int = 12, zeta: float = 0.0) -> dict:
    p = FSI2Parameters()
    fsi = StrongCouplingFSI2(p, cells_per_D=cells, u_lat=u_lat, n_modes=n_modes,
                             max_sub_iterations=max_sub, tolerance=1e-4,
                             zeta=zeta)
    n_steps = int(round(duration / fsi.dt))
    t0 = time.time()
    hist = fsi.run(n_steps, record_every=record_every)
    runtime = time.time() - t0
    transient = 0.55 * duration
    try:
        metrics = fsi.analyse(transient=transient)
    except RuntimeError:
        metrics = {"frequency": float("nan"), "amplitude": float("nan"),
                   "mean_CD": float("nan"), "amp_CL": float("nan")}
    h = hist
    sub = np.asarray(h.sub_iterations)
    fres = np.asarray(h.force_residual)

    # ---- energy ledger over the periodic window ---------------------------
    t = np.asarray(h.time)
    sel = t >= transient
    dt_rec = float(np.mean(np.diff(t))) if t.size > 1 else fsi.dt
    e_tot = np.asarray(h.fluid_ke) + np.asarray(h.struct_ke) + np.asarray(h.struct_pe)
    p_in = np.asarray(h.inlet_power)
    p_out = np.asarray(h.outlet_power)
    p_visc = np.asarray(h.viscous_dissipation)
    p_int = np.asarray(h.interface_power)
    dE = float(e_tot[sel][-1] - e_tot[sel][0])
    work_in = float((p_in[sel].sum() - p_out[sel].sum()) * dt_rec)
    visc = float(p_visc[sel].sum() * dt_rec)
    w_int = float(p_int[sel].sum() * dt_rec)
    # fluid-side interface work: momentum handed over by the boundary cells
    # equals the structural work plus the coupling error
    residual_rel = abs(dE - (work_in - visc)) / max(abs(work_in), 1e-30)

    payload = {
        "case": "fsi2",
        "cells_per_D": cells,
        "u_lat": u_lat,
        "dt": fsi.dt,
        "nu_lat": fsi.nu_lat,
        "n_steps": n_steps,
        "runtime_s": runtime,
        "ms_per_step": 1000.0 * runtime / max(n_steps, 1),
        "n_modes": n_modes,
        "zeta": zeta,
        "mean_sub_iterations": float(sub.mean()),
        "max_sub_iterations": int(sub.max()),
        "converged_fraction": float(np.mean(fres < 1e-4)),
        "force_residual_median": float(np.median(fres[np.isfinite(fres)])),
        "frequency_Hz": metrics.get("frequency"),
        "amplitude_m": metrics.get("amplitude"),
        "mean_tip_m": metrics.get("mean_tip"),
        "mean_CD": metrics.get("mean_CD"),
        "amp_CL": metrics.get("amp_CL"),
        "mean_CL": metrics.get("mean_CL"),
        "window": metrics.get("window"),
        "reference": {"frequency_Hz": p.ref_frequency,
                      "amplitude_m": p.ref_amplitude},
        "error_frequency": (abs(metrics.get("frequency", 0.0) - p.ref_frequency)
                            / p.ref_frequency),
        "error_amplitude": (abs(metrics.get("amplitude", 0.0) - p.ref_amplitude)
                            / p.ref_amplitude),
        "ledger": {
            "delta_E_J_per_m": dE,
            "work_in_J_per_m": work_in,
            "viscous_J_per_m": visc,
            "interface_work_J_per_m": w_int,
            "residual_relative": float(residual_rel),
            "window_s": [float(t[sel][0]), float(t[sel][-1])] if sel.any() else None,
        },
        "history": {
            "t": t.tolist(),
            "tip": np.asarray(h.tip_displacement).tolist(),
            "drag": np.asarray(h.drag).tolist(),
            "lift": np.asarray(h.lift).tolist(),
            "fluid_ke": np.asarray(h.fluid_ke).tolist(),
            "struct_ke": np.asarray(h.struct_ke).tolist(),
            "struct_pe": np.asarray(h.struct_pe).tolist(),
            "inlet_power": p_in.tolist(),
            "outlet_power": p_out.tolist(),
            "viscous": p_visc.tolist(),
            "interface_power": p_int.tolist(),
            "sub_iterations": sub.tolist(),
            "force_residual": fres.tolist(),
        },
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    choices=["all", "cylinder", "structure", "interface", "fsi2"])
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--u-lat", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--n-modes", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--zeta", type=float, default=0.0)
    args = ap.parse_args(argv)

    suffix = f"_{args.tag}" if args.tag else ""
    results = {}
    if args.case in ("all", "structure"):
        results["structure"] = run_structure()
        _write(f"structure{suffix}.json", results["structure"])
        print("structure:", json.dumps(results["structure"])[:400])
    if args.case in ("all", "interface"):
        results["interface"] = run_interface()
        _write(f"interface{suffix}.json", results["interface"])
        print("interface:", json.dumps(results["interface"])[:400])
    if args.case in ("all", "cylinder"):
        for cells in (12, 20):
            res = run_cylinder(cells=cells, n_steps_override=args.n_steps)
            _write(f"cylinder_D{cells}{suffix}.json", res)
            print(f"cylinder D={cells}: CD={res['CD_mean']:.3f} "
                  f"CLamp={res['CL_amp']:.3f} St={res['St']:.3f}")
    if args.case in ("all", "fsi2"):
        res = run_fsi2(cells=args.cells, u_lat=args.u_lat,
                       duration=args.duration, n_modes=args.n_modes,
                       zeta=args.zeta)
        _write(f"fsi2_D{args.cells}_u{args.u_lat}{suffix}.json", res)
        print("fsi2:", {k: res[k] for k in
                        ("frequency_Hz", "amplitude_m", "mean_CD", "amp_CL",
                         "mean_sub_iterations", "converged_fraction")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())