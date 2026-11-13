#!/usr/bin/env python3
"""Board audit m17408 — re-check the fluid metrics of m17361's XVAL (predeclared).

Context
-------
m17361 delivered `benchmarks/xval_aitken_ibm/run_xval.py` (untracked in its
worktree) whose section 3 reports, at cells_per_D = 12:

    CD_mean = 6.022   (ref 3.23, err 86.4 %)
    CL_amp  = 0.374   (ref 1.00, err 62.6 %)
    St      = 0.01616 (ref 0.295, err 94.5 %)      -> declared honest FAIL

Two independent formula defects are visible by inspection of that script
(see docs/board_audit_m17361/AUDIT_m17361.md, findings F2/F3).  This script
re-runs the *same* solver configuration and recomputes the three coefficients
with the formulas the rest of the repository already uses, so that the audit
verdict rests on numbers rather than on arithmetic.

Predeclared expectations (written before the run, NOT tuned afterwards)
-----------------------------------------------------------------------
P1  CD with the test-consistent normalisation  CD = F / (0.5 rho U^2 D)
    (== tests/test_fsi_lbm_strong_coupling.py: 2*F/(u^2*cells)) equals
    delivered CD / 2  = 3.011 +- 0.02, i.e. |err| < 15 % vs 3.23 -> PASS.
P2  St with the solver's own lattice->physical step (fsi_lbm_strong_coupling.py
    line 578: dt = u_lat*dx/U_mean) equals delivered St * 15.625
    = 0.2526 +- 0.005, i.e. err ~ -14 % -> PASS (barely).
P3  CL_amp is already normalised consistently in the delivered script, so the
    re-check should reproduce 0.374 within 1 %  -> still FAIL (>15 %).
P4  Refinement cells_per_D 12 -> 20 must move CD and St towards the reference
    monotonically (the report's section 5.2 claims this without evidence).

Falsification: any of P1-P4 failing is reported as such, not papered over.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorfem.fsi_lbm_strong_coupling import (  # noqa: E402
    FSI2Parameters, _dominant_frequency)
from tensorfem.lbm_channel2d import ChannelConfig, ChannelLBM  # noqa: E402

REF = {"CD": 3.23, "CL_amp": 1.0, "St": 0.295}   # same ref dict as run_xval.py
TOL = 0.15                                        # mission criterion
OUT = os.path.join(REPO, "artifacts", "audit_m17361")


def measure(cells: int, n_steps: int, n_skip: int, u_lat: float = 0.08):
    """Run cylinder-in-channel LBM and return raw + restated coefficients."""
    p = FSI2Parameters()
    dx = p.D / cells
    nu_lat = u_lat * cells / 100.0                      # Re = u_lat*cells/nu = 100
    cfg = ChannelConfig(nx=int(p.W / dx), ny=int(p.H / dx), u_mean=u_lat,
                        nu=nu_lat,
                        cylinder_center=(p.cx / dx, p.cy / dx),
                        cylinder_radius=0.5 * p.D / dx,
                        perturbation=0.02 * u_lat)
    s = ChannelLBM(cfg)
    hist = []
    for _ in range(n_steps):
        s.step()
        hist.append(s.momentum_exchange_force())
    hist = np.array(hist)[n_skip:]
    Fx, Fy = hist[:, 0].mean(), hist[:, 1]

    q_dyn = 0.5 * u_lat ** 2 * cells                     # 0.5 rho U^2 D, rho_lat = 1
    # (a) delivered normalisation: extra factor 2 on drag
    CD_delivered = 2.0 * Fx / q_dyn
    # (b) repo/test-consistent normalisation
    CD_fixed = Fx / q_dyn
    CL_amp = 0.5 * (Fy.max() - Fy.min()) / q_dyn

    st = {"cells_per_D": cells, "n_steps": n_steps, "n_averaged": len(hist),
          "Fx_mean_lat": float(Fx), "q_dyn_lat": float(q_dyn),
          "CD_delivered_formula": float(CD_delivered),
          "CD_fixed_formula": float(CD_fixed),
          "CL_amp": float(CL_amp)}

    # (c) Strouhal: delivered time step vs the solver's own time step
    dt_delivered = dx / (u_lat * p.U_mean / p.D)         # run_xval.py:261
    dt_solver = u_lat * dx / p.U_mean                    # fsi_lbm_strong_coupling.py:578
    for tag, dt in (("delivered", dt_delivered), ("solver", dt_solver)):
        t = np.arange(len(hist)) * dt
        f = _dominant_frequency(t, Fy)
        st["St_%s_dt" % tag] = float(f * p.D / p.U_mean)
        st["f_Hz_%s_dt" % tag] = float(f)
    st["dt_delivered_s"] = float(dt_delivered)
    st["dt_solver_s"] = float(dt_solver)
    st["dt_ratio"] = float(dt_delivered / dt_solver)
    st["err_pct"] = {
        "CD_delivered": 100 * (CD_delivered - REF["CD"]) / REF["CD"],
        "CD_fixed": 100 * (CD_fixed - REF["CD"]) / REF["CD"],
        "CL_amp": 100 * (CL_amp - REF["CL_amp"]) / REF["CL_amp"],
        "St_delivered": 100 * (st["St_delivered_dt"] - REF["St"]) / REF["St"],
        "St_fixed": 100 * (st["St_solver_dt"] - REF["St"]) / REF["St"],
    }
    st["pass_15"] = {k: bool(abs(v) < TOL * 100) for k, v in st["err_pct"].items()}
    return st


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = {"ref": REF, "tolerance": TOL,
           "predeclared": ["P1 CD_fixed = CD_delivered/2 ~ 3.011, |err|<15%",
                           "P2 St_fixed = St_delivered*15.625 ~ 0.2526, |err|<15%",
                           "P3 CL_amp reproduced ~0.374, still FAIL",
                           "P4 CD/St improve monotonically 12 -> 20 cells/D"],
           "runs": []}

    # delivered configuration, delivered averaging window (steps 1001..2000)
    res["runs"].append(dict(measure(12, 2000, 1000), tag="delivered_cfg"))
    # same grid, longer window (averaging sensitivity)
    res["runs"].append(dict(measure(12, 4000, 2000), tag="delivered_cfg_long"))
    # refinement (the claim in report section 5.2, previously unsupported)
    res["runs"].append(dict(measure(20, 4000, 2000), tag="refined_D20"))

    r12 = res["runs"][0]
    checks = {
        "P1_CD_fixed_matches_half_delivered": abs(
            r12["CD_fixed_formula"] - r12["CD_delivered_formula"] / 2) < 1e-9,
        "P1_CD_fixed_within_15pct": r12["pass_15"]["CD_fixed"],
        "P2_dt_ratio_15.625": abs(r12["dt_ratio"] - 15.625) < 1e-6,
        "P2_St_fixed_within_15pct": r12["pass_15"]["St_fixed"],
        "P3_CL_amp_unaffected": abs(r12["CL_amp"] - 0.3737092317455427) < 1e-3,
        "P3_CL_amp_still_fails": not r12["pass_15"]["CL_amp"],
    }
    r20 = res["runs"][2]
    checks["P4_CD_moves_towards_ref"] = abs(r20["err_pct"]["CD_fixed"]) < abs(
        r12["err_pct"]["CD_fixed"])
    checks["P4_St_moves_towards_ref"] = abs(r20["err_pct"]["St_fixed"]) < abs(
        r12["err_pct"]["St_fixed"])
    res["checks"] = checks
    res["all_predeclared_hold"] = all(checks.values())

    with open(os.path.join(OUT, "recheck_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    for r in res["runs"]:
        print("== %-18s cells/D=%2d steps=%d avg=%d" % (r["tag"], r["cells_per_D"],
                                                        r["n_steps"], r["n_averaged"]))
        print("   CD delivered=%.3f fixed=%.3f (err %+.1f%%) | CL_amp=%.3f (err %+.1f%%)"
              % (r["CD_delivered_formula"], r["CD_fixed_formula"],
                 r["err_pct"]["CD_fixed"], r["CL_amp"], r["err_pct"]["CL_amp"]))
        print("   St delivered=%.5f fixed=%.5f (err %+.1f%%) | dt ratio=%.3f"
              % (r["St_delivered_dt"], r["St_solver_dt"], r["err_pct"]["St_fixed"],
                 r["dt_ratio"]))
    print("checks:", json.dumps(checks, indent=1))
    print("all_predeclared_hold:", res["all_predeclared_hold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())