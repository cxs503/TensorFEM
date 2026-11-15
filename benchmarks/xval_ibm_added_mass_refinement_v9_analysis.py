#!/usr/bin/env python3
"""v9: close the v8 open item -- does the implicit-IBM added-mass excess have a
finite limit or vanish as ``dx -> 0``?

The v8 report (m21392) measured ``C = 2.79`` at ``dx = 1`` (c_phys = 4, box 64,
calibrated protocol) and explicitly left the refinement points ``dx = 1/2, 1/4``
and the body-size sweep ``c = 4, 8, 16`` as cluster jobs ``cj3404``/``cj3405``
"still queued", with the resolution dependence **open**.  Those jobs have since
succeeded; this driver reads their artefacts and answers the question:

* refinement axis (fixed ``c_phys = 4``, ``dx -> 0``): ``C`` vs ``dx``
* body-size axis (fixed ``dx = 1``, ``dx/c -> 0``): ``C`` vs ``1/c``

and reconciles the two against the thin-plate reference ``C = 1`` (Lamb 1932
art. 71, ``m_a = rho pi c^2``) and the BEM thickness table of v8.

Every number below is injected from the JSON artefacts; nothing is hand-typed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "xval_ibm_added_mass_refinement_v8_artifacts"
OUT = HERE / "xval_ibm_added_mass_refinement_v9_artifacts"
OUT.mkdir(exist_ok=True)

TOL = 0.15          # owner acceptance: < 15 %
TOL_TIGHT = 0.03    # "is the key quantity within 3 %" question


def load(tag):
    return json.loads((ART / f"{tag}.json").read_text())


def linfit(x, y):
    """Least-squares line y = a + b x; returns (a, b, rms, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * x
    rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss if ss > 0 else 1.0
    return a, b, rms, r2


def main():
    # ---- refinement axis: fixed c_phys=4, box=64, dx -> 0 -----------------
    ref = [load(f"clean_refine_{r}") for r in (1, 2, 4)]
    dx = np.array([r["dx"] for r in ref])
    Cm = np.array([r["C_momentum"] for r in ref])
    Cf = np.array([r["C_force"] for r in ref])
    # linear in dx (kernel support is a fixed number of cells -> smearing ~ dx)
    a_m, b_m, rms_m, r2_m = linfit(dx, Cm)
    a_f, b_f, rms_f, r2_f = linfit(dx, Cf)
    # also try quadratic in dx for a curvature check
    A = np.vstack([np.ones_like(dx), dx, dx ** 2]).T
    (q0, q1, q2), *_ = np.linalg.lstsq(A, Cm, rcond=None)
    limit_lin_m = a_m          # dx -> 0
    limit_quad_m = q0
    # ---- body-size axis: fixed dx=1, c grows (dx/c -> 0) ------------------
    cs = [load(f"clean_c_{int(c)}") for c in (4, 8, 16)]
    c = np.array([r["c_phys"] for r in cs])
    Cmc = np.array([r["C_momentum"] for r in cs])
    Cfc = np.array([r["C_force"] for r in cs])
    # linear in 1/c (kernel support fixed cells -> smearing ~ 1/c)
    a_c, b_c, rms_c, r2_c = linfit(1.0 / c, Cmc)
    limit_c_m = a_c            # 1/c -> 0
    # ---- reconcile: both axes should give the same thin-kernel limit ------
    # refinement: C(dx) = C_inf + b*dx ; body: C(1/c) = C_inf + b_c*(1/c)
    # at the common thin-kernel limit (dx->0 AND 1/c->0) both -> C_inf
    # ---- verdict ----------------------------------------------------------
    err_lin = (limit_lin_m - 1.0) / 1.0          # vs thin plate C=1
    err_quad = (limit_quad_m - 1.0) / 1.0
    err_c = (limit_c_m - 1.0) / 1.0
    # finite-configuration error at the coarsest point (dx=1, c=4)
    err_coarse = (Cm[0] - 1.0) / 1.0

    ev = {
        "generated": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "acceptance_tol": TOL,
        "tight_tol": TOL_TIGHT,
        "refinement_axis": {
            "dx": dx.tolist(), "C_momentum": Cm.tolist(),
            "C_force": Cf.tolist(),
            "lin_fit": {"C_inf": a_m, "slope": b_m, "rms": rms_m, "r2": r2_m},
            "quad_fit": {"C_inf": q0, "c1": q1, "c2": q2},
            "limit_lin": limit_lin_m, "limit_quad": limit_quad_m,
        },
        "body_size_axis": {
            "c": c.tolist(), "C_momentum": Cmc.tolist(),
            "C_force": Cfc.tolist(),
            "lin_fit_1overc": {"C_inf": a_c, "slope": b_c,
                               "rms": rms_c, "r2": r2_c},
            "limit": limit_c_m,
        },
        "verdict": {
            "err_lin_limit": err_lin, "err_quad_limit": err_quad,
            "err_c_limit": err_c, "err_coarse_dx1": err_coarse,
            "within_15_lin": bool(abs(err_lin) < TOL),
            "within_15_quad": bool(abs(err_quad) < TOL),
            "within_15_c": bool(abs(err_c) < TOL),
            "within_3_lin": bool(abs(err_lin) < TOL_TIGHT),
            "within_3_quad": bool(abs(err_quad) < TOL_TIGHT),
            "within_3_c": bool(abs(err_c) < TOL_TIGHT),
        },
        "table": {
            "refine": [{"refine": r["refine"], "dx": r["dx"],
                        "C_momentum": r["C_momentum"],
                        "C_force": r["C_force"],
                        "r2": r["r2_momentum"],
                        "n_cell": r["n_cell"]} for r in ref],
            "csize": [{"c_phys": r["c_phys"], "C_momentum": r["C_momentum"],
                       "C_force": r["C_force"], "r2": r["r2_momentum"],
                       "n_cell": r["n_cell"]} for r in cs],
        },
    }
    (OUT / "evidence.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev["verdict"], indent=1))
    print("refinement table:")
    for r in ev["table"]["refine"]:
        print(f"  dx={r['dx']:.3f}  C_mom={r['C_momentum']:.4f}  "
              f"C_force={r['C_force']:.4f}  r2={r['r2']:.4f}")
    print("body-size table:")
    for r in ev["table"]["csize"]:
        print(f"  c={r['c_phys']:.1f}  C_mom={r['C_momentum']:.4f}  "
              f"C_force={r['C_force']:.4f}  r2={r['r2']:.4f}")


if __name__ == "__main__":
    main()
