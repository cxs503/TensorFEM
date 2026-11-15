"""XVAL v12 -- validity domain of the v11 Aitken cost law: one DOF vs the real solver.

A. rank-one map (``R_min == R_max``): the v11 law, re-measured -- ``n = 3`` for every R.
B. multi-mode map with spectral spread: the law breaks; equation (2) of
   ``tensorfem.aitken_multimode`` predicts the cost, ``n = 3`` does not.
C. the production *flexural* solver (30 free dof, ``CorotationalBeam2D``): Aitken
   count vs fixed-relaxation counts, and the interface gain implied by each count
   (the rank-one-ness test of v11's "gain constancy").

Usage::

    python benchmarks/xval_aitken_multimode_v12.py       # tables + JSON + 2 PNGs (~90 s)
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.aitken_multimode import (  # noqa: E402
    beam_step_count,
    closed_form_count,
    gain_from_count,
    multimode_loop,
    scalar_count,
    submerged_flexural_beam,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "xval_aitken_multimode_artifacts")
TOL = 1e-8


def part_a_rank_one():
    rows = []
    for R in (0.1, 0.3, 0.6, 1.0, 2.1284, 4.0, 12.0, 30.0):
        r = multimode_loop([R], tol=TOL, maxit=400)
        rows.append({"R": R, "n_measured": r["n"], "n_closed_form": 3,
                     "n_scalar_law": scalar_count(r["residual_norms"][0],
                                                  0.0, TOL),
                     "q_after_aitken": 1.0 - (1.0 / (1.0 + R)) * (1.0 + R),
                     "final_err": r["final_err"]})
    return rows


def part_b_multimode():
    rows = []
    grids = [(1.0, 1.0), (1.0, 2.0), (1.0, 4.0), (1.0, 12.0), (1.0, 30.0),
             (0.5, 30.0), (2.0, 30.0), (0.1, 30.0)]
    for R_min, R_max in grids:
        modes = np.linspace(R_min, R_max, 4)
        r = multimode_loop(modes, tol=TOL, maxit=400)
        r1 = r["residual_norms"][0]
        cf = closed_form_count(R_min, R_max, r1, TOL)
        # v11's law predicts 3 sub-iterations for a rank-one map, i.e. it
        # predicts the same cost for every spectrum spread
        naif = 3.0
        rows.append({"R_min": R_min, "R_max": R_max,
                     "modes": [float(m) for m in modes],
                     "spread": R_max / R_min,
                     "n_measured": r["n"], "n_closed_form_2": float(cf),
                     "n_predicted_by_v11_law": float(naif),
                     "omega_after_aitken": r["omegas"][1] if len(r["omegas"]) > 1
                     else float("nan"),
                     "omega_star_dominant": 1.0 / (1.0 + R_max),
                     "q_slow_closed_form": (R_max - R_min) / (1.0 + R_max),
                     "residual_norms": r["residual_norms"]})
    return rows


def part_c_real_beam():
    rows = []
    # (i) Aitken on the production flexural solver
    rec = beam_step_count(mode="aitken")
    rows.append({"mode": "aitken", "omega": None, "n_measured": rec["sub_iterations"],
                 "residual": rec["residual"], "converged": rec["converged"],
                 "omega_applied": rec["omega"]})
    # first sub-iteration residual: max_sub_iter = 1 with sub_tol = 0 reports it
    probe = beam_step_count(mode="fixed", omega=1.0, sub_tol=0.0, max_sub_iter=1)
    r1 = probe["residual"]
    rows.append({"mode": "first_residual", "omega": None, "n_measured": 1,
                 "residual": r1})
    for w in (1.0, 0.5, 0.2, 0.05):
        rec = beam_step_count(mode="fixed", omega=w)
        res = rec["residual"]
        R_implied = gain_from_count(rec["sub_iterations"], w, r1, TOL)
        q = 1.0 - w * (1.0 + R_implied) if np.isfinite(R_implied) else float("nan")
        rows.append({"mode": "fixed", "omega": w,
                     "n_measured": rec["sub_iterations"], "residual": res,
                     "converged": rec["converged"],
                     "R_implied_from_count": R_implied,
                     "q_implied": q,
                     "n_closed_form_from_R": scalar_count(r1, q, TOL)
                     if np.isfinite(q) else float("nan")})
    return rows


def figure_cost(rows_a, rows_b, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sp = [r["spread"] for r in rows_b]
    ax.loglog(sp, [r["n_measured"] for r in rows_b], "o-", label="measured (production Aitken)")
    ax.loglog(sp, [r["n_closed_form_2"] for r in rows_b], "x--", label="closed form (eq. 2)")
    ax.axhline(3.0, color="C3", ls=":", label="v11 scalar law (n = 3)")
    ax.set_xlabel(r"spectral spread $R_{\max}/R_{\min}$")
    ax.set_ylabel("sub-iterations to tol = 1e-8")
    ax.set_title("Aitken cost law: rank-one vs multi-mode interface map")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_real_beam(rows_c, path):
    w = [r["omega"] for r in rows_c if r["mode"] == "fixed"]
    n = [r["n_measured"] for r in rows_c if r["mode"] == "fixed"]
    w = [x for x in w]
    ait = [r for r in rows_c if r["mode"] == "aitken"][0]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(w, n, "o-", label="fixed relaxation, production beam")
    ax.loglog(w, [ait["n_measured"]] * len(w), "s", label="Aitken, same step")
    ax.invert_xaxis()
    ax.set_xlabel(r"fixed under-relaxation $\omega$")
    ax.set_ylabel("sub-iterations to tol = 1e-8")
    ax.set_title("30-dof flexural solver: relaxation cost (rho_f = 1, dt = 0.1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    a = part_a_rank_one()
    b = part_b_multimode()
    c = part_c_real_beam()

    print("A. rank-one map (one DOF):")
    for r in a:
        print("   R=%-7.4g n=%-3d closed=%s final_err=%.1e"
              % (r["R"], r["n_measured"], r["n_closed_form"], r["final_err"]))
    print("B. multi-mode map:")
    for r in b:
        print("   R=[%.2f..%.2f] spread=%-7.2f n_meas=%-4d closed=%-5.0f naive=%s"
              % (r["R_min"], r["R_max"], r["spread"], r["n_measured"],
                 r["n_closed_form_2"], r["n_predicted_by_v11_law"]))
    print("C. production flexural beam (30 free dof):")
    for r in c:
        print("   %-7s omega=%-6s n=%-4d res=%.2e R_implied=%s"
              % (r["mode"], r["omega"], r["n_measured"], r["residual"],
                 "%.4f" % r["R_implied_from_count"]
                 if np.isfinite(r.get("R_implied_from_count", np.nan)) else "-"))

    figure_cost(a, b, os.path.join(OUT, "xval_aitken_v12_cost.png"))
    figure_real_beam(c, os.path.join(OUT, "xval_aitken_v12_beam.png"))
    payload = {"tol": TOL, "A_rank_one": a, "B_multimode": b, "C_real_beam": c}
    with open(os.path.join(OUT, "xval_aitken_multimode_v12.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print("written:", OUT)
    return payload


if __name__ == "__main__":
    main()