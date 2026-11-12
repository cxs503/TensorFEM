#!/usr/bin/env python3
"""m15654 — focused re-derivation of the `poisson_crosscheck` gate (finding F-CRIT).

Question: gate `poisson_crosscheck` reports rel_error = 0.0. Where does the "measured"
ratio come from -- computed tip deflections, or the two closed-form references?

  benchmarks/verified/fem_bending_patch/run.py:542
      meas = fre[0]["w_exact_kirchhoff"] / sym[0]["w_exact_kirchhoff"]

Both operands are closed-form (M L^2 / 2D_cyl and M L^2 / 2D_beam), so the ratio is
algebraically 1/(1-nu^2) and the gate is satisfied by construction: it can never see a
Poisson-coupling defect in the element.

Real cross-check = ratio of the *computed* tip deflections. Quantified below for every
mesh and both shear treatments, plus the free-edge error convergence order.
"""
from __future__ import annotations

import json
import math
import pathlib

RES = pathlib.Path("/root/tp-agent/work/m15654/benchmarks/verified/fem_bending_patch/result.json")
OUT = pathlib.Path(__file__).resolve().parent / "poisson_crosscheck_rederived.json"
POOL_TOL = 0.02

r = json.loads(RES.read_text())
strip = r["strip"]
theory = 1.0 / (1.0 - r["material"]["poisson"] ** 2)

rows = []
for t in sorted({x["thickness"] for x in strip}):
    for treat in ("sri", "mitc"):
        for n in sorted({x["n_x"] for x in strip}):
            sym = next((x for x in strip if x["thickness"] == t and x["lateral"] == "symmetry"
                        and x["shear_treatment"] == treat and x["n_x"] == n
                        and x["case"].startswith("strip/")), None)
            fre = next((x for x in strip if x["thickness"] == t and x["lateral"] == "free"
                        and x["shear_treatment"] == treat and x["n_x"] == n
                        and x["case"].startswith("strip/")), None)
            if not (sym and fre):
                continue
            meas = fre["w_tip"] / sym["w_tip"]
            rows.append({
                "t": t, "shear": treat, "n_x": n,
                "w_free": fre["w_tip"], "w_symmetry": sym["w_tip"],
                "ratio_computed": meas, "ratio_theory": theory,
                "rel_error_computed": abs(meas - theory) / theory,
                "gate_value_declared": r["strip_ratio"]["rel_error"],
                "gate_still_passes": abs(meas - theory) / theory <= POOL_TOL,
            })

# free-edge error convergence: e(n) vs h = 1/n
free_err = {}
for treat in ("sri", "mitc"):
    pts = [(1.0 / x["n_x"], abs(x["w_tip"] - x["w_exact_kirchhoff"]) / x["w_exact_kirchhoff"])
           for x in strip if x["lateral"] == "free" and x["shear_treatment"] == treat
           and x["thickness"] == 0.01 and x["case"].startswith("strip/")]
    pts.sort(reverse=True)
    lx = [math.log(p[0]) for p in pts]
    ly = [math.log(p[1]) for p in pts]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    s = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    free_err[treat] = {"points_h_vs_err": pts, "fitted_order": s,
                       "err_to_theory_ratio": pts[0][1] / pts[-1][1]}

out = {
    "gate": "poisson_crosscheck",
    "declared_value": r["strip_ratio"],
    "declared_operand_source": "run.py:542 divides w_exact_kirchhoff(free) by "
                               "w_exact_kirchhoff(symmetry) -- both closed-form",
    "finding": "F-CRIT: the gate is algebraically satisfied by construction (0.0 to round-off);"
               " it never touches a computed number, so it cannot detect a Poisson-coupling defect",
    "recomputed_from_computed_tip_deflections": rows,
    "max_computed_rel_error": max(x["rel_error_computed"] for x in rows),
    "all_computed_checks_still_within_tolerance": all(x["gate_still_passes"] for x in rows),
    "free_edge_error_convergence": free_err,
    "remediation": "replace the two w_exact_kirchhoff operands by w_tip (computed); then the gate"
                   " value becomes max_computed_rel_error above (still < 2%, no physics changes)",
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("declared:", r["strip_ratio"])
for x in rows:
    print(f"t={x['t']} {x['shear']} n={x['n_x']:>2} ratio={x['ratio_computed']:.6f} "
          f"rel_err={x['rel_error_computed']:.4e} pass={x['gate_still_passes']}")
print("max computed rel err:", out["max_computed_rel_error"])
print("free-edge order:", {k: round(v["fitted_order"], 3) for k, v in free_err.items()})
print("wrote", OUT)