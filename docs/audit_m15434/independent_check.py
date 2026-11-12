#!/usr/bin/env python3
"""Independent audit check for CASE-THERMAL-BAR (m15434).

Deliberately does NOT import ``tensorfem``: the point is to reproduce the
reference numbers and the convergence behaviour from first principles, so the
audit cannot inherit a bug from the module under review.

Checks
------
C1  closed-form reference ``sigma_xx = -E alpha dT``: E and alpha are re-derived
    here by a least-squares fit of the repo's own EH36 anchor table (20 .. -100
    degC), then compared with ``reference.value_Pa`` of the evidence JSON.
C2  quadratic-temperature target ``-E alpha dT / 3``: re-derived from
    equilibrium (``d sigma/dx = 0`` with ``u(0)=u(L)=0`` -> ``sigma = -E alpha
    Tbar``, ``Tbar = mean(T)``) and reproduced with two independent 1-D FE
    implementations written here:
      (a) thermal strain integrated *exactly* (2-point Gauss per element) --
          expected to hit the reference at every mesh (the mechanical solution
          is exact: ``u = 0``);
      (b) nodal (trapezoid) thermal strain -- the convention of the module under
          review; expected to reproduce the module's own O(h^2) values, which
          shows where the reported "order 2.0" comes from.
C3  four restraint families of the closed forms vs the evidence JSON rows.
C4  the evidence JSON's own headline gates recomputed from its own raw numbers,
    plus bit-level re-run agreement of the JSON (done in the shell driver, see
    docs/audit_m15434_thermal_bar.md).

Writes ``docs/audit_m15434/independent_check.json``.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "benchmarks" / "constrained_thermal_bar_evidence.json"
OUT = pathlib.Path(__file__).resolve().parent / "independent_check.json"

# material anchors transcribed from the repo's EH36 table
# (src/tensorfem/polar_thermal_stress.py MATERIAL_ANCHORS['EH36']); the case's
# E / alpha are re-derived by regression below instead of read from the module.
T_C = np.array([20.0, 0.0, -20.0, -40.0, -60.0, -80.0, -100.0])
E_GPA = np.array([206.0, 206.7, 207.3, 208.0, 208.7, 209.3, 210.0])
ALPHA_1E6 = np.array([12.0, 11.65, 11.3, 10.95, 10.6, 10.25, 9.9])
NU = 0.30
T_REF = 20.0
DT = 100.0
L = 1.0


def fit(T, y):
    """Least-squares line y = a + b (T - T_ref); returns the value at T_ref."""
    A = np.vstack([np.ones_like(T), T - T_REF]).T
    (a, _b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a)


def _T_quad(x):
    return T_REF + DT * (x / L) ** 2


def bar_fe(n_elem: int, E: float, alpha: float, thermal: str) -> float:
    """Independent 1-D thermo-elastic FE solve, quadratic dT, both ends fixed.

    ``thermal='exact'``     : consistent thermal load with 2-point Gauss (exact
                              for a quadratic field) -> exact element stresses.
    ``thermal='trapezoid'`` : thermal strain from the *nodal* linear
                              interpolation of T (nodal average per element),
                              i.e. the convention of the module under review.
    Returns the element-averaged axial stress.
    """
    x = np.linspace(0.0, L, n_elem + 1)
    K = np.zeros((n_elem + 1, n_elem + 1))
    f = np.zeros(n_elem + 1)
    le = L / n_elem
    gp = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])
    for e in range(n_elem):
        dN_dx = np.array([-1.0, 1.0]) / le
        if thermal == "exact":
            for g, wgt in zip(gp, [1.0, 1.0]):
                xg = (1 - g) / 2 * x[e] + (1 + g) / 2 * x[e + 1]
                K[np.ix_([e, e + 1], [e, e + 1])] += E * np.outer(dN_dx, dN_dx) * (le / 2) * wgt
                f[[e, e + 1]] += E * alpha * (_T_quad(xg) - T_REF) * dN_dx * (le / 2) * wgt
        else:  # trapezoid on the nodal temperatures
            Te = 0.5 * (_T_quad(x[e]) + _T_quad(x[e + 1]))
            K[np.ix_([e, e + 1], [e, e + 1])] += E * np.outer(dN_dx, dN_dx) * le
            f[[e, e + 1]] += E * alpha * (Te - T_REF) * dN_dx * le
    free = np.arange(1, n_elem)
    u = np.zeros(n_elem + 1)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    sig = []
    for e in range(n_elem):
        dN_dx = np.array([-1.0, 1.0]) / le
        eps = dN_dx @ u[[e, e + 1]]
        if thermal == "exact":
            acc = []
            for g in gp:
                xg = (1 - g) / 2 * x[e] + (1 + g) / 2 * x[e + 1]
                acc.append(E * (eps - alpha * (_T_quad(xg) - T_REF)))
            sig.append(float(np.mean(acc)))
        else:
            Te = 0.5 * (_T_quad(x[e]) + _T_quad(x[e + 1]))
            sig.append(float(E * (eps - alpha * (Te - T_REF))))
    return float(np.mean(sig))


def slope(hs, errs):
    xs = np.log(np.asarray(hs, float))
    ys = np.log(np.maximum(np.asarray(errs, float), 1e-18))
    n = len(xs)
    return float((n * (xs * ys).sum() - xs.sum() * ys.sum())
                 / (n * (xs * xs).sum() - xs.sum() ** 2))


def main() -> dict:
    ev = json.loads(EV.read_text())
    E = fit(T_C, E_GPA * 1e9)
    alpha = fit(T_C, ALPHA_1E6 * 1e-6)

    ref_uniaxial = -E * alpha * DT
    ref_biaxial = -E * alpha * DT / (1 - NU)
    ref_triaxial = -E * alpha * DT / (1 - 2 * NU)
    ref_quad = -E * alpha * DT / 3.0

    # ---- C1 -------------------------------------------------------------
    got_ref = ev["reference"]["value_Pa"]
    c1 = {
        "E_Pa_recomputed": E,
        "E_Pa_in_evidence": ev["reference"]["material"]["E_ref_Pa"],
        "alpha_recomputed": alpha,
        "alpha_in_evidence": ev["reference"]["material"]["alpha_ref_per_K"],
        "ref_recomputed_Pa": ref_uniaxial,
        "ref_in_evidence_Pa": got_ref,
        "rel_diff": (got_ref - ref_uniaxial) / abs(ref_uniaxial),
        "pass": abs((got_ref - ref_uniaxial) / ref_uniaxial) < 1e-12,
    }

    # ---- C2 -------------------------------------------------------------
    ns = (2, 4, 8, 16, 32)
    ex = [{"n_elem": n, "sigma_Pa": bar_fe(n, E, alpha, "exact")} for n in ns]
    tr = [{"n_elem": n, "sigma_Pa": bar_fe(n, E, alpha, "trapezoid"),
           "rel_error": (bar_fe(n, E, alpha, "trapezoid") - ref_quad) / abs(ref_quad)}
          for n in ns]
    ev_rows = ev["studies"]["bar_quadratic_convergence"]["rows"]
    repro = [{"n_elem": r["n_elem"], "audit_sigma_Pa": t["sigma_Pa"],
              "evidence_sigma_Pa": r["sigma_Pa"],
              "rel_diff": (r["sigma_Pa"] - t["sigma_Pa"]) / abs(r["sigma_Pa"])}
             for r, t in zip(ev_rows, tr)]
    order_trap = slope([L / n for n in ns], [abs(r["rel_error"]) for r in tr])
    c2 = {
        "target_recomputed_Pa": ref_quad,
        "target_in_evidence_Pa": ev["studies"]["bar_quadratic_convergence"]["reference_Pa"],
        "rel_diff_target": (ev["studies"]["bar_quadratic_convergence"]["reference_Pa"]
                            - ref_quad) / abs(ref_quad),
        "exact_integration_rows": ex,
        "exact_integration_max_rel_error": max(
            abs(r["sigma_Pa"] - ref_quad) / abs(ref_quad) for r in ex),
        "trapezoid_rows": tr,
        "trapezoid_observed_order": order_trap,
        "evidence_observed_order": ev["studies"]["bar_quadratic_convergence"]["observed_order"],
        "row_by_row_reproduction_vs_evidence": repro,
        "max_row_reproduction_rel_diff": max(abs(r["rel_diff"]) for r in repro),
        "interpretation": ("the mechanical solution is exact at every mesh (u=0); the "
                           "reported O(h^2) error is the trapezoid quadrature error of the "
                           "element mean temperature, i.e. thermal-field interpolation, not "
                           "solver accuracy"),
        "pass": (abs(max(abs(r["rel_diff"]) for r in repro)) < 1e-10
                 and abs(order_trap - 2.0) < 0.05
                 and abs((ev["studies"]["bar_quadratic_convergence"]["reference_Pa"]
                          - ref_quad) / ref_quad) < 1e-12),
    }

    # ---- C3 -------------------------------------------------------------
    fam = {}
    for row in ev["studies"]["hex_restraint_family"]:
        r = row["restraint"]
        if r in fam:
            continue
        ref = {"uniaxial": ref_uniaxial, "biaxial": ref_biaxial,
               "triaxial": ref_triaxial, "free": 0.0}[r]
        got = row["sigma_xx_Pa"]
        fam[r] = {"got_Pa": got, "ref_recomputed_Pa": ref,
                  "rel_error": (got - ref) / abs(ref) if ref else got,
                  "pass_task_tol": (abs((got - ref) / ref) < 0.03) if ref
                  else abs(got) < 1e-3,
                  "pass_roundoff": (abs((got - ref) / ref) < 1e-8) if ref
                  else abs(got) < 1e-6}
    c3 = {"families": fam,
          "pass": len(fam) == 4 and all(v["pass_roundoff"] for v in fam.values())}

    # ---- C4 -------------------------------------------------------------
    g = ev["gates"]
    worst = max(g["bar_uniform_max_abs_rel_error"], g["hex_max_abs_rel_error_xx"])
    c4 = {
        "worst_rel_error_task_case": worst,
        "recomputed_task_gate_3pct": worst < 0.03,
        "recomputed_roundoff_gate_1e-8": worst < 1e-8,
        "bar_quadratic_observed_order": g["bar_quadratic_observed_order"],
        "recomputed_order_gate": abs(g["bar_quadratic_observed_order"] - 2.0) <= 0.35,
        "evidence_verdict_all_pass": ev["verdict"]["all_pass"],
        "pass": (worst < 1e-8
                 and abs(g["bar_quadratic_observed_order"] - 2.0) <= 0.35
                 and ev["verdict"]["all_pass"] is True),
    }

    res = {"case": "CASE-THERMAL-BAR", "mission": "m15434", "audit_mission": "m15714",
           "checks": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
           "all_pass": all(c["pass"] for c in (c1, c2, c3, c4))}
    OUT.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    r = main()
    print("C1 ref rel_diff:", r["checks"]["C1"]["rel_diff"])
    print("C2 rows reproduced, max rel_diff:", r["checks"]["C2"]["max_row_reproduction_rel_diff"],
          "| trapezoid order:", round(r["checks"]["C2"]["trapezoid_observed_order"], 6),
          "| exact-integration max err:", r["checks"]["C2"]["exact_integration_max_rel_error"])
    print("C3 rel errors:", {k: v["rel_error"] for k, v in r["checks"]["C3"]["families"].items()})
    print("C4 worst:", r["checks"]["C4"]["worst_rel_error_task_case"])
    print("ALL PASS:", r["all_pass"])