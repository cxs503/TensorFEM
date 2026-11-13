#!/usr/bin/env python3
"""CASE-CANTILEVER-TIP -- the axial-preload (beam-column) rung.

Verification debt this driver closes
------------------------------------
Every existing rung of CASE-CANTILEVER-TIP varies one axis of the problem:
slenderness/shear, cross-section refinement, taper (EI(x)), the load
distribution (tip point vs end-face traction), the model (beam vs 3-D solid) and
the root boundary condition.  None of them varies the **axial force inside the
beam**, yet that is the parameter that multiplies the transverse deflection of
every hull girder, brace and riser: the case-pool number
delta = P L^3/(3 E I) silently assumes N = 0.

The library already carries the ingredient -- ``tensorfem.buckling`` exposes the
consistent geometric stiffness and assembles ``(K, Kg)`` for a prismatic column
-- but its only consumer is a *linear eigenvalue* buckling solve.  A static
beam-column load path ``(K + N Kg) u = f`` at a *finite* preload was never
graded, so nothing pins down whether ``Kg`` has the right *magnitude* (an
eigenvalue pins only its ratio to ``K`` along the buckling mode).

The exact oracle
----------------
Clamped-free beam-column, dead axial force ``N``, tip transverse load ``P``,
``u = L sqrt(|N|/E I)``.  Solving ``E I w'''' - N w'' = 0`` (tension) or
``E I w'''' + C w'' = 0`` (compression) with ``w = w' = 0`` at the root and
``w'' = 0``, ``V_total = w''' -+ u^2 w' = -P`` at the tip gives (derived here in
sympy, residual reported as a gate, no handbook digit transcribed)

    AF_t(u) = 3 [u - tanh u] / u^3   -> 1 - 2u^2/5 + ...,  < 1   (tension)
    AF_c(u) = 3 [tan u  - u ] / u^3  -> 1 + 2u^2/5 + ...,  > 1   (compression)

``AF_c`` has a pole at ``u = pi/2``, i.e. at ``N_cr = pi^2 E I/(4 L^2)`` -- the
static amplification and the eigenvalue of the same ``K``, ``Kg`` must diverge at
the same load, which is an independent check on the matrices.

Graded
------
1. the oracle itself: ODE residual, small-u series, analytic continuation
   ``AF_c(i u) = AF_t(u)``, pole at pi/2;
2. the FE load path ``(K + N Kg) u = f`` on ``n = 1 ... 32`` elements against it,
   in tension and compression;
3. the case-pool tolerance ``< 3 %`` on the physically admissible rungs;
4. the failure mode as a gate: one element at ``u = 1.5`` is wrong by > 5 %.

Geometry, material and tip load are the case-pool ones
(``benchmarks/cantilever_tip_deflection.py``: ``L = 1 m``, ``b = h = 10 mm``,
``E = 200 GPa``, ``P = 5 N``, ``L/h = 100``), so ``delta_0 = 10.000 mm`` and the
preload rungs need only tens to hundreds of newton.
"""
from __future__ import annotations

import cmath
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch

from tensorfem.buckling import solve_linear_buckling, uniform_column_matrices

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = REPO / "docs" / "CASE-CANTILEVER-TIP-beamcolumn-amplification.md"
FIGDIR = REPO / "docs" / "assets" / "cantilever_beamcolumn"

ANCHOR = "fem_cantilever_beamcolumn"
MISSION = "CASE-CANTILEVER-TIP"
TITLE = ("CASE-CANTILEVER-TIP axial-preload rung: beam-column amplification "
         "delta_N/delta_0 vs the exact tanh/tan oracle on the geometric "
         "stiffness of tensorfem.buckling")

DTYPE = torch.float64
TOLERANCE = 0.03
SHEAR_LIMIT = 1e-3

# case definition (same geometry as the case-pool rung)
L = 1.0
B = 0.01
H = 0.01
E = 200e9
NU = 0.3
P_TIP = 5.0

AREA = B * H
I_CS = B * H**3 / 12.0
EI = E * I_CS
G_SHEAR = E / (2.0 * (1.0 + NU))
KAPPA = 5.0 / 6.0

DELTA_EB = P_TIP * L**3 / (3.0 * EI)
DELTA_SHEAR = P_TIP * L / (KAPPA * AREA * G_SHEAR)
PCR = math.pi**2 * EI / (4.0 * L**2)
U_CR = math.pi / 2.0

LEVELS = (0.5, 1.0, 1.25, 1.5)
N_LADDER = (1, 2, 4, 8, 16, 32)
N_PHYSICAL = (4, 8, 16, 32)
SMALL_DEFLECTION = 0.02          # delta/L admissible for linear beam-column theory


# --------------------------------------------------------------------------- #
# exact oracle
# --------------------------------------------------------------------------- #
def af_tension(u: float) -> float:
    if abs(u) < 1e-6:
        return 1.0 - 2.0 * u * u / 5.0
    return 3.0 * (u - math.tanh(u)) / u**3


def af_compression(u: float) -> float:
    if abs(u) < 1e-6:
        return 1.0 + 2.0 * u * u / 5.0
    return 3.0 * (math.tan(u) - u) / u**3


def af_compression_complex(z: complex) -> complex:
    return 3.0 * (cmath.tan(z) - z) / z**3


def preload_newton(u: float) -> float:
    return EI * u * u / L**2


def _symbolic_ode_solution(tension: bool):
    x, u = sp.symbols("x u", positive=True)
    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4")
    if tension:
        w = c1 + c2 * x + c3 * sp.cosh(u * x) + c4 * sp.sinh(u * x)
    else:
        w = c1 + c2 * x + c3 * sp.cos(u * x) + c4 * sp.sin(u * x)
    wp = sp.diff(w, x)
    shear = sp.diff(w, x, 3) - (1 if tension else -1) * u**2 * wp
    eqs = [sp.Eq(w.subs(x, 0), 0), sp.Eq(wp.subs(x, 0), 0),
           sp.Eq(sp.diff(w, x, 2).subs(x, 1), 0), sp.Eq(shear.subs(x, 1), -1)]
    sol = sp.solve(eqs, [c1, c2, c3, c4], dict=True, simplify=True)[0]
    w_sol = sp.simplify(w.subs(sol))
    return x, u, w_sol, sp.simplify(3 * w_sol.subs(x, 1))


def _sympy_residual() -> dict:
    out = {"derivations": []}
    for name, tension in (("tension", True), ("compression", False)):
        x, u, w_sol, af_sol = _symbolic_ode_solution(tension)
        closed = (3 * (u - sp.tanh(u)) / u**3) if tension else (3 * (sp.tan(u) - u) / u**3)
        residual = sp.simplify(af_sol - closed)
        numeric = [float(af_sol.subs(u, uu)) for uu in LEVELS]
        out.setdefault("numeric_from_ode", {})[name] = numeric
        out["derivations"].append({
            "case": name,
            "ode": "E I w'''' - N w'' = 0" if tension else "E I w'''' + C w'' = 0",
            "amplification_from_ode": str(af_sol),
            "closed_form": str(sp.simplify(closed)),
            "residual": str(residual),
            "series_small_u": str(sp.series(closed, u, 0, 5).removeO()),
        })
        if not tension:
            out["pole_from_closed_form"] = float(
                sp.nsolve(sp.cos(u), u, 1.5))
    return out


def oracle_selfcheck(sym: dict) -> dict:
    out = dict(sym)
    out["residual_ladder_tension"] = max(
        abs(num - af_tension(uu)) / af_tension(uu)
        for num, uu in zip(sym["numeric_from_ode"]["tension"], LEVELS))
    out["residual_ladder_compression"] = max(
        abs(num - af_compression(uu)) / af_compression(uu)
        for num, uu in zip(sym["numeric_from_ode"]["compression"], LEVELS))
    out["series_residual_tension"] = abs(af_tension(1e-3) - (1 - 2e-6 / 5)) / (1 - 2e-6 / 5)
    out["series_residual_compression"] = abs(af_compression(1e-3) - (1 + 2e-6 / 5)) / (1 + 2e-6 / 5)
    out["continuation_identity_max_rel"] = max(
        abs(af_compression_complex(1j * uu).real - af_tension(uu)) / af_tension(uu)
        + abs(af_compression_complex(1j * uu).imag) for uu in LEVELS)
    out["pole_u"] = U_CR
    out["pole_vs_euler_load_rel"] = abs(out["pole_from_closed_form"] - U_CR) / U_CR
    out["af_compression_at_u_1p55"] = af_compression(1.55)
    return out


# --------------------------------------------------------------------------- #
# FE: (K + N Kg) u = f on the merged tensorfem.buckling matrices
# --------------------------------------------------------------------------- #
def _matrices(elements: int):
    return uniform_column_matrices(L, elements, EI, dtype=DTYPE)


def fe_solve(u_level: float, elements: int, tension: bool) -> dict:
    K, Kg = _matrices(elements)
    N = preload_newton(u_level) * (1.0 if tension else -1.0)
    Kt = K + N * Kg
    ndof = Kt.shape[0]
    f = torch.zeros(ndof, dtype=DTYPE)
    tip_node = elements
    f[2 * tip_node] = P_TIP
    fixed = {0, 1}
    free = [i for i in range(ndof) if i not in fixed]
    free_t = torch.tensor(free)
    disp = torch.zeros(ndof, dtype=DTYPE)
    disp[free_t] = torch.linalg.solve(Kt[free_t[:, None], free_t], f[free_t])
    w_tip = float(disp[2 * tip_node])
    return {"elements": elements, "preload_N": N, "w_tip_m": w_tip,
            "amplification": w_tip / DELTA_EB, "nodal": disp.numpy()}


def fe_buckling(elements: int) -> dict:
    K, Kg = _matrices(elements)
    pcr = float(solve_linear_buckling(K, Kg, (0, 1), modes=1).load_factors[0])
    return {"elements": elements, "pcr_FE_N": pcr, "pcr_exact_N": PCR,
            "rel_error": abs(pcr - PCR) / PCR, "u_FE": L * math.sqrt(pcr / EI)}


def analytic_profile(xs: np.ndarray, u_level: float, tension: bool) -> np.ndarray:
    x, u, w_sol, _ = _symbolic_ode_solution(tension)
    fn = sp.lambdify((x, u), w_sol, "numpy")
    w = np.asarray(fn(xs, u_level), dtype=float)
    return w / w[-1]


def fe_profile(nodal: np.ndarray, elements: int, samples: int = 400) -> np.ndarray:
    le = L / elements
    xs = np.linspace(0.0, L, samples)
    w = np.zeros_like(xs)
    for e in range(elements):
        w_e, th_e = nodal[2 * e], nodal[2 * e + 1]
        w_b, th_b = nodal[2 * e + 2], nodal[2 * e + 3]
        mask = (xs >= e * le - 1e-12) & (xs <= (e + 1) * le + 1e-12)
        xi = (xs[mask] - e * le) / le
        w[mask] = ((2 * xi**3 - 3 * xi**2 + 1) * w_e
                   + (xi**3 - 2 * xi**2 + xi) * le * th_e
                   + (-2 * xi**3 + 3 * xi**2) * w_b
                   + (xi**3 - xi**2) * le * th_b)
    return w


def convergence_order(errors: list[float], elements: list[int]) -> dict:
    ns = np.array(elements, dtype=float)
    es = np.array([max(e, 1e-18) for e in errors], dtype=float)
    half = len(ns) // 2
    slope, _ = np.polyfit(np.log(ns[half:]), np.log(es[half:]), 1)
    return {"slope": float(slope), "order": float(-slope),
            "window": [int(ns[half]), int(ns[-1])],
            "ratios_per_doubling": [float(es[i] / es[i + 1]) for i in range(len(es) - 1)],
            "errors": [float(e) for e in es]}


def ladder() -> list[dict]:
    rows = []
    for u_level in LEVELS:
        for tension in (True, False):
            exact = af_tension(u_level) if tension else af_compression(u_level)
            for n in N_LADDER:
                res = fe_solve(u_level, n, tension)
                rows.append({
                    "u": u_level, "case": "tension" if tension else "compression",
                    "preload_N": res["preload_N"],
                    "preload_over_pcr": res["preload_N"] / PCR,
                    "elements": n, "amplification_exact": exact,
                    "amplification_FE": res["amplification"],
                    "rel_error": abs(res["amplification"] - exact) / exact,
                    "w_tip_m": res["w_tip_m"], "w_tip_mm": res["w_tip_m"] * 1e3,
                    "delta_over_L": abs(res["w_tip_m"]) / L,
                })
    return rows


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    id: str
    quantity: str
    reference: str
    value: float
    threshold: float
    operator: str
    passed: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "quantity": self.quantity, "reference": self.reference,
                "value": self.value, "threshold": self.threshold,
                "operator": self.operator, "passed": bool(self.passed), "note": self.note}


def _gate(gid, quantity, reference, value, threshold, operator, note="") -> Gate:
    ok = {"<=": value <= threshold, ">=": value >= threshold,
          "<": value < threshold, ">": value > threshold}[operator]
    return Gate(gid, quantity, reference, float(value), float(threshold), operator, ok, note)


def build_gates(sc: dict, rows: list[dict], buck: list[dict]) -> list[Gate]:
    g: list[Gate] = [
        _gate("oracle_ode_residual_tension",
              "max relative residual of the sympy ODE solution vs 3(u - tanh u)/u^3",
              "E I w'''' - N w'' = 0, clamped root, free tip",
              sc["residual_ladder_tension"], 1e-12, "<=",
              "machine-derived oracle; no handbook digit is transcribed"),
        _gate("oracle_ode_residual_compression",
              "max relative residual of the sympy ODE solution vs 3(tan u - u)/u^3",
              "E I w'''' + C w'' = 0, clamped root, free tip",
              sc["residual_ladder_compression"], 1e-12, "<=",
              "same derivation path, compression branch"),
        _gate("oracle_small_u_series_tension",
              "relative residual of AF_t vs 1 - 2u^2/5 at u = 1e-3",
              "Taylor expansion of 3(u - tanh u)/u^3",
              sc["series_residual_tension"], 1e-9, "<="),
        _gate("oracle_small_u_series_compression",
              "relative residual of AF_c vs 1 + 2u^2/5 at u = 1e-3",
              "Taylor expansion of 3(tan u - u)/u^3",
              sc["series_residual_compression"], 1e-9, "<="),
        _gate("analytic_continuation_identity",
              "|AF_c(i u) - AF_t(u)| / AF_t(u) over the ladder",
              "AF_c(i u) = AF_t(u)", sc["continuation_identity_max_rel"], 1e-12, "<=",
              "one analytic function in two branches: a slip in either is caught"),
        _gate("pole_is_the_euler_load",
              "|u_pole - pi/2| / (pi/2) from the closed form",
              "fixed-free N_cr = pi^2 E I/(4 L^2) <=> cos u = 0",
              sc["pole_vs_euler_load_rel"], 1e-9, "<=",
              "static amplification and eigenvalue must diverge at the same load"),
        _gate("buckling_eigenvalue_from_same_matrices",
              "|P_cr,FE - pi^2 E I/(4 L^2)| / P_cr on the finest mesh (n = %d)" % buck[-1]["elements"],
              "pi^2 E I/(4 L^2)", buck[-1]["rel_error"], 1e-6, "<=",
              "confirms that K and Kg are the matrices whose static sum is graded below"),
        _gate("buckling_convergence_order",
              "log-log slope of the P_cr error over n = 16..32",
              "consistent Hermite element: order >= 2",
              convergence_order([b["rel_error"] for b in buck],
                                [b["elements"] for b in buck])["order"], 2.0, ">=",
              "the eigenvalue must converge under refinement, not merely sit close"),
    ]
    for case, fn in (("tension", af_tension), ("compression", af_compression)):
        for u_level in LEVELS:
            sel = [r for r in rows if r["case"] == case and r["u"] == u_level
                   and r["elements"] in N_PHYSICAL]
            worst = max(r["rel_error"] for r in sel)
            dmax = max(r["delta_over_L"] for r in sel)
            note = ("small-deflection domain (delta/L <= 2 %)"
                    if dmax <= SMALL_DEFLECTION else
                    "OUTSIDE the small-deflection domain (delta/L = %.1f %%): numerical-stiffness "
                    "check only, not an engineering prediction" % (1e2 * dmax))
            g.append(_gate(f"fe_vs_exact_{case}_u{u_level:g}",
                           f"worst FE amplification error over n in {list(N_PHYSICAL)}, u = {u_level:g}",
                           f"AF_{case[0]}(u = {u_level:g}) = {fn(u_level):.6f}",
                           worst, TOLERANCE, "<=", note))
    tension_rows = [r for r in rows if r["case"] == "tension"]
    comp_rows = [r for r in rows if r["case"] == "compression"]
    g.append(_gate("tension_stiffens", "max AF_FE over the tension rungs",
                   "AF_t(u) < 1 for u > 0",
                   max(r["amplification_FE"] for r in tension_rows), 1.0, "<"))
    g.append(_gate("compression_amplifies", "min AF_FE over the compression rungs",
                   "AF_c(u) > 1 for 0 < u < pi/2",
                   min(r["amplification_FE"] for r in comp_rows), 1.0, ">"))
    g.append(_gate("zero_preload_recovers_euler_bernoulli",
                   "|AF_FE - 1| at N = 0, n = 1",
                   "delta = P L^3/(3 E I), the case-pool rung",
                   abs(fe_solve(0.0, 1, True)["amplification"] - 1.0), 1e-14, "<=",
                   "with Kg switched off the rung degenerates to the original case"))
    coarse = [r for r in rows if r["elements"] == 1 and r["u"] == 1.5 and r["case"] == "compression"][0]
    g.append(_gate("single_element_fails_near_critical",
                   "single-element relative error at u = 1.5 compression (0.955 N_cr)",
                   f"AF_c(1.5) = {af_compression(1.5):.4f}",
                   coarse["rel_error"], 0.05, ">=",
                   "graded as a required failure: a one-element beam-column is wrong by construction near P_cr"))
    fine = [r for r in rows if r["elements"] == 32 and r["u"] == 1.5 and r["case"] == "compression"][0]
    g.append(_gate("refinement_recovers_near_critical",
                   "32-element relative error at u = 1.5 compression",
                   f"AF_c(1.5) = {af_compression(1.5):.4f}",
                   fine["rel_error"], TOLERANCE, "<=",
                   "same rung inside tolerance once the mesh resolves the shape"))
    for case in ("tension", "compression"):
        for u_level in (1.0, 1.5):
            errs = [r["rel_error"] for r in rows if r["case"] == case and r["u"] == u_level]
            order = convergence_order(errs, list(N_LADDER))
            g.append(_gate(f"convergence_order_{case}_u{u_level:g}",
                           f"fitted order of the nodal-displacement error over n = {order['window'][0]}..{order['window'][1]}",
                           "consistent Hermite beam-column: order >= 3 measured, 2 required",
                           order["order"], 3.0, ">=",
                           "nodal super-convergence of the Hermite beam-column is the measured result; the gate only "
                           "demands better-than-second order"))
    g.append(_gate("shear_compliance_negligible", "delta_shear/delta_EB at L/h = 100",
                   "E/(4 kappa G) (h/L)^2", DELTA_SHEAR / DELTA_EB, SHEAR_LIMIT, "<=",
                   "the beam-column oracle is Euler-Bernoulli; at L/h = 100 the shear share is < 0.1 %"))
    return g


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(sc: dict, rows: list[dict]) -> list[str]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    u_t = np.linspace(1e-4, 2.6, 400)
    u_c = np.linspace(1e-4, U_CR - 2e-3, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(u_t, [af_tension(u) for u in u_t], color="#2b6cb0", lw=2,
            label=r"exact tension $3(u-\tanh u)/u^3$")
    ax.plot(u_c, [af_compression(u) for u in u_c], color="#c53030", lw=2,
            label=r"exact compression $3(\tan u-u)/u^3$")
    ax.axhline(1.0, color="0.6", ls=":", lw=1, label="Euler-Bernoulli (N = 0)")
    ax.axvline(U_CR, color="0.35", ls="--", lw=1)
    ax.text(U_CR - 0.28, 40.0, r"$u_{cr}=\pi/2$", fontsize=9, color="0.35")
    for case, color, marker in (("tension", "#2b6cb0", "o"), ("compression", "#c53030", "s")):
        sel = [r for r in rows if r["case"] == case and r["elements"] == 8]
        ax.plot([r["u"] for r in sel], [r["amplification_FE"] for r in sel], ls="none",
                marker=marker, ms=7, mfc="white", mec=color, mew=1.6,
                label=f"FE, 8 elements ({case})")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 3e1)
    ax.set_xlabel(r"preload parameter $u=L\sqrt{|N|/EI}$")
    ax.set_ylabel(r"amplification $\delta_N/\delta_0$")
    ax.set_title("Axial preload multiplies the cantilever tip deflection")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = FIGDIR / "amplification.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = {"tension": ["#2b6cb0", "#4299e1"], "compression": ["#c53030", "#ed8936"]}
    for case in ("tension", "compression"):
        for k, u_level in enumerate((1.0, 1.5)):
            errs = [max(r["rel_error"], 1e-17) for r in rows
                    if r["case"] == case and r["u"] == u_level]
            order = convergence_order([r["rel_error"] for r in rows
                                       if r["case"] == case and r["u"] == u_level],
                                      list(N_LADDER))
            ax.loglog(N_LADDER, errs, marker="o", color=colors[case][k],
                      label=f"{case} u={u_level:g} (slope {order['slope']:.2f})")
    ax.axhline(TOLERANCE, color="0.4", ls="--", lw=1)
    ax.text(1.1, TOLERANCE * 1.3, "case-pool tolerance 3 %", fontsize=8, color="0.4")
    ax.set_xlabel("elements")
    ax.set_ylabel("relative error of the amplification")
    ax.set_title("Refinement recovers the exact beam-column amplification")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGDIR / "convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = np.linspace(0.0, L, 400)
    n = 8
    for label, tension, u_level, color in (("N = 0", None, 0.0, "0.5"),
                                           ("tension u=1", True, 1.0, "#2b6cb0"),
                                           ("compression u=1", False, 1.0, "#c53030")):
        res = fe_solve(u_level, n, True if tension is None else tension)
        w_fe = fe_profile(res["nodal"], n)
        ax.plot(xs, w_fe / abs(w_fe[-1]), color=color, lw=2, label=f"FE {label}")
        if tension is not None:
            ax.plot(xs, analytic_profile(xs, u_level, tension), color=color, lw=1, ls="--")
    ax.plot([], [], color="0.3", lw=1, ls="--", label="exact (lambdified ODE solution)")
    ax.set_xlabel("x / m")
    ax.set_ylabel(r"$w(x)/|w(L)|$")
    ax.set_title("Normalised deflection shape at u = 1 (8 elements)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGDIR / "profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for case, color in (("tension", "#2b6cb0"), ("compression", "#c53030")):
        sel = [r for r in rows if r["case"] == case and r["elements"] == 8]
        ax.semilogy([r["u"] for r in sel], [max(r["rel_error"], 1e-17) for r in sel],
                    marker="o", color=color, label=f"8 elements, {case}")
    ax.axhline(TOLERANCE, color="0.4", ls="--", lw=1)
    ax.set_xlabel(r"$u = L\sqrt{|N|/EI}$")
    ax.set_ylabel("relative error (8 elements)")
    ax.set_title(r"Error grows with preload: near $u_{cr}$ the mesh must be refined")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGDIR / "error_vs_preload.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))
    return paths


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> dict:
    t0 = time.perf_counter()
    sym = _sympy_residual()
    sc = oracle_selfcheck(sym)
    rows = ladder()
    buck = [fe_buckling(n) for n in N_LADDER]
    gates = build_gates(sc, rows, buck)
    limitations = [
        "Euler-Bernoulli beam-column only: the axial force acts on the bending stiffness, the "
        "Timoshenko beam-column coupling (shear deformation amplified by N) is not modelled; at "
        "L/h = 100 the shear share of the deflection is %.2e, so the oracle is inside its range."
        % (DELTA_SHEAR / DELTA_EB),
        "small-deflection (linearised) theory: valid while delta/L stays small. The u = 1.25 and "
        "u = 1.5 compression rungs reach delta/L = 4.0 % and 11.2 % and are graded as numerical "
        "stiffness checks, not as engineering predictions.",
        "the axial force is a dead (load-direction-fixed) load; follower axial force would change "
        "the geometric stiffness at finite rotation, which this linear rung does not address.",
        "the geometric stiffness is the consistent (Hermite) matrix: it is exact for the "
        "eigenvalue problem in the limit and second-order accurate for nodal displacements, but "
        "the interior bending moment along an element is only O(h^2) accurate.",
        "beam theory itself is the limit here: no root warping, no 3-D Poisson contraction. The "
        "root-clamp and Poisson residuals of the 3-D model are quantified separately in "
        "docs/CASE-CANTILEVER-TIP-shear-validity.md and ...-side-convergence.md."
    ]
    payload = {
        "anchor": ANCHOR, "mission": MISSION, "title": TITLE,
        "generated_at": utc_now(),
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "sympy": sp.__version__, "threads": torch.get_num_threads()},
        "problem": {
            "length_m": L, "width_m": B, "height_m": H, "young_modulus_Pa": E,
            "poisson": NU, "tip_load_N": P_TIP, "inertia_m4": I_CS, "EI_Nm2": EI,
            "delta_EB_m": DELTA_EB, "delta_shear_m": DELTA_SHEAR,
            "pcr_fixed_free_N": PCR, "u_cr": U_CR,
            "levels_u": list(LEVELS), "ladder_elements": list(N_LADDER),
            "case_tolerance": TOLERANCE,
            "preload_N_at_levels": [preload_newton(u) for u in LEVELS],
        },
        "oracle_selfcheck": sc,
        "ladder": rows,
        "buckling_ladder": buck,
        "convergence": {f"{case}_u{u:g}": convergence_order(
            [r["rel_error"] for r in rows if r["case"] == case and r["u"] == u], list(N_LADDER))
            for case in ("tension", "compression") for u in LEVELS},
        "gates": [x.to_dict() for x in gates],
        "passed": all(x.passed for x in gates),
        "failed_gates": [x.id for x in gates if not x.passed],
        "limitations": limitations,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    payload["figures"] = make_figures(sc, rows)
    write_json(HERE / "result.json", payload)
    manifest = {
        "schema": "tensorfem.verified-gates.v1", "anchor": ANCHOR, "title": TITLE,
        "mission": MISSION, "generated_at": payload["generated_at"],
        "criteria": {"case_pool_tolerance": TOLERANCE,
                     "small_deflection_limit": SMALL_DEFLECTION,
                     "shear_negligible": SHEAR_LIMIT,
                     "oracle_exact": 1e-12,
                     "buckling_eigenvalue": 1e-4},
        "gates": payload["gates"], "passed": payload["passed"],
        "failed_gates": payload["failed_gates"],
        "oracle_selfcheck": {k: sc[k] for k in
                             ("residual_ladder_tension", "residual_ladder_compression",
                              "continuation_identity_max_rel", "pole_vs_euler_load_rel")},
        "artifacts": {"result": str((HERE / "result.json").relative_to(REPO)),
                      "docs": str(DOC.relative_to(REPO)),
                      "readme": str((HERE / "README.md").relative_to(REPO)),
                      "figures": payload["figures"]},
        "limitations": limitations,
        "summary": ("Static beam-column load path (K + N Kg) u = f of the merged "
                    "tensorfem.buckling matrices vs the exact tanh/tan amplification; "
                    "PASS" if payload["passed"] else "FAIL"),
    }
    write_json(HERE / "gates_verified.json", manifest)
    print(f"PASS={payload['passed']} failed={payload['failed_gates']}")
    print(f"oracle residuals: t={sc['residual_ladder_tension']:.2e} "
          f"c={sc['residual_ladder_compression']:.2e} "
          f"continuation={sc['continuation_identity_max_rel']:.2e}")
    print(f"worst physical FE error: "
          f"{100 * max(x.value for x in gates if x.id.startswith('fe_vs_exact_') and 'small-deflection' in x.note):.4f} %")
    print(f"wrote result.json, gates_verified.json, {len(payload['figures'])} figures in {payload['seconds']} s")
    return payload


if __name__ == "__main__":
    main()