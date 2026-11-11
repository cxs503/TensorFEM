#!/usr/bin/env python3
"""CASE-KIRSCH-PLATE: stress concentration of a circular hole in a tension plate.

Physics
-------
An isotropic plate of width ``W``, thickness ``t``, with a central circular hole
of diameter ``d``, carries a uniform far-field uniaxial tension ``T`` (gross
stress ``sigma_0 = T``).  Kirsch (1898) solved the *infinite* plate exactly; in
polar coordinates centred on the hole (radius ``a``, ``theta`` measured from the
load axis) the classical field is

    sigma_rr = T/2 (1 - a^2/r^2) + T/2 (1 - 4 a^2/r^2 + 3 a^4/r^4) cos 2 theta
    sigma_tt = T/2 (1 + a^2/r^2) - T/2 (1 + 3 a^4/r^4) cos 2 theta
    sigma_rt = -T/2 (1 + 2 a^2/r^2 - 3 a^4/r^4) sin 2 theta

(Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., Art. 39, eqs. (a)-(c)).
On the hole edge ``r = a`` this collapses to ``sigma_tt = T (1 - 2 cos 2 theta)``,
so the hoop stress runs from ``-T`` at the load-axis poles to

    K_t = sigma_tt(a, pi/2) / T = 3                                    (1)

at the two poles perpendicular to the load: the ``3`` of every stress
concentration handbook.  ``K_t`` is independent of ``E``, of ``nu`` and of the
absolute scale, which is itself used below as a sensitivity check.

Reference and quoted value (zero hand-copy)
-------------------------------------------
``K_t = 3`` is evaluated from the cited closed form (1), not copied from a table.
The driver also *verifies the oracle itself*: it re-evaluates (a)-(c) on a dense
grid and asserts the hole edge is traction free (``sigma_rr = sigma_rt = 0``)
and that the far field tends to uniaxial ``T``.  No handbook digit enters the
evidence chain.

Models (two independent discretisations of the same physics)
------------------------------------------------------------
1. **Traction-driven finite-width quarter plate** (production path
   :mod:`tensorfem.finite_width_hole`, Q4 plane stress, starburst mesh graded to
   the hole).  Quarter model ``0 <= x <= L``, ``0 <= y <= W/2``, hole quadrant at
   the origin: ``u_x = 0`` on ``x = 0``, ``u_y = 0`` on ``y = 0``, hole and the
   lateral edge ``y = W/2`` traction free, uniform tension on ``x = L``.  No
   exact solution is injected anywhere: this is a genuine boundary-value problem,
   so its limit carries both the discretisation error *and* the finite-width
   model error.  The finite-width part is removed by the ``d/W`` ladder and the
   length by a Saint-Venant study.
2. **Dirichlet-driven Q8 polar annulus** (production path
   :mod:`tensorfem.structural_canonical_face`, quadratic serendipity plane
   stress).  The exact Kirsch displacement is prescribed on ``r = R_out`` and the
   hole edge is left free, so only discretisation error survives.  Different
   element family, different mesh topology, different boundary formulation: a
   model-form cross-check rather than a re-run of (1).

Outputs (this directory): ``result.json``, ``gates_verified.json`` and the four
PNGs.  Generated, never hand-edited.

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # benchmarks/verified -> _harness
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import __version__  # noqa: E402
from tensorfem.finite_width_hole import (  # noqa: E402
    FiniteWidthPlate,
    heywood_stress_concentration,
    solve_case,
)
from tensorfem.structural_canonical_face import (  # noqa: E402
    solve_kirsch_q8_concentration,
)

from _harness import (  # noqa: E402  (same directory)
    Gate,
    gate_manifest,
    pyplot,
    repo_root,
    richardson,
    utc_now,
    write_json,
)

ANCHOR = "fem_kirsch_plate"
MISSION = "CASE-KIRSCH-PLATE"
TITLE = "KIRSCH-PLATE: circular hole in a tension plate, K_t = 3 (Kirsch 1898)"

#: Predeclared thresholds (fixed before the runs; the campaign tolerance is 5 %).
CRITERIA = {
    "kirsch_limit_rel_error_max": 0.05,          # pool mandate, d/W -> 0
    "kirsch_extrapolated_rel_error_max": 0.01,   # stricter, Richardson limit
    "q8_crosscheck_rel_error_max": 0.05,
    "observed_order_min": 1.0,
    "q8_observed_order_min": 1.5,
    "load_balance_rel_max": 1.0e-10,
    "nu_E_invariance_rel_max": 0.01,
    "saint_venant_band_rel_max": 0.01,
    "finite_width_trend_rel_max": 0.03,
    "hoop_profile_rms_over_kt_max": 0.05,
    "refinement_levels_min": 3,
}

RATIOS = (0.02, 0.05, 0.10)
LADDER = (12, 24, 48)          # n_boundary, refinement ratio r = 2
HEADLINE_RATIO = 0.02          # closest to the infinite plate (Kirsch limit)
GRADING = 6.0
HALF_LENGTH_OVER_WIDTH = 1.5

L_STUDY_RATIO = 0.02
L_STUDY_LEVEL = 24
L_STUDY_HALF_LENGTHS = (1.0, 1.5, 2.5, 4.0)
#: Resolution-matched arms: the starburst mesh spreads the same ray count over a longer
#: plate, so ``n ~ L`` is required or the study measures discretisation, not physics.
L_STUDY_UNMATCHED = (1.0, 1.5, 2.5, 4.0)

NU_E_RATIO = 0.02
NU_E_LEVEL = 24
NU_CASES = ((0.0, 2.0e11), (0.30, 2.0e11), (0.45, 2.0e11), (0.30, 2.0e9))

Q8_OUTER = 2.0
Q8_LADDER = ((24, 6), (48, 12), (96, 24))   # clean r = 2 sequence

TREND_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50)
TREND_LEVEL = 40

FIGURES: list[str] = []


# ---------------------------------------------------------------------------
# Oracle: Kirsch closed form + its own boundary/far-field self-check
# ---------------------------------------------------------------------------

def kirsch_stress(r, theta, radius=1.0, tension=1.0):
    """Kirsch infinite-plate stress (``sigma_rr, sigma_tt, sigma_rt``)."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a, T = radius, tension
    a2, a4 = a * a, a * a * a * a
    c2, s2 = np.cos(2.0 * theta), np.sin(2.0 * theta)
    srr = 0.5 * T * (1.0 - a2 / r**2) + 0.5 * T * (1.0 - 4.0 * a2 / r**2 + 3.0 * a4 / r**4) * c2
    stt = 0.5 * T * (1.0 + a2 / r**2) - 0.5 * T * (1.0 + 3.0 * a4 / r**4) * c2
    srt = -0.5 * T * (1.0 + 2.0 * a2 / r**2 - 3.0 * a4 / r**4) * s2
    return srr, stt, srt


def oracle_selfcheck() -> dict:
    """Verify the analytic oracle by its own boundary conditions and limit."""
    theta = np.linspace(0.0, 2.0 * math.pi, 721)
    srr, stt, srt = kirsch_stress(np.ones_like(theta), theta)
    far = kirsch_stress(np.full(theta.size, 200.0), theta)
    # uniaxial far field: sigma_xx -> T, sigma_yy -> 0, sigma_xy -> 0
    sxx_far = far[0] * np.cos(theta) ** 2 + far[1] * np.sin(theta) ** 2 - 2.0 * far[2] * np.sin(theta) * np.cos(theta)
    syy_far = far[0] * np.sin(theta) ** 2 + far[1] * np.cos(theta) ** 2 + 2.0 * far[2] * np.sin(theta) * np.cos(theta)
    return {
        "hole_edge_traction_free_max": float(max(np.abs(srr).max(), np.abs(srt).max())),
        "hole_edge_hoop_min": float(stt.min()),
        "hole_edge_hoop_max": float(stt.max()),
        "edge_hoop_at_pole_degrees": 90.0,
        "far_field_sxx_deviation": float(np.abs(sxx_far - 1.0).max()),
        "far_field_syy_max": float(np.abs(syy_far).max()),
        "kt_exact": float(stt.max() / 1.0),
        "kt_exact_closed_form": 3.0,
    }


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def kirsch_limit_ladder() -> list:
    """Traction-driven quarter plates: h-ladder at three hole/width ratios."""
    rows = []
    for ratio in RATIOS:
        plate = FiniteWidthPlate(diameter=ratio, half_length=HALF_LENGTH_OVER_WIDTH)
        for level in LADDER:
            started = time.time()
            case = solve_case(plate, n_boundary=level, grading=GRADING)
            row = {k: v for k, v in case.items() if not k.startswith("_")}
            row["wall_seconds"] = time.time() - started
            row["ktg_error_vs_kirsch"] = abs(row["ktg_fe"] - 3.0) / 3.0
            profile = np.asarray(row["hole_edge_hoop_profile_over_tension"], dtype=float)
            th = np.radians(profile[:, 0])
            exact = 1.0 - 2.0 * np.cos(2.0 * th)
            row["hoop_profile_rms_over_kt"] = float(np.sqrt(np.mean((profile[:, 1] - exact) ** 2)) / 3.0)
            row["_case"] = case  # kept in memory for the field/profile figures
            rows.append(row)
    return rows


def ladder_summary(rows: list) -> dict:
    """Richardson limit in h and observed order, per ratio."""
    summary = {}
    for ratio in RATIOS:
        series = [r for r in rows if abs(r["ratio"] - ratio) < 1e-12]
        series.sort(key=lambda r: r["n_boundary"])
        values = [r["ktg_fe"] for r in series]
        rich = richardson(*values, refinement_ratio=2.0)
        summary[f"{ratio:.2f}"] = {
            "levels": [r["n_boundary"] for r in series],
            "ktg_fe": values,
            "errors_vs_kirsch": [abs(v - 3.0) / 3.0 for v in values],
            "richardson": rich,
            "richardson_error_vs_kirsch": (
                None if not math.isfinite(rich["richardson_limit"]) else abs(rich["richardson_limit"] - 3.0) / 3.0
            ),
        }
    return summary


def saint_venant_study(resolution_matched: bool = True) -> list:
    """Plate length sensitivity at the anchor ratio (load-introduction / rigid arm).

    ``resolution_matched`` keeps the *physical* element size at the hole fixed by
    scaling the ray count with ``L``.  With the ray count held fixed instead, the
    same rays span a longer plate, the angular spacing near the ``theta = pi/2``
    apex coarsens, and ``K_t`` drifts monotonically by several percent -- an
    artefact that is easy to mistake for a load-introduction effect.  Both tables
    are written out so the confounder is on the record.
    """
    rows = []
    for half_length in L_STUDY_HALF_LENGTHS:
        n = max(12, int(round(L_STUDY_LEVEL * half_length / HALF_LENGTH_OVER_WIDTH))) if resolution_matched else L_STUDY_LEVEL
        plate = FiniteWidthPlate(diameter=L_STUDY_RATIO, half_length=half_length)
        case = solve_case(plate, n_boundary=n, n_radial=n, grading=GRADING)
        rows.append(
            {
                "half_length_over_width": half_length,
                "n_boundary": n,
                "elements": case["elements"],
                "ktg_fe": case["ktg_fe"],
                "ktn_fe": case["ktn_fe"],
                "load_balance_rel": case["load_balance"]["relative_error"],
            }
        )
    return rows


def nu_e_invariance() -> list:
    """Kirsch K_t does not depend on nu or on the absolute stiffness."""
    rows = []
    for nu, young in NU_CASES:
        plate = FiniteWidthPlate(diameter=NU_E_RATIO, young=young, poisson=nu)
        case = solve_case(plate, n_boundary=NU_E_LEVEL, grading=GRADING)
        rows.append(
            {
                "poisson": nu,
                "young_pa": young,
                "ktg_fe": case["ktg_fe"],
                "ktn_fe": case["ktn_fe"],
                "load_balance_rel": case["load_balance"]["relative_error"],
            }
        )
    return rows


def element_order_crosscheck() -> list:
    """Q8 polar annulus driven by the exact Kirsch displacement on r = R_out."""
    rows = []
    for n_theta, n_radial in Q8_LADDER:
        started = time.time()
        result = solve_kirsch_q8_concentration(
            n_theta=n_theta, n_radial=n_radial, outer_radius=Q8_OUTER
        )
        rows.append(
            {
                "n_theta": n_theta,
                "n_radial": n_radial,
                "outer_over_inner": Q8_OUTER,
                "nodes": result.n_node,
                "kt": result.factor,
                "error_vs_kirsch": result.absolute_error / 3.0,
                "displacement_relative_l2": result.displacement_relative_l2,
                "cg_relative_residual": result.cg_relative_residual,
                "wall_seconds": time.time() - started,
            }
        )
    return rows


def finite_width_trend() -> list:
    """d/W ladder 0.1 .. 0.5: gross K_t rises, net K_t falls (Howland/Heywood)."""
    rows = []
    for ratio in TREND_RATIOS:
        plate = FiniteWidthPlate(diameter=ratio, half_length=HALF_LENGTH_OVER_WIDTH)
        case = solve_case(plate, n_boundary=TREND_LEVEL, grading=GRADING)
        heywood_net = heywood_stress_concentration(ratio)
        heywood_gross = heywood_net / (1.0 - ratio)
        rows.append(
            {
                "ratio": ratio,
                "ktg_fe": case["ktg_fe"],
                "ktn_fe": case["ktn_fe"],
                "heywood_ktn": heywood_net,
                "heywood_ktg": heywood_gross,
                "peterson_ktn": case["ktn_peterson"],
                "ktg_error_vs_heywood": abs(case["ktg_fe"] - heywood_gross) / heywood_gross,
                "load_balance_rel": case["load_balance"]["relative_error"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_field(case: dict, ratio: float, path: Path) -> None:
    """sigma_xx / T on the quarter plate (nodal average of the Q4 element values)."""
    plt = pyplot()
    nodes = np.asarray(case["_meta"]["nodes"], dtype=float)
    elements = np.asarray(case["_meta"]["elements"])
    stress = np.asarray(case["_stress"], dtype=float)
    tension = FiniteWidthPlate(diameter=ratio).tension
    nodal = np.zeros(nodes.shape[0])
    count = np.zeros(nodes.shape[0])
    for element, connectivity in enumerate(elements):
        nodal[connectivity] += stress[element, 0] / tension
        count[connectivity] += 1.0
    nodal = nodal / np.maximum(count, 1.0)
    triangles = np.concatenate([elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    level = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, nodal, levels=24, cmap="turbo")
    ax.tricontour(nodes[:, 0], nodes[:, 1], triangles, nodal, levels=[-1.0, 0.0, 1.0, 2.0, 3.0],
                  colors="k", linewidths=0.6)
    fig.colorbar(level, ax=ax, label=r"$\sigma_{xx} / T$")
    ax.set_aspect("equal")
    ax.set_xlabel("x / W")
    ax.set_ylabel("y / W")
    ax.set_title(f"Kirsch plate, d/W = {ratio:g}, n = {case['n_boundary']}: "
                 r"$\sigma_{xx}/T$ (quarter model)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    FIGURES.append(str(path.relative_to(repo_root())))


def figure_convergence(ladder_rows: list, summary: dict, q8_rows: list, path: Path) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for ratio in RATIOS:
        series = sorted([r for r in ladder_rows if abs(r["ratio"] - ratio) < 1e-12], key=lambda r: r["n_boundary"])
        h = [1.0 / r["n_boundary"] for r in series]
        err = [max(r["ktg_error_vs_kirsch"], 1e-12) for r in series]
        ax.loglog(h, err, "o-", label=f"Q4 plate, d/W = {ratio:g}")
    h8 = [1.0 / r["n_theta"] for r in q8_rows]
    err8 = [max(r["error_vs_kirsch"], 1e-12) for r in q8_rows]
    ax.loglog(h8, err8, "s--", label="Q8 annulus (exact rim)")
    ax.axhline(0.05, color="crimson", ls=":", label="pool tolerance 5%")
    ax.set_xlabel("discretisation size h  (1/n)")
    ax.set_ylabel(r"$|K_t - 3| / 3$")
    ax.set_title("Kirsch plate: error against the analytic $K_t = 3$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    FIGURES.append(str(path.relative_to(repo_root())))


def figure_profile(case: dict, ratio: float, path: Path) -> None:
    plt = pyplot()
    profile = np.asarray(case["hole_edge_hoop_profile_over_tension"], dtype=float)
    th = np.linspace(0.0, 0.5 * math.pi, 401)
    _, stt, _ = kirsch_stress(np.ones_like(th), th)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(np.degrees(th), stt, "-", lw=2, label=r"Kirsch exact $1 - 2\cos 2\theta$")
    ax.plot(profile[:, 0], profile[:, 1], "o", ms=3.2,
            label=f"FE Q4, d/W = {ratio:g}, n = {case['n_boundary']}")
    ax.axhline(3.0, color="crimson", ls=":", label="$K_t = 3$")
    ax.set_xlabel(r"$\theta$ from the load axis (deg)")
    ax.set_ylabel(r"$\sigma_{\theta\theta}(a,\theta) / T$")
    ax.set_title("Hole-edge hoop stress: field-level comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    FIGURES.append(str(path.relative_to(repo_root())))


def figure_width_trend(trend_rows: list, path: Path) -> None:
    plt = pyplot()
    ratios = [r["ratio"] for r in trend_rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ratios, [r["ktg_fe"] for r in trend_rows], "o-", label=r"FE $K_t$ (gross basis)")
    ax.plot(ratios, [r["heywood_ktg"] for r in trend_rows], "s--", label="Howland/Heywood (gross)")
    ax.plot(ratios, [r["ktn_fe"] for r in trend_rows], "o-", color="darkorange", label=r"FE $K_{tn}$ (net basis)")
    ax.plot(ratios, [r["heywood_ktn"] for r in trend_rows], "s--", color="saddlebrown", label="Howland/Heywood (net)")
    ax.axhline(3.0, color="crimson", ls=":", label="Kirsch limit 3")
    ax.set_xlabel("d / W")
    ax.set_ylabel("stress concentration factor")
    ax.set_title(f"Finite width, uniaxial tension (n = {TREND_LEVEL})")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    FIGURES.append(str(path.relative_to(repo_root())))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_gates(oracle, summary, ladder_rows, q8_rows, nu_rows, l_rows, trend_rows) -> list:
    fine_anchor = [r for r in ladder_rows
                   if abs(r["ratio"] - HEADLINE_RATIO) < 1e-12 and r["n_boundary"] == LADDER[-1]][0]
    coarse_anchor = [r for r in ladder_rows
                     if abs(r["ratio"] - HEADLINE_RATIO) < 1e-12 and r["n_boundary"] == LADDER[0]][0]
    q8_fine = q8_rows[-1]
    q8_rich = richardson(*[r["kt"] for r in q8_rows[-3:]], refinement_ratio=2.0)
    nu_ref = [r for r in nu_rows if r["poisson"] == 0.30 and abs(r["young_pa"] - 2.0e11) < 1.0][0]["ktg_fe"]
    nu_spread = max(abs(r["ktg_fe"] - nu_ref) / nu_ref for r in nu_rows)
    l_by = {r["half_length_over_width"]: r["ktg_fe"] for r in l_rows}
    band_keys = (1.0, 1.5, 2.5)
    band_values = [l_by[k] for k in band_keys]
    l_designated = (max(band_values) - min(band_values)) / l_by[2.5]
    trend_monotone = all(b["ktg_fe"] > a["ktg_fe"] for a, b in zip(trend_rows, trend_rows[1:]))
    trend_err = max(r["ktg_error_vs_heywood"] for r in trend_rows)
    load_max = max([r["load_balance"]["relative_error"] for r in ladder_rows]
                   + [r["load_balance_rel"] for r in nu_rows]
                   + [r["load_balance_rel"] for r in l_rows])
    orders = [summary[f"{r:.2f}"]["richardson"]["observed_order"] for r in RATIOS]
    orders = [o for o in orders if math.isfinite(o)]
    return [
        Gate("oracle_selfcheck", "hole-edge traction |sigma_rr|,|sigma_rt| / T of the analytic oracle",
             "Kirsch (1898) closed form; Timoshenko & Goodier 3rd ed. Art. 39 eqs. (a)-(c)",
             oracle["hole_edge_traction_free_max"], 1.0e-12),
        Gate("oracle_kt_value", "analytic K_t from the cited closed form",
             "sigma_tt(a,pi/2)/T = 3", oracle["kt_exact"], 3.0, ">=",
             note="self-consistency: the oracle evaluates to 3 before any FE is run"),
        Gate("kirsch_limit", f"|K_t(d/W={HEADLINE_RATIO:g}, n={LADDER[-1]}) - 3|/3",
             "Kirsch limit 3 (pool tolerance)", fine_anchor["ktg_error_vs_kirsch"],
             CRITERIA["kirsch_limit_rel_error_max"]),
        Gate("kirsch_extrapolated", f"|Richardson limit of the d/W={HEADLINE_RATIO:g} ladder - 3|/3",
             "Kirsch limit 3", summary[f"{HEADLINE_RATIO:.2f}"]["richardson_error_vs_kirsch"],
             CRITERIA["kirsch_extrapolated_rel_error_max"]),
        Gate("mesh_convergence_order", "observed order p of the K_t ladder (min over d/W)",
             "p >= 1 for Q4 recovered stress", min(orders), CRITERIA["observed_order_min"], ">="),
        Gate("mesh_error_monotone", "coarse/fine error ratio at d/W = 0.02",
             "error must fall with refinement",
             fine_anchor["ktg_error_vs_kirsch"] / coarse_anchor["ktg_error_vs_kirsch"],
             0.5, "<="),
        Gate("q8_crosscheck", "|K_t(Q8 annulus, finest) - 3|/3",
             "Kirsch limit 3, independent element family", q8_fine["error_vs_kirsch"],
             CRITERIA["q8_crosscheck_rel_error_max"]),
        Gate("q8_convergence_order", "observed order p of the Q8 annulus ladder",
             "p >= 1.5 for quadratic elements", q8_rich["observed_order"],
             CRITERIA["q8_observed_order_min"], ">="),
        Gate("nu_E_invariance", "max relative spread of K_t over nu in {0, .3, .45} and E in {2, 200} GPa",
             "Kirsch K_t is independent of nu and E", nu_spread,
             CRITERIA["nu_E_invariance_rel_max"]),
        Gate("saint_venant_length",
             "K_t band over resolution-matched L/W in {1.0, 1.5, 2.5}, relative to L/W = 2.5",
             "load introduction must not move K_t", l_designated,
             CRITERIA["saint_venant_band_rel_max"],
             note="resolution-matched arms (n ~ L); the fixed-n variant drifts several percent "
                  "purely from mesh coarsening and is reported separately as a confounder"),
        Gate("finite_width_trend", "max |K_tg - Howland/Heywood gross| / value over d/W 0.1..0.5",
             "Heywood (1952) fit K_tn = 2 + (1-d/W)^3", trend_err,
             CRITERIA["finite_width_trend_rel_max"]),
        Gate("finite_width_monotone", "gross K_t strictly increasing over d/W 0.1..0.5",
             "Howland trend", 1.0 if trend_monotone else 0.0, 1.0, ">="),
        Gate("hoop_profile_shape", f"RMS of (sigma_tt(a,theta) - exact) / 3 at d/W = {HEADLINE_RATIO:g}",
             "Kirsch edge law 1 - 2 cos 2 theta", fine_anchor["hoop_profile_rms_over_kt"],
             CRITERIA["hoop_profile_rms_over_kt_max"]),
        Gate("load_balance", "max |applied - reacted| / applied",
             "statics on the symmetry plane", load_max, CRITERIA["load_balance_rel_max"]),
    ]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                        help="directory for result.json / gates_verified.json / PNGs")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    oracle = oracle_selfcheck()
    print(f"[oracle] K_t analytic = {oracle['kt_exact']:.12f} "
          f"(edge traction {oracle['hole_edge_traction_free_max']:.2e})")

    ladder_rows = kirsch_limit_ladder()
    for row in ladder_rows:
        print(f"[plate] d/W={row['ratio']:.2f} n={row['n_boundary']:3d} "
              f"K_t={row['ktg_fe']:.6f} err={100 * row['ktg_error_vs_kirsch']:.3f}%")
    summary = ladder_summary(ladder_rows)

    q8_rows = element_order_crosscheck()
    for row in q8_rows:
        print(f"[Q8] {row['n_theta']}x{row['n_radial']} K_t={row['kt']:.6f} "
              f"err={100 * row['error_vs_kirsch']:.3f}%")

    nu_rows = nu_e_invariance()
    l_rows = saint_venant_study(resolution_matched=True)
    l_unmatched = saint_venant_study(resolution_matched=False)
    trend_rows = finite_width_trend()

    gates = build_gates(oracle, summary, ladder_rows, q8_rows, nu_rows, l_rows, trend_rows)

    headline = [r for r in ladder_rows
                if abs(r["ratio"] - HEADLINE_RATIO) < 1e-12 and r["n_boundary"] == LADDER[-1]][0]
    figure_field(headline["_case"], HEADLINE_RATIO, out / "kirsch_field.png")
    figure_convergence(ladder_rows, summary, q8_rows, out / "kirsch_convergence.png")
    figure_profile(headline["_case"], HEADLINE_RATIO, out / "kirsch_hoop_profile.png")
    figure_width_trend(trend_rows, out / "kirsch_width_trend.png")

    slim = lambda rows: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    result = {
        "benchmark": ANCHOR,
        "mission": MISSION,
        "title": TITLE,
        "generated_at": utc_now(),
        "reference": {
            "citation": "Kirsch (1898) VDI 42, 797; Timoshenko & Goodier, Theory of Elasticity, "
                        "3rd ed., Art. 39, eqs. (a)-(c)",
            "closed_form": "sigma_tt(a, theta) = T (1 - 2 cos 2 theta); K_t = sigma_tt(a, pi/2)/T = 3",
            "basis": "gross stress T (far-field tension); net basis K_tn = K_tg (1 - d/W)",
            "value": 3.0,
            "hand_copied_digits": None,
            "note": "K_t = 3 is evaluated from the cited closed form and the oracle is "
                    "self-checked (traction-free edge, uniaxial far field) inside oracle_selfcheck",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorfem": __version__,
            "machine": platform.machine(),
            "dtype": "float64",
            "solver": "tensorfem.continuum.solve_continuum (sparse direct) / "
                      "tensorfem.quadratic_plane.solve_q8_plane",
        },
        "model": {
            "geometry": "quarter plate 0<=x<=L, 0<=y<=W/2, hole quadrant of radius a=d/2 at origin",
            "boundary_conditions": "u_x=0 on x=0; u_y=0 on y=0; hole edge and y=W/2 traction free; "
                                   "uniform tension T on x=L",
            "half_length_over_width": HALF_LENGTH_OVER_WIDTH,
            "grading": GRADING,
            "element": "Q4 plane stress, starburst mesh graded to the hole (production path)",
            "crosscheck_element": "Q8 serendipity plane stress, polar annulus, exact Kirsch "
                                  "displacement on r = R_out",
            "ratio_definition": "ratio = d/W",
        },
        "oracle_selfcheck": oracle,
        "ladder": slim(ladder_rows),
        "ladder_summary": summary,
        "q8_crosscheck": q8_rows,
        "nu_E_invariance": nu_rows,
        "saint_venant": l_rows,
        "saint_venant_fixed_n_confounder": l_unmatched,
        "finite_width_trend": trend_rows,
        "figures": FIGURES,
        "repro_cmd": "OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py",
    }
    write_json(out / "result.json", result)

    manifest = gate_manifest(
        anchor=ANCHOR,
        title=TITLE,
        criteria=CRITERIA,
        gates=gates,
        artifacts={
            "result.json": "every number of the ladder, cross-checks and sensitivity studies",
            "figures": FIGURES,
            "run.py": "this driver",
        },
        limitations=[
            "K_t = 3 is the infinite-plate limit; the finite-width plate is pushed to d/W = 0.02 "
            "and Richardson-extrapolated in h, and the d/W trend is graded against the "
            "Howland/Heywood fit rather than claimed to be 3.",
            "The Q8 annulus check prescribes the exact Kirsch displacement on r = 2a, so it "
            "certifies the discretisation only and cannot see the far-field/finite-width model error.",
            "Plane stress, linear elasticity, isotropic homogeneous material, single hole, "
            "no plasticity and no contact: this anchor says nothing about those regimes.",
            "The plate-length study must be resolution-matched (n ~ L): with the ray count held "
            "fixed the starburst mesh coarsens near the apex and K_t drifts by percent-level in a "
            "way that mimics a load-introduction effect. Both tables are in result.json.",
            "The hole edge is a stress-raiser: the Q4 recovered nodal stress converges at an "
            "observed order near 2 on this mesh family, which is a property of the starburst "
            "mesh plus apex node, not a general O(h^2) claim.",
        ],
        mission=MISSION,
    )
    write_json(out / "gates_verified.json", manifest)
    print(f"[gates] passed={manifest['passed']} failed={manifest['failed_gates']}")
    for gate in manifest["gates"]:
        print(f"  {gate['id']:24s} value={gate['value']:.6g} thr={gate['threshold']:.6g} "
              f"{'OK' if gate['passed'] else 'FAIL'}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())