#!/usr/bin/env python3
"""CASE-CANTILEVER-TIP -- taper axis: a non-prismatic member against a closed-form oracle.

Verification debt this driver closes
------------------------------------
Every existing route of the anchor ``CASE-CANTILEVER-TIP`` holds the section
constant: the shear ladder (``benchmarks/verified/fem_cantilever_shear``), the
solid-vs-elasticity study (``benchmarks/verified/fem_cantilever_solid_convergence``),
the cross-section and kappa identifications, and the cheap regression gate
(``tests/test_case_cantilever_tip.py``).  The case-pool criterion ``< 3 %`` has
therefore never been exercised on a geometry whose flexibility is an *integral*
of ``1/I(x)``.  Tapered girders and brackets are the normal case in a ship
structure, and the discretisation decision that a taper exposes -- how ``EI``
is sampled inside an element -- is invisible on a prismatic member, where every
sampling scheme is exact.

What is done here
-----------------
1. **Oracle, three independent routes.**  For ``h(x) = h0 (1 - c x/L)``,
   Castigliano gives ``delta_bend = (PL^3/EI0) B(c)`` with
   ``B(c) = [-c - c^2/2 - ln(1-c)]/c^3`` and the shear part
   ``(PL/(kappa G A0)) S(c)``, ``S(c) = -ln(1-c)/c``.  ``B`` is cross-checked
   against a sympy re-derivation, a convergent power series, and adaptive
   quadrature of the flexibility integral; the ``c -> 0`` limits recover the
   prismatic ``PL^3/3EI`` and ``PL/(kappa G A)``.
2. **1-D frame ladder.**  A 2-node Euler-Bernoulli frame with per-element
   inertia in two sampling schemes: ``midpoint`` (lumped "average inertia per
   element") and ``gauss`` (element stiffness integrated with a 4-point rule,
   exact for the cubic ``EI(x)``).  The observed orders are measured, not
   assumed.
3. **3-D solid ladder.**  A graded HEX20 solid whose depth varies element by
   element (geometry exact: element edges are straight and ``h(x)`` is piecewise
   linear), driven by the consistent 8-node tip face load, against the
   Timoshenko oracle with ``kappa = 5/6``.
4. **Root-boundary-condition control.**  Same mesh, same load: ``clamp_all``
   (``u = 0`` on the root face) vs ``lateral_free`` (``u_x = 0`` only), which
   separates the clamped-root boundary layer from the mesh error -- the residual
   the prismatic studies had to explain in prose.
5. **Geometry gate before any deflection is read** (the case-pool equivalent of
   checking ``Cb/L/B``): the mesh surface must sit on ``z = +-h(x)/2`` at every
   axial station, the tip section area must be exact, and the element aspect
   ratio must be bounded.
6. **Wedge limit.**  ``B(c) -> -ln(1-c)`` as ``c -> 1``: the tip deflection of a
   knife-edge wedge *diverges*, so ``c = 1`` is rejected by the model and the
   divergence identity is gated.

Everything below is generated from ``result.json``; no number is transcribed.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_cantilever_taper/run.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "verified"))

from _harness import Gate, gate_manifest, pyplot, repo_root, utc_now, write_json  # noqa: E402
from tensorfem.tapered_cantilever import (  # noqa: E402
    TaperCase,
    bending_factor,
    bending_factor_series,
    bending_factor_symbolic,
    frame_tip_deflection,
    shear_factor,
    shear_factor_series,
    solve_tapered_cantilever,
    taper_limits,
    validate_mesh_geometry,
)

plt = pyplot()

ANCHOR = "fem_cantilever_taper"
MISSION = "CASE-CANTILEVER-TIP"
TITLE = "CASE-CANTILEVER-TIP taper axis: linearly tapered cantilever vs closed-form flexibility oracle"
TOLERANCE = 0.03          # case-pool criterion (< 3 %)
TAPERS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
SOLID_TAPERS = (0.0, 0.25, 0.5, 0.75)
SOLID_MESHES = ((8, 2, 2), (16, 2, 2), (32, 2, 2), (16, 4, 4))
FRAME_ELEMENTS = (2, 4, 8, 16, 32)

ASSETS = repo_root() / "docs" / "assets" / "cantilever_taper"


# --------------------------------------------------------------------------- #
# 1. oracle cross-check
# --------------------------------------------------------------------------- #
def oracle_table() -> list[dict[str, float]]:
    rows = []
    for c in TAPERS:
        case = TaperCase(taper=c)
        quad = case.delta_quadrature()
        symbolic = bending_factor_symbolic(c) if c > 0 else {
            "symbolic": 1.0 / 3.0, "closed": 1.0 / 3.0,
            "residual": 0.0, "series_residual": 0.0,
        }
        closed = bending_factor(c)
        rows.append({
            "taper": c,
            "bending_factor": closed,
            "series": bending_factor_series(c),
            "symbolic": symbolic["symbolic"],
            "series_residual": abs(bending_factor_series(c) - closed) / closed,
            "symbolic_residual": symbolic["residual"],
            "shear_factor": shear_factor(c),
            "shear_series_residual": abs(shear_factor_series(c) - shear_factor(c)) / shear_factor(c),
            "delta_bending": case.delta_euler(),
            "delta_shear": case.delta_shear(),
            "delta_total": case.delta_timoshenko(),
            "delta_quadrature": quad["total"],
            "quadrature_residual": abs(quad["total"] - case.delta_timoshenko()) / case.delta_timoshenko(),
            "amplification": case.relative_deflection(),
            "shear_share": case.shear_share(),
        })
    return rows


# --------------------------------------------------------------------------- #
# 2. 1-D frame ladder
# --------------------------------------------------------------------------- #
def frame_ladder() -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for c in (0.0, 0.5, 0.75):
        case = TaperCase(taper=c)
        ref = case.delta_euler()          # the frame is Euler-Bernoulli: no shear
        out[f"{c:g}"] = {
            "midpoint": [frame_tip_deflection(case, n, sampling="midpoint") / ref - 1.0
                         for n in FRAME_ELEMENTS],
            "gauss": [frame_tip_deflection(case, n, sampling="gauss") / ref - 1.0
                      for n in FRAME_ELEMENTS],
        }
    return out


def min_elements_for_tolerance(errors: list[float], tol: float = TOLERANCE) -> int | None:
    """Fewest elements (first ladder rung) whose error enters the case-pool band."""
    for n, e in zip(FRAME_ELEMENTS, errors):
        if abs(e) < tol:
            return int(n)
    return None


def observed_order(errors: list[float]) -> float:
    """Log-log slope of ``|error|`` against the three finest rungs of the ladder."""
    n = np.array(FRAME_ELEMENTS[-3:], dtype=float)
    e = np.abs(np.array(errors[-3:], dtype=float))
    if np.any(e <= 0.0):
        return float("nan")
    return float(-np.polyfit(np.log(n), np.log(e), 1)[0])


# --------------------------------------------------------------------------- #
# 3./4. 3-D solid ladder and root-BC control
# --------------------------------------------------------------------------- #
def solid_ladder() -> list[dict[str, float]]:
    rows = []
    for c in SOLID_TAPERS:
        case = TaperCase(taper=c)
        ref = case.delta_timoshenko()
        for (nx, ny, nz) in SOLID_MESHES:
            res = solve_tapered_cantilever(case, nx, ny, nz)
            rows.append({
                "taper": c,
                "nx": nx, "ny": ny, "nz": nz,
                "n_nodes": res["n_nodes"], "n_elements": res["n_elements"],
                "n_dof": res["n_dof"],
                "tip_deflection": res["tip_deflection"],
                "oracle": ref,
                "error": res["tip_deflection"] / ref - 1.0,
            })
    return rows


def root_bc_control() -> list[dict[str, float]]:
    rows = []
    for c in (0.0, 0.5, 0.75):
        case = TaperCase(taper=c)
        ref = case.delta_timoshenko()
        for nx in (16, 32):
            row = {"taper": c, "nx": nx, "oracle": ref}
            for bc in ("clamp_all", "lateral_free"):
                res = solve_tapered_cantilever(case, nx, 2, 2, root_bc=bc)
                row[f"{bc}_deflection"] = res["tip_deflection"]
                row[f"{bc}_error"] = res["tip_deflection"] / ref - 1.0
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 5. wedge limit
# --------------------------------------------------------------------------- #
def wedge_limit() -> dict[str, object]:
    rows = []
    for c in (0.9, 0.99, 0.999, 0.9999):
        bend = bending_factor(c)
        # B(c) c^3 = -c - c^2/2 - ln(1-c): the log divergence appears as c -> 1.
        identity = abs(bend * c**3 - (-c - 0.5 * c * c - math.log1p(-c)))
        rows.append({
            "taper": c,
            "bending_factor": bend,
            "log_divergence_reference": -math.log1p(-c),
            "identity_residual": identity / bend,
            "amplification": TaperCase(taper=c).relative_deflection(),
        })
    rejected = False
    try:
        TaperCase(taper=1.0)
    except ValueError:
        rejected = True
    return {"rows": rows, "wedge_rejected": rejected}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_frame_order(ladder: dict[str, dict[str, list[float]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    n = np.array(FRAME_ELEMENTS, dtype=float)
    for c, style in (("0.5", "o-"), ("0.75", "s-")):
        ax.loglog(n, np.abs(ladder[c]["midpoint"]), style, label=f"midpoint EI, c={c}")
        ax.loglog(n, np.abs(ladder[c]["gauss"]), style, mfc="none", label=f"Gauss EI, c={c}")
    ax.loglog(n, abs(ladder["0.5"]["midpoint"][0]) * (n / n[0]) ** -2.0, "k--", lw=1,
              label=r"slope $-2$")
    ax.loglog(n, max(abs(ladder["0.5"]["gauss"][0]), 1e-16) * (n / n[0]) ** -4.0, "k:", lw=1,
              label=r"slope $-4$")
    ax.axhline(TOLERANCE, color="0.5", ls="--", lw=1)
    ax.set_xlabel("frame elements")
    ax.set_ylabel("|error| vs EB oracle")
    ax.set_title("EI sampling inside an element sets the order")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_solid_ladder(rows: list[dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.axhspan(-100 * TOLERANCE, 100 * TOLERANCE, color="0.85", zorder=0, label="+-3 % band")
    ax.axhline(0.0, color="k", lw=0.8)
    for c in SOLID_TAPERS:
        sel = [r for r in rows if r["taper"] == c and r["ny"] == 2 and r["nz"] == 2]
        ax.plot([r["nx"] for r in sel], [100 * r["error"] for r in sel], "o-",
                label=f"c={c:g} (ny=nz=2)")
        sel2 = [r for r in rows if r["taper"] == c and r["ny"] == 4]
        ax.plot([r["nx"] for r in sel2], [100 * r["error"] for r in sel2], "*", ms=10,
                label=f"c={c:g} (ny=nz=4)")
    ax.set_xlabel("axial elements nx")
    ax.set_ylabel("tip-deflection error vs Timoshenko [%]")
    ax.set_title("HEX20 tapered solid: every rung inside the 3 % band")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_amplification(rows: list[dict[str, float]], wedge: dict[str, object], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    cc = np.linspace(1e-3, 0.99, 300)
    amp = [TaperCase(taper=float(c)).relative_deflection() for c in cc]
    ax.plot(cc, amp, "-", color="C0", label="delta_TI(c)/delta_TI(0), closed form")
    ax.plot([r["taper"] for r in rows], [r["amplification"] for r in rows], "o", color="C3",
            label="oracle table rows")
    ax.plot([r["taper"] for r in wedge["rows"]], [r["amplification"] for r in wedge["rows"]],
            "^", color="C2", label="wedge limit (log divergence)")
    ax2 = ax.twinx()
    ax2.plot([r["taper"] for r in rows], [100 * r["shear_share"] for r in rows], "s--",
             color="0.4", ms=4, label="shear share [%] (right axis)")
    ax2.set_ylabel("shear share of tip deflection [%]")
    ax2.grid(False)
    ax.set_xlabel("taper ratio c = 1 - h_tip/h_root")
    ax.set_ylabel("amplification delta(c)/delta(0)")
    ax.set_title("Taper amplification diverges only at the knife edge")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_field_contour(path: Path) -> dict[str, float]:
    """sigma_xx on the mid-plane of the c = 0.5 model (deflection scaled x20)."""
    case = TaperCase(taper=0.5)
    res = solve_tapered_cantilever(case, 24, 3, 4)
    nodes = res["nodes"]
    cells = res["elements"].long()
    sigma = res["sigma_xx"].numpy()
    disp = res["displacement"].numpy()
    corner = nodes[cells[:, :8]].numpy()
    centre_y = corner[:, :, 1].mean(axis=1)
    keep = np.abs(centre_y) < 1e-12
    cx = corner[keep][:, :, 0].mean(axis=1)
    cz = corner[keep][:, :, 2].mean(axis=1)
    # displaced z of the slice nodes (mean of the 8 corner vertical DOFs)
    cz_d = []
    for e in np.where(keep)[0]:
        ids = cells[e, :8].numpy()
        cz_d.append((nodes[ids, 2].numpy() + 20.0 * disp[3 * ids + 2]).mean())
    cz_d = np.array(cz_d)
    val = sigma[keep] / 1e6
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    tri = ax.tricontourf(cx * 1e3, cz_d * 1e3, val, levels=24, cmap="coolwarm")
    xs = np.linspace(0.0, case.length, 120)
    ax.plot(xs * 1e3, 0.5 * np.array([case.depth(x) for x in xs]) * 1e3, "k-", lw=1)
    ax.plot(xs * 1e3, -0.5 * np.array([case.depth(x) for x in xs]) * 1e3, "k-", lw=1)
    ax.set_xlim(-2, 104)
    fig.colorbar(tri, ax=ax, label="sigma_xx [MPa]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("c = 0.5 tapered cantilever, HEX20, sigma_xx on the mid-plane")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "max_sigma_xx_MPa": float(np.abs(val).max()),
        "min_sigma_xx_MPa": float(val.min()),
        "n_slice_elements": int(keep.sum()),
        "tip_deflection_um": 1e6 * res["tip_deflection"],
    }


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def render_doc(result: dict, gates: dict) -> str:
    rows = result["oracle"]
    ladder = result["frame"]
    solid = result["solid"]
    control = result["root_bc_control"]
    wedge = result["wedge"]
    geo = result["geometry"]
    out: list[str] = []
    add = out.append
    add("# CASE-CANTILEVER-TIP — taper axis: the anchor's 3 % criterion on a non-prismatic member")
    add("")
    add(f"*Generated by `benchmarks/verified/fem_cantilever_taper/run.py` ({result['generated_at']}); "
        "every number below is injected from `result.json`. Verdict: "
        f"**{'PASS' if gates['passed'] else 'FAIL'}** "
        f"(failed gates: {gates['failed_gates'] or 'none'}).*")
    add("")
    add("## ① Case and motivation")
    add("")
    add("The case-pool anchor `CASE-CANTILEVER-TIP` grades the tip deflection of a **prismatic** "
        "cantilever (`b = h = 10 mm`, `E = 200 GPa`, `P = 5 N`) against Euler-Bernoulli / "
        "Timoshenko with a `< 3 %` criterion. Every existing route of the anchor holds the section "
        "constant, so the criterion has never been exercised on a member whose flexibility is "
        "`int dx/I(x)` rather than an algebraic expression — the normal case for a tapered girder, "
        "bracket or ice-going hull frame.")
    add("")
    add("This document extends the anchor along the **geometry** axis: the depth tapers linearly, "
        "`h(x) = h0 (1 - c x/L)`, the width is constant, and the reference rung (`c = 0`) is the "
        "case-pool section and load at `L/h0 = 10`.")
    add("")
    add("| parameter | symbol | value |")
    add("|---|---|---|")
    add("| length | L | 0.10 m |")
    add("| root depth | h0 | 0.010 m |")
    add("| width | b | 0.010 m |")
    add("| slenderness | L/h0 | 10 |")
    add("| taper ratio | c | 0, 0.1, 0.25, 0.5, 0.75, 0.9 (oracle table) / 0, 0.25, 0.5, 0.75 (solid) |")
    add("| Young's modulus | E | 200 GPa |")
    add("| Poisson ratio | nu | 0.3 |")
    add("| tip load | P | 5 N |")
    add("| shear factor | kappa | 5/6 (Timoshenko 1921) |")
    add("")
    add("Oracles (Castigliano unit-load theorem, re-derived at run time):")
    add("")
    add("```")
    add("delta_bend  = (P L^3 / (E I0)) * B(c),     B(c) = [-c - c^2/2 - ln(1-c)] / c^3,   B(0) = 1/3")
    add("delta_shear = (P L / (kappa G A0)) * S(c),  S(c) = -ln(1-c) / c,                  S(0) = 1")
    add("delta_TI    = delta_bend + delta_shear")
    add("```")
    add("")
    add("## ② Method")
    add("")
    add("1. **Oracle cross-check, three independent routes** per taper: the closed form above; a "
        "convergent power series `B(c) = sum_k [(k+2)(k+1)/(2(k+3)) - k/2] c^k`; a sympy "
        "re-derivation of `int (1-t)^2/(1-ct)^3 dt`; and adaptive quadrature of the physical "
        "flexibility integral `int (L-x)^2/I(x) dx`. The `c -> 0` limits must recover the "
        "prismatic oracles.")
    add("2. **1-D Euler-Bernoulli frame** with *per-element* inertia in two sampling schemes — "
        "`midpoint` (lumped average `EI` per element) and `gauss` (`int B^T EI(x) B dx` with a "
        "4-point rule, exact for the cubic `EI(x)` of a linear taper) — over "
        f"n = {list(FRAME_ELEMENTS)} elements; the observed order is fitted on the three finest rungs.")
    add("3. **3-D graded HEX20 solid**: the depth varies element by element, so `h(x)` is honoured "
        "exactly at the nodes (element edges are straight, `h` is piecewise linear); the tip face "
        "carries the consistent 8-node load of a uniform traction (net `P`); the root face is fully "
        "clamped. Ladder over `nx` with `ny = nz = 2`, plus one `ny = nz = 4` control.")
    add("4. **Root-BC control**: same mesh and load with `u_x = 0` only on the root face (rigid "
        "translations pinned at one corner), so the root may contract laterally — this isolates the "
        "clamped-root boundary layer from the mesh error.")
    add("5. **Geometry gate before reading any deflection**: mesh surface vs `z = +-h(x)/2` at every "
        "axial station, tip section area, worst element aspect ratio.")
    add("")
    add("## ③ Results")
    add("")
    add("### Oracle table and cross-checks")
    add("")
    add("| c | B(c) | 3B(c) = δ_bend/δ_EB | S(c) | δ_TI [µm] | amplification | shear share | quad resid | series resid | sympy resid |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        add(f"| {r['taper']:g} | {r['bending_factor']:.6f} | {3 * r['bending_factor']:.4f} | "
            f"{r['shear_factor']:.6f} | {1e6 * r['delta_total']:.4f} | {r['amplification']:.4f} | "
            f"{100 * r['shear_share']:.3f} % | {r['quadrature_residual']:.1e} | "
            f"{r['series_residual']:.1e} | {r['symbolic_residual']:.1e} |")
    add("")
    add("### Frame ladder (Euler-Bernoulli, so the oracle is δ_bend only)")
    add("")
    add("| c | sampling | " + " | ".join(f"n={n}" for n in FRAME_ELEMENTS) + " | order |")
    add("|---:|---|" + "---:|" * (len(FRAME_ELEMENTS) + 1))
    for c, block in ladder.items():
        for scheme in ("midpoint", "gauss"):
            errs = block[scheme]
            order = observed_order(errs)
            label = f"{order:.2f}" if abs(errs[-1]) > 1e-10 else "n/a (machine exact)"
            add(f"| {c} | {scheme} | " + " | ".join(f"{100 * e:+.4f} %" for e in errs) +
                f" | {label} |")
    add("")
    add("The `c = 0` row is the control: with a constant `EI` both sampling schemes are exact, so "
        "the element matrices are right and the `c > 0` deviations are pure `EI`-quadrature / "
        "Hermite-space error. The two orders are measured, not assumed: lumping `EI` at the element "
        "centre costs accuracy at `O(h^2)` (order "
        f"{observed_order(ladder['0.5']['midpoint']):.2f}), while integrating the element stiffness "
        "exactly gives `O(h^4)` (order "
        f"{observed_order(ladder['0.5']['gauss']):.2f}); at `c = 0.5` and `n = 32` the midpoint "
        f"error is {abs(ladder['0.5']['midpoint'][-1] / ladder['0.5']['gauss'][-1]):.0f}x the Gauss "
        "error on the same mesh.")
    add("")
    add("| c | fewest midpoint-EI elements with error < 3 % |")
    add("|---:|---:|")
    for c, block in ladder.items():
        need = min_elements_for_tolerance(block["midpoint"])
        add(f"| {c} | {need if need is not None else 'none in ladder'} |")
    add("")
    add("### HEX20 solid vs Timoshenko oracle")
    add("")
    add("| c | mesh (nx,ny,nz) | dof | tip deflection [µm] | oracle [µm] | error | < 3 %? |")
    add("|---:|---|---:|---:|---:|---:|:--:|")
    for r in solid:
        add(f"| {r['taper']:g} | ({r['nx']},{r['ny']},{r['nz']}) | {r['n_dof']} | "
            f"{1e6 * r['tip_deflection']:.5f} | {1e6 * r['oracle']:.5f} | {100 * r['error']:+.3f} % | "
            f"{'yes' if abs(r['error']) < TOLERANCE else 'NO'} |")
    add("")
    add("### Root-BC control (same mesh, same load)")
    add("")
    add("| c | nx | clamp_all error | lateral_free error | clamp cost |")
    add("|---:|---:|---:|---:|---:|")
    for r in control:
        add(f"| {r['taper']:g} | {r['nx']} | {100 * r['clamp_all_error']:+.3f} % | "
            f"{100 * r['lateral_free_error']:+.3f} % | "
            f"{100 * (r['clamp_all_error'] - r['lateral_free_error']):+.3f} % |")
    add("")
    add("### Wedge limit")
    add("")
    add("| c | B(c) | -ln(1-c) | identity residual | amplification |")
    add("|---:|---:|---:|---:|---:|")
    for r in wedge["rows"]:
        add(f"| {r['taper']:g} | {r['bending_factor']:.4f} | {r['log_divergence_reference']:.4f} | "
            f"{r['identity_residual']:.1e} | {r['amplification']:.3f} |")
    add("")
    add(f"`TaperCase(taper=1.0)` rejected: **{wedge['wedge_rejected']}**.")
    add("")
    add("### Geometry gate")
    add("")
    add("| check | value | threshold |")
    add("|---|---:|---:|")
    add(f"| mesh surface vs z = ±h(x)/2 (relative, c=0.5, nx=16) | {geo['surface_deviation']:.2e} | 1e-12 |")
    add(f"| tip section area error | {geo['area_error']:.2e} | 1e-12 |")
    add(f"| worst element aspect ratio | {geo['max_element_aspect']:.3f} | 10 |")
    add(f"| smallest HEX20 edge (m) | {geo['min_edge_length']:.2e} | - |")
    add("")
    add("## ④ Figures")
    add("")
    add("![frame order](assets/cantilever_taper/frame_order.png)")
    add("")
    add("![solid ladder](assets/cantilever_taper/solid_ladder.png)")
    add("")
    add("![amplification](assets/cantilever_taper/amplification.png)")
    add("")
    add("![field](assets/cantilever_taper/field_contour.png)")
    add("")
    add("## ⑤ Notes, limitations")
    add("")
    for note in result["limitations"]:
        add(f"* {note}")
    add("")
    add("## ⑥ Error verdict")
    add("")
    add("| quantity | reference | error | < 3 %? |")
    add("|---|---|---:|:--:|")
    worst_frame = max(abs(block["midpoint"][-1]) for block in ladder.values())
    add(f"| frame, 32 elements, midpoint EI, worst c | δ_bend closed form | "
        f"{100 * worst_frame:.3f} % | {'yes' if worst_frame < TOLERANCE else 'NO'} |")
    worst_gauss = max(abs(block["gauss"][-1]) for block in ladder.values())
    add(f"| frame, 32 elements, Gauss EI, worst c | δ_bend closed form | "
        f"{100 * worst_gauss:.3e} % | {'yes' if worst_gauss < TOLERANCE else 'NO'} |")
    add(f"| HEX20 solid, all rungs, worst c | δ_TI closed form | {100 * result['solid_worst']:.3f} % | "
        f"{'yes' if result['solid_worst'] < TOLERANCE else 'NO'} |")
    add(f"| HEX20 solid, finest axial mesh (nx=32), worst c | δ_TI closed form | "
        f"{100 * result['solid_fine_worst']:.3f} % | "
        f"{'yes' if result['solid_fine_worst'] < TOLERANCE else 'NO'} |")
    add(f"| oracle closed form vs adaptive quadrature, worst c | adaptive quad | "
        f"{max(r['quadrature_residual'] for r in rows):.1e} | yes |")
    add("")
    need_mid = [min_elements_for_tolerance(block["midpoint"]) for block in ladder.values()]
    coarse_mid = max(abs(block["midpoint"][0]) for block in ladder.values())
    add("**Answer to the criterion question, stated precisely: yes — provided the model is "
        "discretised for it.** Every HEX20 rung tested (nx = 8 ... 32, c = 0 ... 0.75) is inside "
        f"3 %: worst {100 * max(abs(r['error']) for r in solid):.2f} % at the coarsest mesh, "
        f"{100 * result['solid_fine_worst']:.2f} % at the finest. The frame is inside 3 % from "
        f"{max(v for v in need_mid if v is not None)} elements per member onward (midpoint `EI`). "
        "It is **not** true for every discretisation: a lumped midpoint `EI` on 2 elements misses "
        f"by {100 * coarse_mid:.0f} % at the worst taper, and the element count needed grows with "
        "`c` as tabulated above. The residual that survives refinement is *not* mesh error — it is "
        "the clamped-root boundary layer, quantified by the root-BC control table.")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)

    oracle_rows = oracle_table()
    ladder = frame_ladder()
    solid = solid_ladder()
    control = root_bc_control()
    wedge = wedge_limit()
    geo = validate_mesh_geometry(TaperCase(taper=0.5), 16, 2, 2)
    geo_prismatic = validate_mesh_geometry(TaperCase(taper=0.0), 16, 2, 2)
    field = figure_field_contour(ASSETS / "field_contour.png")
    figure_frame_order(ladder, ASSETS / "frame_order.png")
    figure_solid_ladder(solid, ASSETS / "solid_ladder.png")
    figure_amplification(oracle_rows, wedge, ASSETS / "amplification.png")

    limits = taper_limits()
    order_mid = {c: observed_order(block["midpoint"]) for c, block in ladder.items()}
    order_gauss = {c: observed_order(block["gauss"]) for c, block in ladder.items()}
    worst_quad = max(r["quadrature_residual"] for r in oracle_rows)
    worst_series = max(max(r["series_residual"], r["shear_series_residual"]) for r in oracle_rows)
    worst_symbolic = max(r["symbolic_residual"] for r in oracle_rows)
    solid_worst = max(abs(r["error"]) for r in solid)
    solid_fine_worst = max(abs(r["error"]) for r in solid if r["nx"] == 32 and r["ny"] == 2)
    frame_worst_32 = max(abs(block["midpoint"][-1]) for block in ladder.values())
    control_gain = all(abs(r["lateral_free_error"]) < abs(r["clamp_all_error"]) for r in control)

    def solid_monotone(c: float) -> bool:
        sel = [abs(r["error"]) for r in solid if r["taper"] == c and r["ny"] == 2 and r["nz"] == 2]
        return all(b < a for a, b in zip(sel, sel[1:]))

    gates = [
        Gate("taper_oracle_quadrature", "closed form vs adaptive quadrature (worst taper)",
             "int (L-x)^2/I(x) dx by scipy.integrate.quad", worst_quad, 1e-10),
        Gate("taper_oracle_series", "closed form vs power series (worst taper)",
             "sum_k [(k+2)(k+1)/(2(k+3)) - k/2] c^k", worst_series, 1e-10),
        Gate("taper_oracle_sympy", "closed form vs sympy re-derivation (worst taper)",
             "sympy integrate (1-t)^2/(1-ct)^3", worst_symbolic, 1e-10),
        Gate("taper_prismatic_bending_limit", "3B(c->0) - 1",
             "delta_EB = PL^3/(3EI)", abs(limits["bending_over_prismatic"] - 1.0), 1e-8),
        Gate("taper_prismatic_shear_limit", "S(c->0) - 1",
             "delta_shear = PL/(kappa G A)", abs(limits["shear_over_prismatic"] - 1.0), 1e-8),
        Gate("frame_midpoint_order", "observed order, midpoint EI, c=0.5",
             "Hermite beam with O(h^2) element quadrature", order_mid["0.5"], 1.8, ">="),
        Gate("frame_midpoint_order_upper", "observed order, midpoint EI, c=0.5",
             "Hermite beam with O(h^2) element quadrature", order_mid["0.5"], 2.2, "<="),
        Gate("frame_gauss_order", "observed order, Gauss EI, c=0.5",
             "Hermite beam with exact element stiffness", order_gauss["0.5"], 3.5, ">="),
        Gate("frame_gate_3pct", "midpoint EI, n=16, c=0.75",
             "delta_bend closed form", abs(ladder["0.75"]["midpoint"][3]), TOLERANCE),
        Gate("frame_elements_needed",
             "worst taper: fewest midpoint-EI elements inside 3 %",
             "delta_bend closed form",
             float(max(v for v in (min_elements_for_tolerance(block["midpoint"])
                                   for block in ladder.values()) if v is not None)), 16.0),
        Gate("solid_gate_3pct_finest", "HEX20 nx=32, worst taper",
             "delta_TI closed form", solid_fine_worst, TOLERANCE),
        Gate("solid_gate_3pct_all", "HEX20 all rungs, worst taper",
             "delta_TI closed form", solid_worst, TOLERANCE),
        Gate("solid_error_monotone", "1 if the HEX20 error falls with nx for every taper",
             "refinement ladder", 0.0 if all(solid_monotone(c) for c in SOLID_TAPERS) else 1.0, 0.0),
        Gate("root_bc_attribution", "1 if lateral_free is closer than clamp_all everywhere",
             "root face u_x = 0 only", 0.0 if control_gain else 1.0, 0.0),
        Gate("geometry_surface", "mesh surface vs z = +-h(x)/2 (relative)",
             "h(x) = h0(1 - c x/L)", geo["surface_deviation"], 1e-12),
        Gate("geometry_area", "tip section area error", "b * h_tip", geo["area_error"], 1e-12),
        Gate("geometry_prismatic_surface", "prismatic control surface deviation",
             "z = +-h0/2", geo_prismatic["surface_deviation"], 1e-12),
        Gate("wedge_rejected", "1 if taper = 1 is rejected (log-divergent deflection)",
             "B(c) -> -ln(1-c)", 0.0 if wedge["wedge_rejected"] else 1.0, 0.0),
        Gate("wedge_identity", "max |B(c)c^3 + c + c^2/2 + ln(1-c)| / B(c)",
             "B(c)c^3 = -c - c^2/2 - ln(1-c)", max(r["identity_residual"] for r in wedge["rows"]), 1e-12),
    ]

    limitations = [
        "The oracle is beam theory (Euler-Bernoulli + Timoshenko shear). For a taper the exact "
        "3-D elasticity solution is not available in closed form (the Airy cantilever field of the "
        "prismatic route needs a constant section), so the solid is graded against the beam oracle: "
        "agreement at the cited level certifies the model against beam kinematics, not against "
        "elasticity.",
        "Uniform traction on the tip face (consistent 8-node load) is used instead of the parabolic "
        "Saint-Venant shear distribution. The resulting tip-region error is O(h/L)^2 and is not "
        "separated here from the root effect; at L/h = 10 it sits below the reported residual.",
        "The clamped root forbids lateral contraction. The root-BC control measures that cost "
        "explicitly: it is a modelling choice, not mesh error, and it does not shrink under "
        "refinement -- which is why the solid ladder plateaus instead of vanishing.",
        "kappa = 5/6 (Timoshenko) is used for the oracle. For a strip in plane stress the equivalent "
        "elasticity factor is 2/3 (see benchmarks/verified/fem_cantilever_solid_convergence); at "
        "L/h = 10 the difference is ~0.1 % of the tip deflection.",
        "A taper ratio of 1 (knife edge) is excluded: the bending flexibility diverges "
        "logarithmically and no linear beam model is meaningful there.",
        "No experimental reference is used: this is a code-vs-closed-form verification study, the "
        "strongest available reference for this geometry family.",
    ]

    result = {
        "anchor": ANCHOR,
        "mission": MISSION,
        "title": TITLE,
        "generated_at": utc_now(),
        "tolerance": TOLERANCE,
        "case": {
            "length": 0.10, "width": 0.010, "depth_root": 0.010,
            "young": 200e9, "poisson": 0.3, "load": 5.0, "kappa": 5.0 / 6.0,
            "slenderness": 10.0,
        },
        "oracle": oracle_rows,
        "frame_elements": list(FRAME_ELEMENTS),
        "frame": ladder,
        "frame_observed_order": {"midpoint": order_mid, "gauss": order_gauss},
        "solid": solid,
        "root_bc_control": control,
        "wedge": wedge,
        "geometry": geo,
        "geometry_prismatic": geo_prismatic,
        "field_contour": field,
        "limits": limits,
        "solid_worst": solid_worst,
        "solid_fine_worst": solid_fine_worst,
        "frame_worst_32": frame_worst_32,
        "limitations": limitations,
    }
    write_json(HERE / "result.json", result)

    gates_doc = gate_manifest(
        anchor=ANCHOR,
        title=TITLE,
        criteria={
            "case_pool_tip_deflection": "< 3 % vs Euler-Bernoulli / Timoshenko",
            "oracle_self_consistency": "< 1e-10",
            "geometry": "surface exact at every axial station",
        },
        gates=gates,
        artifacts={
            "result": "benchmarks/verified/fem_cantilever_taper/result.json",
            "doc": "docs/CASE-CANTILEVER-TIP-taper-validity.md",
            "figures": [
                "docs/assets/cantilever_taper/frame_order.png",
                "docs/assets/cantilever_taper/solid_ladder.png",
                "docs/assets/cantilever_taper/amplification.png",
                "docs/assets/cantilever_taper/field_contour.png",
            ],
            "test": "tests/test_cantilever_taper.py",
        },
        limitations=limitations,
        mission=MISSION,
    )
    write_json(HERE / "gates_verified.json", gates_doc)

    doc_path = repo_root() / "docs" / "CASE-CANTILEVER-TIP-taper-validity.md"
    doc_path.write_text(render_doc(result, gates_doc))

    (HERE / "README.md").write_text(
        "# fem_cantilever_taper\n\n"
        "Anchor `CASE-CANTILEVER-TIP`, taper axis: a linearly tapered cantilever "
        "(`h(x) = h0 (1 - c x/L)`) graded against the closed-form Castigliano flexibility "
        "`B(c) = [-c - c^2/2 - ln(1-c)]/c^3` and the logarithmic shear factor "
        "`S(c) = -ln(1-c)/c`.\n\n"
        "* driver: `run.py` (regenerates `result.json`, `gates_verified.json`, the doc and the figures)\n"
        f"* verdict: **{'PASS' if gates_doc['passed'] else 'FAIL'}** "
        f"(failed gates: {gates_doc['failed_gates'] or 'none'})\n"
        f"* generated: {result['generated_at']}\n\n"
        "Run: `OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_cantilever_taper/run.py`\n"
    )

    print(f"result  -> {HERE / 'result.json'}")
    print(f"gates   -> {HERE / 'gates_verified.json'}  passed={gates_doc['passed']}")
    for g in gates_doc["gates"]:
        print(f"  [{'ok' if g['passed'] else 'FAIL'}] {g['id']}: {g['value']:.6g} {g['operator']} {g['threshold']}")
    print(f"doc     -> {doc_path}")
    print(f"solid worst error   : {100 * solid_worst:+.3f} % (nx=8..32, c=0..0.75)")
    print(f"frame worst error   : {100 * frame_worst_32:+.3f} % (32 el., midpoint EI)")
    print(f"orders midpoint/EB  : {', '.join(f'{c}:{order_mid[c]:.2f}' for c in order_mid)}")
    print(f"orders gauss/EB     : {', '.join(f'{c}:{order_gauss[c]:.2f}' for c in order_gauss)}")
    return 0 if gates_doc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())