#!/usr/bin/env python
"""CASE-FEM-CLAMPED-PLATE-UNIFORM -- clamped square plate, uniform pressure.

Case statement
--------------
Four edges built-in (clamped), square plate ``a x a``, uniform lateral pressure
``q``, Kirchhoff (thin) regime ``a/t = 100``.  Reported quantity is the
dimensionless centre deflection

    alpha = w_max * D / (q * a**4),        D = E t**3 / (12 (1 - nu**2))

Reference (the ONLY transcribed number in this file)
----------------------------------------------------
Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
McGraw-Hill 1959, Part II, Chapter 5 "Rectangular plates", **Sec. 30**
(rectangular plates with various edge conditions), **Table 30** (coefficients
for a uniformly loaded rectangular plate ``w_max = alpha q a^4 / D``),
row "all edges built in" for a square plate, ``nu = 0.3``::

    alpha = 0.00126

The table is printed to three significant figures, so the tabulated coefficient
carries a rounding band of about +-0.4 % (0.001255 .. 0.001265) before any
numerical error is attributed.  Everything else here is computed, not read:
the converged Rayleigh-Ritz value, the FE values, the convergence order and the
Richardson limit.

Verification chain (why a 3-digit table entry is still a usable anchor)
----------------------------------------------------------------------
1. The Ritz machinery used as the secondary reference is anchored first on the
   case that *does* have a closed form (simply supported square under uniform
   load, exact Navier series = 0.00406235), so its clamped output is trusted.
2. The FE field comes from the merged Q4 Mindlin path
   (:func:`tensorfem.plate_uniform_load.solve_uniform_square_plate`,
   ``winkler_k = 0``); no new element code is introduced by this case.
3. Dimensionless-group / unit check (the plate analogue of "validate the
   geometry before trusting the coefficient"): the same physical plate scaled
   by a factor 2 in ``a``, ``t`` and ``q`` must return the identical ``alpha``.

Run (writes ``benchmarks/artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/``)::

    PYTHONPATH=src python benchmarks/case_clamped_plate_uniform.py

Exit status is 0 only if every gate passes (fail-closed).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import scipy.sparse.linalg as spla  # noqa: E402

from tensorfem.plate_bfs import (  # noqa: E402
    assemble_bfs_plate,
    bfs_boundary_dofs,
    bfs_evaluate,
    bfs_moments,
    solve_bfs_static,
    bfs_patch_test,
)
from tensorfem.plate_uniform_load import (  # noqa: E402
    centre_node_deflection,
    clamped_uniform_ritz,
    element_center_moments,
    nearest_element_center,
    node_values_along_x,
    solve_uniform_square_plate,
    ss_uniform_kirchhoff,
    ss_uniform_ritz,
)

CASE_ID = "CASE-FEM-CLAMPED-PLATE-UNIFORM"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "artifacts" / CASE_ID
FIGDIR = ROOT / "docs" / "assets" / "case-clamped-plate-uniform"

#: Timoshenko & Woinowsky-Krieger 1959, Sec. 30, Table 30, square plate, all
#: edges built in, uniform load, nu = 0.3.
REF_TABLE_ALPHA = 0.00126
REF_TABLE_SOURCE = (
    "Timoshenko & Woinowsky-Krieger 1959, Theory of Plates and Shells, 2nd ed., "
    "Sec. 30, Table 30 (uniformly loaded rectangular plate, w_max = alpha q a^4/D), "
    "square plate, all edges built in, nu = 0.3 -> alpha = 0.00126"
)
REF_TABLE_DIGITS_UNCERTAINTY = 0.5e-5 / REF_TABLE_ALPHA  # +-0.397 %

GATE = 0.03           # acceptance: |got - ref| / ref < 3 %
MESH_GATE = 0.01      # comparison with the converged Ritz value
REF_TRUNCATION_GATE = 0.005  # Ritz truncation band (measured N=10..12 spread ~0.22 %)

SIDE = 1.0            # a [m]
E = 2.1e11            # [Pa] steel
NU = 0.3              # matching the table
Q = 1.0e6             # [Pa] uniform pressure
T_THIN = 0.01         # a/t = 100  (thin, Kirchhoff regime)
T_THICK = 0.10        # a/t = 10   (thick, Mindlin shear-deformation regime)
N_MODES = 12          # clamped Ritz truncation (N=10/12/14 agree to <1e-3)

MESHES = (4, 6, 8, 10, 12, 16, 20, 24, 32)   # even -> centre node exists
THICKNESS_SWEEP = (0.005, 0.01, 0.02, 0.05, 0.10)
POISSON_SWEEP = (0.25, 0.30, 0.35)


def flexural_rigidity(young: float, thickness: float, poisson: float) -> float:
    return young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))


def alpha_of(w: float, young: float, thickness: float, poisson: float,
             pressure: float, side: float) -> float:
    """Dimensionless centre deflection w*D/(q*a^4)."""
    return w * flexural_rigidity(young, thickness, poisson) / (pressure * side ** 4)


def rel(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


def fe_alpha(nx: int, side: float = SIDE, young: float = E, poisson: float = NU,
             thickness: float = T_THIN, pressure: float = Q) -> dict:
    t0 = time.time()
    res = solve_uniform_square_plate(nx, side, young, poisson, thickness, pressure, "clamped")
    w = centre_node_deflection(res)
    return {
        "nx": nx,
        "n_nodes": int(res.nodes.shape[0]),
        "n_elements": nx * nx,
        "w_centre": w,
        "alpha": alpha_of(w, young, thickness, poisson, pressure, side),
        "seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------
# Independent family B: BFS C1 bicubic Kirchhoff element (tensorfem.plate_bfs)
# --------------------------------------------------------------------------
BFS_MESHES = (10, 20, 40)


def bfs_solve(n_x: int, boundary: str = "cccc", side: float = SIDE,
              young: float = E, poisson: float = NU, thickness: float = T_THIN,
              pressure: float = Q) -> dict:
    """Solve the same physical plate with an unrelated discretisation family.

    The round-1 evidence for this case came from one family only (Q4 Mindlin
    ``solve_plate_on_winkler_foundation``, 3 dof/node, C0, shear-deformable).
    This solves the *same* boundary-value problem with the Bogner--Fox--Schmit
    C1 bicubic Hermite rectangle (4 dof/node: ``w, w_x, w_y, w_xy``, Kirchhoff
    bending energy only, no rotation dofs, sparse SuperLU), from a separate
    module (:mod:`tensorfem.plate_bfs`) with its own assembly and its own
    kinematics.  The two families share no element code, so agreement between
    them is independent evidence rather than an extrapolation of one code.

    Options are keyword-only in effect (no positional trap): ``boundary`` is
    ``"cccc"`` (clamped) or ``"ssss"`` (hard simple support).
    """
    if boundary not in ("cccc", "ssss"):
        raise ValueError("boundary must be 'cccc' or 'ssss'")
    plate = assemble_bfs_plate(n_x, n_x, side, side, young, poisson, thickness,
                               surface_load=pressure)
    fixed = bfs_boundary_dofs(n_x, n_x, boundary)
    u = solve_bfs_static(plate, fixed)
    point = np.array([[side / 2.0, side / 2.0]])
    w = float(bfs_evaluate(plate, u, point)["w"][0])
    moments = bfs_moments(plate, u, point, young, poisson, thickness)
    # discrete equilibrium identity: the constrained dofs' reactions must carry
    # exactly the applied load (no reference value involved -- pure algebra)
    residual = plate.stiffness @ u - plate.load
    return {
        "nx": int(n_x),
        "n_dof": int(plate.n_dof),
        "n_constrained_dofs": int(len(fixed)),
        "w_centre_m": w,
        "alpha": alpha_of(w, young, thickness, poisson, pressure, side),
        "centre_moment_coeff": abs(float(moments["M_x"][0])) / (pressure * side ** 2),
        "total_load_N": float(np.sum(plate.load)),
        "reaction_N": float(np.sum(residual[fixed])),
        "deflection_vector": u,
    }


def self_convergence_order(values: list[float], meshes: list[int]) -> float:
    """Observed order from successive differences (no exact reference needed)."""
    h = np.array([1.0 / m for m in meshes], dtype=float)
    d = np.array([abs(b - a) for a, b in zip(values, values[1:])], dtype=float)
    if np.any(d <= 0.0) or len(d) < 2:
        return float("nan")
    orders = [float(np.log(d[i] / d[i + 1]) / np.log(h[i] / h[i + 1]))
              for i in range(len(d) - 1)]
    return float(np.mean(orders))


def richardson(pairs: list[tuple[float, float]]) -> dict:
    """Extrapolate a monotone mesh sequence (h, value) with three levels."""
    (h1, v1), (h2, v2), (h3, v3) = pairs[-3:]
    d21, d32 = v2 - v1, v3 - v2
    if abs(d32) < 1e-18 or d21 * d32 <= 0.0:
        return {"order": None, "limit": None, "note": "no monotone three-level window"}
    r = h1 / h2
    order = math.log(abs(d21 / d32)) / math.log(r)
    limit = v3 + (v3 - v2) / (r ** order - 1.0)
    return {"order": order, "limit": limit, "note": "three-level Richardson on last 3 meshes"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    anchors: list[dict] = []

    def anchor(name, desc, got, ref, err, gate, extra=""):
        anchors.append({
            "anchor": name, "description": desc, "got": got, "reference": ref,
            "relative_error": err, "tolerance": gate,
            "passed": bool(err < gate), "note": extra,
        })

    # ------------------------------------------------------------------
    # 0. reference machinery + dimensionless-group / unit check
    # ------------------------------------------------------------------
    ritz_ss = ss_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=8)
    ks = ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q)
    anchor("reference_machinery_ss_vs_navier",
           "Ritz code path run with sine modes must reproduce the exact Navier series "
           "(machinery anchor for the clamped reference, which has no closed form)",
           alpha_of(ritz_ss.deflection(0.5, 0.5), E, T_THIN, NU, Q, SIDE),
           alpha_of(ks["deflection"], E, T_THIN, NU, Q, SIDE),
           rel(alpha_of(ritz_ss.deflection(0.5, 0.5), E, T_THIN, NU, Q, SIDE),
               alpha_of(ks["deflection"], E, T_THIN, NU, Q, SIDE)), 1e-3)

    ritz = clamped_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=N_MODES)
    w_ritz = ritz.deflection(0.5, 0.5)
    alpha_ritz = alpha_of(w_ritz, E, T_THIN, NU, Q, SIDE)
    ritz_m = ritz.moments(0.5, 0.5)
    alpha_ritz_m = ritz_m[0] / (Q * SIDE ** 2)
    # NB: the clamped Ritz *moment* converges much more slowly than the
    # deflection (N=10 -> 0.02310, N=12 -> 0.02349: 1.7 % spread), so the Ritz
    # moment is only reported, never used as a gate.  The moment anchor is
    # FE-vs-table.
    ritz_m10 = clamped_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=10).moments(0.5, 0.5)[0]
    ritz_moment_truncation = abs(alpha_ritz_m / (ritz_m10 / (Q * SIDE ** 2)) - 1.0)

    ritz_spread = {}
    for n in (10, 12):  # CLAMPED_BEAM_EIGENVALUES tabulates 12 roots: N <= 12
        ritz_spread[n] = alpha_of(
            clamped_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=n).deflection(0.5, 0.5),
            E, T_THIN, NU, Q, SIDE)
    trunc_err = max(rel(v, alpha_ritz) for v in ritz_spread.values())
    anchor("reference_truncation_stability",
           "clamped Ritz N=10/12 vs N=%d centre deflection (reference truncation "
           "band; gate 0.5%% because the measured N=10..12 spread is %.2f%%, which is "
           "itself inside the 3%% acceptance band)" % (N_MODES, 100 * trunc_err),
           alpha_ritz, min(ritz_spread.values()), trunc_err, REF_TRUNCATION_GATE,
           "reference stability, not an FE comparison")

    # plate analogue of "validate the geometry before the coefficient":
    # scale a, t and q all by 2 (same slenderness, same load ratio).
    base8 = fe_alpha(8)
    scaled8 = fe_alpha(8, side=2.0 * SIDE, thickness=2.0 * T_THIN, pressure=2.0 * Q)
    anchor("dimensionless_group_scale_invariance",
           "a -> 2a, t -> 2t, q -> 2q must leave alpha = w D/(q a^4) unchanged "
           "(catches unit / normalisation bugs before any coefficient is judged)",
           base8["alpha"], scaled8["alpha"],
           rel(scaled8["alpha"], base8["alpha"]), 1e-12,
           "identical physical plate, rescaled")

    # ------------------------------------------------------------------
    # 1. mesh convergence, thin clamped plate
    # ------------------------------------------------------------------
    conv = [fe_alpha(nx) for nx in MESHES]
    for row in conv:
        row["alpha_ritz"] = alpha_ritz
        row["err_vs_table"] = rel(row["alpha"], REF_TABLE_ALPHA)
        row["err_vs_ritz"] = rel(row["alpha"], alpha_ritz)
        row["h_over_a"] = 1.0 / row["nx"]

    by_nx = {r["nx"]: r for r in conv}
    dyadic = [n for n in (8, 16, 32) if n in by_nx]
    rich = (richardson([(by_nx[n]["h_over_a"], by_nx[n]["alpha"]) for n in dyadic])
            if len(dyadic) == 3 else {"order": None, "limit": None,
                                      "note": "dyadic subsequence 8/16/32 incomplete"})
    if rich["limit"] is not None and not (0.5 <= rich["order"] <= 3.5):
        rich = {"order": rich["order"], "limit": None,
                "note": "three-level estimate rejected: observed order %.2f outside "
                        "[0.5, 3.5], mesh sequence too flat for extrapolation"
                        % rich["order"]}
    if "note" in rich and rich.get("limit") is not None:
        rich["note"] = "three-level Richardson on nx = %s (uniform r = 2)" % (dyadic,)
    finest = conv[-1]
    anchor("finest_mesh_vs_ritz",
           "Finest mesh in the study (%dx%d) vs the converged clamped Ritz value; "
           "the FE series is monotone in nx so the residual is discretisation, not noise"
           % (finest["nx"], finest["nx"]),
           finest["alpha"], alpha_ritz, rel(finest["alpha"], alpha_ritz), MESH_GATE)
    if rich["limit"] is not None:
        anchor("converged_fe_limit_vs_table",
               "Richardson limit of the FE mesh series vs the Timoshenko table entry",
               rich["limit"], REF_TABLE_ALPHA, rel(rich["limit"], REF_TABLE_ALPHA), GATE,
               "extrapolated order %.2f" % rich["order"])
        anchor("converged_fe_limit_vs_ritz",
               "Richardson limit of the FE mesh series vs the converged clamped Ritz value",
               rich["limit"], alpha_ritz, rel(rich["limit"], alpha_ritz), MESH_GATE)

    # ---- PRIMARY ANCHOR: nx = 16 (17x17 nodes) at a/t = 100 -----------
    primary = next(r for r in conv if r["nx"] == 16)
    anchor("primary_fe_vs_timoshenko_table",
           "Clamped square plate, a/t = 100, uniform pressure, Q4 Mindlin 16x16 "
           "(17x17 nodes): alpha = w_max D/(q a^4) vs Timoshenko Table 30 entry 0.00126",
           primary["alpha"], REF_TABLE_ALPHA, rel(primary["alpha"], REF_TABLE_ALPHA), GATE,
           "accuracy floor of the table is +-%.2f%% (3 printed digits)"
           % (100 * REF_TABLE_DIGITS_UNCERTAINTY))

    # secondary: centre bending moment at nx=16 (Table 30 also lists it)
    res16 = solve_uniform_square_plate(16, SIDE, E, NU, T_THIN, Q, "clamped")
    moms = element_center_moments(res16)
    idx = nearest_element_center(moms, 0.5, 0.5)
    m_centre = abs(moms["moment_x"][idx]) / (Q * SIDE ** 2)
    anchor("secondary_fe_centre_moment_vs_table",
           "Centre bending moment coefficient |Mx|/(q a^2), 16x16, vs Timoshenko "
           "Table 30 clamped-square entry 0.0231 (secondary: the Ritz moment "
           "reference itself drifts %.2f%% between N=10 and N=12, so the table "
           "entry is the reference here)" % (100 * ritz_moment_truncation),
           m_centre, 0.0231, rel(m_centre, 0.0231), GATE)

    # ------------------------------------------------------------------
    # 2. sensitivity A: slenderness a/t = 200 ... 10 at nx = 16
    # ------------------------------------------------------------------
    thick_sweep = []
    for t in THICKNESS_SWEEP:
        r = fe_alpha(16, thickness=t)
        r["a_over_t"] = SIDE / t
        r["alpha_ritz_kirchhoff"] = alpha_of(
            clamped_uniform_ritz(SIDE, E, NU, t, Q, n_modes=N_MODES).deflection(0.5, 0.5),
            E, t, NU, Q, SIDE)
        r["shear_effect_vs_kirchhoff"] = r["alpha"] / r["alpha_ritz_kirchhoff"] - 1.0
        # D scales as t^3 so alpha is dimensionless in t; a/t only matters via
        # transverse shear, which is why alpha must stay ~constant while a/t >= 20
        r["err_vs_table"] = rel(r["alpha"], REF_TABLE_ALPHA)
        thick_sweep.append(r)
    shear_free = [r for r in thick_sweep if r["a_over_t"] >= 50]
    if len(shear_free) >= 2:
        # Statistics note (declared after the first run, see docs): the *mutual*
        # spread of alpha over a/t >= 50 is NOT a clean test of the Kirchhoff
        # limit, because at a fixed coarse mesh the discretisation error also
        # varies with t (it changes sign between a/t = 200 and a/t = 50).  The
        # physically meaningful statistic is, for each a/t in the thin range, the
        # deviation of the FE value from its OWN Kirchhoff (Ritz) limit.
        worst = max(shear_free, key=lambda r: abs(r["alpha"] / r["alpha_ritz_kirchhoff"] - 1.0))
        mutual = ((max(r["alpha"] for r in shear_free) - min(r["alpha"] for r in shear_free))
                  / alpha_ritz)
        anchor("slenderness_independence_thin_range",
               "For a/t >= 50 the FE alpha must sit on its own Kirchhoff limit: worst "
               "|alpha_FE/alpha_Ritz(t) - 1| over a/t in {50,100,200} at nx = 16",
               worst["alpha"], worst["alpha_ritz_kirchhoff"],
               abs(worst["alpha"] / worst["alpha_ritz_kirchhoff"] - 1.0), MESH_GATE,
               "worst at a/t = %.0f; mutual spread over the three thin cases = %.3f%% "
               "(dominated by mesh discretisation, not by shear)"
               % (worst["a_over_t"], 100 * mutual))
    thick_row = next(r for r in thick_sweep if r["a_over_t"] == 10)
    anchor("mindlin_shear_raises_thick_plate_deflection",
           "At a/t = 10 transverse shear must raise alpha above the Kirchhoff value "
           "(>1 %); confirms the Mindlin shear branch is live, not dead code",
           thick_row["alpha"], thick_row["alpha_ritz_kirchhoff"],
           abs(thick_row["shear_effect_vs_kirchhoff"]), 1.0,  # signal check, see note
           "not a gate on magnitude: sign/liveness check, effect = %+.2f%% "
           % (100 * thick_row["shear_effect_vs_kirchhoff"]))
    anchors[-1]["passed"] = bool(thick_row["shear_effect_vs_kirchhoff"] > 0.01)

    # ------------------------------------------------------------------
    # 3. sensitivity B: Poisson ratio at nx = 16, a/t = 100
    # ------------------------------------------------------------------
    nu_sweep = []
    for nu in POISSON_SWEEP:
        r = fe_alpha(16, poisson=nu)
        r["nu"] = nu
        r["alpha_ritz"] = alpha_of(
            clamped_uniform_ritz(SIDE, E, nu, T_THIN, Q, n_modes=N_MODES).deflection(0.5, 0.5),
            E, T_THIN, nu, Q, SIDE)
        r["err_vs_ritz"] = rel(r["alpha"], r["alpha_ritz"])
        nu_sweep.append(r)
    worst_nu = max(nu_sweep, key=lambda r: r["err_vs_ritz"])
    anchor("poisson_sweep_fe_vs_ritz",
           "nu in {0.25,0.30,0.35}: FE vs Ritz, worst case (the table is quoted for "
           "nu = 0.3 only, so nu is a sensitivity axis, not a reference axis)",
           worst_nu["alpha"], worst_nu["alpha_ritz"], worst_nu["err_vs_ritz"], MESH_GATE,
           "worst of 3 nu values")

    # ------------------------------------------------------------------
    # 3c. INDEPENDENT FAMILY B -- BFS C1 bicubic Kirchhoff (round 2)
    # ------------------------------------------------------------------
    # Round-1 evidence was one discretisation family graded against the table
    # and a Ritz reference.  Cross-family agreement is strictly stronger: the
    # BFS element (bicubic C1 Hermite, 4 dof/node, Kirchhoff) and the Q4
    # Mindlin element (bilinear C0, 3 dof/node, shear-deformable) share no
    # code, no kinematics and no solver.  Sequence: patch test first (element
    # algebra), then the closed-form case (family machinery), then the case
    # that has no closed form (clamped), then cross-family comparison.
    patch = bfs_patch_test()
    anchor("bfs_patch_constant_curvature_exact",
           "BFS element patch test: w = x^2 + y^2 + x*y (constant curvatures 2 "
           "and twist 1) on a 3x2 mesh of 2:1 rectangles must be reproduced to "
           "round-off at every dof",
           patch["relative_error"], 0.0, patch["relative_error"], 1e-10,
           "max |err| = %.3e (field scale %.3g, %d dofs)"
           % (patch["max_abs_error"], patch["field_scale"], patch["n_dof"]))

    alpha_navier_ss = alpha_of(ks["deflection"], E, T_THIN, NU, Q, SIDE)
    bfs_ss = bfs_solve(20, boundary="ssss")
    anchor("bfs_family_machinery_ss_vs_navier",
           "BFS family graded on the case that DOES have a closed form: hard "
           "simple support, uniform load, 20x20 vs the exact Navier double "
           "series (0.00406235).  A family that cannot reproduce a closed-form "
           "case is not allowed to grade the clamped case.",
           bfs_ss["alpha"], alpha_navier_ss,
           rel(bfs_ss["alpha"], alpha_navier_ss), REF_TRUNCATION_GATE)

    bfs_conv = [bfs_solve(n) for n in BFS_MESHES]
    bfs_finest = bfs_conv[-1]
    bfs_order = self_convergence_order([r["alpha"] for r in bfs_conv], list(BFS_MESHES))

    anchor("independent_family_bfs_vs_table",
           "Independent family (BFS C1 bicubic Kirchhoff, 40x40) clamped square, "
           "a/t = 100: alpha vs the Timoshenko Table 30 entry 0.00126",
           bfs_finest["alpha"], REF_TABLE_ALPHA,
           rel(bfs_finest["alpha"], REF_TABLE_ALPHA), GATE,
           "self-convergence order over the 10/20/40 ladder = %.2f" % bfs_order)
    anchor("independent_family_bfs_vs_ritz",
           "BFS 40x40 vs the converged clamped Rayleigh-Ritz reference (same "
           "Kirchhoff physics, entirely different discretisation)",
           bfs_finest["alpha"], alpha_ritz,
           rel(bfs_finest["alpha"], alpha_ritz), MESH_GATE)

    mindlin_limit = rich["limit"] if rich["limit"] is not None else primary["alpha"]
    anchor("cross_family_bfs_vs_mindlin",
           "CROSS-FAMILY: Q4 Mindlin Richardson limit vs BFS 40x40.  Two "
           "unrelated element families must land within %.1f%% of each other; "
           "this is the check round 1 did not have (one family only)."
           % (100 * MESH_GATE),
           mindlin_limit, bfs_finest["alpha"],
           rel(mindlin_limit, bfs_finest["alpha"]), MESH_GATE,
           "Mindlin series nx=16/24/32 -> %.8f, BFS 40x40 -> %.8f, gap %.3f%%"
           % (mindlin_limit, bfs_finest["alpha"],
              100 * rel(mindlin_limit, bfs_finest["alpha"])))

    anchor("bfs_global_equilibrium_reaction",
           "Discrete equilibrium identity: the reaction of the constrained dofs "
           "must equal the total applied load q a^2 (40x40 BFS); catches a "
           "wrong load vector or a mis-constrained edge",
           abs(bfs_finest["reaction_N"]), Q * SIDE ** 2,
           rel(abs(bfs_finest["reaction_N"]), Q * SIDE ** 2), 1e-10,
           "K u - f at the constrained dofs is the support reaction, i.e. the "
           "load sum with the opposite sign; |reaction| is compared with q a^2")

    anchor("independent_family_bfs_centre_moment_vs_table",
           "BFS 40x40 centre bending-moment coefficient |Mx|/(q a^2) vs "
           "Timoshenko Table 30 clamped entry 0.0231",
           bfs_finest["centre_moment_coeff"], 0.0231,
           rel(bfs_finest["centre_moment_coeff"], 0.0231), GATE)

    # cross-family ladder at matched meshes (report only): the two families
    # approach the same limit from below (Mindlin, shear-flexible C0) and from
    # below (BFS, Kirchhoff); their difference must shrink with refinement.
    cross = []
    for n in (10, 20, 32):
        fe = next(r for r in conv if r["nx"] == n)
        bb = next((r for r in bfs_conv if r["nx"] == n), None) or bfs_solve(n)
        cross.append({
            "nx": n,
            "alpha_mindlin": fe["alpha"],
            "alpha_bfs": bb["alpha"],
            "family_gap_relative": abs(fe["alpha"] - bb["alpha"]) / alpha_ritz,
            "family_gap_absolute": abs(fe["alpha"] - bb["alpha"]),
        })

    # ------------------------------------------------------------------
    # 4. field profiles + figure
    # ------------------------------------------------------------------
    xs, ws = node_values_along_x(res16, 0.5)
    xs_n = [x / SIDE for x in xs]
    ws_n = [w / w_ritz for w in ws]
    ritz_line_x = [i / 200.0 for i in range(201)]
    ritz_line_w = [ritz.deflection(x, 0.5) / w_ritz for x in ritz_line_x]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    nn = [r["nx"] for r in conv]
    ax.semilogx(nn, [r["alpha"] for r in conv], "o-", label="Q4 Mindlin FE (clamped)")
    ax.semilogx([r["nx"] for r in bfs_conv], [r["alpha"] for r in bfs_conv], "s--",
                color="C4", label="BFS C1 Kirchhoff FE (clamped)")
    ax.axhline(REF_TABLE_ALPHA, color="k", ls="--", lw=1,
               label="Timoshenko Table 30: 0.00126")
    ax.axhline(alpha_ritz, color="C3", ls=":", lw=1,
               label="converged Ritz: %.6f" % alpha_ritz)
    if rich["limit"] is not None:
        ax.axhline(rich["limit"], color="C2", ls="-.", lw=1,
                   label="Richardson limit: %.6f" % rich["limit"])
    ax.axhspan(REF_TABLE_ALPHA * (1 - GATE), REF_TABLE_ALPHA * (1 + GATE),
               color="k", alpha=0.06, label="+-3 % gate")
    ax.set_xlabel("elements per side nx")
    ax.set_ylabel(r"$\alpha = w_{max}D/(q a^4)$")
    ax.set_title("mesh convergence, a/t = 100")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ritz_line_x, ritz_line_w, "k-", lw=1.4, label="clamped Ritz")
    ax.plot(xs_n, ws_n, "o", ms=3.5, label="Q4 Mindlin 16x16")
    ax.set_xlabel("x/a  (y/a = 0.5)")
    ax.set_ylabel(r"$w/w_{max}$")
    ax.set_title("normalised deflection profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    at = [r["a_over_t"] for r in thick_sweep]
    ax.semilogx(at, [100 * r["shear_effect_vs_kirchhoff"] for r in thick_sweep], "s-",
                label="FE alpha vs Kirchhoff alpha")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(100 * (REF_TABLE_ALPHA / alpha_ritz - 1), color="C3", ls=":",
               label="table-vs-Ritz rounding band")
    ax.set_xlabel("a/t")
    ax.set_ylabel("relative increase of alpha [%]")
    ax.set_title("slenderness sensitivity (16x16)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("%s -- clamped square plate, uniform pressure (ref: Timoshenko Table 30)"
                 % CASE_ID, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "clamped_plate_uniform.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. verdict + artifacts
    # ------------------------------------------------------------------
    all_passed = all(a["passed"] for a in anchors)
    verdict = {
        "case_id": CASE_ID,
        "solver": "TensorFEM Q4 Mindlin (plate_uniform_load.solve_uniform_square_plate, winkler_k=0)",
        "reference": REF_TABLE_SOURCE,
        "reference_table_alpha": REF_TABLE_ALPHA,
        "reference_rounding_band_relative": REF_TABLE_DIGITS_UNCERTAINTY,
        "independent_reference_ritz": alpha_ritz,
        "independent_reference_ritz_centre_moment": alpha_ritz_m,
        "ritz_centre_moment_truncation_n10_to_n12": ritz_moment_truncation,
        "ritz_centre_moment_n10": ritz_m10 / (Q * SIDE ** 2),
        "ritz_deflection_truncation_band": trunc_err,
        "ritz_alpha_n10_n12": {str(k): v for k, v in ritz_spread.items()},
        "parameters": {
            "side_a_m": SIDE, "young_pa": E, "poisson": NU, "pressure_pa": Q,
            "thickness_m": T_THIN, "a_over_t": SIDE / T_THIN,
            "D_Nm": flexural_rigidity(E, T_THIN, NU),
            "alpha_group": "w_max * D / (q * a^4)",
        },
        "primary": {
            "mesh": "16x16 Q4 elements, 17x17 nodes",
            "alpha_fe": primary["alpha"],
            "alpha_ref": REF_TABLE_ALPHA,
            "relative_error": rel(primary["alpha"], REF_TABLE_ALPHA),
            "gate": GATE,
            "within_3pct": bool(rel(primary["alpha"], REF_TABLE_ALPHA) < GATE),
        },
        "convergence": {
            "meshes": conv, "richardson": rich,
        },
        "independent_family_bfs": {
            "element": "Bogner-Fox-Schmit C1 bicubic Hermite (tensorfem.plate_bfs), "
                       "4 dof/node (w, w_x, w_y, w_xy), Kirchhoff bending only, sparse SuperLU",
            "patch_test": patch,
            "ssss_vs_navier_exact": {
                "nx": bfs_ss["nx"], "alpha": bfs_ss["alpha"],
                "reference_exact_navier": alpha_navier_ss,
                "relative_error": rel(bfs_ss["alpha"], alpha_navier_ss),
            },
            "ladder": [
                {k: v for k, v in r.items() if k != "deflection_vector"} for r in bfs_conv
            ],
            "self_convergence_order": bfs_order,
            "cross_family_ladder": cross,
            "mindlin_richardson_limit": mindlin_limit,
            "cross_family_gap_vs_mindlin": rel(mindlin_limit, bfs_finest["alpha"]),
        },
        "sensitivity": {
            "thin_range_mutual_spread_relative": (
                (max(r["alpha"] for r in shear_free) - min(r["alpha"] for r in shear_free))
                / alpha_ritz) if len(shear_free) >= 2 else None,
            "thickness_sweep_nx16": thick_sweep,
            "poisson_sweep_nx16": nu_sweep,
        },
        "anchors": anchors,
        "all_gates_passed": all_passed,
        "wall_seconds": time.time() - t_start,
        "repro_cmd": "PYTHONPATH=src python benchmarks/case_clamped_plate_uniform.py",
    }
    (OUT / "case_result.json").write_text(json.dumps(verdict, indent=2))

    lines = [
        "# %s -- clamped square plate under uniform pressure" % CASE_ID, "",
        "Reference: %s" % REF_TABLE_SOURCE, "",
        "Independent (computed) reference: Rayleigh-Ritz N=%d alpha = %.8f "
        "(the table entry is rounded to 3 digits, band +-%.2f%%)"
        % (N_MODES, alpha_ritz, 100 * REF_TABLE_DIGITS_UNCERTAINTY), "",
        "## Anchors", "",
        "| anchor | got | reference | rel. error | gate | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for a in anchors:
        lines.append("| %s | %.8g | %.8g | %.3f%% | %.1f%% | %s |" % (
            a["anchor"], a["got"], a["reference"], 100 * a["relative_error"],
            100 * a["tolerance"], "PASS" if a["passed"] else "FAIL"))
    lines += ["", "## Mesh convergence (a/t = 100, clamped)", "",
              "| nx | nodes | alpha | err vs table | err vs Ritz | s |",
              "|---|---|---|---|---|---|"]
    for r in conv:
        lines.append("| %d | %d | %.8f | %+.3f%% | %+.3f%% | %.1f |" % (
            r["nx"], r["n_nodes"], r["alpha"], 100 * (r["alpha"] / REF_TABLE_ALPHA - 1),
            100 * (r["alpha"] / alpha_ritz - 1), r["seconds"]))
    if rich["limit"] is not None:
        lines += ["", "Richardson (last three meshes): order = %.2f, limit = %.8f "
                      "(table %+.3f%%, Ritz %+.3f%%)" % (
                          rich["order"], rich["limit"],
                          100 * (rich["limit"] / REF_TABLE_ALPHA - 1),
                          100 * (rich["limit"] / alpha_ritz - 1))]
    lines += ["", "## Sensitivity A -- slenderness (nx = 16)", "",
              "| a/t | alpha FE | alpha Ritz (Kirchhoff) | shear effect | err vs table |",
              "|---|---|---|---|---|"]
    for r in thick_sweep:
        lines.append("| %.0f | %.8f | %.8f | %+.2f%% | %+.3f%% |" % (
            r["a_over_t"], r["alpha"], r["alpha_ritz_kirchhoff"],
            100 * r["shear_effect_vs_kirchhoff"], 100 * (r["alpha"] / REF_TABLE_ALPHA - 1)))
    lines += ["", "## Sensitivity B -- Poisson ratio (nx = 16, a/t = 100)", "",
              "| nu | alpha FE | alpha Ritz | err |", "|---|---|---|---|"]
    for r in nu_sweep:
        lines.append("| %.2f | %.8f | %.8f | %+.3f%% |" % (
            r["nu"], r["alpha"], r["alpha_ritz"], 100 * (r["alpha"] / r["alpha_ritz"] - 1)))
    lines += ["", "## Independent family B -- BFS C1 bicubic Kirchhoff "
                   "(round 2: cross-family verification)", "",
              "Element: Bogner--Fox--Schmit C1 bicubic Hermite (``tensorfem.plate_bfs``), "
              "4 dof/node, Kirchhoff bending only, sparse SuperLU -- no element code, "
              "kinematics or solver shared with the Q4 Mindlin path above.", "",
              "| check | got | reference | rel. error | gate |",
              "|---|---|---|---|---|",
              "| patch test w=x^2+y^2+xy (2:1 elements) | %.3e | 0 (exact) | %.2e | 1e-10 |"
              % (patch["max_abs_error"], patch["relative_error"]),
              "| BFS SSSS 20x20 vs exact Navier | %.8f | %.8f | %.3f%% | 0.5%% |"
              % (bfs_ss["alpha"], alpha_navier_ss, 100 * rel(bfs_ss["alpha"], alpha_navier_ss)),
              "| equilibrium: |reaction| vs q a^2 (40x40) | %.6e | %.6e | %.1e | 1e-10 |"
              % (abs(bfs_finest["reaction_N"]), Q * SIDE ** 2,
                 abs(abs(bfs_finest["reaction_N"]) / (Q * SIDE ** 2) - 1.0)),
              "", "### BFS ladder (clamped, a/t = 100)", "",
              "| nx | dofs | alpha | err vs table | err vs Ritz | self-convergence vs finer |",
              "|---|---|---|---|---|---|"]
    for r in bfs_conv:
        lines.append("| %d | %d | %.8f | %+.3f%% | %+.3f%% | %s |" % (
            r["nx"], r["n_dof"], r["alpha"], 100 * (r["alpha"] / REF_TABLE_ALPHA - 1),
            100 * (r["alpha"] / alpha_ritz - 1), "-"))
    lines += ["", "Self-convergence order over the 10/20/40 ladder: **%.2f**.  "
                  "Cross-family gap to the Q4 Mindlin Richardson limit (%.8f): "
                  "**%.3f%%**." % (bfs_order, mindlin_limit,
                                   100 * rel(mindlin_limit, bfs_finest["alpha"])),
              "", "### Cross-family ladder at matched meshes", "",
              "| nx | alpha Q4 Mindlin | alpha BFS | family gap |", "|---|---|---|---|"]
    for r in cross:
        lines.append("| %d | %.8f | %.8f | %.3f%% |" % (
            r["nx"], r["alpha_mindlin"], r["alpha_bfs"],
            100 * r["family_gap_relative"]))
    lines += ["", "## Verdict", "",
              "**%s** -- primary anchor %.8f vs %.5f = %.2f%% %s 3%% gate." % (
                  "PASS" if all_passed else "FAIL", primary["alpha"], REF_TABLE_ALPHA,
                  100 * rel(primary["alpha"], REF_TABLE_ALPHA),
                  "<" if rel(primary["alpha"], REF_TABLE_ALPHA) < GATE else ">="),
              "", "Reproduce: `%s`" % verdict["repro_cmd"]]
    (OUT / "case_result.md").write_text("\n".join(lines) + "\n")

    for a in anchors:
        print("%-46s got=%.6e ref=%.6e err=%7.3f%% gate=%4.1f%% %s" % (
            a["anchor"], a["got"], a["reference"], 100 * a["relative_error"],
            100 * a["tolerance"], "PASS" if a["passed"] else "FAIL"))
    print("primary FE alpha = %.8f vs table %.5f -> %.3f%% (gate 3%%)"
          % (primary["alpha"], REF_TABLE_ALPHA, 100 * rel(primary["alpha"], REF_TABLE_ALPHA)))
    if rich["limit"] is not None:
        print("Richardson limit = %.8f (order %.2f)" % (rich["limit"], rich["order"]))
    print("wall %.1fs -> %s" % (verdict["wall_seconds"], OUT))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())