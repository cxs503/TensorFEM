#!/usr/bin/env python3
"""P22-N02: Hertz + Cattaneo-Mindlin + unilateral complementarity gates.

Model
-----
Two contact problems are solved with the half-space influence-coefficient
contact solver in :mod:`tensorfem.hertz_mindlin` (exact Flamant/Love elastic
compliance + active-set Lagrange multipliers, so the unilateral constraint is
enforced exactly rather than by penalty):

* rigid sphere on an elastic half-space (axisymmetric, Love disc kernel) --
  Hertz contact radius, peak pressure, and the load--approach law ``P(delta)``;
* rigid cylinder on an elastic half-space (plane strain, Flamant kernel) --
  Hertz half-width / peak pressure, then the **Cattaneo-Mindlin** partial-slip
  problem: after the normal preload the indenter is displaced tangentially and
  the interface splits into a stick zone and two slip strips.

Deliverables: ``result.json`` (numbers), ``gates.json`` (machine verdicts) and
three figures (pressure profiles, load--approach law, partial-slip state).

Usage::

    PYTHONPATH=src python benchmarks/mindlin_contact/run.py

Everything below is deterministic (no random numbers, no fitted parameters
entering a gate).
"""
from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE
import sys

sys.path.insert(0, str(HERE.parents[1] / "src"))

import tensorfem.hertz_mindlin as hm  # noqa: E402

# ---- declared problem parameters ------------------------------------------
YOUNG = 210.0e9
POISSON = 0.3
RADIUS = 0.05                     # indenter radius 50 mm
CYL_LOAD = 3.625e6                # N/m   -> b = 1.0 mm, p0 = 2.31 GPa
SPHERE_LOADS = [1.0e5 * 1.6 ** k for k in range(6)]
FRICTION = 0.3
TANGENTIAL_FRACTIONS = (0.15, 0.35, 0.60, 0.85)
MESH_3D = (60, 150, 300)
MESH_2D = (400, 800, 1600)

CRITERIA = {
    "E1_hertz_contact_radius_rel_error": 0.02,
    "E2_hertz_peak_pressure_rel_error": 0.05,
    "E3_load_approach_r2": 0.99,
    "E4_stick_slip_boundary_rel_error": 0.05,
    "E5_complementarity_residual": 1e-8,
}


def gate(name, value, limit, mode, note):
    ok = (value <= limit) if mode == "max" else (value >= limit)
    return {"gate": name, "value": float(value), "limit": float(limit),
            "mode": mode, "passed": bool(ok), "note": note}


def main() -> int:
    t_start = time.perf_counter()
    e_star = hm.plane_strain_modulus(YOUNG, POISSON)
    results = {
        "schema": "mindlin-contact-verification/1",
        "campaign_node": "P22-N02",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "parameters": {
            "young_modulus_Pa": YOUNG, "poisson": POISSON,
            "indenter_radius_m": RADIUS, "plane_strain_modulus_Pa": e_star,
            "cylinder_load_N_per_m": CYL_LOAD, "sphere_loads_N": SPHERE_LOADS,
            "friction": FRICTION, "tangential_fractions": list(TANGENTIAL_FRACTIONS),
            "mesh_3d_rings": list(MESH_3D), "mesh_2d_segments": list(MESH_2D),
        },
        "module": {"file": "src/tensorfem/hertz_mindlin.py",
                   "operator": "Flamant (2-D plane strain) / Love (axisymmetric) "
                               "half-space compliance + exact active-set complementarity"},
    }

    # ---------------- 3-D Hertz: radius, peak pressure, load-approach --------
    cyl3d = hm.hertz_cylinder_reference(load=CYL_LOAD, radius=RADIUS,
                                        effective_modulus=e_star)
    sphere_rows = []
    for rings in MESH_3D:
        load = 5.0e5
        sol = hm.solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                       load=load, rings=rings)
        ref = hm.hertz_sphere_reference(load=sol.load, radius=RADIUS,
                                        effective_modulus=e_star)
        loaded = np.nonzero(sol.active)[0]
        a_raw = float(sol.grid["edges"][loaded.max() + 1]) if loaded.size else 0.0
        p0_raw = float(np.max(sol.pressure)) if loaded.size else 0.0
        sphere_rows.append({
            "contact_radius_raw_m": a_raw,
            "contact_radius_raw_rel_error": abs(a_raw / ref.contact_radius - 1.0),
            "peak_pressure_raw_Pa": p0_raw,
            "peak_pressure_raw_rel_error": abs(p0_raw / ref.peak_pressure - 1.0),
            "rings": rings,
            "load_N": sol.load,
            "contact_radius_m": sol.half_width,
            "contact_radius_reference_m": ref.contact_radius,
            "contact_radius_rel_error": abs(sol.half_width / ref.contact_radius - 1.0),
            "peak_pressure_Pa": sol.peak_pressure,
            "peak_pressure_reference_Pa": ref.peak_pressure,
            "peak_pressure_rel_error": abs(sol.peak_pressure / ref.peak_pressure - 1.0),
            "complementarity_residual": sol.complementarity_residual,
            "iterations": sol.iterations,
            "converged": sol.converged,
        })
    finest = sphere_rows[-1]

    # load-approach law: solve loads, compare P with the 3-D Hertz P(delta)
    approach_rows = []
    for load in SPHERE_LOADS:
        sol = hm.solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                       load=load, rings=MESH_3D[-1])
        ref = hm.hertz_sphere_reference(approach=sol.indentation, radius=RADIUS,
                                        effective_modulus=e_star)
        approach_rows.append({"load_N": sol.load, "approach_m": sol.indentation,
                              "load_reference_N": ref.load,
                              "rel_error": abs(sol.load / ref.load - 1.0),
                              "contact_radius_m": sol.half_width})
    p_fe = np.array([row["load_N"] for row in approach_rows])
    p_ref = np.array([row["load_reference_N"] for row in approach_rows])
    r2 = float(1.0 - np.sum((p_fe - p_ref) ** 2) / np.sum((p_ref - p_ref.mean()) ** 2))
    slope = float(np.polyfit(np.log([row["approach_m"] for row in approach_rows]),
                             np.log(p_fe), 1)[0])
    results["hertz_3d"] = {"mesh_ladder": sphere_rows,
                           "load_approach": approach_rows,
                           "load_approach_r2": r2,
                           "power_law_exponent": slope,
                           "power_law_exponent_reference": 1.5}

    # ---------------- 2-D Hertz (cylinder) cross-check ----------------------
    cyl_rows = []
    for segments in MESH_2D:
        sol = hm.solve_hertz_2d(load=CYL_LOAD, radius=RADIUS, young=YOUNG,
                                poisson=POISSON, segments=segments)
        cyl_rows.append({
            "segments": segments,
            "load_N_per_m": sol.load,
            "half_width_m": sol.half_width,
            "half_width_reference_m": cyl3d.half_width,
            "half_width_rel_error": abs(sol.half_width / cyl3d.half_width - 1.0),
            "peak_pressure_Pa": sol.peak_pressure,
            "peak_pressure_reference_Pa": cyl3d.peak_pressure,
            "peak_pressure_rel_error": abs(sol.peak_pressure / cyl3d.peak_pressure - 1.0),
            "complementarity_residual": sol.complementarity_residual,
            "converged": sol.converged,
        })
    results["hertz_2d_cylinder"] = {"mesh_ladder": cyl_rows,
                                    "reference": {"half_width_m": cyl3d.half_width,
                                                  "peak_pressure_Pa": cyl3d.peak_pressure}}

    # ---------------- finite-element cross-check (volumetric mesh) ----------
    fe_rows = []
    for nx, ny in ((60, 40), (100, 60)):
        nodes, elems = hm.block_mesh(nx, ny, 30.0 * cyl3d.half_width,
                                     20.0 * cyl3d.half_width)
        K = hm.plane_strain_stiffness(nodes, elems, YOUNG, POISSON)
        surf = hm.surface_nodes(nodes, ny)
        solver = hm.ElasticSolver(K, hm.dirichlet_dofs(nodes, ny))
        def fe_load(depth):
            return hm.solve_indentation_fe(nodes, elems, K, surf, RADIUS, depth,
                                           solver=solver)["load"]

        lo, hi = 0.0, 1.0e-6
        while fe_load(hi) < CYL_LOAD:
            hi *= 2.0
            if hi > 1.0:
                raise RuntimeError("FE indentation bracket failed")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fe_load(mid) < CYL_LOAD:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        bracket = hm.solve_indentation_fe(nodes, elems, K, surf, RADIUS,
                                          0.5 * (lo + hi), solver=solver)
        fe_rows.append({
            "mesh": f"{nx}x{ny}",
            "load_N_per_m": bracket["load"],
            "contact_extent_m": bracket["contact_extent"],
            "contact_extent_rel_error": abs(bracket["contact_extent"] / cyl3d.half_width - 1.0),
            "peak_nodal_pressure_Pa": bracket["peak_pressure"],
            "peak_pressure_rel_error": abs(bracket["peak_pressure"] / cyl3d.peak_pressure - 1.0),
            "complementarity_residual": bracket["complementarity_residual"],
            "sawtooth_fraction": bracket["sawtooth_fraction"],
            "converged": bracket["converged"],
        })
    results["finite_element_cross_check"] = {
        "note": "plane-strain Q4 node-to-node contact, exact Lagrange multipliers; "
                "independent check of the half-space operator on a volumetric mesh",
        "rows": fe_rows,
    }

    # ---------------- Cattaneo-Mindlin partial slip -------------------------
    cm_rows = []
    for segments in MESH_2D:
        for frac in TANGENTIAL_FRACTIONS:
            tangential = frac * FRICTION * CYL_LOAD
            sol = hm.solve_cattaneo_mindlin_2d(
                normal_load=CYL_LOAD, tangential_load=tangential, friction=FRICTION,
                radius=RADIUS, young=YOUNG, poisson=POISSON, segments=segments)
            c_ref = hm.stick_half_width_2d(sol.contact_half_width, sol.tangential_load,
                                          sol.normal_load, FRICTION)
            cm_rows.append({
                "segments": segments,
                "tangential_fraction": frac,
                "normal_load_N_per_m": sol.normal_load,
                "tangential_load_N_per_m": sol.tangential_load,
                "contact_half_width_m": sol.contact_half_width,
                "stick_half_width_m": sol.stick_half_width,
                "stick_half_width_reference_m": c_ref,
                "stick_boundary_rel_error": abs(sol.stick_half_width / c_ref - 1.0),
                "stick_segments": int(sol.stick.sum()),
                "contact_segments": int(sol.active.sum()),
                "tangential_translation_m": sol.tangential_translation,
                "complementarity_residual": sol.complementarity_residual,
                "stick_residual": sol.stick_residual,
                "cone_residual": sol.cone_residual,
                "converged": sol.converged,
            })
    finest_cm = [row for row in cm_rows if row["segments"] == MESH_2D[-1]]
    results["cattaneo_mindlin"] = {
        "law": "c = b sqrt(1 - Q/(mu P))  (two-dimensional Flamant correspondence)",
        "rows": cm_rows,
        "stick_boundary_rel_error_max_finest": max(r["stick_boundary_rel_error"]
                                                   for r in finest_cm),
    }

    # ---------------- unilateral complementarity / negative control ---------
    no_contact_2d = hm.solve_hertz_2d(load=CYL_LOAD, radius=RADIUS, young=YOUNG,
                                      poisson=POISSON, segments=MESH_2D[-1],
                                      indentation=-1.0e-6)
    no_contact_3d = hm.solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                             indentation=-1.0e-6, rings=MESH_3D[-1])
    loaded_2d = hm.solve_hertz_2d(load=CYL_LOAD, radius=RADIUS, young=YOUNG,
                                  poisson=POISSON, segments=MESH_2D[-1])
    results["unilateral"] = {
        "contact_complementarity_residual_2d": loaded_2d.complementarity_residual,
        "contact_complementarity_residual_3d": finest["complementarity_residual"],
        "contact_min_gap_2d": float(np.min(loaded_2d.gap)),
        "contact_min_pressure_2d": float(np.min(loaded_2d.pressure)),
        "open_gap_min_2d": float(np.min(no_contact_2d.gap)),
        "open_contact_force_2d": no_contact_2d.load,
        "open_contact_force_3d": no_contact_3d.load,
        "open_contact_force_exact_zero": bool(no_contact_2d.load == 0.0
                                              and no_contact_3d.load == 0.0),
    }

    # ---------------- gates ------------------------------------------------
    gates = [
        gate("E1_hertz_contact_radius_rel_error", finest["contact_radius_rel_error"],
             CRITERIA["E1_hertz_contact_radius_rel_error"], "max",
             f"3-D Hertz, {finest['rings']} rings (mesh ladder "
             f"{[r['rings'] for r in sphere_rows]})"),
        gate("E2_hertz_peak_pressure_rel_error", finest["peak_pressure_rel_error"],
             CRITERIA["E2_hertz_peak_pressure_rel_error"], "max",
             "peak of the recovered p(r) against p0 = 3F/(2 pi a^2)"),
        gate("E3_load_approach_r2", r2, CRITERIA["E3_load_approach_r2"], "min",
             f"P(delta) over {len(SPHERE_LOADS)} loads; power-law exponent {slope:.6f}"),
        gate("E4_stick_slip_boundary_rel_error",
             max(r["stick_boundary_rel_error"] for r in finest_cm),
             CRITERIA["E4_stick_slip_boundary_rel_error"], "max",
             f"Cattaneo c vs c=b sqrt(1-Q/(mu P)), {MESH_2D[-1]} segments, "
             f"Q/(mu P) in {[float(f) for f in TANGENTIAL_FRACTIONS]}"),
        gate("E5_complementarity_residual",
             max(loaded_2d.complementarity_residual, finest["complementarity_residual"]),
             CRITERIA["E5_complementarity_residual"], "max",
             "max |gap * contact_force|, 2-D and 3-D"),
        gate("E6_no_contact_force_zero",
             0.0 if results["unilateral"]["open_contact_force_exact_zero"] else 1.0,
             0.0, "max",
             "retracted indenter (gap > 0 everywhere): total contact force must be "
             "exactly 0.0 in floating point"),
    ]
    results["gates"] = gates
    results["all_gates_passed"] = all(g["passed"] for g in gates)
    results["runtime_seconds"] = time.perf_counter() - t_start

    (OUT / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "gates.json").write_text(json.dumps({"gates": gates,
                                                "all_passed": results["all_gates_passed"]},
                                               indent=2), encoding="utf-8")

    _figures(results, loaded_2d, no_contact_2d)
    for g in gates:
        print(f"{'PASS' if g['passed'] else 'FAIL'}  {g['gate']:38s} "
              f"value={g['value']:.4e} limit={g['limit']:.4e}")
    print(f"all gates passed: {results['all_gates_passed']}  "
          f"({results['runtime_seconds']:.1f} s)")
    return 0 if results["all_gates_passed"] else 1


def _figures(results, loaded_2d, no_contact_2d):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figures are optional
        print("figures skipped:", exc)
        return
    # --- figure 1: 2-D Hertz pressure and gap ------------------------------
    ref = loaded_2d.grid["target"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = loaded_2d.grid["centres"] / ref.half_width
    axes[0].plot(x, loaded_2d.pressure / ref.peak_pressure, lw=1.4,
                 label="influence-coefficient solution")
    axes[0].plot(np.linspace(-1, 1, 200),
                 ref.pressure(np.linspace(-1, 1, 200)) / ref.peak_pressure, "--",
                 label="Hertz p0 sqrt(1-(x/b)^2)")
    axes[0].set_xlabel("x / b"), axes[0].set_ylabel("p / p0")
    axes[0].set_title("2-D Hertz pressure (P = %.3g N/m)" % ref.load)
    axes[0].legend(fontsize=8), axes[0].grid(alpha=0.3)
    axes[1].plot(x, loaded_2d.gap * 1e6, lw=1.2, label="gap (loaded)")
    axes[1].plot(no_contact_2d.grid["centres"] / ref.half_width,
                 no_contact_2d.gap * 1e6, lw=1.0, ls=":",
                 label="gap (retracted indenter)")
    axes[1].set_xlabel("x / b"), axes[1].set_ylabel("gap [um]")
    axes[1].set_title("unilateral gap: g >= 0, g*p = 0")
    axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_hertz_2d.png", dpi=160)
    plt.close(fig)

    # --- figure 2: 3-D Hertz pressure profile and load-approach law --------
    sol = hm.solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                   load=5.0e5, rings=MESH_3D[-1])
    ref3 = hm.hertz_sphere_reference(load=sol.load, radius=RADIUS,
                                     effective_modulus=hm.plane_strain_modulus(YOUNG, POISSON))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rr = sol.grid["centres"] / ref3.contact_radius
    axes[0].plot(rr, sol.pressure / ref3.peak_pressure, lw=1.4, label="computed p(r)")
    rr_an = np.linspace(0, 1, 200)
    axes[0].plot(rr_an, np.sqrt(1 - rr_an ** 2), "--", label="Hertz p0 sqrt(1-(r/a)^2)")
    axes[0].set_xlabel("r / a"), axes[0].set_ylabel("p / p0")
    axes[0].set_title("3-D Hertz pressure (F = %.3g N)" % sol.load)
    axes[0].legend(fontsize=8), axes[0].grid(alpha=0.3)
    rows = results["hertz_3d"]["load_approach"]
    delta = np.array([r["approach_m"] for r in rows])
    pfe = np.array([r["load_N"] for r in rows])
    pref = np.array([r["load_reference_N"] for r in rows])
    axes[1].loglog(delta, pfe, "o", ms=5, label="computed P(delta)")
    axes[1].loglog(delta, pref, "-", lw=1.0,
                   label="Hertz P = 4/3 E* sqrt(R) delta^1.5")
    axes[1].set_xlabel("rigid approach delta [m]"), axes[1].set_ylabel("load P [N]")
    axes[1].set_title("load-approach law, R^2 = %.10f" % results["hertz_3d"]["load_approach_r2"])
    axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_hertz_3d.png", dpi=160)
    plt.close(fig)

    # --- figure 3: Cattaneo-Mindlin partial slip ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = [r for r in results["cattaneo_mindlin"]["rows"]
            if r["segments"] == MESH_2D[-1]]
    profile = None
    for row in rows:
        frac = row["tangential_fraction"]
        solution = hm.solve_cattaneo_mindlin_2d(
            normal_load=CYL_LOAD, tangential_load=frac * FRICTION * CYL_LOAD,
            friction=FRICTION, radius=RADIUS, young=YOUNG, poisson=POISSON,
            segments=MESH_2D[-1])
        b = solution.contact_half_width
        cx = solution.grid["centres"] / b
        axes[0].plot(cx, solution.traction / (FRICTION * solution.peak_pressure),
                     lw=1.1, label="Q/(mu P) = %.2f  (c/b = %.3f)"
                     % (frac, solution.stick_half_width / b))
        if profile is None:
            profile = solution
    if profile is not None:
        axes[0].plot(np.linspace(-1, 1, 400),
                     np.sqrt(np.clip(1 - np.linspace(-1, 1, 400) ** 2, 0, None)),
                     "k--", lw=0.9, label="local cone mu p / (mu p0)")
    axes[0].set_xlabel("x / b"), axes[0].set_ylabel("q / (mu p0)")
    axes[0].set_title("Cattaneo-Mindlin tangential traction")
    axes[0].legend(fontsize=7), axes[0].grid(alpha=0.3)
    fractions = np.array([r["tangential_fraction"] for r in rows])
    ratios = np.array([r["stick_half_width_m"] / r["contact_half_width_m"] for r in rows])
    axes[1].plot(fractions, ratios, "o", ms=5, label="computed c/b")
    ff = np.linspace(0.0, 0.95, 100)
    axes[1].plot(ff, np.sqrt(1 - ff), "-", lw=1.0, label="Cattaneo sqrt(1-Q/(mu P))")
    axes[1].set_xlabel("Q / (mu P)"), axes[1].set_ylabel("c / b")
    axes[1].set_title("stick-zone boundary (finest grid)")
    axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_cattaneo_mindlin.png", dpi=160)
    plt.close(fig)
    print("figures written: fig1_hertz_2d.png fig2_hertz_3d.png fig3_cattaneo_mindlin.png")


if __name__ == "__main__":
    raise SystemExit(main())