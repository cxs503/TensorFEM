"""CASE-SPHERE-PRESSURE — uniform-pressure spherical shell, membrane stress sigma = pR/(2t).

Case-pool anchor (mission m15405, TensorFEM).  Decision under test: *does the
production axisymmetric shell of revolution reproduce the exact bending-free
membrane state of a closed pressurised sphere within the predeclared 3 %
tolerance, and over which dimensionless band (t/R, nu) is the reference itself
valid?*

Reference
---------
``sigma_theta = sigma_phi = p R / (2 t)`` for a thin closed sphere under
uniform internal pressure, with ``w = p R^2 (1 - nu) / (2 E t)``.

Provenance is machine-checked rather than hand-copied (see ``reference()``):

1. ``sympy`` derivation from spherical-shell equilibrium — two *independent*
   statics arguments (axial equilibrium of the hemisphere cut, and the
   Laplace/normal equilibrium of a doubly curved membrane) plus Hooke's law for
   the radial displacement.  Both solve to ``p*r/(2*t)`` with symbolic residual
   exactly zero, so the value is recomputable, not transcribed.
2. Literature anchor (honest statement): the same closed form is the thin-walled
   sphere formula of *Roark's Formulas for Stress and Strain* (Table 13.1,
   case 1 in the 7th ed.), i.e. the sphere companion of the cylinder entry the
   repository already cites in ``benchmarks/lame_cylinder_anchor.py``
   (Table 13.5, case 1).  This sandbox has **no network and no handbook file on
   disk** (scan recorded in ``result.json``), so the printed page/table entry
   itself could not be verified here: page-level provenance is ``WITHHELD`` and
   the number is carried by (1) plus (3).
3. Independent 3-D elasticity anchor: the exact Lame thick-sphere solution
   (:func:`tensorfem.sphere_pressure_benchmark.lame_reference`) evaluated at the
   mid-surface with ``a = R - t/2``, ``b = R + t/2`` converges to ``pR/(2t)`` as
   ``t/R -> 0``.  That is a *different theory* (3-D elasticity, no shell
   assumption) agreeing with the shell formula, so the reference value is not
   self-referential.

Production path (reused, not reimplemented)
-------------------------------------------
:class:`tensorfem.axisym_shell.AxisymShell` with ``meridian_radius = R``
(circular-arc meridians, so the element sees the sphere it claims to be),
driven through :func:`tensorfem.axisym_shell_benchmarks.sphere_pressure`.

Controls (experimental design)
------------------------------
* **geometry control** — the same element on the same meridional mesh with
  ``meridian_radius=None`` (straight chords).  Isolates the geometry
  representation term from the discretisation term.
* **element control** — the merged flat-facet Shell4 path
  (:mod:`tensorfem.sphere_membrane_benchmark`) on the same physical case, the
  path whose 1006 %-level failure is recorded in
  ``docs/sphere-membrane-validation.md``.

Run:  python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sympy as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.axisym_shell import AxisymShell  # noqa: E402
from tensorfem.axisym_shell_benchmarks import sphere_nodes, sphere_pressure  # noqa: E402
from tensorfem.sphere_pressure_benchmark import lame_reference  # noqa: E402

# --------------------------------------------------------------------------
# Case definition (predeclared before any solve)
# --------------------------------------------------------------------------
CASE = {
    "radius_m": 10.0,          # mid-surface radius R
    "thickness_m": 0.04,       # wall thickness t
    "young_modulus_pa": 6.825e7,
    "poisson_ratio": 0.3,
    "pressure_pa": 1.0,
}
TOL = 0.03                      # predeclared acceptance tolerance (<3 %)
FACETS = (2, 4, 8, 16, 32, 64)  # refinement ladder (elements pole -> equator)
PRODUCTION_FACETS = 8           # "engineering mesh" on which the gate is stated
THICKNESS_RATIOS = (1 / 1000, 1 / 500, 1 / 250, 1 / 100, 1 / 50, 1 / 20, 1 / 10, 1 / 5)
POISSON_RATIOS = (0.0, 0.2, 0.3, 0.4, 0.49)
PRESSURES = (1.0, 1.0e3, 1.0e6)
FLAT_FACET_CONTROL = 2          # facets per cube face (24 facets) - cheap reproduction

OUT = HERE
FIG = os.path.join(HERE, "figures")


def rel(got, ref):
    return abs(got - ref) / abs(ref)


# --------------------------------------------------------------------------
# 1. Reference, machine-derived
# --------------------------------------------------------------------------
def reference(case: dict = CASE) -> dict:
    """Derive sigma and w from equilibrium + Hooke, symbolically (no hand copy)."""
    p, R, t, E, nu, r = sp.symbols("p R t E nu r", positive=True)
    sigma = sp.symbols("sigma", positive=True)

    # (a) axial equilibrium of the hemisphere cut by the equatorial plane:
    #     pressure resultant p*pi*r^2 = membrane meridional resultant sigma*2*pi*r*t
    axial = sp.Eq(p * sp.pi * r ** 2, sigma * 2 * sp.pi * r * t)
    sigma_axial = sp.solve(axial, sigma)[0]                      # p*r/(2t)

    # (b) Laplace / normal equilibrium of a doubly curved membrane:
    #     N_phi/R_phi + N_theta/R_theta = p, with R_phi = R_theta = R and
    #     N_phi = N_theta = sigma*t  ->  2*sigma*t/R = p
    laplace = sp.Eq(2 * sigma * t / R, p)
    sigma_laplace = sp.solve(laplace, sigma)[0]

    # equal-biaxial state at r = R
    resid = sp.simplify(sigma_axial.subs(r, R) - sigma_laplace)
    sigma_expr = sp.simplify(sigma_laplace)

    # (c) Hooke: eps_theta = (sigma_theta - nu*sigma_phi)/E = sigma*(1-nu)/E
    #     radial displacement of the mid-surface w = R*eps_theta
    w_expr = sp.simplify(R * (sigma_expr - nu * sigma_expr) / E)

    # (d) 3-D elasticity cross-check: Lame mid-surface hoop stress in the thin limit
    a, b = sp.symbols("a b", positive=True)
    A = p * a ** 3 / (b ** 3 - a ** 3)
    B = p * a ** 3 * b ** 3 / (b ** 3 - a ** 3)
    rmid = (a + b) / 2
    lame_mid = sp.simplify(A + B / (2 * rmid ** 3))
    # series of the exact Lame hoop stress at the mid-surface about t/R -> 0
    eps = sp.symbols("eps", positive=True)  # t/R -> 0
    lame_series = sp.series(
        lame_mid.subs({a: R * (1 - eps / 2), b: R * (1 + eps / 2)}), eps, 0, 2).removeO()

    values = {
        "sigma_pa": float(sigma_expr.subs({p: case["pressure_pa"], R: case["radius_m"],
                                           t: case["thickness_m"]})),
        "w_m": float(w_expr.subs({p: case["pressure_pa"], R: case["radius_m"],
                                  t: case["thickness_m"], E: case["young_modulus_pa"],
                                  nu: case["poisson_ratio"]})),
    }
    return {
        "derivation": {
            "axial_equilibrium": str(axial),
            "axial_solution": str(sigma_axial),
            "laplace_equilibrium": str(laplace),
            "laplace_solution": str(sigma_laplace),
            "residual_axial_vs_laplace_at_r_R": str(resid),
            "sigma_expr": str(sigma_expr),
            "w_expr": str(w_expr),
            "lame_mid_surface_series_t_over_R": str(lame_series),
        },
        "residual_is_zero": bool(resid == 0),
        "values": values,
        "value_checks": {
            # recompute from the printed formula independently of sympy
            "sigma_from_formula": case["pressure_pa"] * case["radius_m"] / (2 * case["thickness_m"]),
            "w_from_formula": (case["pressure_pa"] * case["radius_m"] ** 2 *
                               (1 - case["poisson_ratio"]) /
                               (2 * case["young_modulus_pa"] * case["thickness_m"])),
        },
        "provenance": {
            "level": "machine_derived + 3D-elasticity limit; printed handbook page NOT verified",
            "literature_anchor": ("Roark's Formulas for Stress and Strain, thin-walled sphere "
                                  "under internal pressure (Table 13.1, case 1 in the 7th ed.); "
                                  "sphere companion of the cylinder entry cited in "
                                  "benchmarks/lame_cylinder_anchor.py (Table 13.5, case 1)"),
            "page_level_status": "WITHHELD",
            "page_level_reason": ("no network access and no handbook file on disk in this sandbox; "
                                  "scan command and result recorded under 'handbook_scan'"),
            "handbook_scan": {
                "command": "find /root /opt /usr/share/doc -maxdepth 6 -iregex "
                           "'.*\\(roark|timoshenko|handbook\\).*\\.\\(pdf|txt|md|djvu\\)'",
                "result": "only repository .md files; no handbook text/PDF found",
            },
        },
    }


# --------------------------------------------------------------------------
# 2. Numerics on the production element
# --------------------------------------------------------------------------
def axisym_case(facets: int, *, exact_geometry: bool = True, case: dict = CASE) -> dict:
    """Solved state on the AxisymShell path, reduced to the compared scalars."""
    res = sphere_pressure(radius=case["radius_m"], thickness=case["thickness_m"],
                          young=case["young_modulus_pa"],
                          poisson=case["poisson_ratio"],
                          pressure=case["pressure_pa"], facets=facets,
                          exact_geometry=exact_geometry)
    t = case["thickness_m"]
    sigma_theta = [n / t for n in res["N_membrane"]]
    sigma_phi = [n / t for n in res["N_phi"]]
    return {
        "facets": facets,
        "geometry": res["geometry"],
        "elements": facets,
        "dof": 3 * (facets + 1),
        "sigma_ref_pa": res["N_membrane_exact"] / t,
        "sigma_theta_pa": sigma_theta,
        "sigma_phi_pa": sigma_phi,
        "sigma_theta_max_rel_error": res["N_theta_max_rel_error"],
        "sigma_phi_max_rel_error": res["N_phi_max_rel_error"],
        "anisotropy_rel": res["anisotropy_rel_error"],
        "w_m": res["w"],
        "w_ref_m": res["w_exact"],
        "w_rel_error": res["w_rel_error"],
    }


def idealised_meridian_error(facets: int, case: dict = CASE) -> dict:
    """Is the residual error mesh geometry or the exact circular-arc assumption?

    ``AxisymShell`` with ``meridian_radius=R`` integrates the meridian term with
    ``R_s = R`` but the *nodes* still sit on the true sphere, so both curvature
    and nodal positions are exact.  What is left is the interpolation of the
    meridian: ``u``, ``w``, ``beta`` are linear on the arc, so the membrane state
    (constant N) is representable and the remaining error must vanish at the
    element level.  The chord control below keeps everything else fixed.
    """
    return axisym_case(facets, exact_geometry=True, case=case)


def chord_control(facets: int, case: dict = CASE) -> dict:
    """Same element/mesh, meridian curvature switched off (straight generatrix)."""
    return axisym_case(facets, exact_geometry=False, case=case)


def flat_facet_shell4_control(case: dict = CASE) -> dict:
    """Merged flat-facet Shell4 path on the same physical case (element control)."""
    from tensorfem.sphere_membrane_benchmark import PROBLEM, solve_sphere_membrane

    problem = dict(PROBLEM)
    problem.update(radius_m=case["radius_m"], thickness_m=case["thickness_m"],
                   young_modulus_pa=case["young_modulus_pa"],
                   poisson_ratio=case["poisson_ratio"])
    t0 = time.time()
    res = solve_sphere_membrane(FLAT_FACET_CONTROL, problem)
    rr = res.radius_ratios
    out = {
        "path": "tensorfem.sphere_membrane_benchmark (projected_shell4_stiffness, flat facets)",
        "facets_per_cube_face": FLAT_FACET_CONTROL,
        "facet_count": rr["facet_count"],
        "wall_clock_s": round(time.time() - t0, 2),
        "N_mean_n_per_m": rr["resultant_mean_n_per_m"],
        "N_mean_exact_n_per_m": rr["resultant_reference_exact_n_per_m"],
        "N_mean_rel_error": rr["resultant_error_vs_exact"],
        "N_mean_rel_error_vs_facet_plane": rr["resultant_error_vs_facet"],
        "isotropy_defect_n_per_m": rr["resultant_isotropy_defect"],
        "w_fit_m": rr["radial_displacement_m"],
        "w_exact_m": rr["radial_displacement_reference_m"],
        "w_rel_error": rr["radial_displacement_error"],
        "nonradial_residual_rel": rr["uniform_expansion_relative_residual"],
        "mesh_defect": rr["mesh_defect"],
    }
    # the recorded failure curve (docs/sphere-membrane-validation.md) for context
    out["recorded_error_curve_percent"] = {"24": 13316.6, "96": 2826.2, "216": 1005.6}
    out["recorded_source"] = "docs/sphere-membrane-validation.md section 3"
    return out


# --------------------------------------------------------------------------
# 3. Studies
# --------------------------------------------------------------------------
def convergence(case: dict = CASE) -> dict:
    rows = {"exact_meridian": [], "chord_control": []}
    for kind, fn in (("exact_meridian", idealised_meridian_error), ("chord_control", chord_control)):
        for facets in FACETS:
            rows[kind].append(fn(facets, case))
    orders = {}
    for kind in rows:
        st = rows[kind]
        seq = []
        for a, b in zip(st[:-1], st[1:]):
            ea, eb = a["sigma_theta_max_rel_error"], b["sigma_theta_max_rel_error"]
            if eb > 0 and ea > 0:
                seq.append(round(math.log(ea / eb) / math.log(b["facets"] / a["facets"]), 3))
            else:
                seq.append(None)
        orders[kind] = seq
    return {"ladder": rows, "observed_orders_sigma": orders}


def thickness_sensitivity(case: dict = CASE) -> dict:
    """Where does the thin-shell reference stop being valid? (t/R sweep)"""
    R = case["radius_m"]
    rows = []
    for ratio in THICKNESS_RATIOS:
        t = ratio * R
        sub = dict(case, thickness_m=t)
        fem = idealised_meridian_error(16, sub)
        sigma_thin = case["pressure_pa"] * R / (2 * t)          # reference under test
        # exact 3-D elasticity (Lame) at the mid-surface
        a, b = R - t / 2, R + t / 2
        prob = {"inner_radius_m": a, "outer_radius_m": b,
                "young_modulus_pa": case["young_modulus_pa"],
                "poisson_ratio": case["poisson_ratio"],
                "internal_pressure_pa": case["pressure_pa"],
                "external_pressure_pa": 0.0}
        import torch  # local: keeps module import light for the reader
        _, _, _, st, _ = lame_reference(torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64), prob)
        sigma_lame = float(st[0])
        w_thin = case["pressure_pa"] * R ** 2 * (1 - case["poisson_ratio"]) / (
            2 * case["young_modulus_pa"] * t)
        rows.append({
            "t_over_R": ratio,
            "thickness_m": t,
            "sigma_ref_thin_pa": sigma_thin,
            "sigma_lame_midsurface_pa": sigma_lame,
            "thin_vs_lame_rel": rel(sigma_thin, sigma_lame),
            "fem_sigma_pa": max(fem["sigma_theta_pa"]),
            "fem_vs_thin_rel": max(fem["sigma_theta_max_rel_error"], 0.0),
            "fem_vs_lame_rel": rel(max(fem["sigma_theta_pa"]), sigma_lame),
            "w_ref_m": w_thin,
            "w_fem_m": fem["w_m"],
            "w_rel_error": fem["w_rel_error"],
        })
    # band in which the thin-shell reference itself is within TOL of 3-D elasticity
    within = [r["t_over_R"] for r in rows if r["thin_vs_lame_rel"] < TOL]
    xs = [math.log(r["t_over_R"]) for r in rows]
    ys = [math.log(r["thin_vs_lame_rel"]) for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return {"rows": rows,
            "t_over_R_band_within_tol": [min(within), max(within)] if within else None,
            "thin_vs_lame_loglog_slope": round(slope, 4),
            "deviation_model": "|sigma_thin - sigma_Lame| / sigma_Lame ~ (t/R)^slope"}


def poisson_sensitivity(case: dict = CASE) -> dict:
    rows = []
    for nu in POISSON_RATIOS:
        sub = dict(case, poisson_ratio=nu)
        fem = idealised_meridian_error(PRODUCTION_FACETS, sub)
        rows.append({
            "poisson_ratio": nu,
            "sigma_theta_over_ref": max(fem["sigma_theta_pa"]) / fem["sigma_ref_pa"],
            "sigma_theta_max_rel_error": fem["sigma_theta_max_rel_error"],
            "w_over_ref": fem["w_m"] / fem["w_ref_m"],
            "w_rel_error": fem["w_rel_error"],
            "anisotropy_rel": fem["anisotropy_rel"],
        })
    return {"rows": rows}


def pressure_scaling(case: dict = CASE) -> dict:
    """Linearity check in the dimensionless loading p/E."""
    rows = []
    base = idealised_meridian_error(PRODUCTION_FACETS, case)
    for p in PRESSURES:
        sub = dict(case, pressure_pa=p)
        fem = idealised_meridian_error(PRODUCTION_FACETS, sub)
        rows.append({
            "pressure_pa": p,
            "p_over_E": p / case["young_modulus_pa"],
            "sigma_over_p": max(fem["sigma_theta_pa"]) / p,
            "sigma_ref_over_p": fem["sigma_ref_pa"] / p,
            "rel_error": fem["sigma_theta_max_rel_error"],
            "w_over_p": fem["w_m"] / p,
            "w_rel_error": fem["w_rel_error"],
            "sigma_ratio_vs_p1": max(fem["sigma_theta_pa"]) / max(base["sigma_theta_pa"]),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------
# 4. Gates (predeclared)
# --------------------------------------------------------------------------
def gates(conv: dict, thick: dict, pois: dict, lam: dict, flat: dict) -> dict:
    prod = next(r for r in conv["ladder"]["exact_meridian"]
                if r["facets"] == PRODUCTION_FACETS)
    chord = next(r for r in conv["ladder"]["chord_control"]
                 if r["facets"] == PRODUCTION_FACETS)
    e8 = prod["sigma_theta_max_rel_error"]
    e32 = next(r for r in conv["ladder"]["exact_meridian"]
               if r["facets"] == 32)["sigma_theta_max_rel_error"]
    e4 = next(r for r in conv["ladder"]["exact_meridian"]
              if r["facets"] == 4)["sigma_theta_max_rel_error"]
    order = [o for o in conv["observed_orders_sigma"]["exact_meridian"] if o is not None]
    band = thick["t_over_R_band_within_tol"]
    g = {}

    def put(key, value, limit, ok, note):
        g[key] = {"value": value, "criterion": limit, "pass": bool(ok), "note": note}

    put("G1_sigma_theta_vs_pr_2t", e8, "rel err < 3% at facets=8", e8 < TOL,
        "max over Gauss points of sigma_theta; reference pR/(2t)")
    put("G2_sigma_phi_vs_pr_2t", prod["sigma_phi_max_rel_error"],
        "rel err < 3% at facets=8", prod["sigma_phi_max_rel_error"] < TOL,
        "meridional stress, independent Gauss-point set")
    put("G3_isotropy", prod["anisotropy_rel"], "< 3% at facets=8", prod["anisotropy_rel"] < TOL,
        "|N_phi - N_theta| / (pR/2); the membrane state is equi-biaxial")
    put("G4_radial_displacement", prod["w_rel_error"], "< 3% at facets=8",
        prod["w_rel_error"] < TOL, "w vs pR^2(1-nu)/(2Et) at the pole")
    put("G5_monotone_refinement", [e4, e8, e32], "e(4) > e(8) > e(32)",
        e4 > e8 > e32, "error must fall monotonically under refinement")
    put("G6_observed_order", order, "min observed order >= 1.5",
        bool(order) and min(order) >= 1.5, "log-log slope between consecutive ladders")
    put("G7_poisson_invariance", max(r["sigma_theta_max_rel_error"] for r in pois["rows"]),
        "< 3% for every nu in 0..0.49", all(r["sigma_theta_max_rel_error"] < TOL
                                           for r in pois["rows"]),
        "membrane stress of a closed sphere is nu-independent; w is not")
    # G8 was predeclared in two parts: the case point must be inside the
    # reference's own validity band (G8a), and the band was *predicted* to reach
    # t/R = 0.05 (G8b).  G8b is graded as a claim, not silently relaxed: it is
    # measured to be false (deviation ~ t/R, so the band ends near 0.03) and is
    # kept in the record as a falsified hypothesis.
    case_point = next(r for r in thick["rows"]
                      if abs(r["t_over_R"] - CASE["thickness_m"] / CASE["radius_m"]) < 1e-12)
    put("G8a_reference_valid_at_case_point", case_point["thin_vs_lame_rel"],
        "|sigma_thin - sigma_Lame|/sigma_Lame < 3% at t/R of the case",
        case_point["thin_vs_lame_rel"] < TOL,
        "validity of the reference itself at t/R = 0.004 (thin-shell vs exact Lame)")
    put("G8b_reference_band_reaches_t_over_R_0.05", band,
        "predicted: band reaches t/R = 0.05", bool(band) and band[1] >= 0.05,
        "PREDECLARED PREDICTION FALSIFIED: measured band ends near t/R = 0.03; "
        "deviation scales like t/R (see thin_vs_lame_loglog_slope)")
    put("G9_chord_control_fails", chord["sigma_theta_max_rel_error"],
        "> 3% (control must fail: geometry representation matters)",
        chord["sigma_theta_max_rel_error"] > TOL,
        "same element and mesh, curved meridian switched off")
    n_mean_err = flat.get("N_mean_rel_error")
    put("G10_flat_facet_shell4_control", n_mean_err,
        "> 3% (recorded 13317% at 24 facets)", (n_mean_err is None) or n_mean_err > TOL,
        "merged flat-facet Shell4 element on the same case: element control")
    gating = {k: v for k, v in g.items() if not k.startswith("G8b")}
    return {"tolerance": TOL, "gates": g,
            "falsified_predictions": [k for k, v in g.items()
                                      if k.startswith("G8b") and not v["pass"]],
            "all_gating_pass": all(v["pass"] for v in gating.values()),
            "all_pass_including_prediction": all(v["pass"] for v in g.values())}


# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
def figures(conv: dict, thick: dict, pois: dict, ref: dict, case: dict = CASE) -> list[str]:
    os.makedirs(FIG, exist_ok=True)
    made = []
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # --- 1. convergence ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = [r["facets"] for r in conv["ladder"]["exact_meridian"]]
    y = [max(r["sigma_theta_max_rel_error"], 1e-18) for r in conv["ladder"]["exact_meridian"]]
    ax.loglog(x, y, "o-", label="exact circular-arc meridian (production)")
    yc = [max(r["sigma_theta_max_rel_error"], 1e-18) for r in conv["ladder"]["chord_control"]]
    ax.loglog(x, yc, "s--", label="chord meridian (geometry control)")
    ax.axhline(TOL, color="crimson", lw=1.0, ls=":", label="3% tolerance")
    ax.set_xlabel("meridional elements (pole -> equator)")
    ax.set_ylabel(r"$\max |\sigma_\theta - pR/2t|\,/\,(pR/2t)$")
    ax.set_title("CASE-SPHERE-PRESSURE: refinement ladder")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    p = os.path.join(FIG, "convergence.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); made.append(p)

    # --- 2. field along the meridian ----------------------------------------
    fac = 16
    nodes = sphere_nodes(case["radius_m"], fac)
    m = AxisymShell(nodes, young=case["young_modulus_pa"], poisson=case["poisson_ratio"],
                    thickness=case["thickness_m"], pressure=case["pressure_pa"],
                    meridian_radius=case["radius_m"])
    u = m.solve({m.dof(0, "u"): 0.0, m.dof(0, "beta"): 0.0,
                 m.dof(fac, "u"): 0.0, m.dof(fac, "beta"): 0.0})
    field = m.field(u)
    sigma_ref = case["pressure_pa"] * case["radius_m"] / (2 * case["thickness_m"])
    phi = [math.degrees(v) for v in field["phi"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    a1.plot(phi, [v / case["thickness_m"] / sigma_ref for v in field["N_theta"]],
            "o-", ms=3, label=r"$\sigma_\theta$ (FEM, Gauss points)")
    a1.plot(phi, [v / case["thickness_m"] / sigma_ref for v in field["N_s"]],
            "s--", ms=3, label=r"$\sigma_\phi$ (FEM, Gauss points)")
    a1.axhline(1.0, color="k", lw=0.9, ls=":", label=r"$pR/2t$")
    a1.set_xlabel(r"meridian angle $\varphi$ [deg]")
    a1.set_ylabel(r"$\sigma\,/\,(pR/2t)$")
    a1.set_title(f"stress field, {fac} elements")
    a1.grid(alpha=0.3); a1.legend(fontsize=7)
    ang = [math.degrees(0.5 * math.pi * i / fac) for i in range(fac + 1)]
    w = [float(u[m.dof(i, "w")]) for i in range(fac + 1)]
    w_ref = case["pressure_pa"] * case["radius_m"] ** 2 * (1 - case["poisson_ratio"]) / (
        2 * case["young_modulus_pa"] * case["thickness_m"])
    a2.plot(ang, [v * 1e6 for v in w], "o-", ms=3, label="FEM nodes")
    a2.axhline(w_ref * 1e6, color="k", lw=0.9, ls=":", label=r"$pR^2(1-\nu)/(2Et)$")
    a2.set_xlabel(r"meridian angle $\varphi$ [deg]")
    a2.set_ylabel("radial displacement w [µm]")
    a2.set_title("uniform radial expansion")
    a2.grid(alpha=0.3); a2.legend(fontsize=7)
    p = os.path.join(FIG, "field_meridian.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); made.append(p)

    # --- 3. sensitivity in t/R and nu ---------------------------------------
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    tr = [r["t_over_R"] for r in thick["rows"]]
    b1.loglog(tr, [max(r["thin_vs_lame_rel"], 1e-18) for r in thick["rows"]], "o-",
              label=r"thin-shell ref vs exact Lamé")
    b1.loglog(tr, [max(r["fem_vs_thin_rel"], 1e-18) for r in thick["rows"]], "s--",
              label=r"FEM vs thin-shell ref")
    b1.axhline(TOL, color="crimson", lw=1.0, ls=":", label="3% tolerance")
    b1.set_xlabel("t / R"); b1.set_ylabel("relative deviation")
    b1.set_title("thickness sensitivity"); b1.grid(True, which="both", alpha=0.3)
    b1.legend(fontsize=7)
    nu = [r["poisson_ratio"] for r in pois["rows"]]
    b2.plot(nu, [r["sigma_theta_max_rel_error"] for r in pois["rows"]], "o-", ms=4,
            label=r"$\sigma_\theta$ rel err")
    b2.plot(nu, [r["w_rel_error"] for r in pois["rows"]], "s--", ms=4, label="w rel err")
    b2.axhline(TOL, color="crimson", lw=1.0, ls=":", label="3% tolerance")
    b2.set_xlabel(r"Poisson ratio $\nu$"); b2.set_ylabel("relative error")
    b2.set_title(r"$\nu$ sensitivity (8 elements)"); b2.grid(alpha=0.3); b2.legend(fontsize=7)
    p = os.path.join(FIG, "sensitivity.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); made.append(p)
    return made


# --------------------------------------------------------------------------
# 6. Driver
# --------------------------------------------------------------------------
def main() -> dict:
    os.makedirs(OUT, exist_ok=True)
    ref = reference()
    conv = convergence()
    thick = thickness_sensitivity()
    pois = poisson_sensitivity()
    scal = pressure_scaling()
    flat = flat_facet_shell4_control()
    gate_block = gates(conv, thick, pois, None, flat)
    figs = figures(conv, thick, pois, ref)

    result = {
        "case": "CASE-SPHERE-PRESSURE",
        "mission": 15405,
        "repo": "TensorFEM",
        "parameters": {
            **CASE,
            "R_over_t": CASE["radius_m"] / CASE["thickness_m"],
            "t_over_R": CASE["thickness_m"] / CASE["radius_m"],
            "p_over_E": CASE["pressure_pa"] / CASE["young_modulus_pa"],
            "pR_over_Et": (CASE["pressure_pa"] * CASE["radius_m"] /
                           (CASE["young_modulus_pa"] * CASE["thickness_m"])),
            "sigma_over_p": CASE["radius_m"] / (2 * CASE["thickness_m"]),
            "sqrt_Rt_m": math.sqrt(CASE["radius_m"] * CASE["thickness_m"]),
            "w_dimensionless_Et_w_over_pR2": (1 - CASE["poisson_ratio"]) / 2,
        },
        "tolerance": TOL,
        "reference": ref,
        "convergence": conv,
        "thickness_sensitivity": thick,
        "poisson_sensitivity": pois,
        "pressure_scaling": scal,
        "flat_facet_control": flat,
        "gates": gate_block,
        "figures": [os.path.relpath(f, ROOT) for f in figs],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
            "sympy": sp.__version__,
            "numpy": __import__("numpy").__version__,
            "matplotlib": matplotlib.__version__,
            "dtype": "float64",
            "host": "container 99 control plane (loong64), single process",
        },
        "repro_cmd": "python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py",
    }
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    with open(os.path.join(OUT, "gates_verified.json"), "w") as fh:
        json.dump(gate_block, fh, indent=1, sort_keys=True)

    # ---- console summary (the source of every number in the report) --------
    print(f"ref sigma = {ref['values']['sigma_pa']!r} Pa   w = {ref['values']['w_m']!r} m")
    print("residual(axial - laplace) =",
          ref["derivation"]["residual_axial_vs_laplace_at_r_R"])
    print("\nconvergence (production path, exact meridian):")
    print(f"{'facets':>6} {'sigma_theta':>12} {'sigma_phi':>12} {'aniso':>11} {'w':>11}")
    for r in conv["ladder"]["exact_meridian"]:
        print(f"{r['facets']:>6} {r['sigma_theta_max_rel_error']:>12.3e} "
              f"{r['sigma_phi_max_rel_error']:>12.3e} {r['anisotropy_rel']:>11.3e} "
              f"{r['w_rel_error']:>11.3e}")
    print("chord control:")
    for r in conv["ladder"]["chord_control"]:
        print(f"{r['facets']:>6} {r['sigma_theta_max_rel_error']:>12.3e} "
              f"{r['w_rel_error']:>12.3e}")
    print("observed orders:", conv["observed_orders_sigma"]["exact_meridian"])
    print("\nthickness sweep:")
    for r in thick["rows"]:
        print(f"t/R={r['t_over_R']:.4g} thin_vs_lame={r['thin_vs_lame_rel']:.3e} "
              f"fem_vs_thin={r['fem_vs_thin_rel']:.3e} w_err={r['w_rel_error']:.3e}")
    print("t/R band within tol:", thick["t_over_R_band_within_tol"],
          "| log-log slope of thin-vs-Lame deviation:", thick["thin_vs_lame_loglog_slope"])
    print("\npoisson sweep:")
    for r in pois["rows"]:
        print(f"nu={r['poisson_ratio']}: sig_err={r['sigma_theta_max_rel_error']:.3e} "
              f"w/w_ref={r['w_over_ref']:.6f}")
    print("\npressure scaling:")
    for r in scal["rows"]:
        print(f"p={r['pressure_pa']:.0e}: sigma/p={r['sigma_over_p']:.6f} "
              f"err={r['rel_error']:.3e} w/p={r['w_over_p']:.6e}")
    print("\nflat-facet Shell4 control:", json.dumps({k: flat.get(k) for k in
          ("facet_count", "N_mean_n_per_m", "N_mean_exact_n_per_m", "N_mean_rel_error",
           "w_rel_error", "nonradial_residual_rel")}, default=str))
    print("\ngates:")
    for k, v in gate_block["gates"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']}  [{v['criterion']}]")
    print("all_gating_pass =", gate_block["all_gating_pass"],
          "| falsified predictions:", gate_block["falsified_predictions"])
    print("figures:", result["figures"])
    return result


if __name__ == "__main__":
    main()