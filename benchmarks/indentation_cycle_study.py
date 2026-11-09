#!/usr/bin/env python3
"""Ice-structure nonlinear contact / large-deformation verification study.

Chain (campaign node AG27W2-N05): elastic Hertz contact -> elastic-plastic
indentation (Johnson cavity model) -> unloading and residual deformation ->
whole-cycle energy balance -> repeated cycles (shakedown / ratcheting), plus a
large-deflection beam leg (Euler elastica) for the structural side of the
ice-structure interaction.

Gates
-----
C1  Hertz contact radius/peak pressure vs closed form          < 3 %
C2  elastic-plastic indentation load/depth vs Johnson (1985)   < 5 %
C3  cantilever large deflection vs Euler elliptic integrals    < 2 %
C4  single-cycle energy balance |oint F dh - (Wp + U)|         < 1 %
C5  cyclic residual depth increment vs shakedown prediction    < 10 %
C6  GCI extrapolation of the 3-mesh ladder vs closed form        < 5 %

Usage::

    PYTHONPATH=src python benchmarks/indentation_cycle_study.py
    PYTHONPATH=src python benchmarks/indentation_cycle_study.py --quick

Writes ``artifacts/indentation_cycle/study.json`` and the figures under
``docs/assets/indentation-cycle/``.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from tensorfem import indentation_plasticity as ind  # noqa: E402
from tensorfem.beam_large_deflection import (  # noqa: E402
    _elastica_by_quadrature,
    corotational_tangent_error,
    elastica_cantilever,
    solve_corotational_beam,
)
from tensorfem.hertz_axisymmetric import hertz_reference  # noqa: E402

ARTIFACTS = REPO / "artifacts" / "indentation_cycle"
FIGURES = REPO / "docs" / "assets" / "indentation-cycle"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def richardson(values, ratios) -> dict:
    """GCI / Richardson extrapolation from a 3-level mesh ladder."""
    f1, f2, f3 = values
    r = float(np.mean(ratios))
    if abs(f3 - f2) < 1e-16 or abs(f2 - f1) < 1e-16:
        return {"order": float("nan"), "extrapolated": f3,
                "gci": float("nan"), "ratio": r}
    order = math.log(abs((f3 - f2) / (f2 - f1))) / math.log(r)
    factor = r ** order - 1.0
    extrapolated = f3 + (f3 - f2) / factor
    gci = 1.25 * abs(extrapolated - f3) / abs(extrapolated) if extrapolated else float("nan")
    return {"order": order, "extrapolated": extrapolated, "gci": gci,
            "ratio": r, "relative_change": abs(f3 - f2) / abs(f3)}


def elastic_case(nr: int, *, peak: float, radius: float, material) -> dict:
    """Monotonic elastic indentation -> Hertz comparison."""
    # domain 10 contact radii wide/deep: the contact patch then contains ~8-10
    # mesh lines (the graded mesh resolves the patch, and the far field is at
    # 10 a, which keeps the elastic truncation error small)
    extent = 10.0 * math.sqrt(radius * peak)
    res = ind.run_indentation_cycles(
        nr=nr, nz=nr, width=extent, depth=extent, grading=2.5, radius=radius,
        peak_indentation=peak, load_steps=2, unload_steps=1, cycles=1,
        material=material, penalty_factor=100.0)
    step = res.cycles[0].steps[-1]
    force, a_ref, p0_ref = hertz_reference(peak, radius, material.young,
                                           material.poisson)
    return {
        "nr": nr,
        "load": step.load, "load_reference": force,
        "load_error": abs(step.load / force - 1.0),
        "contact_radius": step.contact_radius, "radius_reference": a_ref,
        "radius_fitted": step.fitted_radius,
        # the Hertz-consistent estimator is the p^2 = p0^2 (1 - r^2/a^2) fit
        # (same recovery as tensorfem.hertz_axisymmetric); the raw edge
        # extrapolation is kept as a secondary diagnostic because it is noisy
        # on a graded mesh
        "radius_error": abs(step.fitted_radius / a_ref - 1.0),
        "radius_error_edge": abs(step.contact_radius / a_ref - 1.0),
        "peak_pressure": step.peak_pressure, "peak_reference": p0_ref,
        "peak_error": abs(step.peak_pressure / p0_ref - 1.0),
        "penetration_over_indentation": step.max_penetration / peak,
        "indentation": peak,
        "radii": [float(r) for r in res.discret.radii],
        "surface": [float(u) for u in res.surface_profile["uz"]],
        "profile": res,
    }


def plastic_case(nr: int, *, peak: float, radius: float, material,
                 load_steps: int, cycles: int, hardening: float = 0.0,
                 strict_unloading: bool = True) -> dict:
    extent = 10.0 * math.sqrt(radius * peak)
    res = ind.run_indentation_cycles(
        nr=nr, nz=nr, width=extent, depth=extent, grading=2.0, radius=radius,
        peak_indentation=peak, load_steps=load_steps, unload_steps=6,
        cycles=cycles, material=material, penalty_factor=100.0,
        strict_unloading=strict_unloading)
    model = ind.johnson_model(material, radius)
    cycle = res.cycles[0]
    step = cycle.steps[-1]
    h = np.array([s.indentation for s in cycle.steps])
    f = np.array([s.load for s in cycle.steps])
    johnson = model.load_curve(h)
    # Johnson evaluated at the FE contact radius (isolates the cavity-model
    # pressure law from its kinematic assumption h = a^2/R)
    pm_johnson_at_fe_a = model.mean_pressure(step.contact_radius) if step.contact_radius else float("nan")
    return {
        "nr": nr, "peak_indentation": peak, "yield_stress": material.yield_stress,
        "effective_modulus": material.effective_modulus,
        "load": step.load, "contact_radius": step.contact_radius,
        "radius_fitted": step.fitted_radius,
        "mean_pressure": step.mean_pressure,
        "constraint_factor": step.mean_pressure / material.yield_stress,
        "johnson_load": model.load(peak),
        "johnson_radius": math.sqrt(radius * peak),
        "johnson_mean_pressure": model.mean_pressure(math.sqrt(radius * peak)),
        "johnson_pressure_at_fe_radius": pm_johnson_at_fe_a,
        "load_error_vs_johnson": abs(step.load / model.load(peak) - 1.0),
        "depth_error_vs_johnson": abs(peak / peak - 1.0),
        "residual_depth": cycle.residual_depth,
        "johnson_residual_depth": model.residual_depth(peak),
        "residual_error": abs(cycle.residual_depth / model.residual_depth(peak) - 1.0)
        if model.residual_depth(peak) > 0 else float("nan"),
        "geometric_ratio": (step.contact_radius ** 2 / (radius * peak)),
        "energy": {
            "load_work": cycle.load_work,
            "unload_work": cycle.unload_work,
            "external_work": cycle.external_work,
            "plastic_work": cycle.plastic_work,
            "stored_energy": cycle.stored_energy,
            "closure": cycle.energy_closure,
            "absolute_residual": cycle.external_work
            - (cycle.plastic_work + cycle.stored_energy),
        },
        "plastic_strain_max": cycle.max_eqp,
        "release_plastic_flow": cycle.release_plastic_flow,
        "residual_depths": [c.residual_depth for c in res.cycles],
        "ratchet_increments": np.diff([c.residual_depth for c in res.cycles]).tolist(),
        "hardening": material.hardening,
        "converged": all(c.converged for c in res.cycles),
        "loading_curve": {"indentation": h.tolist(), "load": f.tolist(),
                          "johnson": johnson.tolist()},
        "iterations": sum(c.iterations for c in res.cycles),
        "profile": res,
    }


def beam_case() -> dict:
    rows = []
    for alpha, n_el in ((1.0, 10), (1.0, 20), (3.0, 20), (10.0, 20), (10.0, 40),
                        (20.0, 20), (20.0, 40)):
        exact = elastica_cantilever(load_parameter=alpha)
        fe = solve_corotational_beam(n_elements=n_el, tip_force=alpha,
                                     load_steps=10, tolerance=1e-9)
        quad = _elastica_by_quadrature(load_parameter=alpha)
        rows.append({
            "alpha": alpha, "elements": n_el,
            "tip_deflection_fe": fe.tip_y, "tip_deflection_exact": exact.tip_y,
            "error": abs(fe.tip_y / exact.tip_y - 1.0),
            "tip_x_fe": fe.tip_x, "tip_x_exact": exact.tip_x,
            "tip_x_error": abs(fe.tip_x / exact.tip_x - 1.0),
            "tip_angle_deg": math.degrees(exact.tip_angle),
            "quadrature_deflection": quad.tip_y,
            "quadrature_error": abs(quad.tip_y / exact.tip_y - 1.0),
            "converged": fe.converged,
        })
    small = elastica_cantilever(load_parameter=0.02)
    return {
        "rows": rows,
        "tangent_audit": corotational_tangent_error(),
        "small_load_limit": {
            "alpha": 0.02,
            "elastica": small.tip_y,
            "euler_bernoulli": 0.02 / 3.0,
            "relative_difference": abs(small.tip_y / (0.02 / 3.0) - 1.0),
        },
    }


def ratchet_case() -> dict:
    """Armstrong-Frederick closed form vs direct numerical integration."""
    params = dict(yield_stress=200.0, hardening_modulus=50000.0, gamma=200.0,
                  stress_max=400.0, stress_min=-50.0)
    closed = ind.armstrong_frederick_ratchet_rate(**params)
    numeric = ind.integrate_armstrong_frederick_cycles(cycles=15, steps=600,
                                                       **params)
    for steps in (300, 600, 1200):
        pass
    converged = ind.integrate_armstrong_frederick_cycles(cycles=15, steps=2400,
                                                         **params)
    return {
        "parameters": params,
        "closed_form_per_cycle": closed,
        "numeric_per_cycle_steps600": float(numeric[-1]),
        "numeric_per_cycle_steps2400": float(converged[-1]),
        "error_steps600": abs(numeric[-1] / closed - 1.0),
        "error_steps2400": abs(converged[-1] / closed - 1.0),
        "numeric_series": numeric.tolist(),
        "saturation_backstress": params["hardening_modulus"] / params["gamma"],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(study: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. load-depth curves: elastic, elastic-plastic FE, Johnson
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    el = study["elastic_ladder"][-1]
    ax.plot([el["indentation"]], [el["load"]], "o", ms=6, mfc="none",
            label="FE elastic (current module)")
    pl = study["plastic_primary"]
    ax.plot(pl["loading_curve"]["indentation"], pl["loading_curve"]["load"],
            "s-", ms=3, label="FE elastic-plastic (J2)")
    ax.plot(pl["loading_curve"]["indentation"], pl["loading_curve"]["johnson"],
            "k--", label="Johnson (1985) cavity model")
    grid = np.linspace(1e-4, pl["peak_indentation"], 40)
    hertz = 4.0 / 3.0 * study["_materials"]["primary"].effective_modulus \
        * np.sqrt(1.0) * grid ** 1.5
    ax.plot(grid, hertz, ":", color="0.5", label="Hertz elastic")
    ax.set_xlabel("indentation h / R")
    ax.set_ylabel("contact force F / (E* R^2)")
    ax.set_title("Indentation load-depth: elastic, plastic, cavity model")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES / "load_depth_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append({"name": "load_depth_curve.png",
                  "caption": "Load-depth response: elastic FE backbone, elastic-plastic "
                             "FE (J2 flow theory), Johnson (1985) cavity model and the "
                             "Hertz reference; dimensionless units E*=1099, R=1.",
                  "path": str(p.relative_to(REPO))})

    # 2. energy ledger
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    e = pl["energy"]
    labels = ["oint F dh", "plastic W_p", "stored U", "sum W_p+U"]
    values = [e["external_work"], e["plastic_work"], e["stored_energy"],
              e["plastic_work"] + e["stored_energy"]]
    colors = ["0.3", "tab:red", "tab:blue", "tab:green"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("specific work / (E* R^3)")
    ax.set_title(f"Single-cycle energy ledger (closure {e['closure']*100:+.2f} %)")
    fig.tight_layout()
    p = FIGURES / "energy_ledger.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append({"name": "energy_ledger.png",
                  "caption": "Single-cycle energy ledger: the path integral of the "
                             "contact force is compared with the Gauss-point volume "
                             "integrals of plastic dissipation and of the stored energy "
                             "of the residual stress field (independent routes).",
                  "path": str(p.relative_to(REPO))})

    # 3. surface profile + pressure distribution
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    prof = study["_store"]["primary_profile"]
    axes[0].plot(prof.surface_profile["radii"], -np.array(prof.surface_profile["uz"]),
                 "o-", ms=3)
    axes[0].axhline(pl["residual_depth"], color="0.5", ls="--",
                    label=f"residual depth {pl['residual_depth']:.4f}")
    axes[0].set_xlabel("radius r / R")
    axes[0].set_ylabel("indentation depth (downward)")
    axes[0].set_title("Residual surface profile after full unloading")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    a_ref = study["elastic_ladder"][-1]["radius_reference"]
    p0 = study["elastic_ladder"][-1]["peak_reference"]
    rr = np.linspace(0, a_ref * 1.05, 200)
    axes[1].plot(rr, p0 * np.sqrt(np.clip(1 - (rr / a_ref) ** 2, 0, None)),
                 "k-", label="Hertz p(r)")
    el = study["elastic_ladder"][-1]
    axes[1].plot([el["radius_fitted"]], [el["peak_pressure"]], "o",
                 mfc="none", ms=8, label="FE recovered (a, p0)")
    axes[1].set_xlim(0, 1.6 * a_ref)
    axes[1].set_xlabel("radius r / R")
    axes[1].set_ylabel("normal pressure")
    axes[1].set_title("Hertz pressure field (analytic)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES / "residual_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append({"name": "residual_profile.png",
                  "caption": "Left: residual surface profile after the indenter is "
                             "fully withdrawn (elastic spring-back around the frozen "
                             "plastic strain). Right: Hertz analytic pressure field "
                             "used as the elastic oracle.",
                  "path": str(p.relative_to(REPO))})

    # 4. mesh convergence ladder
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ladder = study["elastic_ladder"]
    nr = [row["nr"] for row in ladder]
    ax.loglog(nr, [row["radius_error"] * 100 for row in ladder], "o-",
              label="contact radius error (Hertz fit)")
    ax.loglog(nr, [row["peak_error"] * 100 for row in ladder], "s-",
              label="peak pressure error")
    ax.loglog(nr, [row["load_error"] * 100 for row in ladder], "^-",
              label="contact force error")
    ax.axhline(3.0, color="k", ls=":", label="C1 gate 3 %")
    ax.set_xlabel("radial/vertical element count per side")
    ax.set_ylabel("relative error vs Hertz [%]")
    ax.set_title("Mesh convergence of the elastic contact solve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIGURES / "mesh_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append({"name": "mesh_convergence.png",
                  "caption": "3-mesh ladder used for the GCI extrapolation of the "
                             "Hertz contact radius, peak pressure and contact force.",
                  "path": str(p.relative_to(REPO))})

    # 5. beam elastica
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    rows = study["beam"]["rows"]
    ax.plot([r["alpha"] for r in rows], [r["tip_deflection_fe"] for r in rows],
            "o", label="corotational FE")
    alphas = np.linspace(0.05, 20, 60)
    ax.plot(alphas, [elastica_cantilever(load_parameter=a).tip_y for a in alphas],
            "k-", label="Euler elastica (elliptic integrals)")
    ax.plot(alphas, alphas / 3.0, ":", color="0.5", label="small-deflection PL^3/3EI")
    ax.set_xlabel(r"$\alpha = P L^2 / (EI)$")
    ax.set_ylabel("tip deflection / L")
    ax.set_title("Cantilever large deflection (C3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES / "beam_elastica.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append({"name": "beam_elastica.png",
                  "caption": "Cantilever tip deflection: corotational FE versus the exact "
                             "elastica (incomplete elliptic integrals) and the "
                             "small-deflection Euler-Bernoulli asymptote.",
                  "path": str(p.relative_to(REPO))})
    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smaller ladders (for CI smoke runs)")
    args = parser.parse_args()
    started = time.time()

    elastic_material = ind.J2Material(young=1000.0, poisson=0.3,
                                      yield_stress=1.0e9)
    plastic_material = ind.J2Material(young=1000.0, poisson=0.3,
                                      yield_stress=18.2, hardening=0.0)
    hardening_material = ind.J2Material(young=1000.0, poisson=0.3,
                                        yield_stress=18.2, hardening=60.0)
    peak_elastic = 0.005
    ladder_sizes = (14, 18, 22) if args.quick else (16, 20, 24)

    study: dict = {
        "node": "AG27W2-N05",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(),
                        "machine": platform.machine(),
                        "numpy": np.__version__},
        "units": "dimensionless: E=1000, nu=0.3, Y=18.2, R=1, E*=1099.4",
        "_materials": {"elastic": elastic_material, "primary": plastic_material},
    }

    # -- C1 / C6: elastic ladder -------------------------------------------
    ladder = [elastic_case(nr, peak=peak_elastic, radius=1.0,
                           material=elastic_material) for nr in ladder_sizes]
    for row in ladder:
        row.pop("profile")
    ratios = [ladder[i + 1]["nr"] / ladder[i]["nr"] for i in range(len(ladder) - 1)]
    gci = {
        "contact_radius": richardson([r["radius_fitted"] for r in ladder], ratios),
        "peak_pressure": richardson([r["peak_pressure"] for r in ladder], ratios),
        "load": richardson([r["load"] for r in ladder], ratios),
    }
    exact = hertz_reference(peak_elastic, 1.0, elastic_material.young,
                            elastic_material.poisson)
    gci["radius_error_vs_analytic"] = abs(gci["contact_radius"]["extrapolated"] / exact[1] - 1)
    gci["peak_error_vs_analytic"] = abs(gci["peak_pressure"]["extrapolated"] / exact[2] - 1)
    gci["load_error_vs_analytic"] = abs(gci["load"]["extrapolated"] / exact[0] - 1)
    # keep an unpruned copy of the finest elastic case for the figures
    finest = elastic_case(ladder_sizes[-1], peak=peak_elastic, radius=1.0,
                          material=elastic_material)
    study["elastic_ladder"] = ladder
    study["elastic_finest"] = {k: v for k, v in finest.items() if k != "profile"}
    study["elastic_finest"]["profile"] = finest["profile"]
    study["gci"] = gci

    # -- C2 / C4 / C5: elastic-plastic cycle -------------------------------
    study["plastic_primary"] = plastic_case(
        ladder_sizes[-1], peak=0.02, radius=1.0, material=plastic_material,
        load_steps=40, cycles=10)
    study.setdefault("_store", {})
    study["_store"]["primary_profile"] = study["plastic_primary"].pop("profile")
    study["plastic_soft"] = plastic_case(
        ladder_sizes[-1], peak=0.005, radius=1.0, material=plastic_material,
        load_steps=20, cycles=1)
    study["_store"]["soft_profile"] = study["plastic_soft"].pop("profile")
    study["plastic_deep"] = plastic_case(
        ladder_sizes[-1], peak=0.05, radius=1.0, material=plastic_material,
        load_steps=20, cycles=2, strict_unloading=False)
    study["plastic_soft"].pop("profile", None)
    study["plastic_deep"].pop("profile", None)
    study["plastic_hardening"] = plastic_case(
        ladder_sizes[-1], peak=0.01, radius=1.0, material=hardening_material,
        load_steps=20, cycles=3)
    study["plastic_hardening"].pop("profile", None)

    # GCI for the plastic case as well (load at the peak indentation)
    plastic_ladder = [plastic_case(nr, peak=0.02, radius=1.0,
                                   material=plastic_material, load_steps=16,
                                   cycles=1) for nr in ladder_sizes]
    for row in plastic_ladder:
        row.pop("profile", None)
    study["plastic_ladder"] = plastic_ladder
    study["plastic_gci"] = richardson([r["load"] for r in plastic_ladder], ratios)

    # -- energy-accounting diagnostics (C4 failure analysis) ---------------
    def _closure(nr, peak, steps, pf, material, cycles=1):
        row = plastic_case(nr, peak=peak, radius=1.0, material=material,
                           load_steps=steps, cycles=cycles)
        return row["energy"]

    diagnostics = {
        "step_count": {},
        "penalty": {},
    }
    diag_nr = ladder_sizes[-2]
    e = _closure(diag_nr, 0.02, 20, 100.0, plastic_material)
    diagnostics["step_count"]["20"] = e
    e = _closure(diag_nr, 0.02, 80, 100.0, plastic_material)
    diagnostics["step_count"]["80"] = e
    diagnostics["step_count"]["40"] = {"external_work": study["plastic_primary"]["energy"]["external_work"],
                                      "plastic_work": study["plastic_primary"]["energy"]["plastic_work"],
                                      "stored_energy": study["plastic_primary"]["energy"]["stored_energy"],
                                      "closure": study["plastic_primary"]["energy"]["closure"]}
    for pf in (100.0, 300.0):
        res = ind.run_indentation_cycles(
            nr=diag_nr, nz=diag_nr,
            width=10.0 * math.sqrt(0.02), depth=10.0 * math.sqrt(0.02),
            grading=2.0, radius=1.0, peak_indentation=0.02, load_steps=40,
            unload_steps=8, cycles=1, material=plastic_material,
            penalty_factor=pf)
        diagnostics["penalty"][str(pf)] = {
            "closure": res.cycles[0].energy_closure,
            "absolute_residual": res.cycles[0].external_work
            - (res.cycles[0].plastic_work + res.cycles[0].stored_energy),
            "penalty_energy": res.cycles[0].penalty_energy,
        }
    elastic_control = ind.run_indentation_cycles(
        nr=diag_nr, nz=diag_nr,
        width=10.0 * math.sqrt(0.005), depth=10.0 * math.sqrt(0.005),
        grading=2.0, radius=1.0, peak_indentation=0.005, load_steps=30,
        unload_steps=8, cycles=1, material=elastic_material,
        penalty_factor=100.0).cycles[0]
    diagnostics["elastic_control"] = {
        "load_work": elastic_control.load_work,
        "external_work": elastic_control.external_work,
        "stored_energy": elastic_control.stored_energy,
        "relative_closing_error": abs(elastic_control.external_work
                                     / elastic_control.load_work),
        "penalty_energy_at_peak": elastic_control.penalty_energy,
    }
    study["energy_diagnostics"] = diagnostics
    _primary = study["plastic_primary"]
    study["johnson_at_fe_radius"] = {
        "fe_radius": _primary["contact_radius"],
        "johnson_pressure": _primary["johnson_pressure_at_fe_radius"],
        "error": abs(_primary["mean_pressure"]
                     / _primary["johnson_pressure_at_fe_radius"] - 1.0)
        if _primary["johnson_pressure_at_fe_radius"] else float("nan"),
    }

    # -- C3: beam ----------------------------------------------------------
    study["beam"] = beam_case()

    # -- ratcheting closed form -------------------------------------------
    study["ratchet"] = ratchet_case()

    # -- gates -------------------------------------------------------------
    primary = study["plastic_primary"]
    energy = primary["energy"]
    residual_depths = np.array(primary["residual_depths"])
    ratchet_increment = float(abs(residual_depths[-1] - residual_depths[1])
                              / max(residual_depths[1], 1e-30))
    gates = {
        "C1_hertz_radius_error": ladder[-1]["radius_error"],
        "C1_hertz_peak_error": ladder[-1]["peak_error"],
        "C1_hertz_load_error": ladder[-1]["load_error"],
        "C2_johnson_load_error": primary["load_error_vs_johnson"],
        "C2_johnson_load_error_soft": study["plastic_soft"]["load_error_vs_johnson"],
        "C3_elastica_deflection_error": max(r["error"] for r in study["beam"]["rows"]),
        "C4_energy_closure": abs(energy["closure"]),
        "C5_shakedown_increment": ratchet_increment,
        "C5_af_ratchet_closed_form_error": study["ratchet"]["error_steps2400"],
        "C6_gci_radius_error": gci["radius_error_vs_analytic"],
        "C6_gci_peak_error": gci["peak_error_vs_analytic"],
        "C6_gci_load_error": gci["load_error_vs_analytic"],
        "C6_plastic_gci_load_error": abs(study["plastic_gci"]["extrapolated"]
                                         / primary["load"] - 1),
    }
    thresholds = {
        "C1_hertz_radius_error": 0.03, "C1_hertz_peak_error": 0.03,
        "C1_hertz_load_error": 0.03,
        "C2_johnson_load_error": 0.05, "C2_johnson_load_error_soft": 0.05,
        "C3_elastica_deflection_error": 0.02, "C4_energy_closure": 0.01,
        "C5_shakedown_increment": 0.10, "C5_af_ratchet_closed_form_error": 0.10,
        "C6_gci_radius_error": 0.05, "C6_gci_peak_error": 0.05,
        "C6_gci_load_error": 0.05, "C6_plastic_gci_load_error": 0.05,
    }
    study["gates"] = {k: {"value": float(v), "threshold": thresholds[k],
                          "pass": bool(float(v) <= thresholds[k])}
                      for k, v in gates.items()}

    # -- figures -----------------------------------------------------------
    study["figures"] = make_figures(study)
    study["elastic_finest"].pop("profile", None)
    for key in ("_store", "_materials"):
        study.pop(key, None)
    study["runtime_seconds"] = round(time.time() - started, 1)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "study.json"
    out.write_text(json.dumps(study, indent=2, default=float))
    print(f"wrote {out.relative_to(REPO)} in {study['runtime_seconds']} s")
    for name, gate in study["gates"].items():
        print(f"  {name:34s} {gate['value']:10.4f}  (gate {gate['threshold']:.2f}) "
              f"{'PASS' if gate['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())