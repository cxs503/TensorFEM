"""Benchmark run for the ice-belt elastoplastic progressive-collapse surface.

Produces ``artifacts/ice_belt_collapse/suite.json`` (all numbers, programmatic)
and figures under ``docs/assets/ice_belt_collapse/``:

* ``moment_curvature.png``  -- FEM moment-curvature vs the 1-D hardening law (C2)
* ``plate_buckling.png``    -- FEM plate-field buckling vs the CSR ``k(a/b)`` (C3)
* ``ice_belt_limit.png``    -- ice-belt load-deflection with the limit-load
  bracket and the code plastic-collapse pressure (C6)
* ``mesh_convergence.png``  -- h-refinement of the ice-belt deflection (C7)

Run:  PYTHONPATH=src python3 benchmarks/ice_belt_progressive_collapse.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ice_belt_collapse import (  # noqa: E402
    AH36, E_STEEL, IACSPressureArea, NU_STEEL, PlateModel, analytic_moment_curvature,
    csr_plate_buckling_stress, elastic_perfectly_plastic, ice_belt_case,
    inplane_free_bcs, mesh_convergence_case, plate_buckling_case,
    plastic_hinge_case, uniform_pressure_load, solve_load_control,
    structured_mesh, navier_ssss_centre, elastic_plate_case, fenves_twice_elastic_slope,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts", "ice_belt_collapse")
FIG = os.path.join(ROOT, "docs", "assets", "ice_belt_collapse")


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


def figure_moment_curvature(plt):
    rows = plastic_hinge_case(ratios=(0.25, 0.5, 1.0, 1.5, 2.0))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, colour in (("EPP", "tab:red"), (AH36.name, "tab:blue")):
        sel = [r for r in rows["rows"] if r["grade"] == label and r["converged"]]
        if not sel:
            continue
        k = [r["kappa_ratio"] for r in sel]
        ax.plot(k, [r["moment_fe"] / 1e3 for r in sel], "o", color=colour,
                label=f"FE ({label})")
        grade = elastic_perfectly_plastic() if label == "EPP" else AH36
        fine = np.linspace(0.05, max(k) * 1.05, 200)
        ax.plot(fine, [analytic_moment_curvature(grade, f * 2 * grade.sigma_0
                                                 / (grade.young * 0.1), 0.1) / 1e3
                       for f in fine], "-", color=colour, lw=1.0,
                label=f"1-D law ({label})")
    ax.set_xlabel(r"$\kappa/\kappa_y$")
    ax.set_ylabel(r"$M$ [kN m/m]")
    ax.set_title("C2 plastic hinge: layered FE vs 1-D section law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "moment_curvature.png")
    plt.close(fig)
    return rows


def figure_plate_buckling(plt):
    meshes = ((12, 8), (18, 12), (24, 16), (32, 21))
    values = [plate_buckling_case(nx=nx, ny=ny)["sigma_fe"] for nx, ny in meshes]
    code = csr_plate_buckling_stress(1.2, 0.8, 0.01, E_STEEL, NU_STEEL)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([m[0] for m in meshes], [v / 1e6 for v in values], "o-",
            label="FE plate-field buckling")
    ax.axhline(code / 1e6, color="tab:red", ls="--",
               label=r"CSR $k(a/b)\pi^2D/(b^2t)$")
    ax.set_xlabel("elements across the span")
    ax.set_ylabel(r"$\sigma_{cr}$ [MPa]")
    ax.set_title("C3 plate-field buckling vs the rule formula")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "plate_buckling.png")
    plt.close(fig)
    return {"meshes": [list(m) for m in meshes],
            "sigma_fe": values, "sigma_code": code}


def figure_ice_belt(plt):
    span_x, width, thickness = 3.0, 0.5, 0.016
    ice = IACSPressureArea(p0=2.0e6)
    pressure = float(ice.pressure(span_x * width))
    grade = elastic_perfectly_plastic()
    nodes, elems = structured_mesh(span_x, width, 24, 8)
    model = PlateModel(nodes, elems, thickness, grade)
    model.von_karman = False
    fixed = inplane_free_bcs(model)
    unit = uniform_pressure_load(model, pressure)
    lam = np.linspace(0.0, 1.0, 31)[1:]
    path = solve_load_control(model, unit, fixed, lam, tol=1e-7, max_iter=30)
    elastic = PlateModel(nodes, elems, thickness, AH36)
    probe = solve_load_control(elastic, unit, fixed, [0.05], tol=1e-9, max_iter=10)
    slope = 0.05 * pressure / probe.centre_w[-1]
    loads = np.array([l * pressure for l in path.lam]) / 1e3
    disp = np.array(path.centre_w) * 1e3
    code = 8.0 * grade.sigma_0 * thickness ** 2 / 4.0 / width ** 2 / 1e3
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(disp, loads, "o-", label="FE ice-belt response (ideal plastic)")
    ax.plot(disp, 0.5 * slope * np.array(path.centre_w) / 1e3,
            ":", color="grey", label="twice-elastic-slope line")
    ax.axhline(loads[-1], color="tab:green", ls="-.",
               label=f"highest converged load {loads[-1]:.0f} kPa")
    ax.axhline(code, color="tab:red", ls="--",
               label=f"strip yield-line {code:.0f} kPa")
    ax.set_xlabel("centre deflection [mm]")
    ax.set_ylabel("ice pressure [kPa]")
    ax.set_title("C6 ice-belt limit pressure (bracket below the limit load)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "ice_belt_limit.png")
    plt.close(fig)
    return {"loads_kPa": loads.tolist(), "deflection_mm": disp.tolist(),
            "elastic_slope": slope, "code_kPa": code,
            "lower_kPa": float(loads[-1])}


def figure_mesh_convergence(plt, conv):
    rows = conv["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    n = [np.sqrt(r["elements"]) for r in rows]
    ax.semilogx(n, [r["rel_to_finest"] * 100 for r in rows], "o-")
    ax.axhline(3.0, color="tab:red", ls="--", label="3 % gate")
    ax.axhline(-3.0, color="tab:red", ls="--")
    ax.set_xlabel("sqrt(elements)")
    ax.set_ylabel("deflection error vs finest [%]")
    ax.set_title("C7 h-refinement of the ice-belt response")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    _save(fig, "mesh_convergence.png")
    plt.close(fig)


def main():
    os.makedirs(ART, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hinge = figure_moment_curvature(plt)
    buckling = figure_plate_buckling(plt)
    ice = figure_ice_belt(plt)
    conv = mesh_convergence_case()
    figure_mesh_convergence(plt, conv)

    payload = {
        "C1_elastic_plate": elastic_plate_case(16),
        "C1_reference_navier": navier_ssss_centre(1.0, 0.8, 0.005, E_STEEL, NU_STEEL, 1e4),
        "C2_plastic_hinge": hinge,
        "C3_plate_field_buckling": buckling,
        "C6_ice_belt": ice,
        "C7_mesh_convergence": conv,
    }
    path = os.path.join(ART, "benchmark.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()