"""Audit of the 2-D plane-strain Q4 locking debt and its B-bar cure.

Produces ``benchmarks/plane_strain_bbar_locking.json`` (evidence) and the figures
under ``docs/assets/plane_strain_bbar_locking/``.  Run with::

    OMP_NUM_THREADS=4 PYTHONPATH=src python3 scripts/audit_plane_strain_bbar_locking.py

Scoring is the plane-strain Lamé field of a pressurised thick cylinder, i.e. the
same oracle :func:`tensorfem.incompressible_locking.plane_strain_sweep` already
uses for the merged fully integrated Q4.  Nothing here is a new solver: it
qualifies the mean-dilatation element of :mod:`tensorfem.continuum_bbar` against
an exact solution and re-states the uncured standard path.

Every number in the Markdown report is copied from this script's stdout/JSON by
the audit run, not typed by hand.
"""
from __future__ import annotations

import json
import math
import pathlib

import torch

torch.set_default_dtype(torch.float64)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.incompressible_locking import plane_strain_case  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "plane_strain_bbar_locking"
EVIDENCE = ROOT / "benchmarks" / "plane_strain_bbar_locking.json"

POISSONS = (0.3, 0.45, 0.49, 0.499, 0.4999)
FORMULATIONS = ("standard", "bbar")
BASE = dict(n_theta=16, n_radial=8)
REFINEMENT = (8, 4), (16, 8), (32, 16), (48, 24)


def sweep():
    records = []
    for poisson in POISSONS:
        for formulation in FORMULATIONS:
            record = plane_strain_case(poisson, formulation=formulation, **BASE)
            records.append(record)
            print(f"nu={poisson:<7} {record['formulation']:<22} "
                  + " ".join(f"{key}={100.0 * value:9.4f}%" for key, value in record["errors"].items()))
    return records


def refinement(poisson=0.4999):
    records = []
    for n_theta, n_radial in REFINEMENT:
        for formulation in FORMULATIONS:
            record = plane_strain_case(poisson, formulation=formulation,
                                       n_theta=n_theta, n_radial=n_radial)
            records.append(record)
            print(f"refinement {n_theta}x{n_radial:<3} {record['formulation']:<22} "
                  f"dofs={record['dofs']:<5} "
                  + " ".join(f"{key}={100.0 * value:9.4f}%" for key, value in record["errors"].items()))
    return records


def observed_order(records, formulation, key):
    """Observed order from the two finest meshes, h proportional to 1/n_theta."""
    rows = [r for r in records if r["formulation"].endswith("bbar") == (formulation == "bbar")]
    rows.sort(key=lambda r: r["n_theta"])
    if len(rows) < 2:
        return None
    coarse, fine = rows[-2], rows[-1]
    ratio = fine["errors"][key] / coarse["errors"][key]
    h_ratio = coarse["n_theta"] / fine["n_theta"]
    if ratio <= 0:
        return None
    return math.log(ratio) / math.log(h_ratio)


def figure_error_vs_poisson(records):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for axis, key, title in zip(axes, ("displacement_l2", "radial_stress_l2"),
                                ("Radial displacement $L_2$", "Radial stress $L_2$")):
        for formulation, style, label in (("q4 plane strain", "o-", "Q4 full integration"),
                                         ("q4 plane strain bbar", "s-", "Q4 B-bar (mean dilatation)")):
            rows = sorted((r for r in records if r["formulation"] == formulation),
                          key=lambda r: r["poisson_ratio"])
            axis.semilogy([r["poisson_ratio"] for r in rows],
                          [100.0 * r["errors"][key] for r in rows], style, label=label)
        axis.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % gate")
        axis.set_xlabel(r"Poisson ratio $\nu$")
        axis.set_ylabel("relative error [%]")
        axis.set_title(f"{title}, 16x8 Q4, 306 DOF")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "error_vs_poisson.png", dpi=150)
    plt.close(fig)


def figure_refinement(records, poisson=0.4999):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for axis, key, title in zip(axes, ("displacement_l2", "radial_stress_l2"),
                                ("Radial displacement $L_2$", "Radial stress $L_2$")):
        for formulation, style, label in (("q4 plane strain", "o-", "Q4 full integration"),
                                         ("q4 plane strain bbar", "s-", "Q4 B-bar (mean dilatation)")):
            rows = sorted((r for r in records if r["formulation"] == formulation),
                          key=lambda r: r["n_theta"])
            axis.loglog([r["n_theta"] for r in rows], [100.0 * r["errors"][key] for r in rows],
                        style, label=label)
        axis.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % gate")
        axis.set_xlabel(r"$n_\theta$ (elements along the circumference)")
        axis.set_ylabel("relative error [%]")
        axis.set_title(f"{title}, $\\nu$ = {poisson}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "refinement.png", dpi=150)
    plt.close(fig)


def figure_field(poisson=0.4999):
    """Radial stress along the radius: exact vs both formulations (element centres)."""
    from tensorfem.continuum import ContinuumModel, solve_continuum
    from tensorfem.continuum_bbar import solve_continuum_bbar
    from tensorfem.lame_reference import (cartesian_to_polar_stress, circular_pressure_loads,
                                          lame_state, sector_q4_mesh, symmetry_fixed_dofs)

    a, b, pressure, young = 0.5, 0.8, 100.0e6, 210.0e9
    n_theta, n_radial = BASE["n_theta"], BASE["n_radial"]
    nodes, cells = sector_q4_mesh(a, b, n_theta, n_radial)
    forces = circular_pressure_loads(n_theta, n_radial, a, pressure, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(nodes, cells, torch.tensor(young), torch.tensor(float(poisson)),
                           torch.tensor(1.0), forces, fixed, plane="strain")
    centres = nodes[cells].mean(1)
    radii = torch.linalg.vector_norm(centres, dim=1)
    exact = lame_state(radii, inner_radius=a, outer_radius=b, inner_pressure=pressure,
                       young_modulus=young, poisson_ratio=float(poisson), plane="strain")
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    order = torch.argsort(radii)
    axis.plot(radii[order] / 1e3, exact.radial_stress[order] / 1e6, "k-", lw=1.6, label="exact Lamé")
    for solver, style, label in ((solve_continuum, "o", "Q4 full integration"),
                                 (solve_continuum_bbar, "s", "Q4 B-bar")):
        result = solver(model)
        sigma_r, _, _ = cartesian_to_polar_stress(result.stress, centres)
        axis.plot(radii[order] / 1e3, sigma_r[order] / 1e6, style, ms=4, alpha=0.8, label=label)
    axis.set_xlabel("radius $r$ [m]")
    axis.set_ylabel(r"radial stress $\sigma_r$ [MPa]")
    axis.set_title(f"plane strain, $\\nu$ = {poisson}, {n_theta}x{n_radial} Q4")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "radial_stress_profile.png", dpi=150)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("== Poisson sweep, 16x8 Q4 (306 DOF) ==")
    sweep_records = sweep()
    print("== refinement at nu = 0.4999 ==")
    refinement_records = refinement()
    figure_error_vs_poisson(sweep_records)
    figure_refinement(refinement_records)
    figure_field()
    orders = {}
    for formulation in ("standard", "bbar"):
        orders[formulation] = {
            key: observed_order(refinement_records, formulation, key)
            for key in ("displacement_l2", "radial_stress_l2")
        }
    print("observed orders (two finest meshes):", json.dumps(orders, indent=2))
    payload = {
        "case": "pressurised thick cylinder, plane strain, inner radius 0.5 m, outer 0.8 m, "
                "p = 100 MPa, E = 210 GPa",
        "reference": "plane-strain Lame field (tensorfem.lame_reference.lame_state)",
        "sweep": sweep_records,
        "refinement": refinement_records,
        "observed_order": orders,
        "figures": ["error_vs_poisson.png", "refinement.png", "radial_stress_profile.png"],
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()