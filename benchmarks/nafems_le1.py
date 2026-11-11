#!/usr/bin/env python
"""NAFEMS LE1 benchmark: Elliptical membrane stress concentration.

Reference: NAFEMS "A Finite Element Benchmark Set" (1990), Test LE1.
Geometry: Plate with central elliptical hole, semi_x=100mm (along load),
          semi_y=50mm (transverse), normalized to semi_x=1.0, semi_y=0.5.
Material: E=210 GPa, nu=0.3 (normalized to E=1.0, nu=0.3).
Loading: Uniaxial tension sigma=100 MPa (normalized to T=1.0).
Target: Hoop stress at transverse tip (0, semi_y) = K_t * T = 2.0 * T.

The analytical solution is the Inglis (1913) exact solution for an elliptical
hole in an infinite plate under uniaxial tension:
    K_t = 1 + 2 * (semi_y / semi_x) = 1 + 2 * 0.5 = 2.0

This script runs the TensorFEM production plane-stress path (Q4 elements)
on a confocal elliptical annulus and compares with the exact solution.

Usage:
    PYTHONPATH=src python benchmarks/nafems_le1.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.elliptic_hole import (
    InglisHole,
    boundary_hoop_profile,
    convergence_study,
    solve_case,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts" / "nafems_le1"
RUNS = REPO / "runs" / "nafems_le1"
ASSETS = REPO / "docs" / "assets" / "nafems_le1"

# NAFEMS LE1 geometry (normalized)
SEMI_X = 1.0  # along load direction
SEMI_Y = 0.5  # transverse to load
FAR_FIELD = 1.0  # normalized tension
YOUNG = 1.0
POISSON = 0.3
RHO_OUTER = 5.0  # confocal outer ellipse
GRADING = 3.0  # radial grading factor

# Mesh levels for convergence study
LEVELS = (16, 20, 24, 32, 40, 48)

# Tolerance
TOLERANCE_PCT = 5.0


def run_convergence():
    """Run convergence study for NAFEMS LE1."""
    hole = InglisHole(
        semi_x=SEMI_X, semi_y=SEMI_Y,
        far_field=FAR_FIELD, young=YOUNG, poisson=POISSON
    )

    print(f"NAFEMS LE1 Benchmark")
    print(f"  semi_x={hole.semi_x}, semi_y={hole.semi_y}")
    print(f"  tip_ratio={hole.tip_ratio}")
    print(f"  K_t (exact)={hole.stress_concentration}")
    print(f"  transverse_tip={hole.transverse_tip}")
    print(f"  load_axis_tip={hole.load_axis_tip}")
    print()

    results = []
    for n in LEVELS:
        t0 = time.time()
        result = solve_case(
            hole=hole, rho_outer=RHO_OUTER,
            n_theta=n, n_radial=n, grading=GRADING
        )
        elapsed = time.time() - t0
        tip = result['transverse_tip']
        load_tip = result['load_axis_tip']

        row = {
            "n": n,
            "dofs": result["dofs"],
            "elapsed_s": round(elapsed, 2),
            "transverse_tip": {
                "hoop_fe": tip["hoop_fe"],
                "exact": tip["exact"],
                "relative_error_pct": round(tip["relative_error"] * 100, 4),
                "within_tolerance": tip["relative_error"] * 100 < TOLERANCE_PCT,
            },
            "load_axis_tip": {
                "hoop_fe": load_tip["hoop_fe"],
                "exact": load_tip["exact"],
                "relative_error_pct": round(load_tip["relative_error"] * 100, 4),
            },
            "displacement_error": {
                "max_relative": result["displacement_error"]["max_relative"],
                "rms_absolute": result["displacement_error"]["rms_absolute"],
            },
            "force_balance_relative": result["force_balance_relative"],
            "symmetry_edge_relative_error": result["symmetry_edge"]["relative_error"],
        }
        results.append(row)
        status = "PASS" if row["transverse_tip"]["within_tolerance"] else "FAIL"
        print(f"  n={n:3d}: DOF={result['dofs']:5d}, "
              f"tip_hoop={tip['hoop_fe']:.6f}, "
              f"exact={tip['exact']:.6f}, "
              f"error={tip['relative_error']*100:.4f}% [{status}] "
              f"({elapsed:.1f}s)")

    return hole, results


def generate_figures(hole, results):
    """Generate report figures."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    # Figure 1: Convergence of tip stress
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r["n"] for r in results]
    errors = [r["transverse_tip"]["relative_error_pct"] for r in results]
    ax.semilogy(ns, errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=TOLERANCE_PCT, color='r', linestyle='--', linewidth=1.5,
               label=f'{TOLERANCE_PCT}% tolerance')
    ax.set_xlabel('Mesh level (n_theta = n_radial = n)')
    ax.set_ylabel('Relative error in tip hoop stress (%)')
    ax.set_title('NAFEMS LE1: Convergence of Stress Concentration Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png", dpi=150)
    plt.close(fig)

    # Figure 2: Stress profile along transverse axis
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use the finest mesh result for the profile
    finest = results[-1]
    n_finest = finest["n"]
    result_finest = solve_case(
        hole=hole, rho_outer=RHO_OUTER,
        n_theta=n_finest, n_radial=n_finest, grading=GRADING
    )
    sample_rows = result_finest["sample_rows"]
    y_vals = [s["y"] for s in sample_rows]
    sigma_fe = [s["sigma_xx_fe"] for s in sample_rows]
    sigma_exact = [s["sigma_xx_exact"] for s in sample_rows]

    ax.plot(y_vals, sigma_fe, 'bo-', label='TensorFEM Q4', linewidth=2, markersize=6)
    ax.plot(y_vals, sigma_exact, 'r--', label='Inglis exact', linewidth=2)
    ax.axvline(x=hole.semi_y, color='g', linestyle=':', alpha=0.5,
               label='Hole boundary')
    ax.set_xlabel('y (transverse direction)')
    ax.set_ylabel(r'$\sigma_{xx}$ / T')
    ax.set_title('NAFEMS LE1: Stress Profile along Transverse Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "stress_profile.png", dpi=150)
    plt.close(fig)

    # Figure 3: von Mises stress field
    fig, ax = plt.subplots(figsize=(8, 6))
    # Generate field on a grid
    nx, ny = 100, 100
    x = np.linspace(0, hole.semi_x * RHO_OUTER * 0.8, nx)
    y = np.linspace(0, hole.semi_y * RHO_OUTER * 0.8, ny)
    X, Y = np.meshgrid(x, y)
    vm = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            # Check if point is outside the ellipse
            if (X[j, i] / hole.semi_x) ** 2 + (Y[j, i] / hole.semi_y) ** 2 >= 1.0:
                vm[j, i] = hole.von_mises_cartesian(X[j, i], Y[j, i])
            else:
                vm[j, i] = np.nan

    # Mask the hole
    mask = (X / hole.semi_x) ** 2 + (Y / hole.semi_y) ** 2 < 1.0
    vm = np.ma.array(vm, mask=mask)

    im = ax.contourf(X, Y, vm, levels=30, cmap='viridis')
    # Draw hole boundary
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(hole.semi_x * np.cos(theta), hole.semi_y * np.sin(theta),
            'r-', linewidth=2, label='Hole boundary')
    ax.set_xlabel('x (load direction)')
    ax.set_ylabel('y (transverse direction)')
    ax.set_title('NAFEMS LE1: von Mises Stress Field (Exact)')
    ax.set_aspect('equal')
    ax.legend()
    fig.colorbar(im, ax=ax, label=r'$\sigma_{vm}$ / T')
    fig.tight_layout()
    fig.savefig(ASSETS / "von_mises_field.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {ASSETS}")


def write_report(hole, results):
    """Write JSON report and markdown summary."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "benchmark": "NAFEMS LE1",
        "description": "Elliptical membrane stress concentration",
        "reference": "NAFEMS A Finite Element Benchmark Set (1990), Test LE1",
        "analytical_solution": "Inglis (1913) exact solution for elliptical hole",
        "geometry": {
            "semi_x": hole.semi_x,
            "semi_y": hole.semi_y,
            "tip_ratio": hole.tip_ratio,
            "rho_outer": RHO_OUTER,
            "grading": GRADING,
        },
        "material": {
            "young": hole.young,
            "poisson": hole.poisson,
            "plane": hole.plane,
        },
        "loading": {
            "far_field": hole.far_field,
            "type": "uniaxial tension along x",
        },
        "target": {
            "quantity": "hoop stress at transverse tip (0, semi_y)",
            "exact_value": hole.stress_concentration * hole.far_field,
            "stress_concentration_factor": hole.stress_concentration,
        },
        "tolerance_pct": TOLERANCE_PCT,
        "convergence_results": results,
        "summary": {
            "finest_mesh": results[-1],
            "coarsest_mesh": results[0],
            "all_within_tolerance": all(
                r["transverse_tip"]["within_tolerance"] for r in results
            ),
            "first_passing_level": next(
                (r["n"] for r in results if r["transverse_tip"]["within_tolerance"]),
                None
            ),
        },
    }

    json_path = ARTIFACTS / "nafems_le1_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report: {json_path}")

    # Compute SHA256 for provenance
    sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
    print(f"Report SHA256: {sha}")

    # Markdown summary
    md_path = ARTIFACTS / "nafems_le1_summary.md"
    with open(md_path, "w") as f:
        f.write("# NAFEMS LE1 Benchmark Report\n\n")
        f.write("## 1. Problem Description\n\n")
        f.write("**Reference:** NAFEMS \"A Finite Element Benchmark Set\" (1990), Test LE1\n\n")
        f.write("**Physical background:** Stress concentration in a plate with a central ")
        f.write("elliptical hole under uniaxial tension. This is a classical elasticity ")
        f.write("problem with an exact analytical solution due to Inglis (1913).\n\n")
        f.write("## 2. Geometry and Parameters\n\n")
        f.write("| Parameter | Symbol | Value |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Semi-axis along load | semi_x | {hole.semi_x} |\n")
        f.write(f"| Semi-axis transverse | semi_y | {hole.semi_y} |\n")
        f.write(f"| Tip ratio | semi_y/semi_x | {hole.tip_ratio} |\n")
        f.write(f"| Far-field tension | T | {hole.far_field} |\n")
        f.write(f"| Young's modulus | E | {hole.young} |\n")
        f.write(f"| Poisson's ratio | ν | {hole.poisson} |\n")
        f.write(f"| Plane condition | — | {hole.plane} |\n")
        f.write(f"| Outer boundary | ρ₀ | {RHO_OUTER} |\n")
        f.write(f"| Radial grading | — | {GRADING} |\n\n")
        f.write("## 3. Analytical Solution\n\n")
        f.write("The exact stress concentration factor for an elliptical hole is:\n\n")
        f.write("```\n")
        f.write(f"K_t = 1 + 2 × (semi_y / semi_x) = 1 + 2 × {hole.tip_ratio} = {hole.stress_concentration}\n")
        f.write("```\n\n")
        f.write(f"The target hoop stress at the transverse tip (0, {hole.semi_y}) is:\n\n")
        f.write("```\n")
        f.write(f"σ_xx = K_t × T = {hole.stress_concentration} × {hole.far_field} = {hole.stress_concentration * hole.far_field}\n")
        f.write("```\n\n")
        f.write("## 4. Convergence Results\n\n")
        f.write("| n | DOF | σ_xx (FE) | σ_xx (exact) | Error (%) | Status |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            tip = r["transverse_tip"]
            status = "✓ PASS" if tip["within_tolerance"] else "✗ FAIL"
            f.write(f"| {r['n']} | {r['dofs']} | {tip['hoop_fe']:.6f} | "
                    f"{tip['exact']:.6f} | {tip['relative_error_pct']:.4f}% | {status} |\n")
        f.write("\n")
        f.write("## 5. Error Assessment\n\n")
        finest = results[-1]
        tip = finest["transverse_tip"]
        f.write(f"**Finest mesh (n={finest['n']}):**\n\n")
        f.write(f"- Tip hoop stress (FE): {tip['hoop_fe']:.6f}\n")
        f.write(f"- Tip hoop stress (exact): {tip['exact']:.6f}\n")
        f.write(f"- Relative error: {tip['relative_error_pct']:.4f}%\n")
        f.write(f"- Tolerance: {TOLERANCE_PCT}%\n")
        f.write(f"- **Result: {'PASS' if tip['within_tolerance'] else 'FAIL'}**\n\n")
        f.write("## 6. Figures\n\n")
        f.write("- `convergence.png`: Convergence of tip stress error vs mesh level\n")
        f.write("- `stress_profile.png`: Stress profile along transverse axis\n")
        f.write("- `von_mises_field.png`: von Mises stress field\n\n")

    print(f"Markdown summary: {md_path}")
    return report


def main():
    print("=" * 60)
    print("NAFEMS LE1 Benchmark: Elliptical Membrane Stress Concentration")
    print("=" * 60)
    print()

    hole, results = run_convergence()
    print()
    generate_figures(hole, results)
    print()
    report = write_report(hole, results)

    # Final verdict
    print()
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    finest = results[-1]
    tip = finest["transverse_tip"]
    if tip["within_tolerance"]:
        print(f"PASS: Error {tip['relative_error_pct']:.4f}% < {TOLERANCE_PCT}% tolerance")
    else:
        print(f"FAIL: Error {tip['relative_error_pct']:.4f}% >= {TOLERANCE_PCT}% tolerance")
    print(f"First passing mesh level: n={report['summary']['first_passing_level']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
