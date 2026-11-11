#!/usr/bin/env python3
"""CASE-CANTILEVER-TIP: Cantilever beam tip deflection benchmark.

Reference: Euler-Bernoulli beam theory (δ = PL³/3EI) and Timoshenko beam theory
(δ = PL³/3EI + PL/κAG).  Geometry: L=1 m, b=0.01 m, h=0.01 m (L/h=100).
Material: E=200 GPa, ν=0.3.  Load: P=5 N at free end.

Tolerance: <3% relative error vs Euler-Bernoulli analytic.
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorfem.frame2d import FrameModel, solve_frame_static
from tensorfem.hex20 import Hex20Model, solve_hex20, structured_hex20_mesh

# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
L = 1.0          # beam length [m]
b = 0.01         # width [m]
h = 0.01         # height [m]
E = 200e9        # Young's modulus [Pa]
nu = 0.3         # Poisson's ratio
P = 5.0          # tip load [N]

A = b * h
I = b * h**3 / 12.0
G = E / (2.0 * (1.0 + nu))
kappa = 5.0 / 6.0  # shear correction factor (rectangular section)

# Analytic references
delta_eb = P * L**3 / (3.0 * E * I)          # Euler-Bernoulli
delta_shear = P * L / (kappa * A * G)         # shear contribution
delta_timoshenko = delta_eb + delta_shear     # Timoshenko

DTYPE = torch.float64
TOLERANCE = 0.03  # 3%

OUT_DIR = Path(__file__).resolve().parent / "cantilever_tip_artifacts"
OUT_DIR.mkdir(exist_ok=True)


def run_frame_levels(n_elements_list=(4, 8, 16, 32)):
    """Cantilever with cubic-Hermite Euler-Bernoulli frame elements."""
    rows = []
    for n in n_elements_list:
        t0 = time.perf_counter()
        nodes = torch.stack([
            torch.linspace(0.0, L, n + 1, dtype=DTYPE),
            torch.zeros(n + 1, dtype=DTYPE)
        ], dim=1)
        elements = torch.stack([torch.arange(n), torch.arange(1, n + 1)], dim=1)
        forces = torch.zeros(3 * (n + 1), dtype=DTYPE)
        forces[3 * n + 1] = -P
        model = FrameModel(
            nodes=nodes, elements=elements,
            young_modulus=torch.tensor([E], dtype=DTYPE),
            area=torch.tensor([A], dtype=DTYPE),
            inertia=torch.tensor([I], dtype=DTYPE),
            forces=forces, fixed_dofs=torch.tensor([0, 1, 2])
        )
        result = solve_frame_static(model)
        tip = -float(result.displacement[3 * n + 1])
        rows.append({
            "n_elements": n,
            "dofs": int(model.n_dofs),
            "tip_deflection_m": tip,
            "relative_error_vs_eb": (tip - delta_eb) / delta_eb,
            "seconds": time.perf_counter() - t0,
        })
    return rows


def run_hex20_levels(levels=((8, 1), (16, 1), (16, 2), (32, 2))):
    """Cantilever as 3-D HEX20 solid (h-refinement ladder)."""
    rows = []
    for n_axial, n_cross in levels:
        t0 = time.perf_counter()
        nodes, cells = structured_hex20_mesh(
            L, b, h, n_axial, n_cross, n_cross, dtype=DTYPE
        )
        right = torch.where(nodes[:, 0] == L)[0]
        forces = torch.zeros(3 * len(nodes), dtype=DTYPE)
        forces[3 * right + 2] = -P / len(right)
        left = torch.where(nodes[:, 0] == 0.0)[0]
        fixed = torch.stack([3 * left, 3 * left + 1, 3 * left + 2], dim=1).reshape(-1)
        model = Hex20Model(
            nodes=nodes, elements=cells,
            young_modulus=torch.tensor(E, dtype=DTYPE),
            poisson_ratio=torch.tensor(nu, dtype=DTYPE),
            forces=forces, fixed_dofs=fixed
        )
        result = solve_hex20(model)
        tip = -float(result.displacement[3 * right + 2].mean())
        rows.append({
            "axial_elements": n_axial,
            "cross_elements": n_cross,
            "nodes": int(len(nodes)),
            "dofs": int(3 * len(nodes)),
            "tip_deflection_m": tip,
            "relative_error_vs_eb": (tip - delta_eb) / delta_eb,
            "seconds": time.perf_counter() - t0,
        })
    return rows


def make_convergence_figure(frame_rows, hex20_rows):
    """Plot convergence of tip deflection vs DOFs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Frame elements
    frame_dofs = [r["dofs"] for r in frame_rows]
    frame_err = [abs(r["relative_error_vs_eb"]) for r in frame_rows]
    ax.loglog(frame_dofs, frame_err, "bo-", label="Frame (EB beam elem)", linewidth=2, markersize=8)

    # HEX20 solid
    hex_dofs = [r["dofs"] for r in hex20_rows]
    hex_err = [abs(r["relative_error_vs_eb"]) for r in hex20_rows]
    ax.loglog(hex_dofs, hex_err, "rs-", label="HEX20 solid", linewidth=2, markersize=8)

    # 3% tolerance line
    ax.axhline(y=TOLERANCE, color="gray", linestyle="--", linewidth=1, label="3% tolerance")

    ax.set_xlabel("Number of DOFs", fontsize=12)
    ax.set_ylabel("Absolute relative error", fontsize=12)
    ax.set_title("Cantilever tip deflection: convergence vs DOFs\n"
                 f"L={L} m, b={b} m, h={h} m, L/h={L/h:.0f}, E={E/1e9:.0f} GPa, P={P} N",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / "convergence.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def make_comparison_figure(frame_rows, hex20_rows):
    """Bar chart: tip deflection comparison."""
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = ["Euler-Bernoulli\n(analytic)", "Timoshenko\n(analytic)"]
    values = [delta_eb, delta_timoshenko]
    colors = ["#2196F3", "#4CAF50"]

    for i, r in enumerate(frame_rows):
        labels.append(f"Frame\n(n={r['n_elements']})")
        values.append(r["tip_deflection_m"])
        colors.append("#FF9800")

    for i, r in enumerate(hex20_rows):
        labels.append(f"HEX20\n({r['axial_elements']}×{r['cross_elements']})")
        values.append(r["tip_deflection_m"])
        colors.append("#9C27B0")

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate values
    for i, (xi, vi) in enumerate(zip(x, values)):
        ax.text(xi, vi, f"{vi:.4e}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.axhline(y=delta_eb, color="blue", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Tip deflection [m]", fontsize=11)
    ax.set_title("Cantilever tip deflection: method comparison", fontsize=12)
    fig.tight_layout()
    png_path = OUT_DIR / "comparison.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def main():
    print("=" * 60)
    print("CASE-CANTILEVER-TIP: Cantilever beam tip deflection benchmark")
    print("=" * 60)
    print(f"Geometry: L={L} m, b={b} m, h={h} m, L/h={L/h:.0f}")
    print(f"Material: E={E/1e9:.0f} GPa, ν={nu}")
    print(f"Load: P={P} N at free end")
    print(f"Analytic (Euler-Bernoulli): δ = {delta_eb:.6e} m")
    print(f"Analytic (Timoshenko):      δ = {delta_timoshenko:.6e} m")
    print(f"Shear contribution:          {delta_shear/delta_eb*100:.4f}%")
    print(f"Tolerance: {TOLERANCE*100:.0f}%")
    print()

    # Run frame element model
    print("[1/2] Frame elements (Euler-Bernoulli beam theory)...")
    frame_rows = run_frame_levels()
    for r in frame_rows:
        print(f"    n={r['n_elements']:3d} dofs={r['dofs']:4d} "
              f"tip={r['tip_deflection_m']:.6e} err={r['relative_error_vs_eb']*100:+.2e}% "
              f"t={r['seconds']:.3f}s")

    # Run HEX20 solid model
    print("\n[2/2] HEX20 solid elements...")
    hex20_rows = run_hex20_levels()
    for r in hex20_rows:
        print(f"    axial={r['axial_elements']:3d} cross={r['cross_elements']} "
              f"nodes={r['nodes']:5d} dofs={r['dofs']:5d} "
              f"tip={r['tip_deflection_m']:.6e} err={r['relative_error_vs_eb']*100:+.4f}% "
              f"t={r['seconds']:.1f}s")

    # Verdict
    frame_pass = all(abs(r["relative_error_vs_eb"]) < TOLERANCE for r in frame_rows)
    hex20_pass = all(abs(r["relative_error_vs_eb"]) < TOLERANCE for r in hex20_rows)
    finest_frame_err = abs(frame_rows[-1]["relative_error_vs_eb"])
    finest_hex20_err = abs(hex20_rows[-1]["relative_error_vs_eb"])

    print(f"\n--- Verdict ---")
    print(f"Frame elements: {'PASS' if frame_pass else 'FAIL'} "
          f"(finest error={finest_frame_err*100:.2e}%)")
    print(f"HEX20 solid:    {'PASS' if hex20_pass else 'FAIL'} "
          f"(finest error={finest_hex20_err*100:.4f}%)")

    # Generate figures
    print("\nGenerating figures...")
    conv_png = make_convergence_figure(frame_rows, hex20_rows)
    comp_png = make_comparison_figure(frame_rows, hex20_rows)
    print(f"  {conv_png}")
    print(f"  {comp_png}")

    # Build results dict
    results = {
        "benchmark": "cantilever_tip_deflection",
        "case_key": "CASE-CANTILEVER-TIP",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "machine": os.uname().machine if hasattr(os, "uname") else "unknown",
            "dtype": str(DTYPE),
        },
        "geometry": {
            "length_m": L, "width_m": b, "height_m": h,
            "slenderness_L_over_h": L / h,
            "area_m2": A, "inertia_m4": I,
        },
        "material": {"young_modulus_Pa": E, "poisson_ratio": nu, "shear_modulus_Pa": G},
        "load": {"tip_force_N": P, "type": "concentrated_transverse"},
        "analytic_references": {
            "euler_bernoulli": {
                "formula": "delta = P*L^3 / (3*E*I)",
                "tip_deflection_m": delta_eb,
            },
            "timoshenko": {
                "formula": "delta = P*L^3/(3*E*I) + P*L/(kappa*A*G)",
                "shear_correction_kappa": kappa,
                "shear_contribution_m": delta_shear,
                "shear_fraction_pct": delta_shear / delta_eb * 100,
                "tip_deflection_m": delta_timoshenko,
            },
        },
        "tolerance": TOLERANCE,
        "frame_results": frame_rows,
        "hex20_results": hex20_rows,
        "verdict": {
            "frame_passed": frame_pass,
            "hex20_passed": hex20_pass,
            "finest_frame_error_pct": finest_frame_err * 100,
            "finest_hex20_error_pct": finest_hex20_err * 100,
            "all_passed": frame_pass and hex20_pass,
        },
        "artifacts": {
            "convergence_png": str(conv_png.relative_to(Path(__file__).resolve().parents[1])),
            "comparison_png": str(comp_png.relative_to(Path(__file__).resolve().parents[1])),
        },
    }

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")

    return 0 if results["verdict"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
