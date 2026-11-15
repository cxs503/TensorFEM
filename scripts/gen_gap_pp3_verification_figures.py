#!/usr/bin/env python3
"""Generate verification figures for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

Uses analytic solutions (Navier series, yield-line, strip bounds) — no FE solve.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    elastic_patch_plate,
    ilyushin_sine_upper_bound,
    plastic_moment,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

# Anchor geometry / material
A = B = 1.0
T = 0.02
E = 210.0e9
NU = 0.3
SY = 355.0e6
PATCH = (0.5, 0.5)

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Compute analytic bounds
    py = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH, grid=41, terms=61)
    ub = yield_line_patch_upper_bound(A, B, T, SY, PATCH)
    sine = ilyushin_sine_upper_bound(A, B, T, SY, PATCH)
    lb = strip_lower_bound(A, B, T, SY, PATCH)
    mp = plastic_moment(T, SY)

    print(f"First yield pressure: {py['pressure']/1e6:.3f} MPa")
    print(f"Yield-line upper bound: {ub['pressure']/1e6:.3f} MPa")
    print(f"Ilyushin sine upper bound: {sine['pressure']/1e6:.3f} MPa")
    print(f"Strip lower bound: {lb['pressure']/1e6:.3f} MPa")
    print(f"Plastic moment: {mp:.1f} N·m/m")

    # -----------------------------------------------------------------------
    # Figure 1: Elastic deflection field (Navier solution)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    nx, ny = 50, 50
    x = np.linspace(0, A, nx)
    y = np.linspace(0, B, ny)
    xx, yy = np.meshgrid(x, y)
    points = [(xx[i, j], yy[i, j]) for i in range(nx) for j in range(ny)]

    elastic = elastic_patch_plate(A, B, T, E, NU, PATCH, py["pressure"],
                                  terms=61, points=points)
    w = np.array(elastic["deflection"]).reshape(nx, ny) * 1e3  # mm

    im = ax.contourf(xx, yy, w, levels=30, cmap="viridis")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Elastic deflection at first yield (p = {py['pressure']/1e6:.3f} MPa)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Deflection w [mm]")
    fig.tight_layout()
    fig1_path = os.path.join(out_dir, "gap_pp3_elastic_deflection_field.png")
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"Figure 1: {fig1_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Mises stress field at first yield
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sig = np.array(elastic["sigma_eq"]).reshape(nx, ny) / 1e6  # MPa

    im = ax.contourf(xx, yy, sig, levels=30, cmap="plasma")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Von Mises surface stress at first yield (σ_y = {SY/1e6:.0f} MPa)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="σ_eq [MPa]")
    fig.tight_layout()
    fig2_path = os.path.join(out_dir, "gap_pp3_mises_stress_field.png")
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"Figure 2: {fig2_path}")

    # -----------------------------------------------------------------------
    # Figure 3: Bounds bracket (horizontal bar chart)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Strip LB", "First Yield", "Ilyushin Sine UB", "Yield-line UB"]
    values = [lb["pressure"], py["pressure"], sine["pressure"], ub["pressure"]]
    colors_bar = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
    bars = ax.barh(labels, [v / 1e6 for v in values], color=colors_bar, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val / 1e6:.3f} MPa", va="center", fontsize=10)
    ax.set_xlabel("Pressure [MPa]")
    ax.set_title("GAP-PP-3: Analytic bounds bracket\n(DH36, t=20mm, 1m×1m plate, 50%×50% central patch)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig3_path = os.path.join(out_dir, "gap_pp3_bounds_bracket.png")
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    print(f"Figure 3: {fig3_path}")

    # -----------------------------------------------------------------------
    # Figure 4: Load-deflection schematic (elastic + plastic bounds)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Elastic line (linear up to first yield)
    w_elastic = np.linspace(0, py["centre_deflection"] * 1e3 * 1.5, 100)
    p_elastic = w_elastic / (py["centre_deflection"] * 1e3) * py["pressure"] / 1e6
    ax.plot(w_elastic, p_elastic, "b-", linewidth=2, label="Elastic (Navier)")

    # First yield point
    ax.plot(py["centre_deflection"] * 1e3, py["pressure"] / 1e6, "ro", markersize=8,
            label=f"First yield ({py['pressure']/1e6:.2f} MPa)")

    # Plastic bounds as horizontal lines
    ax.axhline(y=lb["pressure"] / 1e6, color="green", linestyle="--", alpha=0.7,
               label=f"Strip LB ({lb['pressure']/1e6:.2f} MPa)")
    ax.axhline(y=sine["pressure"] / 1e6, color="blue", linestyle="--", alpha=0.7,
               label=f"Ilyushin Sine UB ({sine['pressure']/1e6:.2f} MPa)")
    ax.axhline(y=ub["pressure"] / 1e6, color="red", linestyle="--", alpha=0.7,
               label=f"Yield-line UB ({ub['pressure']/1e6:.2f} MPa)")

    # Schematic plastic path (illustrative)
    w_plastic = np.linspace(py["centre_deflection"] * 1e3, py["centre_deflection"] * 1e3 * 4, 50)
    p_plastic = np.linspace(py["pressure"] / 1e6, ub["pressure"] / 1e6 * 0.95, 50)
    ax.plot(w_plastic, p_plastic, "m--", linewidth=1.5, alpha=0.7,
            label="Schematic elastoplastic path")

    ax.set_xlabel("Centre deflection w [mm]")
    ax.set_ylabel("Patch pressure p [MPa]")
    ax.set_title("GAP-PP-3: Schematic load-deflection response\nwith analytic bounds")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig4_path = os.path.join(out_dir, "gap_pp3_schematic_load_deflection.png")
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    print(f"Figure 4: {fig4_path}")

    # -----------------------------------------------------------------------
    # Figure 5: Temperature-dependent flow stress for Arctic steels
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    grades = ["DH36", "EH36", "FH36"]
    colors_grade = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    temps = np.linspace(-70, 30, 100)

    for grade, color in zip(grades, colors_grade):
        from tensorfem.plate_patch_plastic_limit import low_temperature_flow_stress
        stresses = []
        ductile_flags = []
        for t in temps:
            try:
                fs = low_temperature_flow_stress(grade, t)
                stresses.append(fs.yield_stress / 1e6)
                ductile_flags.append(fs.ductile)
            except (KeyError, ValueError):
                stresses.append(np.nan)
                ductile_flags.append(False)

        # Plot ductile region
        ductile_temps = [t for t, d in zip(temps, ductile_flags) if d]
        ductile_stresses = [s for s, d in zip(stresses, ductile_flags) if d]
        brittle_temps = [t for t, d in zip(temps, ductile_flags) if not d]
        brittle_stresses = [s for s, d in zip(stresses, ductile_flags) if not d]

        if ductile_temps:
            ax.plot(ductile_temps, ductile_stresses, "-", color=color, linewidth=2, label=f"{grade} (ductile)")
        if brittle_temps:
            ax.plot(brittle_temps, brittle_stresses, "--", color=color, linewidth=1, alpha=0.5,
                    label=f"{grade} (brittle)")

    ax.axvline(x=-65, color="gray", linestyle=":", alpha=0.5, label="DH36/EH36 DBTT")
    ax.axvline(x=-80, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Flow stress σ_y [MPa]")
    ax.set_title("GAP-PP-3: Low-temperature flow stress for Arctic hull steels")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig5_path = os.path.join(out_dir, "gap_pp3_flow_stress_vs_temperature.png")
    fig.savefig(fig5_path, dpi=150)
    plt.close(fig)
    print(f"Figure 5: {fig5_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GAP-PP-3-ICE-STRUCT-PLASTIC-SET — Analytic Verification Summary")
    print("=" * 60)
    print(f"Geometry: {A}m × {B}m × {T*1000:.0f}mm, patch {PATCH[0]*100:.0f}% × {PATCH[1]*100:.0f}%")
    print(f"Material: DH36, σ_y = {SY/1e6:.0f} MPa @ 20°C, E = {E/1e9:.0f} GPa")
    print(f"Plastic moment: m_p = {mp:.1f} N·m/m")
    print(f"\nBounds bracket:")
    print(f"  Strip lower bound:      {lb['pressure']/1e6:.3f} MPa")
    print(f"  First yield (Navier):   {py['pressure']/1e6:.3f} MPa")
    print(f"  Ilyushin sine UB:       {sine['pressure']/1e6:.3f} MPa")
    print(f"  Yield-line UB:          {ub['pressure']/1e6:.3f} MPa")
    print(f"\nBracket width: UB/LB = {ub['pressure']/lb['pressure']:.2f}")
    print(f"First yield / LB = {py['pressure']/lb['pressure']:.2f}")
    print(f"UB / First yield = {ub['pressure']/py['pressure']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
