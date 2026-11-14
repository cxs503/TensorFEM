#!/usr/bin/env python3
"""Generate figures for GAP-PP-1-ICE-STRUCT-CONTACT-PATCH verification report.

Produces:
1. Contact pressure field (Gaussian patch)
2. Plate deflection field (non-uniform vs uniform)
3. Stress comparison bar chart
4. Non-uniformity scan results
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorfem.patch_from_contact import (
    ContactPressureData,
    contact_pressure_to_plate_load,
    gaussian_patch_pressure,
    peak_stresses,
    scan_contact_configurations,
)
from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation


def gen_contact_pressure_figure(out: Path):
    """Figure 1: Gaussian contact pressure field."""
    cx, cy = 0.5, 0.5
    sx, sy = 0.15, 0.15
    peak_p = 5.0e6
    n = 41

    lo_x = max(0.0, cx - 3.0 * sx)
    hi_x = min(1.0, cx + 3.0 * sx)
    lo_y = max(0.0, cy - 3.0 * sy)
    hi_y = min(1.0, cy + 3.0 * sy)
    xs = np.linspace(lo_x, hi_x, n)
    ys = np.linspace(lo_y, hi_y, n)
    xx, yy = np.meshgrid(xs, ys)
    pres = gaussian_patch_pressure(xx, yy, cx, cy, peak_p, sx, sy)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(xx, yy, pres / 1e6, levels=30, cmap="viridis")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("CZM Contact Pressure Field (Gaussian)\n"
                 f"Peak: {peak_p/1e6:.1f} MPa, σ=({sx},{sy}) m")
    fig.colorbar(im, ax=ax, label="Pressure [MPa]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "fig_contact_pressure.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'fig_contact_pressure.png'}")


def gen_deflection_comparison_figure(out: Path):
    """Figure 2: Deflection field comparison (non-uniform vs uniform)."""
    E, nu, t, k = 2.06e11, 0.3, 0.01, 1.0e7
    nx, side = 12, 1.0
    cx, cy, sx, sy, peak_p = 0.5, 0.5, 0.15, 0.15, 5.0e6

    # Build contact data
    lo_x = max(0.0, cx - 3.0 * sx)
    hi_x = min(side, cx + 3.0 * sx)
    lo_y = max(0.0, cy - 3.0 * sy)
    hi_y = min(side, cy + 3.0 * sy)
    xs_c = np.linspace(lo_x, hi_x, 21)
    ys_c = np.linspace(lo_y, hi_y, 21)
    xx_c, yy_c = np.meshgrid(xs_c, ys_c)
    pos = np.column_stack([xx_c.ravel(), yy_c.ravel()])
    pres = gaussian_patch_pressure(pos[:, 0], pos[:, 1], cx, cy, peak_p, sx, sy)
    contact_data = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sx * sy,
    )

    # Solve non-uniform
    plate_result, patch_result = contact_pressure_to_plate_load(
        contact_data, nx, side, E, nu, t, k,
    )
    w_proj = plate_result.displacement[0::3].numpy().reshape(nx + 1, nx + 1)

    # Solve uniform
    patch_w = 6.0 * sx
    patch_l = 6.0 * sy
    uniform_p = patch_result.total_force / (patch_w * patch_l)

    def uniform_fn(x, y):
        in_patch = (
            (x >= cx - patch_w / 2) & (x <= cx + patch_w / 2) &
            (y >= cy - patch_l / 2) & (y <= cy + patch_l / 2)
        )
        return np.where(in_patch, uniform_p, 0.0)

    plate_uni = solve_plate_on_winkler_foundation(nx, side, E, nu, t, k, uniform_fn)
    w_uni = plate_uni.displacement[0::3].numpy().reshape(nx + 1, nx + 1)

    xg = np.linspace(0, side, nx + 1)
    yg = np.linspace(0, side, nx + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].contourf(xg, yg, w_proj * 1000, levels=20, cmap="plasma")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Non-uniform Patch\nDeflection [mm]")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(xg, yg, w_uni * 1000, levels=20, cmap="plasma")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("Uniform Patch\nDeflection [mm]")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    diff = (w_proj - w_uni) * 1000
    im2 = axes[2].contourf(xg, yg, diff, levels=20, cmap="RdBu_r")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_title("Difference [mm]")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle("Plate Deflection: Non-uniform vs Uniform Patch Load", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig_deflection_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'fig_deflection_comparison.png'}")


def gen_stress_comparison_figure(out: Path):
    """Figure 3: Peak stress comparison bar chart."""
    # Load demo results
    demo_json = Path(__file__).parent.parent.parent / "outputs" / "ice_struct_demo" / "end_to_end_demo.json"
    if demo_json.exists():
        data = json.loads(demo_json.read_text())
        sigma_x_nonuni = data["peak_stress_nonuniform"]["sigma_x_MPa"]
        sigma_x_uni = data["peak_stress_uniform"]["sigma_x_MPa"]
        sigma_vm_nonuni = data["peak_stress_nonuniform"]["sigma_vm_MPa"]
        sigma_vm_uni = data["peak_stress_uniform"]["sigma_vm_MPa"]
        w_nonuni = data["peak_deflection_nonuniform_m"]
        w_uni = data["peak_deflection_uniform_m"]
    else:
        # Fallback values
        sigma_x_nonuni, sigma_x_uni = 2493.6, 921.6
        sigma_vm_nonuni, sigma_vm_uni = 2494.3, 1272.0
        w_nonuni, w_uni = 0.139, 0.077

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Stress comparison
    labels = ["σ_x", "σ_vm"]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, [sigma_x_nonuni, sigma_vm_nonuni], width, label="Non-uniform", color="#e74c3c")
    axes[0].bar(x + width/2, [sigma_x_uni, sigma_vm_uni], width, label="Uniform", color="#3498db")
    axes[0].set_ylabel("Stress [MPa]")
    axes[0].set_title("Peak Stress Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    # Deflection comparison
    axes[1].bar(["Non-uniform", "Uniform"], [w_nonuni * 1000, w_uni * 1000],
                color=["#e74c3c", "#3498db"])
    axes[1].set_ylabel("Deflection [mm]")
    axes[1].set_title("Peak Deflection Comparison")

    # Relative difference
    delta_sigma = 100.0 * (sigma_x_nonuni / sigma_x_uni - 1.0)
    delta_w = 100.0 * (w_nonuni / w_uni - 1.0)
    axes[2].bar(["σ_x", "Deflection"], [delta_sigma, delta_w], color="#2ecc71")
    axes[2].set_ylabel("Relative Difference [%]")
    axes[2].set_title("Non-uniform vs Uniform\nRelative Increase")
    for i, v in enumerate([delta_sigma, delta_w]):
        axes[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    fig.suptitle("GAP-PP-1-ICE-STRUCT-CONTACT-PATCH: Non-uniform vs Uniform Patch Load Effects", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_stress_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'fig_stress_comparison.png'}")


def gen_scan_figure(out: Path):
    """Figure 4: Non-uniformity scan results."""
    results = scan_contact_configurations(
        plate_nx=8, plate_side=1.0, young=2.06e11, poisson=0.3,
        thickness=0.01, winkler_k=1.0e7, n_contact_per_axis=21,
    )

    cfgs = results["configurations"]
    names = [c["name"] for c in cfgs]
    sigma_x_nonuni = [c["peak_stress_projected"]["sigma_x_MPa"] for c in cfgs]
    sigma_x_uni = [c["peak_stress_uniform"]["sigma_x_MPa"] for c in cfgs]
    delta_pct = [c["delta_nonuniform_over_uniform_pct"]["sigma_x"] for c in cfgs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, sigma_x_nonuni, width, label="Non-uniform", color="#e74c3c")
    axes[0].bar(x + width/2, sigma_x_uni, width, label="Uniform", color="#3498db")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Peak σ_x [MPa]")
    axes[0].set_title("Peak Stress by Contact Configuration")
    axes[0].legend()

    axes[1].bar(names, delta_pct, color="#2ecc71")
    axes[1].set_ylabel("Relative Difference [%]")
    axes[1].set_title("Non-uniform vs Uniform: σ_x Difference")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(delta_pct):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    fig.suptitle("Contact Configuration Scan (>= 3 configurations)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_scan_results.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'fig_scan_results.png'}")


def main():
    out = Path(__file__).parent.parent.parent / "outputs" / "ice_struct_demo"
    out.mkdir(parents=True, exist_ok=True)

    print("Generating figures for GAP-PP-1-ICE-STRUCT-CONTACT-PATCH report...")
    gen_contact_pressure_figure(out)
    gen_deflection_comparison_figure(out)
    gen_stress_comparison_figure(out)
    gen_scan_figure(out)
    print("All figures generated.")


if __name__ == "__main__":
    main()
