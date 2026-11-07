"""Generate plots for hull-girder strength benchmark."""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.hull_girder_strength import (
    HullSection,
    csr_check,
    design_wave_bending_moment,
    run_kvlcc2_benchmark,
    still_water_bending_moment,
)


def generate_plots():
    os.makedirs("artifacts", exist_ok=True)

    # --- 1. Bending moment distribution along ship length ---
    L = 320.0
    x = np.linspace(-L / 2, L / 2, 100)
    # Simplified wave bending moment distribution (parabolic)
    M_w_x = design_wave_bending_moment(L=L, B=58, Cb=0.708, h_w=13.5) * (1 - (2 * x / L) ** 2)
    M_sw_x = np.zeros_like(x)  # symmetric loading
    M_total_x = M_sw_x + M_w_x

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, M_total_x / 1e9, label="Total Bending Moment (MNm)")
    ax1.plot(x, M_w_x / 1e9, label="Wave Bending Moment (MNm)")
    ax1.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("Position from Midships (m)")
    ax1.set_ylabel("Bending Moment (MNm)")
    ax1.set_title("KVLCC2 Longitudinal Bending Moment Distribution")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig("artifacts/bending_moment_distribution.png", dpi=150)
    plt.close(fig1)

    # --- 2. Stress distribution across cross-section ---
    report = run_kvlcc2_benchmark()
    sec = HullSection()
    sec.add_component("deck_plate", 14445.0, 14.0, 210e9, 235e6)
    sec.add_component("deck_long", 11235.0, 13.5, 210e9, 235e6)
    sec.add_component("side_shell", 9630.0, 7.0, 210e9, 235e6)
    sec.add_component("bottom_plate", 17655.0, -14.0, 210e9, 235e6)
    sec.add_component("bottom_long", 12840.0, -13.5, 210e9, 235e6)
    sec.add_component("inner_bottom", 8025.0, -7.0, 210e9, 235e6)

    y_na = sec.neutral_axis()
    y_vals = np.linspace(-14.5, 14.5, 200)
    sigma_hog = np.array([abs(report['moments']['M_hog']) / (report['section']['I'] / (y - y_na)) for y in y_vals])
    sigma_sag = np.array([abs(report['moments']['M_sag']) / (report['section']['I'] / (y - y_na)) for y in y_vals])
    sigma_allow = report['hogging']['sigma_allow']

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(sigma_hog / 1e6, y_vals, label="Hogging Stress (MPa)")
    ax2.plot(sigma_sag / 1e6, y_vals, label="Sagging Stress (MPa)")
    ax2.axvline(sigma_allow / 1e6, color="r", linestyle="--", label=f"Allowable Stress ({sigma_allow/1e6:.1f} MPa)")
    ax2.set_xlabel("Bending Stress (MPa)")
    ax2.set_ylabel("y-coordinate (m)")
    ax2.set_title("KVLCC2 Cross-Section Bending Stress Distribution")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("artifacts/stress_distribution.png", dpi=150)
    plt.close(fig2)

    print("Plots saved to artifacts/")


if __name__ == "__main__":
    generate_plots()
