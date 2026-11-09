#!/usr/bin/env python
"""Generate verification figures for p106w1-n10 weld toe fatigue maturity report."""
from __future__ import annotations

import base64
import io
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.fatigue import (
    ENVIRONMENTS,
    WeldDetail,
    fatigue_life,
    hot_spot_stress,
    hot_spot_stress_alternative,
    mesh_convergence,
    negative_control_large_radius,
    negative_control_small_radius,
    solve_weld_detail,
    verify_fatigue_power_law,
)

OUTPUT_DIR = ROOT / "artifacts" / "m7916"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# --------------------------------------------------------------------------- #
# Figure 1: S-N curve with power law verification
# --------------------------------------------------------------------------- #
def fig_sn_curve():
    curve = ENVIRONMENTS["air"].curve
    stresses = np.logspace(1, 3, 200)  # 10-1000 MPa
    lives = [curve.cycles_to_failure(s * 1e6) for s in stresses]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(stresses, lives, "b-", linewidth=2, label="DNV-RP-C203 Curve D (air)")
    ax.axvline(x=curve.knee_stress_range_pa / 1e6, color="r", linestyle="--", alpha=0.7,
               label=f"Knee ({curve.knee_stress_range_pa/1e6:.1f} MPa)")
    ax.axhline(y=curve.knee_cycles, color="r", linestyle="--", alpha=0.7)

    # Mark power law verification points
    for ds, label in [(100, "m1=3 regime"), (20, "m2=5 regime")]:
        n = curve.cycles_to_failure(ds * 1e6)
        n2 = curve.cycles_to_failure(2 * ds * 1e6)
        ax.plot([ds, 2*ds], [n, n2], "o-", markersize=8, label=f"{label}: 2x stress → {n/n2:.0f}x life")

    ax.set_xlabel("Stress Range Δσ (MPa)")
    ax.set_ylabel("Cycles to Failure N")
    ax.set_title("DNV-RP-C203 Curve D: S-N Curve with Power Law Verification")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    path = OUTPUT_DIR / "fig_sn_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 2: Mesh convergence study
# --------------------------------------------------------------------------- #
def fig_mesh_convergence():
    detail = WeldDetail(
        joint_type="t_fillet",
        plate_thickness_m=0.020,
        toe_radius_m=0.001,
        attachment_thickness_m=0.020,
        attachment_height_m=0.024,
        weld_leg_m=0.014,
    )
    study = mesh_convergence(detail, levels=(1, 2, 3))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    levels_data = study["levels"]
    levels = [l["level"] for l in levels_data]
    kt_toe = [l["kt_toe"] for l in levels_data]
    hs_lin = [l["hotspot_linear"] for l in levels_data]
    hs_quad = [l["hotspot_quadratic"] for l in levels_data]
    n_elem = [l["n_elements"] for l in levels_data]

    axes[0].plot(levels, kt_toe, "ro-", markersize=8, label="K_t (toe)")
    axes[0].set_xlabel("Mesh Level")
    axes[0].set_ylabel("Stress Concentration Factor")
    axes[0].set_title("Notch SCF Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(levels, hs_lin, "bs-", markersize=8, label="Linear (0.4t/0.9t)")
    axes[1].plot(levels, hs_quad, "g^-", markersize=8, label="Quadratic (0.4t/0.9t/1.4t)")
    axes[1].axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Mesh Level")
    axes[1].set_ylabel("Hot Spot SCF")
    axes[1].set_title("Hot Spot SCF Convergence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(levels, n_elem, "ko-", markersize=8)
    axes[2].set_xlabel("Mesh Level")
    axes[2].set_ylabel("Number of Elements")
    axes[2].set_title("Mesh Size")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Mesh Convergence Study: T-Joint Weld Toe (t=20mm, r=1mm)")
    plt.tight_layout()

    path = OUTPUT_DIR / "fig_mesh_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 3: Negative controls
# --------------------------------------------------------------------------- #
def fig_negative_controls():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Large radius
    large_r = negative_control_large_radius()
    valid = [r for r in large_r["results"] if r["kt_toe"] is not None]
    radii_large = [r["radius_m"] * 1000 for r in valid]  # mm
    kt_large = [r["kt_toe"] for r in valid]

    axes[0].plot(radii_large, kt_large, "ro-", markersize=8)
    axes[0].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="SCF = 1 (no concentration)")
    axes[0].set_xlabel("Toe Radius r (mm)")
    axes[0].set_ylabel("Notch SCF K_t")
    axes[0].set_title("Negative Control: r → ∞, SCF → 1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Small radius
    small_r = negative_control_small_radius()
    valid_s = [r for r in small_r["results"] if r["kt_toe"] is not None]
    radii_small = [r["radius_m"] * 1000 for r in valid_s]  # mm
    kt_small = [r["kt_toe"] for r in valid_s]

    axes[1].plot(radii_small, kt_small, "bs-", markersize=8)
    axes[1].set_xlabel("Toe Radius r (mm)")
    axes[1].set_ylabel("Notch SCF K_t")
    axes[1].set_title("Negative Control: r → 0, Singularity Trend")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Negative Controls: Toe Radius Effects on Stress Concentration")
    plt.tight_layout()

    path = OUTPUT_DIR / "fig_negative_controls.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 4: Fatigue life power law
# --------------------------------------------------------------------------- #
def fig_power_law():
    fig, ax = plt.subplots(figsize=(8, 5))

    # m1=3 regime
    stresses_m1 = np.array([50, 100, 200, 400])  # MPa
    lives_m1 = [fatigue_life(s * 1e6, environment="air") for s in stresses_m1]

    # m2=5 regime
    stresses_m2 = np.array([10, 20, 40])  # MPa
    lives_m2 = [fatigue_life(s * 1e6, environment="air") for s in stresses_m2]

    ax.loglog(stresses_m1, lives_m1, "ro-", markersize=8, linewidth=2, label="m1=3 regime (high stress)")
    ax.loglog(stresses_m2, lives_m2, "bs-", markersize=8, linewidth=2, label="m2=5 regime (low stress)")

    # Annotate ratios
    ax.annotate("2x stress → 8x life", xy=(150, lives_m1[1] * 0.3), fontsize=10, color="red")
    ax.annotate("2x stress → 32x life", xy=(20, lives_m2[1] * 0.3), fontsize=10, color="blue")

    ax.set_xlabel("Stress Range Δσ (MPa)")
    ax.set_ylabel("Fatigue Life N (cycles)")
    ax.set_title("Fatigue Life Power Law Verification: N ∝ Δσ^{-m}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    path = OUTPUT_DIR / "fig_power_law.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating verification figures...")
    for fn in (fig_sn_curve, fig_mesh_convergence, fig_negative_controls, fig_power_law):
        path = fn()
        print(f"  Generated: {path}")
    print("Done.")
