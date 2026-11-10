"""Generate publication-quality figures for the fatigue crack growth verification.

Produces:
  fig1_sif_anchors.png        - SIF geometry factors vs analytical references
  fig2_paris_integration.png  - Paris law closed-form vs RK4 numerical integration
  fig3_residual_life.png      - Residual life monotonicity & Paris exponent sweep
  fig4_paris_vs_sn.png        - Paris life vs IIW S-N life cross-adjudication
  fig5_gci_convergence.png    - GCI discretisation convergence study
  fig6_detail_comparison.png  - Three typical ship details: K_I distribution & N_f
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from tensorfem.fatigue_crack_propagation import (
    ParisLaw,
    k_edge_crack_finite_width,
    k_hole_edge_crack,
    k_infinite_plate,
    k_semi_elliptical_surface_crack,
    paris_integrate_closed_form,
)
from tensorfem.fatigue.crack_growth_geometry import (
    SNCurveIIW,
    centre_crack_finite_width_delta_k,
    compare_paris_vs_iiw,
    convergence_study,
    critical_crack_length,
    edge_crack_finite_width_delta_k,
    hole_edge_crack_delta_k,
    integrate_paris_ode,
    surface_crack_delta_k,
    weld_toe_surface_crack_life,
    hole_edge_crack_life,
    fillet_weld_toe_life,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fatigue_crack_growth")
os.makedirs(OUT_DIR, exist_ok=True)

SIGMA = 100.0e6
K_IC = 60.0e6
STEEL = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def fig1_sif_anchors():
    """SIF geometry factors for the four canonical configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a1) Finite-width centre crack: Y vs 2a/W
    ax = axes[0, 0]
    ratios = np.linspace(0.01, 0.49, 100)
    W = 0.2
    Y_model = [1.0 / math.sqrt(math.cos(math.pi * r * W / W)) for r in ratios]
    # Isida table points
    isida = {0.2: 1.0254, 0.4: 1.1118}
    ax.plot(ratios * 2, Y_model, "b-", lw=2, label="Secant (Feddersen)")
    ax.plot(*zip(*sorted(isida.items())), "ro", ms=8, label="Isida table")
    ax.set_xlabel("2a / W")
    ax.set_ylabel("Geometry factor Y")
    ax.set_title("(a1) Finite-width centre crack")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (a2) Edge crack: Y vs a/W
    ax = axes[0, 1]
    a_ratios = np.linspace(0.001, 0.59, 100)
    Y_edge = []
    for r in a_ratios:
        Y_edge.append(k_edge_crack_finite_width(SIGMA, r * W, W).geometry_factor)
    ax.plot(a_ratios, Y_edge, "g-", lw=2, label="Tada/Paris/Irwin polynomial")
    ax.axhline(1.12, color="r", ls="--", label="Semi-infinite limit (1.12)")
    ax.set_xlabel("a / W")
    ax.set_ylabel("Geometry factor Y")
    ax.set_title("(a2) Edge crack finite-width plate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (a3) Semi-elliptical surface crack: F vs a/t at deepest point
    ax = axes[1, 0]
    a_t_vals = np.linspace(0.01, 0.79, 80)
    t_ref = 0.02
    F_deep = []
    for at in a_t_vals:
        a = at * t_ref
        sif = k_semi_elliptical_surface_crack(SIGMA, a, 2 * a, t_ref, front_location="deepest")
        F_deep.append(sif.k_i_pa_sqrt_m / (SIGMA * math.sqrt(math.pi * a)))
    ax.plot(a_t_vals, F_deep, "m-", lw=2, label="Newman-Raju (deepest, a/c=0.5)")
    ax.axhline(0.6637, color="r", ls="--", label="Semicircular limit (0.6637)")
    ax.set_xlabel("a / t")
    ax.set_ylabel("F = K_I / (σ√(πa))")
    ax.set_title("(a3) Semi-elliptical surface crack")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (a4) Hole-edge crack: F vs a/R
    ax = axes[1, 1]
    R = 5e-3
    a_R_vals = np.logspace(-3, 1, 80)
    F_hole = []
    for ar in a_R_vals:
        a = ar * R
        F_hole.append(k_hole_edge_crack(SIGMA, a, R).geometry_factor)
    ax.semilogx(a_R_vals, F_hole, "c-", lw=2, label="Tada correlation")
    ax.axhline(1.12 * 3.0, color="r", ls="--", label="Kirsch limit (1.12 K_t = 3.36)")
    ax.set_xlabel("a / R")
    ax.set_ylabel("Geometry factor F")
    ax.set_title("(a4) Hole-edge crack")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("SIF Geometry Factors — Analytical Anchors (Criterion a)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_sif_anchors.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig2_paris_integration():
    """Paris law: closed-form vs RK4 numerical integration."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: crack growth history a(N) for different m
    ax = axes[0]
    a0, ac, dsigma = 1e-4, 5e-3, 80e6
    dk_func = centre_crack_finite_width_delta_k(dsigma, 1e3)
    for m, color in [(2.0, "blue"), (3.0, "red"), (4.0, "green")]:
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
        n_f = integrate_paris_ode(dk_func, law, a0, ac, n_steps=500)
        # sample the history
        n_steps = 500
        t_end = math.log(ac / a0)
        h = t_end / n_steps
        a_hist, N_hist = [a0], [0.0]
        a = a0
        N = 0.0
        for _ in range(n_steps):
            dk = dk_func(a)
            da_dN = law.da_dN(dk)
            if da_dN <= 0:
                break
            da = 0.005 * a
            dN = da / da_dN
            a += da
            N += dN
            a_hist.append(a)
            N_hist.append(N)
        ax.semilogy(N_hist, [x * 1e3 for x in a_hist], color=color, lw=1.5,
                    label=f"m = {m}, N_f = {n_f:.2e}")
    ax.set_xlabel("Cycles N")
    ax.set_ylabel("Crack depth a (mm)")
    ax.set_title("Paris Law Crack Growth History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: closed-form vs numerical error vs m
    ax = axes[1]
    m_vals = np.linspace(2.0, 5.0, 30)
    errors = []
    for m in m_vals:
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
        n_closed = paris_integrate_closed_form(a0, ac, dsigma, law)
        n_rk4 = integrate_paris_ode(dk_func, law, a0, ac, n_steps=200)
        errors.append(abs(n_rk4 - n_closed) / n_closed * 100)
    ax.plot(m_vals, errors, "ko-", ms=4, lw=1.5)
    ax.axhline(1.0, color="r", ls="--", label="1% tolerance")
    ax.set_xlabel("Paris exponent m")
    ax.set_ylabel("|N_RK4 − N_closed| / N_closed (%)")
    ax.set_title("Closed-form vs RK4 Integration Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Paris Law Integration Verification (Criterion b)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_paris_integration.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig3_residual_life():
    """Residual life monotonicity and Paris exponent sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: N_f vs a_0 (monotonicity)
    ax = axes[0]
    law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
    a0_vals = np.logspace(-5, -2.5, 30)
    dk_func = hole_edge_crack_delta_k(100e6, 0.01, plate_width_m=0.15)
    n_f_vals = []
    for a0 in a0_vals:
        try:
            ac = critical_crack_length(dk_func, K_IC, a0 * 1.01, a0 * 100)
            n_f = integrate_paris_ode(dk_func, law, a0, ac, n_steps=200)
            n_f_vals.append(n_f)
        except Exception:
            n_f_vals.append(float("nan"))
    ax.loglog(a0_vals * 1e3, n_f_vals, "b-o", ms=4, lw=1.5)
    ax.set_xlabel("Initial crack size a₀ (mm)")
    ax.set_ylabel("Cycles to failure N_f")
    ax.set_title("(d) Residual Life Monotonicity")
    ax.grid(True, alpha=0.3, which="both")
    # annotate monotonicity
    valid = [(a, n) for a, n in zip(a0_vals, n_f_vals) if math.isfinite(n)]
    if len(valid) > 1:
        mono = all(valid[i][1] > valid[i + 1][1] for i in range(len(valid) - 1))
        ax.text(0.05, 0.95, f"Monotone decreasing: {mono}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right: N_f vs m (exponent sweep)
    ax = axes[1]
    m_vals = [2.0, 2.5, 3.0, 3.5, 4.0]
    n_f_sweep = []
    dk_fw = centre_crack_finite_width_delta_k(80e6, 1e3)
    for m in m_vals:
        law_m = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
        n_f = integrate_paris_ode(dk_fw, law_m, 1e-4, 5e-3, n_steps=200)
        n_f_sweep.append(n_f)
    ax.semilogy(m_vals, n_f_sweep, "rs-", ms=8, lw=2)
    ax.set_xlabel("Paris exponent m")
    ax.set_ylabel("Cycles to failure N_f")
    ax.set_title("(g) Paris Exponent Sweep")
    ax.grid(True, alpha=0.3)
    mono_m = all(n_f_sweep[i] > n_f_sweep[i + 1] for i in range(len(n_f_sweep) - 1))
    ax.text(0.05, 0.95, f"N_f monotonically decreasing: {mono_m}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Residual Life & Parameter Sensitivity (Criteria d, g)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_residual_life.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig4_paris_vs_sn():
    """Paris life vs IIW S-N life cross-adjudication."""
    fig, ax = plt.subplots(figsize=(8, 5))

    law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
    sn = SNCurveIIW(detail_category_pa=71e6)

    # Vary stress range and compare Paris vs S-N
    sigma_vals = np.linspace(40e6, 200e6, 30)
    n_paris_list, n_sn_list = [], []
    for dsigma in sigma_vals:
        try:
            life = weld_toe_surface_crack_life(
                thickness_m=0.020, crack_depth_m=2e-4, crack_half_length_m=1e-3,
                sigma_range_pa=dsigma, law=law,
                fracture_toughness_pa_sqrt_m=K_IC, plate_half_width_m=1.0,
            )
            n_paris_list.append(life.n_f)
        except Exception:
            n_paris_list.append(float("nan"))
        scf = 1.2
        n_sn = (sn.detail_category_pa / (dsigma * scf)) ** sn.slope_1
        n_sn_list.append(n_sn)

    ax.loglog(sigma_vals / 1e6, n_paris_list, "b-o", ms=5, lw=1.5, label="Paris law (LEFM)")
    ax.loglog(sigma_vals / 1e6, n_sn_list, "r--s", ms=5, lw=1.5, label="IIW S-N (FAT 71)")
    ax.set_xlabel("Stress range Δσ (MPa)")
    ax.set_ylabel("Cycles to failure N_f")
    ax.set_title("(e) Paris Law vs IIW S-N Curve — Weld Toe Detail")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Annotate the ratio at the reference point
    ref_sigma = 100e6
    ref_life = weld_toe_surface_crack_life(
        thickness_m=0.020, crack_depth_m=2e-4, crack_half_length_m=1e-3,
        sigma_range_pa=ref_sigma, law=law,
        fracture_toughness_pa_sqrt_m=K_IC, plate_half_width_m=1.0,
    )
    agreement = compare_paris_vs_iiw(ref_life.n_f, sn, ref_sigma * 1.2)
    ax.annotate(
        f"Δσ=100 MPa\nParis: {ref_life.n_f:.1e}\nS-N: {agreement.n_f_sn:.1e}\n"
        f"log₁₀ ratio: {agreement.log10_ratio:.2f}",
        xy=(100, ref_life.n_f), xytext=(130, ref_life.n_f * 5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_paris_vs_sn.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig5_gci_convergence():
    """GCI discretisation convergence study."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
    dk_func = centre_crack_finite_width_delta_k(80e6, 0.2)

    # Left: N_f vs n_steps showing convergence
    ax = axes[0]
    n_steps_list = [10, 20, 50, 100, 200, 400, 800]
    n_f_list = [integrate_paris_ode(dk_func, law, 1e-4, 5e-3, n_steps=n) for n in n_steps_list]
    ax.semilogx(n_steps_list, n_f_list, "bo-", ms=6, lw=1.5)
    ax.axhline(n_f_list[-1], color="r", ls="--", alpha=0.5, label="Finest grid")
    ax.set_xlabel("Number of integration steps")
    ax.set_ylabel("N_f (cycles)")
    ax.set_title("(f) Convergence of Paris Integration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: GCI vs refinement level
    ax = axes[1]
    study = convergence_study(dk_func, law, 1e-4, 5e-3, base_steps=50)
    labels = ["Coarse", "Medium", "Fine"]
    colors = ["red", "orange", "green"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(i, study.values[i] / 1e9, color=color, alpha=0.7,
               label=f"{labels[i]}: N={study.step_counts[i]}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("N_f (×10⁹ cycles)")
    ax.set_title(f"GCI = {study.gci_fine * 100:.4f}% (≤5% required)\n"
                 f"Observed order p = {study.observed_order:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Grid Convergence Index Study (Criterion f)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_gci_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig6_detail_comparison():
    """Three typical ship structural details: K_I distribution and N_f."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
    dsigma = 100e6

    # Left: Delta K(a) for three details
    ax = axes[0]
    a_vals = np.logspace(-4, -2, 50)

    # Centre crack (finite width)
    dk_centre = centre_crack_finite_width_delta_k(dsigma, 0.2)
    dk_centre_vals = [dk_centre(a) / 1e6 for a in a_vals]

    # Hole edge crack
    dk_hole = hole_edge_crack_delta_k(dsigma, 0.01, plate_width_m=0.15)
    dk_hole_vals = [dk_hole(a) / 1e6 for a in a_vals]

    # Surface crack (weld toe)
    dk_surf = surface_crack_delta_k(dsigma, 0.020, 2e-4, 1e-3, plate_half_width_m=1.0)
    dk_surf_vals = []
    for a in a_vals:
        try:
            dk_surf_vals.append(dk_surf(a) / 1e6)
        except Exception:
            dk_surf_vals.append(float("nan"))

    ax.loglog(a_vals * 1e3, dk_centre_vals, "b-", lw=2, label="Centre crack (W=200mm)")
    ax.loglog(a_vals * 1e3, dk_hole_vals, "g-", lw=2, label="Hole edge (R=10mm)")
    ax.loglog(a_vals * 1e3, dk_surf_vals, "r-", lw=2, label="Weld toe surface crack")
    ax.set_xlabel("Crack size a (mm)")
    ax.set_ylabel("ΔK (MPa√m)")
    ax.set_title("Stress Intensity Factor Range vs Crack Size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Right: N_f for three details at different stress levels
    ax = axes[1]
    sigma_levels = [60, 80, 100, 120, 150]
    n_centre_list, n_hole_list, n_weld_list = [], [], []
    for ds in sigma_levels:
        ds_pa = ds * 1e6
        # Centre crack
        dk_c = centre_crack_finite_width_delta_k(ds_pa, 0.2)
        try:
            ac_c = critical_crack_length(dk_c, K_IC, 1e-4, 0.05)
            n_c = integrate_paris_ode(dk_c, law, 1e-4, ac_c, n_steps=200)
        except Exception:
            n_c = float("nan")
        n_centre_list.append(n_c)

        # Hole edge
        dk_h = hole_edge_crack_delta_k(ds_pa, 0.01, plate_width_m=0.15)
        try:
            ac_h = critical_crack_length(dk_h, K_IC, 1e-4, 0.05)
            n_h = integrate_paris_ode(dk_h, law, 1e-4, ac_h, n_steps=200)
        except Exception:
            n_h = float("nan")
        n_hole_list.append(n_h)

        # Weld toe surface crack
        try:
            life_w = weld_toe_surface_crack_life(
                thickness_m=0.020, crack_depth_m=2e-4, crack_half_length_m=1e-3,
                sigma_range_pa=ds_pa, law=law,
                fracture_toughness_pa_sqrt_m=K_IC, plate_half_width_m=1.0,
            )
            n_weld_list.append(life_w.n_f)
        except Exception:
            n_weld_list.append(float("nan"))

    x = np.arange(len(sigma_levels))
    w = 0.25
    ax.bar(x - w, [n / 1e3 for n in n_centre_list], w, label="Centre crack", color="steelblue")
    ax.bar(x, [n / 1e3 for n in n_hole_list], w, label="Hole edge", color="seagreen")
    ax.bar(x + w, [n / 1e3 for n in n_weld_list], w, label="Weld toe", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} MPa" for s in sigma_levels])
    ax.set_xlabel("Stress range Δσ")
    ax.set_ylabel("N_f (×10³ cycles)")
    ax.set_title("Fatigue Life Comparison — Three Ship Details")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    fig.suptitle("Typical Ship Structural Details (Criterion e + detail verification)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_detail_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all():
    """Generate all figures and return paths."""
    paths = {}
    for func in [fig1_sif_anchors, fig2_paris_integration, fig3_residual_life,
                 fig4_paris_vs_sn, fig5_gci_convergence, fig6_detail_comparison]:
        name = func.__name__
        paths[name] = func()
        print(f"  Generated: {paths[name]}")
    return paths


if __name__ == "__main__":
    print("Generating fatigue crack growth verification figures...")
    paths = generate_all()
    print(f"\nAll {len(paths)} figures generated in {OUT_DIR}")
