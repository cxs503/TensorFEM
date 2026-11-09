#!/usr/bin/env python3
"""Generate verification figures for FSI flutter analysis (IN98-N10)."""
import math
import os
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensorfem.theodorsen import (
    theodorsen_exact,
    theodorsen_critical_velocity,
    theodorsen_flutter_frequency,
    theodorsen_growth_rate,
    TypicalSectionParams,
    solve_flutter_eigenvalues,
    fsi_time_march,
)

OUTDIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(OUTDIR, exist_ok=True)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Figure 1: Theodorsen function C(k)
# ---------------------------------------------------------------------------
def plot_theodorsen_function():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    k_vals = np.linspace(0.001, 10, 500)
    F_vals = [theodorsen_exact(k).real for k in k_vals]
    G_vals = [theodorsen_exact(k).imag for k in k_vals]

    ax1.plot(k_vals, F_vals, 'b-', linewidth=2, label='F(k) = Re[C(k)]')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='C(∞) = 0.5')
    ax1.set_xlabel('Reduced frequency k = ωb/U')
    ax1.set_ylabel('F(k)')
    ax1.set_title('Theodorsen Function — Real Part')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_vals, G_vals, 'r-', linewidth=2, label='G(k) = Im[C(k)]')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Reduced frequency k = ωb/U')
    ax2.set_ylabel('G(k)')
    ax2.set_title('Theodorsen Function — Imaginary Part')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig1_theodorsen_function.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Critical flutter velocity vs mass ratio
# ---------------------------------------------------------------------------
def plot_u_cr_vs_mu():
    fig, ax = plt.subplots(figsize=(8, 5))
    mu_vals = np.array([5, 10, 20, 50])
    U_cr_vals = [theodorsen_critical_velocity(mu) for mu in mu_vals]
    U_cr_scaling = 2.0 * np.sqrt(mu_vals)

    ax.plot(mu_vals, U_cr_vals, 'bo-', linewidth=2, markersize=8, label='Theodorsen U_cr')
    ax.plot(mu_vals, U_cr_scaling, 'r--', linewidth=1.5, label='2√μ scaling')
    ax.set_xlabel('Mass ratio μ = m/(πρb²)')
    ax.set_ylabel('Critical reduced velocity U_cr/(b·ω_α)')
    ax.set_title('Critical Flutter Velocity vs Mass Ratio\n(Criterion a: error < 5%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add error annotations
    for mu, U_cr in zip(mu_vals, U_cr_vals):
        ref = 2.0 * math.sqrt(mu)
        err = abs(U_cr - ref) / ref * 100
        ax.annotate(f'{err:.1f}%', xy=(mu, U_cr), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, color='green')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig2_u_cr_vs_mu.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Growth rate vs velocity (Dowell Fig. 5.3 comparison)
# ---------------------------------------------------------------------------
def plot_growth_rate():
    fig, ax = plt.subplots(figsize=(8, 5))
    mu = 10
    U_cr = theodorsen_critical_velocity(mu)

    U_vals = np.linspace(0.5 * U_cr, 1.5 * U_cr, 100)
    sigma_vals = [theodorsen_growth_rate(U, mu) for U in U_vals]

    ax.plot(U_vals, sigma_vals, 'b-', linewidth=2, label='Theodorsen growth rate')
    ax.axvline(x=U_cr, color='red', linestyle='--', alpha=0.7, label=f'U_cr = {U_cr:.2f}')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.fill_betweenx([-0.15, 0.15], 0.5 * U_cr, U_cr, alpha=0.1, color='green', label='Stable')
    ax.fill_betweenx([-0.15, 0.15], U_cr, 1.5 * U_cr, alpha=0.1, color='red', label='Unstable')

    ax.set_xlabel('Reduced velocity U/(b·ω_α)')
    ax.set_ylabel('Growth rate σ/ω_α')
    ax.set_title(f'Growth Rate vs Velocity (μ={mu})\n(Criterion c: error < 10% vs Dowell 1995)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig3_growth_rate.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Time histories — subcritical vs supercritical
# ---------------------------------------------------------------------------
def plot_time_histories():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    mu = 10
    params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
    U_cr = theodorsen_critical_velocity(mu)

    # Subcritical
    U_sub = 0.7 * U_cr
    sim_sub = fsi_time_march(params, U_sub, t_max=50.0, dt=0.01)
    ax1.plot(sim_sub.time, sim_sub.alpha, 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (non-dim, /ω_α)')
    ax1.set_ylabel('Pitch angle α (rad)')
    ax1.set_title(f'Subcritical: U = {U_sub:.2f} < U_cr = {U_cr:.2f}\nAmplitude decays (σ < 0)')
    ax1.grid(True, alpha=0.3)

    # Supercritical
    U_super = 1.2 * U_cr
    sim_super = fsi_time_march(params, U_super, t_max=50.0, dt=0.01)
    ax2.plot(sim_super.time, sim_super.alpha, 'r-', linewidth=0.8)
    ax2.set_xlabel('Time (non-dim, /ω_α)')
    ax2.set_ylabel('Pitch angle α (rad)')
    ax2.set_title(f'Supercritical: U = {U_super:.2f} > U_cr = {U_cr:.2f}\nAmplitude grows (σ > 0)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig4_time_histories.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: Mass-ratio scaling (log-log)
# ---------------------------------------------------------------------------
def plot_scaling():
    fig, ax = plt.subplots(figsize=(7, 5))
    mu_vals = np.array([5, 10, 20, 50])
    U_cr_vals = [theodorsen_critical_velocity(mu) for mu in mu_vals]

    ax.loglog(mu_vals, U_cr_vals, 'bo-', linewidth=2, markersize=10, label='Theodorsen U_cr')

    # Reference line with slope 0.5
    mu_ref = np.linspace(3, 60, 100)
    U_ref = 2.0 * np.sqrt(mu_ref)
    ax.loglog(mu_ref, U_ref, 'r--', linewidth=1.5, label='U_cr ∝ √μ (slope = 0.5)')

    ax.set_xlabel('Mass ratio μ')
    ax.set_ylabel('U_cr/(b·ω_α)')
    ax.set_title('Mass-Ratio Scaling Law\n(Criterion e: exponent error < 2%)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig5_scaling.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Generating verification figures...")
    paths = []
    paths.append(('Theodorsen function C(k)', plot_theodorsen_function()))
    paths.append(('U_cr vs mass ratio', plot_u_cr_vs_mu()))
    paths.append(('Growth rate vs velocity', plot_growth_rate()))
    paths.append(('Time histories', plot_time_histories()))
    paths.append(('Mass-ratio scaling', plot_scaling()))

    for name, path in paths:
        print(f"  {name}: {path}")

    print("\nAll figures generated successfully.")
