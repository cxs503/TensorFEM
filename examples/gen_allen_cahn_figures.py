"""Generate figures for the Allen-Cahn D3Q15/D3Q19 validation report."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.lbm_allen_cahn_3d import (
    run_diffusion_decay,
    run_shrinking_sphere,
    compare_d3q15_d3q19,
    effective_radius,
)

ASSET_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets" / "lbm_allen_cahn_d3q15_d3q19_xval"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def fig_diffusion_decay():
    """Figure 1: Diffusion decay — amplitude vs time for D3Q15 and D3Q19."""
    r15 = run_diffusion_decay("D3Q15", n=32, tau=1.0, amplitude=0.1, mode=1,
                               steps=200, probe_every=5)
    r19 = run_diffusion_decay("D3Q19", n=32, tau=1.0, amplitude=0.1, mode=1,
                               steps=200, probe_every=5)

    D = r15["D_theory"]
    k = 2 * np.pi / 32
    t_ana = np.linspace(0, 200, 200)
    A_ana = r15["amplitudes"][0] * np.exp(-D * k**2 * t_ana)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(r15["times"], r15["amplitudes"], "o-", ms=3, label="D3Q15 LBM", color="C0")
    ax.semilogy(r19["times"], r19["amplitudes"], "s-", ms=3, label="D3Q19 LBM", color="C1")
    ax.semilogy(t_ana, A_ana, "--", label=f"Analytical (D={D:.4f})", color="k", lw=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    ax.set_title("Diffusion Decay: Cosine Mode (τ=1.0, n=32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "fig1_diffusion_decay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_diffusion_decay.png")


def fig_shrinking_sphere():
    """Figure 2: Shrinking sphere — R² vs time for D3Q15 and D3Q19."""
    r15 = run_shrinking_sphere("D3Q15", n=40, tau=1.0, epsilon=3.0,
                                radius0=12.0, steps=150, report_every=10)
    r19 = run_shrinking_sphere("D3Q19", n=40, tau=1.0, epsilon=3.0,
                                radius0=12.0, steps=150, report_every=10)

    M = r15["M"]
    R0_eff_15 = r15["radii"][0]
    R0_eff_19 = r19["radii"][0]

    t_ana = np.linspace(0, 150, 200)
    R2_ana_15 = R0_eff_15**2 - 4*M*t_ana
    R2_ana_19 = R0_eff_19**2 - 4*M*t_ana

    fig, ax = plt.subplots(figsize=(7, 4.5))
    R2_15 = [r**2 for r in r15["radii"]]
    R2_19 = [r**2 for r in r19["radii"]]
    ax.plot(r15["times"], R2_15, "o-", ms=3, label="D3Q15 LBM", color="C0")
    ax.plot(r19["times"], R2_19, "s-", ms=3, label="D3Q19 LBM", color="C1")
    ax.plot(t_ana, R2_ana_15, "--", label="Analytical (dR²/dt = −4M)", color="k", lw=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("R² (effective)")
    ax.set_title(f"Shrinking Sphere: R²(t) (τ=1.0, ε=3.0, n=40, M={M:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "fig2_shrinking_sphere.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_shrinking_sphere.png")


def fig_d3q15_vs_d3q19():
    """Figure 3: D3Q15 vs D3Q19 — dR²/dt error at different resolutions."""
    ns = [24, 28, 32, 36, 40]
    errs_15 = []
    errs_19 = []
    for n in ns:
        r0 = min(n * 0.3, n / 2.0 - 6.0)
        if r0 < 5:
            continue
        r15 = run_shrinking_sphere("D3Q15", n=n, tau=1.0, epsilon=3.0,
                                    radius0=r0, steps=150, report_every=30)
        r19 = run_shrinking_sphere("D3Q19", n=n, tau=1.0, epsilon=3.0,
                                    radius0=r0, steps=150, report_every=30)
        errs_15.append(r15["dR2_dt_rel_err"] * 100)
        errs_19.append(r19["dR2_dt_rel_err"] * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns[:len(errs_15)], errs_15, "o-", ms=5, label="D3Q15 (rank-3 isotropy)", color="C0")
    ax.plot(ns[:len(errs_19)], errs_19, "s-", ms=5, label="D3Q19 (rank-4 isotropy)", color="C1")
    ax.axhline(15, ls=":", color="r", lw=1, label="15% acceptance")
    ax.set_xlabel("Domain size n (R₀ ≈ 0.3n)")
    ax.set_ylabel("dR²/dt relative error (%)")
    ax.set_title("D3Q15 vs D3Q19: Shrinking Sphere Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "fig3_d3q15_vs_d3q19.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_d3q15_vs_d3q19.png")


if __name__ == "__main__":
    print("Generating Allen-Cahn D3Q15/D3Q19 validation figures...")
    fig_diffusion_decay()
    fig_shrinking_sphere()
    fig_d3q15_vs_d3q19()
    print("Done.")
