"""Generate the FE-vs-closed-form conservatism figure for the ice-grillage
ultimate cross-check (GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE).

Produces:
  * fig1: FE collapse load factor vs closed-form first-yield multiplier
          (three combined cases + uniform yield-line anchor), with the
          conservatism margin annotated.
  * fig2: arc-length load path (lam vs centre deflection) for the design-ice
          case, showing the limit-point bracketing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.grillage_ice_fe_crosscheck import (
    run_fe_crosscheck,
    fe_lateral_collapse,
    yield_line_uniform_pressure,
)
from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    case_study_geometry,
)

OUT = Path("docs/assets/grillage_ice_fe_crosscheck")
OUT.mkdir(parents=True, exist_ok=True)


def fig1_conservatism():
    payload = run_fe_crosscheck(nx=14, ny=4, n_steps=100, dlam0=0.03)
    cases = payload["cases"]
    names = [c["scenario"] for c in cases]
    lam = [c["lam_peak_fe"] for c in cases]
    closed = [c["closed_multiplier"] for c in cases]
    ratio = [c["conservatism_ratio"] for c in cases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(names))
    w = 0.35
    ax1.bar(x - w / 2, lam, w, label="FE collapse (arc-length)", color="#1f77b4")
    ax1.bar(x + w / 2, closed, w, label="Closed-form first-yield", color="#ff7f0e")
    for i, r in enumerate(ratio):
        ax1.text(x[i], max(lam[i], closed[i]) * 1.02, f"{r:.2f}x",
                 ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.split(" (")[0] for n in names], rotation=20, ha="right")
    ax1.set_ylabel("Load multiplier on combined lateral reference")
    ax1.set_title("FE collapse vs closed-form first-yield\n(conservatism margin)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(x, [r - 1 for r in ratio], color="#2ca02c")
    ax2.axhline(0, color="k", lw=0.8)
    for i, r in enumerate(ratio):
        ax2.text(x[i], (r - 1) * 0.5 + 0.02, f"+{(r-1)*100:.0f}%",
                 ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split(" (")[0] for n in names], rotation=20, ha="right")
    ax2.set_ylabel("Conservatism margin (FE above closed-form)")
    ax2.set_title("Closed-form is a conservative lower bound\n(FE collapse >= first-yield)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_fe_vs_closedform_conservatism.png", dpi=150)
    plt.close(fig)


def fig2_arc_path():
    geom = case_study_geometry()
    case = CombinedLoadCase(
        axial_stress=0.30 * geom.yield_stress, ice_pressure=2.0e6,
        patch_length=0.70, patch_width=0.35, hydrostatic_pressure=1.0e5)
    fe = fe_lateral_collapse(geom, case, nx=14, ny=4, n_steps=100, dlam0=0.03)
    lam = np.array(fe["path_lam"])
    w = np.array(fe["path_w"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(w * 1e3, lam, "-o", ms=3, color="#1f77b4", label="arc-length path")
    ax.axhline(fe["lam_peak"], color="r", ls="--", lw=1.2,
               label=f"peak load factor = {fe['lam_peak']:.3f}")
    ax.set_xlabel("Centre deflection w [mm]")
    ax.set_ylabel("Load factor $\\lambda$ (on combined lateral)")
    ax.set_title("Design-ice case: arc-length collapse path\n(limit-point bracketing)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_arc_length_path.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_conservatism()
    fig2_arc_path()
    print("figures written to", OUT)
