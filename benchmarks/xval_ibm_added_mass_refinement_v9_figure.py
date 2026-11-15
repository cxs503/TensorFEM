#!/usr/bin/env python3
"""v9 figure: implicit-IBM added-mass coefficient vs lattice refinement and
body size, with the thin-plate reference (C=1) and the coupled-frequency
implication.

Two panels:
  (a) refinement axis -- C vs dx at fixed c_phys=4, box=64, with the linear
      dx->0 extrapolation (dashed) and the thin-plate value (dotted);
  (b) body-size axis -- C vs 1/c at fixed dx=1, with the 1/c->0 extrapolation;
  plus the coupled-frequency error implied by each C at m_a/m_s = 0.6
  (f/f_vac = sqrt(1/(1 + 0.6 C))), which is the quantity the anchor's
  implicit-IBM-Aitken relaxation actually produces.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "xval_ibm_added_mass_refinement_v8_artifacts"
OUT = HERE / "xval_ibm_added_mass_refinement_v9_artifacts"
OUT.mkdir(exist_ok=True)

MASS_RATIO = 0.6   # m_a/m_s of the convergence-study convention


def load(tag):
    return json.loads((ART / f"{tag}.json").read_text())


def f_ratio(C):
    """Coupled frequency ratio f/f_vac = sqrt(m_s/(m_s + m_a)) with m_a/m_s=0.6*C."""
    return math.sqrt(1.0 / (1.0 + MASS_RATIO * C))


def main():
    ref = [load(f"clean_refine_{r}") for r in (1, 2, 4)]
    dx = np.array([r["dx"] for r in ref])
    Cm = np.array([r["C_momentum"] for r in ref])
    Cf = np.array([r["C_force"] for r in ref])
    cs = [load(f"clean_c_{int(c)}") for c in (4, 8, 16)]
    c = np.array([r["c_phys"] for r in cs])
    Cmc = np.array([r["C_momentum"] for r in cs])

    # linear fits
    coef_m, *_ = np.linalg.lstsq(np.vstack([np.ones_like(dx), dx]).T, Cm,
                                 rcond=None)
    a_m, b_m = coef_m
    coef_c, *_ = np.linalg.lstsq(
        np.vstack([np.ones_like(1.0 / c), 1.0 / c]).T, Cmc, rcond=None)
    a_c, b_c = coef_c

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) refinement axis
    ax = axes[0]
    ax.plot(dx, Cm, "o-", color="#1f77b4", label="C (momentum)")
    ax.plot(dx, Cf, "s--", color="#ff7f0e", label="C (force)")
    dxs = np.linspace(0, 1.05, 50)
    ax.plot(dxs, a_m + b_m * dxs, "--", color="#1f77b4", alpha=0.5,
            label=f"lin fit  C_inf={a_m:.3f}")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, label="thin plate C=1")
    ax.axhline(1.15, color="r", ls="--", lw=1.0, label="+15 %")
    ax.set_xlabel("lattice spacing  dx  (c_phys = 4, box = 64)")
    ax.set_ylabel("added-mass coeff  C = m_a / (ρπc²)")
    ax.set_title("(a) refinement axis  dx → 0")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)

    # (b) body-size axis
    ax = axes[1]
    ax.plot(1.0 / c, Cmc, "o-", color="#1f77b4", label="C (momentum)")
    invc = np.linspace(0, 0.28, 50)
    ax.plot(invc, a_c + b_c * invc, "--", color="#1f77b4", alpha=0.5,
            label=f"lin fit  C_inf={a_c:.3f}")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, label="thin plate C=1")
    ax.axhline(1.15, color="r", ls="--", lw=1.0, label="+15 %")
    ax.set_xlabel("1 / c  (body half-width, dx = 1)")
    ax.set_ylabel("added-mass coeff  C")
    ax.set_title("(b) body-size axis  1/c → 0")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.28)

    fig.tight_layout()
    png = OUT / "v9_refinement_limit.png"
    fig.savefig(png, dpi=150)
    print("wrote", png)

    # coupled-frequency implication table
    print("\ncoupled-frequency implication (m_a/m_s = 0.6):")
    for tag, C in [("dx=1", Cm[0]), ("dx=1/2", Cm[1]), ("dx=1/4", Cm[2]),
                   ("limit(lin)", a_m), ("limit(1/c)", a_c)]:
        fr = f_ratio(C)
        fr_ref = f_ratio(1.0)
        err = (fr_ref - fr) / fr_ref
        print(f"  {tag:12s} C={C:.4f}  f/f_vac={fr:.4f}  "
              f"vs thin-plate {fr_ref:.4f}  err={100*err:.2f}%")


if __name__ == "__main__":
    main()
