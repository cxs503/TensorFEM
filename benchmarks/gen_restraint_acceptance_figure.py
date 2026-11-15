#!/usr/bin/env python3
"""Generate the GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT verification-closure
figure: centre von-Mises bending stress vs embedding coefficient beta,
showing the SSSS (beta=0) -> clamped (beta=1) band for the ice-belt panel.

Data is read from the committed benchmark artifacts (no re-run of the FEM
solver).  Two curves are shown:
  * diagonal_beta (both edges restrained by the same beta) -- the paper band
  * beta_x_only_beta_y_0 (only the longitudinal edges restrained) -- the
    one-way restraint that isolates the longitudinal contribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = HERE / "artifacts" / "restraint_acceptance_band.png"

band = json.loads((ART / "beta_restraint_panel_band.json").read_text())

# diagonal band: beta_x == beta_y
diag = [r for r in band["diagonal_beta"] if r["beta_x"] == r["beta_y"]]
# one-way: only x edges restrained (beta_y = 0)
oneway = [r for r in band.get("beta_x_only_beta_y_0", [])]

fig, ax = plt.subplots(figsize=(7, 5))
bx = [r["beta_x"] for r in diag]
by = [r["sigma_vm_centre_MPa"] for r in diag]
ax.plot(bx, by, "o-", color="#1f77b4", lw=2, label="diagonal $\\beta_x=\\beta_y$ (both edges)")

if oneway:
    ox = [r["beta_x"] for r in oneway]
    oy = [r["sigma_vm_centre_MPa"] for r in oneway]
    ax.plot(ox, oy, "s--", color="#d62728", lw=1.5, label="$\\beta_y=0$ (longitudinal only)")

# baselines
ssss = band["baseline_navier_N401_sigma_vm_centre_MPa"]
ax.axhline(ssss, color="gray", ls=":", lw=1, label=f"SSSS Navier N=401 ({ssss:.2f} MPa)")
clamp = band["band"]["sigma_vm_clamped_MPa"]
ax.axhline(clamp, color="gray", ls=":", lw=1, label=f"clamped ({clamp:.2f} MPa)")

ax.set_xlabel("embedding coefficient $\\beta$ (0 = simply supported, 1 = clamped)")
ax.set_ylabel("centre von-Mises bending stress (MPa)")
ax.set_title("Ice-belt panel: elastic rotational restraint band\n"
             "(0.8 m × 0.7 m × 22 mm, E=206 GPa, 1 MPa full-breadth patch)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
