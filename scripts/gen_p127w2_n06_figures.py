#!/usr/bin/env python
"""Generate figures and final report for P127W2-N06."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.stiffened_plate_ultimate import (
    StiffenedPlateGeometry,
    MaterialProperties,
    run_full_verification,
    classical_stiffened_panel_buckling_load,
    run_arc_length_postbuckling,
    smith_stiffened_plate_mk_curve,
    mesh_convergence_gci,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "p127w2_n06"
FIGURES = OUTPUT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

geo = StiffenedPlateGeometry()
mat = MaterialProperties()

# Run full verification
result = run_full_verification(
    geometry=geo, material=mat,
    mesh_levels=(10, 20, 40),
    arc_steps=50,
    save_dir=str(OUTPUT),
)

# ---------------------------------------------------------------------------
# Figure 1: Load-displacement curve (post-buckling)
# ---------------------------------------------------------------------------
arc_result = run_arc_length_postbuckling(geo, mat, nx=20, ny=10, steps=50)
points = arc_result["points"]

fig, ax = plt.subplots(figsize=(8, 5))
strains = [p["strain"] * 100 for p in points]  # %
loads = [p["load"] / 1e3 for p in points]  # kN
ax.plot(strains, loads, "b-", lw=2, label="Post-buckling path")
ax.axhline(arc_result["buckling_load_Pcr"] / 1e3, ls="--", color="red",
           label=f"P_cr = {arc_result['buckling_load_Pcr']/1e3:.1f} kN")
ax.axhline(arc_result["peak_load_Pu"] / 1e3, ls="-.", color="green",
           label=f"P_u = {arc_result['peak_load_Pu']/1e3:.1f} kN")
ax.set_xlabel("End shortening strain [%]")
ax.set_ylabel("Compressive load [kN]")
ax.set_title("Stiffened Panel: Load-Displacement Curve\n800x400mm, t=6mm, Tee 150x10mm, AH36")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "load_displacement.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Smith M-kappa curve
# ---------------------------------------------------------------------------
smith_result = smith_stiffened_plate_mk_curve(geo, mat)
mk = smith_result["mk_curve"]

fig, ax = plt.subplots(figsize=(8, 5))
kappas = [p["curvature_ratio"] for p in mk]
moments = [p["moment"] / 1e3 for p in mk]
ax.plot(kappas, moments, "b-", lw=2, label="Smith M-kappa")
ax.axhline(smith_result["ultimate_moment"] / 1e3, ls="--", color="red",
           label=f"M_u = {smith_result['ultimate_moment']/1e3:.1f} kN.m")
ax.set_xlabel(r"Curvature ratio $\kappa/\kappa_y$ [-]")
ax.set_ylabel("Section moment [kN.m]")
ax.set_title("Smith Progressive Collapse: M-kappa Curve")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "smith_mk_curve.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Mesh convergence
# ---------------------------------------------------------------------------
gci_result = mesh_convergence_gci(geo, mat, mesh_levels=(10, 20, 40))

fig, ax = plt.subplots(figsize=(7, 5))
nx_vals = [r["nx"] for r in gci_result["mesh_levels"]]
Pcr_vals = [r["P_cr"] / 1e3 for r in gci_result["mesh_levels"]]
ax.plot(nx_vals, Pcr_vals, "o-", lw=2, color="tab:blue")
ax.axhline(gci_result["extrapolated_P_cr"] / 1e3, ls="--", color="red",
           label=f"Extrapolated: {gci_result['extrapolated_P_cr']/1e3:.1f} kN")
ax.set_xlabel("Number of elements along length (nx)")
ax.set_ylabel("Critical buckling load P_cr [kN]")
ax.set_title(f"Mesh Convergence (GCI = {gci_result['GCI_fine']*100:.2f}%)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "mesh_convergence.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: Acceptance criteria summary
# ---------------------------------------------------------------------------
criteria = result["acceptance_criteria"]
names = ["(a) P_cr <5%", "(b) P_u <10%", "(c) Energy <15%",
         "(d) Smith <8%", "(e) CSR >=1", "(f) GCI <3%"]
values = []
passes = []
for key in ["a_Pcr_vs_classical_5pct", "b_Pu_vs_experiment_10pct",
            "c_postbuckling_energy_15pct", "d_smith_Mu_vs_plastic_8pct",
            "e_CSR_Z_act_over_Z_req", "f_GCI_3pct"]:
    c = criteria[key]
    if c["value"] is not None:
        values.append(c["value"])
        passes.append(c["passes"])
    else:
        values.append(None)
        passes.append(None)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(names))
colors = ["green" if p else "red" if p is False else "gray" for p in passes]
bars = ax.bar(x, [v if v is not None else 0 for v in values], color=colors, alpha=0.7)
for i, (name, val, p) in enumerate(zip(names, values, passes)):
    if val is not None:
        ax.text(i, val, f"{val:.3f}" if val < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=9)
        status = "PASS" if p else "FAIL"
        ax.text(i, 0, status, ha="center", va="top", fontsize=8,
                color="white" if p else "white", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Error ratio / Value")
ax.set_title("P127W2-N06 Acceptance Criteria Summary")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "criteria_summary.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("P127W2-N06 Verification Summary")
print("=" * 60)
print(f"Overall status: {result['overall_status']}")
print(f"Criteria passed: {result['criteria_passed']}/{result['criteria_total']}")
print()
print("Geometry:")
print(f"  Panel: {geo.length*1000:.0f} x {geo.width*1000:.0f} mm")
print(f"  Plate thickness: {geo.plate_thickness*1000:.0f} mm")
print(f"  Stiffener: Tee {geo.web_height*1000:.0f}x{geo.web_thickness*1000:.0f} mm")
print(f"  Plate slenderness beta: {geo.plate_slenderness:.2f}")
print()
print("Classical buckling:")
print(f"  P_cr = {result['classical_buckling']['P_cr_kN']:.1f} kN")
print(f"  P_squash = {result['classical_buckling']['P_squash_kN']:.1f} kN")
print()
print("Post-buckling:")
print(f"  P_u = {result['postbuckling']['Pu_kN']:.1f} kN")
print(f"  P_u/P_cr = {result['postbuckling']['Pu_over_Pcr']:.3f}")
print(f"  Energy = {result['postbuckling']['energy_absorption_J']:.1f} J")
print()
print("Smith method:")
print(f"  M_u = {result['smith_method']['ultimate_moment_kNm']:.1f} kN.m")
print(f"  Smith vs plastic error: {result['smith_vs_plastic']['relative_error']*100:.1f}%")
print()
print("CSR check:")
print(f"  Z_act/Z_req = {result['csr_check']['Z_act_over_Z_req']:.3f}")
print(f"  Passes: {result['csr_check']['passes']}")
print()
print("Mesh convergence:")
print(f"  GCI = {result['mesh_convergence']['GCI_fine']*100:.2f}%")
print(f"  Passes <3%: {result['mesh_convergence']['passes_3pct']}")
print()
print("Experimental comparison (SYNTHETIC):")
exp = result["experimental_comparison"]
print(f"  Avg error: {exp.get('avg_relative_error', 'N/A')}")
print(f"  Passes <10%: {exp.get('passes_10pct', 'N/A')}")
print()
print("Figures saved to:", FIGURES)
print("Results saved to:", OUTPUT / "results.json")
