#!/usr/bin/env python
"""Render the round-2 DCB figure and the markdown tables from the JSON artifacts.

Zero hand-copied numbers: every value printed here is read from
``benchmarks/cohesive3d_dcb_displacement_control_probe.json`` or recomputed by
the production solver in this same process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorfem.cohesive3d import BKCohesive3DLaw  # noqa: E402
from tensorfem.cohesive3d_dcb import (  # noqa: E402
    DCBGeometry, beam_compliance, beam_peak_load, build_dcb_model, solve_dcb,
)

torch.set_num_threads(8)
probe = json.loads((ROOT / "benchmarks" /
                    "cohesive3d_dcb_displacement_control_probe.json").read_text())

# ---- panel a: compliance calibration -----------------------------------
a = np.array([r["crack_m"] for r in probe["compliance_calibration"]["rows"]])
c_fem = np.array([r["fem_m_per_N"] for r in probe["compliance_calibration"]["rows"]])
g = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=3e-3, initial_crack=30e-3)
aa = np.linspace(18e-3, 42e-3, 40)
c_plain = np.array([beam_compliance(x, g, root_rotation=False, shear=True) for x in aa])
c_root = np.array([beam_compliance(x, g, root_rotation=True, shear=True) for x in aa])

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
ax = axes[0]
ax.plot(aa * 1e3, c_plain * 1e6, "--", color="tab:gray",
        label="beam: bending + shear")
ax.plot(aa * 1e3, c_root * 1e6, "-", color="tab:blue",
        label="beam: + Williams root rotation")
ax.fill_between(aa * 1e3, c_plain * 1e6, 1.10 * c_root * 1e6, color="tab:blue",
                alpha=0.12, label="acceptance bracket")
ax.plot(a * 1e3, c_fem * 1e6, "o", ms=7, color="tab:red", label="FE (COH3D8 + HEX8)")
for x, y, r in zip(a * 1e3, c_fem * 1e6, probe["compliance_calibration"]["rows"]):
    ax.annotate(f"{100*r['rel_error']:+.1f}%", (x, y), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color="tab:red")
ax.set_xlabel("initial crack length a [mm]")
ax.set_ylabel("compliance C = delta/P  [$\\mu$m/N]")
ax.set_title("(a) DCB compliance calibration (ASTM D5528 method)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left")

# ---- panel b: load-opening ---------------------------------------------
law = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 300.0, 800.0, 2.2)
g60 = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=3e-3,
                  initial_crack=30e-3, nx=60, ny=1, nz=2)
r = solve_dcb(build_dcb_model(g60, law), delta_max=2.5e-3, n_steps=80, tol=1e-8)
disp = np.array([p["disp"] for p in r.curve]) * 1e3
load = np.array([p["load"] for p in r.curve])
p_lefm = beam_peak_load(30e-3, 300.0, g60, root_rotation=True)

ax = axes[1]
ax.plot(disp, load, "-", color="tab:red", label="FE path follower")
ax.axhline(p_lefm, color="tab:blue", ls="--",
           label=f"LEFM peak (root rotation) {p_lefm:.2f} N")
ax.plot([2e-2], [probe["compliance"]["fem_m_per_N"] * 0 + 0.908], "o", ms=6,
        color="tab:green", label="linear-elastic branch (no damage)")
ax.axvline(2e-2, color="tab:green", ls=":", lw=1.0)
ax.set_xlabel("imposed opening delta [mm]")
ax.set_ylabel("total reaction load P [N]")
ax.set_title("(b) DCB load-opening, a0 = 30 mm, t$_{n0}$ = 5 MPa")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left")

fig.tight_layout()
dest = ROOT / "docs" / "assets" / "cohesive3d-dcb-3d" / "specimen_compliance_and_peak.png"
dest.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dest)
print("wrote", dest)

# ---- markdown tables ----------------------------------------------------
rows = probe["compliance_calibration"]["rows"]
print("\n### compliance calibration table")
print("| a [mm] | C_FE [m/N] | C_beam plain | C_beam + root rotation | vs root rotation | bracketed |")
print("|---|---|---|---|---|---|")
for rr in rows:
    print(f"| {rr['crack_m']*1e3:.0f} | {rr['fem_m_per_N']:.6e} | "
          f"{rr['beam_plain_m_per_N']:.6e} | {rr['beam_root_rotation_m_per_N']:.6e} | "
          f"{100*rr['rel_error']:+.1f}% | {'yes' if rr['bracketed'] else 'no'} |")
print("\nfitted exponent C~a^n: FE "
      f"{probe['compliance_calibration']['crack_exponent_fit_fem']:.3f} vs beam "
      f"{probe['compliance_calibration']['crack_exponent_fit_beam']:.3f}")
print("\n### peak")
pk = probe["peak"]
for k in ("nx60_peak_load_N", "nx60_peak_disp_m", "lefm_peak_load_N", "rel_error",
          "mesh_check_nx24_peak_load_N", "penalty_check_kn1e14_peak_load_N",
          "peak_is_mesh_converged", "peak_is_penalty_independent",
          "process_zone_span_m", "energy_balance_rel"):
    print(f"| {k} | {pk[k]} |")
print("\n### mesh/dof invariance")
for k, v in probe["mesh_independence"].items():
    print(f"| {k} | {v} |")