"""Figure + JSON dump for the GAP-PP-3 acceptance-discriminability note."""
from __future__ import annotations

import base64
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    criterion_discrimination, plastic_bound_bracket, plastic_moment)
from tensorfem.ice_plate_plasticity import flow_stress_at_temperature  # noqa: E402
from plate_mechanism_lp_bound import mechanism_upper_bound  # noqa: E402

A, B, T, PATCH = 2.0, 1.0, 0.020, (0.5, 0.5)
SY = flow_stress_at_temperature("DH36", -40.0)
MP = plastic_moment(T, SY)
bracket = plastic_bound_bracket(A, B, T, SY, PATCH, terms=3)
lp = {f"{nx}x{ny}": mechanism_upper_bound(A, B, MP, PATCH, nx=nx, ny=ny)["pressure"]
      for nx, ny in ((16, 8), (24, 12))}
sq = {n: mechanism_upper_bound(1.0, 1.0, 1.0, (1.0, 1.0), nx=n, ny=n)["pressure"]
      for n in (6, 8, 12, 16)}
gate = criterion_discrimination(bracket["tightest_admissible_upper"],
                                bracket["lower_strip"], 0.10)
out = {"case": {"a": A, "b": B, "t": T, "patch": PATCH, "grade": "DH36",
                "temperature_C": -40.0, "sigma_y_MPa": SY / 1e6,
                "m_p_Nm_per_m": MP},
       "bracket_MPa": {k: v / 1e6 for k, v in bracket.items() if k not in (
           "admissible_bracket_width_ratio", "bracket_width_ratio",
           "yield_line_minus_sine_ratio", "agreement_of_upper_bounds_pct")},
       "bracket_ratios": {k: v for k, v in bracket.items() if k in (
           "admissible_bracket_width_ratio", "bracket_width_ratio",
           "yield_line_minus_sine_ratio", "agreement_of_upper_bounds_pct")},
       "mechanism_lp_MPa": {k: v / 1e6 for k, v in lp.items()},
       "square_plate_lp_over_mp": sq, "criterion_gate": gate,
       "ladder_jobs": {"cj4560": "nx=8 queued", "cj4562": "nx=12 queued",
                       "cj4563": "nx=16 queued"}}
(pathlib.Path(__file__).resolve().parents[1] / "artifacts"
 / "gap_pp3_acceptance_discriminability.json").write_text(json.dumps(out, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
lab = ["strip LB", "sine UB\n(inadmissible)", "pyramid UB",
       "Johansen UB", "LP 16x8", "LP 24x12"]
val = [bracket["lower_strip"], bracket["upper_ilyushin_sine"],
       bracket["upper_yield_line"], bracket["upper_johansen_twist"],
       lp["16x8"], lp["24x12"]]
col = ["tab:green", "tab:gray", "tab:blue", "tab:blue", "tab:orange", "tab:orange"]
ax.bar(range(len(val)), np.array(val) / 1e6, color=col)
ax.set_xticks(range(len(val)))
ax.set_xticklabels(lab, fontsize=7)
ax.axhspan(bracket["lower_strip"] / 1e6,
           bracket["tightest_admissible_upper"] / 1e6, color="tab:green", alpha=0.10)
ax.set_ylabel("collapse pressure [MPa]")
ax.set_title(f"anchor panel {A}x{B}x{T} m, patch {PATCH} m\n"
             f"admissible bracket width {gate['bracket_width_ratio']:.2f}", fontsize=9)

ax = axes[1]
n = np.array(sorted(sq))
ax.plot(n, [sq[k] for k in n], "o-", label="LP mechanism UB / m_p")
ax.axhline(24.0, color="k", ls="--", label="exact Johansen 24 m_p/a^2")
ax.set_xlabel("grid n (n x n triangles cells)")
ax.set_ylabel("p_u / m_p  (square plate, uniform load)")
ax.set_ylim(23.9, 24.15)
ax.set_title("validation: LP recovers the exact collapse load", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
png = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "gap_pp3_acceptance_discriminability.png"
fig.savefig(png, dpi=150)
print("wrote", png)
print(json.dumps({"bracket_MPa": out["bracket_MPa"],
                  "bracket_ratios": out["bracket_ratios"], "lp_MPa": out["mechanism_lp_MPa"],
                  "square": sq, "gate": gate}, indent=2))
