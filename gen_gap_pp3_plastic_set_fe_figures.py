#!/usr/bin/env python3
"""Generate the 4 report figures locally from results.json (the cluster node
has no matplotlib).  Reads artifacts/gap_pp3_plastic_set_fe/results.json and
writes fig_*.png next to it.  Every number comes from the JSON."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts" / "gap_pp3_plastic_set_fe"
R = json.loads((ART / "results.json").read_text())
C = R["case"]
PY = C["first_yield_pressure_Pa"] / 1e6
PUB = C["yield_line_upper_bound_Pa"] / 1e6
SY = C["yield_stress_Pa"] / 1e6
GRIDS = ["8x4", "12x6", "16x8"]


def g(section, key, field):
    try:
        return R[section][key][field]
    except (KeyError, TypeError):
        return float("nan")


# Fig 1: transition
fig, ax = plt.subplots(figsize=(7, 4.5))
fe_uni = [g("transition", f"uniaxial_{n}", "fe_yield_load") / 1e6
           for n in GRIDS]
fe_lat = [g("transition", f"lateral_{n}", "fe_yield_load") / 1e6
          for n in GRIDS]
x = range(len(GRIDS))
ax.bar([i - 0.2 for i in x], fe_uni, width=0.4, label="FE uniaxial")
ax.bar([i + 0.2 for i in x], fe_lat, width=0.4, label="FE lateral")
ax.axhline(SY, color="k", ls="--", lw=1)
ax.text(0.02, SY * 1.01, f"analytical = {SY:.0f} MPa", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(GRIDS)
ax.set_ylabel("yield load [MPa]")
ax.set_title("Elastic->plastic transition vs analytical yield load")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ART / "fig_transition.png", dpi=140)
plt.close(fig)

# Fig 2: limit
fig, ax = plt.subplots(figsize=(7, 4.5))
fe_lim = [g("limit", f"lateral_{n}", "fe_limit") / 1e6 for n in GRIDS]
ax.bar([i - 0.2 for i in x], fe_lim, width=0.4,
        label="FE limit (hardening)")
ax.axhline(PUB, color="r", ls="--", lw=1)
ax.text(0.02, PUB * 1.01, f"yield-line upper bound = {PUB:.3f} MPa",
         color="r", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(GRIDS)
ax.set_ylabel("limit pressure [MPa]")
ax.set_title("Plastic limit pressure vs yield-line upper bound")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ART / "fig_limit.png", dpi=140)
plt.close(fig)

# Fig 3: convergence
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, q in zip(axes, ("limit_load", "max_deflection")):
    d = R["convergence"].get(q)
    if d is None:
        ax.text(0.5, 0.5, "not run", ha="center")
        continue
    ne = [m["n_elements"] for m in d["meshes"]]
    vals = [v / 1e6 if q == "limit_load" else v * 1e3 for v in d["values"]]
    ax.plot(ne, vals, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("n elements")
    ax.set_ylabel("limit [MPa]" if q == "limit_load"
                 else "max deflection [mm]")
    ax.set_title(f"{q} (passed={d['passed']})")
fig.tight_layout()
fig.savefig(ART / "fig_convergence.png", dpi=140)
plt.close(fig)

# Fig 4: permanent
fig, ax = plt.subplots(figsize=(7, 4.5))
PEAKS = (("below_yield_0p5", 0.5), ("at_upper_bound_1p0", 1.0),
         ("above_upper_bound_1p2", 1.2))
for n in GRIDS:
    w = [g("permanent", f"lateral_{n}.{l}", "permanent_deflection_m")
          for l, _ in PEAKS]
    ax.plot([f * PUB for _, f in PEAKS], [w * 1e3 for w in w],
            marker="o", label=n)
ax.axvline(PUB, color="r", ls=":", lw=1)
ax.text(PUB * 1.01, ax.get_ylim()[1] * 0.9, "yield-line upper bound",
         color="r", fontsize=8)
ax.set_xlabel("peak patch pressure [MPa]")
ax.set_ylabel("max permanent deflection [mm]")
ax.set_title("Permanent deflection after full unloading")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ART / "fig_permanent.png", dpi=140)
plt.close(fig)

print("figures written to", ART)
