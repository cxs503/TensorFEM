"""Artifacts and figures for the discrete beam-grid verification (round 4).

Writes into ``artifacts/grillage_beam_grid/``:

* ``grillage_beam_grid_verification.json`` -- gates, findings, sensitivity table;
* ``fig_beam_grid_field.png`` -- discrete grid deflection field with the ice patch;
* ``fig_beam_grid_convergence.png`` -- peak member moments vs mesh refinement,
  against the smeared-layer demands;
* ``fig_beam_grid_shares.png`` -- share comparison and ``load_split`` sensitivity.

    PYTHONPATH=src python gen_grillage_beam_grid_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.grillage_beam_grid import (
    beam_grid_case_study,
    run_beam_grid_verification,
    smeared_member_demands,
)

OUT = Path("artifacts/grillage_beam_grid")
OUT.mkdir(parents=True, exist_ok=True)

payload = run_beam_grid_verification(OUT)
findings = payload["findings"]
demands = smeared_member_demands()
print("gates:", payload["n_gates"], "failed:", payload["n_failed"])

# --------------------------------------------------------------------------- #
# 1. field of the discrete grid
# --------------------------------------------------------------------------- #
grid = beam_grid_case_study(sub=12)
coords = grid["coords"]
w = grid["w_field"] * 1e3                     # mm
fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
sc = ax.scatter(coords[:, 0], coords[:, 1], c=w, s=6, cmap="viridis")
fx1, fx2, fy1, fy2 = grid["patch"]
ax.add_patch(plt.Rectangle((fx1 * grid["span_x"], fy1 * grid["span_y"]),
                           (fx2 - fx1) * grid["span_x"], (fy2 - fy1) * grid["span_y"],
                           fill=False, ec="crimson", lw=1.6, label="ice patch"))
for x in np.linspace(0.0, grid["span_x"], grid["n_x_lines"]):
    ax.axvline(x, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("x [m]  (stiffener direction)")
ax.set_ylabel("y [m]  (frame direction)")
ax.set_title(f"discrete grillage deflection, p_ice = {grid['pressure'] / 1e6:.1f} MPa "
             f"(w_max = {grid['w_max'] * 1e3:.2f} mm)")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(sc, ax=ax, label="w [mm]")
fig.savefig(OUT / "fig_beam_grid_field.png", dpi=160)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 2. convergence of the peak member moments
# --------------------------------------------------------------------------- #
subs = [6, 12, 24, 48]
mx, my = [], []
for s_ in subs:
    r = beam_grid_case_study(sub=s_)
    mx.append(r["moment_x_max"] / 1e3)
    my.append(r["moment_y_max"] / 1e3)
fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
ax.plot(subs, mx, "o-", label="discrete peak |M| x-members (stiffeners)")
ax.plot(subs, my, "s-", label="discrete peak |M| y-members (frames)")
ax.axhline(demands["stiffener_moment"] / 1e3, ls="--", color="C0",
           label="smeared stiffener demand")
ax.axhline(demands["frame_moment_equivalent"] / 1e3, ls="--", color="C1",
           label="smeared frame demand (equivalent)")
ax.axhline(demands["frame_plastic_moment"] / 1e3, ls=":", color="0.3",
           label="frame plastic moment $M_p$")
ax.set_xscale("log")
ax.set_xlabel("beam elements per structural span ('sub')")
ax.set_ylabel("member bending moment [kN m]")
ax.set_title("mesh convergence and the smeared-layer demand")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
fig.savefig(OUT / "fig_beam_grid_convergence.png", dpi=160)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 3. shares and the load-split assumption
# --------------------------------------------------------------------------- #
cmp_ = payload["discrete_vs_smeared"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
labels = ["smeared\nenergy", "discrete\nenergy", "discrete\nreaction"]
vals = [cmp_["smeared_share_x"], cmp_["discrete_share_x_energy"],
        cmp_["discrete_share_x_reaction"]]
bars = ax1.bar(labels, vals, color=["C0", "C2", "C3"])
ax1.axhline(0.0, color="0.4", lw=0.8)
ax1.set_ylabel("share handed to the stiffener system $f_x$")
ax1.set_title("the smeared share vs the discrete grid")
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
             ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax1.grid(alpha=0.3, axis="y")

splits = payload["load_split_sensitivity"]
keys = sorted(splits, key=float)
ax2.plot([float(k) for k in keys], [splits[k]["share_x_energy"] for k in keys],
         "o-", label="$f_x$ (energy share)")
ax2.plot([float(k) for k in keys],
         [splits[k]["moment_y_max"] / 1e4 for k in keys], "s-",
         label="$M_y^{max}$ / 10$^4$ [kN m]")
ax2.set_xlabel("load_split (fraction applied directly to the stiffener lines)")
ax2.set_ylabel("value")
ax2.set_title("sensitivity to the load-application assumption")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig.savefig(OUT / "fig_beam_grid_shares.png", dpi=160)
plt.close(fig)

summary = {
    "gates": payload["n_gates"],
    "gates_failed": payload["n_failed"],
    "smeared_share_x": cmp_["smeared_share_x"],
    "discrete_share_x_energy": cmp_["discrete_share_x_energy"],
    "discrete_share_x_reaction": cmp_["discrete_share_x_reaction"],
    "stiffener_moment_smeared": demands["stiffener_moment"],
    "stiffener_moment_discrete": findings["discrete_peak_moment_x"],
    "frame_moment_smeared_equivalent": demands["frame_moment_equivalent"],
    "frame_moment_discrete": findings["discrete_peak_moment_y"],
    "frame_capacity_pressure_smeared": findings["frame_capacity_pressure_smeared"],
    "frame_capacity_pressure_discrete": findings["frame_capacity_pressure_discrete"],
    "case_study_ultimate_pressure": findings["case_study_ultimate_pressure"],
    "governing_member_at_ultimate": findings["governing_member_at_ultimate"],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))