#!/usr/bin/env python3
"""Figures for the m22176 board audit (all data read from verify_output.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text())

# --- Figure 1: piston anchor convergence (the one genuine numeric check) -----
lv = V["piston"]["levels"]
n = np.array([r["panels"] for r in lv], dtype=float)
err = np.array([r["rel_error"] for r in lv]) * 100.0
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
ax.loglog(n, err, "o-", lw=1.8, ms=6, label="dipole operator vs (8/3)ρa³ (auditor rerun)")
for x, y, r in zip(n, err, lv):
    ax.annotate(f"n_cells={r['n_cells']}\n{y:.4g}%", (x, y), textcoords="offset points",
                xytext=(8, 8), fontsize=7.5)
ax.axhline(3.0, color="crimson", ls="--", lw=1.2, label="3 % gate (ecosystem)")
ax.axhline(5.0, color="orange", ls=":", lw=1.2, label="5 % band (mission test at n=256)")
ax.plot([205887], [0.01], "x", color="grey", ms=8, label="n_cells=512: NOT RUN (315.8 GiB dense)")
ax.set_xlabel("panels N (dense N×N operator)")
ax.set_ylabel("relative error vs Rayleigh [%]")
ax.set_title("GAP-ICE-GATE-POWER: the only real numeric check,\nwhich m22156's own acceptance command deselected", fontsize=9.5)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(HERE / "fig_piston_convergence.png")

# --- Figure 2: what the reference actually says vs the verified anchor ------
b = V["barge"]
bem = 228.6e3
bars = [
    ("verified BEM 2D anchor\n(report 19234 §Q5)", bem / 1e3, "#2b6cb0"),
    ("m22156 strip ref\n(T=6, chain geometry)", b["chain_geometry_L100_B20_T6"]["A33_2D_kg_per_m"] / 1e3, "#c53030"),
    ("m22156 strip ref\n(T=8, own geometry)", b["mission_geometry_L100_B20_T8"]["A33_2D_kg_per_m"] / 1e3, "#e07a5f"),
    ("chain declaration\nA∞/L = 86.1 t/m", 86.1, "#4a5568"),
]
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
xs = np.arange(len(bars))
vals = [v for _, v, _ in bars]
ax.bar(xs, vals, color=[c for _, _, c in bars], width=0.62)
ax.axhspan(bem * 0.85 / 1e3, bem * 1.15 / 1e3, color="green", alpha=0.12,
           label="±15 % band around the verified anchor")
for x, v in zip(xs, vals):
    ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([nm for nm, _, _ in bars], fontsize=7.2)
ax.set_ylabel("2D heave added mass [t/m]")
ax.set_title("The mission 'reference' is +92.5 % off the verified anchor\n(6× outside its own 15 % band); declaration is −62.3 %",
             fontsize=9.5)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_reference_vs_anchor.png")

# --- Figure 3: doc vs shipped code ----------------------------------------
dc = V["doc_vs_code"]
quants = [("C_a", "C_a", 1.0), ("f_3d", "f_3d", 1.0), ("A33_3D [1e7 kg]", "A33_3D_kg", 1e-7)]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
w = 0.36
for i, (label, key, scale) in enumerate(quants):
    d = dc["doc"][key] * scale
    c = dc["code"][key] * scale
    ax.bar(i - w / 2, d, width=w, color="#e07a5f", label="mission document" if i == 0 else None)
    ax.bar(i + w / 2, c, width=w, color="#2b6cb0", label="shipped code" if i == 0 else None)
    rd = dc["rel_diff"][key]
    ax.annotate(f"{rd:+.1%}", (i, max(d, c)), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=7.5)
ax.set_xticks(range(len(quants)))
ax.set_xticklabels([q[0] for q in quants], fontsize=8)
ax.set_title("Mission report numbers are not its code's output\n(A33_3D: 3.6e6 kg documented vs 4.06e7 kg computed)",
             fontsize=9.5)
ax.legend(fontsize=7.5)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_doc_vs_code.png")

print("figures written:", [p.name for p in sorted(HERE.glob("fig_*.png"))])