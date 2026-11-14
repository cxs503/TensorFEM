#!/usr/bin/env python3
"""Figures for the m20186 board audit of m20143 / CASE-FEM-HEAT-L2."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "verify_heat_l2.json").read_text())
EXTRA = HERE / "verify_heat_l2_ext.json"
if EXTRA.exists():
    ext = json.loads(EXTRA.read_text())
    # merge extended ladder points by mesh label
    have = {r["mesh"] for r in DATA["reproduced_levels"]}
    for r in ext["reproduced_levels"]:
        if r["mesh"] not in have:
            DATA["reproduced_levels"].append(r)

levels = sorted(DATA["reproduced_levels"], key=lambda r: r["h"], reverse=True)
h = np.array([r["h"] for r in levels])
l2_all = np.array([r["l2_relative_all_nodes"] for r in levels])
l2_free = np.array([r["l2_relative_free_dofs"] for r in levels])

n_all = np.log(l2_all[:-1] / l2_all[1:]) / np.log(h[:-1] / h[1:])
n_free = np.log(l2_free[:-1] / l2_free[1:]) / np.log(h[:-1] / h[1:])

declared_orders = DATA["orders"]["declared_orders_all_nodes"]
declared_mean = DATA["orders"]["declared_mean_order"]

# ---- Figure 1: convergence curves -----------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

a = ax[0]
a.loglog(h, l2_all, "o-", color="#c0392b", lw=1.8,
         label="mission metric: node RMS, all nodes (incl. exact Dirichlet)")
a.loglog(h, l2_free, "s-", color="#1f6fb4", lw=1.8,
         label="audit metric: node RMS, free dofs only")
href = np.array([h[-1], h[0]])
a.loglog(href, l2_all[-1] * (href / h[-1]) ** 2.0, "k--", lw=1.0, label="slope h$^2$ (target order 2)")
for x, y, lab in zip(h, l2_all, [f"{n:.3f}" for n in n_all] + [""]):
    a.annotate(lab, (x, y), textcoords="offset points", xytext=(4, -12), fontsize=8, color="#c0392b")
a.set_xlabel("element size h")
a.set_ylabel(r"relative $L_2$ error")
a.set_title("CASE-FEM-HEAT-L2: reproduced convergence (m20186 re-derivation)")
a.grid(True, which="both", alpha=0.3)
a.legend(fontsize=7.5, loc="lower right")
a.invert_xaxis()

b = ax[1]
labels = [f"{r['mesh']}" for r in levels]
x = np.arange(len(labels))
b.plot(x[:len(n_all)], n_all, "o-", color="#c0392b", label="all nodes (mission metric)")
b.plot(x[:len(n_free)], n_free, "s-", color="#1f6fb4", label="free dofs (audit metric)")
b.axhline(2.0, color="k", ls="--", lw=1.0, label="anchor target: order 2")
b.axhline(1.8, color="#e67e22", ls=":", lw=1.4, label="mission self-declared gate: 1.8")
pairs = x[:len(n_all)]
b.set_xticks(pairs)
b.set_xticklabels([f"{labels[i]}\u2192{labels[i+1]}" for i in range(len(pairs))], fontsize=8)
b.set_ylabel("observed order")
b.set_ylim(1.5, 2.6)
b.set_title("Observed h-refinement order per grid pair")
b.grid(True, alpha=0.3)
b.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(HERE / "fig_m20186_convergence.png", dpi=150)
plt.close(fig)

# ---- Figure 2: claim vs evidence ------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.4))
bars = ["declared\nmean order\n(1.876)", "finest declared\npair (1.935)",
        "audit 64x64 free-dof\n(2.038)", "m20054 Gauss L2\n(1.998-2.000)"]
vals = [declared_mean, max(declared_orders),
        DATA["orders"]["observed_order_free_dofs"][-1], 2.000]
colors = ["#c0392b", "#c0392b", "#1f6fb4", "#27ae60"]
ax.bar(bars, vals, color=colors)
ax.axhline(2.0, color="k", ls="--", lw=1.2, label="anchor requirement: order 2")
ax.axhline(1.8, color="#e67e22", ls=":", lw=1.4, label="mission gate: 1.8")
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylim(1.6, 2.12)
ax.set_ylabel("L2 convergence order")
ax.set_title("m20143 CASE-FEM-HEAT-L2: '>=2nd order' claim vs reproducible evidence")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_m20186_claim.png", dpi=150)
plt.close(fig)
print("wrote", HERE / "fig_m20186_convergence.png", HERE / "fig_m20186_claim.png")