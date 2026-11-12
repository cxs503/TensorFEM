#!/usr/bin/env python3
"""m15654 audit figure: what the mission's numbers actually support.

panel 1  distorted-patch h-convergence (the only non-tautological patch evidence) vs slope 2
panel 2  gate margin: declared value / threshold for all 17 gates (log x), tautological gate flagged
panel 3  Poisson cross-check: declared 0.0 vs ratio recomputed from the *computed* tip
         deflections (the number the gate was meant to test), + 2% tolerance band

Reads only committed artefacts + the audit's re-derivation JSON; nothing is hand-entered.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
RES = json.loads((REPO / "benchmarks/verified/fem_bending_patch/result.json").read_text())
GAT = json.loads((REPO / "benchmarks/verified/fem_bending_patch/gates_verified.json").read_text())
POI = json.loads((HERE / "poisson_crosscheck_rederived.json").read_text())
POOL = GAT["criteria"]["case_pool_tolerance"]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

# ── panel 1: h-convergence of the distorted patch ─────────────────────────────
ax = axes[0]
lad = RES["ladder"]
for treat, style in (("sri", "-o"), ("mitc", "--s")):
    rows = sorted([r for r in lad if r["shear_treatment"] == treat], key=lambda r: r["h"])
    h = [r["h"] for r in rows]
    ax.loglog(h, [r["displacement_rel_error"] for r in rows], style, color="C0",
              label=f"w ({treat})")
    ax.loglog(h, [r["moment_rel_error"] for r in rows], style, color="C3", alpha=0.75,
              label=f"M ({treat})")
h = [r["h"] for r in sorted(lad, key=lambda r: r["h"])][::2]
ax.loglog(h, [POOL * (x / h[0]) ** 2 for x in h], "k:", lw=1, label="slope 2 guide")
ax.axhline(POOL, color="k", ls="--", lw=1)
ax.text(h[0], POOL * 1.15, "case-pool tolerance 2%", fontsize=8)
ax.set_xlabel("h (element size, distorted patch)")
ax.set_ylabel("relative error")
ax.set_title("(1) distorted patch: only non-tautological\npatch evidence; w-order 1.72 (sri)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3, which="both")

# ── panel 2: gate margins ────────────────────────────────────────────────────
ax = axes[1]
names, margins, colors = [], [], []
for g in GAT["gates"]:
    v, t, op = g["value"], g["threshold"], g["operator"]
    m = (t / v) if (op == "<=" and v) else (v / t if t else float("nan"))
    names.append(g["id"])
    margins.append(m)
    colors.append("#c0392b" if g["id"] == "poisson_crosscheck" else
                  ("#e67e22" if m < 1.5 else "#27ae60"))
y = range(len(names))
ax.barh(list(y), margins, color=colors)
ax.axvline(1.0, color="k", lw=1)
ax.axvline(1.5, color="grey", ls=":", lw=1)
ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=6.5)
ax.set_xscale("log")
ax.set_xlabel("margin = threshold/value (>=1 passes); dotted 1.5x = thin")
ax.set_title("(2) 17 gates: red = tautological (F-CRIT),\norange = margin < 1.5x")
ax.grid(alpha=0.3, axis="x", which="both")

# ── panel 3: Poisson cross-check, declared vs recomputed ─────────────────────
ax = axes[2]
theory = POI["recomputed_from_computed_tip_deflections"][0]["ratio_theory"]
ax.axhline(theory, color="k", lw=1, label=f"theory 1/(1-$\\nu^2$)={theory:.5f}")
ax.axhspan(theory * (1 - POOL), theory * (1 + POOL), color="grey", alpha=0.2,
           label="2% tolerance band")
for treat, mk in (("sri", "o"), ("mitc", "s")):
    rows = [r for r in POI["recomputed_from_computed_tip_deflections"]
            if r["shear"] == treat and r["t"] == 0.01]
    ax.plot([r["n_x"] for r in rows], [r["ratio_computed"] for r in rows], mk + "-",
            label=f"recomputed from w_tip ({treat})")
ax.axhline(POI["declared_value"]["measured"], color="#c0392b", ls="--", lw=1.6,
           label="declared 'measured' (closed-form ratio) = theory exactly")
ax.set_xscale("log", base=2)
ax.set_xlabel("n_x (elements along the strip)")
ax.set_ylabel("w_free / w_symmetry")
ax.set_title("(3) F-CRIT: gate reports 0.0 error because both\noperands are closed form, not w_tip")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

fig.suptitle("m15654 board audit — CASE-BENDING-PATCH (m15427, TensorFEM): what the numbers support",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "fig_bending_patch_audit.png"
fig.savefig(out, dpi=140)
print("wrote", out)