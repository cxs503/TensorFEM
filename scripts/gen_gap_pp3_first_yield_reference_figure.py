#!/usr/bin/env python
"""Figure for docs/CASE-GAP-PP-3-FIRST-YIELD-REFERENCE.md (programmatic injection)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ev = json.load(open(os.path.join(ROOT, "docs", "assets", "gap_pp3_transition_reference.json")))
b = json.load(open(os.path.join(ROOT, "docs", "assets", "gap_pp3_3grid_results.json")))
py = ev["analytic"]["first_yield_pressure_Pa"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
names = ["8x4", "16x8", "24x12"]
errs = [100 * b["meshes"][n]["first_yield_error"] for n in names]
ax1.plot([1, 2, 3], errs, "o-", color="C3", label="nonlinear-fixed (geometric=True, restrained)")
lin = 100 * ev["measured"]["8x4"]["linear-free"]["first_yield_error"]
ax1.plot([1], [lin], "s", color="C0", markersize=9,
         label="linear-free (same mesh, comparable reference)")
ax1.axhline(5.0, color="k", ls="--", lw=1, label="acceptance +5 %")
ax1.set_xticks([1, 2, 3]); ax1.set_xticklabels(names)
ax1.set_xlabel("mesh"); ax1.set_ylabel("first-yield error vs linear Navier [%]")
ax1.set_title("transition point: plateau is not an acceptance failure\n(it is a model/reference mismatch + an unexplained offset)")
ax1.legend(fontsize=8); ax1.grid(alpha=.3)

ax2.bar(["linear-free\n8x4", "nonlinear-fixed\n8x4"], [lin, errs[0]], color=["C0", "C3"])
for i, v in enumerate([lin, errs[0]]):
    ax2.text(i, v + 0.6, f"{v:+.1f} %", ha="center", fontsize=9)
ax2.axhline(5.0, color="k", ls="--", lw=1)
ax2.set_ylabel("first-yield error [%]")
ax2.set_title("membrane action moves the transition ~9 points\ntowards the reference, and does not explain the rest")
ax2.grid(axis="y", alpha=.3)
fig.tight_layout()
out = os.path.join(ROOT, "outputs", "figures", "gap_pp3_first_yield_reference.png")
fig.savefig(out, dpi=140)
print(out, "lin=%.2f fixed8x4=%.2f plateau=%s" % (lin, errs[0], [round(e, 2) for e in errs]))
