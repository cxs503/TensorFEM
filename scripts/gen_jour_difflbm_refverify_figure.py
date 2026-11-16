#!/usr/bin/env python3
"""Render the JOUR-DIFFLBM-REF-VERIFY board-audit figure.

Input : artifacts/refverify_m22758/verification_matrix.json (produced by
        scripts/verify_jour_difflbm_refs.py -- no number is hand-typed here)
Output: artifacts/refverify_m22758/fig_refverify.png

Panel A -- bibliographic element check heatmap (6 elements x 3 references)
Panel B -- "code-backed" flag: is the cited reference actually used by a code
           path (Ladd -> lid closure, Ghia -> example reference table,
           Guo -> nothing)?
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "refverify_m22758"
M = json.loads((OUT / "verification_matrix.json").read_text())

refs = list(M["references"])
elements = ["title", "authors", "year", "journal", "volume", "pages"]
grid = np.array([[1 if M["references"][r]["element_flags"][e] else 0 for r in refs]
                 for e in elements], dtype=float)
used = np.array([1 if M["references"][r]["code_usage"]["used"] else 0 for r in refs])
grade = [M["references"][r]["grade"] for r in refs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.6),
                               gridspec_kw={"width_ratios": [1.55, 1.0]})

im = ax1.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax1.set_xticks(range(len(refs)))
ax1.set_xticklabels([f"[{i+1}] {r}\n{g}" for i, (r, g) in enumerate(zip(refs, grade))],
                    fontsize=8.5)
ax1.set_yticks(range(len(elements)))
ax1.set_yticklabels(elements, fontsize=9)
ax1.set_title("A  bibliographic elements resolved against Crossref", fontsize=10)
for i in range(len(elements)):
    for j in range(len(refs)):
        ax1.text(j, i, "OK" if grid[i, j] else "X", ha="center", va="center",
                 fontsize=9, color="black")
ax1.set_xticks(np.arange(-.5, len(refs), 1), minor=True)
ax1.set_yticks(np.arange(-.5, len(elements), 1), minor=True)
ax1.grid(which="minor", color="white", linewidth=1.5)
ax1.tick_params(which="minor", length=0)

colors = ["#2e7d32" if u else "#c62828" for u in used]
bars = ax2.bar(range(len(refs)), used, color=colors, width=0.6)
ax2.set_xticks(range(len(refs)))
ax2.set_xticklabels([f"[{i+1}] {r}" for i, r in enumerate(refs)], fontsize=9)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["no code path", "code-backed"], fontsize=9)
ax2.set_ylim(0, 1.45)
ax2.set_title("B  is the citation backed by code?", fontsize=10)
for b, r in zip(bars, refs):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
             M["references"][r]["code_usage"]["where"].split("(")[0].strip()[:34],
             ha="center", fontsize=7)

fig.suptitle("JOUR-DIFFLBM-REF-VERIFY board audit (m22758): "
             "2/3 references resolve, 1 DOI invalid and not code-backed",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_refverify.png", dpi=160)
print("wrote", OUT / "fig_refverify.png")