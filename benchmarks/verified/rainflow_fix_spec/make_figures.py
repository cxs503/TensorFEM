#!/usr/bin/env python3
"""Figures for docs/verification/rainflow-fix-spec-contradiction.md.

Reads ``result.json`` produced by ``run.py`` (no hand-copied numbers) and writes
``rainflow_fix_spec.png``: left panel = nested-zigzag cycle decomposition per
candidate counter, right panel = narrow-band Rayleigh gate error per candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "result.json").read_text())
cands = list(data["verdict"])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# --- left: nested zigzag decomposition -------------------------------------- #
names = [f"{r:.0f}" for r in data["anchors"]["nested_zigzag"]["counters"]["remainder"]["full_ranges"]]
width = 0.26
for k, cname in enumerate(cands):
    st = data["anchors"]["nested_zigzag"]["counters"][cname]
    rows = [(r, 1.0) for r in st["full_ranges"]] + [(r, 0.5) for r in st["half_ranges"]]
    for i, (rng, cnt) in enumerate(sorted(rows)):
        axes[0].bar(i + (k - 1) * width, cnt * rng, width=width, label=cname if i == 0 else None)
axes[0].axhline(data["anchors"]["nested_zigzag"]["half_path"], ls="--", lw=1, color="k")
axes[0].set_xlabel("cycle index (sorted by range)")
axes[0].set_ylabel("count x range  [stress units]")
axes[0].set_title("nested 0,1,-1,2,-2,3,-3,0\narc conservation target = dashed")
axes[0].legend(fontsize=7)

# --- middle: sine ------------------------------------------------------------#
for k, cname in enumerate(cands):
    st = data["anchors"]["sine_40_periods"]["counters"][cname]
    axes[1].bar(k, st["sum_count"], width=0.6, label=cname)
    axes[1].text(k, st["sum_count"] + 1.0, f"{st['sum_count']:.1f}\nmax {st['max_range']:.0f}",
                 ha="center", fontsize=7)
axes[1].axhline(40.0, ls="--", lw=1, color="k")
axes[1].set_xticks(range(len(cands)))
axes[1].set_xticklabels(cands, rotation=15, fontsize=7)
axes[1].set_ylabel("equivalent cycles  sum(count)")
axes[1].set_title("40-period sine: C3 wants 40 cycles\nof range 2A = 100 (dashed)")

# --- right: Rayleigh gate ----------------------------------------------------#
for k, cname in enumerate(cands):
    err = data["narrow_band_gate"][cname]["relative_error"]
    axes[2].bar(k, 100.0 * err, width=0.6)
    axes[2].text(k, 100.0 * err + 1.0, f"{100.0 * err:.2f}%", ha="center", fontsize=8)
axes[2].axhline(3.0, ls="--", lw=1, color="k")
axes[2].set_xticks(range(len(cands)))
axes[2].set_xticklabels(cands, rotation=15, fontsize=7)
axes[2].set_ylabel("|damage / Rayleigh - 1|  [%]")
axes[2].set_title("narrow-band Rayleigh gate (C2)\ntolerance < 3% (dashed)")

fig.tight_layout()
out = HERE / "rainflow_fix_spec.png"
fig.savefig(out, dpi=140)
print("wrote", out)