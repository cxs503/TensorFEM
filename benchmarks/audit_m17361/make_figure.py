#!/usr/bin/env python3
"""Board audit m17408 — figures for the m17361 (XVAL-FEM-IBX-Aitken) audit.

Panel (a): Aitken omega sequence from the delivered artifacts (synthetic-residual
           unit test, xval_results.json -> aitken.convergence_test.omegas).
Panel (b): reference-vs-got error for CD / CL_amp / St, as delivered vs as
           restated by the audit with the repository's own normalisation.
Panel (c): CL_amp and St error sensitivity to the averaging window / grid
           (the quantity the delivered report claimed without evidence).

Every number is read from JSON files written by a run; nothing is hand-typed.
"""
from __future__ import annotations

import base64
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
AU = os.path.join(REPO, "artifacts", "audit_m17361")
OUT_PNG = os.path.join(AU, "audit_m17361_metrics.png")

delivered = json.load(open(os.path.join(AU, "raw", "xval_results.json")))
fixed = json.load(open(os.path.join(AU, "fixed_run_results.json")))
recheck = json.load(open(os.path.join(AU, "recheck_metrics.json")))

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

# (a) Aitken omega sequence (delivered artefacts)
om = delivered["aitken"]["convergence_test"]["omegas"]
ax[0].plot(np.arange(1, len(om) + 1), om, "o-", ms=3.5, lw=1.2, color="#1f77b4")
ax[0].axhline(2.0, ls="--", lw=1, color="grey")
ax[0].axhline(-2.0, ls="--", lw=1, color="grey")
ax[0].set_xlabel("interface iteration k")
ax[0].set_ylabel(r"$\omega_k$")
ax[0].set_title("(a) Aitken $\\Delta^2$ relaxation, delivered trace\n"
                "(synthetic residual, omega_max=2 clamp)")
ax[0].grid(alpha=.3)

# (b) error as delivered vs restated
labels = ["$C_D$", "$C_{L,amp}$", "$St$"]
d = delivered["fluid"]["cylinder_Re100"]
f = fixed["fluid"]["cylinder_Re100"]
err_del = [d["error_CD"], d["error_CL"], d["error_St"]]
err_fix = [f["error_CD"], f["error_CL"], f["error_St"]]
x = np.arange(3)
w = 0.36
ax[1].bar(x - w / 2, [100 * e for e in err_del], w, color="#d62728",
          label="delivered formulas")
ax[1].bar(x + w / 2, [100 * e for e in err_fix], w, color="#2ca02c",
          label="audit restatement")
ax[1].axhline(15, ls="--", color="k", lw=1.2)
ax[1].text(2.45, 17, "15 % criterion", ha="right", fontsize=9)
for xi, e in zip(x - w / 2, err_del):
    ax[1].text(xi, 100 * e + 1.5, "%.1f" % (100 * e), ha="center", fontsize=8)
for xi, e in zip(x + w / 2, err_fix):
    ax[1].text(xi, 100 * e + 1.5, "%.1f" % (100 * e), ha="center", fontsize=8)
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].set_ylabel("|error| vs Turek-Hron 2D-2 ref  [%]")
ax[1].set_title("(b) same force history, two normalisations\n"
                "D/dx=12, 1000-step average")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3, axis="y")

# (c) window / grid sensitivity of the restated metrics
runs = recheck["runs"]
tags = ["D/dx=12\n1000-step avg", "D/dx=12\n2000-step avg", "D/dx=20\n2000-step avg"]
cd = [abs(r["err_pct"]["CD_fixed"]) for r in runs]
cl = [abs(r["err_pct"]["CL_amp"]) for r in runs]
st = [abs(r["err_pct"]["St_fixed"]) for r in runs]
x = np.arange(len(runs))
for i, (vals, lab, col) in enumerate(((cd, "$C_D$", "#1f77b4"),
                                      (cl, "$C_{L,amp}$", "#ff7f0e"),
                                      (st, "$St$", "#2ca02c"))):
    ax[2].bar(x + (i - 1) * 0.26, vals, 0.26, label=lab, color=col)
ax[2].axhline(15, ls="--", color="k", lw=1.2)
ax[2].set_xticks(x)
ax[2].set_xticklabels(tags)
ax[2].set_ylabel("|error| after restatement [%]")
ax[2].set_title("(c) averaging-window / grid sensitivity\n(recheck_metrics.json)")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=.3, axis="y")

fig.suptitle("Board audit m17408 — m17361 XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION "
             "(TensorFEM vs joconnor22/LIFE)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PNG, dpi=140)
print("wrote", OUT_PNG, os.path.getsize(OUT_PNG), "bytes")

# also emit base64 for the platform report
with open(OUT_PNG, "rb") as fh:
    b64 = base64.b64encode(fh.read()).decode()
json.dump({"figures": [{
    "name": "audit_m17361_metrics.png",
    "caption": "(a) delivered Aitken omega trace; (b) CD/CL/St error with the "
               "delivered vs restated normalisation at identical force history; "
               "(c) sensitivity of the restated errors to averaging window and grid.",
    "png_base64": b64}]},
    open(os.path.join(AU, "figures.json"), "w"))
print("wrote figures.json")