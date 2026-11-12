#!/usr/bin/env python3
"""Figures for the m15434 / CASE-THERMAL-BAR board audit (numbers from JSON only)."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = json.loads((ROOT / "benchmarks/constrained_thermal_bar_evidence.json").read_text())
IC = json.loads((ROOT / "docs/audit_m15434/independent_check.json").read_text())
OUT = ROOT / "docs/audit_m15434"


def fig_error_budget() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

    rows = EV["studies"]["bar_quadratic_convergence"]["rows"]
    h = [r["h_m"] for r in rows]
    err_mod = [abs(r["abs_rel_error"]) for r in rows]
    err_aud = [abs(r["rel_error"]) for r in IC["checks"]["C2"]["trapezoid_rows"]]
    ax[0].loglog(h, err_mod, "o-", label="module (evidence JSON)")
    ax[0].loglog(h, err_aud, "s--", mfc="none", label="audit re-implementation (trapezoid T)")
    ref = np.array(h) ** 2
    ax[0].loglog(h, ref * err_mod[0] / ref[0], "k:", lw=1.0, label=r"slope 2 reference")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("h [m]")
    ax[0].set_ylabel("relative stress error")
    ax[0].set_title("C2  quadratic-dT stress error vs h\n(order 2.000 reproduced independently)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)

    fam = IC["checks"]["C3"]["families"]
    names = list(fam)
    reff = [abs(fam[k]["ref_recomputed_Pa"]) / 1e6 for k in names]
    got = [abs(fam[k]["got_Pa"]) / 1e6 for k in names]
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, reff, 0.4, label="closed form |sigma| [MPa]")
    ax[1].bar(x + 0.2, got, 0.4, label="FEM (hex8) |sigma| [MPa]")
    ax[1].set_xticks(x, names)
    ax[1].set_ylabel("|sigma_xx| [MPa]")
    ax[1].set_title("C3  restraint family: 4/4 within round-off\n"
                    "(free control = 1.3e-7 Pa vs 2.5e8 Pa scale)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_error_budget.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_error_budget()
    print("wrote", OUT / "fig_error_budget.png")