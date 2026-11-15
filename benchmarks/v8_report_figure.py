#!/usr/bin/env python3
"""Aggregate the v8 artefacts into ``summary.json`` + the report figure.

Reads (all written by the sweeps, none retyped by hand):

* ``bem.json``             -- independent potential-flow reference
* ``clean_refine_*.json``  -- clean-protocol grid refinement (dx = 1, 1/2, 1/4)
* ``clean_c_*.json``       -- clean-protocol body-size sweep at dx = 1
* ``refine.json`` / ``csw.json`` -- the archived-style protocol, kept as the
  control that shows the window/period systematic

    python3 benchmarks/v8_report_figure.py
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "xval_ibm_added_mass_refinement_v8_artifacts")


def load(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(ART, pattern))):
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def implied_thickness(target, thickness):
    hc = np.array([r["h_over_c"] for r in thickness])
    ca = np.array([r["C_a"] for r in thickness])
    if target <= ca[0]:
        return None
    if target <= ca[-1]:
        return float(np.interp(target, ca, hc))
    # logarithmic extrapolation beyond the table
    k = (ca[-1] - ca[-2]) / (math.log(hc[-1]) - math.log(hc[-2]))
    return float(hc[-1] * math.exp((target - ca[-1]) / k))


def main():
    bem = json.load(open(os.path.join(ART, "bem.json")))
    clean_refine = load("clean_refine_*.json")
    clean_c = load("clean_c_*.json")
    arch = json.load(open(os.path.join(ART, "refine.json"))) if \
        os.path.exists(os.path.join(ART, "refine.json")) else {"series": [], "controls": []}

    thick = {r["h_over_c"]: r["C_a"] for r in bem["thickness"]}
    series = [{"refine": r["refine"], "dx": r["dx"], "c_lat": r["c_lat"],
               "n_cell": r["n_cell"], "C_momentum": r["C_momentum"],
               "C_force": r["C_force"], "r2_momentum": r["r2_momentum"],
               "constraint_rel_max": r["constraint_rel_max"],
               "wall_s": r.get("wall_s")} for r in clean_refine]
    summary = {
        "bem_validation_max_rel_err": max(
            max(abs(v) for k, v in row.items() if k.startswith("err_")) for row in bem["validation"]),
        "bem_thickness": [{"h_over_c": k, "C_a": v} for k, v in sorted(thick.items())],
        "clean_refine": series,
        "clean_csize": [{"c_lat": r["c_lat"], "box": r["box_phys"],
                         "C_momentum": r["C_momentum"]} for r in clean_c],
        "archived_protocol": [{"refine": r["refine"], "c_lat": r["c_lat"],
                               "period_phys": 64.0, "n_cycles": 4.0,
                               "C_momentum": r["C_momentum"]} for r in arch["series"]],
    }
    if series:
        c0 = series[0]["C_momentum"]
        summary["C_dx1"] = c0
        summary["implied_h_over_c"] = implied_thickness(c0, bem["thickness"])
        summary["implied_thickness_cells"] = (None if summary["implied_h_over_c"] is None
                                              else summary["implied_h_over_c"] * 4.0)
        summary["C_kernel_thickness"] = thick.get(0.375)
        summary["geometry_ratio"] = c0 / thick[0.375]
    if len(series) >= 2:
        xs = np.array([r["c_lat"] for r in series])
        ys = np.array([r["C_momentum"] for r in series])
        A = np.stack([np.ones_like(1.0 / xs), 1.0 / xs], axis=1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        summary["fit_C_inf_plus_k_over_c"] = {"C_inf": float(coef[0]),
                                             "k": float(coef[1]),
                                             "max_resid": float(np.max(np.abs(A @ coef - ys)))}
    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.3))

    hc = [r["h_over_c"] for r in bem["thickness"]]
    ca = [r["C_a"] for r in bem["thickness"]]
    ax[0].plot(hc, ca, "o-", ms=4, lw=1.4, label="rectangle, BEM")
    ax[0].axhline(1.0, color="k", ls=":", lw=1, label="thin plate $C_a=1$")
    ax[0].axhline(thick[0.375], color="tab:green", ls="--", lw=1.2,
                  label="body of kernel thickness ($3\\Delta$)")
    if series:
        ax[0].axhline(series[0]["C_momentum"], color="tab:red", ls="-", lw=1.2,
                      label="measured, IBM plate ($\\Delta$ lattice)")
        ax[0].axhline(summary["bem_thickness"][-1]["C_a"], color="0.5", ls=":", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$h/c$ (half-thickness / half-chord)")
    ax[0].set_ylabel("$C_a=m_a/(\\rho\\pi c^2)$")
    ax[0].set_title("(a) finite thickness cannot explain it")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)

    if series:
        xs = [r["c_lat"] for r in series]
        ax[1].plot(xs, [r["C_momentum"] for r in series], "o-", color="tab:red",
                   label="momentum route (clean protocol)")
        ax[1].plot(xs, [r["C_force"] for r in series], "s--", color="tab:orange",
                   label="marker-force route")
        if arch["series"]:
            ax[1].plot([r["c_lat"] for r in arch["series"]],
                       [r["C_momentum"] for r in arch["series"]], "v:", color="0.4",
                       label="archived protocol (2 cycles)")
        ax[1].axhline(1.0, color="k", ls=":", label="thin plate")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel("$c/\\Delta$ (half-chord in cells)")
        ax[1].set_ylabel("$C=m_a/(\\rho\\pi c^2)$")
        ax[1].set_title("(b) grid refinement at fixed physical geometry")
        ax[1].grid(alpha=0.3)
        ax[1].legend(fontsize=7)

    jrow = None
    for path in sorted(glob.glob(os.path.join(ART, "clean_refine_*.json"))):
        with open(path) as fh:
            cand = json.load(fh)
        if jrow is None or cand["refine"] > jrow["refine"]:
            jrow = cand
    if jrow is not None:
        ax[2].plot(jrow["momentum_history"], lw=0.8, label="$\\Pi_y$ fluid momentum")
        ax[2].plot(jrow["force_history"], lw=0.8, alpha=0.75, label="$F_y$ marker force")
        ax[2].set_title(f"(c) finest clean run, $c/\\Delta={jrow['c_lat']:.0f}$, "
                        f"$r^2={jrow['r2_momentum']:.4f}$")
        ax[2].set_xlabel("step")
        ax[2].set_ylabel("lattice units")
        ax[2].grid(alpha=0.3)
        ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(ART, "v8_added_mass_refinement.png")
    fig.savefig(out, dpi=150)
    print("figure ->", out)
    return summary


if __name__ == "__main__":
    main()