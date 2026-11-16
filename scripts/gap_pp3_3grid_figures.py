#!/usr/bin/env python
"""Figures for the GAP-PP-3 3-grid matrix (all numbers read from the merged JSON).

Three panels, each one an acceptance criterion:

1. ``gap_pp3_3grid_transition_error.png`` -- raw FE-vs-Navier first-yield error
   vs element size ``h = b/ny`` on log-log axes, with the observed order and the
   Richardson limit drawn as the h -> 0 asymptote;
2. ``gap_pp3_3grid_limit_bracket.png`` -- measured limit pressure per mesh
   against the plastic bracket (strip lower bound .. yield-line upper bound),
   with the +-10 % band around the upper bound shaded;
3. ``gap_pp3_3grid_paths.png`` -- load-deflection path per mesh with the unload
   branch, so the permanent deflection is visible as the residual at p = 0.

No number is typed into this file: every value comes from
``outputs/gap_pp3_3grid_results.json`` produced by the study + merge scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")


def _meshes(results):
    return sorted(results["meshes"].items(), key=lambda kv: kv[1]["elements"])


def fig_transition(results, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    conv = results["criteria"]["raw_error_convergence"]
    h = conv["h"]
    err = [abs(e) * 100.0 for e in conv["errors"]]
    ax.plot(h, err, "o-", lw=1.8, ms=7, label="raw FE vs Navier ($|e|$, %)")
    for (name, _), hh, ee in zip(_meshes(results), h, err):
        ax.annotate(name, (hh, ee), textcoords="offset points", xytext=(6, 6), fontsize=9)
    if "extrapolated_error" in conv:
        ax.axhline(abs(conv["extrapolated_error"]) * 100.0, ls="--", lw=1.2,
                   color="tab:red",
                   label=("Richardson limit = %.2f %%  (order p = %.2f)"
                          % (abs(conv["extrapolated_error"]) * 100.0,
                             conv.get("observed_order", float("nan")))))
    ax.axhline(5.0, ls=":", lw=1.2, color="k", label="5 % acceptance line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("element size  h = b/ny  [m]")
    ax.set_ylabel("$|p_{y,FE} - p_{y,Navier}| / p_{y,Navier}$  [%]")
    ax.set_title("GAP-PP-3 transition: raw error vs mesh")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_limit(results, path):
    meshes = _meshes(results)
    ub = results["analytic"]["yield_line_upper_pressure"] / 1e6
    labels = [n for n, _ in meshes]
    limits = [m["limit_load_measured"] for _, m in meshes]
    lower = results.get("bounds", {}).get("strip_lower_pressure")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = range(len(labels))
    ax.plot(list(xs), [None if v is None else v / 1e6 for v in limits], "s-", lw=1.8,
            ms=7, color="tab:blue", label="measured limit (path tangent)")
    ax.axhline(ub, color="tab:red", ls="--", lw=1.4, label="yield-line UB (patch)")
    ax.axhspan(0.9 * ub, 1.1 * ub, color="tab:red", alpha=0.12, label="UB $\\pm$ 10 %")
    if lower:
        ax.axhline(lower / 1e6, color="tab:green", ls="-.", lw=1.4, label="strip lower bound")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_xlabel("mesh (coarse $\\to$ fine)")
    ax.set_ylabel("patch pressure  [MPa]")
    ax.set_title("GAP-PP-3 plastic limit load vs plastic bracket")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_paths(results, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, mesh in _meshes(results):
        pts = mesh.get("path") or []      # (load_factor, centre_deflection, max_alpha)
        if not pts:
            continue
        w = [row[1] for row in pts]
        p = [row[0] for row in pts]
        ax.plot([v * 1e3 for v in w], [v / 1e6 for v in p], "o-", lw=1.5, ms=4,
                label=f"{name} (set {mesh['permanent_deflection']*1e3:.2f} mm)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("centre deflection  [mm]")
    ax.set_ylabel("patch pressure  [MPa]")
    ax.set_title("GAP-PP-3 load-deflection paths and permanent set")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
                           "gap_pp3_3grid_results.json"),
         figdir=FIGDIR):
    with open(path) as handle:
        results = json.load(handle)
    os.makedirs(figdir, exist_ok=True)
    made = [
        fig_transition(results, os.path.join(figdir, "gap_pp3_3grid_transition_error.png")),
        fig_limit(results, os.path.join(figdir, "gap_pp3_3grid_limit_bracket.png")),
        fig_paths(results, os.path.join(figdir, "gap_pp3_3grid_paths.png")),
    ]
    for m in made:
        print("wrote", os.path.abspath(m))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
        "gap_pp3_3grid_results.json"))
    ap.add_argument("--figdir", default=FIGDIR)
    args = ap.parse_args()
    main(args.results, args.figdir)