#!/usr/bin/env python3
"""Figures for the out-of-sample Hertz verification.

Reads (never recomputes) ``gates_outofsample.json`` and
``probes/profile_aR0075.json``; writes PNGs into
``docs/assets/case-fem-hertz-contact/``:

* ``hertz_oos_error_vs_ratio.png`` -- |a| and |delta| error vs a/R on both mesh
  levels, with the 3% pool band, the certified band and the 0.02 control.
* ``hertz_oos_pressure_aR0075.png`` -- recovered contact pressure at the
  interpolation point a/R = 0.075 vs the Hertz ellipse.

Run:
    python3 benchmarks/verified/fem_hertz_contact_outofsample/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "docs", "assets",
                                   "case-fem-hertz-contact"))
GATES = os.path.join(HERE, "gates_outofsample.json")
PROFILE = os.path.join(HERE, "probes", "profile_aR0075.json")

COLORS = {"a": "#1f77b4", "delta": "#d62728"}


def error_figure(doc):
    rows = doc["rows"]
    ratios = sorted({r["a_over_R"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, qty, label in (
        (axes[0], "contact_radius_linear", "contact radius  a  (linear zero-crossing)"),
        (axes[1], "indentation", "indentation  $\\delta$  (load-control unknown)"),
    ):
        for h, style, marker in ((0.0075, "-o", "o"), (0.03, "--s", "s")):
            xs, ys = [], []
            for ratio in ratios:
                for r in rows:
                    if abs(r["a_over_R"] - ratio) < 1e-12 and abs(r["h_over_a"] - h) < 1e-12:
                        xs.append(ratio)
                        ys.append(r["error_pct"][qty])
            ax.plot(xs, ys, style, marker=marker, lw=1.6, ms=5,
                    label=f"h/a = {h:g}")
        ax.axhspan(-3, 3, color="0.85", zorder=0, label="pool band $\\pm$3%")
        ax.axvspan(0.05, 0.10, color="#c7e9c0", alpha=0.45, zorder=0,
                   label="certified band 0.05..0.10")
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0.02, color="#e6550d", lw=1.2, ls=":", label="0.02 control (documented failure)")
        ax.set_xlabel("a/R (Hertz)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error vs Hertz closed form [%]")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Case pool FEM-HERTZ-CONTACT: out-of-sample error of the "
                 "load-controlled FE face", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, "hertz_oos_error_vs_ratio.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def pressure_figure(prof):
    s = prof["surface"]
    x = np.asarray(s["r_over_a"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(x, s["analytic_over_p0"], "k-", lw=1.8,
            label="Hertz ellipse $\\sqrt{1-(r/a)^2}$")
    ax.plot(x, s["pressure_consistent_over_p0"], "-", color="#1f77b4", lw=1.4,
            label="FE, mass-matrix consistent")
    ax.plot(x, s["pressure_lumped_over_p0"], ":", color="#ff7f0e", lw=1.2,
            label="FE, lumped (ring area)")
    ax.set_xlabel("r / a")
    ax.set_ylabel("p / p$_0$")
    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1.15)
    ax.set_title("Surface pressure at the out-of-sample point a/R = 0.075 "
                 f"(h/a = {prof['point']['h_over_a']:g})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "hertz_oos_pressure_aR0075.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(GATES) as fh:
        doc = json.load(fh)
    print(error_figure(doc))
    if os.path.exists(PROFILE):
        with open(PROFILE) as fh:
            prof = json.load(fh)
        print(pressure_figure(prof))


if __name__ == "__main__":
    main()