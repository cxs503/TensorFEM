#!/usr/bin/env python3
"""Figures for the indentation-plateau mechanism study (zero compute: reads the
three study artifacts plus the anchor certificate)."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from _harness import pyplot  # noqa: E402

FIGDIR = os.path.join(HERE, "..", "..", "..", "docs", "assets", "case-fem-hertz-contact")


def load(path):
    with open(os.path.join(HERE, path)) as fh:
        return json.load(fh)


def main():
    plt = pyplot()
    os.makedirs(FIGDIR, exist_ok=True)
    cert = load("result.json")
    far = load("farfield_discretisation.json")
    zres = load("nearfield_resolution.json")
    conv = load("converged_model.json")
    names = []

    # ---------------------------------------------------------------- fig 1
    # (a) radial ladder (as certified) vs vertical ladder (new).
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=True)
    rad = sorted((r for r in cert["runs"].values()
                  if abs(r["a_over_R"] - 0.05) < 1e-12
                  and abs(r["domain_depth_over_a"] - 96.0) < 1e-9),
                 key=lambda r: -r["h_over_a"])
    h = [r["h_over_a"] for r in rad]
    ax[0].plot(h, [r["error_pct"]["indentation"] for r in rad], "o-", color="C3",
               label=r"radial ladder (certified): $dz=a/6$ fixed")
    ax[0].plot(h, [r["error_pct"]["contact_radius_linear"] for r in rad], "s--",
               color="C0", label=r"radial ladder: $a$")
    vert = [a for a in zres["arms"]]
    dz = [a["near_field_dz_over_a"] for a in vert]
    ax[0].plot(dz, [a["error_pct"]["indentation"] for a in vert], "^-", color="C2",
               label=r"vertical ladder (new): $h/a=0.015$ fixed")
    for a in vert:
        ax[0].annotate(f"{a['n_zfine']}", (a["near_field_dz_over_a"],
                                           a["error_pct"]["indentation"]),
                       textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"spacing in the refined direction  [$/a$]")
    ax[0].set_ylabel(r"relative error of $\delta$ vs Hertz  [%]")
    ax[0].set_title(r"only the vertical spacing moves the plateau ($a/R=0.05$, depth $768a$)")
    ax[0].legend(fontsize=7.5, loc="lower left")

    # (b) far-field refinement: flat -> H_far rejected
    farrows = far["farfield_ladder"]
    x = [r["max_far_vertical_element"] for r in farrows]
    ax[1].plot(x, [r["error_pct"]["indentation"] for r in farrows], "o-", color="C3",
               label=r"$\delta$")
    ax[1].plot(x, [r["error_pct"]["contact_radius_linear"] for r in farrows], "s--",
               color="C0", label=r"$a$")
    ax[1].plot(x, [r["error_pct"]["peak_pressure_consistent"] for r in farrows], "^:",
               color="C2", label=r"$p_0$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"tallest far-field element  [$a$]")
    ax[1].set_ylabel(r"relative error  [%]")
    ax[1].set_title("far-field refinement (26/16 $\\to$ 104/64 layers): flat")
    ax[1].legend(fontsize=8)
    fig.savefig(os.path.join(FIGDIR, "hertz_plateau_mechanism.png"), dpi=150)
    plt.close(fig)
    names.append("hertz_plateau_mechanism.png")

    # ---------------------------------------------------------------- fig 2
    # error of the three gated/recorded quantities for every mechanism arm.
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    labels, ea, ed, ep = [], [], [], []
    for a in zres["arms"]:
        labels.append(f"dz=a/{int(round(1.0 / a['near_field_dz_over_a']))}")
        ea.append(a["error_pct"]["contact_radius_linear"])
        ed.append(a["error_pct"]["indentation"])
        ep.append(a["error_pct"]["peak_pressure_consistent"])
    if conv.get("vertical_ladder_a48"):
        r = conv["vertical_ladder_a48"]
        labels.append(f"dz=a/{int(round(1.0 / r['near_field_dz_over_a']))}")
        ea.append(r["error_pct"]["contact_radius_linear"])
        ed.append(r["error_pct"]["indentation"])
        ep.append(r["error_pct"]["peak_pressure_consistent"])
    import numpy as np
    idx = np.arange(len(labels))
    w = 0.26
    ax.bar(idx - w, ea, w, label=r"$a$ (geometric estimator)", color="C0")
    ax.bar(idx, ed, w, label=r"$\delta$", color="C3")
    ax.bar(idx + w, ep, w, label=r"$p_0$ (consistent)", color="C2")
    ax.axhline(0.0, color="k", lw=0.8)
    for s in (3.0, -3.0):
        ax.axhline(s, ls="--", color="0.5", lw=1)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("relative error vs Hertz  [%]")
    ax.set_title(r"anisotropy of the near field: refining only $dz$ ($h/a=0.015$, depth $768a$)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGDIR, "hertz_plateau_arms.png"), dpi=150)
    plt.close(fig)
    names.append("hertz_plateau_arms.png")

    # ---------------------------------------------------------------- fig 3
    # depth ladder: certified (dz=a/6) vs vertical-refined (dz=a/24) + limits
    fig, ax = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    zr = conv["depth_ladder_dz_a24"]
    ax.plot([r["depth_over_a"] for r in zr], [r["error_pct"]["indentation"] for r in zr],
            "o-", color="C2", label=r"$\delta$, $dz=a/24$ (this study)")
    ax.plot([r["depth_over_a"] for r in zr],
            [r["error_pct"]["contact_radius_linear"] for r in zr], "s--", color="C0",
            label=r"$a$, $dz=a/24$")
    prev = load("truncation_extrapolation.json")
    lad = prev.get("ladders") or prev.get("runs") or {}
    ax.axhline(-0.4278, ls=":", color="0.45",
               label=r"prior limit, $dz=a/6$: $-0.428\%$")
    e_inf = conv["extrapolation"]["e_inf_pct"]
    if e_inf is not None:
        ax.axhline(e_inf, ls="-", lw=1.2, color="C2",
                   label=rf"this study limit: ${e_inf:+.3f}\%$")
    ax.axhline(3.0, ls="--", color="0.6", lw=1)
    ax.axhline(-3.0, ls="--", color="0.6", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"half-space depth / $a$")
    ax.set_ylabel(r"relative error vs Hertz  [%]")
    ax.set_title(r"depth extrapolation with the vertical-refined near field ($a/R=0.05$)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGDIR, "hertz_plateau_depth.png"), dpi=150)
    plt.close(fig)
    names.append("hertz_plateau_depth.png")

    print("figures:", names)
    return 0


if __name__ == "__main__":
    sys.exit(main())