#!/usr/bin/env python3
"""Figures for the physically calibrated restraint band (anchor GAP-PP-4).

Reads ``benchmarks/artifacts/beta_stiffener_calibration.json`` only -- no
recomputation -- and writes three PNGs next to it.

Run:  python gen_stiffener_calibration_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "benchmarks" / "artifacts"
DATA = json.loads((ART / "beta_stiffener_calibration.json").read_text())


def fig_calibration() -> Path:
    """beta and mobilised moment fixity vs the spacing of the fixing members."""
    rows = DATA["sensitivity"]
    spans = sorted({r["support_span"] for r in rows})
    labels = sorted({r["longitudinal"] for r in rows})
    dp = DATA["design_point_summary"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for lab in labels:
        sel = sorted((r for r in rows if r["longitudinal"] == lab and r["frame"] == "FB 200x12"),
                     key=lambda r: r["support_span"])
        ax.plot([r["support_span"] for r in sel], [r["beta_x"] for r in sel], "o-", label=f"longitudinal {lab}")
        ax2.plot([r["support_span"] for r in sel], [r["moment_fixity_x"] for r in sel], "o-", label=f"{lab}")

    ax.axhline(0.0, color="0.6", ls="--", lw=1)
    ax.axhline(1.0, color="0.6", ls="--", lw=1)
    ax.text(spans[0], 0.03, "simply supported (beta = 0)", color="0.35", fontsize=8)
    ax.text(spans[0], 0.93, "clamped idealisation (beta = 1)", color="0.35", fontsize=8)
    ax.plot([0.8], [dp["beta_x"]], "k*", ms=16, label="ice-belt design point")
    ax.plot([0.8], [dp["beta_y"]], "kP", ms=9, label="frames (design point)")
    ax.set_xlabel("spacing of the torsionally fixing members  L  [m]")
    ax.set_ylabel(r"embedding coefficient  $\beta$")
    ax.set_title(r"(a) $\beta$ from real scantlings: never at either idealisation")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax2.plot([0.8], [dp["moment_fixity_x"]], "k*", ms=16)
    ax2.plot([0.8], [dp["moment_fixity_y"]], "kP", ms=9)
    ax2.set_xlabel("spacing of the torsionally fixing members  L  [m]")
    ax2.set_ylabel("mobilised fraction of clamped edge moment")
    ax2.set_title(r"(b) The physical quantity: moment fixity $\neq \beta$")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = ART / "beta_stiffener_calibration_band.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_field() -> Path:
    """von Mises bending stress field at the calibrated restraint."""
    f = DATA["field"]
    x = np.asarray(f["x"])
    y = np.asarray(f["y"])
    s = np.asarray(f["sigma_vm_MPa"])

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    cs = ax.contourf(x, y, s, levels=24, cmap="viridis")
    ax.contour(x, y, s, levels=10, colors="k", linewidths=0.35, alpha=0.5)
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(r"von Mises bending stress  $\sigma_{vM}$  [MPa]")
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]  (span between longitudinals)")
    ax.set_ylabel("y  [m]  (span between transverse frames)")
    ax.set_title(
        "Calibrated restraint  $\\beta_x$=%.3f (longitudinals),  $\\beta_y$=%.3f (frames)\n"
        "uniform pressure 1 MPa, $\\sigma_{vM}^{max}$=%.1f MPa"
        % (f["beta_x"], f["beta_y"], f["max_MPa"]),
        fontsize=10,
    )
    fig.tight_layout()
    out = ART / "beta_stiffener_calibration_field.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_band() -> Path:
    """Centre stress vs beta: FE band, exact Levy anchors, calibrated design point."""
    b = DATA["band"]
    rows = b["rows"]
    betas = [r["beta"] for r in rows]
    sig = [r["sigma_centre_MPa"] for r in rows]
    ex = b["exact_levy_check"]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(betas, sig, "o-", color="C0", label="Mindlin FE panel (40 x 46), $\\beta_y$=0")
    ax.plot([r["beta"] for r in ex], [r["sigma_centre_MPa"] for r in ex], "x",
            color="C3", ms=11, mew=2, label="exact Kirchhoff Levy at the same $K$")
    ax.axhline(b["sigma_at_calibrated_both_edges_MPa"], color="C2", ls="-.",
               label="calibrated design point, both edges (%.1f MPa)" % b["sigma_at_calibrated_both_edges_MPa"])
    ax.axvline(b["calibrated_beta"], color="C2", ls=":", lw=1)
    ax.plot([b["calibrated_beta"]], [b["sigma_at_calibrated_beta_MPa"]], "k*", ms=16,
            label=r"calibrated $\beta_x$ = %.3f (longitudinals only)" % b["calibrated_beta"])
    ax.annotate("", xy=(1.0, b["sigma_clamped_MPa"]), xytext=(0.0, b["sigma_SSSS_MPa"]),
                arrowprops=dict(arrowstyle="<->", color="0.6", lw=1))
    ax.set_xlabel(r"embedding coefficient  $\beta$")
    ax.set_ylabel("centre bending stress  [MPa]")
    ax.set_title(
        "SSSS  %.1f MPa  ->  clamped  %.1f MPa  (-%.1f%%);\n"
        "real restraint delivers only -%.1f%% at this design point"
        % (b["sigma_SSSS_MPa"], b["sigma_clamped_MPa"], b["reduction_vs_SSSS_pct"],
           b["reduction_both_edges_pct"]),
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = ART / "beta_stiffener_calibration_stress_band.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (fig_calibration(), fig_field(), fig_band()):
        print("wrote", path)