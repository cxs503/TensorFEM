"""Sandwich-panel bending benchmark: every model vs the layered 3-D solution.

Produces the tables and figures of ``docs/sandwich-panel-bending.md``:

  python examples/sandwich_panel_benchmark.py --out artifacts/sandwich_panel

Outputs (JSON + PNG, all numbers programmatic -- nothing is typed in by hand):
  results.json          every deflection/stress/error of the sweep
  fig_deflection.png    centre deflection vs a/h, all models + 3-D reference
  fig_errors.png        relative error of each model vs the 3-D reference
  fig_through_thickness.png  sigma_xx and tau_xz through the thickness
  fig_core_stiffness.png     error vs core shear modulus (the validity map)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.composite_laminate_bending import Ply3D
from tensorfem.sandwich_panel import (
    BALSA_CORE,
    ELASTIC_FACE,
    PVC_CORE,
    SandwichPanel,
    deflection_budget,
)


def _scaled(core: Ply3D, factor: float) -> Ply3D:
    return Ply3D(E1=core.E1 * factor, E2=core.E2 * factor, E3=core.E3 * factor,
                 nu12=core.nu12, nu13=core.nu13, nu23=core.nu23,
                 G12=core.G12 * factor, G13=core.G13 * factor, G23=core.G23 * factor)


def run(outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    panel = SandwichPanel(face=ELASTIC_FACE, core=PVC_CORE, t_face=2e-3, t_core=40e-3)
    h = panel.thickness

    # ── sweep 1: span (thick -> thin) ──────────────────────────────────────
    spans = [2.0 * h, 3.0 * h, 4.5 * h, 9.0 * h, 18.0 * h, 40.0 * h, 60.0 * h]
    span_rows = [deflection_budget(panel, a) for a in spans]

    # ── sweep 2: core stiffness (foam -> balsa-like) ───────────────────────
    a_ref = 18.0 * h
    core_rows = []
    for factor in (1.0, 2.0, 5.0, 10.0, 30.0, 100.0):
        p = SandwichPanel(face=ELASTIC_FACE, core=_scaled(PVC_CORE, factor),
                          t_face=2e-3, t_core=40e-3)
        row = deflection_budget(p, a_ref)
        row["core_shear_modulus"] = p.core_shear_modulus()
        row["G_ratio"] = p.core_shear_modulus() / ELASTIC_FACE.G12
        core_rows.append(row)

    # ── sweep 3: face thickness at fixed core ─────────────────────────────
    face_rows = []
    for t_f in (1e-3, 2e-3, 4e-3, 8e-3):
        p = SandwichPanel(face=ELASTIC_FACE, core=PVC_CORE, t_face=t_f, t_core=40e-3)
        row = deflection_budget(p, 18.0 * p.thickness)
        row["t_face"] = t_f
        face_rows.append(row)

    # ── through-thickness fields of the reference and of the engineering model
    ex = panel.exact3d(a_ref)
    z = np.linspace(-0.5 * h + 1e-9, 0.5 * h - 1e-9, 241)
    sig = np.array([ex.stress_amplitudes(float(zz))[0] for zz in z])
    tau = np.array([ex.face_shear(float(zz), 0.0, 0.5 * a_ref) for zz in z])
    cl = panel.classical(a_ref)

    result = {
        "panel": {"t_face": panel.t_face, "t_core": panel.t_core,
                  "thickness": h, "d": panel.d,
                  "E_face": ELASTIC_FACE.E1, "G_core": panel.core_shear_modulus(),
                  "D_classic_11": float(panel.classic_D()[0, 0]),
                  "D_Q": panel.classic_shear_rigidity(),
                  "As_core_only": panel.core_only_shear_rigidity()},
        "span_sweep": span_rows,
        "core_sweep": core_rows,
        "face_sweep": face_rows,
        "through_thickness": {"z": z.tolist(), "sigma_xx": sig.tolist(),
                              "tau_xz": tau.tolist()},
        "classical_reference": cl["w_center"],
        "classical_bending_share": cl["w_bending"] / cl["w_center"],
        "classical_shear_share": cl["w_shear"] / cl["w_center"],
    }
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)

    # ── figure 1: deflection vs span ──────────────────────────────────────
    ah = [r["a_over_h"] for r in span_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(ah, [r["exact3d"] for r in span_rows], "k-o", label="layered 3-D elasticity (ref)")
    ax.loglog(ah, [r["classical"] for r in span_rows], "r--s", label="antiplane-core (Allen)")
    ax.loglog(ah, [r["fsdt_sandwich"] for r in span_rows], "b-.^", label="FSDT, $D_Q=G_c d^2/t_c$")
    ax.loglog(ah, [r["fsdt_k56"] for r in span_rows], "g:v", label="FSDT, $k=5/6$ (homogeneous)")
    ax.loglog(ah, [r["clt"] for r in span_rows], "m:x", label="CLT (no shear)")
    ax.set_xlabel("span / thickness  $a/h$")
    ax.set_ylabel("centre deflection  $w$  [m]")
    ax.set_title("Sandwich panel (2/40/2 mm, PVC H80 core): bending models vs 3-D elasticity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_deflection.png"), dpi=160)
    plt.close(fig)

    # ── figure 2: errors vs span ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, style, label in (
            ("err_classical", "r--s", "antiplane-core (Allen)"),
            ("err_fsdt_sandwich", "b-.^", "FSDT $D_Q=G_c d^2/t_c$"),
            ("err_fsdt_core", "c-*", "FSDT $A_s=G_c t_c$ (core only)"),
            ("err_fsdt_k56", "g:v", "FSDT $k=5/6$"),
            ("err_clt", "m:x", "CLT")):
        ax.semilogx(ah, [100.0 * r[key] for r in span_rows], style, label=label)
    ax.axhspan(-5, 5, color="0.85", zorder=0, label="+-5 % band")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("span / thickness  $a/h$")
    ax.set_ylabel("centre-deflection error vs 3-D  [%]")
    ax.set_title("Sandwich panel: model error against the layered 3-D reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_errors.png"), dpi=160)
    plt.close(fig)

    # ── figure 3: through-thickness fields ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    axes[0].plot(sig / 1e6, (z + 0.5 * h) * 1e3, "k-", label="3-D elasticity")
    axes[0].axvline(cl["sigma_face_top_x"] / 1e6, color="r", ls="--",
                    label="Allen: $\\mp M/(d\\,t_f)$")
    axes[0].set_xlabel("$\\sigma_{xx}$ amplitude  [MPa]")
    axes[0].set_ylabel("height above bottom face  [mm]")
    axes[0].set_title("In-plane stress through the thickness")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(-tau / 1e3, (z + 0.5 * h) * 1e3, "k-", label="3-D elasticity (sign flipped)")
    axes[1].axvline(cl["tau_core_xz_center"] / 1e3, color="r", ls="--",
                    label="Allen: $Q/d$")
    axes[1].set_xlabel("$\\tau_{xz}$ amplitude  [kPa]")
    axes[1].set_ylabel("height above bottom face  [mm]")
    axes[1].set_title("Core shear through the thickness")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("a/h = %.1f, q0 = 1 Pa (sine mode): 3-D fields vs the engineering model"
                 % (a_ref / h), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_through_thickness.png"), dpi=160)
    plt.close(fig)

    # ── figure 4: validity map in core stiffness ─────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    gr = [r["G_ratio"] for r in core_rows]
    for key, style, label in (
            ("err_classical", "r--s", "antiplane-core (Allen)"),
            ("err_fsdt_sandwich", "b-.^", "FSDT $D_Q=G_c d^2/t_c$"),
            ("err_fsdt_k56", "g:v", "FSDT $k=5/6$"),
            ("err_clt", "m:x", "CLT")):
        ax.semilogx(gr, [100.0 * r[key] for r in core_rows], style, label=label)
    ax.axhspan(-5, 5, color="0.85", zorder=0)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("core shear modulus / face shear modulus   $G_c/G_f$")
    ax.set_ylabel("centre-deflection error vs 3-D  [%]")
    ax.set_title("Validity map: a/h = 18, $t_f$ = 2 mm, $t_c$ = 40 mm")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_core_stiffness.png"), dpi=160)
    plt.close(fig)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/sandwich_panel")
    args = ap.parse_args()
    res = run(args.out)
    print(json.dumps({"panel": res["panel"],
                      "classical_bending_share": res["classical_bending_share"],
                      "classical_shear_share": res["classical_shear_share"]}, indent=1))
    for row in res["span_sweep"]:
        print("a/h=%5.1f  err: clt %+7.1f%%  k56 %+7.1f%%  core %+6.1f%%  DQ %+5.2f%%  allen %+5.2f%%"
              % (row["a_over_h"], 100 * row["err_clt"], 100 * row["err_fsdt_k56"],
                 100 * row["err_fsdt_core"], 100 * row["err_fsdt_sandwich"],
                 100 * row["err_classical"]))
    print("figures ->", args.out)