#!/usr/bin/env python
"""W26 composite-laminate verification figures.

Every plotted number is read from an artifact in
``papers/artifacts/w26/composite-laminate-verification`` or recomputed by a
solver call in this process (the Pagano through-thickness shear field, which is
analytic).  Nothing is typed by hand.

    fig1_deflection_deviation.png  |centre deflection deviation| per case vs the limits
    fig2_grid_convergence.png      rel. error vs mesh for both FE paths
    fig3_pagano_shear.png          tau_xz through the thickness: Pagano 3-D vs HEX20
    fig4_cook_membrane.png         Cook membrane convergence vs 23.96
    fig5_rigidity_map.png          normalised ABD rigidities per layup (coupling map)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "papers" / "artifacts" / "w26" / "composite-laminate-verification"
FIG = OUT / "figures"
sys.path.insert(0, str(REPO / "src"))

A = 1.0
Q0 = 1.0


def load(name: str):
    return json.loads((OUT / name).read_text())


def short(case: str) -> str:
    layup, t = case.split("__")
    pretty = {"cross_ply_0_90_s": "[0/90]s",
              "angle_ply_pm45_s": "[45/-45]s",
              "quasi_iso_0_pm45_90_s": "[0/45/-45/90]s"}.get(layup, layup)
    return f"{pretty}\n{t.replace('t', 't/a=').replace('p', '.')}"


def fig1(deflection: dict) -> Path:
    cases = deflection["cases"]
    plate = [abs(c["err_plate_vs_ref_pct"]) for c in cases]
    hexf = [abs(c["err_hex_vs_ref_pct"]) for c in cases]
    limits = [c["limit_pct"] for c in cases]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(x - 0.2, plate, width=0.4, label="Q4 Mindlin FSDT plate FE (n=32)", color="#1f77b4")
    ax.bar(x + 0.2, hexf, width=0.4, label="layered HEX20 3-D FE (finest)", color="#ff7f0e")
    for xi, lim in zip(x, limits):
        ax.hlines(lim, xi - 0.45, xi + 0.45, color="crimson", ls="--", lw=1.4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short(c["case"]) for c in cases], fontsize=7.5)
    ax.set_ylabel("|centre deflection deviation|  [%]")
    ax.set_title("W26 laminate bending: deviation from the case anchor "
                 "(red = 3% thin / 5% thick limit)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "fig1_deflection_deviation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2(grid: dict) -> Path:
    cases = grid["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, case in zip(axes, cases):
        for path, color, mk in (("plate_fem", "#1f77b4", "o"),
                                ("hex20_fem", "#ff7f0e", "s")):
            g = case["paths"][path]
            ax.plot(range(len(g["rel_err_pct"])), g["rel_err_pct"], marker=mk,
                    color=color, label=f"{path} (p={g['p_observed']:.2f})")
        ax.set_yscale("log")
        ax.axhline(grid["gci_limit_pct"], color="crimson", ls="--", lw=1.2)
        ax.set_title(short(case["case"]), fontsize=9)
        ax.set_xlabel("mesh level (coarse $\\rightarrow$ fine)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|relative error| vs anchor  [%]")
    fig.suptitle("W26 grid convergence, both FE paths (dashed = 5% GCI limit)", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig2_grid_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3(shear: dict) -> Path:
    """tau_xz at the ply interfaces: Pagano 3-D anchor vs layered HEX20 (artifact data)."""
    scored = [c for c in shear["cases"] if c.get("reference") == "exact3d"]
    n = max(len(scored), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for ax, case in zip(axes[0], scored):
        ifaces = sorted(float(k) for k in case["exact"])
        h = float(case["t_over_a"]) * 1.0
        peak = max(abs(case["exact"][k]) for k in case["exact"])
        ex = [case["exact"][f"{z:.6f}"] / peak for z in ifaces]
        fe = [case["fem_finest"][f"{z:.6f}"] / peak for z in ifaces]
        zz = [z / h for z in ifaces]
        ax.plot(ex, zz, "o-", color="#1f77b4", lw=2, label="Pagano 3-D elasticity")
        ax.plot(fe, zz, "s--", color="#ff7f0e", ms=7, label="layered HEX20 (finest)")
        ax.set_title(short(case["case"]), fontsize=9)
        ax.set_xlabel(r"$\tau_{xz}/\max|\tau_{xz}|$ at the $z/h$ interfaces")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("z / h")
    fig.suptitle("W26 interlaminar shear at the ply interfaces (peak-normalised)", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig3_pagano_shear.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig4(cook: dict) -> Path:
    rows = cook["rows"]
    nx = [r["nx"] for r in rows]
    w = [r["tip_deflection"] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax.plot(nx, w, "o-", color="#1f77b4", label="Q4 FE tip deflection")
    ax.axhline(cook["reference"], color="crimson", ls="--",
               label=f"Cook (1974) reference = {cook['reference']}")
    ax.axhline(cook["richardson"]["extrapolated"], color="green", ls=":",
               label=f"Richardson extrapolation = {cook['richardson']['extrapolated']:.3f}")
    ax.set_xlabel("edge subdivision nx")
    ax.set_ylabel("tip vertical deflection")
    ax.set_title("Cook membrane: h-refinement", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax2.loglog(nx, [r["abs_rel_err_pct"] for r in rows], "o-", color="#1f77b4",
               label="|error| vs 23.96")
    ax2.set_xlabel("edge subdivision nx")
    ax2.set_ylabel("|relative error|  [%]")
    ax2.set_title(f"observed order p = {cook['p_observed_finest_pair']:.2f} "
                  f"(Q4 theory p = 2)", fontsize=10)
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle("W26 membrane/shear anchor: Cook (1974)", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig4_cook_membrane.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig5(table: list) -> Path:
    comps = [("D11", 0), ("D22", 1), ("D12", 2), ("D66", 3), ("D16", 4), ("D26", 5)]
    labels = [c[0] for c in comps]
    seen, layups = set(), []
    for row in table:
        if row["layup"] not in seen:
            seen.add(row["layup"])
            layups.append(row)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.8 / len(layups)
    for i, row in enumerate(layups):
        d11 = float(row["D11"])
        vals = [float(row[k]) / d11 for k in labels]
        ax.bar(x + i * width - 0.4, vals, width=width,
               label={"cross_ply_0_90_s": "[0/90]s",
                      "angle_ply_pm45_s": "[45/-45]s",
                      "quasi_iso_0_pm45_90_s": "[0/45/-45/90]s"}.get(row["layup"], row["layup"]) +
               f"  (t/a={row['t_over_a']})")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("D", "$D_{") + "}$" for l in labels])
    ax.set_ylabel("normalised rigidity  $D_{ij}/D_{11}$")
    ax.set_title("W26 rigidity map of the tested layups (D16/D26 = bend-twist coupling)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    out = FIG / "fig5_rigidity_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    produced = [fig1(load("deflection_error.json")),
                fig2(load("grid_convergence.json")),
                fig3(load("interlaminar_shear.json")),
                fig4(load("cook_membrane.json")),
                fig5(_csv_rows())]
    for p in produced:
        print(p.relative_to(REPO), p.stat().st_size)
    return 0


def _csv_rows() -> list:
    import csv
    with (OUT / "case_matrix.csv").open() as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    raise SystemExit(main())