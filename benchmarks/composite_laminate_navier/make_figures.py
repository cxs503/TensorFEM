#!/usr/bin/env python
"""Figures for the W23 composite-laminate verification report.

Reads the artifacts written by ``run_case_matrix.py`` and renders:

    fig1_error_bars.png       deflection deviation per case vs the thresholds
    fig2_grid_convergence.png observed order / GCI per case and FE path
    fig3_shear_profile.png    through-thickness tau_xz: 3-D elasticity vs layered HEX20
    fig4_stiffness.png        D-matrix of the three layup families (coupling map)

Every plotted number comes from the JSON artifacts -- nothing is typed in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGDIR = REPO / "docs" / "assets" / "w23_composite_laminate"
sys.path.insert(0, str(REPO / "src"))


def load(name: str):
    return json.loads((HERE / name).read_text())


def fig_error_bars(deflection: dict) -> Path:
    cases = deflection["cases"]
    labels = [f"{c['case'].split('__')[0].replace('_', ' ')}\n{c['case'].split('__')[1]}"
              for c in cases]
    plate = [abs(c["err_plate_vs_ref_pct"]) for c in cases]
    hexf = [abs(c["err_hex_vs_ref_pct"]) for c in cases]
    limits = [c["limit_pct"] for c in cases]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(cases))
    ax.bar(x - 0.2, plate, width=0.4, label="Q4 Mindlin FSDT plate FE (n=32)", color="#1f77b4")
    ax.bar(x + 0.2, hexf, width=0.4, label="layered HEX20 3-D FE (finest)", color="#ff7f0e")
    for xi, lim in zip(x, limits):
        ax.hlines(lim, xi - 0.45, xi + 0.45, color="crimson", linestyle="--", linewidth=1.4)
    ax.text(len(cases) - 0.6, limits[-1] * 1.06, "criterion limit (3% thin / 5% thick)",
            color="crimson", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("|centre deflection deviation|  [%]")
    ax.set_title("W23 laminate bending: deviation from the case anchor (log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGDIR / "fig1_error_bars.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_grid_convergence(grid: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, path, title in ((axes[0], "plate_fem", "Q4 Mindlin plate FE"),
                            (axes[1], "hex20_fem", "layered HEX20 3-D FE")):
        for entry in grid["cases"]:
            data = entry["paths"][path]
            err = [abs(e) for e in data["rel_err_pct"]]
            if data["mesh"] and isinstance(data["mesh"][0], int):
                h = [1.0 / m for m in data["mesh"]]
            else:
                h = [1.0 / int(m.split("x")[0]) for m in data["mesh"]]
            ax.loglog(h, err, "o-", label=f"{entry['case']} (p={data['p_observed']:.2f})",
                      markersize=3.5, linewidth=1.1)
        hh = np.array([1 / 40.0, 1 / 3.0])
        ax.loglog(hh, 3e-3 * (hh / hh[0]) ** 2, "k--", linewidth=1.0,
                  label="reference slope h$^2$")
        ax.set_xlabel("normalised element size h")
        ax.set_ylabel("relative error  [%]")
        ax.set_title(title)
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("W23 grid-convergence study (criterion: GCI < 5%, order vs theory p = 2)")
    fig.tight_layout()
    out = FIGDIR / "fig2_grid_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_shear_profile() -> Path:
    """tau_xz through the thickness: 3-D elasticity vs layered HEX20 (cross-ply)."""
    from tensorfem.composite_laminate_bending import (
        PAGANO_MATERIAL, Laminate, LayeredExact3D, layered_hex20_fem,
    )

    lam = Laminate([0, 90, 90, 0], 0.125, PAGANO_MATERIAL)         # t/a = 0.5
    ex = LayeredExact3D(lam, 1.0, 1.0)
    z = np.linspace(-0.5 * lam.thickness, 0.5 * lam.thickness, 121)
    tau_exact = ex.tau_xz_profile(z, x=0.0, y=0.5)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(tau_exact, z, "k-", linewidth=2.0, label="layered 3-D elasticity (exact)")
    for nx, nz_pp, style in ((4, 2, "o--"), (6, 3, "s-.")):
        fe = layered_hex20_fem(lam, a=1.0, q0=1.0, nx=nx, nz_per_ply=nz_pp)
        zz = np.array([p[0] for p in fe["tau_xz_profile"]])
        tt = np.array([p[1] for p in fe["tau_xz_profile"]])
        ax.plot(tt, zz, style, markersize=4,
                label=f"HEX20 {nx}x{nx}x{lam.n_plies * nz_pp} (nz/ply={nz_pp})")
    for zi in lam.z_interfaces[1:-1]:
        ax.axhline(zi, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"$\tau_{xz}(0, b/2, z)$  [Pa]")
    ax.set_ylabel("z  [m]")
    ax.set_title("Interlaminar shear, [0/90]s laminate, t/a = 0.5\n"
                 "(dotted lines: ply interfaces)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "fig3_shear_profile.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_stiffness(case_matrix: Path) -> Path:
    import csv

    from tensorfem.composite_laminate_bending import (
        PAGANO_MATERIAL, Laminate, symmetric_layup,
    )

    layups = {"[0/90]s": [0, 90, 90, 0],
              "[+-45]s": symmetric_layup([45, -45]),
              "QI [0/+-45/90]s": symmetric_layup([0, 45, -45, 90])}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (name, angles) in zip(axes, layups.items()):
        D = Laminate(angles, 1.25e-3, PAGANO_MATERIAL).d_matrix()
        Dn = D / D[0, 0]
        im = ax.imshow(Dn, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["11", "22", "12"])
        ax.set_yticklabels(["11", "22", "12"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{Dn[i, j]:+.3f}", ha="center", va="center", fontsize=9)
        ax.set_title(name)
        ax.set_xlabel("D column")
        ax.set_ylabel("D row")
    fig.colorbar(im, ax=axes, shrink=0.8, label="D$_{ij}$ / D$_{11}$")
    fig.suptitle("Laminate bending stiffness (normalised): the +-45 and quasi-isotropic "
                 "families carry D$_{16}$, D$_{26}$\n-> the plate element must use the full 3x3 D")
    out = FIGDIR / "fig4_stiffness.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    deflection = load("deflection_error.json")
    grid = load("grid_convergence.json")
    outs = [fig_error_bars(deflection), fig_grid_convergence(grid),
            fig_shear_profile(), fig_stiffness(HERE / "case_matrix.csv")]
    for out in outs:
        print(out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())