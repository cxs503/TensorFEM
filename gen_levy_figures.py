"""Figures for the elastic-restraint Levy verification (anchor GAP-PP-4).

Produces two PNGs in ``benchmarks/artifacts/``:

``beta_levy_band.png``   (a) normalised centre deflection vs beta: exact Levy
                         curve vs Mindlin FEM points on three meshes, at
                         *interior* beta values, not just the two ends;
                         (b) discretisation error of the FEM vs the exact
                         reference on a log-log grid with a 2:1 reference slope.
``beta_levy_field.png``  (c) centre bending moment along the sprung centreline,
                         FEM vs exact Levy (two meshes) at beta = 0.5, and
                         (d) the Mx field of the FEM panel at beta = 0.5 showing
                         how the moment redistributes towards the frames.

Run: ``python gen_levy_figures.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.plate_elastic_restraint import (  # noqa: E402
    beta_to_zeta,
    moments_at_nodes,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import flexural_rigidity, levy_restrained_plate  # noqa: E402

ART = ROOT / "benchmarks" / "artifacts"
DATA = json.loads((ART / "beta_levy_band_verification.json").read_text())

A = B = 1.0
E, NU, T, Q = 200.0e9, 0.3, 0.01, 1000.0
D = flexural_rigidity(E, NU, T)


def _exact(beta, n_modes=801):
    K = beta_to_zeta(beta) * D / A
    return levy_restrained_plate(A, B, D, NU, K, Q, n_modes=n_modes)


def figure_band():
    rows = DATA["canonical_levy"]["rows"]
    betas = np.linspace(0.0, 1.0, 41)
    w_ex = np.array([_exact(b).w_centre() for b in betas]) * D / (Q * A**4)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.plot(betas, w_ex, "-", color="k", lw=1.8, label="exact Levy series")
    styles = {10: ("o", "#1f77b4"), 20: ("s", "#ff7f0e"), 40: ("^", "#2ca02c")}
    for nx, (mk, col) in styles.items():
        bw = [r["beta"] for r in rows]
        ww = [next(f for f in r["fem"] if f["nx"] == nx)["w_centre_m"] * D / (Q * A**4) for r in rows]
        ax.plot(bw, ww, mk, ms=6, color=col, mfc="none", label=f"Mindlin FEM {nx}x{nx}")
    ax.axhline(0.00406, ls=":", color="0.4", lw=1)
    ax.text(0.02, 0.00411, "Timoshenko SSSS 0.00406", fontsize=7, color="0.35")
    ax.set_xlabel(r"embedding coefficient $\beta$ (0 = simple, 1 = clamped)")
    ax.set_ylabel(r"$w_c\,D/(q a^4)$")
    ax.set_title(r"(a) centre deflection across the whole band ($\beta_y=0$)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        nx = [f["nx"] for f in r["fem"]]
        err = [abs(f["err_w_pct"]) for f in r["fem"]]
        ax.loglog(nx, err, "o-", ms=5, lw=1.2, label=rf"$\beta$ = {r['beta']:.1f}")
    h = np.array([10.0, 40.0])
    ax.loglog(h, 0.28 * (h / 10.0) ** -2, "k--", lw=1, label=r"$O(h^2)$ reference")
    ax.set_xlabel("elements per side $n$")
    ax.set_ylabel(r"$|w_{FEM}/w_{exact}-1|$  [%]")
    ax.set_title("(b) error vs exact Levy reference")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Elastic rotational restraint: Mindlin FEM vs exact Levy solution "
        "(1 m x 1 m x 10 mm plate, $\\nu$=0.3, uniform load)",
        fontsize=10,
    )
    fig.tight_layout()
    out = ART / "beta_levy_band.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def figure_field():
    beta = 0.5
    K = beta_to_zeta(beta) * D / A
    exact = _exact(beta, n_modes=801)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    xs = np.linspace(0.0, A, 41)
    ax.plot(xs, [exact.moments(x, B / 2)[0] for x in xs], "-", color="k", lw=1.8,
            label="exact Levy (Kirchhoff)")
    for n, col in ((20, "#ff7f0e"), (40, "#2ca02c")):
        res = solve_restrained_panel(n, n, A, B, E, NU, T, Q, beta_x=beta, beta_y=0.0)
        nodes = res.nodes.numpy()
        mom = moments_at_nodes(res).numpy()[:, 0]
        row = np.abs(nodes[:, 1] - B / 2) < 1e-9
        order = np.argsort(nodes[row, 0])
        ax.plot(nodes[row, 0][order], mom[row][order], "o", ms=3.5, mfc="none",
                color=col, label=f"Mindlin FEM {n}x{n}")
    ax.set_xlabel("$x$ along the sprung centreline [m]")
    ax.set_ylabel(r"$M_x$ [N$\cdot$m/m]")
    ax.set_title(r"(c) centre bending moment, $\beta=0.5$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    res = solve_restrained_panel(40, 40, A, B, E, NU, T, Q, beta_x=beta, beta_y=0.0)
    nodes = res.nodes.numpy()
    mom = moments_at_nodes(res).numpy()[:, 0].reshape(41, 41)
    cs = ax.contourf(nodes[:, 0].reshape(41, 41), nodes[:, 1].reshape(41, 41), mom,
                     24, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=r"$M_x$ [N$\cdot$m/m]")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(r"(d) $M_x$ field, FEM 40x40, $\beta=0.5$ (moment migrates to the frames)")
    ax.set_aspect("equal")

    fig.suptitle(
        r"Rotational restraint $\beta=0.5$: field-level agreement with the exact solution",
        fontsize=10,
    )
    fig.tight_layout()
    out = ART / "beta_levy_field.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    figure_band()
    figure_field()