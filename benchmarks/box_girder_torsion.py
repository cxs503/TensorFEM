"""Evidence generator: closed thin-walled (box girder) torsion verification.

Produces, programmatically (no hand-copied numbers):

* a finite-difference grid study of the multiply connected Prandtl problem,
* the thin-wall asymptote against Bredt,
* a HEX8 (merged solid3d path) mesh study against the finite-difference field,
* figures (field contours + convergence) and a JSON evidence bundle.

Usage::

    python3 benchmarks/box_girder_torsion.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.box_torsion import (RectBoxSection, box_torsion_run,
                                   solve_box_prandtl)

A, B, T = 1.0, 0.6, 0.1
G, THETA = 1.0, 1.0
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"


def _rel(a, b):
    return 100.0 * abs(a - b) / abs(b)


def _wall_mean(box, frac=0.8):
    """Mean |tau| on the long wall, away from the singular inner corners."""
    x, y, txz, tyz = box.tau_cells
    band = (x > A - T) & (np.abs(y) < frac * (B - T))
    return float(np.hypot(txz[band], tyz[band]).mean())


def fd_study():
    rows = []
    for h in (0.05, 0.025, 0.0125):
        box = solve_box_prandtl(A, B, T, h, G, THETA)
        bredt = RectBoxSection(A, B, T).j
        rows.append({
            "h": h, "nodes": int(box.material.sum()), "sweeps": int(box.sweeps),
            "residual": float(box.residual),
            "compat_residual": float(box.compat_residual),
            "c": float(box.c),
            "j_energy": float(box.j_energy),
            "j_torque": float(box.j_torque),
            "identity_gap_pct": _rel(box.j_energy, box.j_torque),
            "j_bredt": float(bredt),
            "j_vs_bredt_pct": _rel(box.j_torque, bredt),
            "tau_max_fd": float(box.tau_max),
            "tau_bredt": float(RectBoxSection(A, B, T, G, THETA).tau_max),
            "tau_wall_fd": _wall_mean(box),
        })
    for i in range(len(rows) - 1):
        d1 = abs(rows[i]["j_torque"] - rows[i + 1]["j_torque"])
        rows[i]["richardson_delta"] = float(d1)
    d1 = rows[0]["j_torque"] - rows[1]["j_torque"]
    d2 = rows[1]["j_torque"] - rows[2]["j_torque"]
    order = math.log2(abs(d1 / d2)) if d2 else float("inf")
    return rows, order


def thin_wall_study():
    rows = []
    for t in (0.2, 0.1, 0.05):
        h = t / 4.0
        box = solve_box_prandtl(A, B, t, h, G, THETA)
        bredt = RectBoxSection(A, B, t).j
        rows.append({"t": t, "h": h, "j_fd": float(box.j_torque),
                     "j_bredt": float(bredt),
                     "ratio": float(box.j_torque / bredt),
                     "error_pct": _rel(box.j_torque, bredt)})
    return rows


def fe_study(reference):
    rows = []
    for (n_side, n_layers, layers_z) in ((4, 2, 2), (8, 3, 2), (8, 4, 3),
                                         (12, 4, 3)):
        res = box_torsion_run(A, B, T, n_side=n_side, n_layers=n_layers,
                              layers_z=layers_z, young_modulus=2.6,
                              poisson_ratio=0.3, twist=THETA)
        rows.append({
            "n_side": n_side, "n_layers": n_layers, "layers_z": layers_z,
            "nodes": res.nodes, "elements": res.elements, "dofs": res.dofs,
            "j_fe": float(res.j_fe), "j_bredt": float(res.j_exact),
            "j_ratio_bredt": float(res.j_ratio),
            "error_vs_bredt_pct": _rel(res.j_fe, res.j_exact),
            "error_vs_fd_pct": _rel(res.j_fe, reference),
            "warping_max": float(res.warping_max),
            "shear_wall_fe": float(res.shear_wall_fe),
            "tau_max_fe": float(res.tau_max_fe),
        })
    return rows


def figures(fd_rows, fe_rows):
    ASSETS.mkdir(parents=True, exist_ok=True)
    box = solve_box_prandtl(A, B, T, 0.025, G, THETA)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    extent = (-A, A, -B, B)
    im = axes[0].imshow(np.where(box.material, box.phi, np.nan), origin="lower",
                        extent=extent, cmap="viridis")
    axes[0].set_title(r"Prandtl $\phi$ (FD, h=0.025)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    gx, gy = box._cell_gradients()
    full = box._full_cells()
    tau = np.where(full, np.hypot(gx, gy), np.nan)
    im = axes[1].imshow(tau, origin="lower", extent=extent, cmap="inferno")
    axes[1].set_title(r"$|\tau|=\sqrt{\tau_{xz}^2+\tau_{yz}^2}$ (FD)")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    X, Y = box._cell_centres()
    band = full & (X > A - T) & (np.abs(Y) < B - T)
    axes[2].plot(np.sort(tau[band]), ls="", marker=".", ms=3,
                 label="FD cell centres")
    axes[2].axhline(RectBoxSection(A, B, T, G, THETA).tau_max, color="k",
                    ls="--", label="Bredt $q/t$")
    axes[2].set_xlabel("sample index (right wall)")
    axes[2].set_ylabel(r"$\tau_{yz}$")
    axes[2].set_title("wall shear on the long wall")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    p1 = ASSETS / "box-girder-torsion-fields.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    hs = [r["h"] for r in fd_rows]
    axes[0].loglog(hs, [r["j_torque"] for r in fd_rows], "o-",
                   label="FD torque identity")
    axes[0].loglog(hs, [r["j_energy"] for r in fd_rows], "s--",
                   label="FD cell energy")
    axes[0].axhline(fd_rows[0]["j_bredt"], color="k", ls=":", label="Bredt")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("grid spacing h")
    axes[0].set_ylabel("torsional constant J")
    axes[0].set_title("multiply connected Prandtl: grid study")
    axes[0].legend(fontsize=8)
    dofs = [r["dofs"] for r in fe_rows]
    axes[1].plot(dofs, [r["j_ratio_bredt"] for r in fe_rows], "o-",
                 label="HEX8 / Bredt")
    axes[1].axhline(fd_rows[-1]["j_torque"] / fd_rows[0]["j_bredt"], color="r",
                    ls="--", label="FD / Bredt")
    axes[1].axhline(1.0, color="k", ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dofs")
    axes[1].set_ylabel(r"$J_{FE}/J_{ref}$")
    axes[1].set_title("HEX8 ring mesh vs references")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p2 = ASSETS / "box-girder-torsion-convergence.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


def main():
    fd_rows, order = fd_study()
    tw_rows = thin_wall_study()
    reference = fd_rows[-1]["j_torque"] * G * THETA
    fe_rows = fe_study(reference)
    figs = figures(fd_rows, fe_rows)
    evidence = {
        "case": {"a": A, "b": B, "t": T, "G": G, "theta": THETA,
                 "section": "rectangular box girder, closed thin wall"},
        "fd_grid_study": fd_rows,
        "fd_observed_order": float(order),
        "thin_wall_asymptote": tw_rows,
        "hex8_study": fe_rows,
        "figures": [str(p.relative_to(ROOT)) for p in figs],
    }
    out = ROOT / "benchmarks" / "box_girder_torsion_evidence.json"
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"evidence -> {out}")
    return evidence


if __name__ == "__main__":
    main()