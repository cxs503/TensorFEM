#!/usr/bin/env python
"""Figures for the W28 composite-laminate verification artifact.

Reads ``papers/artifacts/w28/composite-laminate-verification/anchor_and_convergence.json``
and writes the PNGs that the mission report embeds.  No number is typed here:
every plotted value comes out of the artifact.

Usage::

    python benchmarks/composite_laminate_w28/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "papers" / "artifacts" / "w28" / "composite-laminate-verification"
FIG = OUT / "figures"


def fig_cook(art: dict) -> Path:
    rows = art["cook"]["mesh_rows"]
    ref = art["cook"]["reference"]["value"]
    n = [r["n"] for r in rows]
    uy = [r["uy_probe"] for r in rows]
    ext = art["cook"]["richardson"]["extrapolated"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogx(n, uy, "o-", label="TensorFEM shell4 (membrane action)")
    ax[0].axhline(ref, color="k", ls="--", label=f"published anchor {ref}")
    ax[0].axhline(ext, color="C3", ls=":", label=f"Richardson {ext:.4f}")
    ax[0].set_xlabel("mesh n per side")
    ax[0].set_ylabel(r"$u_y$ at (48, 52)")
    ax[0].set_title("Cook (1974) membrane panel")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    err = [abs(r["rel_dev_pct_vs_published"]) for r in rows]
    ax[1].loglog(n, err, "s-")
    ax[1].set_xlabel("mesh n per side")
    ax[1].set_ylabel("|relative deviation| vs anchor [%]")
    ax[1].set_title("displacement error ladder")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = FIG / "cook_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_strict(art: dict) -> Path:
    quad = art["analytic_strict"]["quadratic_exact_convergence"]["rows"]
    n = [r["n"] for r in quad]
    e = [r["max_abs_err"] for r in quad]
    patch = art["analytic_strict"]["constant_stress_cases"]
    labels, vals = [], []
    for case, rows in patch.items():
        for r in rows:
            labels.append(f"{case}\nn={r['n']}")
            vals.append(r["rel_dev"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(n, e, "o-", label="bilinear shell, linear-stress exact field")
    # slope reference
    n_arr = np.array(n, dtype=float)
    ax[0].loglog(n_arr, e[0] * (n_arr[0] / n_arr) ** 2, "k--", label="slope -2 reference")
    ax[0].set_xlabel("mesh n per side")
    ax[0].set_ylabel(r"$\max|u_h - u_{exact}|$")
    ax[0].set_title("quadratic exact solution (Airy cubic + body force)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].bar(range(len(vals)), [max(v, 1e-18) for v in vals], color="C2")
    ax[1].set_yscale("log")
    ax[1].axhline(art["provenance"]["thresholds"]["strict_rel_dev_max"], color="r", ls="--",
                  label="1e-6 threshold")
    ax[1].set_xticks(range(len(vals)))
    ax[1].set_xticklabels(labels, fontsize=6, rotation=90)
    ax[1].set_ylabel("relative deviation [-]")
    ax[1].set_title("constant-stress (strict) solutions, distorted grids")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "analytic_strict.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_fpf(art: dict) -> Path:
    cases = [c for c in art["layup_ensemble"]["cases"] if "first_ply_failure" in c]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    labels, dev = [], []
    for c in cases:
        for r in c["first_ply_failure"]["ply_rows"]:
            labels.append(f"{c['layup']}|{c['load_case']}|ply{r['ply_index']}({r['angle_deg']:g}deg)")
            dev.append(r["rel_dev_pct"])
    idx = np.arange(len(dev))
    colors = ["C2" if abs(d) < art["provenance"]["thresholds"]["fpf_rel_dev_pct_max"] else "C3"
              for d in dev]
    ax[0].bar(idx, dev, color=colors)
    ax[0].axhline(15, color="r", ls="--", label="+15%")
    ax[0].axhline(-15, color="r", ls="--")
    ax[0].set_ylabel("per-ply relative deviation vs code-format load level [%]")
    ax[0].set_title("first-ply reconciliation against the transcribed CSR allowable")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(labels, fontsize=4, rotation=90)
    ax[0].legend(fontsize=8)
    # ply stress magnitudes, normalised by the allowable
    for k, layup in enumerate(dict.fromkeys(c["layup"] for c in cases)):
        for ls, load_case in zip(("-", "--"), dict.fromkeys(c["load_case"] for c in cases)):
            pts = []
            for c in cases:
                if c["layup"] != layup or c["load_case"] != load_case:
                    continue
                for r in c["first_ply_failure"]["ply_rows"]:
                    pts.append(r["FI"])
            if pts:
                ax[1].plot(range(len(pts)), pts, ls, marker="o", label=f"{layup} {load_case}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("ply index in the case ordering")
    ax[1].set_ylabel("failure index FI = stress / allowable [-]")
    ax[1].set_title("ply-level utilisation under unit resultants")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = FIG / "first_ply_reconciliation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_transverse(art: dict) -> Path:
    t = art["interlaminar_and_transverse"]
    rows = t["plate_fem_rows"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    n = [r["n"] for r in rows]
    w = [r["w_center"] for r in rows]
    ax[0].semilogx(n, w, "o-", label="Q4 Mindlin FSDT (MITC)")
    ax[0].axhline(t["w_center_exact3d_m"], color="k", ls="--",
                  label="exact 3-D (Pagano state space)")
    ax[0].axhline(t["w_center_navier_fsdt_m"], color="C1", ls=":",
                  label="Navier FSDT (exact for cross-ply)")
    ax[0].axhline(t["w_center_navier_clt_m"], color="C4", ls="-.",
                  label="Navier CLT (thin)")
    ax[0].set_xlabel("mesh n per side")
    ax[0].set_ylabel(r"$w_{center}$ [m]")
    ax[0].set_title("[0/90]s under unit sine pressure")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    shear = t["interface_shear_Pa"]
    ax[1].bar(range(len(shear)), shear, color="C0")
    ax[1].set_xticks(range(len(shear)))
    ax[1].set_xticklabels([f"iface {i}" for i in range(len(shear))], fontsize=7)
    ax[1].set_ylabel(r"$\tau_{xz}$ at the interface [Pa]")
    ax[1].set_title("interlaminar shear, exact 3-D solution")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = FIG / "transverse_interlaminar.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    art = json.loads((OUT / "anchor_and_convergence.json").read_text())
    made = [fig_cook(art), fig_strict(art), fig_fpf(art), fig_transverse(art)]
    print(json.dumps([str(p) for p in made], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())