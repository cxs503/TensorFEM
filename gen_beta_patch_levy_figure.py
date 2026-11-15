#!/usr/bin/env python3
"""Figure for the exact patch-load restraint band (JOUR-ICE-STRUCT Sec. 4.2).

Reads ``benchmarks/artifacts/beta_restraint_patch_levy.json`` and draws

* left  - centre von Mises stress vs embedding coefficient ``beta``: the exact
  Kirchhoff patch reference plus the Mindlin FEM at four mesh sizes, with the
  published Navier baseline marked at ``beta = 0``;
* right - convergence of the FEM centre stress onto the exact reference
  (log-log in element size), with an ``h^2`` guide.

Usage: ``python gen_beta_patch_levy_figure.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "benchmarks" / "artifacts" / "beta_restraint_patch_levy.json"
PNG = ROOT / "docs" / "figures" / "beta_restraint_patch_levy.png"
BASELINE = 73.37101604723352


def main() -> int:
    data = json.loads(ART.read_text())
    refs = data["reference_by_beta"]
    rows = data["fem_rows"]
    betas = [r["beta_x"] for r in refs]
    levy = [r["levy_sigma_vm_centre_MPa"] for r in refs]
    ritz = [r["ritz_sigma_vm_centre_MPa"] for r in refs]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    ax0.plot(betas, levy, "k-", lw=2.2, label="exact Kirchhoff patch (Levy ODE join)")
    ax0.plot(betas, ritz, "k--", lw=1.2, label="Rayleigh-Ritz (variational, 30x80)")
    meshes = sorted({r["nx"] for r in rows})
    markers = {20: "o", 40: "s", 80: "^", 160: "D"}
    for nx in meshes:
        xs = [r["beta_x"] for r in rows if r["nx"] == nx]
        ys = [r["sigma_vm_centre_MPa"] for r in rows if r["nx"] == nx]
        ax0.plot(xs, ys, markers.get(nx, "o") + "-", ms=4.5, lw=1.0,
                 label=f"Mindlin Q4 panel, nx={nx}")
    ax0.axhline(BASELINE, color="0.55", ls=":", lw=1.2,
                label="published Navier N=401 baseline (beta=0)")
    ax0.set_xlabel(r"embedding coefficient $\beta_x$ (longitudinal edge)")
    ax0.set_ylabel(r"centre von Mises bending stress $\sigma_{vM}$ [MPa]")
    ax0.set_title("Ice-belt panel, 1 MPa x 8 cm x 70 cm patch\n"
                  r"$0.8\times0.7\times0.022$ m, $E$=206 GPa, $\nu$=0.3")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=7.6, loc="upper right")

    for beta in betas:
        sub = sorted([r for r in rows if r["beta_x"] == beta], key=lambda r: r["nx"])
        if len(sub) < 2:
            continue
        h = [r["h_mm"] for r in sub]
        err = [abs(r["sigma_vm_err_vs_levy_pct"]) for r in sub]
        ax1.loglog(h, err, "o-", ms=4, lw=1.0, label=rf"$\beta_x$={beta:g}")
    h_ref = [rows[0]["h_mm"], rows[0]["h_mm"] / 8.0]
    ref0 = [abs(rows[0]["sigma_vm_err_vs_levy_pct"]) for _ in h_ref]
    ax1.loglog(h_ref, [ref0[0], ref0[0] / 64.0], "k:", lw=1.2, label=r"$h^2$ guide")
    ax1.set_xlabel("element size $h$ [mm]")
    ax1.set_ylabel(r"$|\sigma_{vM}^{FEM}/\sigma_{vM}^{exact}-1|$ [%]")
    ax1.set_title("FEM convergence onto the exact patch reference")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7.6)

    fig.tight_layout()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=150)
    print(f"wrote {PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())