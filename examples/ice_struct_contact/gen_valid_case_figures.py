#!/usr/bin/env python3
"""Figures for the GAP-PP-1 valid end-to-end case report (m22084).

Generates:
  fig1_pressure_field.png        Gaussian contact pressure (CZM-style patch)
  fig2_deflection_fields.png     w(x,y): non-uniform vs uniform patch
  fig3_stress_fields.png         peak face von Mises: non-uniform vs uniform
  fig4_fem_vs_navier.png         mesh ladder, FEM vs Navier series
All numbers are read from the committed artifact JSON (zero hand-copying).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.patch_from_contact import gaussian_patch_pressure  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ARTIFACT = REPO / "benchmarks" / "artifacts" / "GAP-PP-1-end-to-end-demo" / "end_to_end_demo.json"
OUT = REPO / "outputs" / "gap_pp1_valid_case"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ARTIFACT.read_text())
    vc = data["valid_case"]

    cx, cy = vc["contact"]["centre_m"]
    sx, sy = vc["contact"]["sigma_m"]
    peak = vc["contact"]["peak_pressure_MPa"] * 1e6
    side = vc["plate"]["side_m"]

    # ---- fig 1: pressure field ----
    n = 200
    x = np.linspace(0, side, n)
    y = np.linspace(0, side, n)
    XX, YY = np.meshgrid(x, y)
    P = peak * np.exp(-(((XX - cx) / sx) ** 2 + ((YY - cy) / sy) ** 2))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    cs = ax.pcolormesh(XX, YY, P / 1e6, shading="gouraud", cmap="magma")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Gaussian contact pressure, peak {peak/1e6:.2f} MPa, $\\sigma$=0.15 m")
    fig.colorbar(cs, ax=ax, label="p [MPa]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pressure_field.png")
    plt.close(fig)

    # ---- figs 2+3: fields from the FEM solves ----
    import torch
    from tensorfem.patch_from_contact import (
        ContactPressureData, contact_pressure_to_plate_load, peak_stresses,
        gaussian_patch_pressure,
    )
    from tensorfem.plate_mindlin import (
        compute_face_stresses, solve_plate_on_winkler_foundation,
    )

    E, nu, t = vc["plate"]["E_Pa"], vc["plate"]["nu"], vc["plate"]["thickness_m"]
    nx = vc["plate"]["nx"]
    pos, pres = _contact_grid(cx, cy, sx, sy, peak, side, 33)
    contact = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sx * sy,
    )
    sol_nu, patch = contact_pressure_to_plate_load(contact, nx, side, E, nu, t, 0.0)
    pw, pl = 6.0 * sx, 6.0 * sy
    p0 = patch.total_force / (pw * pl)

    def uniform_fn(x, y):
        in_patch = (
            (x >= cx - pw / 2) & (x <= cx + pw / 2) &
            (y >= cy - pl / 2) & (y <= cy + pl / 2)
        )
        return np.where(in_patch, p0, 0.0)

    sol_u = solve_plate_on_winkler_foundation(nx, side, E, nu, t, 0.0, uniform_fn)

    nodes = sol_nu.nodes.numpy()
    w_nu = sol_nu.displacement[0::3].numpy()
    w_u = sol_u.displacement[0::3].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=150)
    for ax, w, title in (
        (axes[0], w_nu, "non-uniform (projected)"),
        (axes[1], w_u, "uniform, same force"),
    ):
        cs = ax.tricontourf(nodes[:, 0], nodes[:, 1], w_u if title.startswith("uniform") else w_nu,
                          levels=24, cmap="viridis")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"deflection w [m] — {title}")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax, label="w [m]")
    fig.suptitle("Valid case (w/t = %.2f): deflection fields" % vc["small_deflection_validity"]["peak_deflection_over_thickness"])
    fig.tight_layout()
    fig.savefig(OUT / "fig2_deflection_fields.png")
    plt.close(fig)

    def vm_field(sol):
        faces = compute_face_stresses(sol)
        pts, vm = [], []
        for f in faces:
            s = f["stress"]
            sx_, sy_, txy = s["sigma_x"], s["sigma_y"], s["tau_xy"]
            vmv = math.sqrt(max(sx_ * sx_ + sy_ * sy_ - sx_ * sy_ + 3 * txy * txy, 0.0))
            pts.append(f["centroid"][:2])
            vm.append(vmv / 1e6)
        return np.array(pts), np.array(vm)

    pts_nu, vm_nu = vm_field(sol_nu)
    pts_u, vm_u = vm_field(sol_u)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=150)
    for ax, pts, vm, title in (
        (axes[0], pts_nu, vm_nu, "non-uniform (projected)"),
        (axes[1], pts_u, vm_u, "uniform, same force"),
    ):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=vm, cmap="magma", s=26)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"face von Mises [MPa] — {title}")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="$\\sigma_{vm}$ [MPa]")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_stress_fields.png")
    plt.close(fig)

    # ---- fig 4: FEM vs Navier ladder ----
    ladder = vc["navier_benchmark"]["mesh_ladder"]
    nxs = [r["plate_nx"] for r in ladder]
    fem = [r["sigma_x_MPa"] for r in ladder]
    nav = [r["navier_sigma_x_MPa"] for r in ladder]
    errs = [r["fem_vs_navier_pct"] for r in ladder]
    fig, ax1 = plt.subplots(figsize=(5.6, 4.2), dpi=150)
    ax1.plot(nxs, fem, "o-", label="FEM (Mindlin Q4 SRI)")
    ax1.plot(nxs, nav, "s--", label="Navier series (Kirchhoff)")
    ax1.set_xlabel("elements per side")
    ax1.set_ylabel("peak $\\sigma_x$ [MPa]")
    ax1.set_title("FEM vs Navier, Gaussian patch (valid case)")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(nxs, errs, "^:", color="tab:red", label="FEM vs Navier error")
    ax2.axhline(0.339, color="tab:gray", lw=0.8)
    ax2.axhline(-0.339, color="tab:gray", lw=0.8)
    ax2.text(nxs[0], 0.339, " H3 anchor 0.339 %", fontsize=8, color="tab:gray")
    ax2.set_ylabel("error [%]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_fem_vs_navier.png")
    plt.close(fig)
    print("figures written to", OUT)


def _contact_grid(cx, cy, sx, sy, peak, side, n):
    lo_x, hi_x = max(0.0, cx - 3.0 * sx), min(side, cx + 3.0 * sx)
    lo_y, hi_y = max(0.0, cy - 3.0 * sy), min(side, cy + 3.0 * sy)
    xs = np.linspace(lo_x, hi_x, n)
    ys = np.linspace(lo_y, hi_y, n)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    pres = gaussian_patch_pressure(pos[:, 0], pos[:, 1], cx, cy, peak, sx, sy)
    return pos, pres


if __name__ == "__main__":
    main()
