#!/usr/bin/env python3
"""CZM-contact ice patch load -> panel stresses: JOUR-ICE-STRUCT gap closure.

GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.  Numerics live in
``tensorfem.ice_panel_patch_contact``; this driver runs

1. the design-point calibration against the already published uniform-patch
   baseline (``panel_patch_load_results.json``, plate_vs_strip row
   ``lambda_c_over_l = 0.1``, ``full_breadth``) and the Timoshenko Table 8
   coefficients,
2. the non-uniformity scan: four ice-edge contact configurations solved by the
   ice-structure CZM contact (two of them bidirectionally coupled to the panel
   deflection), each compared with a uniform patch of the same total force and
   the same footprint,

and writes machine-readable JSON plus two figures.

Usage
-----
    python benchmarks/ice_panel_patch_contact_scan.py --out outputs --quick
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfem.ice_panel_patch_contact import (  # noqa: E402
    DESIGN_POINT,
    design_point_calibration,
    nonuniformity_scan,
)

OUT_DEFAULT = Path(__file__).resolve().parents[1] / "outputs"


def provenance() -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # pragma: no cover
            return "unknown"

    return {
        "script": Path(__file__).name,
        "git_sha": sh(["git", "rev-parse", "--short", "HEAD"]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def figures(scan: dict, calib: dict, out: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []

    # ---- figure 1: pressure fields of the scanned configurations -----------
    from tensorfem.ice_panel_patch_contact import (
        coupled_patch_solve,
        czm_contact_patch,
        default_law,
        uniform_patch_pressure,
    )

    a, b = scan["setting"]["panel_m"]
    F = scan["setting"]["total_force_N"]
    ceiling = scan["setting"]["ceiling_MPa"] * 1e6
    law = default_law()
    panels = []
    for name in ("C2-spherical-oneway", "C3-spherical-twoway"):
        cfg = next(
            (r for r in scan["configurations"] if r["name"] == name), None
        )
        if cfg is None:
            continue
        if cfg["coupled"]:
            patch, _, _ = coupled_patch_solve(
                nx=32, ny=28, law=law, ceiling=ceiling, edge=cfg["edge"], total_force=F,
                grid=scan["setting"]["contact_grid"],
            )
        else:
            patch = czm_contact_patch(
                law=law, ceiling=ceiling, edge=cfg["edge"], total_force=F,
                grid=scan["setting"]["contact_grid"],
            )
        panels.append((name, patch))
    if panels:
        uni = uniform_patch_pressure(
            total_force=F,
            centre=panels[0][1].centroid,
            half_x=0.5 * float(panels[0][1].x[-1] - panels[0][1].x[0]),
            half_y=0.5 * float(panels[0][1].y[-1] - panels[0][1].y[0]),
            panel=(a, b),
            grid=scan["setting"]["contact_grid"],
        )
        panels.append(("uniform reference", uni))
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
        if len(panels) == 1:
            axes = [axes]
        for ax, (name, patch) in zip(axes, panels):
            X, Y = torch.meshgrid(patch.x, patch.y, indexing="ij")
            im = ax.pcolormesh(
                X.numpy(), Y.numpy(), patch.pressure.numpy() / 1e6, shading="auto",
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="p [MPa]")
            ax.set_title(
                f"{name}\npeak {patch.peak_pressure/1e6:.2f} MPa, "
                f"mean {patch.total_force/patch.contact_area/1e6:.2f} MPa"
            )
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
        fig.tight_layout()
        p1 = out / "fig1_czm_contact_pressure_patches.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        figs.append(
            {
                "name": "fig1_czm_contact_pressure_patches.png",
                "caption": (
                    "Ice-structure CZM contact pressure patches on the 0.8 m x 0.7 m ice-belt "
                    "panel at a total normal force of 40 kN: spherical ice edge one-way "
                    "(plate rigid), the same edge with two-way plate-deflection coupling, "
                    "and the uniform reference patch of identical force and footprint."
                ),
                "path": str(p1),
            }
        )

    # ---- figure 2: peak panel stress, non-uniform vs uniform --------------
    rows = scan["configurations"]
    labels = [r["name"].split("-")[0] for r in rows]
    sig_np = [r["nonuniform_patch"]["sigma_Mx_MPa"] for r in rows]
    sig_u = [r["uniform_reference_patch"]["sigma_Mx_MPa"] for r in rows]
    ratio = [r["delta_nonuniform_over_uniform"]["sigma_Mx"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar([i - 0.2 for i in x], sig_np, width=0.4, label="CZM contact patch", color="#2b6cb0")
    ax.bar([i + 0.2 for i in x], sig_u, width=0.4, label="uniform patch (same force)", color="#dd6b20")
    for i, r in enumerate(ratio):
        ax.text(i, max(sig_np[i], sig_u[i]) * 1.02, f"x{r:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"peak $\sigma_{Mx}$ [MPa]")
    ax.set_title(
        "Peak panel bending stress: non-uniform ice patch vs uniform patch "
        f"({scan['setting']['total_force_N']/1e3:.0f} kN, same footprint)"
    )
    ax.legend()
    fig.tight_layout()
    p2 = out / "fig2_peak_stress_nonuniform_vs_uniform.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig2_peak_stress_nonuniform_vs_uniform.png",
            "caption": (
                "Peak panel bending stress sigma_Mx from the finite-element moment field for "
                "each contact configuration (blue) against a uniform patch of the same total "
                "force and footprint (orange); the multiplier above each pair is the "
                "non-uniform/uniform ratio. Dashed reference: the JOUR-ICE-STRUCT design-point "
                "value 78.67 MPa at 1 MPa uniform patch pressure."
            ),
        }
    )
    # design-point line on fig2 is described in the caption; draw it on a copy
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar([i - 0.2 for i in x], sig_np, width=0.4, label="CZM contact patch", color="#2b6cb0")
    ax.bar([i + 0.2 for i in x], sig_u, width=0.4, label="uniform patch (same force)", color="#dd6b20")
    ax.axhline(
        calib["baseline_panel_patch_load_results"]["sigma_plate_Mx_MPa"],
        ls="--", color="k", lw=1,
        label="design point 1 MPa uniform patch (baseline 78.67 MPa)",
    )
    for i, r in enumerate(ratio):
        ax.text(i, max(sig_np[i], sig_u[i]) * 1.02, f"x{r:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"peak $\sigma_{Mx}$ [MPa]")
    ax.set_title("Peak panel bending stress, CZM patch vs uniform patch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--quick", action="store_true", help="coarser mesh for a fast check")
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=35)
    ap.add_argument("--grid", type=int, default=97)
    args = ap.parse_args(argv)
    if args.quick:
        args.nx, args.ny, args.grid = 32, 28, 65

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    calib = design_point_calibration(nx=args.nx, ny=args.ny)
    scan = nonuniformity_scan(nx=args.nx, ny=args.ny, grid=args.grid)
    result = {
        "case": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH",
        "title": "ice patch load from a CZM contact solve, fed two-way into the panel solver",
        "design_point_calibration": calib,
        "nonuniformity_scan": scan,
        "provenance": provenance(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    jpath = out / "ice_panel_patch_contact_scan.json"
    jpath.write_text(json.dumps(result, indent=1))

    try:
        figs = figures(scan, calib, out)
        result["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]
        jpath.write_text(json.dumps(result, indent=1))
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"figures skipped: {exc}")

    print(f"design point  : FEM {calib['fem']['sigma_Mx_MPa']:.2f} MPa | "
          f"Navier {calib['navier_reference']['sigma_Mx_MPa']:.2f} MPa | "
          f"baseline {calib['baseline_panel_patch_load_results']['sigma_plate_Mx_MPa']:.2f} MPa")
    print(f"Timoshenko    : M coef {calib['timoshenko_table8']['M_coef_got']:.6f} "
          f"({calib['timoshenko_table8']['M_err_pct']:+.3f}% vs 0.0479)")
    for r in scan["configurations"]:
        print(f"{r['name']:34s} p_peak/p_mean={r['pressure_peak_over_mean']:.2f} "
              f"sigma_Mx={r['nonuniform_patch']['sigma_Mx_MPa']:7.2f} MPa "
              f"uniform={r['uniform_reference_patch']['sigma_Mx_MPa']:7.2f} MPa "
              f"ratio={r['delta_nonuniform_over_uniform']['sigma_Mx']:.3f} "
              f"navier_dev={r['nonuniform_patch']['fem_vs_navier_Mx_err_pct']:+.2f}%")
    print(f"written: {jpath}  ({result['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())