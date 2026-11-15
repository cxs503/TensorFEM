#!/usr/bin/env python3
"""Independent discretisation audit of the CZM-contact -> patch -> plate chain.

GAP-PP-1-ICE-STRUCT-CONTACT-PATCH (verifier seat).  The merged chain reports a
*non-uniform vs uniform* peak-stress delta per ice-edge contact configuration
(``benchmarks/ice_panel_patch_contact_scan.py``).  The headline number of such a
study is only meaningful if it is **discretisation independent**: the plate mesh
``(nx, ny)``, the contact sampling ``grid`` and the fixed-point iteration controls
(``relax`` / ``tol``) are all numerical choices, and the acceptance criterion
(``>=3 contact configurations``, machine-readable delta) does not by itself
prove that the reported delta is a property of the *physics* rather than of the
mesh.

This script measures exactly that, with three orthogonal sweeps:

1. ``mesh``      -- plate mesh (40,35)...(120,105) at fixed contact grid 97:
   does the non-uniform/uniform peak-``sigma_Mx`` ratio ``delta`` converge, and
   do the FEM and the Navier series agree inside the 3 % acceptance band at every
   mesh, or only where the chain was calibrated?
2. ``grid``      -- contact sampling 41/97/193/385 at fixed plate mesh: are peak
   pressure, contact area and peak stress contact-grid converged?
3. ``coupling``  -- under-relaxation 0.3/0.5/0.8 and fixed-point tolerance
   1e-4/1e-6 for the two-way cases: is the converged state path independent,
   i.e. is the reported two-way stress a genuine fixed point?

Outputs JSON (machine readable) and, when matplotlib is available, figures.

Usage
-----
    python benchmarks/verify_contact_patch_discretization.py --stage timing
    python benchmarks/verify_contact_patch_discretization.py --stage full \
        --out outputs/verify_contact_patch_discretization
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfem.ice_panel_patch_contact import (  # noqa: E402
    DESIGN_POINT,
    coupled_patch_solve,
    czm_contact_patch,
    default_configurations,
    default_law,
    nonuniformity_scan,
)

OUT_DEFAULT = Path(__file__).resolve().parents[1] / "outputs" / "verify_contact_patch_discretization"
ACCEPT_PCT = 3.0  # platform acceptance band for key quantities


# --------------------------------------------------------------------------
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


def _cfg_by_name(name: str) -> dict:
    for cfg in default_configurations():
        if cfg["name"] == name:
            return cfg
    raise KeyError(name)


# --------------------------------------------------------------------------
# 1. plate-mesh convergence of the reported delta
# --------------------------------------------------------------------------
def mesh_study(
    names: list[str],
    meshes: list[tuple[int, int]],
    *,
    grid: int = 97,
    total_force: float = 4.0e4,
) -> dict:
    rows: list[dict] = []
    for name in names:
        cfg = _cfg_by_name(name)
        for nx, ny in meshes:
            t0 = time.time()
            scan = nonuniformity_scan(
                total_force=total_force,
                nx=nx,
                ny=ny,
                grid=grid,
                configurations=[cfg],
            )
            row = scan["configurations"][0]
            rows.append(
                {
                    "config": name,
                    "coupled": row["coupled"],
                    "mesh": [nx, ny],
                    "h_mm": [
                        1e3 * DESIGN_POINT["l"] / nx,
                        1e3 * DESIGN_POINT["s"] / ny,
                    ],
                    "sigma_Mx_nonuniform_MPa": row["nonuniform_patch"]["sigma_Mx_MPa"],
                    "sigma_Mx_uniform_MPa": row["uniform_reference_patch"]["sigma_Mx_MPa"],
                    "delta_sigma_Mx_pct": row["delta_nonuniform_over_uniform"]["sigma_Mx_pct"],
                    "delta_ratio": row["delta_nonuniform_over_uniform"]["sigma_Mx"],
                    "fem_vs_navier_nonuniform_pct": row["nonuniform_patch"][
                        "fem_vs_navier_Mx_err_pct"
                    ],
                    "fem_vs_navier_uniform_pct": row["uniform_reference_patch"][
                        "fem_vs_navier_Mx_err_pct"
                    ],
                    "peak_pressure_MPa": row["peak_pressure_MPa"],
                    "contact_area_m2": row["contact_area_m2"],
                    "force_residual": row["force_residual"],
                    "force_projection_error_rel": row["force_projection_error_rel"],
                    "coupling_sweeps": (
                        row["coupling"]["sweeps"] if row["coupled"] else 1
                    ),
                    "coupling_converged": (
                        row["coupling"]["converged"] if row["coupled"] else True
                    ),
                    "runtime_s": round(time.time() - t0, 2),
                }
            )
    summary = {}
    for name in names:
        sub = [r for r in rows if r["config"] == name]
        deltas = [r["delta_sigma_Mx_pct"] for r in sub]
        fem_nav = [abs(r["fem_vs_navier_nonuniform_pct"]) for r in sub]
        fine = sub[-1]
        summary[name] = {
            "n_meshes": len(sub),
            "delta_pct_min": min(deltas),
            "delta_pct_max": max(deltas),
            "delta_pct_finest": fine["delta_sigma_Mx_pct"],
            "delta_spread_pct_points": max(deltas) - min(deltas),
            "delta_change_finest_vs_coarsest_pct": (
                100.0 * (fine["delta_sigma_Mx_pct"] / deltas[0] - 1.0) if deltas[0] else float("nan")
            ),
            "max_abs_fem_vs_navier_pct": max(fem_nav),
            "fem_vs_navier_all_within_accept": max(fem_nav) <= ACCEPT_PCT,
        }
    return {"meshes": [list(m) for m in meshes], "contact_grid": grid, "rows": rows, "summary": summary}


# --------------------------------------------------------------------------
# 2. contact-grid convergence
# --------------------------------------------------------------------------
def grid_study(
    names: list[str],
    grids: list[int],
    *,
    nx: int = 64,
    ny: int = 56,
    total_force: float = 4.0e4,
) -> dict:
    law = default_law()
    rows: list[dict] = []
    for name in names:
        cfg = _cfg_by_name(name)
        edge = dict(cfg["edge"])
        a, b = DESIGN_POINT["l"], DESIGN_POINT["s"]
        centre = (
            (0.5 * a + cfg["offset"][0], 0.5 * b + cfg["offset"][1])
            if cfg.get("offset")
            else None
        )
        for g in grids:
            t0 = time.time()
            scan = nonuniformity_scan(
                total_force=total_force,
                nx=nx,
                ny=ny,
                grid=g,
                configurations=[cfg],
            )
            row = scan["configurations"][0]
            p_ow = czm_contact_patch(
                law=law,
                ceiling=2.0e6,
                edge=edge,
                total_force=total_force,
                centre=centre,
                grid=g,
            )
            rows.append(
                {
                    "config": name,
                    "coupled": row["coupled"],
                    "contact_grid": g,
                    "peak_pressure_MPa": p_ow.peak_pressure / 1e6,
                    "contact_area_m2": p_ow.contact_area,
                    "mean_over_wetted_MPa": p_ow.total_force / p_ow.contact_area / 1e6,
                    "centroid_m": list(p_ow.centroid),
                    "approach_m": p_ow.approach,
                    "force_residual": p_ow.residual,
                    "sigma_Mx_nonuniform_MPa": row["nonuniform_patch"]["sigma_Mx_MPa"],
                    "sigma_Mx_uniform_MPa": row["uniform_reference_patch"]["sigma_Mx_MPa"],
                    "delta_sigma_Mx_pct": row["delta_nonuniform_over_uniform"]["sigma_Mx_pct"],
                    "runtime_s": round(time.time() - t0, 2),
                }
            )
    summary = {}
    for name in names:
        sub = [r for r in rows if r["config"] == name]
        fine = sub[-1]
        coarsest = sub[0]

        def _spread(key):
            vals = [r[key] for r in sub]
            return 100.0 * (max(vals) - min(vals)) / max(abs(fine[key]), 1e-30)

        summary[name] = {
            "n_grids": len(sub),
            "peak_pressure_spread_pct": _spread("peak_pressure_MPa"),
            "contact_area_spread_pct": _spread("contact_area_m2"),
            "delta_spread_pct_points": max(r["delta_sigma_Mx_pct"] for r in sub)
            - min(r["delta_sigma_Mx_pct"] for r in sub),
            "delta_change_finest_vs_coarsest_pct": 100.0
            * (
                fine["delta_sigma_Mx_pct"] / coarsest["delta_sigma_Mx_pct"] - 1.0
            ),
            "peak_pressure_finest_two_change_pct": (
                100.0 * (sub[-1]["peak_pressure_MPa"] / sub[-2]["peak_pressure_MPa"] - 1.0)
                if len(sub) >= 2
                else float("nan")
            ),
            "contact_area_finest_two_change_pct": (
                100.0 * (sub[-1]["contact_area_m2"] / sub[-2]["contact_area_m2"] - 1.0)
                if len(sub) >= 2
                else float("nan")
            ),
        }
    return {"plate_mesh": [nx, ny], "grids": grids, "rows": rows, "summary": summary}


# --------------------------------------------------------------------------
# 3. fixed-point path independence (two-way cases)
# --------------------------------------------------------------------------
def coupling_study(
    names: list[str],
    *,
    relaxs: list[float],
    tols: list[float],
    nx: int = 64,
    ny: int = 56,
    grid: int = 97,
    total_force: float = 4.0e4,
) -> dict:
    law = default_law()
    a, b = DESIGN_POINT["l"], DESIGN_POINT["s"]
    rows: list[dict] = []
    for name in names:
        cfg = _cfg_by_name(name)
        edge = dict(cfg["edge"])
        centre = (
            (0.5 * a + cfg["offset"][0], 0.5 * b + cfg["offset"][1])
            if cfg.get("offset")
            else None
        )
        for relax in relaxs:
            for tol in tols:
                t0 = time.time()
                patch, sol, coup = coupled_patch_solve(
                    nx=nx,
                    ny=ny,
                    law=law,
                    ceiling=2.0e6,
                    edge=edge,
                    total_force=total_force,
                    centre=centre,
                    grid=grid,
                    relax=relax,
                    tol=tol,
                )
                peak = sol.peak_stress()
                rows.append(
                    {
                        "config": name,
                        "relax": relax,
                        "tol": tol,
                        "sweeps": coup["sweeps"],
                        "converged": coup["converged"],
                        "final_pressure_change_rel": coup["history"][-1]["pressure_change_rel"],
                        "peak_pressure_MPa": patch.peak_pressure / 1e6,
                        "contact_area_m2": patch.contact_area,
                        "total_force_N": patch.total_force,
                        "force_residual": patch.residual,
                        "approach_m": patch.approach,
                        "centroid_m": list(patch.centroid),
                        "sigma_Mx_MPa": peak["sigma_Mx_MPa"],
                        "plate_centre_deflection_mm": coup["history"][-1][
                            "plate_centre_deflection_mm"
                        ],
                        "history": coup["history"],
                        "runtime_s": round(time.time() - t0, 2),
                    }
                )
    summary = {}
    for name in names:
        sub = [r for r in rows if r["config"] == name]
        ref = min(sub, key=lambda r: r["tol"])  # tightest tolerance as reference
        sig = [r["sigma_Mx_MPa"] for r in sub]
        pp = [r["peak_pressure_MPa"] for r in sub]
        summary[name] = {
            "n_runs": len(sub),
            "all_converged": all(r["converged"] for r in sub),
            "max_sweeps": max(r["sweeps"] for r in sub),
            "sigma_Mx_spread_pct": 100.0 * (max(sig) - min(sig)) / ref["sigma_Mx_MPa"],
            "peak_pressure_spread_pct": 100.0 * (max(pp) - min(pp)) / ref["peak_pressure_MPa"],
            "worst_sigma_Mx_vs_tightest_pct": max(
                100.0 * abs(r["sigma_Mx_MPa"] / ref["sigma_Mx_MPa"] - 1.0) for r in sub
            ),
            "worst_peak_pressure_vs_tightest_pct": max(
                100.0 * abs(r["peak_pressure_MPa"] / ref["peak_pressure_MPa"] - 1.0)
                for r in sub
            ),
            "reference": {"relax": ref["relax"], "tol": ref["tol"]},
        }
    return {"plate_mesh": [nx, ny], "contact_grid": grid, "rows": rows, "summary": summary}



# --------------------------------------------------------------------------
# 4. load-level sensitivity of the delta (is 40 kN representative?)
# --------------------------------------------------------------------------
def load_study(
    names: list[str],
    forces: list[float],
    *,
    nx: int = 64,
    ny: int = 56,
    grid: int = 97,
) -> dict:
    rows: list[dict] = []
    for name in names:
        cfg = _cfg_by_name(name)
        for force in forces:
            t0 = time.time()
            scan = nonuniformity_scan(
                total_force=force, nx=nx, ny=ny, grid=grid, configurations=[cfg]
            )
            row = scan["configurations"][0]
            rows.append(
                {
                    "config": name,
                    "total_force_N": force,
                    "mean_box_pressure_MPa": force / (row["footprint_m"][0] * row["footprint_m"][1]) / 1e6,
                    "peak_pressure_MPa": row["peak_pressure_MPa"],
                    "contact_area_m2": row["contact_area_m2"],
                    "peak_over_mean": row["pressure_peak_over_mean"],
                    "sigma_Mx_nonuniform_MPa": row["nonuniform_patch"]["sigma_Mx_MPa"],
                    "sigma_Mx_uniform_MPa": row["uniform_reference_patch"]["sigma_Mx_MPa"],
                    "delta_sigma_Mx_pct": row["delta_nonuniform_over_uniform"]["sigma_Mx_pct"],
                    "fem_vs_navier_nonuniform_pct": row["nonuniform_patch"]["fem_vs_navier_Mx_err_pct"],
                    "force_residual": row["force_residual"],
                    "runtime_s": round(time.time() - t0, 2),
                }
            )
    summary = {}
    for name in names:
        sub = [r for r in rows if r["config"] == name]
        deltas = [r["delta_sigma_Mx_pct"] for r in sub]
        summary[name] = {
            "n_loads": len(sub),
            "delta_pct_min": min(deltas),
            "delta_pct_max": max(deltas),
            "delta_spread_pct_points": max(deltas) - min(deltas),
            "peak_over_mean_range": [
                min(r["peak_over_mean"] for r in sub),
                max(r["peak_over_mean"] for r in sub),
            ],
        }
    return {"plate_mesh": [nx, ny], "contact_grid": grid, "rows": rows, "summary": summary}

# --------------------------------------------------------------------------
def figures(mesh: dict, gridres: dict, coup: dict, out: Path) -> list[dict]:
    try:
        import base64

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return []

    figs: list[dict] = []

    def _save(fig, name, caption):
        path = out / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        figs.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    # fig 1: delta vs plate mesh
    if mesh["rows"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        names = sorted({r["config"] for r in mesh["rows"]})
        for name in names:
            sub = [r for r in mesh["rows"] if r["config"] == name]
            h = [r["h_mm"][0] for r in sub]
            ax.plot(h, [r["delta_sigma_Mx_pct"] for r in sub], "o-", label=name)
        ax.set_xlabel("plate element size $h$ in x [mm]")
        ax.set_ylabel(r"$\delta = \sigma_{Mx}^{nonunif}/\sigma_{Mx}^{unif}-1$ [%]")
        ax.set_title("Non-uniformity delta vs plate mesh")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, "fig1_delta_vs_plate_mesh.png", "Non-uniformity peak-stress delta versus plate element size, per contact configuration.")

    # fig 2: FEM vs Navier vs mesh + acceptance band
    if mesh["rows"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for name in sorted({r["config"] for r in mesh["rows"]}):
            sub = [r for r in mesh["rows"] if r["config"] == name]
            h = [r["h_mm"][0] for r in sub]
            ax.plot(h, [r["fem_vs_navier_nonuniform_pct"] for r in sub], "s--", label=name)
        ax.axhspan(-ACCEPT_PCT, ACCEPT_PCT, color="tab:green", alpha=0.10, label="±3 % band")
        ax.set_xscale("log")
        ax.set_xlabel("plate element size $h$ in x [mm]")
        ax.set_ylabel("FEM vs Navier peak $\\sigma_{Mx}$ [%]")
        ax.set_title("Q4-Mindlin FEM vs Kirchhoff Navier series")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, "fig2_fem_vs_navier_vs_mesh.png", "FEM peak bending stress error against the independent Navier series versus plate element size, with the ±3 % acceptance band.")

    # fig 3: contact-grid convergence
    if gridres["rows"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for name in sorted({r["config"] for r in gridres["rows"]}):
            sub = [r for r in gridres["rows"] if r["config"] == name]
            ax.plot([r["contact_grid"] for r in sub], [r["peak_pressure_MPa"] for r in sub], "o-", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("contact sampling grid $n$ (n×n)")
        ax.set_ylabel("peak contact pressure [MPa]")
        ax.set_title("Contact-grid convergence of the patch peak pressure")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, "fig3_contact_grid_convergence.png", "Peak CZM contact pressure versus contact sampling grid, per configuration.")
    return figs


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["timing", "full"], default="full")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.stage == "timing":
        mesh = mesh_study(["C1-flat-edge-oneway"], [(40, 35)])
        gridres = grid_study(["C2-spherical-oneway"], [41], nx=40, ny=35)
        coup = coupling_study(
            ["C3-spherical-twoway"], relaxs=[0.5], tols=[1e-4], nx=40, ny=35, grid=41
        )
        payload = {
            "stage": "timing",
            "provenance": provenance(),
            "mesh": mesh,
            "grid": gridres,
            "coupling": coup,
            "runtime_s": round(time.time() - t_start, 2),
        }
        (out / "timing.json").write_text(json.dumps(payload, indent=1, default=float))
        print(json.dumps(payload["provenance"], indent=1))
        print("timing row runtimes:", [r["runtime_s"] for r in mesh["rows"]],
              [r["runtime_s"] for r in gridres["rows"]],
              [r["runtime_s"] for r in coup["rows"]])
        return 0

    cfg_names = ["C1-flat-edge-oneway", "C2-spherical-oneway", "C3-spherical-twoway", "C4-cylindrical-twoway-eccentric"]
    meshes = [(40, 35), (64, 56), (80, 70), (120, 105)]
    grids = [41, 97, 193, 385]

    mesh = mesh_study(cfg_names, meshes, grid=97)
    print("mesh study done", round(time.time() - t_start, 1), "s", flush=True)
    gridres = grid_study(cfg_names, grids, nx=64, ny=56)
    print("grid study done", round(time.time() - t_start, 1), "s", flush=True)
    coup = coupling_study(
        ["C3-spherical-twoway", "C4-cylindrical-twoway-eccentric"],
        relaxs=[0.3, 0.5, 0.8],
        tols=[1e-4, 1e-6],
        nx=64,
        ny=56,
        grid=97,
    )
    print("coupling study done", round(time.time() - t_start, 1), "s", flush=True)
    load = load_study(
        ["C1-flat-edge-oneway", "C3-spherical-twoway"],
        [2.0e4, 4.0e4, 6.0e4],
        nx=64,
        ny=56,
        grid=97,
    )
    print("load study done", round(time.time() - t_start, 1), "s", flush=True)

    figs = figures(mesh, gridres, coup, out)
    payload = {
        "case": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH independent discretisation audit",
        "provenance": provenance(),
        "setting": {
            "panel_m": [DESIGN_POINT["l"], DESIGN_POINT["s"]],
            "thickness_m": DESIGN_POINT["t"],
            "total_force_N": 4.0e4,
            "acceptance_band_pct": ACCEPT_PCT,
            "note": "delta = non-uniform CZM patch peak sigma_Mx / uniform same-force same-footprint patch - 1",
        },
        "mesh_study": mesh,
        "grid_study": gridres,
        "coupling_study": coup,
        "load_study": load,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figs],
        "runtime_s": round(time.time() - t_start, 2),
    }
    (out / "verification.json").write_text(json.dumps(payload, indent=1, default=float))
    import base64

    for f in figs:
        (out / f["name"]).write_bytes(base64.b64decode(f["png_base64"]))
    print("written:", out / "verification.json")
    print(json.dumps(mesh["summary"], indent=1))
    print(json.dumps(gridres["summary"], indent=1))
    print(json.dumps(coup["summary"], indent=1))
    print(json.dumps(load["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())