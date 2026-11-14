#!/usr/bin/env python3
"""Physically calibrated beta band for the ice-belt panel (anchor GAP-PP-4).

Reads nothing, writes
``benchmarks/artifacts/beta_stiffener_calibration.json`` plus three figures.
Every number in the accompanying report is read from that JSON.

Run:  python benchmarks/beta_stiffener_calibration.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# The box runs a shared cluster queue; letting torch grab every core makes the
# small solves here thrash.  Fixed small pool keeps runtime predictable.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(4)

from tensorfem.plate_elastic_restraint import moments_at_nodes, solve_restrained_panel  # noqa: E402
from tensorfem.plate_levy_restraint import flexural_rigidity, levy_restrained_plate  # noqa: E402
from tensorfem.stiffener_rotational_restraint import (  # noqa: E402
    beta_from_rotational_stiffness,
    calibrate_panel_restraint,
    moment_fixity_strip,
    sensitivity_sweep,
)

OUT = HERE / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- paper case
# JOUR-ICE-STRUCT design point (papers/jour-ice-struct/data/panel_patch_load_results.json)
PANEL = dict(young=206.0e9, poisson=0.3, thickness=0.022, side_x=0.7, side_y=0.8)
Q = 1.0e6

# Ice-class scantlings: flat-bar longitudinal, flat-bar transverse frame.
LONG_VARIANTS = [
    ("FB 120x10", [(0.120, 0.010)]),
    ("FB 150x12", [(0.150, 0.012)]),
    ("HP 180x10", [(0.180, 0.010), (0.060, 0.014)]),
]
FRAME_VARIANTS = [
    ("FB 200x12", [(0.200, 0.012)]),
    ("HP 220x12", [(0.220, 0.012), (0.070, 0.016)]),
]
SUPPORT_SPANS = [0.4, 0.6, 0.7, 0.8, 1.0, 1.2, 1.6]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def design_point() -> dict:
    """Calibrated restraint of the paper panel with the reference scantlings."""
    cal = calibrate_panel_restraint(
        **PANEL,
        longitudinal_rectangles=[(0.150, 0.012)],
        frame_rectangles=[(0.200, 0.012)],
    )
    return {"calibration": cal.to_dict(), "solver_kwargs": cal.solver_kwargs()}


def end_to_end(cal) -> dict:
    """Calibrated beta -> Mindlin FE, against the exact Levy reference at the same k_theta."""
    rows = []
    for nx, ny in ((24, 28), (48, 56), (72, 84), (96, 112)):
        panel = solve_restrained_panel(
            nx=nx, ny=ny,
            side_x=cal.side_x, side_y=cal.side_y,
            young=cal.young, poisson=cal.poisson, thickness=cal.thickness,
            pressure=Q, beta_x=cal.longitudinal.beta, beta_y=0.0,
        )
        cx, cy = cal.side_x / 2.0, cal.side_y / 2.0
        i = int(np.argmin((panel.nodes[:, 0] - cx) ** 2 + (panel.nodes[:, 1] - cy) ** 2))
        w = float(panel.displacement[3 * i])
        mx = float(moments_at_nodes(panel)[i, 0])
        exact = levy_restrained_plate(
            a=cal.side_x, b=cal.side_y, D=cal.D, nu=cal.poisson,
            K=cal.longitudinal.k_theta, q=Q, n_modes=201,
        )
        w_e, mx_e = exact.w_centre(), exact.moments_centre()[0]
        rows.append(
            {
                "nx": nx, "ny": ny,
                "w_centre_m": w, "w_exact_m": w_e, "err_w_pct": 100.0 * (w / w_e - 1.0),
                "Mx_centre_Nm_per_m": mx, "Mx_exact_Nm_per_m": mx_e,
                "err_Mx_pct": 100.0 * (mx / mx_e - 1.0),
            }
        )
    return {
        "note": "Mindlin FE with the beta calibrated from scantlings vs exact Kirchhoff Levy "
                "evaluated at that same distributed stiffness k_theta",
        "k_theta_N_per_rad": cal.longitudinal.k_theta,
        "beta": cal.longitudinal.beta,
        "rows": rows,
    }


def band(cal) -> dict:
    """Centre stress vs beta on the calibrated panel (light mesh, uniform load)."""
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    rows = []
    for beta in betas:
        panel = solve_restrained_panel(
            nx=40, ny=46,
            side_x=cal.side_x, side_y=cal.side_y,
            young=cal.young, poisson=cal.poisson, thickness=cal.thickness,
            pressure=Q, beta_x=beta, beta_y=0.0,
        )
        cx, cy = cal.side_x / 2.0, cal.side_y / 2.0
        i = int(np.argmin((panel.nodes[:, 0] - cx) ** 2 + (panel.nodes[:, 1] - cy) ** 2))
        mx = float(moments_at_nodes(panel)[i, 0])
        rows.append(
            {
                "beta": beta,
                "Mx_centre_Nm_per_m": mx,
                "sigma_centre_MPa": 6.0 * mx / cal.thickness**2 / 1e6,
                "moment_fixity_strip": moment_fixity_strip(
                    beta * cal.D / cal.side_x if beta < 1.0 else 1e18,
                    cal.D, cal.side_x,
                ),
            }
        )
    s0, s1 = rows[0]["sigma_centre_MPa"], rows[-1]["sigma_centre_MPa"]
    # Stress at the calibrated physical restraint, by interpolation in beta.
    b = cal.longitudinal.beta
    interp = float(np.interp(b, [r["beta"] for r in rows], [r["sigma_centre_MPa"] for r in rows]))

    # The design point restrains *both* edge families: solve it directly rather
    # than interpolating a one-sided band.
    both = solve_restrained_panel(
        nx=40, ny=46, side_x=cal.side_x, side_y=cal.side_y,
        young=cal.young, poisson=cal.poisson, thickness=cal.thickness,
        pressure=Q, beta_x=cal.longitudinal.beta, beta_y=cal.frame.beta,
    )
    cx, cy = cal.side_x / 2.0, cal.side_y / 2.0
    i = int(np.argmin((both.nodes[:, 0] - cx) ** 2 + (both.nodes[:, 1] - cy) ** 2))
    mx_both = float(moments_at_nodes(both)[i, 0])
    s_both = 6.0 * mx_both / cal.thickness**2 / 1e6

    # Independent exact reference for the one-sided restraint (Kirchhoff Levy).
    exact_rows = []
    for beta in (0.0, cal.longitudinal.beta, 1.0):
        K = 1e15 if beta == 1.0 else beta * cal.D / cal.side_x
        sol = levy_restrained_plate(a=cal.side_x, b=cal.side_y, D=cal.D, nu=cal.poisson,
                                    K=K, q=Q, n_modes=201)
        mx = sol.moments_centre()[0]
        exact_rows.append({"beta": beta, "K_N_per_rad": K, "Mx_centre_Nm_per_m": mx,
                           "sigma_centre_MPa": 6.0 * mx / cal.thickness**2 / 1e6})
    return {
        "rows": rows,
        "sigma_SSSS_MPa": s0,
        "sigma_clamped_MPa": s1,
        "reduction_vs_SSSS_pct": 100.0 * (1.0 - s1 / s0),
        "calibrated_beta": b,
        "sigma_at_calibrated_beta_MPa": interp,
        "reduction_at_calibrated_pct": 100.0 * (1.0 - interp / s0),
        "sigma_at_calibrated_both_edges_MPa": s_both,
        "reduction_both_edges_pct": 100.0 * (1.0 - s_both / s0),
        "exact_levy_check": exact_rows,
        "band_note": "beta_y = 0 for the sweep (longitudinal restraint only); "
                     "the design point row also applies the frame restraint",
    }


def field(cal) -> dict:
    """von Mises bending stress field at the calibrated restraint (for the contour)."""
    panel = solve_restrained_panel(
        nx=64, ny=72,
        side_x=cal.side_x, side_y=cal.side_y,
        young=cal.young, poisson=cal.poisson, thickness=cal.thickness,
        pressure=Q, beta_x=cal.longitudinal.beta, beta_y=cal.frame.beta,
    )
    m = moments_at_nodes(panel).numpy()
    t = cal.thickness
    vm = 6.0 / t**2 * np.sqrt(m[:, 0] ** 2 - m[:, 0] * m[:, 1] + m[:, 1] ** 2 + 3.0 * m[:, 2] ** 2)
    nodes = panel.nodes.numpy()
    nx, ny = panel.nx, panel.ny
    grid = vm.reshape(ny + 1, nx + 1) / 1e6
    xs = nodes[:, 0].reshape(ny + 1, nx + 1)
    ys = nodes[:, 1].reshape(ny + 1, nx + 1)
    return {
        "x": xs.tolist(), "y": ys.tolist(), "sigma_vm_MPa": grid.tolist(),
        "max_MPa": float(grid.max()),
        "beta_x": cal.longitudinal.beta, "beta_y": cal.frame.beta,
    }


def main() -> int:
    dp = design_point()
    full = calibrate_panel_restraint(
        **PANEL,
        longitudinal_rectangles=[(0.150, 0.012)],
        frame_rectangles=[(0.200, 0.012)],
    )
    data = {
        "anchor": "GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT",
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "panel": PANEL,
        "pressure_Pa": Q,
        "design_point": dp,
        "design_point_summary": {
            "beta_x": full.longitudinal.beta,
            "beta_y": full.frame.beta,
            "moment_fixity_x": full.longitudinal.moment_fixity,
            "moment_fixity_y": full.frame.moment_fixity,
            "k_theta_x_N_per_rad": full.longitudinal.k_theta,
            "k_theta_y_N_per_rad": full.frame.k_theta,
            "D_Nm": full.D,
        },
        "sensitivity": sensitivity_sweep(
            **PANEL,
            longitudinal_variants=LONG_VARIANTS,
            frame_variants=FRAME_VARIANTS,
            support_spans=SUPPORT_SPANS,
        ),
        "end_to_end": end_to_end(full),
        "band": band(full),
        "field": field(full),
    }
    (OUT / "beta_stiffener_calibration.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data["design_point_summary"], indent=1))
    print("E2E:", [(r["nx"], round(r["err_w_pct"], 3), round(r["err_Mx_pct"], 3)) for r in data["end_to_end"]["rows"]])
    print("band:", {k: v for k, v in data["band"].items() if k != "rows"})
    print("wrote", OUT / "beta_stiffener_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())