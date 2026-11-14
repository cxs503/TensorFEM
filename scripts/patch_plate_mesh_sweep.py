"""3-level mesh convergence sweep for the plate-patch elastoplastic capacity.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET`` acceptance criterion "3 档网格收敛":
the numerical collapse / first-yield / permanent-set path must converge
monotonically as the mesh is refined, and the transition point must stay within
the same-mesh elastic-field tolerance at every level.

This script runs the layered Shell4 J2 plate path at three even mesh levels
(nx = ny = 4, 6, 8) on the same geometry / material / patch and writes:

* ``artifacts/patch_plate_mesh_sweep.json``  — the per-mesh convergence table
* ``artifacts/fig_patch_plate_mesh_convergence.png`` — collapse / first-yield /
  permanent-set vs mesh, with the analytic yield-line upper and strip lower
  bounds drawn as horizontal reference lines.

Run:  ``PYTHONPATH=src python scripts/patch_plate_mesh_sweep.py``
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    run_patch_plastic_path,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
E, NU, FY = 210e9, 0.3, 355e6
PATCH = (0.5 * A, 0.5 * B)
HARDENING = 2e10
LAYERS = 3
MESH_LEVELS = (4, 6, 8)          # even counts keep the plate centre a node


def _run_level(nx: int) -> dict:
    """Run the FE path at one mesh level and return the convergence observables."""
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    steps = [f * py["pressure"] for f in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4, 3.0)]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=HARDENING, patch=PATCH,
                                  load_factors=steps, nx=nx, ny=nx,
                                  layers=LAYERS, unload_steps=3)
    # same-mesh elastic bias probe (one small elastic step)
    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                   hardening=HARDENING, patch=PATCH,
                                   load_factors=[0.5 * py["pressure"]],
                                   nx=nx, ny=nx, layers=LAYERS)
    fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
    same_mesh_yield = FY / fe_sigma_per_pa
    first = path.first_yield_factor
    collapse = path.collapse_factor or path.loading[-1].load_factor
    return {
        "nx": nx,
        "first_yield_factor": first,
        "first_yield_error_vs_analytic": (first - py["pressure"]) / py["pressure"],
        "first_yield_error_vs_same_mesh": (first - same_mesh_yield) / same_mesh_yield,
        "collapse_factor": collapse,
        "collapse_error_vs_upper": (collapse - yield_line_patch_upper_bound(
            A, B, T, FY, PATCH)["pressure"]) / yield_line_patch_upper_bound(
            A, B, T, FY, PATCH)["pressure"],
        "permanent_deflection": path.permanent_deflection,
        "peak_centre_deflection": path.loading[-1].centre_deflection,
        "elastic_bias": fe_sigma_per_pa / py["sigma_eq_per_pa"] - 1.0,
    }


def main() -> None:
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)

    rows = [_run_level(nx) for nx in MESH_LEVELS]

    # --- convergence assertions (the "3 档网格收敛" acceptance) -------------
    firsts = [r["first_yield_factor"] for r in rows]
    collapses = [r["collapse_factor"] for r in rows]
    perms = [r["permanent_deflection"] for r in rows]
    # collapse must approach the yield-line upper bound from below, monotonically
    assert all(c < upper for c in collapses), "collapse must stay below the upper bound"
    assert all(c > lower for c in collapses), "collapse must stay above the strip lower bound"
    assert all(collapses[i] <= collapses[i + 1] for i in range(len(collapses) - 1)), \
        "collapse load must be monotone non-decreasing with refinement"
    # transition point must stay within the same-mesh elastic-field tolerance at every level
    assert all(abs(r["first_yield_error_vs_same_mesh"]) < 0.05 for r in rows), \
        "transition vs same-mesh elastic field must be < 5% at every mesh level"
    # permanent set must be a bounded, positive fraction of the peak deflection
    assert all(0.0 < p < r["peak_centre_deflection"] for p, r in zip(perms, rows)), \
        "permanent set must be positive and below the peak deflection"

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    nx = [r["nx"] for r in rows]
    ax.plot(nx, [r["collapse_factor"] / 1e6 for r in rows], "o-", label="collapse (FE)")
    ax.plot(nx, [r["first_yield_factor"] / 1e6 for r in rows], "s--", label="first yield (FE)")
    ax.plot(nx, [r["permanent_deflection"] * 1e3 for r in rows], "^-.",
             label="permanent set (mm)")
    ax.axhline(upper / 1e6, color="r", ls=":", label="yield-line upper bound")
    ax.axhline(lower / 1e6, color="g", ls=":", label="strip lower bound")
    ax.axhline(py["pressure"] / 1e6, color="k", ls="--", label="analytic first yield")
    ax.set_xlabel("mesh level (nx = ny)")
    ax.set_ylabel("pressure [MPa] / permanent set [mm]")
    ax.set_title("Plate-patch elastoplastic capacity: 3-level mesh convergence")
    ax.legend()
    ax.grid(True)
    os.makedirs("artifacts", exist_ok=True)
    fig.savefig("artifacts/fig_patch_plate_mesh_convergence.png", dpi=150)
    plt.close(fig)

    # --- json -----------------------------------------------------------------
    out = {
        "geometry": {"a": A, "b": B, "t": T, "patch": PATCH},
        "material": {"E": E, "nu": NU, "fy": FY, "hardening": HARDENING, "layers": LAYERS},
        "bounds": {
            "analytic_first_yield_Pa": py["pressure"],
            "yield_line_upper_Pa": upper,
            "strip_lower_Pa": lower,
        },
        "mesh_levels": rows,
        "convergence": {
            "collapse_monotone": True,
            "transition_vs_same_mesh_max_abs_pct":
                100 * max(abs(r["first_yield_error_vs_same_mesh"]) for r in rows),
            "permanent_set_bounded": True,
        },
    }
    with open("artifacts/patch_plate_mesh_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
