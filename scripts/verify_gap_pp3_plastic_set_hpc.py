#!/usr/bin/env python
"""HPC verification driver for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

Runs the FE-heavy computations (elastic-plastic transition, plastic limit load,
3-grid convergence, permanent deflection) and writes results to JSON.

Usage:
    python scripts/verify_gap_pp3_plastic_set_hpc.py --output outputs/gap_pp3_hpc_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    yield_line_patch_upper_bound,
    strip_lower_bound,
    ilyushin_sine_upper_bound,
    low_temperature_flow_stress,
    run_patch_plastic_path,
    limit_load_from_path,
    patch_plate_mesh,
    elastic_patch_plate,
    section_moment_capacity,
)
from tensorfem.ice_plate_plasticity import (
    flow_stress_at_temperature,
    navier_centre_moments,
    yield_line_limit_pressure,
)


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

CASES = [
    dict(name="DH36_-40C", a=2.0, b=1.0, t=0.020, grade="DH36", temp=-40.0,
         patch=(0.5, 0.5), young=206e9, poisson=0.3, hardening=1000e6),
    dict(name="EH36_-40C", a=2.0, b=1.0, t=0.020, grade="EH36", temp=-40.0,
         patch=(0.5, 0.5), young=206e9, poisson=0.3, hardening=1000e6),
    dict(name="FH36_-60C", a=2.0, b=1.0, t=0.020, grade="FH36", temp=-60.0,
         patch=(0.5, 0.5), young=206e9, poisson=0.3, hardening=1000e6),
]

MESH_CONFIGS = [
    (8, 4),    # coarse
    (16, 8),   # medium
    (24, 12),  # fine
]


def run_single_case(case: dict, mesh_configs: list) -> dict:
    """Run full verification for one case across all mesh levels."""
    a, b, t = case["a"], case["b"], case["t"]
    patch = case["patch"]
    young, poisson, hardening = case["young"], case["poisson"], case["hardening"]

    flow = low_temperature_flow_stress(case["grade"], case["temp"])
    sy = flow.yield_stress

    # Analytical references
    elastic_ref = analytic_first_yield_pressure(a, b, t, young, poisson, sy, patch)
    upper_ref = yield_line_patch_upper_bound(a, b, t, sy, patch)
    lower_ref = strip_lower_bound(a, b, t, sy, patch)
    sine_ref = ilyushin_sine_upper_bound(a, b, t, sy, patch)

    result = {
        "case": case["name"],
        "geometry": {"a": a, "b": b, "t": t, "patch": patch},
        "material": {"grade": case["grade"], "temperature": case["temp"],
                     "yield_stress_MPa": sy / 1e6, "ductile": flow.ductile},
        "analytical": {
            "first_yield_MPa": elastic_ref["pressure"] / 1e6,
            "yield_line_ub_MPa": upper_ref["pressure"] / 1e6,
            "strip_lb_MPa": lower_ref["pressure"] / 1e6,
            "sine_ub_MPa": sine_ref["pressure"] / 1e6,
        },
        "mesh_results": [],
    }

    # FE computations at each mesh level
    for nx, ny in mesh_configs:
        print(f"\n  Mesh {nx}x{ny} ...", end=" ", flush=True)
        t0 = time.time()

        # Load ramp: from 0.2x to 4x first-yield pressure
        p_yield = elastic_ref["pressure"]
        load_factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        load_factors = [lf * p_yield for lf in load_factors]

        try:
            path = run_patch_plastic_path(
                a, b, t, young=young, poisson=poisson,
                yield_stress=sy, hardening=hardening,
                patch=patch, load_factors=load_factors,
                nx=nx, ny=ny, layers=5,
                unload_steps=5, unload_to=0.0,
                inplane_restrained=True,
                max_deflection_factor=10.0,
            )
            dt = time.time() - t0
            print(f"done ({dt:.1f}s)")

            # Extract limit load
            try:
                ll = limit_load_from_path(path)
                limit_load = ll.get("limit_load")
            except Exception:
                limit_load = None

            mesh_result = {
                "nx": nx, "ny": ny,
                "time_s": dt,
                "first_yield_factor": path.first_yield_factor,
                "first_yield_error": (
                    (path.first_yield_factor - p_yield) / p_yield
                    if path.first_yield_factor is not None else None
                ),
                "collapse_factor": path.collapse_factor,
                "ramp_completed": path.ramp_completed,
                "permanent_deflection_mm": (
                    path.permanent_deflection * 1000
                    if path.permanent_deflection is not None else None
                ),
                "permanent_set_valid": path.permanent_set_valid,
                "limit_load_MPa": limit_load / 1e6 if limit_load is not None else None,
                "limit_load_error_vs_ub": (
                    (limit_load - upper_ref["pressure"]) / upper_ref["pressure"]
                    if limit_load is not None else None
                ),
                "n_loading_points": len(path.loading),
                "peak_deflection_mm": path.loading[-1].centre_deflection * 1000,
            }

            # Same-mesh elastic bias correction
            if path.first_yield_factor is not None:
                corrected = sy / (path.first_yield_factor / p_yield * elastic_ref["sigma_eq_per_pa"])
                mesh_result["first_yield_error_bias_corrected"] = \
                    (path.first_yield_factor - corrected) / corrected

        except Exception as e:
            dt = time.time() - t0
            print(f"FAILED ({dt:.1f}s): {e}")
            mesh_result = {
                "nx": nx, "ny": ny, "time_s": dt,
                "error": str(e),
            }

        result["mesh_results"].append(mesh_result)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/gap_pp3_hpc_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_results = []
    for case in CASES:
        print(f"\n{'='*60}")
        print(f"Case: {case['name']}")
        print(f"{'='*60}")
        result = run_single_case(case, MESH_CONFIGS)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        print(f"\nCase: {result['case']}")
        print(f"  First yield (analytical): {result['analytical']['first_yield_MPa']:.4f} MPa")
        print(f"  Yield-line UB: {result['analytical']['yield_line_ub_MPa']:.4f} MPa")
        print(f"  Strip LB: {result['analytical']['strip_lb_MPa']:.4f} MPa")
        for mr in result["mesh_results"]:
            if "error" in mr:
                print(f"  Mesh {mr['nx']}x{mr['ny']}: FAILED - {mr['error']}")
            else:
                fy_err = mr.get("first_yield_error")
                ll_err = mr.get("limit_load_error_vs_ub")
                perm = mr.get("permanent_deflection_mm")
                print(f"  Mesh {mr['nx']}x{mr['ny']}: "
                      f"FY err={fy_err*100:.2f}%" if fy_err is not None else f"  Mesh {mr['nx']}x{mr['ny']}: FY err=N/A",
                      f", LL err={ll_err*100:.2f}%" if ll_err is not None else ", LL err=N/A",
                      f", perm={perm:.4f} mm" if perm is not None else ", perm=N/A")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
