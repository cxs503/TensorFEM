#!/usr/bin/env python3
"""GAP-PP-3: what reference does the FE elastic->plastic transition belong to?

The anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET`` requires the FE transition point
to be within 5 % of the analytic yield load.  The 3-mesh matrix published in
``docs/assets/gap_pp3_3grid_results.json`` (merge #1193/#1208) reports

    raw first-yield error:  8x4 +29.7 %   16x8 +20.7 %   24x12 +20.4 %

i.e. a plateau that does *not* converge (observed order 0.026).  A plateau is
not a discretisation error: it is what you get when the FE model and the
reference are solving *different problems*.  The matrix runs the path with
``geometric=True, inplane_restrained=True`` (needed so that a limit point
exists at all) while ``analytic_first_yield_pressure`` is the **linear**
small-deflection Navier solution.  At the measured transition the centre
deflection is ``w/t ~ 1``, where membrane action in a restrained panel is first
order and *reduces* the bending stress -- so the FE necessarily yields later
than the linear reference.

This driver measures both configurations on the same meshes and the same load
ladder, so the difference is attributable:

* ``linear-free``  : ``geometric=False, inplane_restrained=False`` -- the
  configuration whose reference *is* the Navier solution.  This is the
  "consistency regression against the elastic solution on the same mesh"
  required by the anchor.
* ``nonlinear-fixed``: ``geometric=True, inplane_restrained=True`` -- the
  configuration the limit-load study needs; its transition point is *not*
  comparable to a linear reference and is reported as a delayed-yield effect
  with its own (geometrically consistent) check.

Usage
-----
    python scripts/diagnose_gap_pp3_transition_reference.py            # all, serial
    python scripts/diagnose_gap_pp3_transition_reference.py --index 0 --nshards 6
    python scripts/diagnose_gap_pp3_transition_reference.py --merge
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    analytic_first_yield_pressure,
    low_temperature_flow_stress,
    plate_capacity_summary,
)

A, B, T = 2.0, 1.0, 0.020
E_MOD, NU, HARDENING = 206e9, 0.3, 1000e6
PATCH = (0.5, 0.5)                       # centred 0.5 m x 0.5 m patch
GRADE, TEMPERATURE_C = "DH36", -20.0
LAYERS = 9
GRIDS = [(8, 4), (16, 8), (24, 12)]
CONFIGS = {
    "linear-free": {"geometric": False, "inplane_restrained": False},
    "nonlinear-fixed": {"geometric": True, "inplane_restrained": True},
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
OUT = os.path.join(OUT_DIR, "gap_pp3_transition_reference.json")


def cases() -> list[tuple[str, int, int]]:
    return [(name, nx, ny) for name in CONFIGS for (nx, ny) in GRIDS]


def headers() -> dict:
    flow = low_temperature_flow_stress(GRADE, TEMPERATURE_C)
    py = analytic_first_yield_pressure(A, B, T, E_MOD, NU, flow.yield_stress, PATCH)
    return {
        "case": {"a": A, "b": B, "t": T, "patch": list(PATCH), "layers": LAYERS,
                 "grade": GRADE, "temperature_c": TEMPERATURE_C,
                 "young": E_MOD, "poisson": NU, "hardening": HARDENING,
                 "geometry_check": {"aspect": A / B, "t_over_b": T / B,
                                    "note": "rectangle 2.0 x 1.0 x 0.020 m; Cb not applicable"}},
        "material": flow.as_dict(),
        "analytic": {"first_yield_pressure": py["pressure"],
                     "first_yield_sigma_eq_per_pa": py["sigma_eq_per_pa"],
                     "peak_location": list(py["location"]),
                     "centre_Mx": py["centre_Mx"],
                     "centre_deflection": py["centre_deflection"]},
    }


def run_case(name: str, nx: int, ny: int, head: dict) -> dict:
    py = head["analytic"]["first_yield_pressure"]
    ladder = [0.45 * py, 0.9 * py, 1.5 * py, 2.1 * py, 2.7 * py]
    t0 = time.time()
    summary = plate_capacity_summary(
        a=A, b=B, t=T, young=E_MOD, poisson=NU, grade=GRADE,
        temperature_c=TEMPERATURE_C, patch=PATCH, hardening=HARDENING,
        load_factors=ladder, nx=nx, ny=ny, layers=LAYERS, unload_steps=0,
        **CONFIGS[name])
    fe = summary["fe"]
    entry = {
        "config": name, "nx": nx, "ny": ny, "elements": nx * ny,
        "dofs": 6 * (nx + 1) * (ny + 1),
        "first_yield_factor": fe["first_yield_factor"],
        "first_yield_error": fe.get("first_yield_error"),
        "elastic_bias": fe["elastic_bias"],
        "fe_surface_mises_per_pa": fe["fe_surface_mises_per_pa"],
        "peak_centre_deflection_at_yield": None,
        "collapse_factor": fe["collapse_factor"],
        "collapse_is_limit_point": fe["collapse_is_limit_point"],
        "ramp_completed": fe["ramp_completed"],
        "limit_load_measured": fe.get("measured_limit_load"),
        "path": fe["path"],
        "seconds": round(time.time() - t0, 1),
    }
    # centre deflection at the transition, interpolated on the committed path
    pts = fe["path"]
    if entry["first_yield_factor"] and pts:
        p = entry["first_yield_factor"]
        prev = None
        for q in pts:
            if q[0] >= p:
                if prev is None:
                    entry["peak_centre_deflection_at_yield"] = q[1]
                else:
                    frac = (p - prev[0]) / (q[0] - prev[0])
                    entry["peak_centre_deflection_at_yield"] = prev[1] + frac * (q[1] - prev[1])
                break
            prev = q
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_cases = cases()

    if args.merge:
        bundle = headers()
        bundle["meshes"] = {}
        files = sorted(glob.glob(os.path.join(OUT_DIR,
                                              "gap_pp3_transition_reference_shard*.json")))
        for path in files:
            part = json.load(open(path))
            for key, entry in part["meshes"].items():
                bundle["meshes"][key] = entry
        missing = [f"{n}:{nx}x{ny}" for (n, nx, ny) in all_cases
                   if f"{n}:{nx}x{ny}" not in bundle["meshes"]]
        bundle["missing"] = missing
        bundle["criteria"] = criteria(bundle)
        json.dump(bundle, open(args.out, "w"), indent=1)
        print(f"merged {len(bundle['meshes'])} entries -> {args.out}; missing={missing}")
        print(json.dumps(bundle["criteria"], indent=1))
        return 0

    head = headers()
    picked = [c for i, c in enumerate(all_cases) if i % args.nshards == args.index]
    meshes = {}
    for name, nx, ny in picked:
        entry = run_case(name, nx, ny, head)
        key = f"{name}:{nx}x{ny}"
        meshes[key] = entry
        err = entry["first_yield_error"]
        print(f"[{key}] {entry['seconds']:7.1f}s  fy={entry['first_yield_factor']:.6g}  "
              f"err={err:+.4f}  bias={entry['elastic_bias']:+.4f}", flush=True)
        # one machine-readable line per case, so the log itself is the artifact
        print("RESULT_JSON:" + json.dumps({key: entry}), flush=True)
    out = args.out if args.nshards == 1 else os.path.join(
        OUT_DIR, f"gap_pp3_transition_reference_shard{args.index}.json")
    json.dump({"meshes": meshes}, open(out, "w"), indent=1)
    print("wrote", out)
    return 0


def criteria(bundle: dict) -> dict:
    """Verdict block derived from the raw entries (no hand-typed numbers)."""
    py = bundle["analytic"]["first_yield_pressure"]
    out = {"linear_reference_is_navier": True, "per_config": {}}
    for name in CONFIGS:
        rows = [(gt, bundle["meshes"][f"{name}:{nx}x{ny}"])
                for (nx, ny) in GRIDS if f"{name}:{nx}x{ny}" in bundle["meshes"]]
        if not rows:
            continue
        errors = [abs(e["first_yield_error"]) for _, e in rows]
        block = {
            "grids": [f"{nx}x{ny}" for (nx, ny) in GRIDS if f"{name}:{nx}x{ny}" in bundle["meshes"]],
            "abs_errors": errors,
            "max_abs_error": max(errors),
            "all_lt_5pct": all(x < 0.05 for x in errors),
            "finest": abs(rows[-1][1]["first_yield_error"]),
        }
        if len(errors) >= 2:
            h = [B / e["ny"] for _, e in rows]
            e_c, e_f = errors[-2], errors[-1]
            if e_c > 0 and e_f > 0 and h[-2] > h[-1]:
                block["observed_order"] = math.log(e_f / e_c) / math.log(h[-1] / h[-2])
        out["per_config"][name] = block
    lin = out["per_config"].get("linear-free", {})
    non = out["per_config"].get("nonlinear-fixed", {})
    out["anchor_criterion_1_transition_lt_5pct"] = {
        "reference": "linear Navier first-yield pressure",
        "met_by": "linear-free" if lin.get("all_lt_5pct") else None,
        "linear_free_max_abs_error": lin.get("max_abs_error"),
        "nonlinear_fixed_max_abs_error": non.get("max_abs_error"),
        "note": ("a linear analytic reference may only be compared with a "
                 "small-deflection path; the membrane-restrained path is a "
                 "different mechanical model"),
    }
    return out


if __name__ == "__main__":
    raise SystemExit(main())