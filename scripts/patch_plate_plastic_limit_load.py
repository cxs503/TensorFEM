"""Limit-load ladder for the plate panel under a patch load (anchor GAP-PP-3).

The previous certificate reported the *last converged ramp factor* as the plastic
collapse load and compared it with the yield-line upper bound (+41.6 %, an
inadmissible over-strength).  This ladder establishes the real mechanism:

* the Mindlin layered Shell4 model has **no discrete limit point under load
  control** -- the elastic transverse-shear stiffness keeps the tangent positive
  definite, so once the section moments saturate the deflection runs away at
  almost constant load;
* the saturation plateau is the *layered section* capacity
  ``m_p,n = m_p (1 - 1/n^2)`` for odd ``n`` (measured: 0.889 at n = 3) and
  ``m_p`` for even ``n``;
* the limit load must therefore be read from the load-deflection path with an
  explicit definition (see :func:`tensorfem.plate_patch_plastic_limit.limit_load_from_path`)
  and compared with the *tightest* upper bound (Ilyushin double-sine), not with
  the last ramp factor.

Usage (one shard per cluster task)::

    PYTHONPATH=src python scripts/patch_plate_plastic_limit_load.py --shard 0 --shards 6
    PYTHONPATH=src python scripts/patch_plate_plastic_limit_load.py --configs 8:4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    limit_load_from_path,
    low_temperature_flow_stress,
    plastic_bound_bracket,
    run_patch_plastic_path,
    section_moment_capacity,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
GRADE, TEMP_C = "DH36", -40.0
PATCH = (0.5 * A, 0.5 * B)
HARDENING = 1e6          # near-ideal plasticity: the limit load is a limit load
FACTORS = (0.4, 0.8, 1.2, 1.6, 2.0)

#: ``(nx, layers, shear_scale, unload_steps)``
CONFIGS = (
    (8, 4, 1.0, 3),      # mesh ladder, even layers -> section capacity = m_p exactly
    (12, 4, 1.0, 0),
    (16, 4, 1.0, 0),
    (8, 8, 1.0, 0),      # layer sensitivity at fixed mesh
    (8, 3, 1.0, 0),      # odd layers: capacity deficit (8/9) must show up
    (8, 4, 0.25, 0),     # shear-dominance probe (soft shear)
    (8, 4, 4.0, 0),      # shear-dominance probe (stiff shear)
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def run_config(nx: int, layers: int, shear_scale: float, unload_steps: int) -> dict:
    flow = low_temperature_flow_stress(GRADE, TEMP_C)
    fy = flow.yield_stress
    first_yield = analytic_first_yield_pressure(A, B, T, E, NU, fy, PATCH)
    p_y = first_yield["pressure"]
    t0 = time.time()
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=fy, hardening=HARDENING,
        patch=PATCH, load_factors=[f * p_y for f in FACTORS], nx=nx, ny=nx,
        layers=layers, unload_steps=unload_steps, exact_thickness=True,
        shear_correction=5.0 / 6.0 * shear_scale,
        max_deflection_factor=3.0, bisection_tolerance=1e-2)
    elapsed = time.time() - t0
    estimate = limit_load_from_path(path, criteria=(0.1, 0.2, 0.3))
    capacity = section_moment_capacity(T, fy, layers)
    record = {
        "nx": nx, "layers": layers, "shear_scale": shear_scale,
        "p_yield_analytic": p_y,
        "section_capacity": capacity,
        "limit_load": estimate,
        "first_yield": path.first_yield_factor,
        "ramp_completed": path.ramp_completed,
        "ramp_truncated": path.ramp_truncated,
        "collapse_factor_from_ramp": path.collapse_factor,
        "peak_factor": path.loading[-1].load_factor,
        "permanent_deflection": path.permanent_deflection,
        "permanent_set_valid": path.permanent_set_valid,
        "path": [{"p": s.load_factor, "p_over_py": s.load_factor / p_y,
                  "w": s.centre_deflection, "alpha": s.max_alpha,
                  "moment_ratio": s.max_moment_ratio,
                  "sigma_surface_over_fy": s.max_surface_mises / fy}
                 for s in path.loading],
        "elapsed_s": elapsed,
    }
    if path.unloading:
        record["unloading"] = [{"p": s.load_factor, "w": s.centre_deflection}
                               for s in path.unloading]
    limit = estimate["limit_load"]
    if limit:
        record["limit_error_vs_sine"] = limit / ilyushin_sine_upper_bound(
            A, B, T, fy, PATCH)["pressure"] - 1.0
        record["limit_error_vs_yield_line"] = limit / yield_line_patch_upper_bound(
            A, B, T, fy, PATCH)["pressure"] - 1.0
        record["limit_error_vs_strip_lower"] = limit / strip_lower_bound(
            A, B, T, fy, PATCH)["pressure"] - 1.0
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--configs", type=str, default=None,
                        help="explicit list, e.g. 8:4 or 8:3:0.25:2 (nx:layers:shear:unload)")
    args = parser.parse_args(argv)

    configs = CONFIGS
    if args.configs:
        configs = []
        for token in args.configs.split(","):
            parts = [float(v) for v in token.split(":")]
            nx, layers = int(parts[0]), int(parts[1])
            shear = parts[2] if len(parts) > 2 else 1.0
            unload = int(parts[3]) if len(parts) > 3 else 0
            configs.append((nx, layers, shear, unload))

    flow = low_temperature_flow_stress(GRADE, TEMP_C)
    fy = flow.yield_stress
    bracket = plastic_bound_bracket(A, B, T, fy, PATCH, terms=1)
    first_yield = analytic_first_yield_pressure(A, B, T, E, NU, fy, PATCH)
    bracket.update({"p_yield_analytic": first_yield["pressure"],
                    "yield_stress": fy, "grade": GRADE, "temperature_c": TEMP_C,
                    "hardening": HARDENING, "factors": list(FACTORS),
                    "t_over_a": T / A, "patch_fraction": PATCH[0] * PATCH[1] / (A * B)})

    for index, config in enumerate(configs):
        if index % args.shards != args.shard:
            continue
        nx, layers, shear, unload = config
        label = f"nx{nx}_n{layers}_s{shear:g}"
        record = run_config(nx, layers, shear, unload)
        record["bracket"] = bracket
        out = ART / f"patch_plate_limit_load_{label}.json"
        out.write_text(json.dumps(record, indent=2))
        limit = record["limit_load"]["limit_load"]
        if limit is None:
            print(f"[{label}] NO LIMIT LOAD EXTRACTED; path = "
                  f"{[(round(q['p_over_py'], 3), round(q['w'] * 1e3, 2), round(q['moment_ratio'], 3)) for q in record['path']]}"
                  f" [{record['elapsed_s']:.0f} s]", flush=True)
            continue
        print(f"[{label}] limit {limit/1e6:.4f} MPa "
              f"({100 * record.get('limit_error_vs_sine', float('nan')):+.2f} % vs sine, "
              f"{100 * record.get('limit_error_vs_yield_line', float('nan')):+.2f} % vs yield line) "
              f"spread {record['limit_load']['spread']:.4f} "
              f"plateau |M|/mp {(record['path'][-1]['moment_ratio']):.4f} "
              f"(1-1/n^2 = {record['section_capacity']['ratio']:.4f}) "
              f"w_peak {record['path'][-1]['w']*1e3:.1f} mm "
              f"[{record['elapsed_s']:.0f} s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())