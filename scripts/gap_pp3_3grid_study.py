#!/usr/bin/env python
"""3-grid FE verification matrix for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

Purpose
-------
The anchor (板格低温钢弹塑性 + 永久挠度) has three measurable acceptance
criteria:

1. elastic -> plastic transition point vs the analytical yield load, < 5 %;
2. plastic limit load vs Ilyushin / upper-bound analysis, < 10 %;
3. 3-grid convergence (the criterion must be met as a *mesh limit*, not by
   cancelling errors on one arbitrary mesh);
4. permanent deflection is a reported post-processing quantity.

This driver produces the numerical evidence for all four, on the *same*
geometry / material / patch family as the platform case, and writes
``outputs/gap_pp3_3grid_results.json`` which
``tests/test_gap_pp3_plastic_set_fe_verification.py::Test3GridConvergenceFromCluster``
consumes.

Both the *raw* and the *same-mesh normalised* transition error are reported.
The raw error is the honest comparison against the Navier analytic value; the
normalised one removes the element's own elastic stress bias.  The two are NOT
independent evidence (see docs/CASE-GAP-PP-3-3GRID-LIMIT-EVIDENCE.md, finding
N1): the normalised error is ~1e-14 by construction, so the acceptance
argument rests on the raw error converging with mesh refinement, which is what
the ``raw_error_convergence`` block reports (observed order + Richardson
extrapolation).

Mesh-family constraint (finding N6)
-----------------------------------
``patch_plate_mesh`` requires *both* patch edges to fall on element edges: for
a = 2.0 m, b = 1.0 m and a centred 0.5 m x 0.5 m patch that means
``nx % 8 == 0`` and ``ny % 4 == 0``.  The family used up to 2026-11-15,
``(8,4) (12,6) (16,8)``, violates it at 12x6 and raised ``ValueError`` after
the first mesh, so the 3-grid matrix never produced a complete data set
(compute job cj4232: rc = 1).  ``GRIDS`` is now the aligned family
``(8,4) (16,8) (24,12)`` (mesh ratio 2 in both directions) and
``check_alignment`` rejects an infeasible family *before* any solve starts.

Sharding
--------
One mesh costs minutes on the cluster, so the matrix can be split:
``--grids 16x8`` writes a single-mesh JSON; ``scripts/gap_pp3_3grid_merge.py``
combines the shards and computes the criteria block.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    analytic_first_yield_pressure,
    plastic_bound_bracket,
    section_moment_capacity,
    low_temperature_flow_stress,
    plate_capacity_summary,
    yield_line_patch_upper_bound,
)

# ---------------------------------------------------------------------------
# Case: ice-loaded deck / shell panel, DH36 at its design temperature
# ---------------------------------------------------------------------------
A, B, T = 2.0, 1.0, 0.020          # plate length, width, thickness [m]
PATCH = (0.5, 0.5)                 # ice patch footprint [m] (mesh-aligned on 8x4+)
E_MOD, NU = 206e9, 0.3
HARDENING = 1000e6                 # isotropic hardening modulus [Pa]
GRADE = "DH36"
TEMPERATURE_C = -20.0              # design temperature of DH36 (DBTT = -65 C)
# Centred patch -> the element size must divide the patch offsets (0.75/1.25 m in
# x, 0.25/0.75 m in y), i.e. nx % 8 == 0 and ny % 4 == 0.  Ratio 2 everywhere.
GRIDS = [(8, 4), (16, 8), (24, 12)]
LAYERS = 9
UNLOAD_STEPS = 3

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "gap_pp3_3grid_results.json")


def check_alignment(nx: int, ny: int) -> None:
    """Reject a mesh on which the patch footprint is not element-aligned.

    Cheap *pre-flight* twin of the guard inside
    :func:`tensorfem.plate_patch_plastic_limit.patch_plate_mesh`; the point is to
    fail in milliseconds instead of after the first (expensive) solve of the
    matrix, which is exactly what happened on compute job cj4232.
    """
    u, v = PATCH
    for offset, size, length, n, axis in (
        (0.5 * (A - u), u, A, nx, "x"),
        (0.5 * (B - v), v, B, ny, "y"),
    ):
        step = length / n
        for point in (offset, offset + size):
            ratio = point / step
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"mesh {nx}x{ny} cannot carry the {u} x {v} m patch: the "
                    f"{axis} patch edge at {point:.4f} m is not on an element edge "
                    f"(element size {step:.6f} m). Aligned families need "
                    f"nx % 8 == 0 and ny % 4 == 0 for a = {A} m, b = {B} m."
                )


def _load_factors(py: float) -> list[float]:
    """Load ladder in Pa, scaled by the analytical first-yield pressure.

    The upper end is set beyond the yield-line upper bound so that the path has
    to run through the mechanism; increments that converge onto a state outside
    the section are bisected by :func:`run_patch_plastic_path`.
    """
    return [round(f * py, 3) for f in (0.45, 0.90, 1.50, 2.10, 2.70)]


def _rules(errors: list[float], grids: list[tuple[int, int]]) -> dict:
    """Observed order of convergence and a Richardson limit for an error sequence.

    ``h`` is the element size along the short side, ``b / ny``.  With
    ``e(h) = e_lim + C h^p`` the order from the last two meshes is

        p = ln(e_f / e_c) / ln(h_f / h_c)          (h_f < h_c  ->  both logs < 0)

    and the mesh limit is ``e_lim = e_f + (e_f - e_c) / (r^p - 1)`` with
    ``r = h_c / h_f``.

    Finding N7 (mission m22746): the pre-fix version divided by
    ``ln(h_c / h_f)`` -- the reciprocal of the correct denominator -- so the
    reported ``observed_order`` had the wrong sign and ``extrapolated_error``
    was wrong by a factor of order one (on an exact p = 1.5 power law it
    returned order = -1.5 and a limit of 0.068 instead of 0.0, i.e. it would
    have *failed* an exactly convergent sequence).  Both are asserted in
    ``tests/test_gap_pp3_3grid_mesh_family.py``.
    """
    h = [B / ny for (_, ny) in grids]
    out = {"h": h, "errors": errors}
    if len(errors) < 2:
        return out
    e_c, e_f = errors[-2], errors[-1]
    h_c, h_f = h[-2], h[-1]
    if e_c == 0.0 or e_f == 0.0 or h_c <= h_f:
        return out
    out["refinement_ratio"] = h_c / h_f
    if e_f * e_c < 0:                     # sign change: no meaningful order
        out["note"] = "error changes sign between the two finest meshes"
        return out
    order = math.log(abs(e_f / e_c)) / math.log(h_f / h_c)
    out["observed_order"] = order
    r = h_c / h_f
    denom = r ** order - 1.0
    if abs(denom) > 1e-12:
        out["extrapolated_error"] = e_f + (e_f - e_c) / denom
    return out


def headers() -> dict:
    """Case / material / analytic reference block (mesh independent)."""
    flow = low_temperature_flow_stress(GRADE, TEMPERATURE_C)
    py = analytic_first_yield_pressure(A, B, T, E_MOD, NU, flow.yield_stress, PATCH)
    ub = yield_line_patch_upper_bound(A, B, T, flow.yield_stress, PATCH)
    bracket = plastic_bound_bracket(A, B, T, flow.yield_stress, PATCH)
    ladder = _load_factors(py["pressure"])
    return {
        "case": {
            "a": A, "b": B, "t": T, "patch": list(PATCH),
            "young": E_MOD, "poisson": NU, "hardening": HARDENING,
            "grade": GRADE, "temperature_c": TEMPERATURE_C, "layers": LAYERS,
            "geometry_check": {
                "aspect": A / B, "t_over_b": T / B,
                "note": "rectangle 2.0 x 1.0 x 0.020 m, Cb not applicable (unstiffened plate)",
            },
        },
        "material": flow.as_dict(),
        "analytic": {"first_yield_pressure": py["pressure"],
                     "first_yield_sigma_eq_per_pa": py.get("sigma_eq_per_pa"),
                     "yield_line_upper_pressure": ub["pressure"]},
        # Plastic bracket of the *same* case: an FE collapse load is evidenced by
        # falling inside it, and the bracket width says how strong that evidence is.
        "bounds": {
            "strip_lower_pressure": bracket["lower_strip"],
            "yield_line_upper_pressure": bracket["upper_yield_line"],
            "ilyushin_sine_upper_pressure": bracket["upper_ilyushin_sine"],
            "johansen_twist_upper_pressure": bracket["upper_johansen_twist"],
            "tightest_admissible_upper": bracket["tightest_admissible_upper"],
            "admissible_bracket_width_ratio": bracket["admissible_bracket_width_ratio"],
        },
        "load_ladder_Pa": ladder,
        "meshes": {},
        "criteria": {},
    }


def run_mesh(nx: int, ny: int, results: dict) -> dict:
    """Solve one mesh of the matrix and append its entry to ``results``."""
    check_alignment(nx, ny)
    ladder = results["load_ladder_Pa"]
    ub = results["analytic"]["yield_line_upper_pressure"]
    flow = low_temperature_flow_stress(GRADE, TEMPERATURE_C)
    name = f"{nx}x{ny}"
    t0 = time.time()
    summary = plate_capacity_summary(
        a=A, b=B, t=T, young=E_MOD, poisson=NU, grade=GRADE,
        temperature_c=TEMPERATURE_C, patch=PATCH, hardening=HARDENING,
        load_factors=ladder, nx=nx, ny=ny, layers=LAYERS,
        unload_steps=UNLOAD_STEPS, geometric=True, inplane_restrained=True,
    )
    fe = summary["fe"]

    # The limit load as measured from the path itself (double-slope /
    # offset / stiffness definitions), so the reported number does not
    # depend on "the last prescribed factor of the ramp".
    measured = fe["limit_measurements"]
    limit_pressure = measured.get("limit_load")
    entry = {
        "nx": nx, "ny": ny, "elements": nx * ny, "dofs": 6 * (nx + 1) * (ny + 1),
        "layers": LAYERS,
        "section_capacity_ratio": section_moment_capacity(T, flow.yield_stress, LAYERS)["ratio"],
        "elastic_bias": fe["elastic_bias"],
        "first_yield_factor": fe["first_yield_factor"],
        "first_yield_error": fe["first_yield_error"],
        "first_yield_error_bias_normalised": fe["first_yield_error_bias_normalised"],
        "analytic_first_yield_same_mesh": fe["analytic_first_yield_same_mesh"],
        "collapse_factor": fe["collapse_factor"],
        "collapse_is_limit_point": fe["collapse_is_limit_point"],
        "ramp_completed": fe["ramp_completed"],
        "admissibility": measured.get("admissibility"),
        "limit_measurements": measured.get("measurements"),
        "limit_spread": measured.get("spread"),
        "limit_withheld_reason": measured.get("reason"),
        "limit_load_measured": fe["measured_limit_load"],
        "limit_load_error_vs_ub": (None if limit_pressure is None
                                   else (limit_pressure - ub) / ub),
        "permanent_deflection": summary["fe_permanent_deflection"],
        "permanent_set_valid": summary["fe_permanent_set_valid"],
        "unload_failure": fe["unload_failure"],
        "path_points": len(fe["path"]),
        "path": fe["path"],
        "seconds": round(time.time() - t0, 1),
    }
    results["meshes"][name] = entry
    print(f"[{name}] {entry['seconds']:6.1f}s  bias={entry['elastic_bias']:+.4f}  "
          f"yield_err={entry['first_yield_error']:+.4f}  "
          f"limit={entry['limit_load_measured']}  "
          f"perm={entry['permanent_deflection']}", flush=True)
    return entry


def finalize(results: dict) -> dict:
    """Acceptance-criteria block over whatever meshes are present (coarse -> fine)."""
    names = [n for n, _ in sorted(results["meshes"].items(),
                                 key=lambda kv: kv[1]["elements"])]
    ub = results["analytic"]["yield_line_upper_pressure"]
    raw = [results["meshes"][n]["first_yield_error"] for n in names]
    grids = [(results["meshes"][n]["nx"], results["meshes"][n]["ny"]) for n in names]
    conv = _rules(raw, grids)
    finest = names[-1] if names else None
    finest_raw = results["meshes"][finest]["first_yield_error"] if finest else None
    extrapolated = conv.get("extrapolated_error", None)
    limits = [results["meshes"][n]["limit_load_measured"] for n in names]

    results["criteria"] = {
        "mesh_order": names,
        "raw_error_convergence": conv,
        "transition_lt_5pct": {
            "finest_mesh": finest,
            "raw_error_finest": finest_raw,
            "extrapolated_error": extrapolated,
            "met_at_finest_mesh": (None if finest_raw is None
                                   else bool(abs(finest_raw) < 0.05)),
            "met_in_mesh_limit": bool(extrapolated is not None and abs(extrapolated) < 0.05),
            "note": ("the raw error is the honest FE-vs-Navier comparison; the "
                     "same-mesh normalised error is identically ~0 and carries no "
                     "information (docs finding N1)"),
        },
        "limit_lt_10pct": {
            "finest_mesh": finest,
            "limit_measured": results["meshes"][finest]["limit_load_measured"] if finest else None,
            "upper_bound": ub,
            "error_finest": results["meshes"][finest]["limit_load_error_vs_ub"] if finest else None,
            "limit_trend": limits,
            "met_at_finest_mesh": (None if not finest
                                   or results["meshes"][finest]["limit_load_error_vs_ub"]
                                   is None else
                                   abs(results["meshes"][finest]["limit_load_error_vs_ub"]) < 0.10),
        },
        "permanent_deflection_reported": {
            n: {"value": results["meshes"][n]["permanent_deflection"],
                "valid": results["meshes"][n]["permanent_set_valid"]} for n in names},
    }
    return results


def write(results: dict, out: str) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {os.path.abspath(out)}")
    return results


def parse_grids(text: str) -> list[tuple[int, int]]:
    grids = []
    for token in text.split(","):
        token = token.strip().lower().replace("*", "x")
        if not token:
            continue
        nx, ny = (int(v) for v in token.split("x"))
        grids.append((nx, ny))
    return grids


def main(grids=None, out: str = OUT) -> dict:
    grids = GRIDS if grids is None else grids
    for nx, ny in grids:            # fail fast, before any solve
        check_alignment(nx, ny)

    t_start = time.time()
    results = headers()
    for nx, ny in grids:
        run_mesh(nx, ny, results)
    finalize(results)
    results["seconds_total"] = round(time.time() - t_start, 1)
    write(results, out)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", default=None,
                    help="comma separated mesh list, e.g. '16x8'; default: full family 8x4,16x8,24x12")
    ap.add_argument("--out", default=OUT, help="output JSON path")
    args = ap.parse_args()
    main(parse_grids(args.grids) if args.grids else None, args.out)