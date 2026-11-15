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
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    analytic_first_yield_pressure,
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
GRIDS = [(8, 4), (12, 6), (16, 8)]
LAYERS = 9
UNLOAD_STEPS = 3

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "gap_pp3_3grid_results.json")


def _load_factors(py: float) -> list[float]:
    """Load ladder in Pa, scaled by the analytical first-yield pressure.

    The upper end is set beyond the yield-line upper bound so that the path has
    to run through the mechanism; increments that converge onto a state outside
    the section are bisected by :func:`run_patch_plastic_path`.
    """
    return [round(f * py, 3) for f in (0.45, 0.90, 1.50, 2.10, 2.70)]


def _rules(errors: list[float], grids: list[tuple[int, int]]) -> dict:
    """Observed order of convergence and a Richardson limit for an error sequence.

    ``h`` is taken as the element size along the short side, ``b / ny``.
    The extrapolated value is the limit of the error as h -> 0.
    """
    h = [B / ny for (_, ny) in grids]
    out = {"h": h, "errors": errors}
    if len(errors) >= 2 and errors[-2] != 0.0 and errors[-1] != 0.0:
        ratio = h[-2] / h[-1]
        try:
            order = torch.log(torch.tensor(abs(errors[-1] / errors[-2]))) / torch.log(
                torch.tensor(ratio))
            out["observed_order"] = float(order)
            if abs(1.0 - ratio ** (-float(order))) > 1e-9:
                out["extrapolated_error"] = float(
                    errors[-1] + (errors[-1] - errors[-2]) / (ratio ** float(order) - 1.0))
        except (ValueError, RuntimeError):  # pragma: no cover - defensive
            pass
    return out


def main() -> dict:
    flow = low_temperature_flow_stress(GRADE, TEMPERATURE_C)
    py = analytic_first_yield_pressure(A, B, T, E_MOD, NU, flow.yield_stress, PATCH)
    ub = yield_line_patch_upper_bound(A, B, T, flow.yield_stress, PATCH)
    ladder = _load_factors(py["pressure"])

    results = {
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
        "load_ladder_Pa": ladder,
        "meshes": {},
        "criteria": {},
    }

    t_start = time.time()
    for nx, ny in GRIDS:
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
                                       else (limit_pressure - ub["pressure"]) / ub["pressure"]),
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

    # ---- acceptance criteria -------------------------------------------------
    names = list(results["meshes"])
    raw = [results["meshes"][n]["first_yield_error"] for n in names]
    grids = [(results["meshes"][n]["nx"], results["meshes"][n]["ny"]) for n in names]
    conv = _rules(raw, grids)
    finest = names[-1]
    finest_raw = results["meshes"][finest]["first_yield_error"]
    extrapolated = conv.get("extrapolated_error", None)
    limits = [results["meshes"][n]["limit_load_measured"] for n in names]

    results["criteria"] = {
        "raw_error_convergence": conv,
        "transition_lt_5pct": {
            "finest_mesh": finest,
            "raw_error_finest": finest_raw,
            "extrapolated_error": extrapolated,
            "met_at_finest_mesh": bool(abs(finest_raw) < 0.05),
            "met_in_mesh_limit": bool(extrapolated is not None and abs(extrapolated) < 0.05),
            "note": ("the raw error is the honest FE-vs-Navier comparison; the "
                     "same-mesh normalised error is identically ~0 and carries no "
                     "information (docs finding N1)"),
        },
        "limit_lt_10pct": {
            "finest_mesh": finest,
            "limit_measured": results["meshes"][finest]["limit_load_measured"],
            "upper_bound": ub["pressure"],
            "error_finest": results["meshes"][finest]["limit_load_error_vs_ub"],
            "limit_trend": limits,
            "met_at_finest_mesh": (None if results["meshes"][finest]["limit_load_error_vs_ub"]
                                   is None else
                                   abs(results["meshes"][finest]["limit_load_error_vs_ub"]) < 0.10),
        },
        "permanent_deflection_reported": {
            n: {"value": results["meshes"][n]["permanent_deflection"],
                "valid": results["meshes"][n]["permanent_set_valid"]} for n in names},
    }
    results["seconds_total"] = round(time.time() - t_start, 1)

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {os.path.abspath(OUT)} in {results['seconds_total']}s")
    return results


if __name__ == "__main__":
    main()