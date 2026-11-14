"""Independent bound verification for anchor GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

Reproduces the four anchor acceptance quantities from a clean script and writes
``artifacts/gap_pp3_bound_bracket.json``:

* elastic branch vs the Navier double series (same mesh, raw error),
* elastic -> plastic transition pressure vs the analytic first-yield pressure
  (raw error, *no* same-mesh normalisation),
* the two independent kinematic upper bounds and the static lower bound,
* the permanent-set read-out on the unload leg.

Everything is printed as it is computed; nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State,
    solve_layered_shell4_increment,
)
from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    elastic_patch_plate,
    ilyushin_sine_upper_bound,
    patch_plate_mesh,
    plastic_bound_bracket,
    plastic_moment,
    run_patch_plastic_path,
    strip_lower_bound,
    surface_extrapolated_mises,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
E = 210e9
NU = 0.3
SY = 355e6
HARDENING = 2e10
PATCH = (0.5, 0.5)
MESHES = (8, 12, 16)

out: dict = {"configuration": {"a": A, "b": B, "t": T, "young": E, "poisson": NU,
                               "yield_stress": SY, "hardening": HARDENING,
                               "patch": list(PATCH), "meshes": list(MESHES)}}

mp = plastic_moment(T, SY)
bracket = plastic_bound_bracket(A, B, T, SY, PATCH)
ref = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH)
w_navier = elastic_patch_plate(A, B, T, E, NU, PATCH, 1.0,
                               points=[(0.5 * A, 0.5 * B)])["deflection"][0]

out["analytic"] = {
    "plastic_moment_Nm_per_m": mp,
    "first_yield_pressure_Pa": ref["pressure"],
    "first_yield_location": list(ref["location"]),
    "surface_mises_per_Pa": ref["sigma_eq_per_pa"],
    "navier_centre_deflection_per_Pa_m": w_navier,
    "strip_lower_bound_Pa": bracket["lower_strip"],
    "yield_line_upper_bound_Pa": bracket["upper_yield_line"],
    "ilyushin_sine_upper_bound_Pa": bracket["upper_ilyushin_sine"],
    "tightest_upper_bound_Pa": bracket["tightest_upper"],
    "bracket_width_upper_over_lower": bracket["bracket_width_ratio"],
    "upper_bound_agreement_pct": bracket["agreement_of_upper_bounds_pct"],
}

print(f"m_p                       = {mp:.6e} N m/m")
print(f"analytic first yield      = {ref['pressure']:.6e} Pa at {ref['location']}")
print(f"strip lower bound         = {bracket['lower_strip']:.6e} Pa")
print(f"yield-line upper bound    = {bracket['upper_yield_line']:.6e} Pa")
print(f"Ilyushin sine upper bound = {bracket['upper_ilyushin_sine']:.6e} Pa")
print(f"upper bounds agree within = {bracket['agreement_of_upper_bounds_pct']:.2f} %")
print(f"bracket width (U/L)       = {bracket['bracket_width_ratio']:.3f}")

meshes = []
for nx in MESHES:
    model, info = patch_plate_mesh(A, B, T, nx, nx, young=E, poisson=NU,
                                   yield_stress=SY, hardening=HARDENING,
                                   layers=5, patch=PATCH)
    state = LayeredShell4State.virgin(model)
    step = solve_layered_shell4_increment(
        model, info["reference_load"], 1.0, info["fixed_dofs"], state,
        torch.zeros(model.n_dofs, dtype=torch.float64),
        tolerance=1e-10, max_iterations=30)
    centre = (nx // 2) * (nx + 1) + nx // 2
    w_fe = float(step.displacement[6 * centre + 2])
    sigma_per_pa = surface_extrapolated_mises(step.stress, T)
    p_yield_fe = SY / sigma_per_pa
    entry = {
        "mesh": nx,
        "elastic_centre_deflection_per_Pa_m": w_fe,
        "elastic_deflection_error_pct": 100.0 * (w_fe / w_navier - 1.0),
        "fe_surface_mises_per_Pa": sigma_per_pa,
        "fe_first_yield_pressure_Pa": p_yield_fe,
        "first_yield_error_pct": 100.0 * (p_yield_fe / ref["pressure"] - 1.0),
    }
    print(f"nx={nx:2d}  w err {entry['elastic_deflection_error_pct']:+6.2f} %   "
          f"p_yield(FE) {p_yield_fe:.6e} Pa   raw error {entry['first_yield_error_pct']:+6.2f} %")

    if nx == MESHES[0]:
        path = run_patch_plastic_path(
            A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
            patch=PATCH, load_factors=[6.0e5, 1.2e6], nx=nx, ny=nx, layers=5,
            unload_steps=2)
        entry["path_first_yield_factor_Pa"] = path.first_yield_factor
        entry["path_permanent_deflection_m"] = path.permanent_deflection
        entry["path_points"] = [
            {"pressure_Pa": pt.load_factor, "centre_deflection_m": pt.centre_deflection,
             "max_alpha": pt.max_alpha, "max_surface_mises_Pa": pt.max_surface_mises}
            for pt in path.loading]
        entry["unload_points"] = [
            {"pressure_Pa": pt.load_factor, "centre_deflection_m": pt.centre_deflection}
            for pt in path.unloading]
        print(f"      path first yield {path.first_yield_factor:.4e} Pa, "
              f"permanent set {path.permanent_deflection:.4e} m")
    meshes.append(entry)

out["meshes"] = meshes

firsts = [m["first_yield_error_pct"] for m in meshes]
out["verdicts"] = {
    "elastic_branch_max_abs_error_pct": max(abs(m["elastic_deflection_error_pct"])
                                            for m in meshes),
    "first_yield_raw_error_pct_range": [min(firsts), max(firsts)],
    "first_yield_within_5pct_at_every_mesh": all(abs(e) < 5.0 for e in firsts),
    "mesh_convergence_monotone_in_first_yield": (
        abs(firsts[0]) > abs(firsts[1]) and abs(firsts[1]) > abs(firsts[2])),
    "upper_bounds_agree_within_10pct": bracket["agreement_of_upper_bounds_pct"] < 10.0,
    "bracket_width_ratio": bracket["bracket_width_ratio"],
}
print(json.dumps(out["verdicts"], indent=2))

target = Path(__file__).resolve().parents[1] / "artifacts" / "gap_pp3_bound_bracket.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {target}")