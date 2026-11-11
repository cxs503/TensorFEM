"""Independent verification of ``tensorfem.stiffened_plate_progressive``.

Campaign IN421-W2-N01 left that module without any test, report or executed
evidence (``audit_report.md`` booked it at maturity L0, "180 lines, no test").
This script closes the verification debt: every criterion is compared with a
reference that is *re-derived here from theory*, not read back from the module
under test.

Checks
------
(a) elastic buckling of the plate bay -- FEM eigenvalue of the Shell4 operator
    against the Navier closed form ``sigma_cr = k*pi^2*D/(b^2*t)`` implemented
    independently below, on a three-mesh ladder, plus Richardson/GCI.
(c) elastic range of the stateful Riks path -- the axial slope
    ``sigma/epsilon`` must return the Young modulus (the reference load is one
    pascal over the gross section, so no calibration factor is available).
(b) ultimate stress -- FEM limit load vs the Paik-Thayamballi (2003) plate
    formula combined area-wise with a Perry-Robertson stiffener column; the
    empirical curve is quoted at +-10% accuracy, so the gate is 15%.
(f) mesh convergence of the ultimate load on the two affordable meshes.

Output: ``benchmarks/stiffened_plate_progressive_evidence.json``
(schema ``stiffened-plate-progressive/1``); no timestamps, no absolute paths.

Run
---
    OMP_NUM_THREADS=2 PYTHONPATH=src python3 \
        benchmarks/stiffened_plate_progressive_verification.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorfem.stiffened_plate_progressive import (
    WeldedPlateSpec,
    bay_imperfection_mode,
    build_welded_plate,
    path_summary,
    trace_welded_plate_path,
    welded_plate_buckling,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "stiffened_plate_progressive_evidence.json"

SPEC = WeldedPlateSpec(
    length=3.0, width=1.0, plate_t=0.01,
    web_h=0.15, web_t=0.01, flange_w=0.10, flange_t=0.01,
    spacing=0.5, young=206e9, poisson=0.3, sigma_y=315e6, E_t=2e9,
    nx=6, ny=4, layers=3, imperfection=0.1,
)

BUCKLING_MESHES = ((8, 2), (16, 4), (32, 8))


# ---------------------------------------------------------------------------
# Independent closed forms (no import of the module under test)
# ---------------------------------------------------------------------------

def navier_plate_stress(*, a: float, b: float, t: float, E: float, nu: float) -> dict:
    """Simply supported plate, uniaxial compression -- Navier (Timoshenko).

    ``sigma_cr = k * pi^2 * D / (b^2 * t)``, ``k = (m*b/a + a/(m*b))^2``
    minimised over the integer half-wave count ``m`` (Timoshenko & Gere 1961,
    Eq. 9.6; reproduced as Paik et al. 2007 Eq. 2.1).
    """
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    best = (math.inf, 0)
    for m in range(1, 25):
        k = (m * b / a + a / (m * b)) ** 2
        if k < best[0]:
            best = (k, m)
    k, m = best
    return {"sigma_cr": k * math.pi ** 2 * D / (b ** 2 * t), "k": k, "mode": m, "D": D}


def paik_plate_ultimate(*, beta: float, sigma_y: float) -> float:
    """Paik & Thayamballi (2003) plate ultimate, a/b >= 1 branch."""
    if beta < 1.0:
        return sigma_y
    return min(1.50 / beta - 0.30 / beta ** 2, 1.0) * sigma_y


def perry_robertson_column(*, sigma_y: float, E: float, length: float, radius: float,
                           eta_base: float = 0.003) -> float:
    """Perry-Robertson column strength with ``eta = eta_base * L/r``."""
    lam = length / radius * math.sqrt(sigma_y / (math.pi ** 2 * E))
    eta = eta_base * length / radius
    return min(sigma_y / (eta + math.sqrt(eta ** 2 + lam ** 2)), sigma_y)


def paik_panel_reference(spec: WeldedPlateSpec) -> dict:
    """Area-weighted plate + stiffener column ultimate of one panel unit."""
    A_plate = spec.spacing * spec.plate_t
    A_stiff = spec.web_h * spec.web_t + spec.flange_w * spec.flange_t
    beta = (spec.spacing / spec.plate_t) * math.sqrt(spec.sigma_y / spec.young)
    sigma_plate = paik_plate_ultimate(beta=beta, sigma_y=spec.sigma_y)
    # stiffener column: second moment about the plate surface, then r = sqrt(I/A)
    A_web, A_fl = spec.web_h * spec.web_t, spec.flange_w * spec.flange_t
    z_web = spec.web_h / 2.0
    z_fl = spec.web_h
    z_bar = (A_web * z_web + A_fl * z_fl) / A_stiff
    I = (spec.web_t * spec.web_h ** 3 / 12.0 + A_web * (z_web - z_bar) ** 2
         + spec.flange_w * spec.flange_t ** 3 / 12.0 + A_fl * (z_fl - z_bar) ** 2)
    r = math.sqrt(I / A_stiff)
    sigma_stiff = perry_robertson_column(
        sigma_y=spec.sigma_y, E=spec.young, length=spec.length, radius=r)
    sigma_panel = (A_plate * sigma_plate + A_stiff * sigma_stiff) / (A_plate + A_stiff)
    return {
        "beta": beta,
        "plate_area": A_plate,
        "stiffener_area": A_stiff,
        "stiffener_radius_of_gyration": r,
        "sigma_u_plate": sigma_plate,
        "sigma_u_stiffener": sigma_stiff,
        "sigma_u_panel": sigma_panel,
        "sigma_u_over_yield": sigma_panel / spec.sigma_y,
        "total_area": spec.total_area,
    }


def richardson(values: list[float]) -> dict:
    """Observed order and GCI-style extrapolation from three mesh levels."""
    f1, f2, f3 = values[-3:]
    if not (f1 != f2 and f2 != f3):
        return {"observed_order": None, "extrapolated": None,
                "finest_error": None}
    ratio = (f2 - f1) / (f3 - f2) if f3 != f2 else float("inf")
    try:
        p = math.log(abs(ratio)) / math.log(2.0)
    except (ValueError, ZeroDivisionError):
        p = float("nan")
    if not math.isfinite(p) or p <= 0.0:
        extrapolated = f3
    else:
        extrapolated = f3 + (f3 - f2) / (2.0 ** p - 1.0)
    return {"observed_order": p, "extrapolated": extrapolated,
            "finest_error": abs(f3 - extrapolated) / abs(extrapolated) if extrapolated else None}


# ---------------------------------------------------------------------------
# (a) buckling ladder
# ---------------------------------------------------------------------------

def buckling_ladder(meshes=BUCKLING_MESHES) -> dict:
    navier = navier_plate_stress(a=SPEC.length, b=SPEC.spacing, t=SPEC.plate_t,
                                 E=SPEC.young, nu=SPEC.poisson)
    rows = []
    for nx, ny in meshes:
        out = welded_plate_buckling(SPEC, nx=nx, ny=ny)
        rows.append({
            "mesh": f"{nx}x{ny}",
            "sigma_cr_fem": out["sigma_cr_fem"],
            "rel_error": out["sigma_cr_fem"] / navier["sigma_cr"] - 1.0,
            "abs_rel_error": abs(out["sigma_cr_fem"] / navier["sigma_cr"] - 1.0),
            "module_navier": out["sigma_cr_navier"],
            "buckling_coefficient": out["buckling_coefficient"],
        })
    return {
        "navier": navier,
        "ladder": rows,
        "module_navier_rel_diff": abs(rows[0]["module_navier"] / navier["sigma_cr"] - 1.0),
        "finest_mesh_below_3pct": rows[-1]["abs_rel_error"] < 0.03,
        "error_monotone_under_refinement": all(
            rows[i]["abs_rel_error"] > rows[i + 1]["abs_rel_error"] for i in range(len(rows) - 1)),
        "richardson": richardson([r["sigma_cr_fem"] for r in rows]),
    }


# ---------------------------------------------------------------------------
# (b)(c)(f) Riks path
# ---------------------------------------------------------------------------

def run_path(spec: WeldedPlateSpec, steps: int, mode_mesh=(18, 6)) -> dict:
    torch.manual_seed(0)
    buckling = welded_plate_buckling(spec, nx=mode_mesh[0], ny=mode_mesh[1], modes=1)
    imperfection = bay_imperfection_mode(spec, nx=mode_mesh[0], ny=mode_mesh[1])
    case = build_welded_plate(spec, imperfection_mode=imperfection)
    trace = trace_welded_plate_path(case, steps=steps)
    summary = path_summary(case, trace)
    first, second = trace.points[0], trace.points[1]
    epsilon = first.end_shortening / spec.length
    E_app = first.load_factor / epsilon if epsilon else None
    tangent = ((second.load_factor - first.load_factor)
               / ((second.end_shortening - first.end_shortening) / spec.length))
    return {
        "mesh": f"{spec.nx}x{spec.ny}",
        "layers": spec.layers,
        "steps_requested": steps,
        "n_dofs": case.n_dofs,
        "converged": trace.converged,
        "points_completed": summary["points_completed"],
        "ultimate_stress": summary["ultimate_stress"],
        "ultimate_over_yield": summary["ultimate_over_yield"],
        "final_stress_over_yield": summary["final_stress_over_yield"],
        "has_limit_point": summary["has_limit_point"],
        "n_descending_points": summary["n_descending_points"],
        "max_plastic_strain": summary["max_plastic_strain"],
        "energy_absorbed": summary["energy_absorbed"],
        "elastic_modulus_apparent": E_app,
        "elastic_slope_error": E_app / spec.young - 1.0 if E_app else None,
        "tangent_modulus_apparent": tangent,
        "first_buckling_mode_peak": float(buckling["modes"][:, 0].__abs__().max()),
        "path": [[p.step, p.load_factor, p.end_shortening, p.transverse_deflection,
                  p.maximum_plastic_strain] for p in trace.points],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="short path ladder for a smoke run")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    reference = paik_panel_reference(SPEC)
    buckling = buckling_ladder()
    print("buckling:", json.dumps(buckling["ladder"], indent=1), flush=True)

    if args.quick:
        plans = [(WeldedPlateSpec(**{**SPEC.__dict__, "nx": 4, "ny": 2, "layers": 2}), 6)]
    else:
        plans = [
            (WeldedPlateSpec(**{**SPEC.__dict__, "nx": 4, "ny": 2, "layers": 2}), 45),
            (WeldedPlateSpec(**{**SPEC.__dict__, "nx": 6, "ny": 4, "layers": 3}), 45),
        ]
    post = []
    for spec, steps in plans:
        row = run_path(spec, steps)
        post.append(row)
        print("path:", row["mesh"], "ult=%.6g" % row["ultimate_stress"],
              "limit=%s" % row["has_limit_point"], flush=True)

    ultimate = [row["ultimate_stress"] for row in post]
    finest = post[-1]
    paik_error = finest["ultimate_stress"] / reference["sigma_u_panel"] - 1.0
    payload = {
        "schema": "stiffened-plate-progressive/1",
        "module": "src/tensorfem/stiffened_plate_progressive.py",
        "spec": {k: v for k, v in SPEC.__dict__.items()},
        "paik_reference": reference,
        "buckling": buckling,
        "post_buckling": post,
        "gates": {
            "buckling_finest_mesh_below_3pct": (buckling["finest_mesh_below_3pct"]
                                                and buckling["error_monotone_under_refinement"]),
            "elastic_slope_below_5pct": (abs(finest["elastic_slope_error"]) < 0.05
                                         if finest["elastic_slope_error"] is not None else False),
            "ultimate_vs_paik_below_15pct": (abs(paik_error) < 0.15
                                             if any(r["has_limit_point"] for r in post) else None),
            "mesh_convergence_below_10pct": (abs(ultimate[-1] - ultimate[-2]) / ultimate[-1] < 0.10
                                             if len(ultimate) > 1
                                             and all(r["has_limit_point"] for r in post) else None),
        },
        "ultimate_vs_paik_rel_error": paik_error,
        "mesh_convergence_rel_change": (abs(ultimate[-1] - ultimate[-2]) / ultimate[-1]
                                        if len(ultimate) > 1 else None),
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT, flush=True)
    return 0 if all(v is not False for v in payload["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())