"""CASE-NAFEMS-LE11 -- spheroidal (ellipsoid-of-revolution) shell under pressure.

Runs the TensorFEM axisymmetric shell-of-revolution route
(:mod:`tensorfem.axisym_shell`) on a closed spheroid with the ellipsoid case
module :mod:`tensorfem.ellipsoid_shell`, and writes:

* ``runs/nafems_le11/nafems_le11_results.json`` -- every number in the report,
* ``docs/assets/nafems_le11/*.png``             -- report figures.

Reference values are *computed* by the programme (closed-form membrane forces
from statics + an independent Rayleigh-Ritz membrane displacement solve); no
number in the report is transcribed by hand.

Usage::

    PYTHONPATH=src python3 benchmarks/nafems_le11_ellipsoid.py              # full sweep + figures
    PYTHONPATH=src python3 benchmarks/nafems_le11_ellipsoid.py --facets 64  # single mesh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from tensorfem.ellipsoid_shell import ellipsoid_pressure, membrane_forces

A, B, THICKNESS, YOUNG, POISSON, PRESSURE = 1.0, 0.5, 0.01, 1.0, 0.3, 1.0
TOL = 0.05   # case gate: < 5 %
SWEEP = (4, 8, 16, 32, 64, 128, 256, 512)


def _summary(out: dict) -> dict:
    return {
        "facets": out["facets"],
        "dof": out["dof"],
        "N_pole_fe": out["N_pole_fe"],
        "N_pole_ref": out["N_pole_ref"],
        "N_pole_err": out["N_pole_err"],
        "N_theta_pole_fe": out["N_theta_pole_fe"],
        "N_theta_pole_ref": out["N_theta_pole_ref"],
        "N_theta_pole_err": out["N_theta_pole_err"],
        "N_phi_equator_fe": out["N_phi_equator_fe"],
        "N_phi_equator_ref": out["N_phi_equator_ref"],
        "N_phi_equator_err": out["N_phi_equator_err"],
        "N_theta_equator_fe": out["N_theta_equator_fe"],
        "N_theta_equator_ref": out["N_theta_equator_ref"],
        "N_theta_equator_err": out["N_theta_equator_err"],
        "w_pole_fe": out["w_pole_fe"],
        "w_pole_ref": out["w_pole_ref"],
        "w_equator_fe": out["w_equator_fe"],
        "w_equator_ref": out["w_equator_ref"],
        "w_equator_err": out["w_equator_err"],
        "err_phi_max_scaled": out["err_phi_max_scaled"],
        "err_theta_max_scaled": out["err_theta_max_scaled"],
        "axial_residual_max": out["axial_residual_max"],
    }


def run_case(*, facets: int, a: float = A, b: float = B, thickness: float = THICKNESS,
             young: float = YOUNG, poisson: float = POISSON, pressure: float = PRESSURE,
             shear_integration: str = "reduced", grading: float = 1.0) -> dict:
    t0 = time.time()
    out = ellipsoid_pressure(a=a, b=b, thickness=thickness, young=young, poisson=poisson,
                             pressure=pressure, facets=facets, grading=grading,
                             shear_integration=shear_integration)
    out["wall_seconds"] = time.time() - t0
    return out


def main() -> int:                                              # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--facets", type=int, default=None)
    ap.add_argument("--out", default="runs/nafems_le11/nafems_le11_results.json")
    ap.add_argument("--figdir", default="docs/assets/nafems_le11")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    sweep_facets = (args.facets,) if args.facets else SWEEP
    sweep = [_summary(run_case(facets=f)) for f in sweep_facets]

    payload: dict = {
        "case": "CASE-NAFEMS-LE11",
        "solver": "TensorFEM tensorfem.axisym_shell (2-node ring, Reissner-Mindlin, SRI)",
        "parameters": {
            "a_equatorial": A, "b_polar": B, "thickness": THICKNESS, "young": YOUNG,
            "poisson": POISSON, "pressure": PRESSURE,
            "aspect_ratio_b_over_a": B / A, "slenderness_a_over_t": A / THICKNESS,
            "dimensionless_group": "p*a^2/(E*t^2) = %.1f" % (PRESSURE * A ** 2
                                                             / (YOUNG * THICKNESS ** 2)),
        },
        "reference": {
            "N_phi": "p*a*W/(2*b), W = sqrt(a^2 cos^2 th + b^2 sin^2 th)  [cap statics]",
            "N_theta": "p*(a*W/b)*(1 - a^2/(2 W^2))  [normal equilibrium]",
            "displacement": "Rayleigh-Ritz membrane potential minimisation",
        },
        "tolerance": TOL,
        "sweep": sweep,
    }

    if not args.facets:
        # --- reference self-check: Rayleigh-Ritz reproduces closed-form forces ----
        ref = run_case(facets=32)["membrane_shape"]
        payload["reference_selfcheck"] = {
            "rr_order": ref["order"],
            "N_phi_max_rel_error": ref["N_selfcheck_phi"],
            "N_theta_max_rel_error": ref["N_selfcheck_theta"],
            "sphere_limit_w_equator": _sphere_limit_w(),
        }
        # --- sensitivity studies ------------------------------------------------
        payload["sensitivity"] = {
            "aspect_ratio": [_aspect_point(ratio) for ratio in (0.3, 0.5, 0.75, 1.0, 2.0, 3.0)],
            "thickness": [_thickness_point(t) for t in (0.02, 0.01, 0.005, 0.002)],
            "grading": [
                dict(grading=g, **_summary(run_case(facets=f, grading=g)))
                for g in (0.5, 0.75, 1.0, 1.5) for f in (32, 64)
            ],
            "shear_integration": [
                dict(integr=si, **_summary(run_case(facets=128, shear_integration=si)))
                for si in ("reduced", "full")
            ],
        }
        fine = run_case(facets=256)
        payload["fine_mesh"] = _summary(fine)
        payload["fine_mesh_gauss"] = [
            {k: d[k] for k in ("theta", "r", "phi", "N_s_fe", "N_phi_ref", "N_theta_fe",
                               "N_theta_ref", "M_s_fe", "axial_residual")}
            for d in fine["gc"]
        ]
        payload["profile_128"] = [
            {k: d[k] for k in ("theta", "r", "N_s_fe", "N_phi_ref", "N_theta_fe",
                               "N_theta_ref")}
            for d in run_case(facets=128)["gc"]
        ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)

    if not args.no_figures and not args.facets:
        from tensorfem.ellipsoid_shell_figures import make_figures
        make_figures(payload, args.figdir)

    _print_table(payload)
    return 0


def _sphere_limit_w() -> dict:
    """Sphere special case: w must equal p a^2 (1 - nu) / (2 E t)."""
    out = ellipsoid_pressure(a=1.0, b=1.0, thickness=THICKNESS, young=YOUNG,
                             poisson=POISSON, pressure=PRESSURE, facets=16)
    exact = PRESSURE * 1.0 ** 2 * (1.0 - POISSON) / (2.0 * YOUNG * THICKNESS)
    return {"fe": out["w_equator_fe"], "exact": exact,
            "rel_error": abs(out["w_equator_fe"] - exact) / exact,
            "N_theta_err": out["N_theta_pole_err"]}


def _aspect_point(ratio: float) -> dict:
    b = ratio
    out = run_case(facets=128, a=1.0, b=b)
    n_phi_eq, n_theta_eq = membrane_forces(1.0, b, PRESSURE, 0.5 * math.pi)
    return {
        "b_over_a": ratio, "facets": 128,
        "N_pole_ref": out["N_pole_ref"], "N_theta_pole_ref": out["N_theta_pole_ref"],
        "N_phi_equator_ref": n_phi_eq, "N_theta_equator_ref": n_theta_eq,
        "hoop_sign_change": bool(out["N_theta_pole_ref"] * n_theta_eq < 0.0),
        "N_pole_err": out["N_pole_err"],
        "N_theta_equator_err": out["N_theta_equator_err"],
        "err_phi_max_scaled": out["err_phi_max_scaled"],
        "err_theta_max_scaled": out["err_theta_max_scaled"],
        "w_equator_err": out["w_equator_err"],
    }


def _thickness_point(t: float) -> dict:
    out = run_case(facets=128, thickness=t)
    return {"thickness": t, "slenderness": 1.0 / t, "facets": 128,
            "N_pole_err": out["N_pole_err"],
            "N_theta_equator_err": out["N_theta_equator_err"],
            "err_phi_max_scaled": out["err_phi_max_scaled"],
            "err_theta_max_scaled": out["err_theta_max_scaled"],
            "w_equator_fe": out["w_equator_fe"], "w_equator_ref": out["w_equator_ref"],
            "w_equator_err": out["w_equator_err"],
            "bending_to_membrane": out["bending_to_membrane"]}


def _print_table(payload: dict) -> None:                        # pragma: no cover
    print("CASE-NAFEMS-LE11  spheroidal shell, p=%g a=%g b=%g t=%g"
          % (PRESSURE, A, B, THICKNESS))
    print("%5s %5s | %9s %9s %7s | %9s %9s %7s | %8s %8s %7s | %7s %7s"
          % ("fct", "dof", "N_pole", "ref", "err", "NthEq", "ref", "err",
             "wEq", "ref", "err", "maxNphi", "maxNth"))
    for s in payload["sweep"]:
        print("%5d %5d | %9.5f %9.5f %6.2f%% | %9.5f %9.5f %6.2f%% | %8.3f %8.3f %6.2f%% | %6.2f%% %6.2f%%"
              % (s["facets"], s["dof"], s["N_pole_fe"], s["N_pole_ref"], 100 * s["N_pole_err"],
                 s["N_theta_equator_fe"], s["N_theta_equator_ref"],
                 100 * s["N_theta_equator_err"], s["w_equator_fe"], s["w_equator_ref"],
                 100 * s["w_equator_err"], 100 * s["err_phi_max_scaled"],
                 100 * s["err_theta_max_scaled"]))


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())