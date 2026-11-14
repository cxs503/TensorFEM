"""Integration callers: platform-side invocation of TensorFEM verification cases.

This module is the **integration surface** of the case pool: the cases registered
here are called by platform-side consumers, not by the solver's own unit tests.

Consumers
---------
* ``tensorfem.cli verify`` (via :func:`tensorfem.benchmark_registry.verification_report`)
  runs the registered scalar checks, including CASE-NAFEMS-LE11 and
  CASE-FEM-KIRSCH-PLATE.
* ``tensorfem.marine_structural_qualification`` builds its qualification matrix
  from the same registry.
* The platform / ship-performance-platform case-pool settlement calls
  :func:`run_nafems_le11_case` / :func:`run_kirsch_plate_case` directly to obtain
  a machine-checkable verdict dict (no CLI parsing).

Design rule: no physics lives here. Each caller is a thin adapter that runs an
existing case module and returns ``{verdict, rows, artifacts}``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .ellipsoid_shell import ellipsoid_pressure
from .finite_width_hole import FiniteWidthPlate, solve_case

__all__ = [
    "run_nafems_le11_case",
    "run_kirsch_plate_case",
    "NAFEMS_LE11_QUANTITIES",
    "KIRSCH_PLATE_QUANTITIES",
]

#: Gated quantities: (name, key holding FE value, key holding reference value)
NAFEMS_LE11_QUANTITIES = (
    ("N_phi_pole", "N_pole_fe", "N_pole_ref"),
    ("N_theta_pole", "N_theta_pole_fe", "N_theta_pole_ref"),
    ("N_phi_equator", "N_phi_equator_fe", "N_phi_equator_ref"),
    ("N_theta_equator", "N_theta_equator_fe", "N_theta_equator_ref"),
    ("w_equator", "w_equator_fe", "w_equator_ref"),
)

#: Gated quantities for the Kirsch plate (CASE-FEM-KIRSCH-PLATE).
#: ``K_tg`` is the *gross*-section stress-concentration factor ``sigma_peak / T``;
#: with ``d/W -> 0`` it must collapse onto the infinite-plate Kirsch value ``3``
#: (the pool mandate is < 5 %).  ``K_tn`` is the *net*-section (ligament) factor
#: ``sigma_peak / sigma_net``, compared with Heywood's engineering fit
#: ``K_tn = 2 + (1 - d/W)**3`` (the registry default is < 3 %).
KIRSCH_PLATE_QUANTITIES = (
    ("K_tg", "ktg_fe", "ktg_ref"),
    ("K_tn", "ktn_fe", "ktn_ref"),
)


def run_nafems_le11_case(
    *,
    facets: int = 256,
    a: float = 1.0,
    b: float = 0.5,
    thickness: float = 0.01,
    young: float = 1.0,
    poisson: float = 0.3,
    pressure: float = 1.0,
    tolerance: float = 0.05,
    output_dir: str | None = None,
) -> dict:
    """Run CASE-NAFEMS-LE11 and return a machine-checkable verdict.

    Returns
    -------
    dict with keys
        ``case``, ``verdict`` ('PASS' | 'FAIL'), ``rows`` (per-quantity ref/got/
        rel_error/passed), ``parameters``, ``max_rel_error``, ``artifact``
        (path of the JSON written when ``output_dir`` is given), ``wall_seconds``.
    """
    t0 = time.time()
    out = ellipsoid_pressure(a=a, b=b, thickness=thickness, young=young,
                             poisson=poisson, pressure=pressure, facets=facets)
    rows = []
    for name, fe_key, ref_key in NAFEMS_LE11_QUANTITIES:
        got, ref = float(out[fe_key]), float(out[ref_key])
        err = abs(got - ref) / abs(ref)
        rows.append({"quantity": name, "got": got, "ref": ref,
                     "rel_error": err, "tolerance": tolerance, "passed": err < tolerance})
    payload = {
        "case": "CASE-NAFEMS-LE11",
        "verdict": "PASS" if all(r["passed"] for r in rows) else "FAIL",
        "rows": rows,
        "max_rel_error": max(r["rel_error"] for r in rows),
        "tolerance": tolerance,
        "facets": facets,
        "dof": out["dofs"],
        "parameters": {"a": a, "b": b, "thickness": thickness, "young": young,
                       "poisson": poisson, "pressure": pressure},
        "axial_residual_max": out["axial_residual_max"],
        "artifact": None,
        "wall_seconds": None,
    }
    if output_dir is not None:
        path = Path(output_dir) / "nafems_le11_case_verdict.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n")
        payload["artifact"] = str(path)
    payload["wall_seconds"] = round(time.time() - t0, 2)
    return payload


def run_kirsch_plate_case(
    *,
    diameter: float = 0.02,
    width: float = 1.0,
    half_length: float = 1.5,
    n_boundary: int = 24,
    n_radial: int | None = None,
    grading: float = 6.0,
    tolerance: float = 0.05,
    output_dir: str | None = None,
) -> dict:
    """Run CASE-FEM-KIRSCH-PLATE and return a machine-checkable verdict.

    A thin adapter over :func:`tensorfem.finite_width_hole.solve_case` (the
    *production* plane-stress path, :class:`tensorfem.continuum.ContinuumModel`
    + :func:`solve_continuum`) on a graded quarter-plate Q4 mesh with a central
    circular hole.  With ``d/W -> 0`` the gross-section stress-concentration
    factor ``K_tg`` must collapse onto the infinite-plate Kirsch (1898) value
    ``K_t = 3``; the net-section factor ``K_tn`` is compared with Heywood's
    engineering fit ``2 + (1 - d/W)**3``.  The load balance on the symmetry
    plane is a statically-exact invariant (residual ~ 1e-12).

    Returns
    -------
    dict with keys
        ``case``, ``verdict`` ('PASS' | 'FAIL'), ``rows`` (per-quantity ref/got/
        rel_error/passed), ``parameters``, ``max_rel_error``, ``artifact``
        (path of the JSON written when ``output_dir`` is given), ``wall_seconds``,
        ``load_balance``, ``dof``.
    """
    t0 = time.time()
    plate = FiniteWidthPlate(width=width, diameter=diameter,
                             half_length=half_length)
    out = solve_case(plate, n_boundary=n_boundary, n_radial=n_radial,
                     grading=grading)
    # Reference values: the Kirsch infinite-plate limit for the gross factor,
    # and Heywood's finite-width fit for the net factor.
    ktg_ref = 3.0
    ktn_ref = plate.heywood_ktn
    rows = []
    for name, fe_key, _ in KIRSCH_PLATE_QUANTITIES:
        got = float(out[fe_key])
        ref = ktg_ref if name == "K_tg" else ktn_ref
        err = abs(got - ref) / abs(ref)
        rows.append({"quantity": name, "got": got, "ref": ref,
                     "rel_error": err, "tolerance": tolerance, "passed": err < tolerance})
    # Load balance is a statically-exact invariant (residual vs 0).
    balance = out["load_balance"]
    balance_err = float(balance["relative_error"])
    rows.append({"quantity": "load_balance", "got": balance_err, "ref": 0.0,
                 "rel_error": balance_err, "tolerance": 1e-10,
                 "passed": balance_err < 1e-10})
    payload = {
        "case": "CASE-FEM-KIRSCH-PLATE",
        "verdict": "PASS" if all(r["passed"] for r in rows) else "FAIL",
        "rows": rows,
        "max_rel_error": max(r["rel_error"] for r in rows),
        "tolerance": tolerance,
        "dof": out["dofs"],
        "elements": out["elements"],
        "parameters": {"diameter": diameter, "width": width,
                       "half_length": half_length, "n_boundary": n_boundary,
                       "n_radial": n_radial, "grading": grading,
                       "young": plate.young, "poisson": plate.poisson,
                       "tension": plate.tension},
        "load_balance": balance,
        "stress_concentration": out["stress_concentration"],
        "artifact": None,
        "wall_seconds": None,
    }
    if output_dir is not None:
        path = Path(output_dir) / "kirsch_plate_case_verdict.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n")
        payload["artifact"] = str(path)
    payload["wall_seconds"] = round(time.time() - t0, 2)
    return payload
