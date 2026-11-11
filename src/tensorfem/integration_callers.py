"""Integration callers: platform-side invocation of TensorFEM verification cases.

This module is the **integration surface** of the case pool: the cases registered
here are called by platform-side consumers, not by the solver's own unit tests.

Consumers
---------
* ``tensorfem.cli verify`` (via :func:`tensorfem.benchmark_registry.verification_report`)
  runs the registered scalar checks, including CASE-NAFEMS-LE11.
* ``tensorfem.marine_structural_qualification`` builds its qualification matrix
  from the same registry.
* The platform / ship-performance-platform case-pool settlement calls
  :func:`run_nafems_le11_case` directly to obtain a machine-checkable verdict
  dict (no CLI parsing).

Design rule: no physics lives here. Each caller is a thin adapter that runs an
existing case module and returns ``{verdict, rows, artifacts}``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .ellipsoid_shell import ellipsoid_pressure

__all__ = ["run_nafems_le11_case", "NAFEMS_LE11_QUANTITIES"]

#: Gated quantities: (name, key holding FE value, key holding reference value)
NAFEMS_LE11_QUANTITIES = (
    ("N_phi_pole", "N_pole_fe", "N_pole_ref"),
    ("N_theta_pole", "N_theta_pole_fe", "N_theta_pole_ref"),
    ("N_phi_equator", "N_phi_equator_fe", "N_phi_equator_ref"),
    ("N_theta_equator", "N_theta_equator_fe", "N_theta_equator_ref"),
    ("w_equator", "w_equator_fe", "w_equator_ref"),
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
        "dof": out["dof"],
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
