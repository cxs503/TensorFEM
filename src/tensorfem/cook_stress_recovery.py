"""Production promotion of the audited Cook-membrane stress recovery.

The merged Cook-membrane channel evaluates ``sigma = C @ B @ u`` at every element
centre.  That field is ``O(h)`` accurate, and the pointwise von Mises residual
against the independent Q8 reference therefore stays at 5.02 % on the finest
declared mesh, which is why
``tensorfem.additional_benchmark_fields.cook_fields`` carries
``stress_accuracy_qualified = False``.  ``docs/cook-probe-protocol-audit.md``
showed that part of that residual is a probe-protocol artefact and that a
textbook 2x2-Gauss (Barlow) extrapolation to the nodes plus area-weighted nodal
averaging reaches 0.120 % at ``n = 64`` -- but that construction lived only in
the measurement module ``tensorfem.cook_probe_protocol`` and never reached the
production field builder, so the qualification status could not refer to it.

This module closes that gap without weakening any fail-closed gate:

* :func:`recovered_nodal_stress` is the production entry point for the recovered
  field (same construction the audit measured, now returned with its protocol
  string so a consumer cannot mistake it for the raw element field);
* :func:`qualify` measures the recovered field against the frozen independent
  Q8 reference at increasing mesh levels and returns a **fail-closed** verdict:
  ``stress_recovery_qualified`` is true only when the finest pointwise error is
  below tolerance, the error decreases monotonically over every declared level,
  the observed order exceeds one, and the reference digest is recorded;
* ``stress_accuracy_qualified`` for the element-centre field is *not* touched
  and stays false -- the recovered field is a second-order post-processing of
  the same displacement solution, so citing it while quoting the raw field
  would be a protocol substitution, not an accuracy improvement.

Scope: linear-elastic plane stress, ``E = 1``, ``nu = 1/3``, ``t = 1``, Cook
tapered membrane under a unit vertical resultant on the loaded edge; structured
Q4 grids; 2x2 Gauss recovery.  Clamped-corner and loaded-edge-end singular
zones are *not* covered, so no pointwise claim is made there.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .cook_probe_protocol import (
    canonical_points,
    interpolate_nodal_stress,
    nodal_recovered_stress,
    normalized_error,
    von_mises,
)

SCHEMA = "tensorfem.cook-stress-recovery/1.0"
PROTOCOL = (
    "2x2 Gauss (Barlow) stress extrapolation to element nodes, area-weighted "
    "nodal averaging, bilinear re-interpolation at the declared physical points"
)
SCOPE = (
    "linear-elastic plane stress Cook tapered membrane (E=1, nu=1/3, t=1, unit "
    "vertical resultant on the loaded edge), structured Q4 grids n=8..64; "
    "clamped-corner and loaded-edge singular zones excluded"
)
DEFAULT_LEVELS = (8, 16, 32, 64)
DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "cook_membrane_stress_reference.json"
)


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_reference(path: str | Path | None = None) -> dict:
    """Load the frozen independent Q8 stress reference and record its digest."""
    location = Path(path) if path is not None else DEFAULT_REFERENCE
    frozen = json.loads(location.read_text())
    return {
        "path": str(location),
        "sha256": hashlib.sha256(location.read_bytes()).hexdigest(),
        "formulation": frozen.get("reference", {}).get("formulation", "unknown"),
        "levels": list(frozen.get("reference", {}).get("levels", [])),
        "points": canonical_points(frozen["reference"]),
    }


def recovered_nodal_stress(nodes, elements, displacement) -> dict:
    """The recovered nodal stress field, tagged with its protocol.

    ``field`` has shape ``(nnodes, 3)`` with the ``(sxx, syy, sxy)`` components.
    The tag exists so downstream export cannot silently present the recovered
    field as the raw element-centre field stored under ``cook_fields['stress']``.
    """
    field = nodal_recovered_stress(nodes, elements, displacement)
    return {
        "protocol": PROTOCOL,
        "order": "second-order post-processing of the same displacement solution",
        "field": field,
        "von_mises": np.sqrt(
            field[:, 0] ** 2 - field[:, 0] * field[:, 1] + field[:, 1] ** 2 + 3.0 * field[:, 2] ** 2
        ),
    }


def measure(levels: Sequence[int] = DEFAULT_LEVELS, reference_path=None) -> dict:
    """Pointwise von Mises error of the recovered field at every declared level."""
    from .additional_benchmark_fields import cook_fields

    reference = load_reference(reference_path)
    points = reference["points"]
    scale = max(point["von_mises_reference"] for point in points)
    rows = []
    for level in levels:
        case = cook_fields(int(level), include_recovery=True)
        recovered = case["recovered_nodal_stress"]
        row = {"divisions": int(level), "points": [], "max_relative_error": 0.0}
        for point in points:
            sample = interpolate_nodal_stress(
                case["nodes"], case["elements"], np.asarray(recovered, dtype=np.float64),
                point["point"],
            )
            error = normalized_error(
                sample["von_mises"], point["von_mises_reference"], scale
            )
            row["points"].append(
                {
                    "xi": point["xi"],
                    "eta": point["eta"],
                    "point": point["point"],
                    "von_mises": sample["von_mises"],
                    "von_mises_reference": point["von_mises_reference"],
                    "relative_error": error,
                }
            )
            row["max_relative_error"] = max(row["max_relative_error"], error)
        rows.append(row)
    return {"reference": reference, "scale": scale, "levels": rows}


def _observed_order(values: list[float], levels: list[int]) -> float | None:
    if len(values) < 3:
        return None
    ratio_denominator = values[-2] - values[-3]
    if abs(ratio_denominator) <= 0.0:
        return None
    ratio = abs(values[-1] - values[-2]) / abs(ratio_denominator)
    if ratio <= 0.0:
        return None
    order = math.log(ratio) / math.log(levels[-2] / levels[-1])
    return order if math.isfinite(order) else None


def qualify(
    levels: Sequence[int] = DEFAULT_LEVELS,
    reference_path=None,
    *,
    tolerance: float = 0.03,
    measured: Mapping[str, object] | None = None,
) -> dict:
    """Fail-closed verdict for the recovered field (never promotes the raw field).

    ``measured`` allows an already measured payload to be re-judged, which the
    tests use to prove the gate refuses non-monotone or out-of-tolerance data.
    """
    if not (math.isfinite(tolerance) and 0.0 < tolerance <= 0.03):
        raise ValueError("tolerance must be finite and at most 3%")
    levels = [int(level) for level in levels]
    if len(levels) < 2:
        raise ValueError("at least two mesh levels are required for a monotone gate")

    data = dict(measured) if measured is not None else measure(levels, reference_path)
    rows = list(data["levels"])
    if [row["divisions"] for row in rows] != levels:
        raise ValueError("measured levels do not match the declared levels")

    errors = [float(row["max_relative_error"]) for row in rows]
    finite = all(math.isfinite(value) for value in errors)
    monotone = finite and all(b < a for a, b in zip(errors, errors[1:]))
    order = _observed_order(errors, levels) if finite else None
    reasons = []
    if not finite:
        reasons.append("non_finite_error")
    if not monotone:
        reasons.append("non_monotone_error")
    if errors[-1] > tolerance:
        reasons.append("finest_above_tolerance")
    if order is None or order <= 1.0:
        reasons.append("observed_order_not_above_one")

    reference = data["reference"]
    return {
        "schema": SCHEMA,
        "case": "cook-membrane",
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "tolerance": tolerance,
        "reference": {
            "path": str(reference["path"]),
            "sha256": reference["sha256"],
            "formulation": reference["formulation"],
            "levels": reference["levels"],
        },
        "levels": [
            {
                "divisions": row["divisions"],
                "max_relative_error": float(row["max_relative_error"]),
                "points": list(row["points"]),
            }
            for row in rows
        ],
        "error_sequence": errors,
        "monotone_decreasing": monotone,
        "observed_order": order,
        "stress_recovery_qualified": not reasons,
        "blocking_reasons": reasons,
        "stress_accuracy_qualified": False,
        "element_centre_field": {
            "qualified": False,
            "reason": (
                "the raw element-centre field is O(h) accurate and reports 5.02% "
                "pointwise at n=64 under the merged protocol; the recovered field "
                "is a second-order post-processing and does not certify it"
            ),
        },
        "verdict_sha256": _digest(
            {
                "protocol": PROTOCOL,
                "tolerance": tolerance,
                "errors": errors,
                "reference_sha256": reference["sha256"],
                "qualified": not reasons,
            }
        ),
    }