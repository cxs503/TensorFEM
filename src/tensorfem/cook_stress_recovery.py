"""Fail-closed qualification of the Cook-membrane recovered stress field.

The merged Cook-membrane channel evaluates ``sigma = D B u`` at every element
centre.  That field is ``O(h)`` accurate and its pointwise von Mises residual
against the independent Q8 reference stays at 5.02 % on the finest declared mesh,
so ``tensorfem.additional_benchmark_fields.cook_fields`` keeps
``stress_accuracy_qualified = False``.  ``docs/cook-probe-protocol-audit.md``
showed that most of that residual is a probe-protocol artefact (all five
canonical points sit on Q4 element boundaries, where the element-wise constant
field is not single valued) and that the textbook 2x2-Gauss (Barlow)
extrapolation plus area-weighted nodal averaging reaches 0.120 % at ``n = 64``.

``tensorfem.stress_recovery`` is that construction as production code and
``cook_fields`` already exposes it as ``nodal_stress`` / ``stress_recovery``.
What is still missing -- and what this module adds -- is a *library-level,
fail-closed* verdict: the ``stress_recovery_qualified`` boolean currently only
exists as ad-hoc arithmetic inside an evidence script, with no importable API,
no observed-order criterion, no tolerance validation and no negative controls.
A qualification flag that cannot be re-run and cannot be proven to reject bad
data is not evidence.

Guarantees:

* :func:`recovered_nodal_stress` is the documented entry point for the recovered
  field.  It does not re-implement anything: it delegates to
  :func:`tensorfem.stress_recovery.q4_gauss_stresses` and
  :func:`tensorfem.stress_recovery.recover_nodal_stress` (the same functions the
  production payload uses) and only tags the result with its protocol string so
  a consumer cannot mistake it for the raw element-centre field.
* :func:`measure` drives the *production* payload ``cook_fields(n, reference=...)``
  over a refinement ladder and records the worst pointwise error per level.
* :func:`qualify` is fail-closed.  ``stress_recovery_qualified`` is true only if
  every declared level produced a finite error, the error decreases strictly at
  every level, the finest error is within tolerance, the observed order exceeds
  one, and the frozen reference digest is recorded.  Any missing condition blocks.
* ``stress_accuracy_qualified`` for the raw element-centre field is never
  touched: it stays ``False``, because the recovered field is a second-order
  post-processing of the same displacement solution.  Citing the recovered field
  while quoting the raw field would be a protocol substitution, not an accuracy
  improvement.

Scope: linear-elastic plane stress, ``E = 1``, ``nu = 1/3``, ``t = 1``, Cook
tapered membrane under a unit vertical resultant on the loaded edge; structured
Q4 grids; 2x2 Gauss recovery.  Clamped-corner and loaded-edge-end singular zones
are not covered, so no pointwise claim is made there.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .stress_recovery import (
    q4_gauss_stresses,
    recover_nodal_stress,
)

SCHEMA = "tensorfem.cook-stress-recovery/1.0"
PROTOCOL = (
    "2x2 Gauss (Barlow) stress extrapolation to element nodes, area-weighted "
    "nodal averaging, bilinear re-interpolation at the declared physical points"
)
SCOPE = (
    "linear-elastic plane stress Cook tapered membrane (E=1, nu=1/3, t=1, unit "
    "vertical resultant on the loaded edge), structured Q4 grids; 2x2 Gauss "
    "recovery; clamped-corner and loaded-edge singular zones excluded"
)
CRITERION = (
    "fail-closed: finite errors at every level, strictly monotone decrease over "
    "the ladder, finest worst-site error <= tolerance, observed order > 1, "
    "frozen reference digest recorded"
)
MAX_TOLERANCE = 0.03
DEFAULT_LEVELS = (8, 16, 32, 64)
DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "cook_membrane_stress_reference.json"
)


def _digest_safe(errors: list[float]) -> list[object]:
    """JSON-compliant view of the judged errors (rejected data may be non-finite)."""
    return [value if math.isfinite(value) else repr(value) for value in errors]


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_reference(path: str | Path | None = None) -> dict:
    """Load the frozen independent Q8 reference and record its digest.

    ``raw`` is the frozen ``reference`` block exactly as stored (the production
    payload consumes it); ``points`` is the normalised view used for reporting,
    with the Richardson-extrapolated von Mises as the per-site target.
    """
    location = Path(path) if path is not None else DEFAULT_REFERENCE
    frozen = json.loads(location.read_text())
    raw = frozen["reference"]
    return {
        "path": str(location),
        "sha256": hashlib.sha256(location.read_bytes()).hexdigest(),
        "formulation": raw.get("formulation", "unknown"),
        "levels": [int(level) for level in raw.get("levels", [])],
        "raw": raw,
        "points": [
            {
                "xi": point["xi"],
                "eta": point["eta"],
                "point": [float(component) for component in point["point"]],
                "von_mises_reference": float(point["von_mises"]["extrapolated"]),
                "gci_fine_relative": float(point["von_mises"]["gci_fine_relative"]),
            }
            for point in raw["points"]
        ],
    }


def recovered_nodal_stress(
    nodes,
    elements,
    displacement,
    *,
    young: float = 1.0,
    poisson: float = 1.0 / 3.0,
) -> dict:
    """The recovered nodal stress field, tagged with its protocol.

    ``field`` has shape ``(nnodes, 3)`` with the ``(sxx, syy, sxy)`` components.
    This is a thin, documented front end to the production recovery primitives
    (``q4_gauss_stresses`` + ``recover_nodal_stress``); it intentionally adds no
    second implementation, so the tag is the only new information a caller gets.
    """
    coordinates = torch.as_tensor(nodes, dtype=torch.float64)
    connectivity = torch.as_tensor(elements, dtype=torch.long)
    displacement = torch.as_tensor(displacement, dtype=torch.float64).reshape(-1, 2)
    gauss = q4_gauss_stresses(coordinates, connectivity, displacement, young, poisson)
    field = recover_nodal_stress(coordinates, connectivity, gauss)
    sxx, syy, sxy = field[:, 0], field[:, 1], field[:, 2]
    return {
        "protocol": PROTOCOL,
        "order": "second-order post-processing of the same displacement solution",
        "field": field,
        "von_mises": torch.sqrt(
            (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy).clamp_min(0.0)
        ),
    }


def measure(levels: Sequence[int] = DEFAULT_LEVELS, reference_path=None) -> dict:
    """Worst pointwise von Mises error of the recovered field at every level.

    The measurement goes through the production payload
    ``cook_fields(n, reference=<frozen reference>)``; nothing is recomputed in a
    private path, so the numbers being judged are the numbers a consumer receives.
    """
    from .additional_benchmark_fields import cook_fields

    reference = load_reference(reference_path)
    rows = []
    for level in levels:
        divisions = int(level)
        case = cook_fields(divisions, reference=reference["raw"])
        recovery = case["stress_recovery"]
        rows.append(
            {
                "divisions": divisions,
                "nodes": len(case["nodes"]),
                "elements": len(case["elements"]),
                "scale": float(recovery["scale"]),
                "worst_site_error": float(recovery["worst_site_error"]),
                "worst_boundary_spread": float(recovery["worst_boundary_spread"]),
                "sites": [
                    {
                        "point": site["point"],
                        "von_mises": float(site["von_mises"]),
                        "von_mises_reference": float(site["von_mises_reference"]),
                        "error": float(site["error"]),
                        "on_element_boundary": bool(site["on_element_boundary"]),
                    }
                    for site in recovery["sites"]
                ],
                "level_passed": bool(recovery["passed"]),
                "raw_field_qualified": bool(case["stress_accuracy_qualified"]),
                "displacement_response_relative_error": float(
                    case["verification"]["response_relative_error"]
                ),
            }
        )
    return {"reference": reference, "levels": rows}


def _observed_order(errors: list[float], levels: list[int]) -> float | None:
    """Observed order from the last three levels, using error differences.

    Differencing rather than the raw error ratio is the conservative choice: the
    ladder errors are dominated by a smooth ``h^p`` term, and the difference
    quotient removes the constant offset that a raw ratio would inflate.
    """
    if len(errors) < 3:
        return None
    denominator = errors[-2] - errors[-3]
    if denominator == 0.0:
        return None
    ratio = abs(errors[-1] - errors[-2]) / abs(denominator)
    if ratio <= 0.0:
        return None
    order = math.log(ratio) / math.log(levels[-2] / levels[-1])
    return order if math.isfinite(order) else None


def qualify(
    levels: Sequence[int] = DEFAULT_LEVELS,
    reference_path=None,
    *,
    tolerance: float = MAX_TOLERANCE,
    measured: Mapping[str, object] | None = None,
) -> dict:
    """Fail-closed verdict for the recovered field (never promotes the raw field).

    ``measured`` allows an already measured payload to be re-judged, which the
    test suite uses to prove the gate refuses non-monotone, over-tolerance and
    non-finite data, and that it cannot be widened past the 3 % acceptance limit.
    """
    if not (math.isfinite(tolerance) and 0.0 < tolerance <= MAX_TOLERANCE):
        raise ValueError(
            f"tolerance must be finite and at most {MAX_TOLERANCE:.0%}; "
            "the acceptance limit is not configurable"
        )
    levels = [int(level) for level in levels]
    if len(levels) < 2:
        raise ValueError("at least two mesh levels are required for a monotone gate")
    if any(later <= earlier for earlier, later in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly increasing")

    data = dict(measured) if measured is not None else measure(levels, reference_path)
    reference = data.get("reference") or load_reference(reference_path)
    rows = [dict(row) for row in data["levels"]]
    if [int(row["divisions"]) for row in rows] != levels:
        raise ValueError("measured levels do not match the declared levels")

    errors = [float(row["worst_site_error"]) for row in rows]
    finite = all(math.isfinite(value) for value in errors)
    monotone = finite and all(later < earlier for earlier, later in zip(errors, errors[1:]))
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
    if not reference.get("sha256"):
        reasons.append("reference_digest_missing")

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "case": "cook-membrane",
        "protocol": PROTOCOL,
        "criterion": CRITERION,
        "scope": SCOPE,
        "tolerance": tolerance,
        "levels": levels,
        "reference": {
            "path": str(reference["path"]),
            "sha256": reference["sha256"],
            "formulation": reference["formulation"],
            "levels": list(reference["levels"]),
            "points": list(reference["points"]),
        },
        "ladder": [
            {
                "divisions": int(row["divisions"]),
                "worst_site_error": float(row["worst_site_error"]),
                "sites": list(row.get("sites", [])),
            }
            for row in rows
        ],
        "error_sequence": errors,
        "monotone_decreasing": monotone,
        "observed_order": order,
        "stress_recovery_qualified": qualified,
        "blocking_reasons": reasons,
        "stress_accuracy_qualified": False,
        "element_centre_field": {
            "qualified": False,
            "reason": (
                "the raw element-centre field is O(h) accurate and its five canonical "
                "probe sites all lie on Q4 element boundaries where the value is not "
                "single valued; the recovered field is a second-order post-processing "
                "of the same displacement solution and does not certify it"
            ),
        },
        "verdict_sha256": _digest(
            {
                "schema": SCHEMA,
                "protocol": PROTOCOL,
                "tolerance": tolerance,
                "levels": levels,
                "errors": _digest_safe(errors),
                "reference_sha256": reference.get("sha256"),
                "qualified": qualified,
            }
        ),
    }
