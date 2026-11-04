"""Production qualification of the Cook membrane *recovered* stress field.

Background.  The merged Cook membrane benchmark stores a constant stress per Q4
element (``tensorfem.additional_benchmark_fields.cook_fields``) and carries
``stress_accuracy_qualified = False``: sampled at five published material points the
stored field sits 5.02 % from the independent Q8-Richardson reference at n=64, and
those points fall on element interfaces where a discontinuous field has no unique
value.  The protocol audit (:mod:`tensorfem.cook_probe_protocol`,
``docs/cook-probe-protocol-audit.md``) showed the residual is a *measurement*
defect, not a discretisation defect, and left one item open: the Gauss-point nodal
recovery it used for the argument never entered the production path.

This module closes that item.  It builds the recovered nodal field with the
production implementation (:mod:`tensorfem.stress_recovery`), samples it at the five
published points, and derives a fail-closed qualification record.  The gate is
deliberately narrow:

* the reference must be the frozen, independently coded Q8 Richardson field shipped
  with the repository (``benchmarks/cook_membrane_stress_reference.json``); if the
  file is absent the case is *not* qualified rather than silently grading itself;
* every published point must be below the 3 % error gate on the recovered von Mises
  value *and* on each recovered component, using the reference-scaled 5 % error floor
  of the Cook audit (component gate is the stricter of the two and is what makes the
  n=16 level fail);
* the recovered field must be single valued at the probe (alternate one-sided
  interpolations must agree to rounding), otherwise the "well-posed probe" claim
  would be vacuous;
* the element-centre errors of the merged protocol are reported alongside, never
  used to qualify.

The record never touches ``cook_fields``' own ``stress_accuracy_qualified`` flag:
the stored element-centre field is still first-order accurate and that flag stays
``False``.  What is qualified here is the recovered field built from the *same*
displacement solution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .additional_benchmark_fields import cook_fields
from .cook_stress_reference import SAMPLE_POINTS
from .cook_probe_protocol import one_sided_values
from .stress_recovery import (
    interpolate_nodal_stress,
    nodal_recovered_stress,
    recovery_error,
    von_mises,
)

__all__ = [
    "GATE",
    "DEFAULT_REFERENCE_PATH",
    "load_frozen_reference",
    "cook_recovery_case",
    "cook_recovery_study",
]

#: Acceptance gate for the pointwise von Mises error.
GATE = 0.03

#: Repository copy of the frozen Q8 reference, resolved relative to the source tree.
DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "cook_membrane_stress_reference.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_reference(path: str | Path | None = None) -> dict:
    """Load the frozen Q8 reference points used as the independent stress oracle.

    Raises :class:`FileNotFoundError` when the reference artefact cannot be found;
    callers convert that into a fail-closed "not qualified" record instead of
    fabricating a reference.
    """
    target = Path(path) if path is not None else DEFAULT_REFERENCE_PATH
    if not target.is_file():
        raise FileNotFoundError(f"frozen Cook reference not found: {target}")
    payload = json.loads(target.read_text())
    reference = payload.get("reference", payload)
    points = []
    for entry in reference["points"]:
        points.append(
            {
                "xi": float(entry["xi"]),
                "eta": float(entry["eta"]),
                "point": [float(v) for v in entry["point"]],
                "von_mises": float(entry["von_mises"]["extrapolated"]),
                "components": {
                    key: float(entry[key]["extrapolated"]) for key in ("sxx", "syy", "sxy")
                },
                "gci_fine_relative": float(entry["von_mises"].get("gci_fine_relative", float("nan"))),
            }
        )
    if [tuple((p["xi"], p["eta"])) for p in points] != list(SAMPLE_POINTS):
        raise ValueError("frozen reference point set does not match cook_stress_reference.SAMPLE_POINTS")
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "formulation": reference.get("formulation", ""),
        "levels": list(reference.get("levels", [])),
        "points": points,
        "scale": max(point["von_mises"] for point in points),
    }


def cook_recovery_case(n: int, reference: dict | None = None) -> dict:
    """Recovered Cook membrane stress field at mesh ``n`` with its qualification record.

    The displacement comes from the production ``cook_fields(n)`` solve; only the
    stress post-processing differs from the stored benchmark field.
    """
    case = cook_fields(n)
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    displacement = np.asarray(case["displacement"], dtype=np.float64)
    nodal = nodal_recovered_stress(nodes, elements, displacement)

    record = {
        "mesh": int(n),
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "displacement": displacement.tolist(),
        "recovered_stress": nodal.tolist(),
        "recovered_von_mises": [von_mises(row) for row in nodal],
        "response": case["verification"]["response"],
        "response_relative_error": case["verification"]["response_relative_error"],
        "fail_reasons": [],
    }

    try:
        loaded = reference if reference is not None else load_frozen_reference()
    except FileNotFoundError as error:
        record.update(
            {
                "points": [],
                "max_error": float("nan"),
                "recovery_qualified": False,
                "reference": None,
            }
        )
        record["fail_reasons"].append(f"reference_unavailable: {error}")
        return record

    scale = loaded["scale"]
    rows = []
    for point in loaded["points"]:
        probe = interpolate_nodal_stress(nodes, elements, nodal, point["point"])
        components = probe["stress"]
        error = recovery_error(probe["von_mises"], point["von_mises"], scale)
        component_errors = {
            key: recovery_error(components[index], point["components"][key], scale)
            for index, key in enumerate(("sxx", "syy", "sxy"))
        }
        baseline = one_sided_values(nodes, elements, displacement, point["point"])
        baseline_errors = [
            recovery_error(candidate["von_mises"], point["von_mises"], scale)
            for candidate in baseline["candidates"]
        ]
        rows.append(
            {
                "xi": point["xi"],
                "eta": point["eta"],
                "point": point["point"],
                "recovered_von_mises": probe["von_mises"],
                "recovered_stress": components,
                "reference_von_mises": point["von_mises"],
                "reference_gci_fine_relative": point["gci_fine_relative"],
                "error": error,
                "component_errors": component_errors,
                "well_posed_spread": probe["alternate_element_spread"],
                "candidate_count": probe["candidate_count"],
                "element_centre_error_min": min(baseline_errors),
                "element_centre_error_max": max(baseline_errors),
                "one_sided_spread_relative": baseline["spread_relative"],
            }
        )

    max_error = max(row["error"] for row in rows)
    max_spread = max(row["well_posed_spread"] for row in rows)
    record.update(
        {
            "points": rows,
            "max_error": max_error,
            "max_component_error": max(
                max(row["component_errors"].values()) for row in rows
            ),
            "max_element_centre_error": max(
                row["element_centre_error_max"] for row in rows
            ),
            "max_well_posed_spread": max_spread,
            "reference": {
                "path": loaded["path"],
                "sha256": loaded["sha256"],
                "formulation": loaded["formulation"],
                "levels": loaded["levels"],
            },
            "gate": GATE,
        }
    )

    if not np.all(np.isfinite(nodal)):
        record["fail_reasons"].append("non_finite_recovered_field")
    if max_error >= GATE:
        record["fail_reasons"].append(f"pointwise_error {max_error:.6f} >= gate {GATE}")
    max_component_error = record["max_component_error"]
    if max_component_error >= GATE:
        record["fail_reasons"].append(
            f"component_error {max_component_error:.6f} >= gate {GATE}"
        )
    if max_spread > 1e-9:
        record["fail_reasons"].append(f"probe_not_single_valued ({max_spread:.3e})")
    record["recovery_qualified"] = not record["fail_reasons"]
    return record


def cook_recovery_study(levels=(8, 16, 32, 64), reference: dict | None = None) -> dict:
    """Qualification across a refinement ladder, with its convergence evidence.

    Study-level ``qualified`` requires the *finest* level's record to be qualified and
    the sequence of per-level maximum errors to decrease monotonically; coarse levels
    that miss the gate are kept in the ladder as recorded failures instead of being
    averaged away.  A non-monotone ladder is reported as unqualified even when the
    finest level happens to pass, because that is the signature of a mesh-sampling
    artefact rather than converging accuracy.
    """
    loaded = reference if reference is not None else load_frozen_reference()
    cases = [cook_recovery_case(int(level), loaded) for level in levels]
    maximum = [case["max_error"] for case in cases]
    monotone = all(
        float(fine) < float(coarse) for coarse, fine in zip(maximum, maximum[1:])
    )
    rows = [
        {
            "mesh": case["mesh"],
            "max_error": case["max_error"],
            "max_component_error": case.get("max_component_error"),
            "max_element_centre_error": case.get("max_element_centre_error"),
            "recovery_qualified": case["recovery_qualified"],
            "fail_reasons": list(case["fail_reasons"]),
        }
        for case in cases
    ]
    reasons = []
    finest = rows[-1]
    if not finest["recovery_qualified"]:
        reasons.append(f"mesh {finest['mesh']}: {', '.join(finest['fail_reasons'])}")
    if not monotone:
        reasons.append(f"per-level maximum error not monotone: {maximum}")
    return {
        "levels": [int(level) for level in levels],
        "cases": cases,
        "rows": rows,
        "finest_max_error": maximum[-1],
        "monotone": bool(monotone),
        "gate": GATE,
        "recovery_qualified": not reasons,
        "fail_reasons": reasons,
        "reference": cases[-1].get("reference"),
    }