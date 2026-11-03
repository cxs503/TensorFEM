"""Audit the probe protocol behind the Cook membrane Q4 pointwise stress residual.

The merged qualification channel (``scripts/audit_cook_membrane_stress.py``) reports an
unexplained pointwise von Mises error of 5.02 % at n=64 and therefore keeps
``stress_accuracy_qualified`` false.  This script separates discretisation error from
measurement-protocol ambiguity by measuring, at every canonical reference point:

* the admissible one-sided Q4 values on both sides of the element edge (spread),
* the mesh-dependent tie-break value the merged protocol happens to select,
* an edge-averaged value (ambiguity removed by symmetrisation),
* a Gauss-point/nodal-averaged recovered field interpolated at the same physical point,
* the observed convergence order and the Richardson limit of each protocol,

all against the frozen Q8 reference of ``benchmarks/cook_membrane_stress_reference.json``.

Reproduce: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_probe_protocol.py``
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.additional_benchmark_fields import cook_fields  # noqa: E402
from tensorfem.cook_probe_protocol import (  # noqa: E402
    canonical_points,
    interpolate_nodal_stress,
    nodal_recovered_stress,
    normalized_error,
    one_sided_values,
    probe_grid_contact,
    von_mises,
)
from tensorfem.cook_stress_reference import plane_stress_matrix, q4_strain_matrix  # noqa: E402

REFERENCE = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"
OUTPUT = ROOT / "benchmarks" / "cook_probe_protocol.json"
LEVELS = (8, 16, 32, 64)


def merged_protocol_value(case: dict, point) -> dict:
    """Reproduce the merged protocol's element selection: floor tie-break on a square grid.

    ``cook_stress_reference.platform_q4_stress_at`` selects element ``(floor(xi*n),
    floor(eta*n))``; the arithmetic is replayed here so the audit pins the merged
    number without importing a function whose name promises a validity it lacks.
    """
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    displacement = np.asarray(case["displacement"], dtype=np.float64).reshape(-1, 2)
    divisions = int(round(math.sqrt(len(elements))))
    from tensorfem.cook_stress_reference import inverse_map

    xi, eta = inverse_map(np.asarray(point, dtype=np.float64))
    i = min(int(math.floor(xi * divisions)), divisions - 1)
    j = min(int(math.floor(eta * divisions)), divisions - 1)
    local = (2.0 * (xi * divisions - i) - 1.0, 2.0 * (eta * divisions - j) - 1.0)
    conn = elements[j * divisions + i]
    matrix, _ = q4_strain_matrix(local[0], local[1], nodes[conn])
    stress = plane_stress_matrix() @ matrix @ displacement[conn].ravel()
    return {
        "element": int(j * divisions + i),
        "local": [float(local[0]), float(local[1])],
        "stress": [float(s) for s in stress],
        "von_mises": von_mises(stress),
    }


def restart_order(values: list[float], levels: tuple[int, ...]) -> dict:
    """Observed order from the two finest pairs plus Richardson limit at h -> 0."""
    fine, mid, coarse = values[-1], values[-2], values[-3]
    denominator = mid - coarse
    order = None
    if abs(denominator) > 0.0:
        ratio = abs(fine - mid) / abs(denominator)
        if ratio > 0.0:
            order = math.log(ratio) / math.log(levels[-2] / levels[-1])
    if order is None or not math.isfinite(order) or order <= 0.05:
        order = None
        estimate = fine
    else:
        estimate = fine + (fine - mid) / (2.0**order - 1.0)
    return {
        "levels": list(levels),
        "values": list(values),
        "observed_order": order,
        "richardson_limit": float(estimate),
    }


def main() -> dict:
    frozen = json.loads(REFERENCE.read_text())
    reference = frozen["reference"]
    points = canonical_points(reference)
    scale = max(point["von_mises_reference"] for point in points)

    rows = []
    for level in LEVELS:
        case = cook_fields(level)
        nodal = nodal_recovered_stress(case["nodes"], case["elements"], case["displacement"])
        entry = {"divisions": level, "points": []}
        for point in points:
            one_sided = one_sided_values(
                case["nodes"], case["elements"], case["displacement"], point["point"]
            )
            merged = merged_protocol_value(case, point["point"])
            recovered = interpolate_nodal_stress(
                case["nodes"], case["elements"], nodal, point["point"]
            )
            magnitudes = [candidate["von_mises"] for candidate in one_sided["candidates"]]
            target = point["von_mises_reference"]
            entry["points"].append(
                {
                    "xi": point["xi"],
                    "eta": point["eta"],
                    "point": point["point"],
                    "grid_contact": probe_grid_contact(level, point["xi"], point["eta"]),
                    "reference_von_mises": target,
                    "candidates": [
                        {"element": c["element"], "local": c["local"], "von_mises": c["von_mises"]}
                        for c in one_sided["candidates"]
                    ],
                    "candidate_count": one_sided["candidate_count"],
                    "candidate_min": one_sided["von_mises_min"],
                    "candidate_max": one_sided["von_mises_max"],
                    "candidate_spread_relative": one_sided["spread_relative"],
                    "ambiguous": not one_sided["well_posed"],
                    "reference_inside_candidate_span": bool(
                        one_sided["von_mises_min"] - 1e-12
                        <= target
                        <= one_sided["von_mises_max"] + 1e-12
                    ),
                    "merged": {
                        "von_mises": merged["von_mises"],
                        "element": merged["element"],
                        "local": merged["local"],
                        "relative_error": normalized_error(merged["von_mises"], target, scale),
                    },
                    "symmetrised": {
                        "von_mises": float(np.mean(magnitudes)),
                        "relative_error": normalized_error(
                            float(np.mean(magnitudes)), target, scale
                        ),
                    },
                    "recovered": {
                        "von_mises": recovered["von_mises"],
                        "element": recovered["element"],
                        "alternate_element_spread": recovered["alternate_element_spread"],
                        "relative_error": normalized_error(recovered["von_mises"], target, scale),
                    },
                }
            )
        rows.append(entry)

    protocols = ("merged", "symmetrised", "recovered")
    summary = {}
    for protocol in protocols:
        per_point = {}
        for index, point in enumerate(points):
            series = [row["points"][index][protocol]["von_mises"] for row in rows]
            errors = [row["points"][index][protocol]["relative_error"] for row in rows]
            fit = restart_order(series, LEVELS)
            per_point[f"xi{point['xi']}_eta{point['eta']}"] = {
                "errors_by_level": errors,
                "limit_relative_error": normalized_error(
                    fit["richardson_limit"], point["von_mises_reference"], scale
                ),
                **fit,
            }
        max_errors = [max(row["points"][i][protocol]["relative_error"] for i in range(len(points)))
                      for row in rows]
        monotone = all(
            fine < coarse for coarse, fine in zip(max_errors, max_errors[1:])
        )
        summary[protocol] = {
            "max_relative_error_by_level": max_errors,
            "finest_max_relative_error": max_errors[-1],
            "monotone": bool(monotone),
            "per_point": per_point,
            "qualified_at_finest": bool(max_errors[-1] < 0.03),
        }

    spreads = [point["candidate_spread_relative"] for point in rows[-1]["points"]]
    payload = {
        "case": "cook-membrane",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "source": str(REFERENCE.relative_to(ROOT)),
            "levels": reference["levels"],
            "max_gci_fine_relative": max(
                point["von_mises"]["gci_fine_relative"] for point in reference["points"]
            ),
            "documented_residual": 0.0502,
        },
        "levels": list(LEVELS),
        "probe_protocol": {
            "merged_tie_break": "floor(xi*n), floor(eta*n) in "
                                "cook_stress_reference.platform_q4_stress_at",
            "symmetrised": "arithmetic mean of all admissible one-sided Q4 values at the point",
            "recovered": "2x2 Gauss-point stress extrapolated to nodes, area-weighted nodal "
                         "average, bilinear interpolation at the point",
            "error_convention": "|computed-reference| / max(|reference|, 0.05*von Mises scale)",
        },
        "probe_sites": {
            "on_element_edge_at_all_levels": bool(
                all(
                    any(contact["on_element_edge"])
                    for row in rows
                    for contact in [point["grid_contact"] for point in row["points"]]
                )
            ),
            "ambiguous_points_finest": sum(1 for point in rows[-1]["points"] if point["ambiguous"]),
            "candidate_spread_relative_finest": spreads,
            "max_spread_relative_finest": max(spreads),
            "reference_inside_span_finest": sum(
                1 for point in rows[-1]["points"] if point["reference_inside_candidate_span"]
            ),
        },
        "protocols": summary,
        "cases": rows,
        "verdict": {
            "merged_residual_reproduced": summary["merged"]["finest_max_relative_error"],
            "merged_residual_matches_documented": bool(
                abs(summary["merged"]["finest_max_relative_error"] - 0.0502) < 0.004
            ),
            "ambiguity_exceeds_residual": bool(
                max(spreads) > 3.0 * summary["merged"]["finest_max_relative_error"]
            ),
            "recovered_qualified": summary["recovered"]["qualified_at_finest"],
            "symmetrised_qualified": summary["symmetrised"]["qualified_at_finest"],
        },
        "status": "probe-protocol-artifact" if (
            summary["merged"]["finest_max_relative_error"] > 0.03
            and (summary["recovered"]["qualified_at_finest"]
                 or summary["symmetrised"]["qualified_at_finest"])
        ) else "withheld",
    }
    OUTPUT.write_text(json.dumps(payload, indent=1))
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps({key: result[key] for key in ("status", "verdict", "probe_sites")}, indent=1))
    for name, block in result["protocols"].items():
        print(
            f"{name:14s} max_err_by_level={[round(v, 5) for v in block['max_relative_error_by_level']]}"
            f" monotone={block['monotone']} qualified={block['qualified_at_finest']}"
        )