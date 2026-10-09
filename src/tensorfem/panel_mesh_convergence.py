"""Evidence-only mesh-convergence gate for nonlinear panel peak paths."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


MESH_CONVERGENCE_SCHEMA = "tensorfem.panel-mesh-convergence/1.0"


def evaluate_panel_mesh_convergence(
    manifests: Sequence[Mapping[str, object]], *,
    required_divisions: Sequence[int] = (4, 8, 12),
    relative_tolerance: float = .03,
) -> dict[str, object]:
    """Evaluate adjacent-mesh peak convergence without extrapolating data.

    Only manifests that explicitly confirm both a peak and post-peak response
    are eligible.  Missing or incomplete meshes produce a blocked report, not
    a synthetic peak estimate.
    """
    if not required_divisions or any(n < 2 for n in required_divisions):
        raise ValueError("required_divisions must contain valid meshes")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be finite and positive")

    eligible: dict[int, float] = {}
    rejected: dict[int, str] = {}
    for manifest in manifests:
        divisions = int(manifest.get("divisions", manifest.get("mesh_divisions", 0)))
        if divisions not in required_divisions:
            continue
        peak = manifest.get("peak_force_n")
        if not manifest.get("peak_confirmed"):
            rejected[divisions] = "peak_not_confirmed"
        elif not manifest.get("post_peak_observed"):
            rejected[divisions] = "post_peak_not_observed"
        elif peak is None or not math.isfinite(float(peak)) or float(peak) <= 0:
            rejected[divisions] = "invalid_peak_force"
        else:
            eligible[divisions] = float(peak)

    missing = [n for n in required_divisions if n not in eligible]
    comparisons = []
    for coarse, fine in zip(required_divisions, required_divisions[1:]):
        if coarse not in eligible or fine not in eligible:
            continue
        relative_change = abs(eligible[fine] - eligible[coarse]) / eligible[fine]
        comparisons.append({
            "coarse_divisions": coarse,
            "fine_divisions": fine,
            "relative_peak_change": relative_change,
            "passed": relative_change <= relative_tolerance,
        })
    complete = not missing and len(comparisons) == len(required_divisions) - 1
    passed = complete and all(item["passed"] for item in comparisons)
    return {
        "schema": MESH_CONVERGENCE_SCHEMA,
        "status": "passed" if passed else ("failed" if complete else "blocked"),
        "passed": passed,
        "relative_tolerance": relative_tolerance,
        "required_divisions": list(required_divisions),
        "eligible_peak_force_n": eligible,
        "missing_divisions": missing,
        "rejected": rejected,
        "comparisons": comparisons,
    }
