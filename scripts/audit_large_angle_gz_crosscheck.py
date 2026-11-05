"""Cross-implementation audit of the large-angle `GZ` capability.

Why this audit exists
---------------------
Two independent exact solutions of the same wall-sided problem now live in the
package:

* ``tensorfem.marine_large_angle_stability.BoxBarge`` -- solves the free-surface
  height by bisection on the immersed-area residual and reports the righting arm
  as a horizontal distance; it refuses the open endpoints ``|heel| >= 90 deg``.
* ``tensorfem.marine_large_angle_gz`` -- clips the section polygon against the
  waterline, refines the waterline offset by bisection on the corner bracket and
  builds ``GZ``/``KN`` curves, cross-curves and dynamic-stability areas over the
  *closed* interval ``[-90, 90]``.

Neither implements the other, so their difference is a real measurement rather
than a tautology.  This script quantifies it over several geometries (keel-edge
emergence first, deck immersion first, coincident limits, slender, wide) and on
both heel signs, and records the two events where the modules must differ by
construction:

* the ``|heel| = 90 deg`` endpoint, where the merged oracle is fail-closed and
  this module returns the exact closed geometry;
* the wall-sided envelope, which both must report identically even though they
  derive it from different code paths.

Output: ``benchmarks/large_angle_gz_crosscheck.json`` (deterministic, sorted).
Exit status is non-zero when any tolerance fails, so the audit is usable as a
gate.  Tolerances are set from the problem scale, not fitted to the result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow `python scripts/...` without install
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.marine_large_angle_gz import (  # noqa: E402
    MAX_HEEL_DEG,
    gz_curve,
    section_at_heel,
    wall_sided_validity_deg,
)
from tensorfem.marine_large_angle_stability import BoxBarge  # noqa: E402

JSON_OUT = ROOT / "benchmarks" / "large_angle_gz_crosscheck.json"

# Geometries chosen so that every ordering of the two wall-sided limits occurs.
GEOMETRIES = (
    dict(name="reference", beam=12.0, depth=6.0, draft=2.0, kg=2.1),
    dict(name="deep_draft_deck_first", beam=12.0, depth=6.0, draft=3.0, kg=3.0),
    dict(name="coincident_limits", beam=12.0, depth=4.0, draft=2.0, kg=2.1),
    dict(name="wide_shallow", beam=20.0, depth=3.0, draft=1.0, kg=1.4),
    dict(name="slender", beam=6.0, depth=6.0, draft=1.0, kg=1.2),
)
LENGTH = 40.0

# Heel sweep for the agreement measurement: 0.5 deg resolution, closed endpoint
# excluded because the merged oracle is declared fail-closed there (measured
# separately below).
SWEEP_STEP_DEG = 0.5
SWEEP_MAX_DEG = 89.5

# Agreement tolerance.  Both engines bisect to machine precision, so the only
# expected difference is float round-off amplified by the lever arms; 1e-10 m is
# ~1e-10 of the beam and far below any physical significance.
ABS_TOL_M = 1e-10


def _angles() -> list[float]:
    count = int(round(SWEEP_MAX_DEG / SWEEP_STEP_DEG))
    return [round(index * SWEEP_STEP_DEG, 10) for index in range(count + 1)]


def _audit_geometry(geom: dict) -> dict:
    beam, depth, draft, kg = geom["beam"], geom["depth"], geom["draft"], geom["kg"]
    oracle = BoxBarge(length=LENGTH, beam=beam, depth=depth, draft=draft, kg=kg)
    curve = gz_curve(beam, depth, draft, kg, _angles())

    worst_abs = 0.0
    worst_angle = 0.0
    worst_scale = 0.0
    for angle, arm in zip(curve.angles_deg, curve.gz):
        reference = oracle.gz(angle)
        difference = abs(arm - reference)
        if difference > worst_abs:
            worst_abs, worst_angle, worst_scale = difference, angle, max(abs(arm), abs(reference))

    # Negative heel: the merged oracle accepts the mirror image.
    mirror_abs = max(
        abs(curve.gz[index] + oracle.gz(-angle))
        for index, angle in enumerate(curve.angles_deg[1:], start=1)
    )

    bilge, deck = wall_sided_validity_deg(beam, depth, draft)
    our_envelope = min(bilge, deck)
    return {
        "name": geom["name"],
        "geometry": dict(beam=beam, depth=depth, draft=draft, kg=kg),
        "heel_count": len(curve.angles_deg),
        "max_abs_gz_difference_m": worst_abs,
        "max_rel_gz_difference": worst_abs / worst_scale if worst_scale else 0.0,
        "angle_of_max_difference_deg": worst_angle,
        "max_abs_gz_difference_negative_heel_m": mirror_abs,
        "envelope_this_module_deg": our_envelope,
        "envelope_merged_oracle_deg": oracle.wall_sided_envelope_deg,
        "envelope_difference_deg": abs(our_envelope - oracle.wall_sided_envelope_deg),
        "gm_this_module_m": curve.gm,
        "gm_merged_oracle_m": oracle.gm,
        "gm_difference_m": abs(curve.gm - oracle.gm),
        "max_gz_m": curve.max_gz,
    }


def _audit_ninety_degree(geom: dict) -> dict:
    """Endpoint where the merged oracle is fail-closed and this module is exact."""
    beam, depth, draft, kg = geom["beam"], geom["depth"], geom["draft"], geom["kg"]
    oracle = BoxBarge(length=LENGTH, beam=beam, depth=depth, draft=draft, kg=kg)
    section = section_at_heel(beam, depth, beam * draft, MAX_HEEL_DEG)
    exact_gz = depth / 2.0 - kg
    exact_xb = beam / 2.0 - beam * draft / (2.0 * depth)
    refused = False
    try:
        oracle.gz(MAX_HEEL_DEG)
    except ValueError:
        refused = True
    return {
        "name": geom["name"],
        "merged_oracle_refuses_90_deg": refused,
        "gz_at_90_deg_m": section.xb * math.cos(math.radians(MAX_HEEL_DEG))
        + (section.zb - kg) * math.sin(math.radians(MAX_HEEL_DEG)),
        "gz_at_90_deg_analytic_m": exact_gz,
        "abs_error_m": abs(
            section.xb * math.cos(math.radians(MAX_HEEL_DEG))
            + (section.zb - kg) * math.sin(math.radians(MAX_HEEL_DEG))
            - exact_gz
        ),
        "xb_at_90_deg_m": section.xb,
        "xb_at_90_deg_analytic_m": exact_xb,
        "zb_at_90_deg_m": section.zb,
        "zb_at_90_deg_analytic_m": depth / 2.0,
    }


def build_evidence() -> dict:
    per_geometry = [_audit_geometry(geom) for geom in GEOMETRIES]
    ninety = [_audit_ninety_degree(geom) for geom in GEOMETRIES]
    return {
        "schema": "tensorfem.marine-large-angle-gz.crosscheck.v1",
        "description": (
            "Independent-implementation agreement between "
            "marine_large_angle_gz (polygon clipping) and "
            "marine_large_angle_stability.BoxBarge (area-residual bisection)"
        ),
        "sweep": {"step_deg": SWEEP_STEP_DEG, "max_deg": SWEEP_MAX_DEG},
        "abs_tolerance_m": ABS_TOL_M,
        "max_abs_gz_difference_m": max(row["max_abs_gz_difference_m"] for row in per_geometry),
        "max_envelope_difference_deg": max(row["envelope_difference_deg"] for row in per_geometry),
        "max_gm_difference_m": max(row["gm_difference_m"] for row in per_geometry),
        "geometries": per_geometry,
        "ninety_degree_endpoint": ninety,
        "max_abs_error_at_90_deg_m": max(row["abs_error_m"] for row in ninety),
        "merged_oracle_refuses_90_deg": all(row["merged_oracle_refuses_90_deg"] for row in ninety),
    }


def check(evidence: dict) -> list[str]:
    failures: list[str] = []
    if evidence["max_abs_gz_difference_m"] > ABS_TOL_M:
        failures.append(
            f"GZ disagreement {evidence['max_abs_gz_difference_m']:.3e} m exceeds {ABS_TOL_M:.0e} m"
        )
    if evidence["max_envelope_difference_deg"] > 1e-12:
        failures.append(f"envelope disagreement {evidence['max_envelope_difference_deg']:.3e} deg")
    if evidence["max_gm_difference_m"] > 1e-12:
        failures.append(f"GM disagreement {evidence['max_gm_difference_m']:.3e} m")
    if evidence["max_abs_error_at_90_deg_m"] > 1e-12:
        failures.append(f"90 deg error {evidence['max_abs_error_at_90_deg_m']:.3e} m")
    if not evidence["merged_oracle_refuses_90_deg"]:
        failures.append("merged oracle no longer refuses 90 deg; coverage claim is stale")
    return failures


def main() -> int:
    evidence = build_evidence()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {JSON_OUT.relative_to(ROOT)}")
    print(
        "max |dGZ| {dgz:.3e} m over {n} geometries, max 90 deg error {e90:.3e} m".format(
            dgz=evidence["max_abs_gz_difference_m"],
            n=len(evidence["geometries"]),
            e90=evidence["max_abs_error_at_90_deg_m"],
        )
    )
    failures = check(evidence)
    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
