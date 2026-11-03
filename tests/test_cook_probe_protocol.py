"""Guard tests for the Cook probe-protocol audit.

The audit's claim is narrow and falsifiable, so it is pinned here: the merged
pointwise protocol (i) evaluates the Q4 stress at sites where the field is
discontinuous, (ii) therefore fails to converge while reproducing the documented
5.02 % residual, and (iii) the same displacement field, read through a Gauss-point
nodal recovery, meets the 3 % gate with second-order convergence.  The evidence JSON
that ships with the repository is pinned as well so the published numbers cannot drift
away from the code that produced them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfem.additional_benchmark_fields import cook_fields
from tensorfem.cook_probe_protocol import (
    canonical_points,
    interpolate_nodal_stress,
    nodal_recovered_stress,
    one_sided_values,
    probe_grid_contact,
)
from tensorfem.cook_stress_reference import map_point

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_probe_protocol.json"
REFERENCE = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"
GATE = 0.03


@pytest.fixture(scope="module")
def reference_points() -> list[dict]:
    frozen = json.loads(REFERENCE.read_text())
    return canonical_points(frozen["reference"])


@pytest.fixture(scope="module")
def cases() -> dict:
    return {level: cook_fields(level) for level in (8, 16, 32, 64)}


def _scale(points) -> float:
    """Global von Mises scale used as the 5 % floor in the normalisation."""
    return max(point["von_mises_reference"] for point in points)


def _recovered_errors(case, point) -> float:
    field = nodal_recovered_stress(case["nodes"], case["elements"], case["displacement"])
    return interpolate_nodal_stress(case["nodes"], case["elements"], field, point["point"])


def test_canonical_probe_sites_are_ill_posed_on_the_q4_grid(reference_points, cases):
    """Every canonical point sits on an element edge, so the one-sided values differ."""
    for level in (8, 16, 32):
        case = cases[level]
        for point in reference_points:
            contact = probe_grid_contact(level, point["xi"], point["eta"])
            assert contact["on_element_edge"][0] or contact["on_element_edge"][1]
            values = one_sided_values(
                case["nodes"], case["elements"], case["displacement"], point["point"]
            )
            assert values["candidate_count"] == (
                4 if contact["on_grid_node"] else 2
            ), f"level {level} point {(point['xi'], point['eta'])}"
            assert not values["well_posed"]
    coarse = cases[8]
    spreads = [
        one_sided_values(coarse["nodes"], coarse["elements"], coarse["displacement"], p["point"])[
            "spread_relative"
        ]
        for p in reference_points
    ]
    assert min(spreads) > 0.02


def test_interior_point_has_a_single_admissible_value(cases):
    """Control: off the grid lines the Q4 value is unique, so the ambiguity is site-borne."""
    case = cases[8]
    interior = map_point(0.31, 0.42)
    values = one_sided_values(
        case["nodes"], case["elements"], case["displacement"], interior
    )
    assert values["candidate_count"] == 1
    assert values["well_posed"]


def test_merged_protocol_reproduces_the_documented_residual(reference_points, cases):
    """The audit must reproduce the merged 5.02 % residual before it may reinterpret it."""
    from scripts.audit_cook_probe_protocol import merged_protocol_value

    case = cases[32]
    errors = []
    for point in reference_points:
        merged = merged_protocol_value(case, point["point"])
        target = point["von_mises_reference"]
        errors.append(abs(merged["von_mises"] - target) / max(abs(target), 0.05 * _scale(reference_points)))
    assert max(errors) == pytest.approx(0.0802, abs=1.5e-3)
    assert max(errors) > GATE


def test_recovered_field_is_second_order_and_clears_the_gate(reference_points, cases):
    history = []
    for level in (8, 16, 32):
        case = cases[level]
        errors = []
        for point in reference_points:
            recovered = _recovered_errors(case, point)
            target = point["von_mises_reference"]
            errors.append(abs(recovered["von_mises"] - target) / max(abs(target), 0.05 * _scale(reference_points)))
        history.append(max(errors))
    assert history[-1] < GATE, history
    assert all(fine < coarse for coarse, fine in zip(history, history[1:])), history
    order = math.log(history[-2] / history[-1]) / math.log(2.0)
    assert order > 1.5, f"observed order {order:.2f}"


def test_recovered_interpolation_is_single_valued_at_element_edges(reference_points, cases):
    """A nodal field is continuous; the recovery therefore removes the probe ambiguity."""
    case = cases[16]
    field = nodal_recovered_stress(case["nodes"], case["elements"], case["displacement"])
    for point in reference_points:
        recovered = interpolate_nodal_stress(
            case["nodes"], case["elements"], field, point["point"]
        )
        assert recovered["alternate_element_spread"] < 1e-12


def test_simple_one_sided_averaging_is_rejected_as_a_fix(reference_points, cases):
    """Fail-closed: symmetrising the one-sided values is not a consistent estimator."""
    history = []
    for level in (8, 16, 32, 64):
        case = cases[level]
        errors = []
        for point in reference_points:
            values = one_sided_values(
                case["nodes"], case["elements"], case["displacement"], point["point"]
            )
            mean = float(np.mean([c["von_mises"] for c in values["candidates"]]))
            target = point["von_mises_reference"]
            errors.append(abs(mean - target) / max(abs(target), 0.05 * _scale(reference_points)))
        history.append(max(errors))
    # The estimator looks acceptable at n=32 and then regresses: not a valid fix.
    assert history[-2] < GATE < history[-1], history


def test_published_evidence_matches_the_code():
    payload = json.loads(EVIDENCE.read_text())
    assert payload["levels"] == [8, 16, 32, 64]
    assert payload["status"] == "probe-protocol-artifact"
    verdict = payload["verdict"]
    assert verdict["merged_residual_matches_documented"] is True
    assert verdict["merged_residual_reproduced"] == pytest.approx(0.0502, abs=4e-3)
    assert verdict["merged_residual_reproduced"] > GATE
    assert verdict["recovered_qualified"] is True
    assert verdict["symmetrised_qualified"] is False
    assert verdict["ambiguity_exceeds_residual"] is True
    recovered = payload["protocols"]["recovered"]
    assert recovered["finest_max_relative_error"] < GATE
    assert recovered["monotone"] is True
    assert payload["probe_sites"]["ambiguous_points_finest"] == 5
    assert payload["probe_sites"]["max_spread_relative_finest"] > 3 * 0.0502
    assert payload["reference"]["documented_residual"] == pytest.approx(0.0502)