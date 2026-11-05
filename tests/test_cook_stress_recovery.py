"""Production promotion of the Cook-membrane recovered stress field.

The audited recovery must (a) be exact on a manufactured uniform stress state, so
the operator itself carries no error, (b) reach the 3 % gate monotonically, and
(c) never loosen the fail-closed verdict on the raw element-centre field.  The
gate is also tested with injected bad data, because a gate that only ever sees
good data is not a gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfem.additional_benchmark_fields import cook_fields
from tensorfem.cook_probe_protocol import interpolate_nodal_stress, plane_stress_matrix
from tensorfem.cook_stress_recovery import (
    PROTOCOL,
    load_reference,
    measure,
    qualify,
    recovered_nodal_stress,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "cook_stress_recovery.json"


def _manufactured(target_stress):
    """Displacement field producing ``target_stress`` on the Cook mesh (plane stress)."""
    case = cook_fields(4)
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    matrix = plane_stress_matrix()
    strain = np.linalg.solve(matrix, np.asarray(target_stress, dtype=np.float64))
    grad = np.array([[strain[0], strain[2] / 2.0], [strain[2] / 2.0, strain[1]]])
    return case, nodes @ grad.T


def test_recovery_is_exact_on_a_manufactured_uniform_stress_state():
    """Independent prior: a constant stress field must be recovered to round-off."""
    target = np.array([2.5e-1, -6.0e-2, 4.0e-2])
    case, displacement = _manufactured(target)
    recovered = recovered_nodal_stress(case["nodes"], case["elements"], displacement)
    field = recovered["field"]
    assert recovered["protocol"] == PROTOCOL
    error = np.max(np.abs(field - target)) / np.max(np.abs(target))
    assert error < 1e-12, error
    sample = interpolate_nodal_stress(
        case["nodes"], case["elements"], field, [24.0, 52.0]
    )
    assert np.max(np.abs(np.asarray(sample["stress"]) - target)) / np.max(
        np.abs(target)
    ) < 1e-12


def test_production_payload_keeps_the_raw_field_verdict_and_adds_the_recovery():
    plain = cook_fields(8)
    assert "recovered_nodal_stress" not in plain
    assert plain["stress_accuracy_qualified"] is False

    enriched = cook_fields(8, include_recovery=True)
    assert enriched["stress_accuracy_qualified"] is False
    assert enriched["recovered_stress_protocol"] == PROTOCOL
    nodes = np.asarray(enriched["nodes"])
    assert np.asarray(enriched["recovered_nodal_stress"]).shape == (len(nodes), 3)
    assert np.asarray(enriched["recovered_von_mises"]).shape == (len(nodes),)
    # The raw element field must not be silently replaced by the recovered one.
    assert np.asarray(enriched["stress"]).shape == (len(enriched["elements"]), 3)


def test_recovered_field_is_single_valued_across_element_interfaces():
    """A nodal field removes the one-sided ambiguity that broke the merged protocol."""
    case = cook_fields(8, include_recovery=True)
    field = np.asarray(case["recovered_nodal_stress"], dtype=np.float64)
    for point in (entry["point"] for entry in load_reference()["points"]):
        sample = interpolate_nodal_stress(case["nodes"], case["elements"], field, point)
        assert sample["alternate_element_spread"] < 1e-12, sample["alternate_element_spread"]


def test_gate_qualifies_the_recovered_field_monotonically_below_three_percent():
    verdict = qualify((8, 16, 32))
    errors = verdict["error_sequence"]
    assert verdict["stress_recovery_qualified"] is True, verdict["blocking_reasons"]
    assert verdict["blocking_reasons"] == []
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 0.03
    assert verdict["observed_order"] > 1.0
    assert verdict["stress_accuracy_qualified"] is False
    assert verdict["element_centre_field"]["qualified"] is False


def test_gate_refuses_non_monotone_and_out_of_tolerance_data():
    measured = measure((8, 16, 32))
    inflated = json.loads(json.dumps(measured))
    inflated["levels"][-1]["max_relative_error"] = (
        inflated["levels"][0]["max_relative_error"] * 2.0
    )
    verdict = qualify((8, 16, 32), measured=inflated)
    assert verdict["stress_recovery_qualified"] is False
    assert "non_monotone_error" in verdict["blocking_reasons"]
    assert "finest_above_tolerance" in verdict["blocking_reasons"]

    flat = json.loads(json.dumps(measured))
    for index, row in enumerate(flat["levels"]):
        row["max_relative_error"] = 0.02 + 0.001 * index
    verdict = qualify((8, 16, 32), measured=flat)
    assert verdict["stress_recovery_qualified"] is False
    assert "non_monotone_error" in verdict["blocking_reasons"]


def test_gate_rejects_a_single_level_and_a_loose_tolerance():
    with pytest.raises(ValueError, match="two mesh levels"):
        qualify((8,))
    with pytest.raises(ValueError, match="at most 3%"):
        qualify((8, 16), tolerance=0.05)


def test_frozen_reference_and_published_artifact_are_digest_locked():
    reference = load_reference()
    digest = hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()
    assert reference["sha256"] == digest
    assert reference["levels"] == [12, 24, 48]

    stored = json.loads(ARTIFACT.read_text())
    fresh = measure((16,))
    assert stored["reference"]["sha256"] == reference["sha256"]
    assert stored["levels"][1]["divisions"] == 16
    assert stored["levels"][1]["max_relative_error"] == pytest.approx(
        fresh["levels"][0]["max_relative_error"], rel=1e-9, abs=1e-12
    )
    assert stored["stress_recovery_qualified"] is True
    assert stored["stress_accuracy_qualified"] is False