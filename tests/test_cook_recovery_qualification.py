"""Guard tests for the production Cook membrane stress-recovery qualification.

The claim under test is narrow and falsifiable: the *same* production displacement
solution that leaves the stored element-centre field 5.02 % off the independent Q8
reference yields a Gauss-point nodal recovery below the 3 % gate, the recovery is
single valued where the element-centre field is not, and the qualification record
fails closed on every fabricated defect we can inject.  The published evidence JSON
is pinned as well, so the shipped numbers cannot drift away from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfem import cook_probe_protocol, stress_recovery
from tensorfem.additional_benchmark_fields import cook_fields
from tensorfem.cook_recovery_qualification import (
    GATE,
    cook_recovery_case,
    cook_recovery_study,
    load_frozen_reference,
)
from tensorfem.stress_recovery import (
    interpolate_nodal_stress,
    nodal_recovered_stress,
    recovery_error,
    von_mises,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_recovery_qualification.json"
#: Values documented in docs/cook-membrane-stress-recovery-qualification.md.
DOCUMENTED_MAX_ERROR = {8: 0.0802311, 16: 0.0195144, 32: 0.00483792, 64: 0.00120190}
DOCUMENTED_MAX_COMPONENT_ERROR = {64: 0.0106786}
LEVELS = (8, 16, 32, 64)


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_frozen_reference()


@pytest.fixture(scope="module")
def study(reference) -> dict:
    return cook_recovery_study(LEVELS, reference)


@pytest.fixture(scope="module")
def cases(study) -> dict:
    return {case["mesh"]: case for case in study["cases"]}


def _uniform_stress_mesh(divisions: int = 4):
    """Square mesh with a linear displacement field => exactly uniform stress."""
    coordinates = np.linspace(0.0, 1.0, divisions + 1)
    nodes = np.array([[x, y] for y in coordinates for x in coordinates], dtype=np.float64)
    elements = []
    for row in range(divisions):
        for column in range(divisions):
            lower_left = row * (divisions + 1) + column
            elements.append(
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + divisions + 2,
                    lower_left + divisions + 1,
                ]
            )
    return nodes, np.asarray(elements, dtype=np.int64)


def _uniform_displacement(nodes: np.ndarray) -> np.ndarray:
    """u = (eps_x x, eps_y y) with a prescribed shear, i.e. a constant strain field."""
    eps_x, eps_y, gamma = 1.0e-3, -4.0e-4, 6.0e-4
    return np.stack((eps_x * nodes[:, 0] + 0.5 * gamma * nodes[:, 1],
                     eps_y * nodes[:, 1] + 0.5 * gamma * nodes[:, 0]), axis=1)


def test_single_source_of_truth_with_the_audit_module():
    """The audit re-exports the production recovery; no second implementation exists."""
    assert cook_probe_protocol.nodal_recovered_stress is stress_recovery.nodal_recovered_stress
    assert cook_probe_protocol.gauss_point_stresses is stress_recovery.gauss_point_stresses
    assert cook_probe_protocol.interpolate_nodal_stress is stress_recovery.interpolate_nodal_stress


def test_recovery_reproduces_a_uniform_stress_field_exactly():
    """Patch property: a constant strain field must survive recovery to rounding."""
    nodes, elements = _uniform_stress_mesh()
    displacement = _uniform_displacement(nodes)
    recovered = nodal_recovered_stress(nodes, elements, displacement)
    # |0 - 0.001| / max(|0.001|, 0.05 * 1.0) -> the 5 % reference floor governs.
    assert recovery_error(0.0, 0.001, 1.0) == pytest.approx(0.02)
    expected = recovered[0]
    assert np.allclose(recovered, expected, rtol=0.0, atol=1e-12), "recovery must be uniform"
    # Independent check of the constant: plane stress, E=1, nu=1/3.
    nu = 1.0 / 3.0
    factor = 1.0 / (1.0 - nu * nu)
    eps_x, eps_y, gamma = 1.0e-3, -4.0e-4, 6.0e-4
    assert expected[0] == pytest.approx(factor * (eps_x + nu * eps_y), rel=1e-12)
    assert expected[1] == pytest.approx(factor * (eps_y + nu * eps_x), rel=1e-12)
    assert expected[2] == pytest.approx(factor * (1.0 - nu) / 2.0 * gamma, rel=1e-12)


def test_recovered_field_is_single_valued_at_the_published_probes(cases):
    """The point that the element-centre field cannot answer has a unique answer here."""
    for mesh in (16, 32, 64):
        case = cases[mesh]
        assert case["max_well_posed_spread"] < 1e-12
        for row in case["points"]:
            assert row["well_posed_spread"] < 1e-12
            assert row["candidate_count"] >= 1


def test_element_centre_protocol_defect_is_preserved_not_hidden(cases):
    """The stored field stays ill-posed: the audit's 19.9 % one-sided spread remains."""
    finest = cases[64]
    spreads = {
        (round(row["xi"], 2), round(row["eta"], 2)): row["one_sided_spread_relative"]
        for row in finest["points"]
    }
    assert spreads[(0.5, 0.5)] == pytest.approx(0.19919, abs=5e-5)
    assert max(spreads.values()) > 5 * GATE


def test_finest_recovery_meets_the_gate_while_the_stored_field_does_not(cases):
    finest = cases[64]
    assert finest["max_error"] < GATE
    assert max(
        max(row["component_errors"].values()) for row in finest["points"]
    ) < GATE
    assert finest["max_element_centre_error"] > GATE
    assert finest["recovery_qualified"] is True
    assert finest["fail_reasons"] == []


def test_coarse_levels_fail_closed_and_are_retained(study, cases):
    """n=8 misses on von Mises, n=16 misses on components; both stay in the ladder."""
    assert cases[8]["recovery_qualified"] is False
    assert any("pointwise_error" in reason for reason in cases[8]["fail_reasons"])
    assert cases[16]["max_error"] < GATE
    assert cases[16]["recovery_qualified"] is False
    assert any("component_error" in reason for reason in cases[16]["fail_reasons"])
    assert study["recovery_qualified"] is True
    assert study["monotone"] is True
    assert [row["mesh"] for row in study["rows"]] == list(LEVELS)


def test_documented_ladder_matches_the_published_evidence(cases):
    for mesh, expected in DOCUMENTED_MAX_ERROR.items():
        assert cases[mesh]["max_error"] == pytest.approx(expected, rel=1e-4)
    for mesh, expected in DOCUMENTED_MAX_COMPONENT_ERROR.items():
        assert cases[mesh]["max_component_error"] == pytest.approx(expected, rel=1e-4)


def test_shipped_evidence_json_agrees_with_the_code(study):
    payload = json.loads(EVIDENCE.read_text())
    assert payload["levels"] == list(LEVELS)
    assert payload["gate"] == GATE
    assert payload["recovery_qualified"] is True
    assert payload["monotone"] is True
    assert payload["reference"]["sha256"] == study["reference"]["sha256"]
    for stored, fresh in zip(payload["rows"], study["rows"]):
        assert stored["mesh"] == fresh["mesh"]
        assert stored["max_error"] == pytest.approx(fresh["max_error"], rel=1e-9)
        assert stored["max_component_error"] == pytest.approx(
            fresh["max_component_error"], rel=1e-9
        )
        assert stored["recovery_qualified"] == fresh["recovery_qualified"]
    for stored, fresh in zip(payload["finest_points"], study["cases"][-1]["points"]):
        assert stored["error"] == pytest.approx(fresh["error"], rel=1e-9)


def test_stored_production_field_still_carries_no_stress_claim():
    """The element-centre field is untouched: the old flag and record keys are stable."""
    case = cook_fields(8)
    assert case["stress_accuracy_qualified"] is False
    assert set(case) == {
        "mesh",
        "nodes",
        "elements",
        "displacement",
        "centers",
        "stress",
        "von_mises",
        "reference",
        "verification",
        "stress_accuracy_qualified",
    }


def test_gate_blocks_a_perturbed_field():
    """Negative control: a 5 % high field must be rejected, not averaged into a pass."""
    case = cook_fields(16)
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    field = nodal_recovered_stress(nodes, elements, case["displacement"])
    reference = load_frozen_reference()
    errors = []
    for point in reference["points"]:
        inflated = interpolate_nodal_stress(nodes, elements, field * 1.05, point["point"])
        errors.append(recovery_error(inflated["von_mises"], point["von_mises"], reference["scale"]))
    assert max(errors) > GATE


def test_case_fails_closed_on_a_wrong_reference(reference):
    """Negative control: a reference inflated by 20 % must disqualify the case."""
    inflated = json.loads(json.dumps(reference))
    for point in inflated["points"]:
        point["von_mises"] *= 1.2
        point["components"] = {key: value * 1.2 for key, value in point["components"].items()}
    case = cook_recovery_case(32, inflated)
    assert case["recovery_qualified"] is False
    assert any("pointwise_error" in reason for reason in case["fail_reasons"])


def test_case_fails_closed_when_the_reference_is_missing(tmp_path):
    runs = cook_recovery_case.__module__
    assert runs == "tensorfem.cook_recovery_qualification"
    from tensorfem import cook_recovery_qualification

    missing = tmp_path / "absent.json"
    original = cook_recovery_qualification.DEFAULT_REFERENCE_PATH
    try:
        cook_recovery_qualification.DEFAULT_REFERENCE_PATH = missing
        case = cook_recovery_case(8, None)
    finally:
        cook_recovery_qualification.DEFAULT_REFERENCE_PATH = original
    assert case["recovery_qualified"] is False
    assert case["reference"] is None
    assert any("reference_unavailable" in reason for reason in case["fail_reasons"])


def test_reference_loader_rejects_a_point_set_it_did_not_produce(tmp_path):
    """A reference whose probe sites differ from the production sample set is refused."""
    payload = json.loads((ROOT / "benchmarks" / "cook_membrane_stress_reference.json").read_text())
    payload["reference"]["points"][0]["xi"] = 0.25
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_frozen_reference(tampered)


def test_recovery_error_uses_the_reference_scaled_floor():
    """Small reference values cannot inflate the error beyond the 5 % floor."""
    assert recovery_error(0.02, 0.001, 1.0) == pytest.approx(0.019 / 0.05)
    assert von_mises([1.0, 1.0, 0.0]) == pytest.approx(1.0)