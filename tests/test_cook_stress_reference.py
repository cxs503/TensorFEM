"""Independent-Q8 audit of the Cook membrane stress field (validation-debt record).

The tests encode the two *positive* results (the independent Q8 code reproduces the
published deflection and is self-converged at fixed interior points) and the *negative*
result: the audit could not certify the production Q4 stress field, so the recorded
evidence must stay ``withheld``.  The negative test is deliberate - it forbids turning an
unresolved discrepancy into a certification.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfem import cook_stress_reference as reference

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"


def test_q8_shape_functions_are_a_partition_of_unity_with_nodal_delta_property():
    points = ((-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0))
    values = np.array([reference.q8_shape(xi, eta)[0] for xi, eta in points])
    assert np.allclose(values, np.eye(8), atol=1e-15)
    for xi, eta in ((0.31, -0.42), (-0.7, 0.15)):
        shape, derivative = reference.q8_shape(xi, eta)
        assert abs(shape.sum() - 1.0) < 1e-15
        assert np.allclose(derivative.sum(axis=0), 0.0, atol=1e-15)


def test_q8_parent_derivatives_match_central_differences():
    h = 1e-6
    for xi, eta in ((0.3, -0.4), (-0.55, 0.62)):
        _, derivative = reference.q8_shape(xi, eta)
        plus, _ = reference.q8_shape(xi + h, eta)
        minus, _ = reference.q8_shape(xi - h, eta)
        assert np.allclose(derivative[:, 0], (plus - minus) / (2 * h), atol=1e-9)
        plus, _ = reference.q8_shape(xi, eta + h)
        minus, _ = reference.q8_shape(xi, eta - h)
        assert np.allclose(derivative[:, 1], (plus - minus) / (2 * h), atol=1e-9)


def test_q8_element_has_no_rigid_body_energy_and_the_map_inverts():
    coords = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
                       (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)))
    stiffness = reference.q8_element_stiffness(coords)
    translation = np.tile((0.7, -1.3), 8)
    rotation = np.stack((-coords[:, 1], coords[:, 0]), 1).ravel()
    assert abs(float(translation @ stiffness @ translation)) < 1e-12
    assert abs(float(rotation @ stiffness @ rotation)) < 1e-12
    target = reference.map_point(0.37, 0.61)
    xi, eta = reference.inverse_map(target)
    assert np.allclose(reference.map_point(xi, eta), target, atol=1e-12)


def test_q8_solution_carries_the_full_traction_and_converges():
    solution = reference.solve_q8(8)
    assert abs(solution["load_resultant"] - 1.0) < 1e-14
    assert solution["relative_residual"] < 1e-10
    probe = reference.probe_displacement(solution)
    assert probe["coordinates"] == pytest.approx([48.0, 52.0])
    assert probe["relative_error"] < 0.03


def test_q8_reference_is_self_converged_at_the_interior_sample_points():
    coarse = reference.solve_q8(12)
    fine = reference.solve_q8(24)
    for xi, eta in reference.SAMPLE_POINTS:
        a = reference.field_stress(coarse, xi, eta)["von_mises"]
        b = reference.field_stress(fine, xi, eta)["von_mises"]
        assert abs(b - a) / b < 0.02


def test_published_deflection_is_confirmed_by_both_discretisations():
    from tensorfem.additional_benchmark_fields import cook_fields

    platform = cook_fields(16)
    assert platform["verification"]["response_relative_error"] < 0.03
    independent = reference.probe_displacement(reference.solve_q8(24))
    assert independent["relative_error"] < 0.02


def test_audit_evidence_must_not_claim_stress_certification():
    if not EVIDENCE.exists():
        pytest.skip("audit evidence has not been generated in this checkout")
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["status"] == "withheld"
    assert evidence["comparison"]["qualified"] is False
    cross_check = evidence.get("recovery_cross_check", {"passed": False})
    if not cross_check["passed"]:
        assert evidence["status"] != "qualified"
    for point in evidence["reference"]["points"]:
        assert point["von_mises"]["gci_fine_relative"] < 0.01
