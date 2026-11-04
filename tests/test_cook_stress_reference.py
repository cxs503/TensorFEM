"""Independent-Q8 audit of the Cook membrane stress field (validation-debt record).

The tests encode the two *positive* results (the independent Q8 code reproduces the
published deflection and is self-converged at fixed interior points) and the *negative*
result: the audit does not certify the production Q4 stress field on the arbitrary-point
criterion, so the recorded evidence must never read ``qualified``.  The negative test is
deliberate - it forbids turning an unresolved discrepancy into a certification.
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
    assert evidence["status"] in ("withheld", "measured-not-qualified")
    assert evidence["comparison"]["qualified"] is False
    assert evidence["status"] != "qualified"
    # The committed evidence must describe the code as it is.  After the element-Jacobian
    # repair the audit reproduces the stored element-centre field to machine precision, so
    # a failing cross-check or strain-matrix record in this file is stale data, not a
    # finding: regenerate with scripts/audit_cook_membrane_stress.py.
    cross_check = evidence.get("recovery_cross_check", {"passed": False})
    assert cross_check["passed"] is True
    assert evidence["strain_matrix_verification"]["passed"] is True
    for point in evidence["reference"]["points"]:
        assert point["von_mises"]["gci_fine_relative"] < 0.01


# --------------------------------------------------------------------------------------
# Element-interface aware sampling (2026-11-03 closure of the "not certifiable" debt).
#
# The original sample points lie exactly on inter-element interfaces for every dyadic
# mesh, so their pointwise Q4 stress is two-valued.  These tests lock the machine-checkable
# facts: the legacy points are degenerate, the interior points are not, the two-sided
# bracket reproduces the stored element-centre field, and the recorded evidence may not
# claim certification while the finest error is still above 3 %.
# --------------------------------------------------------------------------------------


def test_legacy_sample_points_sit_on_element_interfaces_on_the_whole_ladder():
    for divisions in (8, 16, 32, 64, 128):
        margins = reference.sample_margins(reference.SAMPLE_POINTS, divisions)
        assert margins["on_interface"] is True
        assert margins["min_margin_cells"] <= reference.INTERFACE_TOLERANCE


def test_interior_sample_points_keep_a_margin_from_every_interface():
    for divisions in (8, 16, 32, 64, 128):
        margins = reference.sample_margins(reference.INTERIOR_POINTS, divisions)
        assert margins["on_interface"] is False
        assert margins["min_margin_cells"] >= 0.1


def test_two_sided_bracket_reproduces_the_stored_field_at_element_centres():
    from tensorfem.additional_benchmark_fields import cook_fields

    case = cook_fields(8)
    divisions = case["mesh"]
    stored = np.asarray(case["stress"], dtype=np.float64)
    checked = 0
    for j in range(0, divisions, 2):
        for i in range(0, divisions, 2):
            point = reference.map_point((i + 0.5) / divisions, (j + 0.5) / divisions)
            sides = reference.platform_q4_stress_sides(
                case["nodes"], case["elements"], case["displacement"], point
            )
            assert sides["inside"] is True
            assert sides["on_interface"] is False
            assert len(sides["sides"]) == 1
            target = stored[j * divisions + i]
            recomputed = np.array(
                [sides["sides"][0][key] for key in ("sxx", "syy", "sxy")]
            )
            assert np.allclose(recomputed, target, rtol=0, atol=1e-12)
            checked += 1
    assert checked == 16


def test_interface_point_returns_two_sides_with_a_finite_bracket():
    from tensorfem.additional_benchmark_fields import cook_fields

    case = cook_fields(8)
    point = reference.map_point(0.5, 0.5)  # exactly on an interface
    sides = reference.platform_q4_stress_sides(
        case["nodes"], case["elements"], case["displacement"], point
    )
    assert sides["on_interface"] is True
    assert len(sides["sides"]) >= 2
    low, high = sides["von_mises_bracket"]
    assert low <= high
    assert sides["von_mises_spread"] >= 0.0


def test_bracket_error_is_zero_inside_the_bracket_and_counts_outside():
    inside = reference.bracket_error(1.0, [0.9, 1.1], 1.0)
    assert inside["inside"] is True and inside["normalized"] == 0.0
    outside = reference.bracket_error(1.0, [1.1, 1.2], 1.0)
    assert outside["inside"] is False
    assert outside["gap"] == pytest.approx(0.1)


def test_convergence_prediction_recovers_a_first_order_sequence():
    levels = [8, 16, 32, 64]
    errors = [0.32 / n for n in levels]
    prediction = reference.convergence_prediction(levels, errors, target=0.03)
    assert prediction["observed_order"] == pytest.approx(1.0, abs=1e-9)
    assert prediction["predicted_divisions"] == pytest.approx(64 * (errors[-1] / 0.03), rel=1e-9)
    assert prediction["valid"] is True
    flat = reference.convergence_prediction(levels, [0.05, 0.05, 0.05, 0.05])
    assert flat["valid"] is False and flat["predicted_divisions"] is None


def test_profile_agreement_is_zero_for_identical_profiles_and_flags_a_shift():
    profile = {"y": [0.0, 1.0, 2.0], "v": [1.0, 2.0, 3.0]}
    assert reference.profile_agreement(profile, profile)["max_abs_difference"] == 0.0
    shifted = {"y": [0.0, 1.0, 2.0], "v": [1.1, 2.1, 3.1]}
    agreement = reference.profile_agreement(profile, shifted)
    assert agreement["max_abs_difference"] == pytest.approx(0.1)
    assert agreement["max_relative_difference"] > 0.03


def test_evidence_records_a_trustworthy_ladder_and_an_honest_verdict():
    if not EVIDENCE.exists():
        pytest.skip("audit evidence has not been generated in this checkout")
    evidence = json.loads(EVIDENCE.read_text())
    cross_check = evidence["recovery_cross_check"]
    assert cross_check["passed"] is True
    assert np.isfinite(cross_check["max_relative_difference"])
    assert cross_check["max_relative_difference"] <= 1e-12
    ambiguity = evidence["interface_ambiguity"]
    assert ambiguity["interior_points_clear_interfaces"] is True
    assert ambiguity["legacy_points_on_interface"] is True
    comparison = evidence["comparison"]
    assert comparison["cases"][0]["divisions"] == 8
    assert comparison["cases"][-1]["divisions"] >= 64
    finest = comparison["cases"][-1]
    assert finest["max_bracket_error"] > 0.03
    assert comparison["finest_passed"] is False
    # ASSERTION-REBASELINE (rebuild arbitration, flagged for owner): the union keeps
    # main/m2814's three-state status vocabulary ("withheld" = tool untrusted,
    # "measured-not-qualified" = measured, misses the 3 %% gate); the fail-closed
    # intent of this test (never "qualified") is preserved by set membership.
    assert evidence["status"] in ("withheld", "measured-not-qualified")
    assert evidence["loaded_edge_profiles"]["agreement_finest"]["max_relative_difference"] < 0.01
    convergence = evidence["convergence"]
    if convergence["valid"]:
        assert convergence["predicted_divisions"] > convergence["levels"][-1]
