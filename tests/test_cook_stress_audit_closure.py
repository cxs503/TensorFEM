"""Closure of the Cook-membrane audit's two open items and its accuracy scope.

The 2026-11 audit left three statements unresolved: (i) its own pointwise Q4 recovery could
not be cross-checked against the stored field, (ii) the Q8 reference loaded-edge profile
seemed to diverge from the production Q4 profile, and (iii) the accuracy of the stored
element-centre field was never measured.  The element-Jacobian repair answered (i) and (ii);
these tests pin the answers and the measurement, and they keep the honest negative result
(the arbitrary-point criterion still misses 3 %) asserted.
"""

import json
from pathlib import Path

import numpy as np

from tensorfem import cook_stress_reference as reference
from tensorfem.additional_benchmark_fields import cook_fields

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"


def test_field_stress_at_point_reproduces_the_parent_coordinate_evaluation():
    solution = reference.solve_q8(12)
    # Parent coordinates on an element boundary are excluded: the Q8 stress field is
    # element-wise discontinuous there, so the containing element (and its value)
    # depends on the last bit of the inverted parent coordinate.
    for xi, eta in ((0.31, 0.52), (0.62, 0.28), (0.85, 0.77)):
        direct = reference.field_stress(solution, xi, eta)
        point = reference.map_point(xi, eta)
        indirect = reference.field_stress_at_point(solution, point)
        assert indirect["map_error"] < 1e-10
        for key in ("sxx", "syy", "sxy", "von_mises"):
            assert abs(indirect[key] - direct[key]) <= 1e-6 * max(abs(direct[key]), 1e-3)


def test_stored_element_centre_field_converges_under_refinement():
    fine_reference = reference.solve_q8(24)
    coarse = reference.element_centre_accuracy(cook_fields(16), fine_reference)
    fine = reference.element_centre_accuracy(cook_fields(32), fine_reference)
    assert fine["all_elements"]["p95_relative_error"] < 0.5 * coarse["all_elements"]["p95_relative_error"]
    assert fine["all_elements"]["median_relative_error"] < 0.01
    assert set(fine) >= {"all_elements", "excluding_clamped_band", "worst", "passed"}


def test_element_centre_gate_fails_on_a_coarse_mesh():
    """Negative control: the scoped gate must not pass at n = 32."""
    accuracy = reference.element_centre_accuracy(cook_fields(32), reference.solve_q8(24))
    assert accuracy["passed"] is False
    assert accuracy["all_elements"]["max_scale_normalized_error"] > 0.03


def test_loaded_edge_profiles_of_both_discretisations_agree():
    """The pre-fix Q8 profile spanned 7.6..52.4; a fixed reference must stay inside 5 %."""
    agreement = reference.loaded_edge_agreement(reference.solve_q8(12), cook_fields(16))
    assert agreement["reference_spread"] < 4.0
    assert agreement["max_relative_to_probe"] < 0.05


def test_convergence_analysis_recovers_a_known_order():
    levels = (10, 20, 40)
    errors = [0.9 * (level / 10.0) ** -2.0 for level in levels]
    fit = reference.convergence_analysis(levels, errors)
    assert fit["observed_order"] == pytest_approx(2.0, 1e-9)
    assert fit["max_fit_residual_relative"] < 1e-9
    assert fit["divisions_for_tolerance"] > levels[-1]


def test_committed_evidence_matches_the_current_recovery_code():
    if not EVIDENCE.exists():
        import pytest

        pytest.skip("audit evidence has not been generated in this checkout")
    evidence = json.loads(EVIDENCE.read_text())
    live = reference.cross_check_pointwise_recovery(cook_fields(8))
    assert evidence["recovery_cross_check"]["passed"] == live["passed"] is True
    assert evidence["status"] == "measured-not-qualified"
    assert evidence["comparison"]["qualified"] is False
    assert evidence["element_centre_accuracy"][-1]["passed"] is True
    assert evidence["loaded_edge_agreement"]["passed"] is True


def pytest_approx(value, tolerance):
    import pytest

    return pytest.approx(value, abs=tolerance)
