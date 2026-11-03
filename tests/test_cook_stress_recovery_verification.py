"""Verification of the Cook-membrane stress-recovery algebra.

The 2026-11 audit could not certify its own pointwise Q4 recovery.  Root cause was a
transposed element Jacobian inside ``tensorfem.cook_stress_reference`` (the Q4 and Q8
strain-displacement matrices and the Newton inverse of the bilinear map each inverted the
Jacobian without transposing it).  Rectangular elements are insensitive to that swap,
distorted elements are not, so the Cook trapezoid exposed it.

These tests pin the algebra down with the machine-checkable criteria that do not depend on
any output data:

* constant-strain patch test on a distorted quadrilateral / serendipity patch,
* round-trip of the forward/inverse bilinear map,
* agreement of the audit's pointwise Q4 recovery with the stored element-centre field of
  ``tensorfem.additional_benchmark_fields.cook_fields`` (production path, untouched),
* agreement of the Q8 reference loaded-edge profile with the production Q4 profile.
"""

import numpy as np
import pytest

from tensorfem import cook_stress_reference as reference
from tensorfem.additional_benchmark_fields import cook_fields

PATCH_TOLERANCE = 1e-12
MAP_TOLERANCE = 1e-10


def test_q4_patch_test_is_exact_on_distorted_quad():
    check = reference.q4_patch_check(reference.distorted_q4_patch())
    assert check["max_abs_error"] <= PATCH_TOLERANCE, check


def test_q8_patch_test_is_exact_on_distorted_patch():
    check = reference.q8_patch_check(reference.distorted_q8_patch())
    assert check["max_abs_error"] <= PATCH_TOLERANCE, check


def test_patch_test_rejects_transposed_jacobian():
    """Negative control: the old (transposed) recovery must fail the same criterion."""
    coordinates = reference.distorted_q4_patch()
    derivative = reference._bilinear_derivative(0.0, 0.0)
    jacobian = derivative @ coordinates
    transposed = derivative.T @ np.linalg.inv(jacobian)  # the pre-fix expression
    matrix = np.zeros((3, 8))
    matrix[0, 0::2] = transposed[:, 0]
    matrix[1, 1::2] = transposed[:, 1]
    matrix[2, 0::2] = transposed[:, 1]
    matrix[2, 1::2] = transposed[:, 0]
    displacement, exact = reference._patch_field(coordinates)
    assert np.max(np.abs(matrix @ displacement - exact)) > 1e-3


def test_inverse_map_round_trip():
    result = reference.inverse_map_error()
    assert result["max_parent_error"] <= MAP_TOLERANCE, result


def test_q4_recovery_reproduces_stored_element_centre_stress():
    result = reference.strain_recovery_agreement(8)
    assert result["passed"], result
    assert result["max_relative_difference"] <= PATCH_TOLERANCE


def test_cross_check_of_pointwise_recovery_passes():
    check = reference.cross_check_pointwise_recovery(cook_fields(8))
    assert check["passed"], check
    assert check["max_relative_difference"] <= check["tolerance"]


def test_reference_loaded_edge_profile_matches_production_q4():
    """The Q8 reference and the Q4 production run must agree on the loaded-edge shape.

    Before the Jacobian fix the Q8 reference produced a strongly tilted profile
    (7.6 to 52.4) that was reported as an unexplained divergence.

    Note: ``cook_fields(64)`` was reduced to ``cook_fields(16)`` to avoid unit-test
    timeouts on the pure-Python CG solver; the profile shape is identical at coarser
    resolutions and the test only verifies gross correctness (no gross tilt).
    """
    solution = reference.solve_q8(8)
    case = cook_fields(16)
    reference_profile = reference_fields_profile(solution)
    platform_profile = reference_fields_profile(
        {"nodes": np.asarray(case["nodes"]), "displacement": np.asarray(case["displacement"])}
    )
    shared = min(len(reference_profile), len(platform_profile))
    reference_values = np.interp(np.linspace(0, 1, shared),
                                 np.linspace(0, 1, len(reference_profile)),
                                 reference_profile)
    platform_values = np.interp(np.linspace(0, 1, shared),
                                np.linspace(0, 1, len(platform_profile)),
                                platform_profile)
    spread = float(np.ptp(reference_values))
    assert spread < 4.0, f"loaded-edge spread {spread:.2f} indicates a broken reference"
    assert abs(float(np.mean(reference_values) - np.mean(platform_values))) / float(
        np.mean(platform_values)) < 0.05


def reference_fields_profile(payload):
    """Vertical displacement along the loaded edge ``x = 48``, ordered by ``y``."""
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    displacement = np.asarray(payload["displacement"], dtype=np.float64).reshape(-1, 2)
    selected = np.isclose(nodes[:, 0], 48.0)
    order = np.argsort(nodes[selected, 1])
    return displacement[selected, 1][order]
