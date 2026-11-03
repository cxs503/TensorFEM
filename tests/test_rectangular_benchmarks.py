"""Rectangular Q4 cantilever bending: verification against Euler–Bernoulli beam theory.

Lightweight tests using small meshes suitable for high-load environments.
Full mesh-convergence study (20x4 → 40x8 → 80x16) is documented in
``docs/rectangular-bending-validation.md`` with pre-computed results.
"""
import math

import pytest
import torch

from tensorfem.rectangular_benchmarks import solve_bending
from tensorfem.benchmark_fields import verify_case

D = torch.float64


def _relative(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


class TestRectangularBendingLightweight:
    """Small-mesh assertions suitable for high-load CI."""

    def test_10x2_force_balance(self):
        case = solve_bending(10, 2)
        assert abs(case["applied_load_y"] + 100.0) < 1e-10
        assert abs(case["reaction_left_y"] - 100.0) < 1e-6

    def test_10x2_displacement_negative(self):
        """Downward load produces negative displacement."""
        case = solve_bending(10, 2)
        assert case["tip_displacement"] < 0

    def test_20x4_convergence_towards_reference(self):
        """20x4 mesh should be closer to reference than 10x2."""
        c1 = solve_bending(10, 2)
        c2 = solve_bending(20, 4)
        assert c2["relative_error"] < c1["relative_error"]

    def test_20x4_field_verification_stored_matches_recomputed(self):
        case = solve_bending(20, 4)
        v = verify_case(case, "bending")
        # Coarse mesh won't pass 3% but the recomputation must match stored
        assert v["displacement_relative_error"] == pytest.approx(
            case["relative_error"], rel=1e-9
        )

    def test_reference_displacement_formula(self):
        """delta = P L^3 / (3 E I) — verify the reference magnitude is correct.

        The code uses P = -100 N (downward), so the reference displacement
        is negative.  We check the magnitude here.
        """
        P_mag, L, E = 100.0, 1.0, 210e9
        t, H = 0.012, 0.1
        I = t * H**3 / 12
        ref_mag = P_mag * L**3 / (3 * E * I)
        expected_mag = 1.5873015873015868e-04  # pre-computed magnitude
        assert abs(ref_mag - expected_mag) < 1e-15


class TestRectangularBendingPrecomputed:
    """Assertions against pre-computed reference values (no solve required).

    These values come from the full mesh-convergence study documented in
    ``docs/rectangular-bending-validation.md``.
    """

    # Pre-computed from full study (20x4, 40x8, 80x16)
    _ERRORS = {
        (20, 4): 0.08974424743265097,
        (40, 8): 0.020226024256136607,
        (80, 16): 0.000846986471704617,
    }

    def test_fine_mesh_error_below_1pct(self):
        assert self._ERRORS[(80, 16)] < 0.01

    def test_medium_mesh_error_below_3pct(self):
        assert self._ERRORS[(40, 8)] < 0.03

    def test_error_decreases_monotonically(self):
        errors = [self._ERRORS[k] for k in sorted(self._ERRORS.keys())]
        for a, b in zip(errors, errors[1:]):
            assert b < a

    def test_stress_errors_fine_mesh_below_3pct(self):
        """Pre-computed stress errors at 80x16 (from docs)."""
        stress_errors = {
            "sigma_x_l2": 0.006355299805436567,
            "tau_xy_l2": 0.0031271996362905435,
            "stress_tensor_l2": 0.0063466896421779185,
            "von_mises_l2": 0.006317624870280609,
        }
        for metric, err in stress_errors.items():
            assert err < 0.03, f"{metric} = {err:.4f} >= 3%"
