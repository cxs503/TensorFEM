"""Tests for combined in-plane loading of simply supported panels.

The oracle chain is: exact Navier (closed form) -> separable finite difference
of :mod:`tensorfem.marine_plate_buckling` -> the non-separable Kronecker
eigensolver of this module -> literature shear coefficient.
"""
import math

import pytest
import torch

from tensorfem.marine_plate_buckling import (
    finite_difference_uniaxial_buckling,
    navier_uniaxial_buckling_load,
)
from tensorfem.marine_plate_combined_buckling import (
    BiaxialBucklingResult,
    classical_shear_buckling_coefficient,
    navier_biaxial_buckling_load,
    panel_buckling_error,
    panel_buckling_load,
    run_combined_panel_buckling_qualification,
)

E, NU, T = 210.0e9, 0.3, 0.012
D = E * T**3 / (12.0 * (1.0 - NU**2))


def sign_changes(values):
    """Number of sign changes along a 1-D profile (zero crossings excluded)."""
    signs = [1 if v > 0 else -1 for v in values if abs(v) > 1e-9 * max(abs(x) for x in values)]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


class TestNavierBiaxial:
    def test_equal_biaxial_square_is_two_pi_squared_d(self):
        """Nx = Ny on a square plate: Nx = 2 pi^2 D / b^2 exactly."""
        result = navier_biaxial_buckling_load(
            length=1.0, width=1.0, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, biaxial_ratio=1.0)
        assert isinstance(result, BiaxialBucklingResult)
        assert result.critical_line_load == pytest.approx(2.0 * math.pi**2 * D, rel=1e-14)
        assert (result.longitudinal_half_waves, result.transverse_half_waves) == (1, 1)

    def test_zero_ratio_reproduces_uniaxial_oracle(self):
        """ratio = 0 must be the existing uniaxial oracle, bit for bit."""
        for length, waves in ((1.0, 1), (2.0, 2), (3.0, 3)):
            uniaxial, m, n = navier_uniaxial_buckling_load(
                length=length, width=1.0, rigidity_x=D, rigidity_y=D, coupling_rigidity=D)
            biaxial = navier_biaxial_buckling_load(
                length=length, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, biaxial_ratio=0.0)
            assert biaxial.critical_line_load == uniaxial
            assert biaxial.longitudinal_half_waves == m

    def test_biaxial_compression_lowers_the_longitudinal_load(self):
        grid = [
            navier_biaxial_buckling_load(
                length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, biaxial_ratio=r).critical_line_load
            for r in (0.0, 0.25, 0.5, 1.0)
        ]
        assert grid[0] > grid[1] > grid[2] > grid[3]

    def test_transverse_tension_removes_modes_but_keeps_a_load(self):
        """A negative ratio is admissible; modes it makes neutrally stable drop out."""
        compressed = navier_biaxial_buckling_load(
            length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, biaxial_ratio=0.25)
        tensioned = navier_biaxial_buckling_load(
            length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, biaxial_ratio=-0.25)
        assert tensioned.critical_line_load > compressed.critical_line_load

    def test_ratio_that_eliminates_every_mode_raises(self):
        with pytest.raises(ValueError, match="no compressive mode"):
            navier_biaxial_buckling_load(
                length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, biaxial_ratio=-100.0, maximum_half_waves=8)

    def test_small_wave_limit_is_rejected(self):
        with pytest.raises(ValueError, match="maximum_half_waves"):
            navier_biaxial_buckling_load(
                length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, biaxial_ratio=0.0, maximum_half_waves=1)

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"length": 0.0}, ValueError),
            ({"length": True}, TypeError),
            ({"poisson_like": float("nan")}, TypeError),
        ],
    )
    def test_input_validation(self, kwargs, error):
        base = dict(length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
                    coupling_rigidity=D, biaxial_ratio=0.0)
        base.update(kwargs)
        bad = {"width": 0.0} if "poisson_like" in base else base
        with pytest.raises(error):
            navier_biaxial_buckling_load(**bad)


class TestAssembledSolver:
    def test_uniaxial_matches_separable_solver(self):
        """The Kronecker assembly must reproduce the closed-form sin^2 eigenvalues."""
        for grid in (8, 16, 32):
            separable = finite_difference_uniaxial_buckling(
                length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=grid, grid_y=grid)
            assembled = panel_buckling_load(
                length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=grid, grid_y=grid,
                longitudinal_share=1.0, transverse_share=0.0, shear_share=0.0)
            assert assembled.critical_line_load == pytest.approx(
                separable.critical_line_load, rel=1e-10)

    def test_biaxial_converges_to_navier(self):
        reference = navier_biaxial_buckling_load(
            length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, biaxial_ratio=0.5).critical_line_load
        errors = []
        for grid in (8, 16, 32):
            result = panel_buckling_load(
                length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=grid, grid_y=grid,
                longitudinal_share=1.0, transverse_share=0.5)
            errors.append(abs(result.critical_line_load / reference - 1.0))
        assert errors[-1] < errors[-2] < errors[0]
        assert errors[-1] < 3.0e-3

    def test_biaxial_mode_numbers_match_navier(self):
        """Sign changes of the discrete mode reproduce the analytic (m, n)."""
        reference = navier_biaxial_buckling_load(
            length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, biaxial_ratio=0.5)
        result = panel_buckling_load(
            length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=24, grid_y=24,
            longitudinal_share=1.0, transverse_share=0.5, keep_mode=True)
        mode = result.mode_shape
        assert mode.shape == (24, 24)
        mid_y = mode[:, 12]
        mid_x = mode[12, :]
        assert sign_changes(mid_y) + 1 == reference.longitudinal_half_waves
        assert sign_changes(mid_x) + 1 == reference.transverse_half_waves

    def test_fundamental_mode_error_equals_closed_form_identity(self):
        """(1,1) mode: discrete/continuous = 4(N+1)^2 sin^2(pi/2(N+1)) / pi^2."""
        for grid in (8, 16, 24):
            reference = navier_biaxial_buckling_load(
                length=1.0, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, biaxial_ratio=1.0).critical_line_load
            measured = panel_buckling_load(
                length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=grid, grid_y=grid,
                longitudinal_share=1.0, transverse_share=1.0).critical_line_load
            factor = 4.0 * (grid + 1) ** 2 * math.sin(math.pi / (2 * (grid + 1))) ** 2 / math.pi**2
            assert measured / reference == pytest.approx(factor, rel=1e-9)

    def test_pure_shear_square_matches_literature(self):
        """k_s = 9.34 for a simply supported square plate under pure shear."""
        result = panel_buckling_load(
            length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=32, grid_y=32,
            longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0)
        coefficient = result.critical_line_load / (math.pi**2 * D)
        assert coefficient == pytest.approx(9.34, rel=0.01)
        assert result.critical_stress == pytest.approx(result.critical_line_load / T)

    def test_shear_sign_is_immaterial(self):
        loads = [
            panel_buckling_load(
                length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=16, grid_y=16,
                longitudinal_share=0.0, shear_share=s).critical_line_load
            for s in (1.0, -1.0)
        ]
        assert loads[0] == pytest.approx(loads[1], rel=1e-10)

    def test_pure_shear_mode_has_definite_symmetry(self):
        """The square pure-shear mode must be symmetric or antisymmetric in x<->y."""
        mode = panel_buckling_load(
            length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=20, grid_y=20,
            longitudinal_share=0.0, shear_share=1.0, keep_mode=True).mode_shape
        scale = mode.abs().max()
        deviations = [((mode.T - mode).abs().max() / scale).item(),
                      ((mode.T + mode).abs().max() / scale).item()]
        assert min(deviations) < 1e-10

    def test_combined_loading_lowers_the_load_below_either_branch(self):
        shear_only = panel_buckling_load(
            length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=16, grid_y=16,
            longitudinal_share=0.0, shear_share=1.0).critical_line_load
        axial_only = panel_buckling_load(
            length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=16, grid_y=16,
            longitudinal_share=1.0).critical_line_load
        share = 0.355
        combined = panel_buckling_load(
            length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=16, grid_y=16,
            longitudinal_share=share, shear_share=share).critical_line_load
        # The share factors scale both components, so compare the *components*.
        axial_ratio = share * combined / axial_only
        shear_ratio = share * combined / shear_only
        # Neither component may reach its own critical value, otherwise the pure
        # mode would have been recovered instead.
        assert 0.0 < axial_ratio < 1.0
        assert 0.0 < shear_ratio < 1.0
        # Interaction sits between the linear (sum = 1) and elliptic
        # (sum of squares = 1) envelopes of the rule formulations.
        assert axial_ratio + shear_ratio > 1.0
        assert axial_ratio**2 + shear_ratio**2 < 1.0

    def test_orthotropic_panel_is_bounded_by_isotropic_extremes(self):
        weak = panel_buckling_load(
            length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D * 0.25,
            coupling_rigidity=D, grid_x=16, grid_y=16, longitudinal_share=1.0)
        plain = panel_buckling_load(
            length=2.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=16, grid_y=16, longitudinal_share=1.0)
        assert weak.critical_line_load < plain.critical_line_load

    def test_tension_only_and_empty_loading_raise(self):
        with pytest.raises(ValueError, match="tension-only"):
            panel_buckling_load(
                length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=8, grid_y=8, longitudinal_share=-1.0)
        with pytest.raises(ValueError, match="at least one load share"):
            panel_buckling_load(
                length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, grid_x=8, grid_y=8,
                longitudinal_share=0.0, transverse_share=0.0, shear_share=0.0)

    @pytest.mark.parametrize(
        "field,value,error",
        [
            ("grid_x", 1, ValueError),
            ("grid_x", 12.0, TypeError),
            ("thickness", 0.0, ValueError),
            ("longitudinal_share", float("inf"), ValueError),
            ("shear_share", True, TypeError),
        ],
    )
    def test_input_validation(self, field, value, error):
        base = dict(length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
                    coupling_rigidity=D, grid_x=8, grid_y=8, longitudinal_share=1.0)
        base[field] = value
        with pytest.raises(error):
            panel_buckling_load(**base)

    def test_reference_attachment_requires_a_reference(self):
        result = panel_buckling_load(
            length=1.0, width=1.0, thickness=T, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=8, grid_y=8, longitudinal_share=1.0)
        with pytest.raises(ValueError, match="no reference load"):
            panel_buckling_error(result)


class TestShearCoefficient:
    def test_square_matches_the_classical_nine_point_three_four(self):
        assert classical_shear_buckling_coefficient(1.0) == pytest.approx(9.34, rel=1e-12)

    def test_asymptote_approaches_five_point_three_four(self):
        assert classical_shear_buckling_coefficient(1.0e6) == pytest.approx(5.34, rel=1e-9)

    def test_monotone_decrease(self):
        values = [classical_shear_buckling_coefficient(r) for r in (1.0, 1.5, 2.0, 3.0, 5.0)]
        assert values == sorted(values, reverse=True)

    @pytest.mark.parametrize("bad,error", [(0.5, ValueError), (0.0, ValueError), ("a", TypeError)])
    def test_validation(self, bad, error):
        with pytest.raises(error):
            classical_shear_buckling_coefficient(bad)


class TestQualification:
    def test_qualification_passes_all_gates(self):
        report = run_combined_panel_buckling_qualification()
        assert report["schema"] == "tensorfem.combined-panel-buckling/1.0"
        assert report["tolerance"] == 0.03
        assert report["passed"] is True
        gates = [case["gate"] for case in report["cases"]]
        assert gates == [
            "equal_biaxial_square_exact",
            "biaxial_reduces_to_uniaxial",
            "assembled_matches_separable_uniaxial",
            "biaxial_ratio_0.5_convergence",
            "pure_shear_square_literature",
            "pure_shear_aspect_two_spectral_reference",
        ]
        assert all(case["passed"] for case in report["cases"])
        discretised = [case for case in report["cases"] if case["gate"].endswith("convergence")]
        assert discretised[0]["errors"][-1] < report["tolerance"]