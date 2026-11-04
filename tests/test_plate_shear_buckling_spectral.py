"""Tests for the double-sine Galerkin reference of ``plate_shear_buckling_spectral``.

The reference only earns the right to judge the finite-difference solver if it is
first pinned to the cases where a closed form exists.  That ordering is the point
of this file: exactness first (Navier), then the classical literature anchor, then
the aspect ratio the ship-panel module previously left unqualified.
"""

from __future__ import annotations

import math

import pytest

from tensorfem.marine_plate_combined_buckling import (
    classical_shear_buckling_coefficient,
    navier_biaxial_buckling_load,
    panel_buckling_load,
)
from tensorfem.plate_shear_buckling_spectral import (
    SQUARE_SHEAR_LITERATURE_COEFFICIENT,
    classical_asymptotic_coefficient,
    converged_shear_buckling_coefficient,
    shear_buckling_coefficient,
    spectral_buckling_load,
    twist_matrix_quadrature_check,
)

D = 1.0


def _load(**kwargs):
    base = dict(length=1.0, width=1.0, rigidity_x=D, rigidity_y=D,
                coupling_rigidity=D, modes_x=20, modes_y=20)
    base.update(kwargs)
    return spectral_buckling_load(**base)


class TestTwistBlock:
    def test_closed_form_matches_quadrature(self):
        """The parity factor is 0/2, not a 0/1 indicator (a 4x trap)."""
        assert twist_matrix_quadrature_check(4) < 1e-10

    def test_parity_factor_is_two(self):
        from tensorfem.plate_shear_buckling_spectral import _parity

        assert _parity(1, 2) == 2.0
        assert _parity(2, 4) == 0.0
        assert _parity(3, 3) == 0.0


class TestExactLimits:
    """Cases with a closed-form coefficient: the reference must be exact there."""

    def test_uniaxial_square_is_navier(self):
        result = _load(longitudinal_share=1.0)
        assert result.critical_line_load / (4.0 * math.pi**2 * D) == pytest.approx(
            1.0, rel=1e-10)

    def test_equal_biaxial_square_is_navier(self):
        result = _load(longitudinal_share=1.0, transverse_share=1.0)
        assert result.critical_line_load / (2.0 * math.pi**2 * D) == pytest.approx(
            1.0, rel=1e-10)

    def test_biaxial_ratio_matches_module_navier(self):
        reference = navier_biaxial_buckling_load(
            length=2.0, width=1.0, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, biaxial_ratio=0.5).critical_line_load
        result = _load(length=2.0, longitudinal_share=1.0, transverse_share=0.5)
        assert result.critical_line_load == pytest.approx(reference, rel=1e-8)

    def test_non_square_uniaxial_is_navier(self):
        # a/b = 2 uniaxial: the minimum over m of (m/2 + 2/m)^2 pi^2 D / b^2 is
        # attained at m = 2, k = 4; at m = 1 it is 6.25.
        result = _load(length=2.0, longitudinal_share=1.0)
        assert result.critical_line_load / (4.0 * math.pi**2 * D) == pytest.approx(
            1.0, rel=1e-8)


class TestPureShearReference:
    def test_square_matches_literature_anchor(self):
        coefficient = shear_buckling_coefficient(
            1.0, modes_x=24, modes_y=24)
        assert abs(coefficient / SQUARE_SHEAR_LITERATURE_COEFFICIENT - 1.0) < 0.003

    def test_truncation_is_a_decreasing_upper_bound(self):
        values = [shear_buckling_coefficient(2.0, modes_x=m, modes_y=m)
                  for m in (8, 12, 16, 20, 24)]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))
        assert values[-1] < values[0]

    def test_aspect_ratio_two_value(self):
        data = converged_shear_buckling_coefficient(
            2.0, budgets=(16, 20, 24, 28))
        assert data["converged"] is True
        assert data["residual"] < 1e-4
        assert data["coefficient"] == pytest.approx(6.5460, abs=2e-3)

    def test_long_panel_approaches_classical_asymptote(self):
        coefficient = converged_shear_buckling_coefficient(
            8.0, budgets=(16, 20, 24))["coefficient"]
        assert 5.34 <= coefficient <= 5.44
        assert coefficient < converged_shear_buckling_coefficient(
            2.0, budgets=(16, 20, 24))["coefficient"]

    def test_shear_sign_symmetry(self):
        plus = _load(longitudinal_share=0.0, shear_share=1.0)
        minus = _load(longitudinal_share=0.0, shear_share=-1.0)
        assert minus.critical_line_load == pytest.approx(
            plus.critical_line_load, rel=1e-9)

    def test_shear_sits_between_the_long_panel_limits(self):
        """Sanity in the long-panel regime: k_x -> 4 while k_s -> 5.34, so k_s > k_x.

        At a/b = 4 the uniaxial coefficient bottoms out at 4 (m = 4 half waves) and
        the converged shear coefficient is 5.62; shear is the *less* critical load
        here, which is the opposite of the square panel.
        """
        shear = converged_shear_buckling_coefficient(
            4.0, budgets=(16, 20, 24))["coefficient"]
        axial = _load(length=4.0, longitudinal_share=1.0).critical_line_load / math.pi**2
        assert axial == pytest.approx(4.0, rel=1e-8)
        assert 5.34 < shear < 6.0
        assert shear > axial


class TestVerdictOnTheFiniteDifferenceSolver:
    """The comparison this module exists to make."""

    def test_fd_solver_agrees_at_aspect_two(self):
        reference = converged_shear_buckling_coefficient(
            2.0, budgets=(16, 20, 24))["coefficient"]
        result = panel_buckling_load(
            length=2.0, width=1.0, thickness=0.012, rigidity_x=D, rigidity_y=D,
            coupling_rigidity=D, grid_x=32, grid_y=64,
            longitudinal_share=0.0, shear_share=1.0)
        coefficient = result.critical_line_load / math.pi**2
        assert abs(coefficient / reference - 1.0) < 0.01

    def test_asymptotic_fit_is_the_worse_reference_at_aspect_two(self):
        """Documents why the old 3.55 % verdict was attributed to the fit."""
        reference = converged_shear_buckling_coefficient(
            2.0, budgets=(16, 20, 24))["coefficient"]
        fit = classical_shear_buckling_coefficient(2.0)
        assert abs(fit / reference - 1.0) > 0.02
        assert abs(fit / reference - 1.0) > abs(
            classical_asymptotic_coefficient(1.0) / 9.3245 - 1.0)

    def test_fd_error_stays_under_three_percent_across_aspects(self):
        pairs = ((1.0, (24, 24)), (2.0, (24, 48)), (3.0, (20, 60)))
        for aspect, (grid_x, grid_y) in pairs:
            reference = converged_shear_buckling_coefficient(
                aspect, budgets=(16, 20, 24))["coefficient"]
            result = panel_buckling_load(
                length=aspect, width=1.0, thickness=0.012, rigidity_x=D,
                rigidity_y=D, coupling_rigidity=D, grid_x=grid_x, grid_y=grid_y,
                longitudinal_share=0.0, shear_share=1.0)
            coefficient = result.critical_line_load / math.pi**2
            assert abs(coefficient / reference - 1.0) < 0.03


class TestValidation:
    def test_aspect_ratio_must_be_positive(self):
        with pytest.raises(ValueError):
            shear_buckling_coefficient(0.0)

    def test_modes_must_be_integer(self):
        with pytest.raises(ValueError):
            _load(modes_x=8.5)

    def test_modes_must_be_at_least_two(self):
        with pytest.raises(ValueError):
            _load(modes_y=1)

    def test_all_zero_shares_rejected(self):
        with pytest.raises(ValueError):
            _load(longitudinal_share=0.0, transverse_share=0.0, shear_share=0.0)

    def test_tension_is_out_of_scope(self):
        with pytest.raises(ValueError):
            _load(longitudinal_share=-1.0)

    def test_single_budget_rejected(self):
        with pytest.raises(ValueError):
            converged_shear_buckling_coefficient(1.0, budgets=(16,))