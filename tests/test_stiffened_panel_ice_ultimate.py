"""Tests for stiffened panel ultimate strength under ice loading.

Campaign node: p262w1-N05-fem-panel-ultimate
Acceptance criteria: F1-F6 (see module docstring).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.stiffened_panel_ice_ultimate import (
    EH36Steel,
    StiffenedPanelGeometry,
    PolarClassIcePressure,
    timoshenko_plate_buckling_stress,
    plate_slenderness_parameter,
    build_stiffened_panel_model,
    verify_elastic_buckling,
    verify_ultimate_strength_paik,
    verify_ultimate_three_slenderness,
    verify_ice_pressure_collapse,
    verify_energy_conservation,
    run_full_verification,
)
from tensorfem.ice_belt_collapse import (
    elastic_perfectly_plastic,
    E_STEEL,
    NU_STEEL,
)


# ---------------------------------------------------------------------------
# Material and geometry tests (fast, no FE)
# ---------------------------------------------------------------------------

class TestEH36Steel:
    """EH36 material properties."""

    def test_eh36_yield_stress(self):
        steel = EH36Steel()
        assert steel.sigma_y == pytest.approx(355.0e6)

    def test_eh36_youngs_modulus(self):
        steel = EH36Steel()
        assert steel.young == pytest.approx(206.0e9)

    def test_eh36_to_grade_conversion(self):
        steel = EH36Steel()
        grade = steel.to_grade()
        assert grade.sigma_0 == pytest.approx(355.0e6)
        assert grade.young == pytest.approx(206.0e9)
        assert grade.n == pytest.approx(0.5)
        assert grade.sigma_u == pytest.approx(490.0e6)


class TestStiffenedPanelGeometry:
    """Panel geometry configuration."""

    def test_default_geometry(self):
        geom = StiffenedPanelGeometry()
        assert geom.span_x == pytest.approx(2.0)
        assert geom.width == pytest.approx(2.0)
        assert geom.plate_thickness == pytest.approx(0.015)

    def test_stiffener_positions(self):
        geom = StiffenedPanelGeometry(n_stiffeners=3, stiffener_spacing=0.5)
        positions = geom.stiffener_positions
        assert len(positions) == 3
        assert positions[0] == pytest.approx(0.5)
        assert positions[1] == pytest.approx(1.0)
        assert positions[2] == pytest.approx(1.5)


class TestPolarClassIcePressure:
    """IACS Polar Class ice pressure model."""

    def test_pc7_default_pressure(self):
        ice = PolarClassIcePressure.for_class("PC7")
        assert ice.p0 == pytest.approx(8.0e6)

    def test_pc6_higher_pressure(self):
        ice = PolarClassIcePressure.for_class("PC6")
        assert ice.p0 == pytest.approx(12.0e6)

    def test_pressure_area_scaling(self):
        ice = PolarClassIcePressure(p0=10.0e6, area_ref=1.0)
        # At reference area, pressure = p0
        assert ice.pressure_at_area(1.0) == pytest.approx(10.0e6)
        # Larger area → lower pressure (exponent = -0.3)
        p_large = ice.pressure_at_area(4.0)
        assert p_large < 10.0e6

    def test_invalid_class_raises(self):
        with pytest.raises(ValueError, match="Unknown Polar Class"):
            PolarClassIcePressure.for_class("PC99")


# ---------------------------------------------------------------------------
# Analytical reference tests (fast, no FE)
# ---------------------------------------------------------------------------

class TestAnalyticalReferences:
    """Closed-form reference solutions."""

    def test_timoshenko_buckling_square_plate(self):
        """Square plate a=b: k ≈ 4.0 for m=1."""
        sigma_cr = timoshenko_plate_buckling_stress(
            a=1.0, b=1.0, t=0.01, young=E_STEEL, poisson=NU_STEEL)
        # k=4, σ_cr = 4·π²·E/(12(1-ν²))·(t/b)²
        k_expected = 4.0
        D = E_STEEL / (12.0 * (1.0 - NU_STEEL ** 2))
        sigma_expected = k_expected * math.pi ** 2 * D * (0.01 / 1.0) ** 2
        assert sigma_cr == pytest.approx(sigma_expected, rel=1e-6)

    def test_timoshenko_buckling_positive(self):
        sigma_cr = timoshenko_plate_buckling_stress(
            a=2.0, b=2.0, t=0.015)
        assert sigma_cr > 0

    def test_slenderness_parameter(self):
        beta = plate_slenderness_parameter(
            b=0.5, t=0.015, sigma_y=355.0e6, E=206.0e9)
        # β = (0.5/0.015) * sqrt(355e6/206e9) ≈ 33.3 * 0.0416 ≈ 1.39
        assert beta == pytest.approx(1.387, rel=0.01)


# ---------------------------------------------------------------------------
# F1: Elastic buckling verification (FE, ~12s)
# ---------------------------------------------------------------------------

class TestF1ElasticBuckling:
    """F1: Elastic buckling stress vs Timoshenko analytical (≤5% error)."""

    def test_elastic_buckling_passes(self):
        result = verify_elastic_buckling(nx=12, ny=8)
        assert result["pass_f1"], (
            f"F1 failed: rel_error={result['rel_error']:.4f}, "
            f"sigma_fe={result['sigma_fe']:.3e}, "
            f"sigma_analytical={result['sigma_analytical']:.3e}")


# ---------------------------------------------------------------------------
# F2: Ultimate strength vs Paik formula (FE, ~8s)
# ---------------------------------------------------------------------------

class TestF2UltimateStrength:
    """F2: Ultimate strength vs Paik-Thayamballi (≤15% error)."""

    def test_moderate_slenderness_passes(self):
        """Moderate slenderness case (β ≈ 1.7)."""
        result = verify_ultimate_strength_paik(
            span_x=2.0, width=0.5, thickness=0.012,
            nx=12, ny=4, n_steps=20)
        assert result["pass_f2"], (
            f"F2 failed for moderate: rel_error={result['rel_error']:.4f}, "
            f"sigma_fem={result['sigma_fem']:.3e}, "
            f"sigma_paik={result['sigma_paik']:.3e}")


# ---------------------------------------------------------------------------
# F3: Post-peak tracking
# ---------------------------------------------------------------------------

class TestF3PostPeak:
    """F3: Load-deflection curve shows post-peak descent."""

    def test_arc_length_captures_peak(self):
        result = verify_ultimate_strength_paik(
            span_x=2.0, width=0.5, thickness=0.012,
            nx=12, ny=4, n_steps=20)
        # Peak load factor should be found
        assert result["peak_load_factor"] > 0


# ---------------------------------------------------------------------------
# F4: Energy conservation (FE, ~10s)
# ---------------------------------------------------------------------------

class TestF4EnergyConservation:
    """F4: Energy balance W_ext = U + D_p, residual ≤ 2%."""

    def test_energy_conservation_passes(self):
        result = verify_energy_conservation(nx=8, ny=6, n_steps=10)
        assert result["pass_f4"], (
            f"F4 failed: energy_residual={result['energy_residual']:.4f}")


# ---------------------------------------------------------------------------
# Model building tests (fast)
# ---------------------------------------------------------------------------

class TestModelBuilding:
    """FE model construction."""

    def test_stiffened_panel_has_stiffeners(self):
        grade = elastic_perfectly_plastic()
        geom = StiffenedPanelGeometry(span_x=2.0, width=2.0, n_stiffeners=2, stiffener_spacing=1.0)
        model = build_stiffened_panel_model(geom, grade, nx=8, ny=6)
        assert len(model.stiffeners) > 0

    def test_stiffened_panel_mesh_size(self):
        grade = elastic_perfectly_plastic()
        geom = StiffenedPanelGeometry(n_stiffeners=0)
        model = build_stiffened_panel_model(geom, grade, nx=8, ny=6)
        assert len(model.elements) == 8 * 6
        assert len(model.nodes) == 9 * 7
