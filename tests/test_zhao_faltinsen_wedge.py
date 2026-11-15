"""Tests for Zhao & Faltinsen (1993) wedge water-entry module."""

import math
import pytest
import numpy as np
import torch

from tensorfem.zhao_faltinsen_wedge import (
    WedgeEntryParams,
    wetted_half_width,
    pressure_coefficient_distribution,
    pressure_distribution,
    peak_pressure_coefficient,
    pressure_time_history,
    structural_pressure_load,
    zhao_faltinsen_similarity_parameter,
    validate_wedge_geometry,
)


class TestValidateWedgeGeometry:
    def test_valid_angle(self):
        result = validate_wedge_geometry(30.0)
        assert result["deadrise_deg"] == 30.0
        assert abs(result["deadrise_rad"] - math.radians(30)) < 1e-12
        assert abs(result["tan_beta"] - math.tan(math.radians(30))) < 1e-12

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_wedge_geometry(0.0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_wedge_geometry(-10.0)

    def test_invalid_90(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_wedge_geometry(90.0)

    def test_flag_very_shallow(self):
        result = validate_wedge_geometry(3.0)
        assert result["flag_very_shallow"] is True

    def test_flag_very_steep(self):
        result = validate_wedge_geometry(65.0)
        assert result["flag_very_steep"] is True


class TestSimilarityParameter:
    def test_wagner_limit_small_angle(self):
        """For very small β, c̃ → π/2 (Wagner limit)."""
        c_tilde = zhao_faltinsen_similarity_parameter(math.radians(2))
        assert abs(c_tilde - math.pi / 2) < 0.01

    def test_beta_30_deg(self):
        """For β = 30°, c̃ ≈ 1.372 from the paper's Table 1."""
        c_tilde = zhao_faltinsen_similarity_parameter(math.radians(30))
        assert abs(c_tilde - 1.372) < 0.01

    def test_beta_45_deg(self):
        """For β = 45°, c̃ ≈ 1.188 from the paper's Table 1."""
        c_tilde = zhao_faltinsen_similarity_parameter(math.radians(45))
        assert abs(c_tilde - 1.188) < 0.02

    def test_monotonic_decrease(self):
        """c̃ should decrease monotonically with β."""
        betas = [math.radians(a) for a in [5, 15, 25, 35, 45, 55, 65]]
        c_tildes = [zhao_faltinsen_similarity_parameter(b) for b in betas]
        for i in range(len(c_tildes) - 1):
            assert c_tildes[i] > c_tildes[i + 1]

    def test_invalid_angle(self):
        with pytest.raises(ValueError):
            zhao_faltinsen_similarity_parameter(0.0)
        with pytest.raises(ValueError):
            zhao_faltinsen_similarity_parameter(math.pi / 2)


class TestWettedHalfWidth:
    def test_linear_growth(self):
        """c(t) should grow linearly with time."""
        V = 2.0
        beta = math.radians(30)
        t1, t2 = 0.01, 0.02
        c1 = wetted_half_width(V, beta, t1)
        c2 = wetted_half_width(V, beta, t2)
        assert abs(c2 / c1 - t2 / t1) < 1e-10

    def test_proportional_to_velocity(self):
        """c(t) should be proportional to V."""
        beta = math.radians(30)
        t = 0.01
        c1 = wetted_half_width(1.0, beta, t)
        c2 = wetted_half_width(2.0, beta, t)
        assert abs(c2 / c1 - 2.0) < 1e-10

    def test_numpy_array(self):
        """Should work with numpy arrays."""
        V = 2.0
        beta = math.radians(30)
        t = np.array([0.01, 0.02, 0.03])
        c = wetted_half_width(V, beta, t)
        assert c.shape == t.shape
        assert c[1] / c[0] == pytest.approx(2.0)

    def test_torch_tensor(self):
        """Should work with torch tensors."""
        V = 2.0
        beta = math.radians(30)
        t = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        c = wetted_half_width(V, beta, t)
        assert c.shape == t.shape

    def test_invalid_velocity(self):
        with pytest.raises(ValueError):
            wetted_half_width(0.0, math.radians(30), 0.01)


class TestPressureCoefficient:
    def test_apex_value(self):
        """At η = 0, Cp should be finite and positive."""
        beta = math.radians(30)
        cp = pressure_coefficient_distribution(beta, 0.0)
        assert cp > 0
        assert math.isfinite(cp)

    def test_increases_toward_spray_root(self):
        """Cp should increase as η → 1."""
        beta = math.radians(30)
        cp1 = pressure_coefficient_distribution(beta, 0.5)
        cp2 = pressure_coefficient_distribution(beta, 0.9)
        assert cp2 > cp1

    def test_peak_for_30_deg(self):
        """For β = 30°, peak Cp should be near 4.5 (from the paper)."""
        beta = math.radians(30)
        cp_peak = peak_pressure_coefficient(beta)
        # The paper reports ~4.5; our approximation should be within 15%
        assert abs(cp_peak - 4.5) / 4.5 < 0.15

    def test_numpy_array(self):
        """Should work with numpy arrays."""
        beta = math.radians(30)
        eta = np.linspace(0, 0.99, 50)
        cp = pressure_coefficient_distribution(beta, eta)
        assert cp.shape == eta.shape
        assert np.all(cp > 0)

    def test_torch_tensor(self):
        """Should work with torch tensors."""
        beta = math.radians(30)
        eta = torch.linspace(0, 0.99, 50, dtype=torch.float64)
        cp = pressure_coefficient_distribution(beta, eta)
        assert cp.shape == eta.shape
        assert torch.all(cp > 0)


class TestPressureDistribution:
    def test_dimensional_scaling(self):
        """p = 0.5 ρ V² Cp should scale correctly."""
        V = 2.0
        rho = 1000.0
        beta = math.radians(30)
        eta = 0.5
        cp = pressure_coefficient_distribution(beta, eta)
        p = pressure_distribution(V, rho, beta, eta)
        expected = 0.5 * rho * V**2 * cp
        assert abs(p - expected) < 1e-6

    def test_quadratic_in_velocity(self):
        """Pressure should scale as V²."""
        rho = 1000.0
        beta = math.radians(30)
        eta = 0.5
        p1 = pressure_distribution(1.0, rho, beta, eta)
        p2 = pressure_distribution(2.0, rho, beta, eta)
        assert abs(p2 / p1 - 4.0) < 1e-10


class TestPressureTimeHistory:
    def test_zero_before_wetting(self):
        """Pressure should be zero before the point is wetted."""
        V = 2.0
        rho = 1000.0
        beta = math.radians(30)
        y_point = 0.05
        time = np.array([0.001, 0.005])
        p = pressure_time_history(V, rho, beta, y_point, time)
        assert np.all(p == 0.0)

    def test_positive_after_wetting(self):
        """Pressure should be positive after the point is wetted."""
        V = 2.0
        rho = 1000.0
        beta = math.radians(30)
        y_point = 0.01
        time = np.array([0.1, 0.2, 0.3])
        p = pressure_time_history(V, rho, beta, y_point, time)
        assert np.all(p > 0)

    def test_peak_then_decay(self):
        """Pressure should peak shortly after wetting then decay."""
        V = 2.0
        rho = 1000.0
        beta = math.radians(30)
        y_point = 0.02
        time = np.linspace(0.001, 0.1, 100)
        p = pressure_time_history(V, rho, beta, y_point, time)
        # Find the peak
        peak_idx = np.argmax(p)
        # After the peak, pressure should generally decrease
        if peak_idx < len(p) - 10:
            post_peak = p[peak_idx + 10:]
            assert np.mean(post_peak) < p[peak_idx]


class TestStructuralPressureLoad:
    def test_valid_params(self):
        params = WedgeEntryParams(
            V=2.0, rho=1000.0, beta_deg=30.0,
            wedge_half_width=0.1, wedge_thickness=0.005,
            young_modulus=2.1e11, poisson_ratio=0.3,
            material_density=7850.0,
        )
        params.validate()

    def test_invalid_poisson(self):
        with pytest.raises(ValueError):
            WedgeEntryParams(
                V=2.0, rho=1000.0, beta_deg=30.0,
                wedge_half_width=0.1, wedge_thickness=0.005,
                young_modulus=2.1e11, poisson_ratio=0.6,
                material_density=7850.0,
            ).validate()

    def test_dynamic_pressure(self):
        params = WedgeEntryParams(
            V=2.0, rho=1000.0, beta_deg=30.0,
            wedge_half_width=0.1, wedge_thickness=0.005,
            young_modulus=2.1e11, poisson_ratio=0.3,
            material_density=7850.0,
        )
        assert params.dynamic_pressure == pytest.approx(0.5 * 1000 * 2**2)

    def test_load_at_time_zero(self):
        params = WedgeEntryParams(
            V=2.0, rho=1000.0, beta_deg=30.0,
            wedge_half_width=0.1, wedge_thickness=0.005,
            young_modulus=2.1e11, poisson_ratio=0.3,
            material_density=7850.0,
        )
        node_pos = np.array([0.01, 0.02, 0.05])
        p = structural_pressure_load(params, node_pos, 0.0)
        assert np.all(p == 0.0)

    def test_load_after_wetting(self):
        params = WedgeEntryParams(
            V=2.0, rho=1000.0, beta_deg=30.0,
            wedge_half_width=0.1, wedge_thickness=0.005,
            young_modulus=2.1e11, poisson_ratio=0.3,
            material_density=7850.0,
        )
        node_pos = np.array([0.001, 0.005, 0.01])
        p = structural_pressure_load(params, node_pos, 0.01)
        assert np.all(p >= 0)


class TestWedgeEntryParams:
    def test_beta_rad(self):
        params = WedgeEntryParams(
            V=2.0, rho=1000.0, beta_deg=30.0,
            wedge_half_width=0.1, wedge_thickness=0.005,
            young_modulus=2.1e11, poisson_ratio=0.3,
            material_density=7850.0,
        )
        assert abs(params.beta_rad - math.radians(30)) < 1e-12

    def test_validation_catches_negative_velocity(self):
        with pytest.raises(ValueError):
            WedgeEntryParams(
                V=-1.0, rho=1000.0, beta_deg=30.0,
                wedge_half_width=0.1, wedge_thickness=0.005,
                young_modulus=2.1e11, poisson_ratio=0.3,
                material_density=7850.0,
            ).validate()


class TestRegressionZhaoFaltinsen1993:
    """Regression tests against known values from the paper."""

    def test_cp_peak_30_deg_within_15pct(self):
        """Peak Cp for β=30° must be within 15% of the paper's ~4.5."""
        beta = math.radians(30)
        cp_peak = peak_pressure_coefficient(beta)
        reference = 4.5
        error_pct = abs(cp_peak - reference) / reference * 100
        assert error_pct < 15.0, f"Cp error {error_pct:.1f}% exceeds 15% tolerance"

    def test_cp_peak_10_deg(self):
        """For β=10°, peak Cp should be higher (sharper wedge = higher pressure)."""
        cp_10 = peak_pressure_coefficient(math.radians(10))
        cp_30 = peak_pressure_coefficient(math.radians(30))
        assert cp_10 > cp_30, "Sharper wedge should have higher peak pressure"

    def test_cp_peak_60_deg(self):
        """For β=60°, peak Cp should be lower (blunter wedge = lower pressure)."""
        cp_60 = peak_pressure_coefficient(math.radians(60))
        cp_30 = peak_pressure_coefficient(math.radians(30))
        assert cp_60 < cp_30, "Blunter wedge should have lower peak pressure"
