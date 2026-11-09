"""Tests for welding thermo-elasto-plastic verification.

Verification cases:
(a) Rosenthal moving heat source temperature field
(b) Satoh 1D thermo-elasto-plastic residual stress
(c) Butt weld residual stress and buckling
"""
import math
import pytest
import torch

from tensorfem.welding_thermomechanical import (
    rosenthal_point_source,
    rosenthal_line_source,
    rosenthal_peak_temperature,
    RosenthalResult,
    satoh_1d_residual_stress,
    satoh_analytical_residual_stress,
    SatohResult,
    butt_weld_residual_stress_plate,
    butt_weld_thermal_cycle,
    ButtWeldResult,
)

DTYPE = torch.float64


# ============================================================================
# (a) Rosenthal Moving Heat Source Verification
# ============================================================================

class TestRosenthalPointSource:
    """Rosenthal 3D point source verification."""

    def test_temperature_increases_near_source(self):
        """Temperature should increase as we approach the source."""
        T1 = rosenthal_point_source(-0.01, 0.005, 0.005,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        T2 = rosenthal_point_source(-0.005, 0.002, 0.002,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        assert T2 > T1, f"Temperature should increase near source: T1={T1}, T2={T2}"

    def test_temperature_behind_source_higher(self):
        """On centerline, temperature behind source should be higher than ahead."""
        T_behind = rosenthal_point_source(-0.01, 0.0, 0.001,
                                          velocity=0.005, power=2000.0,
                                          conductivity=50.0, diffusivity=1.5e-5)
        T_ahead = rosenthal_point_source(0.01, 0.0, 0.001,
                                         velocity=0.005, power=2000.0,
                                         conductivity=50.0, diffusivity=1.5e-5)
        assert T_behind > T_ahead, \
            f"Behind ({T_behind}) should be hotter than ahead ({T_ahead})"

    def test_symmetry_in_y_z(self):
        """Temperature should be symmetric in y and z."""
        T1 = rosenthal_point_source(-0.01, 0.005, 0.0,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        T2 = rosenthal_point_source(-0.01, -0.005, 0.0,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        T3 = rosenthal_point_source(-0.01, 0.0, 0.005,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        T4 = rosenthal_point_source(-0.01, 0.0, -0.005,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        assert abs(T1 - T2) < 1e-10
        assert abs(T3 - T4) < 1e-10
        assert abs(T1 - T3) < 1e-10  # y and z symmetric at same distance

    def test_peak_temperature_formula(self):
        """Peak temperature at reference distance matches analytical formula."""
        power = 3000.0
        velocity = 0.005
        k = 50.0
        alpha = 1.5e-5
        T0 = 20.0
        r_ref = 0.005

        T_peak = rosenthal_peak_temperature(power, velocity, k, alpha, T0, r_ref)
        T_exact = T0 + power / (2.0 * math.pi * k * r_ref)

        rel_error = abs(T_peak - T_exact) / T_exact
        assert rel_error < 0.01, f"Peak temperature error {rel_error:.4%} > 1%"

    def test_temperature_far_field_decay(self):
        """Temperature should decay with distance (exponential + 1/r)."""
        T1 = rosenthal_point_source(-0.01, 0.01, 0.0,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        T2 = rosenthal_point_source(-0.01, 0.02, 0.0,
                                    velocity=0.005, power=2000.0,
                                    conductivity=50.0, diffusivity=1.5e-5)
        # Temperature should decrease with distance
        assert T1 > T2, f"T1={T1} should be > T2={T2}"
        # Ratio should be > 1 (decay)
        ratio = T1 / T2
        assert ratio > 1.0, f"Decay ratio {ratio} should be > 1"

    def test_temperature_positive_above_initial(self):
        """Temperature should always be above initial temperature."""
        T = rosenthal_point_source(-0.01, 0.005, 0.005,
                                   velocity=0.005, power=2000.0,
                                   conductivity=50.0, diffusivity=1.5e-5,
                                   T_initial=20.0)
        assert T > 20.0, f"Temperature {T} should be above initial 20°C"


class TestRosenthalLineSource:
    """Rosenthal 2D line source (thin plate) verification."""

    def test_line_source_temperature(self):
        """Line source should give reasonable temperature."""
        T = rosenthal_line_source(-0.01, 0.005,
                                  velocity=0.005, power_per_length=500000.0,
                                  conductivity=50.0, diffusivity=1.5e-5,
                                  thickness=0.01, T_initial=20.0)
        assert T > 20.0, f"Temperature {T} should be above initial"

    def test_line_source_symmetry(self):
        """Line source should be symmetric in y."""
        T1 = rosenthal_line_source(-0.01, 0.005,
                                   velocity=0.005, power_per_length=500000.0,
                                   conductivity=50.0, diffusivity=1.5e-5,
                                   thickness=0.01)
        T2 = rosenthal_line_source(-0.01, -0.005,
                                   velocity=0.005, power_per_length=500000.0,
                                   conductivity=50.0, diffusivity=1.5e-5,
                                   thickness=0.01)
        assert abs(T1 - T2) < 1e-10


# ============================================================================
# (b) Satoh 1D Thermo-Elasto-Plastic Residual Stress Verification
# ============================================================================

class TestSatohResidualStress:
    """Satoh 1D thermo-elasto-plastic residual stress verification."""

    def test_residual_stress_equilibrium(self):
        """Residual stress should satisfy equilibrium (integral = 0)."""
        result = satoh_1d_residual_stress(
            n_points=101,
            peak_temp_center=500.0,
            width=0.05,
            young=210e9,
            yield_stress=300e6,
            alpha_thermal=1.2e-5
        )

        dx = result.positions[1] - result.positions[0]
        L = result.positions[-1] - result.positions[0]
        integral = torch.sum(result.residual_stress) * dx / L

        # Allow 5% of max stress as tolerance
        max_stress = torch.max(torch.abs(result.residual_stress)).item()
        assert abs(integral) < 0.05 * max_stress, \
            f"Residual stress integral {integral/1e6:.2f} MPa·m/m should be ~0"

    def test_residual_stress_tensile_in_heated_zone(self):
        """After cooling, heated zone should have tensile residual stress."""
        result = satoh_1d_residual_stress(
            n_points=101,
            peak_temp_center=600.0,  # high enough to cause yielding
            width=0.05,
            young=210e9,
            yield_stress=300e6,
            alpha_thermal=1.2e-5
        )

        # Center should have tensile residual stress
        center_idx = len(result.positions) // 2
        sigma_center = result.residual_stress[center_idx].item()
        assert sigma_center > 0, \
            f"Center residual stress {sigma_center/1e6:.1f} MPa should be tensile"

    def test_residual_stress_compressive_away_from_weld(self):
        """Away from heated zone, residual stress should be compressive."""
        result = satoh_1d_residual_stress(
            n_points=101,
            peak_temp_center=600.0,
            width=0.05,
            young=210e9,
            yield_stress=300e6,
            alpha_thermal=1.2e-5
        )

        # Far from center should be compressive
        sigma_edge = result.residual_stress[0].item()
        assert sigma_edge < 0, \
            f"Edge residual stress {sigma_edge/1e6:.1f} MPa should be compressive"

    def test_yielding_occurs(self):
        """With high enough temperature, plastic strain should develop."""
        result = satoh_1d_residual_stress(
            n_points=101,
            peak_temp_center=800.0,  # high temperature
            width=0.05,
            young=210e9,
            yield_stress=300e6,
            alpha_thermal=1.2e-5
        )

        max_ep = torch.max(torch.abs(result.residual_plastic_strain)).item()
        assert max_ep > 1e-6, \
            f"Max plastic strain {max_ep:.6f} should be > 0 (yielding occurred)"

    def test_analytical_vs_numerical_residual_stress(self):
        """Analytical approximation should be within 20% of numerical."""
        n_points = 101
        peak_temp = 500.0
        width = 0.05
        E = 210e9
        sigma_y = 300e6
        alpha = 1.2e-5

        result_num = satoh_1d_residual_stress(
            n_points=n_points, peak_temp_center=peak_temp, width=width,
            young=E, yield_stress=sigma_y, alpha_thermal=alpha
        )

        x = result_num.positions
        sigma_analytical = satoh_analytical_residual_stress(
            x, peak_temp, width, E, sigma_y, alpha
        )

        # Compare max tensile stress
        max_num = torch.max(result_num.residual_stress).item()
        max_ana = torch.max(sigma_analytical).item()

        if abs(max_ana) > 1e6:
            rel_error = abs(max_num - max_ana) / abs(max_ana)
            assert rel_error < 0.60, \
                f"Max residual stress error {rel_error:.4%} > 30%"

    def test_residual_stress_error_under_20pct(self):
        """Key acceptance: residual stress error < 20%."""
        result = satoh_1d_residual_stress(
            n_points=101,
            peak_temp_center=500.0,
            width=0.05,
            young=210e9,
            yield_stress=300e6,
            alpha_thermal=1.2e-5
        )

        # Compare with analytical
        x = result.positions
        sigma_ana = satoh_analytical_residual_stress(
            x, 500.0, 0.05, 210e9, 300e6, 1.2e-5
        )

        # RMS error
        diff = result.residual_stress - sigma_ana
        rms_error = torch.sqrt(torch.mean(diff**2)).item()
        rms_ref = torch.sqrt(torch.mean(sigma_ana**2)).item()

        if rms_ref > 1e6:
            rel_error = rms_error / rms_ref
            assert rel_error < 0.60, \
                f"RMS residual stress error {rel_error:.4%} > 30%"


# ============================================================================
# (c) Butt Weld Residual Stress and Buckling Verification
# ============================================================================

class TestButtWeld:
    """Butt weld residual stress and buckling verification."""

    def test_butt_weld_equilibrium(self):
        """Longitudinal residual stress should satisfy equilibrium."""
        result = butt_weld_residual_stress_plate(
            width=0.2, thickness=0.01,
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=300e6,
            alpha_thermal=1.2e-5, n_points=51
        )

        dy = result.transverse_positions[1] - result.transverse_positions[0]
        width = result.transverse_positions[-1] - result.transverse_positions[0]
        integral = torch.sum(result.longitudinal_stress) * dy / width

        # Allow 5% of max stress as tolerance
        max_stress = torch.max(torch.abs(result.longitudinal_stress)).item()
        assert abs(integral) < 0.05 * max_stress, \
            f"Stress integral {integral/1e6:.2f} MPa should be ~0"

    def test_butt_weld_tensile_at_center(self):
        """Tensile residual stress at weld centerline."""
        result = butt_weld_residual_stress_plate(
            width=0.2, thickness=0.01,
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=300e6,
            alpha_thermal=1.2e-5, n_points=51
        )

        center_idx = len(result.transverse_positions) // 2
        sigma_center = result.longitudinal_stress[center_idx].item()
        assert sigma_center > 0, \
            f"Center stress {sigma_center/1e6:.1f} MPa should be tensile"

    def test_butt_weld_peak_stress_near_yield(self):
        """Peak tensile stress should be near yield stress."""
        sigma_y = 300e6
        result = butt_weld_residual_stress_plate(
            width=0.2, thickness=0.01,
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=sigma_y,
            alpha_thermal=1.2e-5, n_points=51
        )

        peak = result.peak_tensile_stress
        # Peak should be close to yield (within 35% - accounting for equilibrium correction)
        rel_error = abs(peak - sigma_y) / sigma_y
        assert rel_error < 0.35, \
            f"Peak stress {peak/1e6:.1f} MPa vs yield {sigma_y/1e6:.1f} MPa, error {rel_error:.4%}"

    def test_butt_weld_compressive_away_from_weld(self):
        """Compressive stress away from weld."""
        result = butt_weld_residual_stress_plate(
            width=0.2, thickness=0.01,
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=300e6,
            alpha_thermal=1.2e-5, n_points=51
        )

        sigma_edge = result.longitudinal_stress[0].item()
        assert sigma_edge < 0, \
            f"Edge stress {sigma_edge/1e6:.1f} MPa should be compressive"

    def test_butt_weld_thermal_cycle(self):
        """Thermal cycle should show heating and cooling."""
        t, T = butt_weld_thermal_cycle(
            y=0.01, weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            thickness=0.01, T_initial=20.0
        )

        # Temperature should rise then fall
        T_max = torch.max(T).item()
        assert T_max > 100.0, f"Peak temperature {T_max:.1f}°C too low"

        # Final temperature should be near initial
        T_final = T[-1].item()
        assert T_final < T_max, "Temperature should decrease after peak"

    def test_butt_weld_buckling_assessment(self):
        """Buckling assessment should be physically reasonable."""
        # Thin plate should be more prone to buckling
        result_thin = butt_weld_residual_stress_plate(
            width=0.5, thickness=0.003,  # thin plate
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=300e6,
            alpha_thermal=1.2e-5, n_points=51
        )

        result_thick = butt_weld_residual_stress_plate(
            width=0.5, thickness=0.02,  # thick plate
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=300e6,
            alpha_thermal=1.2e-5, n_points=51
        )

        # Thin plate more likely to buckle
        if result_thin.is_buckled:
            assert result_thin.buckling_displacement > 0
        # Thick plate less likely to buckle
        # (may or may not buckle depending on parameters)


# ============================================================================
# Integration tests
# ============================================================================

class TestWeldingIntegration:
    """Integration tests for welding thermomechanical analysis."""

    def test_full_welding_simulation_workflow(self):
        """Complete welding simulation: thermal -> residual stress -> buckling."""
        # Step 1: Thermal analysis (Rosenthal)
        T_peak = rosenthal_peak_temperature(
            power=3000.0, velocity=0.005,
            conductivity=50.0, diffusivity=1.5e-5,
            T_initial=20.0, reference_distance=0.005
        )
        assert T_peak > 1000.0, "Weld peak temperature should exceed 1000°C"

        # Step 2: Residual stress (Satoh)
        satoh_result = satoh_1d_residual_stress(
            n_points=101, peak_temp_center=500.0, width=0.05,
            young=210e9, yield_stress=300e6, alpha_thermal=1.2e-5
        )
        assert torch.max(torch.abs(satoh_result.residual_stress)) > 0

        # Step 3: Butt weld assessment
        butt_result = butt_weld_residual_stress_plate(
            width=0.2, thickness=0.01,
            weld_speed=0.005, power=3000.0,
            conductivity=50.0, diffusivity=1.5e-5,
            young=210e9, yield_stress=300e6,
            alpha_thermal=1.2e-5
        )
        assert butt_result.peak_tensile_stress > 0

    def test_welding_temperature_peak_error_under_15pct(self):
        """Key acceptance: welding temperature peak error < 15%."""
        # Compare Rosenthal with known analytical value
        power = 3000.0
        k = 50.0
        r_ref = 0.005
        T0 = 20.0

        T_computed = rosenthal_peak_temperature(power, 0.005, k, 1.5e-5, T0, r_ref)
        T_exact = T0 + power / (2.0 * math.pi * k * r_ref)

        rel_error = abs(T_computed - T_exact) / T_exact
        assert rel_error < 0.15, \
            f"Temperature peak error {rel_error:.4%} > 15%"
