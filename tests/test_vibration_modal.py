"""Verification of ship hull structural vibration modal analysis.

Tests cover:
1. Dry modal vs Timoshenko-Mindlin plate theory (|err| < 3%)
2. Wet modal vs BEM analytical reference (|err| < 5%)
3. Radiation efficiency vs Leppington (1982) approximation (|err| < 5%)
4. Energy balance computation
5. FRF computation and VAL
6. Input validation and edge cases

All references are independent closed-form solutions or published data.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.vibration import (
    timoshenko_plate_angular_frequency,
    solve_dry_modal,
    solve_wet_modal,
    compute_frf,
    compute_radiated_noise,
    check_energy_balance,
    radiation_efficiency_leppington,
    critical_frequency,
    abramson_wet_ratio,
    DryModalResult,
    WetModalResult,
    FRFResult,
    RadiatedNoiseResult,
    EnergyBalanceResult,
    AbramsonWetRatio,
)
from tensorfem.plate_vibration import kirchhoff_angular_frequency, plate_rigidity
from tensorfem.hydro_added_mass import wet_frequency_ratio, panel_added_mass_ratio

# Standard steel plate parameters
STEEL = dict(
    density=7850.0,
    young=200.0e9,
    poisson=0.3,
    thickness=0.01,
)
WATER = dict(rho_water=1025.0, c_water=1500.0)


def _hz(omega: float) -> float:
    return omega / (2.0 * math.pi)


def _relative_error(value: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if value != 0 else 0.0
    return abs(value - reference) / abs(reference)


def _center_dof(elements_x: int, elements_y: int) -> int:
    """Return the transverse DOF at the plate center (non-zero for (1,1) mode)."""
    n_nodes_x = elements_x + 1
    n_nodes_y = elements_y + 1
    cx = n_nodes_x // 2
    cy = n_nodes_y // 2
    node = cy * n_nodes_x + cx
    return 3 * node  # w-DOF


# ===================================================================
# 1. Dry modal analysis vs Timoshenko-Mindlin theory
# ===================================================================

class TestDryModal:
    """Dry modal analysis verification against Timoshenko-Mindlin plate theory."""

    def test_fundamental_mode_within_three_percent(self):
        """Fundamental mode of simply supported square plate vs Timoshenko theory."""
        result = solve_dry_modal(16, 16, length=1.0, width=1.0, modes=3, **STEEL)
        omega_11 = timoshenko_plate_angular_frequency(
            length=1.0, width=1.0, half_waves_x=1, half_waves_y=1, **STEEL
        )
        f_analytical = _hz(omega_11)
        f_computed = float(result.frequencies_hz[0])
        err = _relative_error(f_computed, f_analytical)
        assert err < 0.03, f"Mode 1: computed={f_computed:.2f} Hz, analytical={f_analytical:.2f} Hz, err={err:.4f}"

    def test_three_modes_within_three_percent(self):
        """First 3 modes vs Timoshenko-Mindlin theory, all |err| < 3%."""
        result = solve_dry_modal(16, 16, length=1.0, width=1.0, modes=3, **STEEL)
        for i in range(3):
            m = (i // 2) + 1
            n = (i % 2) + 1
            omega = timoshenko_plate_angular_frequency(
                length=1.0, width=1.0, half_waves_x=m, half_waves_y=n, **STEEL
            )
            f_analytical = _hz(omega)
            f_computed = float(result.frequencies_hz[i])
            err = _relative_error(f_computed, f_analytical)
            assert err < 0.03, (
                f"Mode {i+1} ({m},{n}): computed={f_computed:.2f} Hz, "
                f"analytical={f_analytical:.2f} Hz, err={err:.4f}"
            )

    def test_rectangular_plate_modes(self):
        """Rectangular plate (2:1 aspect ratio) modes vs Timoshenko theory."""
        result = solve_dry_modal(20, 10, length=2.0, width=1.0, modes=3, **STEEL)
        omega_11 = timoshenko_plate_angular_frequency(
            length=2.0, width=1.0, half_waves_x=1, half_waves_y=1, **STEEL
        )
        f_analytical = _hz(omega_11)
        f_computed = float(result.frequencies_hz[0])
        err = _relative_error(f_computed, f_analytical)
        assert err < 0.03, f"Rectangular mode 1: err={err:.4f}"

    def test_mesh_refinement_converges(self):
        """Frequency converges with mesh refinement."""
        errors = []
        for nx in [8, 12, 16]:
            result = solve_dry_modal(nx, nx, length=1.0, width=1.0, modes=1, **STEEL)
            omega = timoshenko_plate_angular_frequency(
                length=1.0, width=1.0, half_waves_x=1, half_waves_y=1, **STEEL
            )
            f_analytical = _hz(omega)
            errors.append(_relative_error(float(result.frequencies_hz[0]), f_analytical))
        assert errors[-1] < errors[0], f"Errors not decreasing: {errors}"
        assert errors[-1] < 0.03

    def test_timoshenko_reduces_to_kirchhoff_for_thin_plate(self):
        """Timoshenko frequency approaches Kirchhoff for very thin plate."""
        thin_steel = dict(**STEEL, thickness=0.001)
        omega_tim = timoshenko_plate_angular_frequency(
            length=1.0, width=1.0, half_waves_x=1, half_waves_y=1, **thin_steel
        )
        omega_kirch = kirchhoff_angular_frequency(
            length=1.0, width=1.0, half_waves_x=1, half_waves_y=1, **thin_steel
        )
        err = _relative_error(omega_tim, omega_kirch)
        assert err < 0.01, f"Timoshenko vs Kirchhoff for thin plate: err={err:.6f}"

    def test_dry_modal_result_metadata(self):
        """DryModalResult contains expected fields."""
        result = solve_dry_modal(8, 8, length=1.0, width=1.0, modes=3, **STEEL)
        assert isinstance(result, DryModalResult)
        assert result.frequencies_hz.shape == (3,)
        assert result.element_counts == (8, 8)
        assert len(result.analytical_frequencies_hz) == 3
        assert len(result.relative_errors) == 3
        assert torch.all(result.frequencies_hz[1:] > result.frequencies_hz[:-1])


# ===================================================================
# 2. Wet modal analysis vs BEM analytical reference
# ===================================================================

class TestWetModal:
    """Wet modal analysis verification against BEM analytical reference."""

    def test_wet_frequency_lower_than_dry(self):
        """Wet frequencies must be lower than dry frequencies."""
        result = solve_wet_modal(12, 12, length=1.0, width=1.0, modes=3, **STEEL)
        for i in range(3):
            assert float(result.wet_frequencies_hz[i]) < float(result.dry_frequencies_hz[i]), (
                f"Mode {i+1}: wet={result.wet_frequencies_hz[i]:.2f} >= dry={result.dry_frequencies_hz[i]:.2f}"
            )

    def test_wet_dry_ratio_positive(self):
        """Wet/dry frequency ratio must be in (0, 1)."""
        result = solve_wet_modal(12, 12, length=1.0, width=1.0, modes=3, **STEEL)
        for i in range(3):
            ratio = float(result.frequency_ratios[i])
            assert 0.0 < ratio < 1.0, f"Mode {i+1}: ratio={ratio}"

    def test_wet_modal_vs_bem_analytical_within_five_percent(self):
        """Wet/dry ratio vs BEM analytical reference (1/sqrt(1+mu)) within 5%."""
        result = solve_wet_modal(12, 12, length=1.0, width=1.0, modes=3, **STEEL)
        for i in range(3):
            computed_ratio = float(result.frequency_ratios[i])
            analytical_ratio = result.analytical_ratios[i]
            err = _relative_error(computed_ratio, analytical_ratio)
            assert err < 0.05, (
                f"Mode {i+1}: computed_ratio={computed_ratio:.4f}, "
                f"analytical_ratio={analytical_ratio:.4f}, err={err:.4f}"
            )

    def test_added_mass_increases_with_water_density(self):
        """Higher water density -> lower wet frequency."""
        result_fresh = solve_wet_modal(10, 10, length=1.0, width=1.0, modes=1,
                                       rho_water=1000.0, **STEEL)
        result_salt = solve_wet_modal(10, 10, length=1.0, width=1.0, modes=1,
                                      rho_water=1025.0, **STEEL)
        assert float(result_salt.wet_frequencies_hz[0]) < float(result_fresh.wet_frequencies_hz[0])

    def test_abramson_wet_ratio_reasonable(self):
        """Abramson wet ratio is in physical range."""
        abr = abramson_wet_ratio(aspect_ratio=1.0, mode_m=1, mode_n=1, a_over_h=100.0)
        assert isinstance(abr, AbramsonWetRatio)
        assert 0.0 < abr.frequency_ratio < 1.0
        assert 0.5 < abr.frequency_ratio < 1.0

    def test_wet_modal_added_mass_ratios_positive(self):
        """Added mass ratios must be positive."""
        result = solve_wet_modal(10, 10, length=1.0, width=1.0, modes=3, **STEEL)
        for mu in result.added_mass_ratios:
            assert mu > 0, f"mu = {mu}"


# ===================================================================
# 3. Radiation efficiency vs Leppington (1982)
# ===================================================================

class TestRadiationEfficiency:
    """Radiation efficiency verification against Leppington (1982)."""

    def test_critical_frequency_positive(self):
        """Critical frequency must be positive."""
        f_c = critical_frequency(**STEEL, **WATER)
        assert f_c > 0

    def test_efficiency_below_critical(self):
        """Below f_c, radiation efficiency < 1."""
        f_c = critical_frequency(**STEEL, **WATER)
        sigma = radiation_efficiency_leppington(f_c * 0.5, length=1.0, width=1.0, **STEEL, **WATER)
        assert 0.0 < sigma < 1.0

    def test_efficiency_at_critical(self):
        """At f_c, radiation efficiency = 1."""
        f_c = critical_frequency(**STEEL, **WATER)
        sigma = radiation_efficiency_leppington(f_c, length=1.0, width=1.0, **STEEL, **WATER)
        assert sigma == 1.0, f"sigma at f_c = {sigma}"

    def test_efficiency_above_critical(self):
        """Above f_c, radiation efficiency = 1."""
        f_c = critical_frequency(**STEEL, **WATER)
        sigma = radiation_efficiency_leppington(f_c * 2.0, length=1.0, width=1.0, **STEEL, **WATER)
        assert sigma == 1.0

    def test_efficiency_zero_at_zero_frequency(self):
        """At f=0, radiation efficiency = 0."""
        sigma = radiation_efficiency_leppington(0.0, length=1.0, width=1.0, **STEEL, **WATER)
        assert sigma == 0.0

    def test_efficiency_scales_with_frequency_squared_below_fc(self):
        """Below f_c, sigma ~ (f/f_c)^2."""
        f_c = critical_frequency(**STEEL, **WATER)
        f1 = f_c * 0.25
        f2 = f_c * 0.5
        sigma1 = radiation_efficiency_leppington(f1, length=1.0, width=1.0, **STEEL, **WATER)
        sigma2 = radiation_efficiency_leppington(f2, length=1.0, width=1.0, **STEEL, **WATER)
        if sigma1 > 0:
            ratio = sigma2 / sigma1
            assert 3.0 < ratio < 5.0, f"sigma ratio = {ratio}, expected ~4"

    def test_radiated_noise_result_metadata(self):
        """RadiatedNoiseResult contains expected fields."""
        dry = solve_dry_modal(8, 8, length=1.0, width=1.0, modes=3, **STEEL)
        wet = solve_wet_modal(8, 8, length=1.0, width=1.0, modes=3, **STEEL)
        frf = compute_frf(dry, freq_min=10.0, freq_max=200.0, n_points=50)
        noise = compute_radiated_noise(wet, frf, length=1.0, width=1.0, **STEEL, **WATER)
        assert isinstance(noise, RadiatedNoiseResult)
        assert noise.frequencies_hz.shape == (50,)
        assert noise.radiation_efficiency.shape == (50,)
        assert noise.sound_pressure_level.shape == (50,)


# ===================================================================
# 4. FRF and VAL
# ===================================================================

class TestFRF:
    """Frequency response function verification."""

    def test_frf_resonance_peak(self):
        """FRF shows peak near natural frequency."""
        dry = solve_dry_modal(12, 12, length=1.0, width=1.0, modes=3, **STEEL)
        f1 = float(dry.frequencies_hz[0])
        dof = _center_dof(12, 12)
        frf = compute_frf(dry, freq_min=f1 * 0.5, freq_max=f1 * 1.5, n_points=200,
                          response_dof=dof)
        peak_idx = torch.argmax(frf.frf_magnitude)
        peak_freq = float(frf.frequencies_hz[peak_idx])
        assert _relative_error(peak_freq, f1) < 0.1, f"Peak at {peak_freq:.1f} Hz, expected ~{f1:.1f} Hz"

    def test_val_positive_at_resonance(self):
        """VAL is positive (above reference) at resonance."""
        dry = solve_dry_modal(12, 12, length=1.0, width=1.0, modes=3, **STEEL)
        f1 = float(dry.frequencies_hz[0])
        dof = _center_dof(12, 12)
        frf = compute_frf(dry, freq_min=f1 * 0.5, freq_max=f1 * 1.5, n_points=100,
                          excitation_force=1000.0, excitation_freq=f1, response_dof=dof)
        peak_val = float(torch.max(frf.val_db))
        assert peak_val > 0, f"VAL at resonance = {peak_val:.1f} dB"

    def test_frf_result_metadata(self):
        """FRFResult contains expected fields."""
        dry = solve_dry_modal(8, 8, length=1.0, width=1.0, modes=3, **STEEL)
        dof = _center_dof(8, 8)
        frf = compute_frf(dry, freq_min=10.0, freq_max=200.0, n_points=50, response_dof=dof)
        assert isinstance(frf, FRFResult)
        assert frf.frequencies_hz.shape == (50,)
        assert frf.frf_magnitude.shape == (50,)
        assert frf.frf_phase.shape == (50,)
        assert frf.val_db.shape == (50,)
        assert frf.excitation_spectrum.shape == (50,)


# ===================================================================
# 5. Energy balance
# ===================================================================

class TestEnergyBalance:
    """Energy balance verification."""

    def test_energy_balance_computation(self):
        """Energy balance computation produces valid results."""
        dry = solve_dry_modal(10, 10, length=1.0, width=1.0, modes=3, **STEEL)
        wet = solve_wet_modal(10, 10, length=1.0, width=1.0, modes=3, **STEEL)
        dof = _center_dof(10, 10)
        frf = compute_frf(dry, freq_min=10.0, freq_max=200.0, n_points=100,
                          damping_ratio=0.02, response_dof=dof)
        noise = compute_radiated_noise(wet, frf, length=1.0, width=1.0, **STEEL, **WATER)
        balance = check_energy_balance(
            frf, noise, damping_ratio=0.02,
            density=STEEL["density"], thickness=STEEL["thickness"],
            length=1.0, width=1.0, tolerance=1.0,
        )
        assert isinstance(balance, EnergyBalanceResult)
        assert balance.input_power >= 0
        assert balance.radiated_power >= 0
        assert balance.dissipated_power >= 0
        assert 0.0 <= balance.balance_error < 1.0


# ===================================================================
# 6. Input validation
# ===================================================================

class TestInputValidation:
    """Input validation tests."""

    def test_timoshenko_rejects_negative_thickness(self):
        with pytest.raises(ValueError):
            timoshenko_plate_angular_frequency(
                length=1.0, width=1.0, density=7850.0, young=200.0e9,
                poisson=0.3, thickness=-0.01, half_waves_x=1, half_waves_y=1,
            )

    def test_timoshenko_rejects_invalid_poisson(self):
        with pytest.raises(ValueError):
            timoshenko_plate_angular_frequency(
                length=1.0, width=1.0, density=7850.0, young=200.0e9,
                poisson=0.5, thickness=0.01, half_waves_x=1, half_waves_y=1,
            )

    def test_timoshenko_rejects_zero_mode(self):
        with pytest.raises(ValueError):
            timoshenko_plate_angular_frequency(
                length=1.0, width=1.0, density=7850.0, young=200.0e9,
                poisson=0.3, thickness=0.01, half_waves_x=0, half_waves_y=1,
            )

    def test_abramson_rejects_negative_aspect(self):
        with pytest.raises(ValueError):
            abramson_wet_ratio(aspect_ratio=-1.0)

    def test_radiation_efficiency_negative_frequency(self):
        sigma = radiation_efficiency_leppington(-10.0, length=1.0, width=1.0, **STEEL, **WATER)
        assert sigma == 0.0
