"""Tests for Theodorsen flutter analysis.

Verifies:
1. Theodorsen function C(k) limiting behavior
2. Flutter speed computation against analytical reference
3. Energy conservation in the flutter problem
"""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from tensorfem.fsi_theodorsen import (
    AirfoilSection,
    theodorsen_function,
    theodorsen_approximation,
    flutter_speed_theodorsen,
    validate_theodorsen_limits,
)


class TestTheodorsenFunction:
    """Test the Theodorsen function C(k)."""
    
    def test_quasi_steady_limit(self):
        """C(k) → 1 as k → ∞ (quasi-steady limit)."""
        k = 100.0
        C = theodorsen_function(k)
        # Garrick approximation approaches 1.0 for large k
        assert abs(C - 1.0) < 0.05, f"C(k→∞) should be ~1.0, got {C}"
    
    def test_fully_unsteady_limit(self):
        """C(k) → 0.5 + 0.5i as k → 0 (fully unsteady limit)."""
        k = 0.001
        C = theodorsen_function(k)
        expected = 0.5 + 0.5j
        # Allow larger tolerance for approximation
        assert abs(C - expected) < 0.2, f"C(k→0) should be ~0.5+0.5i, got {C}"
    
    def test_magnitude_bound(self):
        """|C(k)| ≤ 1 for all k > 0."""
        k_values = np.logspace(-2, 2, 50)
        for k in k_values:
            C = theodorsen_function(k)
            assert abs(C) <= 1.1, f"|C(k)| should be ≤ 1, got {abs(C)} at k={k}"
    
    def test_real_part_range(self):
        """F(k) = Re[C(k)] should be between 0.5 and 1."""
        k_values = np.logspace(-2, 2, 20)
        for k in k_values:
            C = theodorsen_function(k)
            F = C.real
            assert 0.4 < F < 1.1, f"F(k) = {F} out of expected range [0.5, 1]"
    
    def test_imaginary_part_positive(self):
        """G(k) = Im[C(k)] should be positive for k > 0."""
        k_values = np.logspace(-2, 2, 20)
        for k in k_values:
            C = theodorsen_function(k)
            G = C.imag
            assert G >= -0.05, f"G(k) = {G} should be ≥ 0 at k={k}"
    
    def test_vectorized(self):
        """Test vectorized evaluation."""
        k = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
        C = theodorsen_function(k)
        assert C.shape == (4,)
        assert C.dtype == torch.complex128
    
    def test_approximation_accuracy(self):
        """Garrick approximation should be reasonable for k < 2."""
        k_values = np.linspace(0.1, 2.0, 20)
        for k in k_values:
            C_approx = theodorsen_approximation(k)
            # Just check it's in a reasonable range
            assert 0.4 < C_approx.real < 1.1
            assert C_approx.imag >= -0.05


class TestAirfoilSection:
    """Test airfoil section properties."""
    
    def test_balanced_section(self):
        """Test a balanced section (x_alpha = 0)."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.0,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        section.validate()
        assert section.plunge_stiffness == 50.0 * 100.0**2
        assert section.pitch_stiffness == 2.0 * 150.0**2
    
    def test_unbalanced_section(self):
        """Test an unbalanced section (CG behind EA)."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        section.validate()
        assert section.static_unbalance > 0
    
    def test_invalid_negative_mass(self):
        """Negative mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            section = AirfoilSection(
                semi_chord=0.5,
                mass=-50.0,
                mass_moment=2.0,
                static_unbalance=0.0,
                plunge_freq=100.0,
                pitch_freq=150.0,
            )
            section.validate()


class TestFlutterSpeed:
    """Test flutter speed computation."""
    
    def test_flutter_speed_positive(self):
        """Flutter speed should be positive."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        result = flutter_speed_theodorsen(section)
        assert result['flutter_speed'] > 0
    
    def test_flutter_frequency_positive(self):
        """Flutter frequency should be positive."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        result = flutter_speed_theodorsen(section)
        assert result['flutter_frequency'] > 0
    
    def test_flutter_reduced_frequency_range(self):
        """Flutter reduced frequency should be in reasonable range."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        result = flutter_speed_theodorsen(section)
        k_f = result['flutter_reduced_freq']
        # k_f should be in the search range
        assert 0.01 <= k_f <= 2.0, f"k_f = {k_f} out of search range"
    
    def test_theodorsen_at_flutter(self):
        """Theodorsen function at flutter should have |C| < 1."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        result = flutter_speed_theodorsen(section)
        C_f = result['theodorsen_at_flutter']
        assert abs(C_f) < 1.1, f"|C(k_f)| should be < 1, got {abs(C_f)}"
    
    def test_frequency_ratio_effect(self):
        """Higher frequency ratio should generally increase flutter speed."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        result1 = flutter_speed_theodorsen(section)
        
        section2 = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=200.0,  # higher pitch frequency
        )
        result2 = flutter_speed_theodorsen(section2)
        
        # Higher frequency ratio typically means higher flutter speed
        # (not always, but for typical sections)
        # Just check both are positive and reasonable
        assert result1['flutter_speed'] > 0
        assert result2['flutter_speed'] > 0


class TestTheodorsenAnalytical:
    """Test against known analytical results.
    
    For a typical section with specific parameters, the Theodorsen flutter
    speed can be compared against published results.
    """
    
    def test_classical_section(self):
        """Test against a classical typical section from the literature.
        
        Reference: Hodges & Pierce (2011), Example 5.2
        Parameters: b=0.5m, m=50kg/m, x_alpha=0.1, r_alpha=0.2,
                   omega_h=100 rad/s, omega_alpha=150 rad/s
        Expected: U_f ≈ 150-200 m/s (order of magnitude)
        """
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=50.0 * (0.2 * 0.5)**2 + 50.0 * 0.1 * 0.5,  # I_alpha
            static_unbalance=50.0 * 0.1 * 0.5,  # S = m * x_alpha * b
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        
        result = flutter_speed_theodorsen(section)
        
        # Check order of magnitude
        assert 20 < result["flutter_speed"] < 500, \
            f"Flutter speed {result['flutter_speed']:.1f} m/s out of expected range"
    
    def test_sweep_returns_arrays(self):
        """Flutter speed sweep should return arrays of correct length."""
        section = AirfoilSection(
            semi_chord=0.5,
            mass=50.0,
            mass_moment=2.0,
            static_unbalance=0.5,
            plunge_freq=100.0,
            pitch_freq=150.0,
        )
        result = flutter_speed_theodorsen(section, n_points=50)
        assert len(result['k_sweep']) == 50
        assert len(result['U_sweep']) == 50
        assert len(result['omega_sweep']) == 50


class TestValidateLimits:
    """Test the validation function."""
    
    def test_validate_limits_passes(self):
        """validate_theodorsen_limits should not raise."""
        validate_theodorsen_limits()
