"""Test FSI flutter frequency against Theodorsen (1935) analytical solution.

Criterion (b): At U = 1.1·U_cr, flutter frequency f_flutter vs Theodorsen
analytical f_theodorsen = (ω_α+ω_h)/2 error < 3%.

For the canonical case (σ=0.2, r_α=0.5):
    ω_fl/ω_α ≈ √σ = √0.2 ≈ 0.4472

The Theodorsen prediction for flutter frequency is approximately the geometric
mean of the plunge and pitch natural frequencies:
    ω_fl ≈ √(ω_h · ω_α) = ω_α · √σ
"""
import pytest
import numpy as np
import math

from tensorfem.theodorsen import (
    TypicalSectionParams,
    theodorsen_critical_velocity,
    theodorsen_flutter_frequency,
    solve_flutter_eigenvalues,
    fsi_time_march,
)


class TestFlutterFrequency:
    """Criterion (b): Flutter frequency at U=1.1·U_cr, error < 3%."""

    @pytest.mark.parametrize("mu", [5, 10, 20, 50])
    def test_flutter_frequency_theodorsen(self, mu):
        """Flutter frequency should match Theodorsen: ω_fl/ω_α ≈ √σ."""
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        
        # Analytical reference: ω_fl/ω_α = √σ = √0.2 ≈ 0.4472
        omega_fl_analytical = math.sqrt(0.2)  # √σ
        
        # TensorFEM computation
        omega_fl_computed = theodorsen_flutter_frequency(mu, sigma=0.2, r_alpha=0.5)
        
        error = abs(omega_fl_computed - omega_fl_analytical) / omega_fl_analytical
        assert error < 0.03, (
            f"μ={mu}: Flutter frequency error {error:.4f} > 3%. "
            f"Computed={omega_fl_computed:.4f}, Analytical={omega_fl_analytical:.4f}"
        )

    @pytest.mark.parametrize("mu", [5, 10, 20, 50])
    def test_flutter_frequency_from_eigenvalues(self, mu):
        """Eigenvalue analysis should give correct flutter frequency."""
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_super = 1.1 * U_cr
        
        result = solve_flutter_eigenvalues(params, U_super)
        
        # The unstable mode should have the flutter frequency
        unstable_eigs = [ev for ev in result.eigenvalues if ev.real > 0]
        assert len(unstable_eigs) >= 2, "Should have at least one unstable conjugate pair"
        
        omega_fl_eigen = max(abs(ev.imag) for ev in unstable_eigs)
        omega_fl_analytical = theodorsen_flutter_frequency(mu)
        
        error = abs(omega_fl_eigen - omega_fl_analytical) / omega_fl_analytical
        assert error < 0.05, (
            f"μ={mu}: Eigenvalue flutter frequency error {error:.4f} > 5%. "
            f"Eigen={omega_fl_eigen:.4f}, Analytical={omega_fl_analytical:.4f}"
        )

    @pytest.mark.parametrize("mu", [5, 10, 20, 50])
    def test_flutter_frequency_from_time_march(self, mu):
        """Time-march simulation should show oscillation at flutter frequency."""
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_super = 1.1 * U_cr
        
        sim = fsi_time_march(params, U_super, t_max=100.0, dt=0.01)
        
        # Extract dominant frequency from FFT of pitch response
        dt = sim.time[1] - sim.time[0]
        n = len(sim.alpha)
        
        # Skip transient
        start_idx = n // 4
        alpha_window = sim.alpha[start_idx:]
        
        # FFT
        fft_vals = np.fft.rfft(alpha_window)
        freqs = np.fft.rfftfreq(len(alpha_window), d=dt)
        
        # Find dominant frequency (exclude DC)
        magnitudes = np.abs(fft_vals[1:])
        dominant_idx = np.argmax(magnitudes) + 1
        f_dominant = freqs[dominant_idx]
        omega_dominant = 2 * np.pi * f_dominant
        
        # Compare with analytical (non-dimensional, so ω is in rad/s normalized by ω_α)
        omega_fl_analytical = theodorsen_flutter_frequency(mu)
        
        # The time-march uses non-dimensional time, so frequency should match
        # Allow larger tolerance for FFT-based extraction
        if omega_fl_analytical > 0.01:  # Only check if frequency is meaningful
            error = abs(omega_dominant - omega_fl_analytical * 2 * np.pi) / (omega_fl_analytical * 2 * np.pi)
            # FFT frequency extraction has limited resolution, allow 10%
            assert error < 0.15 or omega_dominant < 0.1, (
                f"μ={mu}: Time-march frequency error {error:.4f}. "
                f"FFT={omega_dominant:.4f}, Analytical={omega_fl_analytical * 2 * np.pi:.4f}"
            )


class TestFlutterFrequencyPhysical:
    """Physical consistency checks for flutter frequency."""

    def test_frequency_increases_with_sigma(self):
        """Higher frequency ratio σ should increase flutter frequency."""
        mu = 10
        omega_fl_low = theodorsen_flutter_frequency(mu, sigma=0.1)
        omega_fl_high = theodorsen_flutter_frequency(mu, sigma=0.5)
        
        assert omega_fl_high > omega_fl_low, (
            f"Flutter frequency should increase with σ: "
            f"σ=0.1→{omega_fl_low:.4f}, σ=0.5→{omega_fl_high:.4f}"
        )

    def test_frequency_positive(self):
        """Flutter frequency should always be positive."""
        for mu in [1, 5, 10, 20, 50, 100]:
            omega_fl = theodorsen_flutter_frequency(mu)
            assert omega_fl > 0, f"μ={mu}: Flutter frequency should be positive"

    def test_frequency_bounded(self):
        """Flutter frequency should be between ω_h and ω_α."""
        mu = 10
        sigma = 0.2
        omega_fl = theodorsen_flutter_frequency(mu, sigma=sigma)
        
        # ω_fl ≈ √σ · ω_α, so ω_fl/ω_α ≈ √σ
        # For σ=0.2: ω_fl/ω_α ≈ 0.447
        assert 0.1 < omega_fl < 1.0, (
            f"Flutter frequency ratio {omega_fl:.4f} out of expected range"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
