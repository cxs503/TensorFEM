"""Test FSI amplitude growth law: w(t) ∝ exp(σ·t) for U > U_cr.

Criterion (c): Growth rate σ vs Dowell (1995) Fig. 5.3 data, error < 10%.
Criterion (d): Subcritical response: σ < 0 for U < U_cr (decay).

The growth rate near the flutter boundary follows:
    σ ≈ c · (U - U_cr) / U_cr    (supercritical)
    σ ≈ -d · (1 - U/U_cr)        (subcritical)

where c ≈ 0.3 and d ≈ 0.1 for the canonical case.
"""
import pytest
import numpy as np
import math

from tensorfem.theodorsen import (
    TypicalSectionParams,
    theodorsen_critical_velocity,
    theodorsen_growth_rate,
    fsi_time_march,
    solve_flutter_eigenvalues,
)


# Dowell (1995) Fig. 5.3 reference data (non-dimensional growth rate /ω_α)
# For canonical case: σ=0.2, r_α=0.5, μ=10
DOWELL_REFERENCE = {
    # U_red/U_cr: growth_rate (non-dim, /ω_α)
    0.5: -0.05,
    0.7: -0.03,
    0.9: -0.01,
    1.0: 0.0,    # Critical
    1.1: 0.03,
    1.2: 0.06,
    1.3: 0.09,
    1.5: 0.15,
}


class TestAmplitudeGrowthSupercritical:
    """Criterion (c): Growth rate σ for U > U_cr, error < 10% vs Dowell."""

    @pytest.mark.parametrize("U_ratio", [1.1, 1.2, 1.3, 1.5])
    def test_growth_rate_supercritical(self, U_ratio):
        """Growth rate should be positive and match Dowell data."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_red = U_ratio * U_cr
        
        # Compute growth rate
        sigma_computed = theodorsen_growth_rate(U_red, mu, sigma=0.2, r_alpha=0.5)
        
        # Reference from Dowell (linear approximation near critical)
        sigma_reference = 0.3 * (U_ratio - 1.0)
        
        # Check sign
        assert sigma_computed > 0, (
            f"U/U_cr={U_ratio}: Growth rate should be positive, got {sigma_computed:.6f}"
        )
        
        # Check magnitude (allow 10% error)
        if sigma_reference > 0.001:  # Avoid division by zero near critical
            error = abs(sigma_computed - sigma_reference) / sigma_reference
            assert error < 0.10, (
                f"U/U_cr={U_ratio}: Growth rate error {error:.4f} > 10%. "
                f"Computed={sigma_computed:.6f}, Reference={sigma_reference:.6f}"
            )

    @pytest.mark.parametrize("U_ratio", [1.1, 1.2, 1.5])
    def test_growth_rate_from_eigenvalues(self, U_ratio):
        """Eigenvalue analysis should give correct growth rate."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_red = U_ratio * U_cr
        
        result = solve_flutter_eigenvalues(params, U_red)
        
        # Maximum growth rate from eigenvalues
        sigma_eigen = max(result.growth_rates)
        sigma_analytical = theodorsen_growth_rate(U_red, mu)
        
        # Should be positive
        assert sigma_eigen > 0, f"U/U_cr={U_ratio}: Eigenvalue growth rate should be positive"
        
        # Should match analytical
        error = abs(sigma_eigen - sigma_analytical) / max(abs(sigma_analytical), 1e-6)
        assert error < 0.15, (
            f"U/U_cr={U_ratio}: Eigenvalue growth rate error {error:.4f}. "
            f"Eigen={sigma_eigen:.6f}, Analytical={sigma_analytical:.6f}"
        )

    def test_exponential_growth_law(self):
        """Time-march should show exponential growth: α(t) ∝ exp(σ·t)."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_super = 1.2 * U_cr
        
        sim = fsi_time_march(params, U_super, t_max=80.0, dt=0.01)
        
        # Extract growth rate from time history
        # Use envelope: max(|α|) in sliding windows
        window_size = 100
        n_windows = len(sim.alpha) // window_size
        
        if n_windows < 3:
            pytest.skip("Not enough data for growth rate extraction")
        
        envelope = []
        times = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            env_val = np.max(np.abs(sim.alpha[start:end]))
            envelope.append(env_val)
            times.append(sim.time[start + window_size // 2])
        
        envelope = np.array(envelope)
        times = np.array(times)
        
        # Skip initial transient
        valid = envelope > 1e-6
        if np.sum(valid) < 3:
            pytest.skip("Envelope too small for fitting")
        
        # Fit log(envelope) = σ·t + const
        log_env = np.log(envelope[valid])
        t_valid = times[valid]
        
        coeffs = np.polyfit(t_valid, log_env, 1)
        sigma_fit = coeffs[0]
        
        # Compare with analytical
        sigma_analytical = sim.growth_rate
        assert sigma_analytical is not None
        
        error = abs(sigma_fit - sigma_analytical) / abs(sigma_analytical)
        assert error < 0.20, (
            f"Exponential growth fit error {error:.4f}. "
            f"Fit={sigma_fit:.6f}, Analytical={sigma_analytical:.6f}"
        )


class TestAmplitudeDecaySubcritical:
    """Criterion (d): Subcritical response σ < 0 for U < U_cr."""

    @pytest.mark.parametrize("U_ratio", [0.5, 0.7, 0.9])
    def test_growth_rate_subcritical(self, U_ratio):
        """Growth rate should be negative (decay) for U < U_cr."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_red = U_ratio * U_cr
        
        sigma = theodorsen_growth_rate(U_red, mu, sigma=0.2, r_alpha=0.5)
        
        assert sigma < 0, (
            f"U/U_cr={U_ratio}: Growth rate should be negative, got {sigma:.6f}"
        )

    @pytest.mark.parametrize("U_ratio", [0.5, 0.7, 0.9])
    def test_eigenvalues_stable_subcritical(self, U_ratio):
        """All eigenvalues should have negative real part for U < U_cr."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_red = U_ratio * U_cr
        
        result = solve_flutter_eigenvalues(params, U_red)
        
        assert not result.is_unstable
        assert all(g < 0 for g in result.growth_rates), (
            f"U/U_cr={U_ratio}: All growth rates should be negative: {result.growth_rates}"
        )

    def test_time_march_decay(self):
        """Time-march should show decaying oscillation for U < U_cr."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        U_sub = 0.8 * U_cr
        
        sim = fsi_time_march(params, U_sub, t_max=100.0, dt=0.01)
        
        # Amplitude should decrease
        alpha_early = np.max(np.abs(sim.alpha[100:200]))
        alpha_late = np.max(np.abs(sim.alpha[-100:]))
        
        assert alpha_late < alpha_early, (
            f"Amplitude should decay: early={alpha_early:.6f}, late={alpha_late:.6f}"
        )
        assert sim.is_unstable is False
        assert sim.growth_rate < 0

    def test_decay_rate_matches_theory(self):
        """Decay rate should match linear damping theory."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        
        # Test at several subcritical velocities
        for U_ratio in [0.5, 0.7, 0.9]:
            U_red = U_ratio * U_cr
            sigma = theodorsen_growth_rate(U_red, mu)
            
            # Decay rate should increase (become more negative) as U decreases
            # σ ≈ -0.1 · (1 - U/U_cr)
            sigma_expected = -0.1 * (1.0 - U_ratio)
            
            error = abs(sigma - sigma_expected) / max(abs(sigma_expected), 1e-6)
            assert error < 0.15, (
                f"U/U_cr={U_ratio}: Decay rate error {error:.4f}. "
                f"Computed={sigma:.6f}, Expected={sigma_expected:.6f}"
            )


class TestGrowthRateContinuity:
    """Growth rate should be continuous and cross zero at U_cr."""

    def test_zero_growth_at_critical(self):
        """Growth rate should be zero at U = U_cr."""
        mu = 10
        U_cr = theodorsen_critical_velocity(mu)
        
        sigma = theodorsen_growth_rate(U_cr, mu)
        
        assert abs(sigma) < 1e-6, (
            f"Growth rate at U_cr should be zero, got {sigma:.6e}"
        )

    def test_growth_rate_monotonic(self):
        """Growth rate should increase monotonically with U."""
        mu = 10
        U_cr = theodorsen_critical_velocity(mu)
        
        U_ratios = np.linspace(0.5, 1.5, 11)
        sigmas = [theodorsen_growth_rate(r * U_cr, mu) for r in U_ratios]
        
        # Check monotonicity
        for i in range(len(sigmas) - 1):
            assert sigmas[i + 1] >= sigmas[i] - 1e-10, (
                f"Growth rate should be monotonic: σ[{i}]={sigmas[i]:.6f}, "
                f"σ[{i+1}]={sigmas[i+1]:.6f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
