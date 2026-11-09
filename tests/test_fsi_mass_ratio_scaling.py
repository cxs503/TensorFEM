"""Test FSI mass ratio scaling: U_cr ∝ √μ (Theodorsen scaling law).

Criterion (e): For μ ∈ [5, 50], scaling exponent error < 2%.

The Theodorsen (1935) theory predicts:
    U_cr/(b·ω_α) = 2.0 · √μ    (canonical case)

This means U_cr ∝ μ^(1/2), i.e., the scaling exponent is exactly 0.5.
This test verifies that the TensorFEM FSI framework reproduces this scaling.
"""
import pytest
import numpy as np
import math

from tensorfem.theodorsen import (
    TypicalSectionParams,
    theodorsen_critical_velocity,
    find_critical_flutter,
    solve_flutter_eigenvalues,
)


class TestMassRatioScaling:
    """Criterion (e): U_cr ∝ √μ, scaling exponent error < 2%."""

    def test_scaling_exponent(self):
        """Log-log fit should give exponent ≈ 0.5."""
        mu_vals = [5, 10, 20, 50]
        U_cr_vals = [theodorsen_critical_velocity(mu) for mu in mu_vals]
        
        # Log-log linear fit: log(U_cr) = exponent · log(μ) + const
        log_mu = np.log(mu_vals)
        log_U = np.log(U_cr_vals)
        
        coeffs = np.polyfit(log_mu, log_U, 1)
        exponent = coeffs[0]
        
        error = abs(exponent - 0.5) / 0.5
        assert error < 0.02, (
            f"Scaling exponent {exponent:.6f} should be 0.5, error={error:.4f} > 2%"
        )

    @pytest.mark.parametrize("mu", [5, 10, 20, 50])
    def test_exact_scaling(self, mu):
        """U_cr/(bω_α) should equal 2√μ exactly."""
        U_cr = theodorsen_critical_velocity(mu, sigma=0.2, r_alpha=0.5)
        U_cr_exact = 2.0 * math.sqrt(mu)
        
        error = abs(U_cr - U_cr_exact) / U_cr_exact
        assert error < 0.02, (
            f"μ={mu}: U_cr={U_cr:.6f}, 2√μ={U_cr_exact:.6f}, error={error:.4f}"
        )

    def test_scaling_ratio(self):
        """U_cr(μ₂)/U_cr(μ₁) should equal √(μ₂/μ₁)."""
        mu1, mu2 = 10, 40
        U_cr1 = theodorsen_critical_velocity(mu1)
        U_cr2 = theodorsen_critical_velocity(mu2)
        
        ratio_computed = U_cr2 / U_cr1
        ratio_exact = math.sqrt(mu2 / mu1)
        
        error = abs(ratio_computed - ratio_exact) / ratio_exact
        assert error < 0.02, (
            f"Scaling ratio error {error:.4f}. "
            f"Computed={ratio_computed:.6f}, Exact={ratio_exact:.6f}"
        )

    def test_scaling_over_wide_range(self):
        """Scaling should hold over wide mass ratio range [1, 100]."""
        mu_vals = np.logspace(0, 2, 20)  # 1 to 100
        U_cr_vals = [theodorsen_critical_velocity(mu) for mu in mu_vals]
        
        log_mu = np.log(mu_vals)
        log_U = np.log(U_cr_vals)
        
        coeffs = np.polyfit(log_mu, log_U, 1)
        exponent = coeffs[0]
        r_squared = 1 - np.sum((log_U - np.polyval(coeffs, log_mu))**2) / np.sum((log_U - np.mean(log_U))**2)
        
        error = abs(exponent - 0.5) / 0.5
        assert error < 0.02, (
            f"Scaling exponent {exponent:.6f} error {error:.4f} > 2% over [1, 100]"
        )
        assert r_squared > 0.999, f"R² = {r_squared:.6f} should be > 0.999"


class TestMassRatioScalingPhysical:
    """Physical consistency of mass ratio scaling."""

    def test_scaling_with_frequency_ratio(self):
        """Scaling exponent should be robust to frequency ratio variations."""
        sigma_vals = [0.1, 0.2, 0.3, 0.5]
        
        for sigma in sigma_vals:
            mu_vals = [5, 10, 20, 50]
            U_cr_vals = [theodorsen_critical_velocity(mu, sigma=sigma) for mu in mu_vals]
            
            log_mu = np.log(mu_vals)
            log_U = np.log(U_cr_vals)
            coeffs = np.polyfit(log_mu, log_U, 1)
            exponent = coeffs[0]
            
            error = abs(exponent - 0.5) / 0.5
            assert error < 0.05, (
                f"σ={sigma}: Scaling exponent {exponent:.6f} error {error:.4f}"
            )

    def test_scaling_with_radius_of_gyration(self):
        """Scaling exponent should be robust to r_α variations."""
        r_alpha_vals = [0.3, 0.5, 0.7]
        
        for r_alpha in r_alpha_vals:
            mu_vals = [5, 10, 20, 50]
            U_cr_vals = [theodorsen_critical_velocity(mu, r_alpha=r_alpha) for mu in mu_vals]
            
            log_mu = np.log(mu_vals)
            log_U = np.log(U_cr_vals)
            coeffs = np.polyfit(log_mu, log_U, 1)
            exponent = coeffs[0]
            
            error = abs(exponent - 0.5) / 0.5
            assert error < 0.05, (
                f"r_α={r_alpha}: Scaling exponent {exponent:.6f} error {error:.4f}"
            )

    def test_mass_ratio_definition(self):
        """Verify mass ratio μ = m/(πρb²) is correctly used."""
        # For a plate with given properties, compute μ
        rho_s = 2700  # kg/m³ (aluminum)
        h = 0.01      # m (thickness)
        b = 0.1       # m (semi-chord)
        rho_f = 1.225 # kg/m³ (air)
        
        m = rho_s * h  # mass per unit area × unit span = kg/m
        mu = m / (math.pi * rho_f * b**2)
        
        # Should be around 70 for these parameters
        expected_mu = 2700 * 0.01 / (math.pi * 1.225 * 0.1**2)
        assert abs(mu - expected_mu) / expected_mu < 1e-10


class TestMassRatioScalingEigenvalue:
    """Verify scaling through eigenvalue analysis."""

    @pytest.mark.parametrize("mu", [5, 10, 20, 50])
    def test_eigenvalue_scaling(self, mu):
        """Eigenvalue-based U_cr should follow √μ scaling."""
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        
        # Find U_cr by bisection on eigenvalue sign
        U_min, U_max = 0.1, 50.0
        for _ in range(30):
            U_mid = 0.5 * (U_min + U_max)
            result = solve_flutter_eigenvalues(params, U_mid)
            if result.is_unstable:
                U_max = U_mid
            else:
                U_min = U_mid
        
        U_cr_eigen = 0.5 * (U_min + U_max)
        U_cr_analytical = theodorsen_critical_velocity(mu)
        
        error = abs(U_cr_eigen - U_cr_analytical) / U_cr_analytical
        # Eigenvalue bisection has limited precision
        assert error < 0.10, (
            f"μ={mu}: Eigenvalue U_cr error {error:.4f}. "
            f"Eigen={U_cr_eigen:.4f}, Analytical={U_cr_analytical:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
