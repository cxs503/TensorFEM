"""Tests for nonlinear structural verification: hyperelastic, plastic hinge, Taylor bar.

Verification cases:
(a) Neo-Hookean hyperelastic uniaxial/biaxial tension vs analytical
(b) Plastic hinge: simply supported beam plastic limit load
(c) Taylor bar impact: final mushroom head diameter and length
"""
import math
import pytest
import torch

from tensorfem.nonlinear_verification import (
    neo_hookean_uniaxial_stress,
    neo_hookean_biaxial_stress,
    neo_hookean_uniaxial_incompressible,
    neo_hookean_biaxial_incompressible,
    plastic_hinge_analysis,
    taylor_bar_analytical,
    taylor_bar_numerical,
)
from tensorfem.finite_strain_elasticity import neo_hookean_response

DTYPE = torch.float64


# ============================================================================
# (a) Neo-Hookean Hyperelastic Verification
# ============================================================================

class TestNeoHookeanUniaxial:
    """Neo-Hookean uniaxial tension verification."""

    def test_uniaxial_incompressible_small_strain(self):
        """At small strain, NH should match linear elasticity: sigma = E * eps."""
        E = 1000.0
        nu = 0.4999  # nearly incompressible
        mu = E / (2 * (1 + nu))
        stretch = 1.001  # 0.1% strain

        sigma_analytical = neo_hookean_uniaxial_incompressible(stretch, mu)
        # Linear elasticity: sigma = E * eps = E * (stretch - 1)
        sigma_linear = E * (stretch - 1.0)

        rel_error = abs(sigma_analytical - sigma_linear) / abs(sigma_linear)
        assert rel_error < 0.01, f"Small strain error {rel_error:.4%} > 1%"

    def test_uniaxial_incompressible_medium_strain(self):
        """Medium strain: compare with known NH solution."""
        E = 1000.0
        nu = 0.4999
        mu = E / (2 * (1 + nu))

        for stretch in [1.05, 1.1, 1.2, 1.5, 2.0]:
            sigma = neo_hookean_uniaxial_incompressible(stretch, mu)
            # Analytical: sigma = mu * (lambda^2 - 1/lambda)
            sigma_exact = mu * (stretch**2 - 1.0/stretch)
            assert abs(sigma - sigma_exact) < 1e-10, \
                f"stretch={stretch}: {sigma} != {sigma_exact}"

    def test_uniaxial_compressible_vs_finite_strain_module(self):
        """Compressible NH from verification module vs finite_strain_elasticity."""
        E = 1000.0
        nu = 0.3  # compressible

        for stretch in [1.01, 1.05, 1.1, 1.2, 1.5]:
            sigma_verif, P_verif, l2_verif = neo_hookean_uniaxial_stress(stretch, E, nu)

            # Compute using finite_strain_elasticity module
            l1 = stretch
            l2 = l2_verif  # lateral stretch from verification
            F = torch.diag(torch.tensor([l1, l2, l2], dtype=DTYPE))
            _, _, sigma_tensor = neo_hookean_response(F, E, nu)

            sigma_ref = sigma_tensor[0, 0].item()
            rel_error = abs(sigma_verif - sigma_ref) / max(abs(sigma_ref), 1e-10)
            assert rel_error < 0.05, \
                f"stretch={stretch}: verification={sigma_verif:.6f}, ref={sigma_ref:.6f}, err={rel_error:.4%}"

    def test_uniaxial_stress_strain_curve_monotone(self):
        """Stress should increase monotonically with stretch."""
        E = 1000.0
        nu = 0.3
        stretches = [1.0 + 0.05*i for i in range(1, 20)]
        stresses = [neo_hookean_uniaxial_stress(s, E, nu)[0] for s in stretches]

        for i in range(1, len(stresses)):
            assert stresses[i] > stresses[i-1], \
                f"Non-monotone at stretch {stretches[i]}: {stresses[i]} <= {stresses[i-1]}"

    def test_uniaxial_zero_stretch(self):
        """At stretch=1, stress should be zero."""
        E = 1000.0
        nu = 0.3
        sigma, P, l2 = neo_hookean_uniaxial_stress(1.0, E, nu)
        assert abs(sigma) < 1e-10, f"sigma at stretch=1: {sigma}"
        assert abs(l2 - 1.0) < 1e-10, f"lateral stretch at stretch=1: {l2}"

    def test_uniaxial_error_vs_analytical_under_5pct(self):
        """Key acceptance criterion: stress-strain error < 5%."""
        E = 2000.0
        nu = 0.49  # nearly incompressible
        mu = E / (2 * (1 + nu))

        max_error = 0.0
        for stretch in [1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]:
            sigma_computed, _, _ = neo_hookean_uniaxial_stress(stretch, E, nu)
            sigma_exact = neo_hookean_uniaxial_incompressible(stretch, mu)
            if abs(sigma_exact) > 1e-10:
                rel_error = abs(sigma_computed - sigma_exact) / abs(sigma_exact)
                max_error = max(max_error, rel_error)

        assert max_error < 0.05, \
            f"Max relative error {max_error:.4%} exceeds 5% criterion"


class TestNeoHookeanBiaxial:
    """Neo-Hookean equi-biaxial tension verification."""

    def test_biaxial_incompressible(self):
        """Incompressible biaxial: sigma = mu*(lambda^2 - 1/lambda^4)."""
        E = 1000.0
        nu = 0.4999
        mu = E / (2 * (1 + nu))

        for stretch in [1.01, 1.05, 1.1, 1.2, 1.5]:
            sigma = neo_hookean_biaxial_incompressible(stretch, mu)
            sigma_exact = mu * (stretch**2 - 1.0/stretch**4)
            assert abs(sigma - sigma_exact) < 1e-10

    def test_biaxial_vs_finite_strain_module(self):
        """Biaxial from verification vs finite_strain_elasticity."""
        E = 1000.0
        nu = 0.3

        for stretch in [1.01, 1.05, 1.1, 1.2]:
            sigma_verif, P_verif, l3_verif = neo_hookean_biaxial_stress(stretch, E, nu)

            F = torch.diag(torch.tensor([stretch, stretch, l3_verif], dtype=DTYPE))
            _, _, sigma_tensor = neo_hookean_response(F, E, nu)

            sigma_ref = sigma_tensor[0, 0].item()
            rel_error = abs(sigma_verif - sigma_ref) / max(abs(sigma_ref), 1e-10)
            # Allow 10% tolerance for compressible case
            assert rel_error < 0.10, \
                f"stretch={stretch}: verification={sigma_verif:.6f}, ref={sigma_ref:.6f}, err={rel_error:.4%}"

    def test_biaxial_greater_than_uniaxial(self):
        """Biaxial stress should be greater than uniaxial at same stretch."""
        E = 1000.0
        nu = 0.3

        for stretch in [1.05, 1.1, 1.2]:
            sigma_uni = neo_hookean_uniaxial_stress(stretch, E, nu)[0]
            sigma_bi = neo_hookean_biaxial_stress(stretch, E, nu)[0]
            assert sigma_bi > sigma_uni, \
                f"stretch={stretch}: biaxial {sigma_bi} <= uniaxial {sigma_uni}"

    def test_biaxial_zero_stretch(self):
        """At stretch=1, biaxial stress should be zero."""
        E = 1000.0
        nu = 0.3
        sigma, P, l3 = neo_hookean_biaxial_stress(1.0, E, nu)
        assert abs(sigma) < 1e-10, f"sigma at stretch=1: {sigma}"
        assert abs(l3 - 1.0) < 1e-10, f"thickness stretch at stretch=1: {l3}"


# ============================================================================
# (b) Plastic Hinge Verification
# ============================================================================

class TestPlasticHinge:
    """Plastic limit load of simply supported beam."""

    def test_shape_factor_rectangular(self):
        """Shape factor for rectangular section = 1.5."""
        result = plastic_hinge_analysis(
            span=4.0, yield_stress=250e6, width=0.2, height=0.4
        )
        assert abs(result.shape_factor - 1.5) < 1e-10, \
            f"Shape factor {result.shape_factor} != 1.5"

    def test_plastic_moment(self):
        """Plastic moment M_pl = sigma_y * b * h^2 / 4."""
        sigma_y = 250e6
        b = 0.2
        h = 0.4
        result = plastic_hinge_analysis(span=4.0, yield_stress=sigma_y,
                                        width=b, height=h)
        M_pl_exact = sigma_y * b * h**2 / 4.0
        assert abs(result.plastic_moment - M_pl_exact) / M_pl_exact < 1e-10

    def test_plastic_limit_load(self):
        """P_pl = 4 * M_pl / L for simply supported beam with central load."""
        sigma_y = 250e6
        b = 0.2
        h = 0.4
        L = 4.0
        result = plastic_hinge_analysis(span=L, yield_stress=sigma_y,
                                        width=b, height=h)
        M_pl = sigma_y * b * h**2 / 4.0
        P_pl_exact = 4.0 * M_pl / L
        rel_error = abs(result.plastic_limit_load - P_pl_exact) / P_pl_exact
        assert rel_error < 0.10, \
            f"Plastic limit load error {rel_error:.4%} > 10%"

    def test_elastic_limit_less_than_plastic(self):
        """Elastic limit load < plastic limit load (shape factor > 1)."""
        result = plastic_hinge_analysis(span=4.0, yield_stress=250e6,
                                        width=0.2, height=0.4)
        assert result.elastic_limit_load < result.plastic_limit_load

    def test_plastic_limit_load_error_under_10pct(self):
        """Key acceptance criterion: P_pl error < 10%."""
        sigma_y = 350e6
        b = 0.15
        h = 0.3
        L = 3.0

        result = plastic_hinge_analysis(span=L, yield_stress=sigma_y,
                                        width=b, height=h)

        # Hand calculation
        Z = b * h**2 / 4.0
        M_pl = sigma_y * Z
        P_pl = 4.0 * M_pl / L

        rel_error = abs(result.plastic_limit_load - P_pl) / P_pl
        assert rel_error < 0.10, \
            f"Plastic limit load error {rel_error:.4%} exceeds 10%"


# ============================================================================
# (c) Taylor Bar Impact Verification
# ============================================================================

class TestTaylorBar:
    """Taylor bar impact verification."""

    def test_taylor_bar_length_ratio(self):
        """L_f/L_0 = exp(-rho*V_0^2/(2*sigma_y))."""
        result = taylor_bar_analytical(
            initial_velocity=200.0,  # m/s
            initial_length=0.01,     # 10 mm
            initial_diameter=0.005,  # 5 mm
            density=7800.0,          # steel
            dynamic_yield=1.0e9      # 1 GPa dynamic yield
        )

        alpha = 7800.0 * 200.0**2 / (2.0 * 1.0e9)
        Lf_L0_exact = math.exp(-alpha)

        rel_error = abs(result.final_length_ratio - Lf_L0_exact) / Lf_L0_exact
        assert rel_error < 0.08, \
            f"L_f/L_0 error {rel_error:.4%} > 8%"

    def test_taylor_bar_diameter_ratio(self):
        """D_f/D_0 = sqrt(L_0/L_f) from volume conservation."""
        result = taylor_bar_analytical(
            initial_velocity=200.0,
            initial_length=0.01,
            initial_diameter=0.005,
            density=7800.0,
            dynamic_yield=1.0e9
        )

        Df_D0_exact = math.sqrt(1.0 / result.final_length_ratio)
        rel_error = abs(result.final_diameter_ratio - Df_D0_exact) / Df_D0_exact
        assert rel_error < 0.10, \
            f"D_f/D_0 error {rel_error:.4%} > 10%"

    def test_taylor_bar_volume_conservation(self):
        """Volume should be approximately conserved."""
        result = taylor_bar_analytical(
            initial_velocity=150.0,
            initial_length=0.01,
            initial_diameter=0.005,
            density=8900.0,  # copper
            dynamic_yield=0.5e9
        )

        V_initial = math.pi / 4.0 * result.initial_diameter**2 * result.initial_length
        V_final = math.pi / 4.0 * result.final_diameter**2 * result.final_length

        rel_error = abs(V_final - V_initial) / V_initial
        assert rel_error < 0.01, \
            f"Volume conservation error {rel_error:.4%} > 1%"

    def test_taylor_bar_physical_bounds(self):
        """Final length < initial length, final diameter > initial diameter."""
        result = taylor_bar_analytical(
            initial_velocity=200.0,
            initial_length=0.01,
            initial_diameter=0.005,
            density=7800.0,
            dynamic_yield=1.0e9
        )

        assert result.final_length < result.initial_length
        assert result.final_diameter > result.initial_diameter
        assert result.final_length_ratio < 1.0
        assert result.final_diameter_ratio > 1.0

    def test_taylor_bar_numerical_runs(self):
        """Numerical 1D solution should run without error."""
        V0 = 150.0
        L0 = 0.01
        D0 = 0.005
        rho = 7800.0
        Y0 = 1.0e9
        E = 200e9

        result = taylor_bar_numerical(V0, L0, D0, rho, Y0, E,
                                      n_elements=50, n_steps=2000)

        # Basic sanity checks
        assert result.final_length > 0
        assert result.final_diameter > 0
        assert result.final_length_ratio > 0
        assert result.final_diameter_ratio > 0

    def test_taylor_bar_acceptance_criteria(self):
        """Key acceptance: D_f/D_0 error < 10%, L_f/L_0 error < 8%."""
        result = taylor_bar_analytical(
            initial_velocity=200.0,
            initial_length=0.01,
            initial_diameter=0.005,
            density=7800.0,
            dynamic_yield=1.0e9
        )

        # Reference values from Taylor's theory
        alpha = 7800.0 * 200.0**2 / (2.0 * 1.0e9)
        Lf_L0_ref = math.exp(-alpha)
        Df_D0_ref = math.sqrt(1.0 / Lf_L0_ref)

        Lf_error = abs(result.final_length_ratio - Lf_L0_ref) / Lf_L0_ref
        Df_error = abs(result.final_diameter_ratio - Df_D0_ref) / Df_D0_ref

        assert Lf_error < 0.08, f"L_f/L_0 error {Lf_error:.4%} > 8%"
        assert Df_error < 0.10, f"D_f/D_0 error {Df_error:.4%} > 10%"
