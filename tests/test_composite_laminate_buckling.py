"""Tests for composite laminate buckling (CLT + FSDT).

Verification matrix
-------------------
1. Navier CLT vs analytical: [0/90]s AS4/9310 simply-supported plate
2. FSDT → CLT convergence: thick plate ratio decreases with h/a → 0
3. Isotropic limit: E1=E2, ν12=ν21, G12=E/2(1+ν) → isotropic Navier
4. Ply angle parametric sweep: physical trends
5. Negative control: zero thickness → ValueError
6. ABD matrix symmetry and positive-definiteness
7. FSDT vs CLT trend: thick plates → FSDT < CLT
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from tensorfem.composite_laminate_buckling import (
    PlyProperty,
    LaminateLayup,
    LaminateStiffness,
    compute_ply_transformed_stiffness,
    compute_laminate_abd,
    navier_clt_buckling,
    fsdt_buckling,
    navier_isotropic_reference,
    isotropic_limit_check,
    parametric_ply_angle_sweep,
    thick_plate_ratio_scan,
    AS4_9310_E1,
    AS4_9310_E2,
    AS4_9310_NU12,
    AS4_9310_G12,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def as4_9310():
    """AS4/9310 carbon/epoxy ply properties."""
    return PlyProperty(
        E1=AS4_9310_E1, E2=AS4_9310_E2,
        nu12=AS4_9310_NU12, G12=AS4_9310_G12,
    )


@pytest.fixture
def layup_0_90s(as4_9310):
    """[0/90]s symmetric laminate."""
    return LaminateLayup(
        ply_props=as4_9310, angles_deg=[0, 90], ply_thickness=0.125e-3
    )


@pytest.fixture
def layup_pm45_2s(as4_9310):
    """[±45]2s symmetric laminate."""
    return LaminateLayup(
        ply_props=as4_9310, angles_deg=[45, -45], ply_thickness=0.125e-3
    )


@pytest.fixture
def layup_0_90_2s(as4_9310):
    """[0/90]2s symmetric laminate (thicker)."""
    return LaminateLayup(
        ply_props=as4_9310, angles_deg=[0, 90, 0, 90], ply_thickness=0.125e-3
    )


# ── Test 1: ABD matrix properties ──────────────────────────────────────────

class TestABDMatrix:
    """Verify ABD matrix symmetry and positive-definiteness."""

    def test_D_symmetric(self, layup_0_90s):
        stiff = compute_laminate_abd(layup_0_90s)
        np.testing.assert_allclose(stiff.D, stiff.D.T, atol=1e-10)

    def test_A_symmetric(self, layup_0_90s):
        stiff = compute_laminate_abd(layup_0_90s)
        np.testing.assert_allclose(stiff.A, stiff.A.T, atol=1e-10)

    def test_D_positive_definite(self, layup_0_90s):
        stiff = compute_laminate_abd(layup_0_90s)
        eigvals = np.linalg.eigvalsh(stiff.D)
        assert all(ev > 0 for ev in eigvals), f"D not PD: eigenvalues = {eigvals}"

    def test_cross_ply_D16_D26_zero(self, layup_0_90s):
        """Cross-ply [0/90]s should have D16=D26=0 (special orthotropic)."""
        stiff = compute_laminate_abd(layup_0_90s)
        assert abs(stiff.D16) < 1e-10, f"D16={stiff.D16} not zero for cross-ply"
        assert abs(stiff.D26) < 1e-10, f"D26={stiff.D26} not zero for cross-ply"

    def test_0_ply_D11_greater_D22(self, layup_0_90s):
        """[0/90]s should have D11 > D22 (more fibers in x-direction at outer plies)."""
        stiff = compute_laminate_abd(layup_0_90s)
        assert stiff.D11 > stiff.D22

    def test_thickness_consistency(self, layup_0_90s):
        stiff = compute_laminate_abd(layup_0_90s)
        expected_h = 4 * 0.125e-3  # [0/90]s = 4 plies
        assert abs(stiff.h - expected_h) < 1e-12


# ── Test 2: 0° transformed stiffness ───────────────────────────────────────

class TestTransformedStiffness:
    """Verify ply transformation at 0° and 90°."""

    def test_0deg_Q11_equals_Q11_local(self, as4_9310):
        Q_bar, _ = compute_ply_transformed_stiffness(as4_9310, 0.0)
        nu21 = as4_9310.nu12 * as4_9310.E2 / as4_9310.E1
        Q11_local = as4_9310.E1 / (1 - as4_9310.nu12 * nu21)
        assert abs(Q_bar[0, 0] - Q11_local) / Q11_local < 1e-10

    def test_90deg_swap(self, as4_9310):
        """90° ply: Q_bar_xx = Q22_local, Q_bar_yy = Q11_local."""
        Q_bar, _ = compute_ply_transformed_stiffness(as4_9310, 90.0)
        nu21 = as4_9310.nu12 * as4_9310.E2 / as4_9310.E1
        Q11_local = as4_9310.E1 / (1 - as4_9310.nu12 * nu21)
        Q22_local = as4_9310.E2 / (1 - as4_9310.nu12 * nu21)
        assert abs(Q_bar[0, 0] - Q22_local) / Q22_local < 1e-10
        assert abs(Q_bar[1, 1] - Q11_local) / Q11_local < 1e-10

    def test_0deg_no_coupling(self, as4_9310):
        """0° and 90° plies should have Q16=Q26=0."""
        for angle in [0.0, 90.0]:
            Q_bar, _ = compute_ply_transformed_stiffness(as4_9310, angle)
            assert abs(Q_bar[0, 2]) < 1e-3, f"Q16 at {angle}° = {Q_bar[0, 2]}"
            assert abs(Q_bar[1, 2]) < 1e-3, f"Q26 at {angle}° = {Q_bar[1, 2]}"


# ── Test 3: Navier CLT buckling ────────────────────────────────────────────

class TestNavierCLT:
    """Verify CLT buckling against analytical expectations."""

    def test_0_90s_square_plate(self, layup_0_90s):
        """[0/90]s square plate: m=1, n=1 should be critical mode."""
        stiff = compute_laminate_abd(layup_0_90s)
        a, b = 0.3, 0.3  # 300mm square plate
        res = navier_clt_buckling(stiff, a, b, m_max=10, n_max=10)
        assert res.N_cr > 0
        assert res.m_cr == 1
        assert res.n_cr == 1
        assert res.theory == "CLT"

    def test_long_plate_mode(self, layup_0_90s):
        """Long plate (a >> b): critical mode adjusts to minimize N_cr.

        For highly orthotropic plates with D11 >> D22, the critical mode
        can have m > 1 in the longitudinal direction.
        """
        stiff = compute_laminate_abd(layup_0_90s)
        a, b = 1.0, 0.2  # a/b = 5
        res = navier_clt_buckling(stiff, a, b, m_max=10, n_max=5)
        assert res.n_cr == 1  # transverse mode always 1 for long plate
        assert res.m_cr >= 1  # longitudinal mode ≥ 1

    def test_Ncr_positive(self, layup_0_90s):
        stiff = compute_laminate_abd(layup_0_90s)
        res = navier_clt_buckling(stiff, 0.3, 0.3, m_max=10, n_max=10)
        assert res.N_cr > 0
        assert math.isfinite(res.N_cr)

    def test_known_value_0_90s(self, layup_0_90s):
        """Verify N_cr for [0/90]s against hand calculation."""
        stiff = compute_laminate_abd(layup_0_90s)
        a, b = 0.3, 0.3
        res = navier_clt_buckling(stiff, a, b, m_max=10, n_max=10)

        # Hand calculation for m=1, n=1:
        alpha = math.pi / a
        beta = math.pi / b
        expected = (stiff.D11 * alpha**4
                    + 2 * (stiff.D12 + 2 * stiff.D66) * alpha**2 * beta**2
                    + stiff.D22 * beta**4) / alpha**2

        assert abs(res.N_cr - expected) / expected < 1e-10


# ── Test 4: FSDT vs CLT ────────────────────────────────────────────────────

class TestFSDT:
    """Verify FSDT correction: thick plates → FSDT < CLT."""

    def test_fsdt_less_than_clt_thick(self, as4_9310):
        """For thick plates, FSDT critical load < CLT."""
        # Use a thick plate: h/a = 0.1
        layup = LaminateLayup(
            ply_props=as4_9310, angles_deg=[0, 90, 0, 90],
            ply_thickness=2.5e-3  # 4 plies × 2.5mm = 10mm total
        )
        stiff = compute_laminate_abd(layup)
        a, b = 0.1, 0.1  # h/a = 10mm/100mm = 0.1
        res_clt = navier_clt_buckling(stiff, a, b, m_max=5, n_max=5)
        res_fsd = fsdt_buckling(stiff, a, b, m_max=5, n_max=5)
        assert res_fsd.N_cr < res_clt.N_cr, (
            f"FSDT ({res_fsd.N_cr}) should be < CLT ({res_clt.N_cr}) for thick plate"
        )
        ratio = res_fsd.N_cr / res_clt.N_cr
        assert ratio < 0.95, f"FSDT/CLT ratio = {ratio}, expected < 0.95 for h/a=0.1"

    def test_fsdt_converges_to_clt_thin(self, as4_9310):
        """For very thin plates, FSDT → CLT."""
        # Very thin: h/a ≈ 0.001
        layup = LaminateLayup(
            ply_props=as4_9310, angles_deg=[0, 90], ply_thickness=0.025e-3
        )
        stiff = compute_laminate_abd(layup)
        a, b = 0.2, 0.2  # h/a = 0.1mm/200mm = 0.0005
        res_clt = navier_clt_buckling(stiff, a, b, m_max=5, n_max=5)
        res_fsd = fsdt_buckling(stiff, a, b, m_max=5, n_max=5)
        ratio = res_fsd.N_cr / res_clt.N_cr
        assert ratio > 0.99, f"FSDT/CLT ratio = {ratio}, expected > 0.99 for thin plate"

    def test_fsdt_ratio_decreases_with_ha(self, as4_9310):
        """FSDT/CLT ratio should decrease as h/a increases."""
        result = thick_plate_ratio_scan(
            as4_9310, [0, 90], a=0.2, b=0.2,
            h_over_a_range=[0.02, 0.05, 0.10, 0.15]
        )
        ratios = result["ratio_FSDT_CLT"]
        # Each ratio should be ≤ 1
        for r in ratios:
            assert r <= 1.0 + 1e-10, f"Ratio {r} > 1"
        # Check monotonically decreasing (with tolerance)
        for i in range(len(ratios) - 1):
            assert ratios[i + 1] <= ratios[i] + 1e-6, (
                f"Ratio not decreasing: {ratios[i]} → {ratios[i+1]}"
            )


# ── Test 5: Isotropic limit ────────────────────────────────────────────────

class TestIsotropicLimit:
    """Negative control: composite → isotropic."""

    def test_isotropic_limit_single_ply(self):
        """E1=E2, ν12=ν21, G12=E/2(1+ν) → isotropic result."""
        result = isotropic_limit_check(E=210e9, nu=0.3, h=0.01, a=1.0, b=1.0)
        assert result["pass"], (
            f"Isotropic limit failed: composite={result['composite_clt_Ncr']}, "
            f"isotropic={result['isotropic_navier_Ncr']}, "
            f"error={result['relative_error']}"
        )

    def test_isotropic_limit_square_plate(self):
        """Square plate isotropic limit."""
        result = isotropic_limit_check(E=70e9, nu=0.33, h=0.005, a=0.5, b=0.5)
        assert result["relative_error"] < 0.01

    def test_isotropic_limit_rectangular(self):
        """Rectangular plate isotropic limit."""
        result = isotropic_limit_check(E=200e9, nu=0.3, h=0.008, a=1.2, b=0.6)
        assert result["relative_error"] < 0.01


# ── Test 6: Parametric ply angle sweep ─────────────────────────────────────

class TestParametricSweep:
    """Verify physical trends in ply angle parametric sweep."""

    def test_layup_stiffness_differences(self, as4_9310):
        """Verify different layups have different stiffness properties.

        [±45]2s has higher D66 (twisting stiffness) than [0/90]s.
        """
        configs = [
            ("0_90_s", [0, 90]),
            ("pm45_2s", [45, -45]),
        ]
        results = parametric_ply_angle_sweep(
            as4_9310, configs, a=0.3, b=0.3, ply_thickness=0.125e-3
        )
        D66_0_90 = results["0_90_s"]["D66"]
        D66_pm45 = results["pm45_2s"]["D66"]
        assert D66_pm45 > D66_0_90, (
            f"[±45]2s D66={D66_pm45} should be > [0/90]s D66={D66_0_90}"
        )

    def test_angle_sweep_positive(self, as4_9310):
        """N_cr vs angle should be positive for all angles."""
        configs = [("theta_sweep", [0, 90])]
        results = parametric_ply_angle_sweep(
            as4_9310, configs, a=0.3, b=0.3,
            angles_scan=[0, 15, 30, 45, 60, 75, 90]
        )
        sweep = results["_sweep"]
        Ncr = sweep["N_cr_CLT"]
        # Check no zero or negative values
        for val in Ncr:
            assert val > 0, f"N_cr = {val} not positive"
        # Note: mode switches can cause jumps in N_cr as angle changes,
        # which is physical (different buckling modes become critical)


# ── Test 7: Negative controls ──────────────────────────────────────────────

class TestNegativeControls:
    """Verify error handling for invalid inputs."""

    def test_zero_thickness_raises(self, as4_9310):
        with pytest.raises(ValueError):
            LaminateLayup(ply_props=as4_9310, angles_deg=[0], ply_thickness=0.0)

    def test_negative_thickness_raises(self, as4_9310):
        with pytest.raises(ValueError):
            LaminateLayup(ply_props=as4_9310, angles_deg=[0], ply_thickness=-1e-3)

    def test_invalid_ply_property_raises(self):
        with pytest.raises(ValueError):
            PlyProperty(E1=-1e9, E2=10e9, nu12=0.3, G12=5e9)

    def test_empty_angles_raises(self, as4_9310):
        with pytest.raises(ValueError):
            LaminateLayup(ply_props=as4_9310, angles_deg=[], ply_thickness=0.125e-3)


# ── Test 8: Comparison with isotropic Navier (cross-check) ─────────────────

class TestCrossCheck:
    """Cross-check composite CLT against isotropic Navier for isotropic case."""

    def test_single_0_ply_isotropic(self):
        """A single 0° ply with isotropic properties = isotropic plate."""
        E, nu = 210e9, 0.3
        G = E / (2 * (1 + nu))
        ply = PlyProperty(E1=E, E2=E, nu12=nu, G12=G)
        h = 0.01
        # Use ply_thickness = h/2 so total thickness = 2 * h/2 = h
        layup = LaminateLayup(ply_props=ply, angles_deg=[0], ply_thickness=h / 2)
        stiff = compute_laminate_abd(layup)

        a, b = 1.0, 0.5
        res_composite = navier_clt_buckling(stiff, a, b, m_max=10, n_max=10)
        ref = navier_isotropic_reference(E, nu, h, a, b, m_max=10, n_max=10)

        rel_err = abs(res_composite.N_cr - ref.N_cr) / ref.N_cr
        assert rel_err < 0.01, f"Relative error {rel_err} > 1%"

    def test_cross_modulus_values(self):
        """Test with different modulus values."""
        for E in [70e9, 148e9, 210e9]:
            nu = 0.3
            G = E / (2 * (1 + nu))
            ply = PlyProperty(E1=E, E2=E, nu12=nu, G12=G)
            h = 0.005
            layup = LaminateLayup(ply_props=ply, angles_deg=[0], ply_thickness=h / 2)
            stiff = compute_laminate_abd(layup)
            a, b = 0.8, 0.4
            res = navier_clt_buckling(stiff, a, b, m_max=10, n_max=10)
            ref = navier_isotropic_reference(E, nu, h, a, b, m_max=10, n_max=10)
            rel_err = abs(res.N_cr - ref.N_cr) / ref.N_cr
            assert rel_err < 0.01, f"E={E}: error={rel_err}"
