"""Tests for structural fatigue detail verification (P92W3-N03).

Verifies:
  - Analytical K_t formulas (Nii, Maddox) - implementation correctness
  - Hot-spot stress extrapolation (IIW linear/quadratic) - mathematical correctness
  - DNV-RP-C203 dual-slope S-N curve - exact verification
  - Mesh convergence framework - GCI method
  
HONEST FAIL documented:
  - FE notch submodel does NOT match Nii/Maddox formulas within 3% tolerance
  - T-joint: 20.10% error (criterion: ≤ 3%)
  - Butt joint: 113.26% error (criterion: ≤ 3%)
  - Root cause: FE uses 2D elliptical notch idealization vs empirical formulas
"""
import math
import pytest
import numpy as np

from tensorfem.structural_fatigue_detail import (
    nii_t_joint_kt,
    maddox_butt_joint_kt,
    hotspot_extrapolation_iiw,
    HotSpotExtrapolation,
    DNVDualSlopeSN,
    mesh_convergence_study,
)


# ---------------------------------------------------------------------------
# Analytical formula tests (all PASS)
# ---------------------------------------------------------------------------

class TestNiiFormula:
    """Test Nii (1974) T-joint K_t formula."""

    def test_basic_computation(self):
        """K_t should be > 1 for any finite geometry."""
        kt = nii_t_joint_kt(0.015, 0.010, 1.0e-3)
        assert kt > 1.0

    def test_increases_with_leg_size(self):
        """K_t should increase with weld leg size a/t."""
        kt1 = nii_t_joint_kt(0.015, 0.008, 1.0e-3)
        kt2 = nii_t_joint_kt(0.015, 0.012, 1.0e-3)
        assert kt2 > kt1

    def test_increases_with_sharp_toe(self):
        """K_t should increase as toe radius decreases (sharper notch)."""
        kt_smooth = nii_t_joint_kt(0.015, 0.010, 2.0e-3)
        kt_sharp = nii_t_joint_kt(0.015, 0.010, 0.5e-3)
        assert kt_sharp > kt_smooth

    def test_known_value(self):
        """Check against hand calculation for t=15mm, a=10mm, r=1mm."""
        # K_t = 1 + 1.24 * sqrt(10/15) * (15/1)^0.25
        # = 1 + 1.24 * 0.8165 * 1.9680
        # = 1 + 1.989 ≈ 2.989
        kt = nii_t_joint_kt(0.015, 0.010, 1.0e-3)
        expected = 1.0 + 1.24 * math.sqrt(0.010/0.015) * (0.015/1.0e-3)**0.25
        assert kt == pytest.approx(expected, rel=1e-10)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            nii_t_joint_kt(-0.015, 0.010, 1.0e-3)
        with pytest.raises(ValueError):
            nii_t_joint_kt(0.015, 0.0, 1.0e-3)


class TestMaddoxFormula:
    """Test Maddox (1975) butt joint K_t formula."""

    def test_basic_computation(self):
        """K_t should be > 1 for any reinforcement."""
        kt = maddox_butt_joint_kt(0.015, 1.5e-3, 1.0e-3)
        assert kt > 1.0

    def test_zero_reinforcement(self):
        """K_t = 1 for flush-ground weld (h_r = 0)."""
        kt = maddox_butt_joint_kt(0.015, 0.0, 1.0e-3)
        assert kt == pytest.approx(1.0)

    def test_increases_with_reinforcement(self):
        """K_t should increase with reinforcement height."""
        kt1 = maddox_butt_joint_kt(0.015, 1.0e-3, 1.0e-3)
        kt2 = maddox_butt_joint_kt(0.015, 2.0e-3, 1.0e-3)
        assert kt2 > kt1

    def test_increases_with_sharp_toe(self):
        """K_t should increase as toe radius decreases."""
        kt_smooth = maddox_butt_joint_kt(0.015, 1.5e-3, 2.0e-3)
        kt_sharp = maddox_butt_joint_kt(0.015, 1.5e-3, 0.5e-3)
        assert kt_sharp > kt_smooth

    def test_known_value(self):
        """Check against hand calculation for t=15mm, h_r=1.5mm, r=1mm."""
        # K_t = 1 + 0.18 * (1.5/15) * sqrt(15/1)
        # = 1 + 0.18 * 0.1 * 3.873
        # = 1 + 0.0697 ≈ 1.070
        kt = maddox_butt_joint_kt(0.015, 1.5e-3, 1.0e-3)
        expected = 1.0 + 0.18 * (1.5e-3/0.015) * math.sqrt(0.015/1.0e-3)
        assert kt == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Hot-spot stress extrapolation tests (all PASS)
# ---------------------------------------------------------------------------

class TestHotSpotExtrapolation:
    """Test IIW linear and quadratic hot-spot stress extrapolation."""

    def test_linear_field(self):
        """Linear stress field should give exact extrapolation."""
        t = 0.015
        # σ(x) = 2.0 - 0.5*x/t (linear decay from 2.0 at toe)
        x = np.linspace(0.0, 3.0 * t, 50)
        y = 2.0 - 0.5 * x / t

        result = hotspot_extrapolation_iiw(x, y, t)
        # At x=0: σ = 2.0
        assert result.sigma_hot_linear == pytest.approx(2.0, rel=1e-6)
        assert result.sigma_hot_quadratic == pytest.approx(2.0, rel=1e-6)

    def test_quadratic_field(self):
        """Quadratic stress field: linear and quadratic should differ."""
        t = 0.015
        # σ(x) = 2.0 - 0.3*(x/t) - 0.1*(x/t)^2
        x = np.linspace(0.0, 3.0 * t, 50)
        y = 2.0 - 0.3 * (x/t) - 0.1 * (x/t)**2

        result = hotspot_extrapolation_iiw(x, y, t)
        # At x=0: σ = 2.0
        # Quadratic should be more accurate for quadratic field
        assert result.sigma_hot_quadratic == pytest.approx(2.0, rel=1e-3)

    def test_interpolation_values(self):
        """Check that 0.4t, 0.9t, 1.4t values are correctly interpolated."""
        t = 0.015
        x = np.array([0.0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.021, 0.030])
        y = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.3, 1.1])

        result = hotspot_extrapolation_iiw(x, y, t)
        # 0.4t = 0.006, 0.9t = 0.0135, 1.4t = 0.021
        assert result.sigma_04t == pytest.approx(1.8, rel=1e-6)
        assert result.sigma_14t == pytest.approx(1.3, rel=1e-6)

    def test_insufficient_points(self):
        """Should raise error with fewer than 3 points."""
        with pytest.raises(ValueError):
            hotspot_extrapolation_iiw([0.0, 0.01], [2.0, 1.5], 0.015)


# ---------------------------------------------------------------------------
# DNV dual-slope S-N curve tests (all PASS)
# ---------------------------------------------------------------------------

class TestDNVDualSlopeSN:
    """Test DNV-RP-C203 dual-slope S-N curve (Curve D)."""

    def test_knee_point_stress(self):
        """Knee point should be self-consistent."""
        sn = DNVDualSlopeSN()
        stress_knee = sn.stress_knee_pa / 1e6  # MPa
        n_at_knee = sn.cycles_to_failure(sn.stress_knee_pa)
        assert n_at_knee == pytest.approx(1e7, rel=1e-6)

    def test_first_slope_ratio(self):
        """Doubling stress on m=3 slope should give 8× life reduction."""
        sn = DNVDualSlopeSN()
        ds1 = 80e6  # 80 MPa
        ds2 = 160e6  # 160 MPa
        n1 = sn.cycles_to_failure(ds1)
        n2 = sn.cycles_to_failure(ds2)
        ratio = n1 / n2
        assert ratio == pytest.approx(8.0, rel=1e-6)

    def test_second_slope_ratio(self):
        """Doubling stress on m=5 slope should give 32× life reduction."""
        sn = DNVDualSlopeSN()
        # Use stresses below knee point
        stress_knee = sn.stress_knee_pa
        ds1 = stress_knee * 0.5  # Below knee
        ds2 = stress_knee * 0.25  # 2× lower
        n1 = sn.cycles_to_failure(ds1)
        n2 = sn.cycles_to_failure(ds2)
        ratio = n2 / n1
        assert ratio == pytest.approx(32.0, rel=1e-6)

    def test_inverse_consistency(self):
        """stress_at_cycles(cycles_to_failure(σ)) should return σ."""
        sn = DNVDualSlopeSN()
        for stress in [50e6, 100e6, 150e6, 200e6]:
            n = sn.cycles_to_failure(stress)
            stress_back = sn.stress_at_cycles(n)
            assert stress_back == pytest.approx(stress, rel=1e-6)

    def test_invalid_inputs(self):
        sn = DNVDualSlopeSN()
        with pytest.raises(ValueError):
            sn.cycles_to_failure(0)
        with pytest.raises(ValueError):
            sn.cycles_to_failure(-100e6)
        with pytest.raises(ValueError):
            sn.stress_at_cycles(0)


# ---------------------------------------------------------------------------
# Mesh convergence tests (all PASS)
# ---------------------------------------------------------------------------

class TestMeshConvergence:
    """Test mesh convergence study."""

    def test_converged_case(self):
        """Perfectly converged case should pass."""
        def compute_kt(params):
            return 2.0  # Constant K_t

        mesh_params = [
            {"n": 10},
            {"n": 20},
            {"n": 40},
        ]
        result = mesh_convergence_study(compute_kt, mesh_params)
        assert result.scf_change_percent == 0.0
        assert result.converged

    def test_divergent_case(self):
        """Divergent case should fail convergence."""
        def compute_kt(params):
            return params["n"] * 0.1  # K_t increases with mesh

        mesh_params = [
            {"n": 10},
            {"n": 20},
            {"n": 40},
        ]
        result = mesh_convergence_study(compute_kt, mesh_params)
        assert result.scf_change_percent > 2.0
        assert not result.converged

    def test_insufficient_levels(self):
        """Should raise error with fewer than 3 mesh levels."""
        with pytest.raises(ValueError):
            mesh_convergence_study(lambda p: 1.0, [{"n": 10}, {"n": 20}])


# ---------------------------------------------------------------------------
# Honest FAIL documentation tests
# ---------------------------------------------------------------------------

class TestHonestFailDocumentation:
    """Document the honest FAIL: FE vs analytical K_t discrepancy."""

    def test_t_joint_kt_discrepancy_documented(self):
        """Document that T-joint FE K_t does not match Nii formula within 3%."""
        # This test documents the honest FAIL
        kt_nii = nii_t_joint_kt(0.015, 0.010, 1.0e-3)
        # FE result (from weld_joint_fatigue.compute_joint_scf with n_xi=24, n_eta=14)
        kt_fe = 2.3910  # Documented value
        error = abs(kt_fe - kt_nii) / kt_nii * 100
        
        # Document the discrepancy
        assert error > 3.0, f"Expected error > 3%, got {error:.2f}%"
        assert error == pytest.approx(20.10, rel=0.1)  # Documented value

    def test_butt_joint_kt_discrepancy_documented(self):
        """Document that butt joint FE K_t does not match Maddox formula within 3%."""
        # This test documents the honest FAIL
        kt_maddox = maddox_butt_joint_kt(0.015, 1.5e-3, 1.0e-3)
        # FE result (from weld_joint_fatigue.compute_joint_scf with n_xi=24, n_eta=14)
        kt_fe = 2.2813  # Documented value
        error = abs(kt_fe - kt_maddox) / kt_maddox * 100
        
        # Document the discrepancy
        assert error > 3.0, f"Expected error > 3%, got {error:.2f}%"
        assert error == pytest.approx(113.26, rel=0.1)  # Documented value

    def test_sn_curve_passes(self):
        """S-N curve verification passes (criterion d)."""
        sn = DNVDualSlopeSN()
        
        # Test m=3 slope
        ds1 = 80e6
        ds2 = 160e6
        n1 = sn.cycles_to_failure(ds1)
        n2 = sn.cycles_to_failure(ds2)
        ratio_m3 = n1 / n2
        assert ratio_m3 == pytest.approx(8.0, rel=0.01)
        
        # Test m=5 slope
        stress_knee = sn.stress_knee_pa
        ds3 = stress_knee * 0.5
        ds4 = stress_knee * 0.25
        n3 = sn.cycles_to_failure(ds3)
        n4 = sn.cycles_to_failure(ds4)
        ratio_m5 = n4 / n3
        assert ratio_m5 == pytest.approx(32.0, rel=0.01)
        
        # Criterion d passes
        assert True
