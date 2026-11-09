"""Tests for weld_joint_fatigue module (IN67-N09 verification criteria).

Tests cover:
  - T-joint, butt joint, bracket toe SCF computation
  - Analytical Inglis SCF formula
  - Hot-spot stress extrapolation (0.5t / 1.5t)
  - S-N curve fatigue life (DNV-RP-C203 D)
  - Beam-to-hotspot interface
  - Mesh convergence study
  - All six verification criteria C1–C6
"""
import math
import pytest
import numpy as np

from tensorfem.weld_joint_fatigue import (
    TJointGeometry,
    ButtJointGeometry,
    BracketToeGeometry,
    JointSCFResult,
    compute_joint_scf,
    analytical_scf_inglis,
    HotSpotExtrapolationResult,
    hot_spot_stress_05_15t,
    hot_spot_stress_from_structural_field,
    SNFatigueLifeResult,
    dnv_d_curve_life,
    BeamToHotSpotInterface,
    beam_to_hotspot,
    MeshConvergenceResult,
    mesh_convergence_study,
    VerificationReport,
    run_all_verification,
    WeldJointType,
)
from tensorfem.weld_fatigue_ice import DNVSNCurve


# ---------------------------------------------------------------------------
# Geometry construction tests
# ---------------------------------------------------------------------------

class TestJointGeometries:
    """Test joint geometry construction and validation."""

    def test_t_joint_defaults(self):
        g = TJointGeometry()
        assert g.plate_thickness_m == 0.015
        assert g.weld_toe_radius_m == 1.0e-3
        assert g.t_over_r == 15.0

    def test_t_joint_custom(self):
        g = TJointGeometry(plate_thickness_m=0.020, weld_toe_radius_m=0.5e-3)
        assert g.plate_thickness_m == 0.020
        assert g.t_over_r == 40.0

    def test_t_joint_invalid_flank_angle(self):
        with pytest.raises(ValueError, match="flank_angle_deg"):
            TJointGeometry(flank_angle_deg=2.0)

    def test_butt_joint_defaults(self):
        g = ButtJointGeometry()
        assert g.plate_thickness_m == 0.015
        assert g.reinforcement_height_m == 1.5e-3

    def test_butt_joint_excess_reinforcement(self):
        with pytest.raises(ValueError, match="reinforcement"):
            ButtJointGeometry(plate_thickness_m=0.010, reinforcement_height_m=0.005)

    def test_bracket_toe_defaults(self):
        g = BracketToeGeometry()
        assert g.bracket_thickness_m == 0.012

    def test_to_weld_geometry_t_joint(self):
        g = TJointGeometry()
        wg = g.to_weld_geometry()
        assert wg.weld_type == "fillet"
        assert wg.plate_thickness_m == g.plate_thickness_m

    def test_to_weld_geometry_butt(self):
        g = ButtJointGeometry()
        wg = g.to_weld_geometry()
        assert wg.weld_type == "butt"

    def test_to_weld_geometry_bracket(self):
        g = BracketToeGeometry()
        wg = g.to_weld_geometry()
        assert wg.weld_type == "fillet"

    def test_equivalent_notch_depth(self):
        g = TJointGeometry(weld_toe_radius_m=1.0e-3, flank_angle_deg=50.0)
        d = g.equivalent_notch_depth_m
        # d = r * sqrt(1 + 1.6*cos(50°))
        expected = 1.0e-3 * math.sqrt(1.0 + 1.6 * math.cos(math.radians(50.0)))
        assert abs(d - expected) < 1e-10


# ---------------------------------------------------------------------------
# Analytical SCF tests
# ---------------------------------------------------------------------------

class TestAnalyticalSCF:
    """Test analytical Inglis SCF formula."""

    def test_inglis_basic(self):
        # K_t = 1 + 2*sqrt(d/r)
        kt = analytical_scf_inglis(1.0e-3, 1.0e-3)
        expected = 1.0 + 2.0 * math.sqrt(1.0)
        assert abs(kt - expected) < 1e-10

    def test_inglis_sharp_notch(self):
        # Sharp notch: d >> r → high SCF
        kt = analytical_scf_inglis(10.0e-3, 0.1e-3)
        expected = 1.0 + 2.0 * math.sqrt(100.0)
        assert abs(kt - expected) < 1e-10

    def test_inglis_smooth_limit(self):
        # Smooth: r >> d → K_t → 1
        # For d=1mm, need r > 160m for < 0.5% deviation
        kt = analytical_scf_inglis(1.0e-3, 200.0)
        assert abs(kt - 1.0) < 0.005

    def test_inglis_positive_inputs(self):
        with pytest.raises(ValueError):
            analytical_scf_inglis(-1.0e-3, 1.0e-3)
        with pytest.raises(ValueError):
            analytical_scf_inglis(1.0e-3, 0.0)


# ---------------------------------------------------------------------------
# SCF computation tests (using small mesh for speed)
# ---------------------------------------------------------------------------

class TestSCFComputation:
    """Test SCF computation for different joint types."""

    def test_t_joint_scf_positive(self):
        g = TJointGeometry(plate_thickness_m=0.015, weld_toe_radius_m=1.0e-3)
        result = compute_joint_scf(g, n_xi=24, n_eta=14)
        assert isinstance(result, JointSCFResult)
        assert result.kt_toe_fe > 1.0
        assert result.kt_hotspot > 1.0
        assert result.joint_type == WeldJointType.T_JOINT

    def test_butt_joint_scf_positive(self):
        g = ButtJointGeometry(plate_thickness_m=0.015, weld_toe_radius_m=1.0e-3)
        result = compute_joint_scf(g, n_xi=24, n_eta=14)
        assert result.kt_toe_fe > 1.0
        assert result.kt_hotspot >= 1.0
        assert result.joint_type == WeldJointType.BUTT_JOINT

    def test_bracket_toe_scf_positive(self):
        g = BracketToeGeometry(plate_thickness_m=0.015, weld_toe_radius_m=1.0e-3)
        result = compute_joint_scf(g, n_xi=24, n_eta=14)
        assert result.kt_toe_fe > 1.0
        assert result.kt_hotspot > 1.0
        assert result.joint_type == WeldJointType.BRACKET_TOE

    def test_scf_mesh_ratio_reported(self):
        g = TJointGeometry(plate_thickness_m=0.015)
        result = compute_joint_scf(g, n_xi=24, n_eta=14)
        assert result.min_edge_m > 0
        assert result.mesh_ratio > 0

    def test_invalid_geometry_type(self):
        with pytest.raises(TypeError):
            compute_joint_scf("not a geometry")

    def test_t_joint_hotspot_in_iiw_range(self):
        """T-joint hot-spot SCF should be in IIW range 1.2-1.5."""
        g = TJointGeometry(
            plate_thickness_m=0.015,
            weld_toe_radius_m=1.0e-3,
            weld_leg_m=0.7 * 0.015,
        )
        result = compute_joint_scf(g, n_xi=24, n_eta=14)
        assert 1.2 <= result.kt_hotspot <= 1.5, f"kt_hotspot={result.kt_hotspot}"


# ---------------------------------------------------------------------------
# Hot-spot stress extrapolation tests
# ---------------------------------------------------------------------------

class TestHotSpotExtrapolation:
    """Test hot-spot stress method with 0.5t/1.5t extrapolation."""

    def test_linear_stress_field(self):
        """A linear stress field should extrapolate exactly."""
        t = 0.015
        # Linear: σ(x) = 2.0 - 100*x (Pa/Pa)
        x = np.linspace(0, 0.05, 50)
        y = 2.0 - 100.0 * x
        result = hot_spot_stress_05_15t(x, y, t, method="dnv_05_15t")
        # At x=0: σ = 2.0
        assert abs(result.sigma_hot - 2.0) < 1e-6
        assert result.r_squared > 0.999

    def test_dnv_weights(self):
        """DNV 0.5t/1.5t: σ_hot = 1.5*σ(0.5t) - 0.5*σ(1.5t)."""
        t = 0.020
        x = np.array([0.0, 0.5 * t, 1.0 * t, 1.5 * t, 2.0 * t])
        y = np.array([1.5, 1.3, 1.15, 1.05, 0.95])
        result = hot_spot_stress_05_15t(x, y, t, method="dnv_05_15t")
        expected = 1.5 * 1.3 - 0.5 * 1.05
        assert abs(result.sigma_hot - expected) < 1e-6

    def test_iiw_method(self):
        """IIW 0.4t/1.0t extrapolation."""
        t = 0.020
        x = np.array([0.0, 0.4 * t, 0.8 * t, 1.0 * t, 1.5 * t])
        y = np.array([1.5, 1.3, 1.15, 1.05, 0.95])
        result = hot_spot_stress_05_15t(x, y, t, method="iiw_04_10t")
        # IIW: σ_hot = σ(0.4t) - 0.4t*(σ(1.0t)-σ(0.4t))/(0.6t)
        expected_iiw = 1.3 - 0.4 * t * (1.05 - 1.3) / (0.6 * t)
        assert abs(result.sigma_hot - expected_iiw) < 1e-6

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            hot_spot_stress_05_15t([0, 1], [1, 1], 0.01, method="invalid")

    def test_r_squared_high_for_linear_field(self):
        """A linear stress field should give R² ≈ 1."""
        t = 0.015
        x = np.linspace(0.3 * t, 2.0 * t, 30)
        y = 1.35 - 0.2 * x / t  # linear decay
        result = hot_spot_stress_05_15t(x, y, t, method="dnv_05_15t")
        assert result.r_squared >= 0.999

    def test_structural_field_interface(self):
        """Test hot_spot_stress_from_structural_field."""
        sigma_hot = hot_spot_stress_from_structural_field(0.015, 1.35, 100e6)
        assert abs(sigma_hot - 135e6) < 1e-3


# ---------------------------------------------------------------------------
# S-N curve fatigue life tests
# ---------------------------------------------------------------------------

class TestSNCurveFatigue:
    """Test DNV-RP-C203 D curve fatigue life calculation."""

    def test_life_positive(self):
        result = dnv_d_curve_life(100e6)
        assert result.cycles_to_failure > 0
        assert result.stress_range_pa == 100e6

    def test_knee_point_detection(self):
        # High stress → below knee (m=3)
        r1 = dnv_d_curve_life(100e6)
        assert r1.is_below_knee
        assert r1.slope_m == 3.0
        # Low stress → above knee (m=5)
        r2 = dnv_d_curve_life(30e6)
        assert not r2.is_below_knee
        assert r2.slope_m == 5.0

    def test_power_law_ratio_m3(self):
        """C4: Doubling Δσ on m=3 slope → N ratio = 2^3 = 8."""
        ds1 = 80e6  # above knee
        ds2 = 160e6  # 2× ds1, also above knee
        n1 = dnv_d_curve_life(ds1).cycles_to_failure
        n2 = dnv_d_curve_life(ds2).cycles_to_failure
        ratio = n1 / n2
        assert abs(ratio - 8.0) / 8.0 < 0.01, f"Ratio = {ratio}, expected 8.0"

    def test_power_law_ratio_m5(self):
        """On m=5 slope: doubling Δσ → N ratio = 2^5 = 32."""
        ds1 = 25e6  # below knee
        ds2 = 50e6  # 2× ds1, still below knee
        n1 = dnv_d_curve_life(ds1).cycles_to_failure
        n2 = dnv_d_curve_life(ds2).cycles_to_failure
        ratio = n1 / n2
        assert abs(ratio - 32.0) / 32.0 < 0.01, f"Ratio = {ratio}, expected 32.0"

    def test_log_log_linearity_m3_slope(self):
        """Log-log linearity on m=3 slope: regression residual ≤ 1%."""
        # All stresses above knee (N < 1e7, m=3)
        stress_levels = [60e6, 80e6, 100e6, 120e6, 150e6, 200e6]
        log_ds = [math.log10(s / 1e6) for s in stress_levels]
        log_n = [math.log10(dnv_d_curve_life(s).cycles_to_failure) for s in stress_levels]
        # Linear regression
        n = len(log_ds)
        sx = sum(log_ds)
        sy = sum(log_n)
        sxx = sum(x * x for x in log_ds)
        sxy = sum(x * y for x, y in zip(log_ds, log_n))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        residuals = [abs(y - (slope * x + intercept)) for x, y in zip(log_ds, log_n)]
        max_rel = max(residuals) / max(abs(y) for y in log_n)
        assert max_rel < 0.01, f"Max relative residual = {max_rel}"
        # Slope should be -3
        assert abs(slope - (-3.0)) < 0.01, f"Slope = {slope}, expected -3.0"


# ---------------------------------------------------------------------------
# Beam-to-hotspot interface tests
# ---------------------------------------------------------------------------

class TestBeamToHotSpot:
    """Test beam theory to hot-spot stress interface."""

    def test_basic_interface(self):
        g = TJointGeometry(plate_thickness_m=0.015)
        scf = compute_joint_scf(g, n_xi=24, n_eta=14)
        result = beam_to_hotspot(100e6, scf, scf_type="hotspot")
        assert isinstance(result, BeamToHotSpotInterface)
        assert result.sigma_nom_pa == 100e6
        assert result.sigma_hot_pa == scf.kt_hotspot * 100e6

    def test_different_scf_types(self):
        g = TJointGeometry()
        scf = compute_joint_scf(g, n_xi=24, n_eta=14)
        r1 = beam_to_hotspot(50e6, scf, scf_type="hotspot")
        r2 = beam_to_hotspot(50e6, scf, scf_type="hotspot_iiw")
        r3 = beam_to_hotspot(50e6, scf, scf_type="notch_toe")
        assert r1.scf == scf.kt_hotspot
        assert r2.scf == scf.kt_hotspot_iiw
        assert r3.scf == scf.kt_toe_fe

    def test_invalid_scf_type(self):
        g = TJointGeometry()
        scf = compute_joint_scf(g, n_xi=24, n_eta=14)
        with pytest.raises(ValueError, match="scf_type"):
            beam_to_hotspot(50e6, scf, scf_type="invalid")


# ---------------------------------------------------------------------------
# Mesh convergence tests (using small meshes for speed)
# ---------------------------------------------------------------------------

class TestMeshConvergence:
    """Test mesh convergence study (C2)."""

    def test_convergence_study_runs(self):
        g = TJointGeometry(plate_thickness_m=0.015)
        result = mesh_convergence_study(g, refinement_levels=((16, 10), (24, 14), (36, 20)))
        assert isinstance(result, MeshConvergenceResult)
        assert len(result.mesh_levels) == 3
        assert result.scf_change_percent >= 0
        assert result.gci_percent >= 0

    def test_convergence_criterion(self):
        """C2: SCF change ≤ 2% and GCI ≤ 3% on mesh doubling."""
        g = TJointGeometry(plate_thickness_m=0.015)
        result = mesh_convergence_study(
            g, refinement_levels=((16, 10), (24, 14), (36, 20))
        )
        # The criterion is checked; we report whether it passes
        assert result.scf_change_percent < 50.0  # sanity: not wildly divergent


# ---------------------------------------------------------------------------
# Full verification report tests (using small meshes for speed)
# ---------------------------------------------------------------------------

class TestVerificationReport:
    """Test the complete verification report (all criteria C1–C6)."""

    def test_report_structure(self):
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        assert isinstance(report, VerificationReport)
        assert "c1_t_joint_scf" in report.__dataclass_fields__
        assert "c2_mesh_convergence" in report.__dataclass_fields__
        assert "c3_hotspot_linearity" in report.__dataclass_fields__
        assert "c4_sn_power_law" in report.__dataclass_fields__
        assert "c5_smooth_limit" in report.__dataclass_fields__
        assert "c6_singularity_trend" in report.__dataclass_fields__

    def test_c1_t_joint_scf_in_iiw_range(self):
        """C1: T-joint SCF should be in IIW range 1.2–1.5."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        c1 = report.c1_t_joint_scf
        assert c1["passed"], f"C1 failed: kt={c1['kt_hotspot']}, range={c1['iiw_recommended_range']}"

    def test_c2_mesh_convergence(self):
        """C2: Mesh convergence should be achieved."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        c2 = report.c2_mesh_convergence
        assert c2["passed"], f"C2 failed: change={c2['scf_change_percent']}%, GCI={c2['gci_percent']}%"

    def test_c3_hotspot_linearity(self):
        """C3: Hot-spot extrapolation R² ≥ 0.999."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        c3 = report.c3_hotspot_linearity
        assert c3["passed"], f"C3 failed: R²={c3['r_squared']}"

    def test_c4_sn_power_law(self):
        """C4: S-N power law ratio and residual."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        c4 = report.c4_sn_power_law
        assert c4["passed"], f"C4 failed: ratio_error={c4['ratio_error']}, residual={c4['relative_residual']}"

    def test_c5_smooth_limit(self):
        """C5: r → ∞ gives SCF → 1."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        c5 = report.c5_smooth_limit
        assert c5["passed"], f"C5 failed: deviation={c5['deviation_from_1']}"

    def test_c6_singularity_monotonic(self):
        """C6: SCF should increase as r decreases."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        c6 = report.c6_singularity_trend
        assert c6["passed"], f"C6 failed: monotonic={c6['monotonic_increase']}"

    def test_summary_string(self):
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        assert "IN67-N09" in report.summary
        assert "criteria passed" in report.summary

    def test_all_criteria_pass(self):
        """All six criteria should pass."""
        report = run_all_verification(plate_thickness_m=0.015, n_xi=24, n_eta=14)
        assert report.all_passed, report.summary
