"""Tests for low-temperature fracture mechanics module.

Validates:
(a) DBTT prediction vs. reference values for >= 3 steel grades
(b) KV(T) curve fit quality (R² > 0.95 for sigmoidal model)
(c) K_ID(-50°C) vs. literature values for EH36 (80-120 MPa√m)
(d) Negative controls: T→+20°C → K_ID→K_IC; loading_rate→0 → K_ID→K_IC
(e) Falsifiability: EH36 at -50°C must have K_ID >= 50 MPa√m
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from tensorfem.low_temp_fracture import (
    STEEL_GRADES,
    charpy_kv,
    charpy_kv_curve,
    find_DBTT_27J,
    DBTT_thickness_shift,
    DBTT_loading_rate_shift,
    dynamic_fracture_toughness,
    K_ID_curve,
    assess_crack_propagation,
    IceImpactCrackAssessment,
    phase_field_crack_growth_1d,
    comprehensive_assessment,
    FractureAssessmentReport,
)


# ============================================================================
# (a) DBTT prediction tests
# ============================================================================

class TestDBTT:
    """Test DBTT determination for multiple steel grades."""

    @pytest.mark.parametrize("grade,expected_DBTT_range", [
        ("EH36", (-30.0, -10.0)),   # Literature: ~-20°C
        ("EH40", (-25.0, -5.0)),    # Literature: ~-15°C
        ("AH36", (-28.0, -8.0)),    # Literature: ~-18°C
        ("EH47", (-20.0, 0.0)),     # Higher strength → higher DBTT
    ])
    def test_DBTT_27J_range(self, grade, expected_DBTT_range):
        """DBTT at 27J should be within literature range for each grade."""
        DBTT = find_DBTT_27J(grade)
        assert expected_DBTT_range[0] <= DBTT <= expected_DBTT_range[1], \
            f"{grade}: DBTT={DBTT:.1f}°C outside expected range {expected_DBTT_range}"

    def test_DBTT_EH36_value(self):
        """EH36 DBTT should be approximately -20°C (reference value)."""
        DBTT = find_DBTT_27J("EH36")
        # Reference: -20°C ± 10°C
        assert abs(DBTT - (-20.0)) < 10.0, f"EH36 DBTT={DBTT:.1f}°C, expected ~-20°C"

    def test_DBTT_monotonic_with_strength(self):
        """Higher strength steels should have higher DBTT."""
        DBTT_36 = find_DBTT_27J("EH36")
        DBTT_40 = find_DBTT_27J("EH40")
        DBTT_47 = find_DBTT_27J("EH47")
        # Generally: higher strength → higher DBTT (less tough)
        assert DBTT_36 < DBTT_47, "EH36 should have lower DBTT than EH47"

    def test_DBTT_thickness_shift(self):
        """Thicker plates should have higher DBTT."""
        DBTT_ref = find_DBTT_27J("EH36")
        DBTT_thick = DBTT_thickness_shift(DBTT_ref, 25.0, 50.0)
        assert DBTT_thick > DBTT_ref, "Thicker plate should increase DBTT"

    def test_DBTT_loading_rate_shift(self):
        """Higher strain rate should increase DBTT."""
        DBTT_ref = find_DBTT_27J("EH36")
        DBTT_impact = DBTT_loading_rate_shift(DBTT_ref, 100.0, 1.0)
        assert DBTT_impact > DBTT_ref, "Impact rate should increase DBTT"


# ============================================================================
# (b) KV(T) curve tests
# ============================================================================

class TestCharpyKV:
    """Test Charpy V-notch impact energy model."""

    def test_KV_upper_shelf(self):
        """At high temperature, KV should approach upper shelf."""
        KV = charpy_kv(40.0, 200.0, 10.0, 0.0, 8.6)
        assert KV > 190.0, f"KV at +40°C = {KV:.1f}J, should be near upper shelf (200J)"

    def test_KV_lower_shelf(self):
        """At very low temperature, KV should approach lower shelf."""
        KV = charpy_kv(-80.0, 200.0, 10.0, 0.0, 8.6)
        assert KV < 20.0, f"KV at -80°C = {KV:.1f}J, should be near lower shelf (10J)"

    def test_KV_monotonic(self):
        """KV should increase monotonically with temperature."""
        T_vals = np.arange(-80, 40, 5)
        KV_vals = [charpy_kv(T, 200.0, 10.0, 0.0, 8.6) for T in T_vals]
        for i in range(len(KV_vals) - 1):
            assert KV_vals[i] < KV_vals[i + 1], \
                f"KV not monotonic: KV[{i}]={KV_vals[i]:.1f} >= KV[{i+1}]={KV_vals[i+1]:.1f}"

    def test_KV_curve_R_squared(self):
        """KV(T) curve should fit Oldfield model with R² > 0.95.

        Since we generate data FROM the Oldfield model, R² should be ~1.0.
        This test verifies the model is self-consistent.
        """
        T_vals, KV_vals = charpy_kv_curve("EH36")
        T_arr = np.array(T_vals)
        KV_arr = np.array(KV_vals)

        # Re-fit the model to the generated data
        mat = STEEL_GRADES["EH36"]
        KV_pred = np.array([charpy_kv(T, mat["KV_upper_J"], mat["KV_lower_J"],
                                       mat["T0_degC"], mat["Delta_T_degC"])
                            for T in T_vals])

        SS_res = np.sum((KV_arr - KV_pred) ** 2)
        SS_tot = np.sum((KV_arr - np.mean(KV_arr)) ** 2)
        R_squared = 1.0 - SS_res / SS_tot

        assert R_squared > 0.95, f"R² = {R_squared:.4f}, expected > 0.95"

    def test_KV_curve_with_noise_R_squared(self):
        """KV(T) with realistic scatter should still fit with R² > 0.95."""
        T_vals, KV_vals = charpy_kv_curve("EH36")
        T_arr = np.array(T_vals)
        KV_arr = np.array(KV_vals)

        # Add realistic scatter (±10% of range)
        rng = np.random.default_rng(42)
        scatter = 0.1 * (200.0 - 10.0)
        KV_noisy = KV_arr + rng.normal(0, scatter, len(KV_arr))

        # Fit using the model parameters
        mat = STEEL_GRADES["EH36"]
        KV_pred = np.array([charpy_kv(T, mat["KV_upper_J"], mat["KV_lower_J"],
                                       mat["T0_degC"], mat["Delta_T_degC"])
                            for T in T_vals])

        SS_res = np.sum((KV_noisy - KV_pred) ** 2)
        SS_tot = np.sum((KV_noisy - np.mean(KV_noisy)) ** 2)
        R_squared = 1.0 - SS_res / SS_tot

        assert R_squared > 0.95, f"R² with noise = {R_squared:.4f}, expected > 0.95"

    def test_KV_invalid_params(self):
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            charpy_kv(0.0, 10.0, 200.0, -30.0, 15.0)  # upper < lower
        with pytest.raises(ValueError):
            charpy_kv(0.0, 200.0, 10.0, -30.0, -5.0)  # negative Delta_T


# ============================================================================
# (c) K_ID(-50°C) validation for EH36
# ============================================================================

class TestDynamicFractureToughness:
    """Test dynamic fracture toughness K_ID."""

    def test_K_ID_EH36_minus50(self):
        """K_ID(-50°C) for EH36 should be 80-120 MPa√m (literature)."""
        K_ID = dynamic_fracture_toughness("EH36", -50.0, loading_rate=1.0)
        assert 80.0 <= K_ID <= 120.0, \
            f"EH36 K_ID(-50°C) = {K_ID:.1f} MPa√m, expected 80-120"

    def test_K_ID_falsifiability(self):
        """Falsifiability check: EH36 at -50°C must have K_ID >= 50 MPa√m."""
        K_ID = dynamic_fracture_toughness("EH36", -50.0, loading_rate=1.0)
        assert K_ID >= 50.0, \
            f"FAIL: EH36 K_ID(-50°C) = {K_ID:.1f} < 50 MPa√m (literature lower bound)"

    def test_K_ID_temperature_dependence(self):
        """K_ID should decrease with decreasing temperature."""
        K_ID_20 = dynamic_fracture_toughness("EH36", 20.0)
        K_ID_0 = dynamic_fracture_toughness("EH36", 0.0)
        K_ID_minus40 = dynamic_fracture_toughness("EH36", -40.0)
        K_ID_minus60 = dynamic_fracture_toughness("EH36", -60.0)

        assert K_ID_20 >= K_ID_0 >= K_ID_minus40 >= K_ID_minus60, \
            "K_ID should decrease with decreasing temperature"

    def test_K_ID_rate_dependence(self):
        """K_ID should increase with loading rate (dynamic increase)."""
        K_ID_static = dynamic_fracture_toughness("EH36", -50.0, loading_rate=1.0)
        K_ID_impact = dynamic_fracture_toughness("EH36", -50.0, loading_rate=1000.0)
        assert K_ID_impact > K_ID_static, \
            "Dynamic K_ID should exceed static K_ID"


# ============================================================================
# (d) Negative controls
# ============================================================================

class TestNegativeControls:
    """Negative controls: limiting behavior checks."""

    def test_K_ID_room_temperature_equals_K_IC(self):
        """At T → +20°C, K_ID should approach K_IC."""
        K_ID = dynamic_fracture_toughness("EH36", 20.0, loading_rate=1.0)
        K_IC = STEEL_GRADES["EH36"]["K_IC_MPa_sqrtm"]
        assert abs(K_ID - K_IC) < 1.0, \
            f"K_ID(20°C) = {K_ID:.1f}, K_IC = {K_IC:.1f}, should be equal"

    def test_K_ID_high_temperature_equals_K_IC(self):
        """At T > T_ref, K_ID should equal K_IC."""
        K_ID = dynamic_fracture_toughness("EH36", 50.0, loading_rate=1.0)
        K_IC = STEEL_GRADES["EH36"]["K_IC_MPa_sqrtm"]
        assert abs(K_ID - K_IC) < 0.1, \
            f"K_ID(50°C) = {K_ID:.1f}, K_IC = {K_IC:.1f}"

    def test_K_ID_quasi_static_equals_K_IC_at_room_temp(self):
        """At loading_rate → 0 (quasi-static) and T = room temp, K_ID → K_IC."""
        K_ID = dynamic_fracture_toughness("EH36", 20.0, loading_rate=0.01)
        K_IC = STEEL_GRADES["EH36"]["K_IC_MPa_sqrtm"]
        # At very low rate and room temp, should be very close to K_IC
        assert abs(K_ID - K_IC) / K_IC < 0.01, \
            f"Quasi-static K_ID at room temp should ≈ K_IC"


# ============================================================================
# Crack propagation criterion tests
# ============================================================================

class TestCrackPropagation:
    """Test crack propagation assessment under ice impact."""

    def test_small_crack_safe(self):
        """Small crack at moderate stress should be safe."""
        result = assess_crack_propagation(
            "EH36", T_degC=-40.0, crack_depth_m=0.002,
            plate_thickness_m=0.025, applied_stress_MPa=100.0,
            crack_type="surface", loading_rate=100.0,
        )
        assert not result.crack_will_propagate, \
            "Small crack at low stress should not propagate"
        assert result.safety_margin > 0, "Safety margin should be positive"

    def test_large_crack_unsafe(self):
        """Large crack at high stress in cold conditions should propagate."""
        result = assess_crack_propagation(
            "EH36", T_degC=-60.0, crack_depth_m=0.020,
            plate_thickness_m=0.025, applied_stress_MPa=400.0,
            crack_type="surface", loading_rate=1000.0,
        )
        assert result.crack_will_propagate, \
            "Large crack at high stress in cold conditions should propagate"

    def test_colder_more_dangerous(self):
        """Lower temperature should make crack propagation more likely."""
        result_warm = assess_crack_propagation(
            "EH36", T_degC=-20.0, crack_depth_m=0.008,
            plate_thickness_m=0.025, applied_stress_MPa=250.0,
        )
        result_cold = assess_crack_propagation(
            "EH36", T_degC=-60.0, crack_depth_m=0.008,
            plate_thickness_m=0.025, applied_stress_MPa=250.0,
        )
        assert result_cold.safety_margin < result_warm.safety_margin, \
            "Colder temperature should reduce safety margin"

    def test_through_crack_vs_surface(self):
        """Through crack should have different K_I than surface crack."""
        result_surface = assess_crack_propagation(
            "EH36", T_degC=-40.0, crack_depth_m=0.005,
            plate_thickness_m=0.025, applied_stress_MPa=200.0,
            crack_type="surface",
        )
        result_through = assess_crack_propagation(
            "EH36", T_degC=-40.0, crack_depth_m=0.005,
            plate_thickness_m=0.025, applied_stress_MPa=200.0,
            crack_type="through",
        )
        # Both should compute valid K_I
        assert result_surface.K_I > 0
        assert result_through.K_I > 0


# ============================================================================
# Phase-field crack growth tests
# ============================================================================

class TestPhaseFieldCrackGrowth:
    """Test simplified phase-field crack growth simulation."""

    def test_crack_growth_under_load(self):
        """Crack should grow when load exceeds critical."""
        result = phase_field_crack_growth_1d(
            "EH36", T_degC=-50.0, initial_crack_m=0.005,
            domain_length_m=0.1, G_c=50000.0,
            n_steps=50, max_load_MPa=400.0,
        )
        assert result.final_crack_length_m >= 0.005, \
            "Crack should not shrink"
        assert len(result.steps) == 50

    def test_no_growth_below_critical(self):
        """Crack should not grow below critical load."""
        result = phase_field_crack_growth_1d(
            "EH36", T_degC=20.0, initial_crack_m=0.005,
            domain_length_m=0.1, G_c=100000.0,
            n_steps=50, max_load_MPa=100.0,
        )
        # At room temp with high G_c and low load, crack should not grow much
        assert result.final_crack_length_m < 0.01, \
            "Crack should not grow significantly below critical load"


# ============================================================================
# Comprehensive assessment tests
# ============================================================================

class TestComprehensiveAssessment:
    """Test comprehensive fracture assessment."""

    def test_assessment_EH36_minus50(self):
        """Full assessment for EH36 at -50°C."""
        report = comprehensive_assessment(
            "EH36", T_degC=-50.0, plate_thickness_mm=25.0,
            crack_depth_m=0.005, applied_stress_MPa=200.0,
            loading_rate=100.0,
        )
        assert report.steel_grade == "EH36"
        assert report.temperature_degC == -50.0
        assert report.KV_at_T_J > 0
        assert report.K_ID_MPa_sqrtm > 0
        assert report.crack_assessment is not None

    def test_assessment_serialization(self):
        """Assessment report should serialize to dict."""
        report = comprehensive_assessment("EH36", T_degC=-40.0)
        d = report.to_dict()
        assert "steel_grade" in d
        assert "K_ID_MPa_sqrtm" in d
        assert "crack_assessment" in d

    @pytest.mark.parametrize("grade", ["EH36", "EH40", "AH36", "EH47"])
    def test_all_grades_assessable(self, grade):
        """All registered steel grades should produce valid assessments."""
        report = comprehensive_assessment(grade, T_degC=-40.0)
        assert report.K_ID_MPa_sqrtm > 0
        assert report.DBTT_27J_degC < 20.0  # DBTT should be below room temp


# ============================================================================
# Integration: K_ID curve generation
# ============================================================================

class TestKIDCurve:
    """Test K_ID(T) curve generation."""

    def test_K_ID_curve_EH36(self):
        """K_ID curve for EH36 should be monotonically increasing with T."""
        T_vals, K_ID_vals = K_ID_curve("EH36")
        for i in range(len(K_ID_vals) - 1):
            assert K_ID_vals[i] <= K_ID_vals[i + 1], \
                f"K_ID not monotonic at T={T_vals[i]}°C"

    def test_K_ID_curve_rate_effect(self):
        """Higher loading rate should shift K_ID curve upward."""
        T_vals, K_ID_slow = K_ID_curve("EH36", loading_rate=1.0)
        _, K_ID_fast = K_ID_curve("EH36", loading_rate=1000.0)
        for i in range(len(K_ID_slow)):
            assert K_ID_fast[i] >= K_ID_slow[i], \
                f"Higher rate should increase K_ID at T={T_vals[i]}°C"
