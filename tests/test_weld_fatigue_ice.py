"""Tests for the ice-region welded-joint fatigue assessment chain.

Covers: weld geometry parameterisation, the FE weld-toe notch submodel (with the
exact Inglis reference), DNV-RP-C203 / IIW S-N anchors, the low-temperature
penalty, the exact SDOF response and its energy ledger, rainflow + Miner
accumulation, and the positive/negative controls of the case matrix.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.marine_fatigue_advanced import RainflowCycle, rainflow_cycles
from tensorfem.weld_fatigue_ice import (
    DNVSNCurve,
    IIWFatClass,
    IceFatigueCase,
    PanelStructure,
    SNTemperatureCorrection,
    WeldGeometry,
    elliptical_hole_kt_fe,
    exact_sdof_response,
    fatigue_assessment,
    hotspot_extrapolation,
    ice_pressure_history,
    low_temperature_factor,
    miner_damage,
    response_energy_ledger,
    toe_stress_concentration,
    weld_geometry_for,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_weld_geometry_validation():
    with pytest.raises(ValueError):
        WeldGeometry(weld_type="lap")
    with pytest.raises(ValueError):
        WeldGeometry(weld_type="butt", plate_thickness_m=-1.0)
    with pytest.raises(ValueError):
        WeldGeometry(weld_type="butt", flank_angle_deg=120.0)
    with pytest.raises(ValueError):
        WeldGeometry(weld_type="butt", plate_thickness_m=0.01, reinforcement_m=0.005)
    with pytest.raises(ValueError):
        WeldGeometry(weld_type="fillet", plate_thickness_m=0.01, reinforcement_m=0.02)
    geometry = WeldGeometry()
    assert geometry.weld_type == "butt"
    assert geometry.notch_depth_m > geometry.toe_radius_m


def test_notch_depth_grows_with_flank_angle():
    shallow = WeldGeometry(flank_angle_deg=20.0)
    steep = WeldGeometry(flank_angle_deg=60.0)
    # a shallower flank gives a narrower equivalent notch
    assert shallow.equivalent_notch_width_ratio > steep.equivalent_notch_width_ratio


# ---------------------------------------------------------------------------
# FE notch submodel
# ---------------------------------------------------------------------------


def test_fe_submodel_reproduces_inglis_solution():
    exact = 3.0
    fe = elliptical_hole_kt_fe(1.0, n_xi=48, n_eta=20)
    assert abs(fe - exact) / exact < 0.03


def test_fe_submodel_converges_under_refinement():
    coarse = elliptical_hole_kt_fe(2.0, n_xi=48, n_eta=20)
    fine = elliptical_hole_kt_fe(2.0, n_xi=64, n_eta=28)
    exact = 5.0
    assert abs(fine - exact) < abs(coarse - exact)
    assert abs(fine - exact) / exact < 0.03


@pytest.mark.parametrize("weld_type", ["butt", "fillet"])
def test_toe_stress_concentration_in_iiw_band(weld_type):
    submodel = toe_stress_concentration(weld_geometry_for(weld_type))
    assert 1.5 <= submodel.kt_toe <= 2.5
    assert submodel.mesh_fine_enough
    assert submodel.n_elements > 500


def test_fillet_toe_is_sharper_than_butt_toe():
    butt = toe_stress_concentration(weld_geometry_for("butt"))
    fillet = toe_stress_concentration(weld_geometry_for("fillet"))
    assert fillet.kt_toe > butt.kt_toe


def test_toe_stress_concentration_is_scale_invariant():
    small = toe_stress_concentration(
        WeldGeometry(plate_thickness_m=0.012, reinforcement_m=0.8e-3, toe_radius_m=0.5e-3)
    )
    large = toe_stress_concentration(
        WeldGeometry(plate_thickness_m=0.040, reinforcement_m=2.7e-3, toe_radius_m=2.0e-3)
    )
    assert abs(small.kt_toe - large.kt_toe) / small.kt_toe < 0.05


def test_hotspot_extrapolation_recovers_linear_path():
    distances = tuple(np.linspace(0.0, 0.05, 51))
    # a linear stress path is reproduced exactly by linear extrapolation
    stresses = tuple(2.0 - 20.0 * d for d in distances)
    value = hotspot_extrapolation(stresses, distances, 0.02, method="iiw")
    assert value == pytest.approx(2.0, rel=1e-6)
    with pytest.raises(ValueError):
        hotspot_extrapolation(stresses, distances, 0.02, method="watson")


# ---------------------------------------------------------------------------
# S-N curves and low temperature
# ---------------------------------------------------------------------------


def test_dnv_curve_knee_stress():
    curve = DNVSNCurve()
    assert curve.knee_stress_pa / 1e6 == pytest.approx(52.63, rel=0.03)


def test_dnv_curve_is_continuous_at_the_knee():
    curve = DNVSNCurve()
    knee = curve.knee_cycles
    below = curve.cycles_to_failure(curve.stress_at_cycles(0.99 * knee))
    above = curve.cycles_to_failure(curve.stress_at_cycles(1.01 * knee))
    assert below == pytest.approx(0.99 * knee, rel=2e-2)
    assert above == pytest.approx(1.01 * knee, rel=2e-2)


def test_iiw_fat90_matches_dnv_curve_d():
    dnv = DNVSNCurve().stress_at_cycles(2.0e6) / 1e6
    iiw = IIWFatClass(fat_mpa=90.0).stress_at_cycles(2.0e6) / 1e6
    assert dnv == pytest.approx(iiw, rel=0.03)


def test_low_temperature_factor_transition():
    assert low_temperature_factor(10.0) == 1.0
    assert low_temperature_factor(0.0) == 1.0
    assert low_temperature_factor(-5.0) == pytest.approx(1.2)
    assert low_temperature_factor(-40.0) == pytest.approx(1.2)
    assert 1.0 < low_temperature_factor(-2.5) < 1.2
    with pytest.raises(ValueError):
        low_temperature_factor(-10.0, factor=0.5)


def test_low_temperature_strength_reduction_at_1e7():
    correction = SNTemperatureCorrection(curve=DNVSNCurve(), temperature_c=-20.0)
    reduction = correction.strength_reduction(1.0e7)
    assert 0.15 <= reduction <= 0.25
    assert correction.stress_at_cycles(1.0e7) < correction.curve.stress_at_cycles(1.0e7)


def test_thickness_correction_inactive_below_25mm():
    curve = DNVSNCurve()
    thin = curve.thickness_corrected(0.020)
    assert thin.log_a1 == pytest.approx(curve.log_a1)
    thick = curve.thickness_corrected(0.040)
    assert thick.stress_at_cycles(1.0e7) < curve.stress_at_cycles(1.0e7)


# ---------------------------------------------------------------------------
# structural response and energy ledger
# ---------------------------------------------------------------------------


def _newmark(force, dt, mass, damping, stiffness):
    beta, gamma = 0.25, 0.5
    n = force.size
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    a[0] = (force[0] - damping * v[0] - stiffness * u[0]) / mass
    k_eff = stiffness + gamma / (beta * dt) * damping + mass / (beta * dt**2)
    for i in range(n - 1):
        rhs = (
            force[i + 1]
            + mass * (u[i] / (beta * dt**2) + v[i] / (beta * dt) + (1.0 / (2 * beta) - 1.0) * a[i])
            + damping
            * (
                gamma * u[i] / (beta * dt)
                + (gamma / beta - 1.0) * v[i]
                + dt * (gamma / (2 * beta) - 1.0) * a[i]
            )
        )
        u[i + 1] = rhs / k_eff
        a[i + 1] = (u[i + 1] - u[i]) / (beta * dt**2) - v[i] / (beta * dt) - (1.0 / (2 * beta) - 1.0) * a[i]
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
    return u, v


def test_exact_sdof_matches_newmark_reference():
    structure = PanelStructure()
    dt = 1.0 / 1000.0
    t = np.arange(6000) * dt
    force = 1.0e5 * np.sin(2.0 * math.pi * 5.0 * t)
    u, v = exact_sdof_response(
        force,
        dt,
        mass=structure.modal_mass_kg,
        damping=structure.modal_damping_n_s_m,
        stiffness=structure.modal_stiffness_n_m,
    )
    u_ref, _ = _newmark(
        force, dt, structure.modal_mass_kg, structure.modal_damping_n_s_m, structure.modal_stiffness_n_m
    )
    scale = np.abs(u_ref).max()
    assert np.abs(u - u_ref).max() / scale < 5e-3


def test_exact_sdof_static_limit():
    structure = PanelStructure()
    dt = 1.0 / 2000.0
    steps = 200000
    force = np.linspace(0.0, 1.0e6, steps)
    u, _ = exact_sdof_response(
        force,
        dt,
        mass=structure.modal_mass_kg,
        damping=structure.modal_damping_n_s_m,
        stiffness=structure.modal_stiffness_n_m,
    )
    assert u[-1] == pytest.approx(1.0e6 / structure.modal_stiffness_n_m, rel=0.05)


def test_response_energy_ledger_closes():
    structure = PanelStructure()
    load = ice_pressure_history(120.0, 50.0, 0.5, 5.0, seed=3)
    force = np.asarray(load["pressure_pa"]) * structure.modal_area_m2
    dt = 1.0 / 50.0
    u, v = exact_sdof_response(
        force,
        dt,
        mass=structure.modal_mass_kg,
        damping=structure.modal_damping_n_s_m,
        stiffness=structure.modal_stiffness_n_m,
    )
    ledger = response_energy_ledger(
        force,
        u,
        v,
        dt,
        mass=structure.modal_mass_kg,
        damping=structure.modal_damping_n_s_m,
        stiffness=structure.modal_stiffness_n_m,
    )
    assert abs(ledger["relative_residual"]) < 1e-6
    assert ledger["damping_dissipation_j"] > 0.0
    assert ledger["work_load_j"] > 0.0
    with pytest.raises(ValueError):
        response_energy_ledger(force, u[:-1], v, dt, mass=1.0, damping=0.0, stiffness=1.0)


# ---------------------------------------------------------------------------
# rainflow / Miner
# ---------------------------------------------------------------------------


def test_rainflow_sine_gives_one_cycle_per_period():
    dt = 1.0 / 200.0
    t = np.arange(int(20.0 / dt)) * dt
    signal = 50.0e6 * np.sin(2.0 * math.pi * 2.0 * t)
    cycles = rainflow_cycles(signal - signal.mean())
    total = sum(c.count for c in cycles)
    assert total == pytest.approx(40.0, rel=0.05)
    assert max(c.range_pa for c in cycles) == pytest.approx(100.0e6, rel=0.02)


def test_miner_damage_matches_hand_calculation():
    curve = DNVSNCurve()
    cycles = (RainflowCycle(100.0e6, 0.0, 1.0), RainflowCycle(50.0e6, 0.0, 1.0))
    result = miner_damage(cycles, curve, temperature_c=0.0)
    expected = 1.0 / curve.cycles_to_failure(100.0e6) + 1.0 / curve.cycles_to_failure(50.0e6)
    assert result["damage"] == pytest.approx(expected, rel=1e-12)
    warm = result["damage"]
    cold = miner_damage(cycles, curve, temperature_c=-20.0)["damage"]
    assert cold > warm


# ---------------------------------------------------------------------------
# load spectrum and case matrix controls
# ---------------------------------------------------------------------------


def test_ice_pressure_grows_with_thickness_and_speed():
    thin = ice_pressure_history(60.0, 50.0, 0.3, 5.0, seed=5)
    thick = ice_pressure_history(60.0, 50.0, 0.8, 5.0, seed=5)
    assert thick["mean_pressure_pa"] > thin["mean_pressure_pa"]
    slow = ice_pressure_history(60.0, 50.0, 0.5, 3.0, seed=5)
    fast = ice_pressure_history(60.0, 50.0, 0.5, 7.0, seed=5)
    assert fast["event_rate_hz"] > slow["event_rate_hz"]
    assert np.all(np.asarray(thin["pressure_pa"]) >= 0.0)


def _short_case(**kwargs):
    defaults = dict(window_s=300.0, sample_rate_hz=50.0, exposure_h=200.0)
    defaults.update(kwargs)
    return defaults


def test_case_assessment_is_positive_and_consistent():
    result = fatigue_assessment(IceFatigueCase(0.5, 5.0, "fillet"), **_short_case())
    assert result["damage_per_window"] > 0.0
    assert result["damage_per_1000h"] > result["damage_per_window"]
    assert result["cycles_counted"] > 50
    assert abs(result["energy_ledger"]["relative_residual"]) < 1e-6
    assert 1.5 <= result["kt_toe"] <= 2.5


def test_weld_effect_is_necessary_negative_control():
    """Without the weld toe (K_t = 1) the fatigue life must be overestimated."""
    with_weld = fatigue_assessment(IceFatigueCase(0.5, 5.0, "fillet"), **_short_case())
    without = fatigue_assessment(
        IceFatigueCase(0.5, 5.0, "fillet"), weld_override=1.0, **_short_case()
    )
    ratio = with_weld["damage_per_window"] / without["damage_per_window"]
    assert ratio > 1.5


def test_damage_grows_with_speed_and_k_t():
    slow = fatigue_assessment(IceFatigueCase(0.5, 3.0, "butt"), **_short_case())
    fast = fatigue_assessment(IceFatigueCase(0.5, 7.0, "butt"), **_short_case())
    assert fast["damage_per_window"] > slow["damage_per_window"]