"""Tests for the propeller ice-milling high-cycle fatigue chain.

Every stage is pinned to an independent anchor: Hertz closed form vs
Boussinesq quadrature, beam FE vs Euler-Bernoulli closed form and mesh
convergence, modal superposition vs direct Newmark-beta, rainflow vs a
known sinusoidal cycle count, and the S-N curve vs DNV-RP-C203 table values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.marine_fatigue_advanced import RainflowCycle, rainflow_cycles
from tensorfem.propeller_ice_milling_fatigue_chain import (
    BladeGeometry,
    DNV_D_TABLE,
    blade_modal_basis,
    blade_section_properties,
    cantilever_beam_modes,
    dnv_d_curve_in_air,
    effective_contact_modulus,
    ellipsoid_pole_radius,
    equivalent_contact_radius,
    euler_bernoulli_frequency,
    half_sine_pulse,
    hertz_engagement,
    hertz_force,
    hertz_pressure_quadrature,
    ice_fragment_geometry,
    milling_bounding_case,
    milling_damage_case,
    milling_parameter_sweep,
    miner_damage,
    modal_stress_history,
    newmark_stress_history,
    pulse_train,
    sine_stress_history,
    wet_mode_frequencies,
)


def uniform_geometry(**kwargs) -> BladeGeometry:
    """Bending-uniform reference blade (root and tip sections identical)."""
    base = dict(
        root_chord_m=0.50,
        tip_chord_m=0.50,
        root_thickness_m=0.18,
        tip_thickness_m=0.18,
        n_elements=32,
    )
    base.update(kwargs)
    return BladeGeometry(**base)


# ---------------------------------------------------------------------------
# Stage 1: Hertz contact
# ---------------------------------------------------------------------------


def test_effective_contact_modulus_matches_hand_value() -> None:
    e_star = effective_contact_modulus(9.0e9, 0.30, 118.0e9, 0.34)
    expected = 1.0 / ((1.0 - 0.30**2) / 9.0e9 + (1.0 - 0.34**2) / 118.0e9)
    assert e_star == pytest.approx(expected, rel=1e-14)


def test_equivalent_contact_radius_is_harmonic() -> None:
    assert equivalent_contact_radius(0.1, 0.1) == pytest.approx(0.05, rel=1e-14)
    # A much softer second body degenerates to the small radius.
    assert equivalent_contact_radius(0.5, 1.0e-3) == pytest.approx(9.98e-4, rel=1e-3)


def test_ellipsoid_pole_radius_sphere_limit() -> None:
    assert ellipsoid_pole_radius(0.2, 0.2, 0.2) == pytest.approx(0.2, rel=1e-15)
    fragment = ice_fragment_geometry(0.2, aspect_ratio=2.0)
    assert fragment["pole_radius_m"] == pytest.approx(0.2, rel=1e-12)
    assert fragment["mass_kg"] == pytest.approx(
        917.0 * 4.0 / 3.0 * math.pi * 0.1 * 0.1 * 0.05, rel=1e-12
    )


def test_hertz_force_quadrature_matches_closed_form() -> None:
    """Criterion (b): Hertz force vs independent quadrature < 1e-3 N."""
    for penetration in (1.0e-5, 1.0e-4, 1.0e-3):
        result = hertz_pressure_quadrature(9.2e9, 0.01, penetration)
        assert abs(result["force_quadrature_n"] - result["force_closed_form_n"]) < 1.0e-3


def test_hertz_pressure_field_gives_parabolic_displacement() -> None:
    result = hertz_pressure_quadrature(9.2e9, 0.01, 2.0e-4, probe_ratios=(0.0, 0.25, 0.5, 0.75))
    # Hertz kinematics: u(r) = delta (1 - r^2 / (2 a^2)) inside the patch.
    assert result["centre_displacement_relative_error"] < 0.01
    assert result["displacement_profile_relative_error"] < 0.02


def test_hertz_force_power_law_exponent() -> None:
    base = hertz_force(9.2e9, 0.01, 1.0e-4)
    scaled = hertz_force(9.2e9, 0.01, 4.0e-4)
    assert scaled / base == pytest.approx(8.0, rel=1e-12)


def test_hertz_engagement_conserves_energy() -> None:
    engagement = hertz_engagement(
        e_star_pa=9.2e9,
        equivalent_radius_m=0.01,
        effective_mass_kg=40.0,
        impact_velocity_m_s=5.0,
    )
    stored = 0.8 * engagement.contact_stiffness_n_m15 * engagement.penetration_m**2.5
    assert stored == pytest.approx(0.5 * 40.0 * 25.0, rel=1e-9)
    assert engagement.force_max_n == pytest.approx(
        engagement.contact_stiffness_n_m15 * engagement.penetration_m**1.5, rel=1e-12
    )


def test_hertz_impact_duration_follows_velocity_scaling() -> None:
    engagement_fast = hertz_engagement(
        e_star_pa=9.2e9, equivalent_radius_m=0.01, effective_mass_kg=40.0, impact_velocity_m_s=8.0
    )
    engagement_slow = hertz_engagement(
        e_star_pa=9.2e9, equivalent_radius_m=0.01, effective_mass_kg=40.0, impact_velocity_m_s=1.0
    )
    ratio = engagement_fast.duration_s / engagement_slow.duration_s
    assert ratio == pytest.approx(8.0 ** (-0.2), rel=1e-12)


def test_crushing_limit_caps_contact_pressure() -> None:
    elastic = hertz_engagement(
        e_star_pa=9.2e9, equivalent_radius_m=0.01, effective_mass_kg=40.0, impact_velocity_m_s=5.0
    )
    limited = hertz_engagement(
        e_star_pa=9.2e9,
        equivalent_radius_m=0.01,
        effective_mass_kg=40.0,
        impact_velocity_m_s=5.0,
        crush_pressure_pa=30.0e6,
    )
    assert elastic.peak_pressure_pa > 1.0e9  # the elastic point contact is unphysical
    assert limited.force_max_n < elastic.force_max_n
    mean_pressure = limited.force_max_n / (math.pi * limited.contact_radius_m**2)
    assert mean_pressure == pytest.approx(2.0 / 3.0 * 30.0e6, rel=1e-9)


# ---------------------------------------------------------------------------
# Stage 2: blade modal analysis
# ---------------------------------------------------------------------------


def test_uniform_beam_frequencies_match_euler_bernoulli() -> None:
    """Criterion (a): first five dry frequencies vs analytic < 5 %."""
    geometry = uniform_geometry()
    ei, rho_a, _ = blade_section_properties(geometry)
    omega, _, _ = cantilever_beam_modes(ei, rho_a, geometry.span_m, n_modes=5)
    for mode in range(5):
        analytic = 2.0 * math.pi * euler_bernoulli_frequency(
            geometry.span_m, ei[0], rho_a[0], mode=mode + 1
        )
        assert abs(omega[mode] - analytic) / analytic < 0.05
        # The uniform FE model is actually far tighter than the criterion.
        assert abs(omega[mode] - analytic) / analytic < 0.01


def test_tapered_blade_frequencies_are_mesh_converged() -> None:
    coarse = blade_modal_basis(BladeGeometry(n_elements=48), n_modes=5)
    fine = blade_modal_basis(BladeGeometry(n_elements=96), n_modes=5)
    for mode in range(5):
        assert abs(fine.frequencies_hz[mode] - coarse.frequencies_hz[mode]) / fine.frequencies_hz[
            mode
        ] < 0.005


def test_wet_frequency_ratio_matches_added_mass() -> None:
    geometry = uniform_geometry(n_elements=24)
    ei, rho_a, _ = blade_section_properties(geometry)
    chunk = geometry.span_m / geometry.n_elements
    added = np.full(geometry.n_elements, 300.0)  # uniform extra mass per unit span
    dry, wet = wet_mode_frequencies(ei, rho_a, added, geometry.span_m, n_modes=3)
    expected = math.sqrt(rho_a[0] * chunk / (rho_a[0] * chunk + added[0] * chunk))
    for mode in range(3):
        assert wet[mode] / dry[mode] == pytest.approx(expected, rel=0.01)


def test_root_curvature_matches_static_cantilever() -> None:
    geometry = uniform_geometry(n_elements=16)
    ei, rho_a, _ = blade_section_properties(geometry)
    stiffness, _ = cantilever_beam_modes.__globals__["_assemble_beam"](ei, rho_a, geometry.span_m)
    free = np.arange(2, stiffness.shape[0])
    load = np.zeros(free.size)
    load[np.searchsorted(free, 2 * geometry.n_elements)] = 1.0e4  # unit-ish tip load
    displacement = np.zeros(stiffness.shape[0])
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load)
    from tensorfem.propeller_ice_milling_fatigue_chain import root_curvature_from_dofs

    curvature = root_curvature_from_dofs(displacement[:4], geometry.span_m / geometry.n_elements)
    analytic = 1.0e4 * geometry.span_m / ei[0]
    assert curvature == pytest.approx(analytic, rel=1e-6)


# ---------------------------------------------------------------------------
# Stage 3: stress response
# ---------------------------------------------------------------------------


def test_modal_static_limit_matches_static_bending() -> None:
    geometry = uniform_geometry(n_elements=16)
    basis = blade_modal_basis(geometry, n_modes=6)
    ei, _, _ = blade_section_properties(geometry)
    force = 1.0e4
    static_curvature = force * (geometry.contact_radius_m - geometry.hub_radius_m) / ei[0]
    modal_curvature = float(
        np.sum(basis.root_curvature_factors * basis.contact_phi / basis.omega_rad_s**2) * force
    )
    assert modal_curvature == pytest.approx(static_curvature, rel=0.02)


def test_modal_superposition_matches_newmark() -> None:
    """Criterion (c): modal stress vs direct Newmark-beta < 3 % peak and RMS."""
    geometry = uniform_geometry(n_elements=12)
    basis = blade_modal_basis(geometry, n_modes=5)
    force_max = 8.0e4
    duration = 2.0e-3
    _, pulse, dt = half_sine_pulse(force_max, duration, n_samples=100)
    # One pulse followed by ring-down.
    history = np.concatenate([pulse, np.zeros(1600)])
    zeta = 0.02
    omega = basis.omega_rad_s
    a0 = 2.0 * zeta * omega[0] * omega[-1] / (omega[0] + omega[-1])
    a1 = 2.0 * zeta / (omega[0] + omega[-1])
    zetas = [0.5 * (a0 / w + a1 * w) for w in omega]
    _, stress_modal = modal_stress_history(
        history,
        dt,
        basis,
        fibre_distance_m=0.5 * geometry.thickness_m(0.0),
        damping_ratios=zetas,
        wet=False,
    )
    _, stress_newmark = newmark_stress_history(
        history, dt, geometry, damping_ratio=zeta, damping_mode_index=4
    )
    peak_modal = float(np.max(np.abs(stress_modal)))
    peak_newmark = float(np.max(np.abs(stress_newmark)))
    rms_modal = float(np.sqrt(np.mean(stress_modal**2)))
    rms_newmark = float(np.sqrt(np.mean(stress_newmark**2)))
    assert abs(peak_modal - peak_newmark) / peak_newmark < 0.03
    assert abs(rms_modal - rms_newmark) / rms_newmark < 0.03


def test_pulse_train_shape() -> None:
    time, values, dt = pulse_train(1.0e5, 2.0e-3, 10.0, 4, samples_per_period=2048)
    assert values.max() == pytest.approx(1.0e5, rel=1e-3)
    peak_indices = np.where(values > 0.99e5)[0]
    # One pulse per period: four separate groups of samples above the threshold.
    groups = 1 + int(np.sum(np.diff(peak_indices) > 1))
    assert groups == 4
    assert time[1] - time[0] == pytest.approx(dt, rel=1e-12)


# ---------------------------------------------------------------------------
# Stage 4/5: rainflow and S-N
# ---------------------------------------------------------------------------


def test_rainflow_sine_cycle_count() -> None:
    """Criterion (d): sinusoidal history cycle-count error < 1."""
    amplitude = 40.0e6
    cycles_expected = 7.0
    _, history, _ = sine_stress_history(amplitude, 1.0, cycles_expected, samples_per_cycle=64)
    rows = rainflow_cycles([float(value) for value in history])
    full = sum(row.count for row in rows if row.range_pa > 1.9 * amplitude)
    assert abs(full - cycles_expected) <= 1.0
    assert max(row.range_pa for row in rows) == pytest.approx(2.0 * amplitude, rel=0.01)


def test_rainflow_two_block_loading_is_exact() -> None:
    big, small = 100.0e6, 40.0e6
    history = [0.0]
    for _ in range(3):
        history.extend([big, -big])
    for _ in range(5):
        history.extend([small, -small])
    rows = rainflow_cycles(history)
    count_big = sum(row.count for row in rows if abs(row.range_pa - 2.0 * big) < 1.0)
    count_small = sum(row.count for row in rows if abs(row.range_pa - 2.0 * small) < 1.0)
    assert count_big == pytest.approx(2.5, abs=0.5)
    assert count_small == pytest.approx(5.0, abs=0.5)


def test_dnv_sn_curve_matches_table_values() -> None:
    """Criterion (e): DNV-RP-C203 curve D vs tabulated values < 5 %."""
    curve = dnv_d_curve_in_air()
    # FAT 90 at 2e6 cycles.
    assert curve.cycles_to_failure(90.0e6) / 2.0e6 == pytest.approx(1.0, rel=0.05)
    # Knee stress at 1e7 cycles.
    assert curve.knee_stress_pa / (DNV_D_TABLE["knee_stress_mpa"] * 1.0e6) == pytest.approx(
        1.0, rel=0.05
    )
    # Two-slope continuity at the knee.
    assert curve.cycles_to_failure(curve.knee_stress_pa) == pytest.approx(1.0e7, rel=2e-3)
    # Above the knee the second slope governs: N(30 MPa) > 1e8.
    assert curve.cycles_to_failure(30.0e6) > 1.0e8


def test_miner_two_block_damage() -> None:
    curve = dnv_d_curve_in_air()
    rows = (
        RainflowCycle(range_pa=100.0e6, mean_pa=0.0, count=1.0),
        RainflowCycle(range_pa=60.0e6, mean_pa=0.0, count=0.5),
    )
    result = miner_damage(rows, curve)
    expected = 1.0 / curve.cycles_to_failure(100.0e6) + 0.5 / curve.cycles_to_failure(60.0e6)
    assert result["damage"] == pytest.approx(expected, rel=1e-12)
    assert result["passed"] is (expected < 1.0)


# ---------------------------------------------------------------------------
# Stage 6: the chain
# ---------------------------------------------------------------------------


def test_milling_engagement_scaling() -> None:
    from tensorfem.propeller_ice_milling_fatigue_chain import MillingCondition, milling_engagement

    slow = milling_engagement(MillingCondition(rpm=100.0))
    fast = milling_engagement(MillingCondition(rpm=200.0))
    assert fast["engagement"].force_max_n > slow["engagement"].force_max_n
    assert fast["impact_frequency_hz"] == pytest.approx(2.0 * slow["impact_frequency_hz"], rel=1e-12)
    assert fast["impact_velocity_m_s"] > slow["impact_velocity_m_s"]


def test_milling_sweep_is_monotone() -> None:
    """Criterion (f): D increases with L_ice, n and V."""
    sweep = milling_parameter_sweep()
    assert sweep["all_pass"]
    for key, value in sweep["checks"].items():
        assert value, key


def test_milling_design_check_and_bounding_case() -> None:
    """Criterion (g): D < 1 for the 20-year design life; elastic bound reported."""
    design = milling_damage_case()
    assert design["fatigue"]["damage_20yr"] < 1.0
    assert design["fatigue"]["passed"] is True
    assert design["contact_model"] == "crushing-limited"
    bounding = milling_bounding_case()
    assert bounding["contact_model"] == "elastic-hertz"
    assert bounding["engagement"]["force_max_n"] > design["engagement"]["force_max_n"]
    assert bounding["fatigue"]["damage_20yr"] > design["fatigue"]["damage_20yr"]


def test_chain_is_deterministic() -> None:
    first = milling_damage_case(n_pulses=4)
    second = milling_damage_case(n_pulses=4)
    assert first["fatigue"]["damage_20yr"] == second["fatigue"]["damage_20yr"]
    assert first["stress"]["sigma_max_pa"] == second["stress"]["sigma_max_pa"]
    assert np.allclose(first["history"]["stress_pa"], second["history"]["stress_pa"], rtol=0.0, atol=0.0)


def test_invalid_inputs_are_rejected() -> None:
    from tensorfem.propeller_ice_milling_fatigue_chain import MillingCondition

    with pytest.raises(ValueError):
        MillingCondition(rpm=0.0)
    with pytest.raises(ValueError):
        MillingCondition(pitch_angle_rad=-0.1)
    with pytest.raises(ValueError):
        hertz_force(9.2e9, 0.01, 0.0)
    with pytest.raises(ValueError):
        rainflow_cycles([1.0, 1.0])
