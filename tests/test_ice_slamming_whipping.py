"""Tests for the ice-slamming / whipping / fatigue chain.

The guards here encode the verification story of
``tensorfem.ice_slamming_whipping``: every claim the module makes must be
reproducible against an independent reference (exact beam theory, a direct ODE
integration, a static equilibrium identity, the added-mass momentum closure) or
against a structural identity of the declared model (the pressure-area/velocity
exponents of the ice crushing law, the three negative controls).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensorfem.ice_slamming_whipping as m


@pytest.fixture(scope="module")
def ship() -> m.IceClassShipCase:
    return m.IceClassShipCase(
        length=230.0,
        breadth=32.0,
        draft=13.5,
        block_coefficient=0.80,
        girder_depth=20.0,
        girder_wall=0.040,
        deadrise_deg=30.0,
        impact_station_x=103.5,
        impact_local_immersion=3.5,
        impact_half_breadth=6.5,
        slam_length=14.0,
    )


@pytest.fixture(scope="module")
def fast_model() -> m.WhippingModel:
    return m.WhippingModel(elements=24, modes=4, response_dt=0.02, duration=20.0)


# --------------------------------------------------------------------------- 
# declared case and geometry
# ---------------------------------------------------------------------------
def test_geometry_is_consistent(ship):
    """C_b / L / B / T must reproduce the displaced volume before any run."""
    checks = ship.geometry_checks()
    assert checks["displacement_volume_m3"] == pytest.approx(
        ship.block_coefficient * ship.length * ship.breadth * ship.draft
    )
    assert checks["displacement_mass_t"] == pytest.approx(
        ship.water_density * checks["displacement_volume_m3"] / 1000.0
    )
    for flag in ("flag_length_breadth", "flag_breadth_draft", "flag_block_coefficient"):
        assert checks[flag] is False, flag
    girder = ship.girder()
    assert girder.inertia > 0.0
    assert ship.section_modulus == pytest.approx(girder.inertia / (0.5 * ship.girder_depth))


@pytest.mark.parametrize(
    "field,value",
    [
        ("length", -1.0),
        ("draft", 0.0),
        ("block_coefficient", 1.4),
        ("deadrise_deg", 95.0),
        ("impact_station_x", 200.0),
    ],
)
def test_invalid_ship_inputs_rejected(ship, field, value):
    payload = {name: getattr(ship, name) for name in ship.__dataclass_fields__}
    payload[field] = value
    with pytest.raises(ValueError):
        m.IceClassShipCase(**payload).validate()


def test_crush_pressure_vanishes_at_zero_velocity():
    ice = m.IceField(thickness=0.5, concentration=0.6)
    assert ice.crush_pressure(6.0, 0.0) == 0.0
    assert ice.crush_pressure(6.0, 5.0) > 0.0
    with pytest.raises(ValueError):
        ice.crush_pressure(0.0, 5.0)


# ---------------------------------------------------------------------------
# beam element and girder assembly
# ---------------------------------------------------------------------------
def test_beam_element_has_two_rigid_body_modes():
    """A single free-free element carries exactly heave and rotation."""
    stiffness, mass = m.beam_element_matrices(6.0e13, 1.2e11, 1.0e6, 5.0e6, 5.0)
    assert np.allclose(stiffness, stiffness.T)
    assert np.allclose(mass, mass.T)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    assert abs(eigenvalues[0]) < 1e-6 * abs(eigenvalues[-1])
    assert abs(eigenvalues[1]) < 1e-6 * abs(eigenvalues[-1])
    assert eigenvalues[2] > 0.0


def test_girder_has_two_rigid_modes_and_positive_spectrum(ship, fast_model):
    modes = m.wet_girder_modes(ship, fast_model)
    assert modes.rigid_modes == 2
    assert modes.omega.shape[0] == fast_model.modes + 2
    assert modes.omega[0] < 1.0 and modes.omega[1] < 1.0
    assert np.all(np.diff(modes.frequencies_hz) > 0.0)
    # the omega threshold for rigid modes is 1 rad/s = 0.159 Hz
    assert np.all(modes.vertical_frequencies_hz > 1.0 / (2.0 * math.pi))


# ---------------------------------------------------------------------------
# static equilibrium: the moment recovery must reproduce P*a*b/L
# ---------------------------------------------------------------------------
def test_static_moment_recovery(ship, fast_model):
    check = m.static_moment_check(ship, fast_model)
    assert check["relative_error"] < 1e-9, check
    assert check["measured_moment"] > 0.0


# ---------------------------------------------------------------------------
# criterion (b): frequencies against the exact beam-theory solution
# ---------------------------------------------------------------------------
def test_frequencies_match_exact_beam_theory(ship, fast_model):
    check = m.frequency_check(ship, fast_model)
    assert check["rigid_modes_detected"] == 2
    # criterion (b) is about the overall modes f2..f4; higher modes are still
    # converging on the 24-element half-model
    assert max(check["relative_residual"][:3]) < 0.05, check["relative_residual"]
    wet = check["frequencies_wet_fe_hz"]
    theory = check["frequencies_exact_dsm_hz"]
    assert len(wet) >= 3 and len(theory) >= 3
    # shear softening is physical and must shrink the Euler-Bernoulli values
    for fe, dsm, eb in zip(wet, theory, check["euler_bernoulli_hz"]):
        assert dsm < eb
        assert fe < eb


def test_euler_bernoulli_root_finder_is_exact():
    """The computed free-free roots must satisfy cos(x)cosh(x) = 1."""
    ship = m.IceClassShipCase(
        length=100.0,
        breadth=20.0,
        draft=8.0,
        block_coefficient=0.7,
        girder_depth=10.0,
        girder_wall=0.03,
        deadrise_deg=30.0,
        impact_station_x=40.0,
        impact_local_immersion=3.0,
        impact_half_breadth=4.0,
        slam_length=8.0,
    )
    f1 = m._euler_bernoulli_free_free_hz(1.0e13, 5.0e5, 100.0, 1)
    expected = (4.730040744862704**2 / 100.0**2) * math.sqrt(1.0e13 / 5.0e5) / (2 * math.pi)
    assert f1 == pytest.approx(expected, rel=1e-6)
    ratio = m._euler_bernoulli_free_free_hz(1.0e13, 5.0e5, 100.0, 2) / f1
    assert ratio == pytest.approx((7.853204624095838 / 4.730040744862704) ** 2, rel=1e-6)
    assert ship.length == 100.0


# ---------------------------------------------------------------------------
# modal integrator against a direct ODE integration
# ---------------------------------------------------------------------------
def test_modal_integrator_matches_direct_integration():
    dt = 1.0e-4
    t = np.arange(0, 6.0 + dt, dt)
    rng = np.random.default_rng(7)
    force = np.sin(2.0 * np.pi * 0.7 * t) ** 2 + 0.3 * rng.standard_normal(t.size)
    omega = 2.0 * math.pi * 1.2
    zeta = 0.03
    q, v = m.piecewise_linear_modal_response(t, force, omega, zeta)
    # independent RK4 integration of the same equations
    q_rk = np.zeros_like(t)
    v_rk = np.zeros_like(t)

    def forcing(tt: float) -> float:
        return float(np.interp(tt, t, force))

    def accel(tt, qq, vv):
        return forcing(tt) - 2.0 * zeta * omega * vv - omega * omega * qq

    for i in range(t.size - 1):
        h = t[i + 1] - t[i]
        t0 = t[i]
        k1q, k1v = v_rk[i], accel(t0, q_rk[i], v_rk[i])
        k2q = v_rk[i] + 0.5 * h * k1v
        k2v = accel(t0 + 0.5 * h, q_rk[i] + 0.5 * h * k1q, v_rk[i] + 0.5 * h * k1v)
        k3q = v_rk[i] + 0.5 * h * k2v
        k3v = accel(t0 + 0.5 * h, q_rk[i] + 0.5 * h * k2q, v_rk[i] + 0.5 * h * k2v)
        k4q = v_rk[i] + h * k3v
        k4v = accel(t0 + h, q_rk[i] + h * k3q, v_rk[i] + h * k3v)
        q_rk[i + 1] = q_rk[i] + h * (k1q + 2 * k2q + 2 * k3q + k4q) / 6.0
        v_rk[i + 1] = v_rk[i] + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    scale = np.max(np.abs(q_rk))
    assert np.max(np.abs(q - q_rk)) / scale < 1e-6
    assert np.max(np.abs(v - v_rk)) / np.max(np.abs(v_rk)) < 1e-5


# ---------------------------------------------------------------------------
# whipping decay time
# ---------------------------------------------------------------------------
def test_decay_time_matches_analytic(ship):
    model = m.WhippingModel(elements=48, modes=4, response_dt=0.01, duration=60.0)
    event = m.isolated_whipping_event(ship, model, 3.0e6, 0.013, duration=60.0)
    decay = m.whipping_decay_time(event, 0)
    assert decay["measured"] == 1.0
    assert decay["tau_decay"] > 0.0
    assert decay["relative_error"] < 0.02, decay
    assert decay["tau_decay"] == pytest.approx(decay["analytic_tau"], rel=0.02)


def test_decay_time_zero_without_event(ship, fast_model):
    t = np.arange(0.0, 2.0, fast_model.response_dt)
    response = m.modal_response_records(ship, fast_model, t, np.zeros_like(t))
    decay = m.whipping_decay_time(response, 0)
    assert decay["tau_decay"] == 0.0
    assert decay["measured"] == 0.0


# ---------------------------------------------------------------------------
# water-entry slam branch: added-mass momentum closure and scaling
# ---------------------------------------------------------------------------
def test_wave_slam_impulse_closure(ship):
    wave = m.WaveField(significant_height=10.0, peak_period=11.0)
    pulses = m.wave_slam_pulses(ship, wave, 5.0, 120.0)
    assert pulses, "the wave branch must fire for a sea state that emerges the bow"
    w = m.seakeeping_relative_motion(ship, wave, 5.0).re_entry_velocity()
    assert w > 0.0
    expected = m.added_mass_impulse(ship, w)
    for pulse in pulses:
        assert pulse.rise_impulse == pytest.approx(expected, rel=1e-9)


def test_von_karman_scaling(ship):
    base = m.von_karman_peak_force(ship, 3.0)
    assert m.von_karman_peak_force(ship, 6.0) == pytest.approx(4.0 * base, rel=1e-12)
    assert m.von_karman_peak_force(ship, 3.0, wagner=True) == pytest.approx(
        base * (0.5 * math.pi) ** 2, rel=1e-12
    )
    assert m.von_karman_peak_force(ship, 0.0) == 0.0


def test_no_emergence_gives_no_wave_slam(ship):
    wave = m.WaveField(significant_height=1.0, peak_period=6.0)
    motion = m.seakeeping_relative_motion(ship, wave, 5.0)
    assert not motion.encounters_emergence
    assert motion.re_entry_velocity() == 0.0
    assert m.wave_slam_pulses(ship, wave, 5.0, 30.0) == []


# ---------------------------------------------------------------------------
# ice impact branch: structural scaling of the declared crushing law
# ---------------------------------------------------------------------------
def test_ice_impact_exponents(ship):
    ice = m.IceField(thickness=0.5, concentration=0.6)
    thin = m.IceField(thickness=0.25, concentration=0.6)
    f = max(p.amplitude for p in m.ice_impact_pulses(ship, ice, 5.0, 120.0))
    f_thin = max(p.amplitude for p in m.ice_impact_pulses(ship, thin, 5.0, 120.0))
    exponent = math.log(f / f_thin) / math.log(0.5 / 0.25)
    assert exponent == pytest.approx(1.0 + ice.area_exponent, rel=0.35)


def test_ice_impacts_vanish_at_zero_speed_and_zero_concentration(ship):
    ice = m.IceField(thickness=0.5, concentration=0.6)
    assert m.ice_impact_pulses(ship, ice, 0.0, 60.0) == []
    assert m.ice_impact_pulses(ship, m.IceField(thickness=0.0, concentration=0.6), 5.0, 60.0) == []
    assert m.ice_impact_pulses(ship, m.IceField(thickness=0.5, concentration=0.0), 5.0, 60.0) == []


def test_ice_train_is_reproducible(ship):
    ice = m.IceField(thickness=0.5, concentration=0.6)
    a = [(p.t0, p.amplitude) for p in m.ice_impact_pulses(ship, ice, 5.0, 60.0)]
    b = [(p.t0, p.amplitude) for p in m.ice_impact_pulses(ship, ice, 5.0, 60.0)]
    assert a == b and a


# ---------------------------------------------------------------------------
# fatigue: S-N curve and rainflow/Miner accumulation
# ---------------------------------------------------------------------------
def test_sn_curve_knee_and_slopes():
    curve = m.SNCurve()
    # Published DNV-RP-C203 curve D in air: log a1 = 12.164 -> knee 52.642 MPa,
    # N(100 MPa) = 1.458814e6.  v1 pinned 53.4 MPa / 1.52e6 (log a1 = 12.182,
    # +4.19 % unconservative life); corrected in
    # docs/fatigue-sn-curve-consistency-audit.md.
    assert curve.knee_stress / 1e6 == pytest.approx(52.642115, rel=1e-6)
    assert curve.cycles_to_failure(100e6) == pytest.approx(1.458814e6, rel=1e-5)
    low = curve.cycles_to_failure(40e6)
    high = curve.cycles_to_failure(80e6)
    assert low > high
    with pytest.raises(ValueError):
        curve.cycles_to_failure(0.0)


def test_fatigue_of_constant_history_is_zero():
    result = m.fatigue_damage(np.full(100, 12.0e6))
    assert result["damage"] == 0.0


def test_fatigue_of_single_cycle_matches_sn_curve():
    peak = 100e6
    history = np.concatenate([np.linspace(0.0, peak, 200), np.linspace(peak, 0.0, 200)])
    result = m.fatigue_damage(history)
    expected = 1.0 / m.SNCurve().cycles_to_failure(peak)
    assert result["damage"] == pytest.approx(expected, rel=0.02)
    assert result["max_range_pa"] == pytest.approx(peak, rel=1e-9)


# ---------------------------------------------------------------------------
# the three hard negative controls
# ---------------------------------------------------------------------------
def test_negative_controls(ship):
    ice = m.IceField(thickness=0.5, concentration=0.6)
    wave = m.WaveField(significant_height=10.0, peak_period=11.0)
    model = m.WhippingModel(elements=24, modes=4, response_dt=0.02, duration=30.0)
    controls = m.negative_control_checks(ship, ice, wave, 5.0, model)
    assert controls["v_to_zero"]["relative_residual"] == 0.0
    assert controls["v_to_zero"]["tau_decay_ice_only_limit"] == 0.0
    assert controls["v_to_zero"]["tau_decay_ice_only_base"] > 0.0
    assert controls["h_to_zero"]["relative_residual"] < 1e-9
    assert controls["h_to_zero"]["peak_wave_branch"] > 0.0
    assert controls["c_to_zero"]["damage_ratio_limit"] == 0.0


# ---------------------------------------------------------------------------
# full case wiring
# ---------------------------------------------------------------------------
def test_full_case_reports_physical_numbers(ship):
    ice = m.IceField(thickness=0.5, concentration=0.6)
    wave = m.WaveField(significant_height=2.0, peak_period=7.0)
    model = m.WhippingModel(elements=24, modes=4, response_dt=0.02, duration=30.0)
    case = m.run_ice_slamming_case(ship, ice, wave, 5.144, model)
    assert case.peak_ice_force > 0.0
    assert case.peak_total_moment > 0.0
    assert case.damage_wave > 0.0
    assert case.damage_slam > 0.0
    assert case.damage_ratio > 0.0
    assert not case.motion.encounters_emergence  # 2 m sea: no bow emergence
    assert case.tau_decay > 0.0
    assert sum(case.modal_peak_share) == pytest.approx(1.0, rel=1e-9)
    # the whipping moment must exceed the quasi-static moment of the same load
    assert case.dynamic_amplification > 1.0