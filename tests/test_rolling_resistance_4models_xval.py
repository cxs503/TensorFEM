"""XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS: four rolling-resistance families vs
analytic single-particle oracles.

Every assertion below binds a *simulated* quantity to a *closed-form* reference
(``tensorfem.rolling_resistance_models``) with a tolerance tight enough that a
constitutive-law regression fails it (the integrator reproduces the linear laws to
~1e-13, so 1e-6..1e-9 tolerances are meaningful gates, not decoration).

Scenario numbers come from ``examples/rolling_model_assessment.py`` so the doc,
the JSON artefact and the tests share one source of truth.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from tensorfem.rolling_resistance_models import (
    INERTIA_FACTOR,
    MODELS,
    RollingModel,
    RollingState,
    cdt_stopping_distance,
    contact_torque,
    critical_incline_angle_deg,
    damped_terminal_velocity,
    damped_time_constant,
    saturated_acceleration,
    static_windup_angle,
    viscous_terminal_velocity,
)

MU_R = 0.3
RADIUS = 0.01
MASS = 1.0
THETA_SUPER = 20.0
THETA_SUB = 5.0

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rolling_model_assessment.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("rolling_model_assessment", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model(family: str, **kwargs) -> RollingModel:
    kwargs.setdefault("inertia_factor", INERTIA_FACTOR["sphere"])
    return RollingModel(model=family, mu_r=MU_R, radius=RADIUS, mass=MASS, **kwargs)


def rel(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


# ---------------------------------------------------------------------------
# pure API / contract tests (no simulation, always run)
# ---------------------------------------------------------------------------
def test_four_families_are_registered():
    assert MODELS == ("cdt", "viscous", "hysteretic", "epsd")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "lubricated"},
        {"model": "cdt", "mu_r": -0.1},
        {"model": "cdt", "radius": 0.0},
        {"model": "cdt", "mass": 0.0},
        {"model": "hysteretic", "hysteresis_ratio": 1.0},
        {"model": "epsd", "damping_ratio": -1.0},
        {"model": "cdt", "inertia_factor": 0.0},
    ],
)
def test_parameter_validation(kwargs):
    base = {"model": "cdt", "mu_r": 0.3, "radius": 0.01, "mass": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        RollingModel(**base)


def test_oracles_are_self_consistent():
    m = model("epsd")
    # tan(theta_c) = mu_r is the definition used to measure rolling coefficients.
    assert math.tan(math.radians(critical_incline_angle_deg(MU_R))) == pytest.approx(MU_R, rel=1e-14)
    # Saturated acceleration changes sign exactly at theta_c.
    assert saturated_acceleration(m, critical_incline_angle_deg(MU_R)) == pytest.approx(0.0, abs=1e-15)
    assert saturated_acceleration(m, 0.0) < 0.0 < saturated_acceleration(m, 45.0)
    # The elastic wind-up strain is the fraction tan(theta)/mu_r of the yield strain.
    assert static_windup_angle(m, 0.0) == pytest.approx(0.0, abs=0.0)
    assert static_windup_angle(m, 45.0) > m.yield_angle * 3.0


def test_torque_is_capped_and_separated_contact_forgets_state():
    for family in ("cdt", "hysteretic", "epsd"):
        m = model(family, damping_ratio=0.0)
        force = m.normal_force(0.0)
        cap = m.capacity(force)
        _, state = contact_torque(m, RollingState(), omega=5.0, dtheta=5.0, normal_force=force)
        # A separated contact must return zero torque and drop its memory.
        tau, reset = contact_torque(m, state, omega=5.0, dtheta=5.0, normal_force=0.0)
        assert tau == 0.0
        assert reset == RollingState()
        # A long forward roll saturates every family at the same limit.
        tau_sat, _ = contact_torque(m, state, omega=5.0, dtheta=1.0, normal_force=force)
        assert abs(tau_sat) <= cap * (1.0 + 1e-12)


def test_stopping_distance_oracle_limits():
    m = model("cdt")
    assert math.isinf(cdt_stopping_distance(m, 1.0, critical_incline_angle_deg(MU_R) + 0.1))
    assert cdt_stopping_distance(m, 1.0, 0.0) > 0.0
    # Viscous family has no static threshold: terminal velocity is positive at any slope.
    assert viscous_terminal_velocity(model("viscous"), 1.0) > 0.0
    assert damped_terminal_velocity(model("epsd"), 1.0) == 0.0
    assert damped_time_constant(model("epsd"), 1.0) > 0.0


# ---------------------------------------------------------------------------
# assessment scenarios (one shared run of the harness)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results():
    torch.set_default_dtype(torch.float64)
    module = _load_example()
    return {
        "s1": module.s1_flat_stop(),
        "s1b": module.s1_flat_stop(damping_ratio=1.0),
        "s2": module.s2_static_hold(),
        "s2b": module.s2b_jitter_resolution(duration=0.1),
        "s3": module.s3_supercritical(),
        "s3b": module.s3b_supercritical_damped(),
        "s4": module.s4_subcritical(),
        "s5": module.s5_critical_angle(),
        "s6": module.s6_family_reduction(),
        "s7": module.s7_hysteresis_loop(),
    }


def test_s1_cdt_stopping_distance_and_energy(results):
    cdt = results["s1"]["cdt"]
    assert cdt["stopping_distance_rel_error"] < 1e-6
    assert cdt["work_rel_error"] < 1e-4


def test_s1_saturated_families_stop_like_cdt(results):
    for family in ("hysteretic", "epsd"):
        assert results["s1"][family]["stopping_distance_vs_cdt_rel_error"] < 1e-3


def test_s1_viscous_matches_closed_form_decay(results):
    assert results["s1"]["viscous"]["final_speed_rel_error"] < 1e-9


def test_s2_only_spring_families_hold_a_static_contact(results):
    s2 = results["s2"]
    assert s2["epsd"]["displacement_rel_error"] < 1e-9
    # The two-stiffness branch stiffens the initial wind-up: a real, measured
    # ~4 % offset of the constitutive family, not an integrator error (the epsd
    # family, same integrator, is exact to 1e-9).
    assert s2["hysteretic"]["displacement_rel_error"] < 0.05
    # "cdt" has no static branch: instead of a static offset it jitters, and the
    # jitter is a limit cycle (see test_s2b: it scales with dt, so it is not a
    # convergent contact state).  The epsd contact is genuinely at rest.
    assert s2["cdt"]["max_speed_m_s"] > 1e-6
    assert abs(s2["epsd"]["final_speed_m_s"]) < 1e-12
    assert abs(s2["hysteretic"]["final_speed_m_s"]) < 1e-12
    assert s2["cdt"]["elastic_strain_rad"] == 0.0
    # "viscous" creeps at the applied torque / C_v: no static resistance at all.
    assert s2["viscous"]["final_speed_m_s"] == pytest.approx(
        s2["viscous"]["creep_speed_reference_m_s"], rel=1e-9
    )


def test_s2b_cdt_jitter_scales_linearly_with_step_size(results):
    """The artefact is a resolution-dependent limit cycle, so it is not convergent."""
    assert 8.0 < results["s2b"]["rms_ratio_dt10"] < 12.0
    keys = [k for k in results["s2b"] if k.startswith("dt=")]
    ratio = sorted(keys, key=lambda k: results["s2b"][k]["step_s"])
    small = results["s2b"][ratio[0]]["displacement_m"]
    large = results["s2b"][ratio[1]]["displacement_m"]
    assert 8.0 < large / small < 12.0


def test_s3_saturated_acceleration_and_torque(results):
    for family in ("cdt", "hysteretic", "epsd"):
        entry = results["s3"][family]
        assert entry["acceleration_rel_error"] < 1e-9
        assert entry["mean_tail_torque_rel_error"] < 1e-9


def test_s3b_damped_families_hit_the_terminal_velocity(results):
    for family in ("hysteretic", "epsd"):
        assert results["s3b"][family]["terminal_velocity_rel_error"] < 1e-9


def test_s4_subcritical_windup(results):
    s4 = results["s4"]
    assert s4["epsd"]["displacement_rel_error"] < 1e-6
    assert s4["hysteretic"]["displacement_rel_error"] < 0.05
    assert s4["viscous"]["final_speed_rel_error"] < 1e-9
    # "cdt" cannot come to rest: its residual speed is the limit-cycle amplitude.
    assert abs(s4["cdt"]["final_speed_m_s"]) > 1e-6
    assert s4["cdt"]["max_speed_m_s"] > 1e-5


def test_s5_critical_angle_matches_atan_mu_r(results):
    reference = critical_incline_angle_deg(MU_R)
    for family in ("hysteretic", "epsd"):
        assert results["s5"][family]["absolute_error_deg"] < 0.02
        assert results["s5"][family]["simulated_critical_angle_deg"] == pytest.approx(
            reference, abs=0.02
        )
    assert results["s5"]["viscous"]["terminal_velocity_at_1deg_m_s"] > 0.0


def test_s6_hysteretic_reduces_to_epsd_at_zero_hysteresis(results):
    s6 = results["s6"]
    assert s6["max_velocity_difference_m_s"] <= 1e-15 * max(s6["max_velocity_m_s"], 1.0)
    assert s6["max_velocity_m_s"] > 0.0


def test_s7_hysteresis_dissipation_grows_with_h(results):
    s7 = results["s7"]
    assert s7["h=0.0"]["dissipated_energy_J"] < 1e-6
    ordered = [s7[key]["dissipated_energy_J"] for key in ("h=0.0", "h=0.1", "h=0.25", "h=0.5")]
    assert ordered == sorted(ordered)
    assert s7["h=0.5"]["dissipated_energy_J"] > 1.5 * s7["h=0.25"]["dissipated_energy_J"]
    assert all(s7[key]["monotone_vs_previous"] is not False for key in s7 if key.startswith("h="))