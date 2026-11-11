"""Tests for the ISO 19906 pressure--area contact source in a 6-DOF integration."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorfem.iso19906_ice_6dof import (
    damping_dissipation_crosscheck,
    CaseConfig,
    IceEdgeContact,
    ISO19906PressureArea,
    Vessel6DOF,
    case_statistics,
    contact_area_m2,
    count_load_cycles,
    energy_balance,
    law_from_anchor,
    load_anchor_parameters,
    natural_periods,
    run_case,
    spearman_rank,
)


# --------------------------------------------------------------------------- #
# pressure--area law
# --------------------------------------------------------------------------- #
def test_pressure_area_law_matches_closed_form():
    law = ISO19906PressureArea(reference_pressure_MPa=1.0, reference_area_m2=1.0, exponent=-0.28)
    # p(1 m2) = C_R by construction; the law is monotone decreasing in area
    assert law.pressure_MPa(1.0) == pytest.approx(1.0)
    areas = [0.1, 0.5, 1.0, 2.0, 10.0]
    pressures = [law.pressure_MPa(a) for a in areas]
    assert all(b < a for a, b in zip(pressures[:-1], pressures[1:]))
    # closed form check away from the reference area
    assert law.pressure_MPa(4.0) == pytest.approx(4.0**-0.28)
    # contact force N = p*A is convex/monotone increasing with a sub-linear exponent
    forces = [law.force_N(a) for a in areas]
    assert all(b > a for a, b in zip(forces[:-1], forces[1:]))
    assert law.force_area_exponent == pytest.approx(0.72)
    assert forces[-1] / forces[2] == pytest.approx((10.0 / 1.0) ** 0.72, rel=1e-12)


def test_pressure_area_force_scales_exactly_with_reference_pressure():
    """Absolute loads are linear in C_R, so conclusions are anchor-independent."""
    base = ISO19906PressureArea(reference_pressure_MPa=1.0, exponent=-0.28)
    scaled = ISO19906PressureArea(reference_pressure_MPa=2.5, exponent=-0.28)
    for area in (0.05, 0.3, 1.0, 7.0):
        assert scaled.force_N(area) == pytest.approx(2.5 * base.force_N(area))


def test_global_pressure_branch_requires_negative_exponent():
    with pytest.raises(ValueError, match="negative exponent"):
        ISO19906PressureArea(exponent=0.5, scope="global")
    # the local branch is explicitly a different convention and is allowed
    ISO19906PressureArea(exponent=0.5, scope="local")
    with pytest.raises(ValueError):
        ISO19906PressureArea(reference_pressure_MPa=-1.0)
    with pytest.raises(ValueError):
        ISO19906PressureArea(scope="anything")


def test_pressure_rejects_non_positive_area():
    law = ISO19906PressureArea()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            law.pressure_MPa(bad)
        with pytest.raises(ValueError):
            law.force_N(bad)


# --------------------------------------------------------------------------- #
# contact geometry
# --------------------------------------------------------------------------- #
def test_wedge_contact_area_geometry():
    contact = IceEdgeContact(half_angle_deg=40.0, breadth_m=20.0, contact_height_ratio=1.0)
    assert contact.area_m2(0.0, 1.0) == 0.0
    width = 2.0 * 0.25 * math.tan(math.radians(40.0))
    assert contact.contact_width_m(0.25) == pytest.approx(width)
    assert contact.area_m2(0.25, 1.5) == pytest.approx(width * 1.5)
    assert contact_area_m2(contact, 0.25, 1.5) == pytest.approx(width * 1.5)
    # the width saturates at the hull breadth
    assert contact.contact_width_m(100.0) == pytest.approx(20.0)
    # monotone in penetration, linear in thickness
    assert contact.area_m2(0.3, 1.0) > contact.area_m2(0.2, 1.0)
    assert contact.area_m2(0.2, 2.0) == pytest.approx(2.0 * contact.area_m2(0.2, 1.0))


def test_contact_validation():
    with pytest.raises(ValueError):
        IceEdgeContact(residual_ratio=1.0)
    with pytest.raises(ValueError):
        IceEdgeContact(half_angle_deg=90.0)
    with pytest.raises(ValueError):
        IceEdgeContact(normal_obliquity_deg=90.0)
    with pytest.raises(ValueError):
        IceEdgeContact(spall_depth_m=0.0)


# --------------------------------------------------------------------------- #
# vessel
# --------------------------------------------------------------------------- #
def test_vessel_mass_matrix_and_natural_periods():
    vessel = Vessel6DOF()
    assert vessel.mass_kg == pytest.approx(1025.0 * 0.70 * 100.0 * 20.0 * 6.0)
    mass = vessel.mass_matrix()
    assert np.allclose(mass, mass.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)
    periods = natural_periods(vessel)
    assert 3.0 < periods["heave"] < 12.0
    assert 5.0 < periods["roll"] < 30.0
    assert 4.0 < periods["pitch"] < 15.0
    with pytest.raises(ValueError):
        Vessel6DOF(z_ice_line_m=1.0)


# --------------------------------------------------------------------------- #
# energy budget
# --------------------------------------------------------------------------- #
def _case(thickness=1.0, speed=0.8, **kwargs):
    law = kwargs.pop("law", ISO19906PressureArea())
    contact = kwargs.pop("contact", IceEdgeContact(normal_obliquity_deg=20.0))
    vessel = kwargs.pop("vessel", Vessel6DOF())
    return CaseConfig(
        thickness_m=thickness,
        commanded_speed_mps=speed,
        law=law,
        contact=contact,
        vessel=vessel,
        **kwargs,
    )


def test_bookkeeping_identity_and_independent_quadrature_agree():
    """Velocity-Verlet makes the displacement quadrature exact; the independent
    power quadrature must agree to well below 5% for the same run."""
    result = run_case(_case(), dt_s=0.01, duration_s=20.0, transient_s=5.0)
    balance = energy_balance(result)
    window = balance["statistics_window"]
    assert abs(window["displacement_quadrature_residual_J"]) < 1e-3 * (
        abs(window["work_thrust_J"]) + 1.0
    )
    assert abs(window["residual_fraction_of_thrust_work"]) < 5.0e-3
    assert abs(window["power_quadrature_residual_fraction"]) < 5.0e-3
    # the budget must actually contain all the terms
    assert window["work_ice_normal_J"] < 0.0
    assert window["work_damping_J"] < 0.0
    assert window["work_thrust_J"] > 0.0


def test_energy_budget_without_ice_reduces_to_thrust_damping_balance():
    """Positive control: with a vanishing area-law pressure the ship is a
    thrust/damping balance, and the budget must close on that alone."""
    law = ISO19906PressureArea(reference_pressure_MPa=1.0e-9, exponent=-0.28)
    result = run_case(
        _case(law=law, contact=IceEdgeContact(normal_obliquity_deg=0.0, friction=0.0)),
        dt_s=0.01,
        duration_s=20.0,
        transient_s=5.0,
    )
    window = energy_balance(result)["statistics_window"]
    assert abs(window["work_ice_normal_J"]) < 1.0
    assert window["work_thrust_J"] + window["work_damping_J"] == pytest.approx(
        window["kinetic_energy_change_J"] + window["potential_energy_change_J"], rel=1e-6
    )


def test_energy_metric_detects_a_mis_accounted_ice_force():
    """The budget must be able to fail: inflating the accumulated ice work by 6%
    has to push the residual past the 5% criterion."""
    case = _case()
    clean = energy_balance(run_case(case, dt_s=0.01, duration_s=20.0, transient_s=5.0))[
        "statistics_window"
    ]["residual_fraction_of_thrust_work"]
    mutated = energy_balance(
        run_case(case, dt_s=0.01, duration_s=20.0, transient_s=5.0, diagnostic_work_scale=1.06)
    )["statistics_window"]["residual_fraction_of_thrust_work"]
    assert abs(clean) < 1.0e-3
    assert abs(mutated) > 0.05
    # the reported residual tracks the injected error roughly one-for-one
    assert abs(mutated) == pytest.approx(0.06, abs=0.01)


def test_damping_dissipation_matches_the_power_law_integral():
    case = _case()
    result = run_case(case, dt_s=0.01, duration_s=20.0, transient_s=5.0)
    check = damping_dissipation_crosscheck(result, case.vessel)
    assert check["power_law_integral_J"] > 0.0
    assert abs(check["relative_difference"]) < 1.0e-3


def test_energy_residual_shrinks_with_time_step():
    case = _case()
    coarse = run_case(case, dt_s=0.02, duration_s=20.0, transient_s=5.0)
    fine = run_case(case, dt_s=0.01, duration_s=20.0, transient_s=5.0)
    coarse_res = abs(
        energy_balance(coarse)["statistics_window"]["residual_fraction_of_thrust_work"]
    )
    fine_res = abs(energy_balance(fine)["statistics_window"]["residual_fraction_of_thrust_work"])
    assert fine_res < 0.05
    assert fine_res < coarse_res


# --------------------------------------------------------------------------- #
# dynamics / statistics
# --------------------------------------------------------------------------- #
def test_load_statistics_and_cycles_are_physical():
    result = run_case(_case(), dt_s=0.01, duration_s=20.0, transient_s=5.0)
    stats = case_statistics(result)
    assert stats["spall_events"] > 5
    assert stats["cycle_count"] > 5
    assert 1.0 < stats["peak_to_mean"] < 4.0
    assert stats["max_pressure_MPa"] > stats["min_pressure_MPa"] > 0.0
    assert 0.0 < stats["mean_speed_mps"] <= 0.8
    assert stats["exceedance_count"] > 0


def test_mean_load_increases_with_ice_thickness():
    means = []
    for thickness in (0.6, 1.0, 1.5):
        result = run_case(_case(thickness=thickness), dt_s=0.02, duration_s=15.0, transient_s=5.0)
        means.append(case_statistics(result)["mean_load_MN"])
    assert means[0] < means[1] < means[2]
    # N ~ h**(1+ex) = h**0.72: the ratio between 0.6 m and 1.5 m is bracketed
    assert 1.45 < means[2] / means[0] < 2.1


def test_run_is_deterministic():
    case = _case()
    first = run_case(case, dt_s=0.02, duration_s=6.0, transient_s=2.0)
    second = run_case(case, dt_s=0.02, duration_s=6.0, transient_s=2.0)
    assert np.array_equal(first.contact_force_N, second.contact_force_N)
    assert np.array_equal(first.displacement, second.displacement)


def test_control_time_step_is_stable_and_converged():
    """Halving dt must move the mean load by less than 3% (criterion C3)."""
    case = _case()
    coarse = case_statistics(run_case(case, dt_s=0.02, duration_s=20.0, transient_s=5.0))
    fine = case_statistics(run_case(case, dt_s=0.01, duration_s=20.0, transient_s=5.0))
    change = abs(fine["mean_load_MN"] - coarse["mean_load_MN"]) / coarse["mean_load_MN"]
    assert change < 0.03


def test_input_validation():
    with pytest.raises(ValueError):
        _case(thickness=0.0)
    with pytest.raises(ValueError):
        _case(speed=-1.0)
    with pytest.raises(ValueError):
        run_case(_case(), dt_s=0.0)
    with pytest.raises(ValueError):
        run_case(_case(), dt_s=10.0, duration_s=1.0)


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def test_count_load_cycles_on_synthetic_saw_tooth():
    n_cycles = 7
    # each ramp rises 0 -> 0.95 and drops back to 0; the trailing decay makes
    # the last peak an interior turning point (edge peaks are not counted)
    samples = np.concatenate(
        [np.linspace(0.0, 0.95, 20, endpoint=False) for _ in range(n_cycles)] + [[0.95, 0.4, 0.0]]
    )
    count, peaks = count_load_cycles(samples)
    assert count == n_cycles
    assert peaks.size == n_cycles


def test_count_load_cycles_ignores_flat_and_noise_free_signals():
    assert count_load_cycles(np.linspace(0.0, 1.0, 100))[0] == 0
    assert count_load_cycles(np.zeros(50))[0] == 0


def test_spearman_exact_permutation_p_value():
    perfect = spearman_rank([0.6, 1.0, 1.5], [1.0, 1.4, 1.9])
    assert perfect["rho"] == pytest.approx(1.0)
    # exact two-sided permutation p for a monotone triple is 2/6
    assert perfect["p_value"] == pytest.approx(2.0 / 6.0)
    reversed_ = spearman_rank([0.6, 1.0, 1.5], [1.9, 1.4, 1.0])
    assert reversed_["rho"] == pytest.approx(-1.0)
    assert reversed_["p_value"] == pytest.approx(2.0 / 6.0)
    six = spearman_rank([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    assert six["n"] == 6
    assert six["rho"] == pytest.approx(1.0)
    assert six["p_value"] == pytest.approx(2.0 / 720.0)
    # tied x (the pooled thickness ensemble) uses average ranks, so rho < 1
    tied = spearman_rank([0.6, 0.6, 1.0, 1.0, 1.5, 1.5], [1, 2, 3, 4, 5, 6])
    assert tied["rho"] == pytest.approx(0.9561828874675149)
    # ties make the exact permutation p conservative: every relabelling inside a
    # tie group reproduces |rho| and is counted, giving 16/720 rather than 2/720
    assert tied["p_value"] == pytest.approx(16.0 / 720.0)
    with pytest.raises(ValueError):
        spearman_rank([1.0, 2.0], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# anchor handling
# --------------------------------------------------------------------------- #
def test_anchor_missing_file_is_reported_not_invented(tmp_path):
    payload = load_anchor_parameters()
    assert payload["status"] == "pending_transcription"
    assert payload["anchor_id"] == "AN-ISO19906"
    payload = load_anchor_parameters(tmp_path / "absent.json")
    assert payload["status"] == "pending_transcription"
    assert "not found" in payload["note"]


def test_anchor_transcription_is_used_verbatim(tmp_path):
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps(
            {
                "anchor_id": "AN-ISO19906",
                "C_R_MPa": 1.7,
                "A_0_m2": 1.0,
                "exponent": -0.31,
                "source": "unit-test transcription",
            }
        ),
        encoding="utf-8",
    )
    payload = load_anchor_parameters(anchor)
    assert payload["status"] == "transcribed"
    law = law_from_anchor(payload)
    assert law.reference_pressure_MPa == pytest.approx(1.7)
    assert law.exponent == pytest.approx(-0.31)
    assert "unit-test transcription" in law.provenance
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"C_R_MPa": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_anchor_parameters(broken)
