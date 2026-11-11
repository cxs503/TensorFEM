"""Tests for the W23 6-DOF Cummins / MMG / random-ice-field model.

These assert *invariants*, not just that the code runs:

* the closed-form ``A(w)``/``B(w)`` of the exponential kernel agree with a
  numerical Kramers-Kronig integration of ``K(tau)``;
* the zero-ice time-domain response reproduces the frequency-domain RAO;
* the per-impact momentum ledger closes, and it *fails loudly* when a channel is
  deliberately broken (mutation controls), i.e. the check has resolving power;
* the Weibull MLE recovers the parameters of synthetic Weibull samples;
* the swept-area encounter rate matches an independent geometric Monte Carlo.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks" / "ice_coupled_6dof"))

from cummins_ice_6dof import (  # noqa: E402
    Cummins6DOF,
    IceField,
    ShipModel,
    harmonic_amplitude,
    weibull_mle,
)


@pytest.fixture(scope="module")
def model():
    return ShipModel()


def test_kernel_closed_form_matches_kramers_kronig(model):
    tau = np.linspace(0.0, 120.0, 240001)
    kernel = model.radiation["heave"].kernel(tau)
    for omega in (0.3, 0.9):
        a_num = model.radiation["heave"].added_mass_inf - np.trapezoid(
            kernel * np.sin(omega * tau), tau) / omega
        b_num = np.trapezoid(kernel * np.cos(omega * tau), tau)
        assert a_num == pytest.approx(float(model.radiation["heave"].added_mass(omega)),
                                      rel=1e-5)
        assert b_num == pytest.approx(float(model.radiation["heave"].damping(omega)),
                                      rel=1e-5)


def test_kernel_is_passive_on_working_band(model):
    omega = np.linspace(0.05, 3.0, 200)
    for dof in ("heave", "pitch", "roll"):
        assert np.all(model.radiation[dof].damping(omega) >= 0.0)
        assert np.all(model.radiation[dof].added_mass(omega) > 0.0)


def test_zero_ice_time_domain_reproduces_frequency_domain_rao(model):
    for omega in (0.5, 1.0):
        fd = model.rao_frequency_domain(omega)
        sim = Cummins6DOF(model)
        exc = model.excitation(omega)
        periods = 25
        out = sim.run(periods * 2.0 * math.pi / omega, 0.02,
                      excitation=lambda t, e=exc, w=omega: e * math.cos(w * t))
        a3 = harmonic_amplitude(out["time"], out["heave_m"], omega, n_periods=8)
        a5 = harmonic_amplitude(out["time"], out["pitch_rad"], omega, n_periods=8)
        assert abs(a3 / fd["amplitude_abs"][2] - 1.0) < 0.01
        assert abs(a5 / fd["amplitude_abs"][4] - 1.0) < 0.01


def _run_ice(model, mutation=None, duration=40.0, seed=11):
    field = IceField(concentration=0.5, median_floe_diameter_m=3.0)
    sim = Cummins6DOF(model)
    return sim.run(duration, 0.01, ice=field, seed=seed, commanded_speed=5.0,
                   initial_velocity=5.0, mutation=mutation)


def test_momentum_ledger_closes(model):
    out = _run_ice(model)
    assert len(out["events"]) > 20
    resid = np.array([e["resid_max"] for e in out["events"]])
    assert np.max(resid) < 1e-3, f"momentum residual too large: {np.max(resid)}"


def test_momentum_ledger_detects_broken_channels(model):
    """Negative controls: a broken channel must blow the residual up."""
    nominal = np.max([e["resid_max"] for e in _run_ice(model)["events"]])
    for mutation in ("floe_reaction_half",):
        resid = np.max([e["resid_max"] for e in _run_ice(model, mutation)["events"]])
        assert resid > 50 * max(nominal, 1e-12), (mutation, resid, nominal)


def test_impulse_momentum_relation(model):
    """``J`` must match the reduced-mass momentum change of the pair.

    The effective hull mass seen by a vertical impulse at the contact point
    ``x_b`` is the harmonic combination of the heave and pitch inertias,
    ``1/m_eff = 1/(m+A33) + x_b^2/(Iyy+A55)``; with that mass the collision
    closure is within a few percent (the residual is the radiation history and
    the sway/roll mobility, which the scalar model ignores).
    """
    out = _run_ice(model)
    ratios = []
    m33 = model.mass_rigid[2] + model.radiation["heave"].added_mass_inf
    i55 = model.mass_rigid[4] + model.radiation["pitch"].added_mass_inf
    for e in out["events"]:
        m_eff = 1.0 / (1.0 / m33 + e["arm_x_m"] ** 2 / i55)
        m_red = m_eff * e["floe_mass_kg"] / (m_eff + e["floe_mass_kg"])
        expected = m_red * (e["v_rel_in_mps"] - e["v_rel_out_mps"])
        ratios.append(abs(e["J_contact_Ns"] / expected - 1.0))
    assert np.median(ratios) < 0.10


def test_weibull_mle_recovers_parameters():
    rng = np.random.default_rng(3)
    k_true, lam_true = 1.4, 0.012
    x = lam_true * rng.weibull(k_true, 4000)
    fit = weibull_mle(x)
    assert fit["shape_k"] == pytest.approx(k_true, rel=0.10)
    assert fit["scale_lambda"] == pytest.approx(lam_true, rel=0.05)
    assert fit["weibull_plot_r2"] > 0.99


def test_encounter_rate_matches_geometric_monte_carlo(model):
    field = IceField(concentration=0.5, median_floe_diameter_m=3.0)
    lam = field.encounter_rate(5.0, model.breadth_m)
    rng = np.random.default_rng(5)
    duration = 50000.0
    speed = 5.0
    length = speed * duration
    width = model.breadth_m + 4.0 * field.mean_floe_diameter
    n = int(rng.poisson(field.number_density() * length * width))
    d = rng.lognormal(math.log(field.median_floe_diameter_m), field.lognormal_sigma, n)
    y = rng.uniform(-0.5 * width, 0.5 * width, n)
    hits = int(np.count_nonzero(np.abs(y) <= 0.5 * (model.breadth_m + d)))
    assert hits / duration == pytest.approx(lam, rel=0.05)


def test_sampled_arrivals_are_poisson(model):
    field = IceField(concentration=0.5, median_floe_diameter_m=3.0)
    lam = field.encounter_rate(5.0, model.breadth_m)
    rng = np.random.default_rng(7)
    duration = 4000.0
    arrivals = field.sample_arrivals(duration, lam, rng)
    assert arrivals.size == pytest.approx(lam * duration, rel=0.06)
    gaps = np.diff(arrivals)
    assert gaps.size > 100
    assert np.std(gaps) / np.mean(gaps) == pytest.approx(1.0, abs=0.2)


def test_pressure_area_law_exponent(model):
    """``N(A)`` must scale as ``A**(1+ex)`` with a negative ``ex``."""
    from cummins_ice_6dof import PressureAreaLaw

    law = PressureAreaLaw()
    a1, a2 = 0.5, 4.0
    ratio = float(law.force_N(a2) / law.force_N(a1))
    assert ratio == pytest.approx((a2 / a1) ** (1.0 + law.exponent), rel=1e-12)
    assert law.exponent < 0.0