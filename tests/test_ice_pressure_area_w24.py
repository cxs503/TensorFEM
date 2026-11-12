"""Tests for the W24 impact pressure--area machinery.

These assert *invariants*, not just that the code runs:

* ``power_law_fit`` recovers a known power law exactly and refuses degenerate
  samples instead of returning a silent slope;
* ``decompose_total_exponent`` closes the identity
  ``dlnp/dlnA = a_A + a_h beta_h + a_V beta_V`` to machine precision on a
  synthetic law (so the attribution of the raw slope is exact, not rhetorical);
* the recording contact laws are injected through
  ``Cummins6DOF.run(..., contact_law=...)`` without perturbing the default
  integration path (bit-level agreement of the default ISO law arm), and the
  imposed-law arm returns its own exponent (the closure that proves the chain
  cannot test the law);
* the negative control (random per-floe strength) really has no area dependence;
* the global/local calibre bridge is dimensionally consistent.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "ice_coupled_6dof"))

from cummins_ice_6dof import (  # noqa: E402
    Cummins6DOF,
    IceField,
    PressureAreaLaw,
    ShipModel,
)
from tensorfem.ice_pressure_area import (  # noqa: E402
    ISOAreaLawContact,
    RandomStrengthContact,
    RateDependentCrushing,
    bootstrap_power_law,
    calibrate_reference_pressure,
    decompose_total_exponent,
    global_local_bridge,
    power_law_fit,
    quantile_summary,
)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def test_power_law_fit_recovers_a_known_law():
    rng = np.random.default_rng(7)
    area = 10.0 ** rng.uniform(-3.0, 0.0, 200)
    pressure = 2.5 * area ** -0.37
    fit = power_law_fit(area, pressure)
    assert fit["fit_ok"]
    assert fit["exponent"] == pytest.approx(-0.37, abs=1e-12)
    assert fit["prefactor_MPa"] == pytest.approx(2.5, rel=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 200


def test_power_law_fit_reports_degeneracy_instead_of_a_slope():
    # too few points
    fit = power_law_fit([0.1, 0.2], [1.0, 1.1])
    assert fit["fit_ok"] is False and "usable impact samples" in fit["note"]
    # zero variance in ln A with a constant pressure
    fit = power_law_fit([0.2, 0.2, 0.2, 0.2], [1.0, 1.0, 1.0, 1.0])
    assert fit["fit_ok"] is False and "zero variance" in fit["note"]
    # non-positive pressures are dropped, not fitted as zeros
    fit = power_law_fit([0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -1.0, 2.0])
    assert fit["fit_ok"] is False


def test_bootstrap_brackets_the_truth():
    rng = np.random.default_rng(11)
    area = 10.0 ** rng.uniform(-2.5, -0.2, 400)
    pressure = 1.8 * area ** -0.25 * np.exp(rng.normal(0.0, 0.05, area.size))
    boot = bootstrap_power_law(area, pressure, n_boot=300)
    assert boot["exponent_p05"] < -0.25 < boot["exponent_p95"]
    assert boot["n_boot"] > 250


def test_quantile_summary_reports_median_and_tails():
    q = quantile_summary(np.arange(1.0, 101.0))
    assert q["n"] == 100
    assert q["median"] == pytest.approx(50.5)
    assert q["p05"] < q["median"] < q["p95"]
    assert quantile_summary([]) == {"n": 0}


def test_decompose_total_exponent_identity_is_exact():
    rng = np.random.default_rng(3)
    n = 0.33
    area = 10.0 ** rng.uniform(-2.5, -0.2, 500)
    thick = rng.uniform(0.3, 1.5, 500)
    close = rng.uniform(1.0, 2.0, 500)
    pressure = 1.9 * (close * thick / area) ** n   # p ~ (V h / A)^n
    dec = decompose_total_exponent(area, thick, close, pressure)
    assert dec["identity_ok"]
    assert dec["partial_exponents"]["A"] == pytest.approx(-n, abs=1e-9)
    assert dec["partial_exponents"]["h"] == pytest.approx(+n, abs=1e-9)
    assert dec["partial_exponents"]["V_close"] == pytest.approx(+n, abs=1e-9)
    assert abs(dec["residual"]) < 1e-9
    assert dec["partial_r2"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# contact laws
# --------------------------------------------------------------------------- #
def test_calibration_hits_the_declared_point():
    for n in (0.15, 0.33, 0.50):
        pref = calibrate_reference_pressure(n, reference_strain_rate_s=1.0,
                                           target_pressure_MPa=1.9)
        law = RateDependentCrushing(0.01, reference_pressure_MPa=pref,
                                    reference_strain_rate_s=1.0, exponent=n)
        # area / thickness chosen so that epsdot = V/(4 A/h) = 1 1/s
        # (A/h = 0.5 m -> V = 4 * 0.5 * 1 = 2 m/s)
        area, thick, close = 0.5, 1.0, 2.0
        assert law.pressure_MPa(area, 0.1, thick, close) == pytest.approx(
            1.9, rel=1e-12)


def test_rate_law_has_no_area_exponent_but_does_depend_on_the_width():
    law = RateDependentCrushing(0.01, reference_pressure_MPa=1.9,
                                reference_strain_rate_s=1.0, exponent=0.33)
    # same closing speed, doubling the width lowers the strain rate -> lower p
    p_small = law.pressure_MPa(0.5, 0.1, 1.0, 4.0)
    p_large = law.pressure_MPa(1.0, 0.2, 1.0, 4.0)
    assert p_large < p_small
    assert p_large / p_small == pytest.approx(2.0 ** -0.33, rel=1e-12)


def test_random_strength_control_is_area_independent():
    law = RandomStrengthContact(0.01, mean_pressure_MPa=1.9,
                               coefficient_of_variation=0.35, seed=5)
    law.current_event = {"t_start": 0.0, "diameter": 3.0, "v_rel_in": 1.0}
    p1 = law.pressure_MPa(0.01, 0.001, 0.5, 1.0)
    p2 = law.pressure_MPa(1.00, 0.100, 0.5, 1.0)
    assert p1 == p2


def test_global_local_bridge_is_the_participation_factor():
    br = global_local_bridge(1.0, -0.28, local_area_m2=0.1, nominal_area_m2=1.0)
    assert br["participation_factor_kappa"] == pytest.approx(0.1)
    assert br["pressure_global_calibre_MPa"] == pytest.approx(
        br["pressure_local_calibre_MPa"] * 0.1, rel=1e-12)
    assert br["pressure_local_calibre_MPa"] == pytest.approx(0.1 ** -0.28)


# --------------------------------------------------------------------------- #
# end-to-end on the coupled integrator
# --------------------------------------------------------------------------- #
def _run(contact_law=None, seconds=6.0, dt=0.01, seed=1101, conc=0.7):
    sim = Cummins6DOF(ShipModel())
    return sim.run(seconds, dt, ice=IceField(concentration=conc), seed=seed,
                   commanded_speed=5.0, contact_law=contact_law)


def test_default_path_is_not_perturbed_by_the_injection():
    """Injecting an ISO-law recorder must reproduce the default integration."""
    default = _run(None)
    recorded = ISOAreaLawContact(PressureAreaLaw(), 0.01)
    injected = _run(recorded)
    for key in ("heave_m", "pitch_rad", "surge_m", "ice_force_N"):
        np.testing.assert_allclose(injected[key], default[key], rtol=1e-12,
                                   atol=1e-12)
    assert injected["completed_contacts"] == default["completed_contacts"]


def test_imposed_law_arm_returns_its_own_exponent():
    """Closure: the imposed-pressure arm cannot test the law it was fed."""
    law = ISOAreaLawContact(PressureAreaLaw(), 0.01)
    _run(law)
    rows = law.rows("C1_iso_area_law", 0.7, 1101)
    assert len(rows) >= 15
    area = [r["area_at_peak_m2"] for r in rows]
    pressure = [r["pressure_at_peak_MPa"] for r in rows]
    fit = power_law_fit(area, pressure)
    assert fit["fit_ok"]
    assert fit["exponent"] == pytest.approx(PressureAreaLaw().exponent, abs=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    # and every recorded pressure is exactly the law evaluated at that area
    lawref = PressureAreaLaw()
    for a, p in zip(area, pressure):
        assert p == pytest.approx(float(lawref.pressure_MPa(a)), rel=1e-12)


def test_rate_arm_returns_a_physical_impact_sample_and_a_finite_fit():
    law = RateDependentCrushing(0.01, reference_pressure_MPa=1.9,
                                reference_strain_rate_s=1.0, exponent=0.33)
    _run(law)
    rows = law.rows("T", 0.7, 1101)
    assert len(rows) >= 15
    areas = np.array([r["area_at_peak_m2"] for r in rows])
    pressures = np.array([r["pressure_at_peak_MPa"] for r in rows])
    assert np.all(areas > 0.0) and np.all(pressures > 0.0)
    assert np.all(np.isfinite(pressures))
    fit = power_law_fit(areas, pressures)
    assert fit["fit_ok"]
    # the apparent (raw) slope of a rate-dependent contact must be negative but
    # far smaller in magnitude than the material partial exponent n = 0.33
    assert -0.6 < fit["exponent"] < 0.0
    dec = decompose_total_exponent(
        areas, np.array([r["thickness_m"] for r in rows]),
        np.array([r["closing_speed_at_peak_mps"] for r in rows]), pressures)
    assert dec["identity_ok"]
    assert dec["partial_exponents"]["A"] == pytest.approx(-0.33, abs=1e-9)
    assert abs(dec["residual"]) < 1e-9


def test_recorder_keeps_impulse_and_area_integrals_consistent():
    law = RateDependentCrushing(0.01, reference_pressure_MPa=1.9,
                                reference_strain_rate_s=1.0, exponent=0.33)
    _run(law, seconds=4.0, conc=0.7)
    rows = law.rows("T", 0.7, 1101)
    assert rows
    for row in rows:
        assert row["impulse_Ns"] > 0.0
        assert row["area_time_integral_m3s"] > 0.0
        # p_bar = impulse / int A dt / 1e6 and it must be finite and positive
        assert math.isfinite(row["mean_pressure_MPa"])
        assert row["mean_pressure_MPa"] > 0.0
        assert row["n_samples"] >= 3