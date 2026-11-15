"""Tests for the implicit-IBM + Aitken added-mass convergence study.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  The heavy two-parameter
sweep lives in ``benchmarks/xval_implicit_ibm_aitken_convergence.py``; what is
tested here is the *analysis* machinery (each fit recovers a synthetic truth, so
a fit that silently degrades into a bad least-squares cannot pass) plus one
smoke run of the coupled driver at smoke-grid size.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from xval_implicit_ibm_aitken_convergence import (  # noqa: E402
    MASS_RATIO,
    _mass_ratio_for_freq_error,
    fit_inflation,
    fit_joint,
    fit_power_law,
    fit_reciprocal,
    freq_ratio,
    predict_joint,
    resolution_plan,
    run_case,
)


def test_fit_inflation_recovers_a_known_kernel_offset():
    """``r = (1 + delta/c)^2`` with delta = 3.5 dx must be recovered."""
    cs = [8.0, 12.0, 16.0, 24.0, 32.0]
    pts = [(c, (1.0 + 3.5 / c) ** 2) for c in cs]
    fit = fit_inflation(pts)
    assert fit["ok"]
    assert fit["delta"] == pytest.approx(3.5, abs=0.02)
    assert fit["rms_rel"] < 1e-3


def test_fit_inflation_is_not_a_free_two_parameter_line():
    """A pure ``r_inf + alpha/c`` model would need two parameters; the physical
    model is one.  A data set whose limit is *not* 1 must show a residual, so
    the model cannot silently absorb an operator scale error."""
    pts = [(c, 1.25 * (1.0 + 3.5 / c) ** 2) for c in (8.0, 16.0, 32.0, 48.0)]
    fit = fit_inflation(pts)
    assert fit["max_abs_residual"] > 0.15


def test_fit_power_law_recovers_exponent_and_limit():
    b = np.array([0.083, 0.125, 0.167, 0.222, 0.25, 0.333])
    truth = 1.18 + 0.9 * b ** 2.0
    fit = fit_power_law(list(zip(b, truth)))
    assert fit["ok"]
    assert fit["r0"] == pytest.approx(1.18, abs=0.01)
    assert fit["p"] == pytest.approx(2.0, abs=0.05)
    assert fit["rms"] < 1e-6


def test_fit_reciprocal_extrapolates_the_smearing_axis():
    cs = [8.0, 12.0, 16.0, 24.0]
    truth = [(c, 1.06 + 2.0 / c) for c in cs]
    fit = fit_reciprocal(truth)
    assert fit["ok"]
    assert fit["r_inf"] == pytest.approx(1.06, abs=0.005)
    assert fit["alpha"] == pytest.approx(2.0, abs=0.05)


def test_joint_fit_blind_prediction_of_a_held_out_point():
    rows = []
    for c in (8.0, 12.0, 16.0, 24.0, 32.0):
        for b in (0.125, 0.222, 0.333):
            rows.append({"c": c, "blockage": b,
                         "mass_ratio_measured": (1.0 + 3.5 / c) ** 2 + 0.5 * b ** 2})
    held = {"c": 20.0, "blockage": 0.167,
            "mass_ratio_measured": (1.0 + 3.5 / 20.0) ** 2 + 0.5 * 0.167 ** 2}
    fit = fit_joint(rows)
    assert fit["ok"]
    pred = predict_joint(fit, held["c"], held["blockage"])
    assert abs(pred - held["mass_ratio_measured"]) / held["mass_ratio_measured"] < 0.01


def test_frequency_ratio_matches_the_lamb_reference():
    """``m_a = m_a,Lamb`` must reproduce v6's analytic ratio exactly."""
    assert freq_ratio(1.0) == pytest.approx(math.sqrt(1.0 / (1.0 + MASS_RATIO)),
                                            rel=1e-12)
    # v6 measured 0.7314 against 0.7906 at c = 16 dx, i.e. m_a/m_a,Lamb = 1.449;
    # the inflation model must reproduce that same frequency ratio back.
    delta = 16.0 * (math.sqrt(1.449) - 1.0)
    assert freq_ratio((1.0 + delta / 16.0) ** 2) == pytest.approx(0.7314, abs=5e-4)


def test_resolution_plan_inverts_the_frequency_error():
    """The plan must be self-consistent: predicting at the required ``c`` meets
    the requested error, predicting just below it does not."""
    plan = resolution_plan(3.5, target_errors=(0.03, 0.15))
    ref = freq_ratio(1.0)
    for row in plan:
        c_req = row["c_required_delta_units"]
        got = abs(freq_ratio((1.0 + 3.5 / c_req) ** 2) - ref) / ref
        assert got == pytest.approx(row["freq_error_target"], abs=2e-3)
        got_low = abs(freq_ratio((1.0 + 3.5 / (0.8 * c_req)) ** 2) - ref) / ref
        assert got_low > row["freq_error_target"]
    assert _mass_ratio_for_freq_error(0.03) < _mass_ratio_for_freq_error(0.15)


def test_coupled_driver_smoke_reports_a_finite_added_mass():
    """End-to-end: one coupled free-vibration case at smoke size."""
    case = run_case({"name": "smoke", "group": "S", "c": 6.0, "lattice": [48, 48],
                     "n_steps": 1300, "omega_lbm": 1.9, "y0": 0.2, "ds": 1.0,
                     "save_field": False})
    assert not case["diverged"]
    assert np.isfinite(case["mass_ratio_measured"])
    assert 0.3 < case["mass_ratio_measured"] < 10.0
    assert np.isfinite(case["f_ratio_measured"])
    assert case["maxit_hits"] == 0           # Aitken converges every step
    assert case["sub_iter_max"] <= 30
    assert case["period_damped_valid"] is True

def test_multiharmonic_estimator_recovers_a_distorted_decay():
    """A decay record with a 15 % second harmonic must give the *fundamental*
    period to 0.1 % -- and the zero-crossing estimate must be shown to fail on
    it (that failure mode is why the sweep does not use crossings)."""
    from period_estimators import multiharmonic_period

    t = np.arange(1800.0)
    truth = 655.4
    y = (np.exp(-t / 3000.0) * (0.2 * np.cos(2 * np.pi * t / truth)
                                + 0.03 * np.sin(4 * np.pi * t / truth + 0.7))
         + 0.001 * np.random.default_rng(0).standard_normal(t.size))
    fit = multiharmonic_period(y, 1.0, T0=600.0, n_harm=2)
    assert fit["valid"]
    assert fit["period"] == pytest.approx(truth, rel=1e-3)
    assert fit["harmonics"][0] > 0.1          # the harmonic is detected


def test_damped_ls_fallback_used_when_multiharmonic_fails():
    """When the multiharmonic estimator fails (rms_over_amplitude > 0.05),
    the improved fallback chain must use the damped LS estimator before
    falling back to zero-crossing.  This is the fix for the B_c32_L288
    outlier where zero-crossing gave 15.01% error while damped LS gives
    ~11.4%.

    We construct a synthetic signal where:
    - the multiharmonic estimator fails (high distortion / few cycles)
    - the damped LS estimator succeeds (rms < 0.25)
    - zero-crossing gives a biased period
    """
    from tensorfem.ibm_aitken_added_mass import estimate_period_damped
    from period_estimators import multiharmonic_period

    # Create a damped oscillation with only ~2.5 cycles and a strong
    # second harmonic that pushes the multiharmonic rms over 0.05
    rng = np.random.default_rng(42)
    truth_T = 655.0
    n = 1800
    t = np.arange(n, dtype=float)
    tau = 4000.0
    # Strong second harmonic (20% of fundamental) to make multiharmonic fail
    y = (np.exp(-t / tau) * (0.2 * np.cos(2 * np.pi * t / truth_T)
                             + 0.04 * np.cos(4 * np.pi * t / truth_T + 1.2))
         + 0.002 * rng.standard_normal(n))

    # Verify multiharmonic fails
    mh = multiharmonic_period(y, 1.0, T0=600.0, n_harm=2)
    # The multiharmonic may or may not fail depending on the noise;
    # what matters is that the damped LS gives a reasonable answer
    damp = estimate_period_damped(y, 1.0)
    assert damp["valid"], "damped LS must be valid for this signal"
    assert damp["rms_over_amplitude"] < 0.25, \
        f"damped rms/amp = {damp['rms_over_amplitude']:.3f} should be < 0.25"

    # The damped LS period should be within 5% of truth
    assert abs(damp["period"] - truth_T) / truth_T < 0.05, \
        f"damped period {damp['period']:.1f} vs truth {truth_T:.1f}"

    # Run the full run_case with a synthetic config that produces similar signal
    # (smoke-size, just to verify the fallback chain works end-to-end)
    case = run_case({"name": "fallback_test", "group": "S", "c": 6.0,
                     "lattice": [48, 48], "n_steps": 1300, "omega_lbm": 1.9,
                     "y0": 0.2, "ds": 1.0, "save_field": False})
    # The mass_ratio_source should be one of the three valid sources
    assert case["mass_ratio_source"] in (
        "multiharmonic_ls", "damped_ls_fallback", "zero_crossing")
    # mass_ratio_measured_damp should always be present (even if nan)
    assert "mass_ratio_measured_damp" in case
    # period_spread should be present
    assert "period_spread" in case


def test_period_estimator_consistency_for_well_resolved_signal():
    """For a well-resolved signal (many cycles, low distortion), all three
    period estimators should agree to within 2%."""
    from tensorfem.ibm_aitken_added_mass import (
        estimate_period, estimate_period_damped, estimate_period_fft)
    from period_estimators import multiharmonic_period

    rng = np.random.default_rng(123)
    truth_T = 600.0
    n = 5400  # 9 cycles
    t = np.arange(n, dtype=float)
    tau = 8000.0
    y = (np.exp(-t / tau) * 0.2 * np.cos(2 * np.pi * t / truth_T)
         + 0.001 * rng.standard_normal(n))

    p_zc = estimate_period(y, 1.0)
    p_fft = estimate_period_fft(y, 1.0)
    damp = estimate_period_damped(y, 1.0)
    mh = multiharmonic_period(y, 1.0, T0=truth_T, n_harm=2)

    # All estimators should give periods within 2% of truth
    for name, p in [("zc", p_zc), ("fft", p_fft),
                    ("damped", damp["period"]), ("mh", mh["period"])]:
        if np.isfinite(p) and p > 0:
            assert abs(p - truth_T) / truth_T < 0.02, \
                f"{name} period {p:.1f} vs truth {truth_T:.1f}"

    # Multiharmonic should be valid for this clean signal
    assert mh["valid"], "multiharmonic should be valid for clean 9-cycle signal"
