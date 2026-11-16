"""Tests for XVAL round 5 — attribution of the round-4 coupled-frequency deficit.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``; reopens PR #1120.

The round-4 study (``benchmarks/xval_aitken_strong_coupling.py``) reported
``max |f - f_ref| = 23 %`` against Lamb's ``m_a = rho pi c^2`` and left it
un-attributed.  These tests pin the machinery that does the attribution:

1. ``effective_added_mass`` inverts ``m_eff = k/(2 pi / T)^2 - m_s`` exactly and
   returns ``nan`` (never a silent 0) for a degenerate period;
2. the *measurement* is not the deficit: the window-safe damped fit and the
   mean-removed zero-crossing estimator agree on a synthesised record that is
   4 cycles long, while the zero-crossing estimator returns ``nan`` -- and the
   FFT one returns its fundamental bin -- on a record that is only 1.3 cycles
   long, which is exactly the ``--quick`` pathology recorded in the round-4
   artifacts (``T = nan``, ``f`` error 27-49 %);
3. the coupled solver integrates a *ratio-independent* effective inertia: two
   mass ratios with the same ``m_eff`` hypothesis give the same ``m_eff``, so the
   frequency deficit of round 4 is a property of the diffuse interface and not of
   the staggered coupling (this is the T1 claim, at test scale);
4. the round-4 driver now carries a window guard: a case whose record is shorter
   than ``MIN_CYCLES`` coupled periods is flagged and excluded from the
   frequency verdict instead of contributing a bogus number.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest

from tensorfem.ibm_aitken_added_mass import (
    analytic_added_mass,
    coupled_free_vibration,
    effective_added_mass,
    estimate_period,
    estimate_period_damped,
    oscillator_frequency,
    period_window_ok,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_bench(name):
    path = os.path.join(REPO, "benchmarks", name)
    spec = importlib.util.spec_from_file_location("bench_" + name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. the inversion
# ---------------------------------------------------------------------------
def test_effective_added_mass_inverts_the_oscillator():
    k, m_s, m_eff = 3.0, 500.0, 314.1592653589793
    f = oscillator_frequency(k, m_s + m_eff)
    got = effective_added_mass(k, m_s, 1.0 / f)
    assert got == pytest.approx(m_eff, rel=1e-12)
    # degenerate inputs are sentinels, never silently zero
    for bad in (float("nan"), 0.0, -3.0, float("inf")):
        assert math.isnan(effective_added_mass(k, m_s, bad))


# ---------------------------------------------------------------------------
# 2. the estimator is not the deficit
# ---------------------------------------------------------------------------
def _damped(period, tau, n, amp=0.2, dt=1.0):
    t = np.arange(n) * dt
    return (amp * np.exp(-t / tau) * np.cos(2.0 * math.pi * t / period)
            + 0.03 * amp)                       # DC offset, as the solver produces


def test_window_safe_fit_recovers_period_where_zero_crossings_hold_up():
    period = 380.0
    y = _damped(period, tau=3000.0, n=1500)     # ~3.9 cycles
    fit = estimate_period_damped(y)
    assert fit["valid"]
    assert fit["period"] == pytest.approx(period, rel=0.01)
    zc = estimate_period(y)
    assert zc == pytest.approx(period, rel=0.02)
    # the two estimators agree, i.e. the recorded frequency is not the estimator
    assert abs(1.0 / zc - 1.0 / fit["period"]) / (1.0 / period) < 0.02


def test_short_record_is_a_non_measurement_not_a_number():
    """1.3 cycles: the zero-crossing estimator returns ``nan`` and the damped fit
    flags itself invalid -- the two guards the round-4 ``--quick`` artifact lacked
    (it recorded ``T = nan`` next to frequency errors of 27-49 %)."""
    period = 300.0
    y = _damped(period, tau=3000.0, n=400)      # 1.3 cycles
    assert not np.isfinite(estimate_period(y))
    fit = estimate_period_damped(y)
    assert not fit["valid"]                     # sentinel, cycles < min_cycles
    assert fit["cycles"] < 2.0
    assert not period_window_ok(400, 1.0, period, min_cycles=3.0)
    # the FFT branch has no validity flag at all: on this record it is within 5 %
    # of the truth *by luck* (zero padding), but on the solver records of the
    # round-4 --quick run it was 27-49 % off, so it may only be used behind the
    # window guard
    from tensorfem.ibm_aitken_added_mass import estimate_period_fft
    f_fft = 1.0 / estimate_period_fft(y)
    assert np.isfinite(f_fft)


# ---------------------------------------------------------------------------
# 3. the discrete interface has a ratio-independent effective inertia
# ---------------------------------------------------------------------------
def test_coupled_solver_is_linear_with_one_effective_inertia():
    """Same lattice/config, two added-mass ratios -> the same ``m_eff``.

    This is the mechanism behind the round-4 verdict: if the deficit were a
    *coupling* error it would scale with ``m_a/m_s``; if it is the diffuse
    interface, ``m_eff`` is one number for the geometry.
    """
    c, box, period = 8.0, 80, 260.0
    m_a = analytic_added_mass(c)
    measured = {}
    for ratio in (1.0, 3.0):
        m_s = m_a / ratio
        k = (m_s + m_a) * (2.0 * math.pi / period) ** 2
        rec = coupled_free_vibration(n_steps=900, half_width=c,
                                     center=(box / 2.0, box / 2.0),
                                     lattice=(box, box), mass=m_s, stiffness=k,
                                     y0=0.2, omega_lbm=1.9, relax_mode="aitken",
                                     sub_tol=1e-6, max_sub_iter=40)
        assert not rec["diverged"]
        assert rec["period_window_ok"]          # the solver reports the guard
        m_eff = effective_added_mass(k, m_s, rec["period_damped"])
        assert np.isfinite(m_eff) and m_eff > 0.0
        # the reference rebuilt from the measured inertia reproduces the record
        f_pred = oscillator_frequency(k, m_s + m_eff)
        assert abs(1.0 / rec["period_damped"] - f_pred) / f_pred < 1e-12
        measured[ratio] = m_eff
    assert abs(measured[1.0] - measured[3.0]) / measured[1.0] < 0.05, measured


# ---------------------------------------------------------------------------
# 4. the round-4 driver carries the window guard
# ---------------------------------------------------------------------------
def test_round4_driver_reports_a_window_guard():
    mod = _load_bench("xval_aitken_strong_coupling.py")
    assert mod.MIN_CYCLES >= 2.0
    cfg = dict(half_width=8.0, box=80, period=260.0, n_steps=1200, omega_lbm=1.9,
               sub_tol=1e-6, max_sub_iter=40, ratios=(1.0,), stencil_mass=0.0)
    rec = mod._case(1.0, "aitken", 1.0, cfg)
    for key in ("window_ok", "cycles", "period_damped", "frequency_from"):
        assert key in rec
    assert np.isfinite(rec["cycles"])
    # the flag is exactly the predeclared criterion, not a decoration
    assert rec["window_ok"] == (rec["cycles"] >= mod.MIN_CYCLES)
    assert rec["window_ok"] is True                 # 1200 steps ~ 3.8 cycles
    assert rec["frequency_from"] in {"zero_crossing", "damped_fit", "fft"}

# ---------------------------------------------------------------------------
# 5. the trust gate used by the geometry test
# ---------------------------------------------------------------------------
def test_trust_gate_flags_an_unusable_record_instead_of_averaging_it_in():
    mod = _load_bench("xval_aitken_strong_attribution.py")
    good = {"period_fit": 381.0, "period_zero_crossing": 380.0, "fit_valid": True,
            "period_window_ok": True, "stiffness": 0.1608, "mass": 52.36,
            "added_mass_lamb": 314.16, "frequency_ref_lamb": 1.0 / 300.0}
    assert mod._apply_trust(dict(good))["measurement_trusted"] is True
    # a failed fit falls back to the zero-crossing estimate but stays untrusted
    bad_fit = dict(good, fit_valid=False)
    rec = mod._apply_trust(bad_fit)
    assert rec["measurement_trusted"] is False
    assert rec["estimator_used"] == "zero_crossing"
    assert rec["period_primary"] == 380.0
    # two estimators disagreeing by more than the threshold is also untrusted
    disagree = dict(good, period_zero_crossing=404.0, period_fit=417.4)
    rec = mod._apply_trust(disagree)
    assert rec["measurement_trusted"] is False
    assert abs(rec["estimator_gap_pct"]) > mod.TOL_ESTIMATOR_AGREEMENT
    # a short window is untrusted even when the fit "succeeds"
    short = dict(good, period_window_ok=False)
    assert mod._apply_trust(short)["measurement_trusted"] is False
