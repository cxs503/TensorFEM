"""Regression tests for the coupled added-mass frequency validation (v5).

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.

Background
----------
``docs/xval_implicit_ibm_aitken_v4.md`` reported the plate added-mass frequency
ratio as ``1.5811`` against the analytic ``0.7906`` (100 % error) and blamed a
unit-handling bug in the plate driver.  The real defect was in the measurement:
the record was ``n_steps * dt = 300`` long while the analytic period of the same
case is ``T = 600``, and every period estimator degenerates to the window length
when the window is shorter than one period:

* zero crossings: fewer than two up-crossings -> ``nan``;
* Hann-windowed FFT: the peak saturates at the fundamental bin ``1/(n dt)``.

``1/300 = 0.003333`` is exactly the ``f_measured`` in the v4 evidence file, so
the "2x frequency" was the window, not the physics.  These tests pin both the
failure mode (so it cannot come back silently) and the window-safe replacement
:func:`tensorfem.ibm_aitken_added_mass.estimate_period_damped`.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    estimate_period,
    estimate_period_damped,
    estimate_period_fft,
    oscillator_frequency,
    period_window_ok,
    plate_markers,
)


# ---------------------------------------------------------------------------
# synthetic signals
# ---------------------------------------------------------------------------
def _decayed(t, period=600.0, tau=1600.0, amp=0.2, offset=0.0):
    return amp * np.cos(2.0 * math.pi * t / period) * np.exp(-t / tau) + offset


def test_estimator_recovers_period_of_decaying_offset_signal():
    """Decay + DC offset (both present in the coupled runs) must not bias T."""
    dt = 1.0
    t = np.arange(2400) * dt
    y = _decayed(t, period=600.0, tau=1600.0, offset=-0.15)
    fit = estimate_period_damped(y, dt)
    assert fit["valid"] is True
    assert fit["cycles"] > 3.5
    assert abs(fit["period"] - 600.0) / 600.0 < 0.02
    assert abs(fit["tau"] - 1600.0) / 1600.0 < 0.15
    assert abs(fit["offset"] + 0.15) < 0.02


def test_estimator_flags_record_shorter_than_one_period():
    """A window shorter than one period is reported invalid, never silently used."""
    dt = 1.0
    fit = estimate_period_damped(_decayed(np.arange(300) * dt), dt)
    assert fit["valid"] is False
    assert fit["cycles"] < 1.0


def test_legacy_estimators_saturate_at_the_window_length():
    """The v4 failure mode, reproduced deterministically.

    With a half-period record the FFT branch returns the fundamental bin
    ``1 / (n_steps * dt)`` and the zero-crossing branch returns ``nan`` -- so the
    "measured" frequency is a property of the window, not of the signal.
    """
    dt = 1.0
    n = 300
    y = _decayed(np.arange(n) * dt, period=600.0)
    assert not np.isfinite(estimate_period(y, dt))          # < 2 up-crossings
    p_fft = estimate_period_fft(y, dt)
    # zero-padded / Hann-windowed peak lands within a few per cent of the window
    assert abs(p_fft - n * dt) / (n * dt) < 0.10            # == window length

    # the legacy combined estimator therefore reports exactly the v4 number
    legacy_f = 1.0 / estimate_period(y, dt) if np.isfinite(estimate_period(y, dt)) \
        else 1.0 / p_fft
    assert abs(legacy_f - 1.0 / 300.0) / (1.0 / 300.0) < 0.05


def test_period_window_ok_guard():
    assert period_window_ok(2400, 1.0, 600.0) is True
    assert period_window_ok(300, 1.0, 600.0) is False
    assert period_window_ok(1200, 1.0, 600.0, min_cycles=2.0) is True
    assert period_window_ok(900, 1.0, 600.0, min_cycles=2.0) is False
    assert period_window_ok(100, 1.0, float("nan")) is False


# ---------------------------------------------------------------------------
# coupled solver: added-mass frequency vs Lamb (1932) m_a = rho pi c^2
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_coupled_plate_frequency_within_acceptance():
    """4-period record, box 144, c = 10: f/f_vac within 15 % of sqrt(m_s/(m_s+m_a)).

    This is the check v4 declared FAIL on an unusable record.  The residual few
    per cent are the documented kernel-smearing and finite-domain biases, not a
    coupling defect (see ``docs/xval_implicit_ibm_aitken_v5.md``).
    """
    box, c, dt = 144, 10.0, 1.0
    n_steps = 2400
    t_target = 600.0
    m_a = analytic_added_mass(c, 1.0)
    m_s = m_a / 0.6
    k = (m_s + m_a) * (2.0 * math.pi / t_target) ** 2

    body = SpringMountedBody(plate_markers((box / 2.0, box / 2.0), c), mass=m_s,
                             stiffness=k, y0=0.2)
    fsi = IBMImplicitFSI(body, LBMFluid2D((box, box), omega=1.9), dt=dt, dx=1.0,
                         ref_length=2.0 * c, sub_tol=1e-6, max_sub_iter=30)
    ys = [fsi.step()["y"] for _ in range(n_steps)]

    fit = estimate_period_damped(ys, dt)
    assert fit["valid"] is True
    assert period_window_ok(n_steps, dt, fit["period"], 2.0)

    f_meas = 1.0 / fit["period"]
    f_vac = oscillator_frequency(k, m_s)
    f_lamb = oscillator_frequency(k, m_s + m_a)
    err = abs(f_meas - f_lamb) / f_lamb
    assert err < 0.15, f"coupled frequency off by {100 * err:.1f} % vs Lamb"

    # the recovered added mass must be in the right ballpark as well
    m_a_eff = k / (2.0 * math.pi * f_meas) ** 2 - m_s
    assert m_a_eff > 0.5 * m_a
    assert not fsi.diverged