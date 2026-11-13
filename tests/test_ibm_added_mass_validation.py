"""Tests for the added-mass cross-validation of the implicit IBM/Aitken coupling.

Covers (a) the analytic references, (b) the implicit Newmark structure control,
(c) the regression that motivated the module: with the one-step lag in the
interface force the coupled oscillator was unstable (it alternated sign and grew
by a factor ~2 per step until the markers left the lattice in ~20 steps), after
moving the force into the same LBM collision it is stable and bounded.
"""

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_added_mass import (
    SpringMountedBody,
    analytic_added_mass,
    coupled_free_vibration,
    estimate_period,
    estimate_period_fft,
    marker_ring,
    oscillator_frequency,
    plate_markers,
    structural_probe,
)


def test_analytic_added_mass_matches_potential_flow():
    # Lamb 1932: m_a = rho pi c^2 for a plate in broadside motion
    assert abs(analytic_added_mass(10.0, 1.0) - math.pi * 100.0) < 1e-12
    assert abs(analytic_added_mass(2.0, 1.5) - 1.5 * math.pi * 4.0) < 1e-12


def test_oscillator_frequency():
    # 1 Hz oscillator: k = m (2 pi)^2
    m, k = 2.0, 2.0 * (2.0 * math.pi) ** 2
    assert abs(oscillator_frequency(k, m) - 1.0) < 1e-14


def test_plate_markers_spacing_and_ends():
    m = plate_markers((50.0, 30.0), 10.0)
    assert m.shape == (21, 2)
    assert m[0, 0] == pytest.approx(40.0) and m[-1, 0] == pytest.approx(60.0)
    assert np.allclose(m[:, 1], 30.0)
    assert np.allclose(np.diff(m[:, 0]), 1.0)
    ring = marker_ring((0.0, 0.0), 10.0)
    assert len(ring) == int(math.ceil(2.0 * math.pi * 10.0))


def test_period_estimators_on_synthetic_damped_signal():
    dt = 1.0
    period = 120.0
    t = np.arange(0, 5 * period, dt)
    y = np.exp(-0.004 * t) * np.cos(2.0 * math.pi * t / period)
    assert abs(estimate_period(y, dt) - period) / period < 0.02
    assert abs(estimate_period_fft(y, dt) - period) / period < 0.05


def test_structural_integrator_matches_analytic_frequency():
    rec = structural_probe(mass=500.0, stiffness=3.0, y0=0.2, n_steps=1200)
    assert abs(rec["error_pct"]) < 0.5


def test_body_newmark_trial_is_pure():
    body = SpringMountedBody(plate_markers((10.0, 10.0), 4.0), mass=100.0,
                             stiffness=2.0, y0=0.1)
    y, v, a = body.newmark_step(0.0, 1.0)
    assert body.y == pytest.approx(0.1)      # trial evaluation must not commit
    body.commit(y, v, a)
    assert body.y == pytest.approx(y)


def test_coupled_oscillator_is_stable_and_converges():
    """Regression for the one-step lag: stable, bounded, sub-iterations converge."""
    m_a = analytic_added_mass(10.0)
    m_s = m_a / 0.6
    k = (m_s + m_a) * (2.0 * math.pi / 300.0) ** 2
    rec = coupled_free_vibration(n_steps=460, half_width=10.0,
                                 center=(48.0, 48.0), lattice=(96, 96),
                                 mass=m_s, stiffness=k, y0=0.2,
                                 omega_lbm=1.9, sub_tol=1e-6)
    assert rec["steps_completed"] == 460
    assert not rec["diverged"], rec["diverged_reason"]
    assert rec["converged_all"] and rec["maxit_hits"] == 0
    ys = np.abs(np.asarray(rec["y_history"]))
    assert ys.max() < 0.5                    # bounded, no added-mass blow-up
    assert ys[0] > ys[-1]                    # oscillates and decays
    # the fluid adds inertia: the coupled frequency must be below the vacuum one
    f = rec["frequency"] if math.isfinite(rec["frequency"]) else rec["frequency_fft"]
    assert math.isfinite(f) and f < rec["frequency_vacuum"]


def test_added_mass_ratio_debug_fields():
    m_a = analytic_added_mass(10.0)
    m_s = m_a / 0.6
    k = (m_s + m_a) * (2.0 * math.pi / 300.0) ** 2
    rec = coupled_free_vibration(n_steps=120, half_width=10.0,
                                 center=(48.0, 48.0), lattice=(96, 96),
                                 mass=m_s, stiffness=k, y0=0.2,
                                 omega_lbm=1.9, sub_tol=1e-6)
    assert rec["added_mass_ratio"] == pytest.approx(0.6, abs=1e-9)
    expected_vac = oscillator_frequency(k, m_s)
    assert rec["frequency_vacuum"] == pytest.approx(expected_vac, rel=1e-12)
    assert rec["frequency_ratio_ref"] == pytest.approx(
        math.sqrt(m_s / (m_s + m_a)), rel=1e-12)