"""Tests for the harmonic (frequency-sweep) added-mass identification.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  These tests pin down

* the legacy single-frequency ``{y'', y', y, 1}`` fit is rank deficient and
  returns no information about ``m_a`` (defect, kept auditable);
* the harmonic projection recovers a synthetic force exactly;
* a multi-frequency sweep separates ``m_a`` from the added stiffness ``k_a``;
* the sweep refuses to answer when the frequency span is too narrow;
* the free-decay mass attribute is an exact algebraic amplification of the
  frequency error (reproduces the archived v5 rows);
* ``forced_oscillation`` labels its output non-identifiable and still emits the
  legacy value.

No test here needs the cluster: the LBM case is a 48^2 lattice, 3 periods.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ibm_added_mass_harmonic import (
    added_mass_gain,
    amplify_frequency_error,
    apparent_added_mass,
    fit_added_mass_sweep,
    harmonic_projection,
    legacy_basis_condition,
    legacy_single_frequency_fit,
    required_frequency_accuracy,
)
from tensorfem.ibm_aitken_added_mass import analytic_added_mass


def _harmonic_record(omega, amplitude=0.2, n_steps=600, dt=1.0, m_a=314.0,
                     k_a=0.0, c_a=0.0, offset=0.0):
    t = dt * np.arange(1, n_steps + 1, dtype=float)
    y = amplitude * np.sin(omega * t)
    v = amplitude * omega * np.cos(omega * t)
    a = -amplitude * omega ** 2 * np.sin(omega * t)
    f = -m_a * a - c_a * v - k_a * y + offset
    return t, y, f


# ---------------------------------------------------------------------------
# 1. the defect
# ---------------------------------------------------------------------------
def test_legacy_basis_is_rank_deficient():
    cond = legacy_basis_condition(2.0 * math.pi / 300.0)
    assert cond > 1.0e12, cond


def test_legacy_fit_carries_no_added_mass_information():
    """Same synthetic force from two very different masses -> same answer."""
    lo = legacy_single_frequency_fit(314.1592653589793,
                                     omega=2.0 * math.pi / 300.0)
    hi = legacy_single_frequency_fit(560.0, omega=2.0 * math.pi / 300.0)
    assert abs(lo["added_mass_fit"] - hi["added_mass_fit"]) < 1.0e-3
    # both are wrong by ~100 %; the archive recorded exactly this (-0.0)
    assert abs(lo["added_mass_fit"]) < 1.0e-2
    assert abs(hi["added_mass_fit"]) < 1.0e-2


# ---------------------------------------------------------------------------
# 2. the replacement estimator
# ---------------------------------------------------------------------------
def test_harmonic_projection_recovers_synthetic_force():
    omega = 0.03
    t, _y, f = _harmonic_record(omega, m_a=314.0, c_a=5.0, k_a=0.4, offset=0.7)
    proj = harmonic_projection(t, f, omega, skip_cycles=0.0)
    amp = 0.2
    assert proj["in_phase"] == pytest.approx(amp * (314.0 * omega ** 2 - 0.4),
                                             rel=1e-6)
    assert proj["quadrature"] == pytest.approx(-5.0 * amp * omega, rel=1e-6)
    assert proj["offset"] == pytest.approx(0.7, abs=1e-8)
    assert proj["r2"] > 0.999999


def test_apparent_mass_conflates_mass_and_stiffness():
    """m_app = m_a - k_a/w^2: one frequency cannot see them apart."""
    omega = 0.03
    t, y, f = _harmonic_record(omega, m_a=314.0, k_a=0.9)
    pt = apparent_added_mass(f, y, t, omega, amplitude=0.2, skip_cycles=0.0)
    assert pt["apparent_added_mass"] == pytest.approx(
        314.0 - 0.9 / omega ** 2, rel=1e-6)
    assert pt["separated"] is False


def test_frequency_sweep_separates_mass_and_stiffness():
    m_a, k_a, c_a = 330.0, 0.75, 4.0
    points = []
    for period in (80.0, 140.0, 260.0, 420.0):
        omega = 2.0 * math.pi / period
        t, y, f = _harmonic_record(omega, n_steps=1200, m_a=m_a, k_a=k_a,
                                   c_a=c_a)
        pt = apparent_added_mass(f, y, t, omega, skip_cycles=0.0)
        points.append(pt)
    fit = fit_added_mass_sweep(points)
    assert fit["identifiable"] is True
    assert fit["added_mass"] == pytest.approx(m_a, rel=1e-6)
    assert fit["added_stiffness"] == pytest.approx(k_a, rel=1e-6)
    assert fit["r2"] > 0.999999
    for pt in points:
        assert pt["damping"] == pytest.approx(c_a, rel=1e-6)


def test_sweep_refuses_narrow_frequency_span():
    m_a, k_a = 330.0, 0.75
    points = []
    for period in (200.0, 230.0):
        omega = 2.0 * math.pi / period
        t, y, f = _harmonic_record(omega, n_steps=1200, m_a=m_a, k_a=k_a)
        points.append(apparent_added_mass(f, y, t, omega, skip_cycles=0.0))
    fit = fit_added_mass_sweep(points, min_omega_span=1.5)
    assert fit["identifiable"] is False
    assert math.isnan(fit["added_mass"])
    assert "span" in fit["reason"]


def test_sweep_needs_two_points():
    omega = 0.03
    t, y, f = _harmonic_record(omega)
    fit = fit_added_mass_sweep(
        [apparent_added_mass(f, y, t, omega, skip_cycles=0.0)])
    assert fit["identifiable"] is False


# ---------------------------------------------------------------------------
# 3. free-decay amplification identity (v5 section 5.3)
# ---------------------------------------------------------------------------
def test_gain_formula():
    assert added_mass_gain(0.6) == pytest.approx(2.0 * 1.6 / 0.6)
    assert added_mass_gain(2.51) == pytest.approx(2.0 * 3.51 / 2.51)
    # 15 % mass from free decay needs 2.8 % frequency at r = 0.6
    assert required_frequency_accuracy(0.6, 15.0) == pytest.approx(2.81, abs=0.01)


@pytest.mark.parametrize("ratio,df,archived", [
    (0.6, -12.109, 78.544),
    (2.51, 11.9, -28.10),
])
def test_amplification_reproduces_archived_v5_rows(ratio, df, archived):
    pred = amplify_frequency_error(ratio, df)
    assert pred == pytest.approx(archived, abs=0.2)


def test_amplification_matches_direct_reconstruction():
    """Independent check: rebuild m_a from the perturbed frequency."""
    k, m_s, r = 0.09187044942311055, 523.5987755982989, 0.6
    m_a = m_s * r
    f0 = math.sqrt(k / (m_s + m_a)) / (2.0 * math.pi)
    for df in (-12.109, -5.0, 0.0, 4.0, 11.9):
        f = f0 * (1.0 + df / 100.0)
        m_measured = k / (2.0 * math.pi * f) ** 2 - m_s
        direct = 100.0 * (m_measured - m_a) / m_a
        assert amplify_frequency_error(r, df) == pytest.approx(direct, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. the LBM driver keeps its labels honest
# ---------------------------------------------------------------------------
def test_forced_oscillation_reports_apparent_mass_and_flags():
    from tensorfem.ibm_aitken_added_mass import forced_oscillation

    rec = forced_oscillation(n_steps=128, half_width=4.0,
                             center=(24.0, 24.0), lattice=(48, 48),
                             amplitude=0.1, drive_period=32.0,
                             omega_lbm=1.7, sub_tol=1e-8, max_sub_iter=20)
    assert rec["added_mass_identifiable"] is False
    assert "apparent" in rec["added_mass_definition"]
    assert rec["added_mass_fit"] == pytest.approx(
        rec["in_phase_force"] / (rec["amplitude"] *
                                 (2.0 * math.pi / rec["drive_period"]) ** 2))
    assert rec["damping_fit"] == pytest.approx(
        -rec["quadrature_force"] / (rec["amplitude"] *
                                    2.0 * math.pi / rec["drive_period"]))
    assert math.isfinite(rec["legacy_added_mass_fit"])
    assert rec["legacy_basis_condition"] > 1.0e12
    assert rec["added_mass_ref"] == pytest.approx(analytic_added_mass(4.0))
    assert np.all(np.isfinite(rec["force_history"]))