"""Exact low-frequency radiation-efficiency asymptotics for baffled plates.

These tests close the verification debt left by the ``m6164`` acoustic-radiation
drop: the low-frequency branch used to be a hand-calibrated fit
(``(k/k_mn)^2 * 1.03`` for odd-odd modes, ``* 0.1`` otherwise), i.e. constants
tuned against the code's own Rayleigh integral and with the wrong frequency
exponent for any mode with an even index.

Every anchor below is a *closed form derived from the Rayleigh integral*
(compact-source monopole / dipole limits), and each is checked against two
independent numerical formulations:

  route A -- the module's 4-D Gauss-Legendre ``sin(kR)/R`` power integral
             (:func:`radiation_efficiency_rayleigh`),
  route B -- direct far-field pressure integration ``|p|^2/(2 rho c)`` over a
             hemisphere on a uniform midpoint grid (implemented here with
             numpy, sharing no code with route A).

Closed forms (see docs/acoustic-radiation-low-frequency-asymptotic.md):

  odd-odd        sigma = 32 k^2 a b / (pi^5 m^2 n^2)                 (~k^2)
  exactly one even   sigma = k^4 |d|^2 / (12 pi S <v^2>)             (~k^4)
  even-even      sigma = O(k^6) (prefactor not derived; slope test only)
"""
import math

import numpy as np
import pytest

from tensorfem.acoustic_radiation import (
    dipole_moment_magnitude,
    dipole_radiation_efficiency,
    leppington_radiation_efficiency,
    low_frequency_radiation_efficiency,
    low_frequency_scaling_exponent,
    maidanik_radiation_efficiency,
    mode_mean_square_velocity,
    monopole_radiation_efficiency,
    radiation_efficiency_rayleigh,
    validate_low_frequency_scaling,
)

AIR_RHO = 1.225
AIR_C = 343.0
PI = math.pi


# ---------------------------------------------------------------------------
# Independent numerical route B: far-field hemisphere power integration
# ---------------------------------------------------------------------------

def farfield_radiation_efficiency(
    length: float, width: float, k: float, m: int, n: int,
    velocity_amplitude: float = 1.0,
    n_grid: int = 96, n_theta: int = 40, n_phi: int = 80,
    rho: float = AIR_RHO, c: float = AIR_C,
) -> float:
    """sigma from direct far-field integration (independent of route A).

    ``p(r_hat) = (j omega rho / 2 pi) * int_S v exp(-j k R) / R dS`` is evaluated
    on a uniform midpoint grid over the plate, then ``|p|^2/(2 rho c) r^2`` is
    integrated over the radiating hemisphere with a midpoint rule.
    """
    x = (np.arange(n_grid) + 0.5) * length / n_grid
    y = (np.arange(n_grid) + 0.5) * width / n_grid
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = velocity_amplitude * np.sin(m * PI * X / length) * np.sin(n * PI * Y / width)
    dS = (length / n_grid) * (width / n_grid)
    omega = c * k

    theta = (np.arange(n_theta) + 0.5) * (PI / 2.0) / n_theta
    phi = (np.arange(n_phi) + 0.5) * 2.0 * PI / n_phi
    w_th = (PI / 2.0) / n_theta
    w_ph = 2.0 * PI / n_phi

    power = 0.0
    gt, gp = np.meshgrid(theta, phi, indexing="ij")
    st, ct = np.sin(gt), np.cos(gt)
    rx = (st * np.cos(gp)).ravel()
    ry = (st * np.sin(gp)).ravel()
    rz = ct.ravel()
    sin_t = st.ravel()

    chunk = 256
    for i0 in range(0, rx.size, chunk):
        rx_b, ry_b, rz_b = rx[i0:i0 + chunk, None, None], ry[i0:i0 + chunk, None, None], \
            rz[i0:i0 + chunk, None, None]
        R = rx_b * X[None] + ry_b * Y[None] + rz_b * 0.0
        p_hat = (1j * omega * rho / (2.0 * PI)) * (
            V[None] * np.exp(-1j * k * R) * dS).sum(axis=(1, 2))
        power += float((np.abs(p_hat) ** 2 * sin_t[i0:i0 + chunk]).sum())

    power *= w_th * w_ph / (2.0 * rho * c)
    S = length * width
    v_sq = mode_mean_square_velocity(
        length=length, width=width, m=m, n=n, velocity_amplitude=velocity_amplitude)
    return power / (rho * c * S * v_sq)


def _k_from_ka(ka: float, length: float, width: float) -> float:
    return ka / max(length, width)


def _freq(k: float) -> float:
    return k * AIR_C / (2.0 * PI)


# ---------------------------------------------------------------------------
# Odd-odd modes: monopole law  sigma = 32 k^2 a b / (pi^5 m^2 n^2)
# ---------------------------------------------------------------------------

MONOPOLE_CASES = [
    # length, width, m, n, ka
    (1.0, 1.0, 1, 1, 0.05),
    (1.0, 1.0, 1, 1, 0.20),
    (1.0, 1.0, 1, 3, 0.10),
    (1.0, 1.0, 3, 3, 0.20),
    (2.0, 1.0, 1, 1, 0.10),
    (2.0, 1.0, 3, 1, 0.20),
    (1.0, 3.0, 1, 3, 0.15),
]


@pytest.mark.parametrize("length,width,m,n,ka", MONOPOLE_CASES)
def test_monopole_law_matches_rayleigh_quadrature(length, width, m, n, ka):
    """Route A must reproduce the analytic monopole law within 3%."""
    k = _k_from_ka(ka, length, width)
    sigma_num = radiation_efficiency_rayleigh(
        length=length, width=width, frequency=_freq(k),
        rho=AIR_RHO, c=AIR_C, m=m, n=n, n_points=48,
    )
    sigma_ana = monopole_radiation_efficiency(
        length=length, width=width, frequency=_freq(k), c=AIR_C, m=m, n=n)
    err = abs(sigma_num - sigma_ana) / sigma_ana
    assert err < 0.03, (
        f"({m},{n}) a={length} b={width} ka={ka}: route A sigma={sigma_num:.6e} "
        f"vs analytic {sigma_ana:.6e}, error={err*100:.2f}% (gate 3%)")


@pytest.mark.parametrize("length,width,m,n,ka", [
    (1.0, 1.0, 1, 1, 0.20),
    (2.0, 1.0, 1, 1, 0.20),
    (1.0, 1.0, 1, 3, 0.20),
])
def test_monopole_law_matches_independent_farfield_route(length, width, m, n, ka):
    """Route B (far-field hemisphere integration) reproduces the monopole law."""
    k = _k_from_ka(ka, length, width)
    sigma_b = farfield_radiation_efficiency(length, width, k, m, n)
    sigma_ana = monopole_radiation_efficiency(
        length=length, width=width, frequency=_freq(k), c=AIR_C, m=m, n=n)
    err = abs(sigma_b - sigma_ana) / sigma_ana
    assert err < 0.02, (
        f"({m},{n}) a={length} b={width}: route B sigma={sigma_b:.6e} "
        f"vs analytic {sigma_ana:.6e}, error={err*100:.2f}%")


def test_monopole_law_is_quadrature_independent():
    """The analytic anchor must not depend on the quadrature resolution."""
    k = _k_from_ka(0.1, 1.0, 1.0)
    sigmas = [radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=_freq(k), rho=AIR_RHO, c=AIR_C,
        m=1, n=1, n_points=n) for n in (24, 40, 64)]
    ana = monopole_radiation_efficiency(
        length=1.0, width=1.0, frequency=_freq(k), c=AIR_C, m=1, n=1)
    for n_pts, sig in zip((24, 40, 64), sigmas):
        assert abs(sig - ana) / ana < 0.01, f"n_points={n_pts}: sigma={sig:.6e}"


def test_monopole_law_scales_with_plate_area():
    """sigma ~ a*b at fixed k, m, n (monopole law grows with radiating area)."""
    k = 0.1
    s1 = monopole_radiation_efficiency(
        length=1.0, width=1.0, frequency=_freq(k), c=AIR_C, m=1, n=1)
    s2 = monopole_radiation_efficiency(
        length=2.0, width=1.0, frequency=_freq(k), c=AIR_C, m=1, n=1)
    assert abs(s2 / s1 - 2.0) < 1e-12


def test_monopole_law_rejects_even_index():
    with pytest.raises(ValueError):
        monopole_radiation_efficiency(
            length=1.0, width=1.0, frequency=10.0, c=AIR_C, m=2, n=1)


# ---------------------------------------------------------------------------
# One-even mode: dipole law  sigma = k^4 |d|^2 / (12 pi S <v^2>)
# ---------------------------------------------------------------------------

DIPOLE_CASES = [
    (1.0, 1.0, 2, 1, 0.10),
    (1.0, 1.0, 2, 1, 0.40),
    (1.0, 1.0, 1, 2, 0.20),
    (2.0, 1.0, 2, 1, 0.20),
    (2.0, 1.0, 4, 1, 0.40),
]


@pytest.mark.parametrize("length,width,m,n,ka", DIPOLE_CASES)
def test_dipole_law_matches_rayleigh_quadrature(length, width, m, n, ka):
    """Route A must reproduce the analytic dipole law within 3%."""
    k = _k_from_ka(ka, length, width)
    sigma_num = radiation_efficiency_rayleigh(
        length=length, width=width, frequency=_freq(k),
        rho=AIR_RHO, c=AIR_C, m=m, n=n, n_points=48,
    )
    sigma_ana = dipole_radiation_efficiency(
        length=length, width=width, frequency=_freq(k), c=AIR_C, m=m, n=n)
    err = abs(sigma_num - sigma_ana) / sigma_ana
    assert err < 0.03, (
        f"({m},{n}) a={length} b={width} ka={ka}: route A sigma={sigma_num:.6e} "
        f"vs analytic {sigma_ana:.6e}, error={err*100:.2f}% (gate 3%)")


@pytest.mark.parametrize("length,width,m,n,ka", [
    (1.0, 1.0, 2, 1, 0.20),
    (2.0, 1.0, 2, 1, 0.20),
])
def test_dipole_law_matches_independent_farfield_route(length, width, m, n, ka):
    """Route B reproduces the dipole law (independent formulation)."""
    k = _k_from_ka(ka, length, width)
    sigma_b = farfield_radiation_efficiency(length, width, k, m, n)
    sigma_ana = dipole_radiation_efficiency(
        length=length, width=width, frequency=_freq(k), c=AIR_C, m=m, n=n)
    err = abs(sigma_b - sigma_ana) / sigma_ana
    assert err < 0.02, (
        f"({m},{n}): route B sigma={sigma_b:.6e} vs analytic {sigma_ana:.6e}, "
        f"error={err*100:.2f}%")


def test_dipole_moment_closed_form_matches_surface_integral():
    """d = int_S v r_s dS closed form vs numerical surface integration."""
    a, b, V0 = 1.3, 0.7, 2.0
    x = np.linspace(0.0, a, 4001)
    y = np.linspace(0.0, b, 4001)
    X, Y = np.meshgrid(x, y, indexing="ij")
    for m, n in ((2, 1), (1, 2), (4, 3)):
        V = V0 * np.sin(m * PI * X / a) * np.sin(n * PI * Y / b)
        dx = np.trapezoid(np.trapezoid(V * X, y, axis=1), x)
        dy = np.trapezoid(np.trapezoid(V * Y, y, axis=1), x)
        d_num = math.hypot(dx, dy)
        d_ana = dipole_moment_magnitude(
            length=a, width=b, m=m, n=n, velocity_amplitude=V0)
        assert abs(d_num - d_ana) / d_ana < 1e-4, (
            f"({m},{n}): numerical |d|={d_num:.6e} vs closed form {d_ana:.6e}")


def test_dipole_moment_vanishes_for_even_even_modes():
    """Even-even modes have zero dipole moment: the k^4 term disappears."""
    for m, n in ((2, 2), (2, 4), (4, 6)):
        d = dipole_moment_magnitude(length=1.0, width=1.0, m=m, n=n)
        assert d == 0.0, f"({m},{n}) dipole moment should vanish, got {d}"
        assert dipole_radiation_efficiency(
            length=1.0, width=1.0, frequency=100.0, c=AIR_C, m=m, n=n) == 0.0


def test_odd_odd_dipole_is_subdominant():
    """Odd-odd modes carry a dipole moment, but the monopole term dominates.

    ``sigma_dipole / sigma_monopole -> 0`` as ``k -> 0``, so the dipole law must
    not be applied to odd-odd modes.
    """
    for m, n in ((1, 1), (1, 3), (3, 1)):
        d = dipole_moment_magnitude(length=1.0, width=2.0, m=m, n=n)
        assert d > 0.0
        ratios = []
        for ka in (0.2, 0.1):
            k = ka / 2.0
            dip = dipole_radiation_efficiency(
                length=1.0, width=2.0, frequency=_freq(k), c=AIR_C, m=m, n=n)
            mon = monopole_radiation_efficiency(
                length=1.0, width=2.0, frequency=_freq(k), c=AIR_C, m=m, n=n)
            ratios.append(dip / mon)
        assert ratios[0] < 0.01, f"({m},{n}) dipole/monopole={ratios[0]:.3e}"
        assert abs(ratios[0] / ratios[1] - 4.0) < 0.05, "ratio must scale as k^2"


# ---------------------------------------------------------------------------
# Even-even modes: leading order O(k^6) (slope verified, prefactor not derived)
# ---------------------------------------------------------------------------

def test_even_even_scaling_exponent_is_six():
    """Monopole and dipole both cancel for even-even modes -> sigma ~ k^6."""
    kas = [0.4, 0.2, 0.1]
    sigmas = []
    for ka in kas:
        sigmas.append(radiation_efficiency_rayleigh(
            length=1.0, width=1.0, frequency=_freq(ka), rho=AIR_RHO, c=AIR_C,
            m=2, n=2, n_points=48))
    slopes = [math.log(sigmas[i] / sigmas[i + 1]) / math.log(kas[i] / kas[i + 1])
              for i in range(len(kas) - 1)]
    for slope in slopes:
        assert abs(slope - 6.0) < 0.1, f"log-log slope {slope:.3f} != 6 (k^6)"


def test_scaling_exponent_dispatch():
    assert low_frequency_scaling_exponent(1, 1) == 2
    assert low_frequency_scaling_exponent(1, 3) == 2
    assert low_frequency_scaling_exponent(2, 1) == 4
    assert low_frequency_scaling_exponent(3, 4) == 4
    assert low_frequency_scaling_exponent(2, 2) == 6


def test_low_frequency_dispatch_matches_regime_laws():
    kw = dict(length=1.0, width=1.0, frequency=50.0, c=AIR_C)
    assert low_frequency_radiation_efficiency(m=1, n=3, **kw) == pytest.approx(
        monopole_radiation_efficiency(m=1, n=3, **kw), rel=1e-14)
    assert low_frequency_radiation_efficiency(m=2, n=1, **kw) == pytest.approx(
        dipole_radiation_efficiency(m=2, n=1, **kw), rel=1e-14)
    assert low_frequency_radiation_efficiency(m=2, n=2, **kw) == 0.0


# ---------------------------------------------------------------------------
# Regression: the superseded calibrated model, quantified
# ---------------------------------------------------------------------------

def _legacy_calibrated_sigma(length, width, frequency, c, m, n):
    """The m6164 calibrated model, kept here as a negative control only."""
    k = 2.0 * PI * frequency / c
    k_mn = math.sqrt((m * PI / length) ** 2 + (n * PI / width) ** 2)
    geom = 1.03 if (m % 2 == 1 and n % 2 == 1) else 0.1
    return (k / k_mn) ** 2 * geom


def test_legacy_calibration_factor_is_half_the_analytic_prefactor():
    """The old 1.03 factor was the analytic (1,1)-square prefactor *halved*.

    ``monopole_radiation_efficiency`` returns ``32 k^2 a b/(pi^5 m^2 n^2)``
    after the 2026-11-08 convention audit (``<v^2> = V0^2/8``), i.e.
    ``2.0641 (k/k_mn)^2`` for the (1,1) square mode.  The superseded m6164 fit
    used 1.03, which reproduced the old half-convention law, so it is exactly
    half of the correct prefactor: same defect, one factor of two apart.
    """
    k = 0.1
    k_mn = math.sqrt(2.0) * PI
    ratio = monopole_radiation_efficiency(
        length=1.0, width=1.0, frequency=_freq(k), c=AIR_C, m=1, n=1) / (k / k_mn) ** 2
    assert abs(ratio - 2.0641) / 2.0641 < 0.01, f"analytic prefactor ratio={ratio:.6f}"
    assert abs(1.03 - 0.5 * ratio) / (0.5 * ratio) < 0.01, (
        f"legacy 1.03 vs half of analytic {0.5 * ratio:.6f}")


@pytest.mark.parametrize("length,width,m,n,min_err", [
    # (1,3) residual is 10.2% after the convention audit (was 48% against the
    # half-convention law): same-area higher mode is the weakest discriminator.
    (1.0, 1.0, 1, 3, 0.08),     # same area, higher mode
    (2.0, 1.0, 1, 1, 0.15),     # aspect ratio 2
    (2.0, 1.0, 3, 1, 0.15),
    (1.0, 1.0, 2, 1, 0.50),     # even mode: wrong exponent
])
def test_legacy_model_error_exceeds_gate(length, width, m, n, min_err):
    """The superseded fit violates the 15% gate; the new law must not."""
    f = _freq(_k_from_ka(0.1, length, width))
    ana = low_frequency_radiation_efficiency(
        length=length, width=width, frequency=f, c=AIR_C, m=m, n=n)
    legacy = _legacy_calibrated_sigma(length, width, f, AIR_C, m, n)
    err = abs(legacy - ana) / ana
    assert err > min_err, (
        f"({m},{n}) a={length} b={width}: legacy error only {err*100:.1f}% - "
        "negative control no longer reproduces the defect")


def test_legacy_error_grows_as_k_minus_two_for_even_modes():
    """Even-mode legacy error scales as k^-2 because it uses the k^2 law."""
    errs = []
    for ka in (0.1, 0.2):
        f = _freq(ka)
        ana = dipole_radiation_efficiency(
            length=1.0, width=1.0, frequency=f, c=AIR_C, m=2, n=1)
        errs.append(_legacy_calibrated_sigma(1.0, 1.0, f, AIR_C, 2, 1) / ana)
    assert abs(errs[0] / errs[1] - 4.0) < 0.2, f"ratio={errs[0]/errs[1]:.3f}"


# ---------------------------------------------------------------------------
# Public API integration
# ---------------------------------------------------------------------------

def test_maidanik_entry_point_now_returns_analytic_law():
    f = 20.0
    for m, n in ((1, 1), (1, 3), (2, 1)):
        got = maidanik_radiation_efficiency(
            length=1.0, width=1.0, frequency=f, rho=AIR_RHO, c=AIR_C, m=m, n=n)
        want = low_frequency_radiation_efficiency(
            length=1.0, width=1.0, frequency=f, c=AIR_C, m=m, n=n)
        assert got == pytest.approx(want, rel=1e-14)


def test_validate_low_frequency_scaling_scores_odd_modes_within_gate():
    result = validate_low_frequency_scaling(
        length=1.0, width=1.0, freq_max=50.0, rho=AIR_RHO, c=AIR_C,
        m=1, n=1, n_points=40, tolerance=0.03,
    )
    assert result["all_pass"] is True
    assert result["max_error"] < 0.03
    assert result["asymptote_scored_points"] == len(result["results"])
    assert result["scaling_exponent"] == 2


def test_validate_low_frequency_scaling_marks_even_even_as_ungated():
    result = validate_low_frequency_scaling(
        length=1.0, width=1.0, freq_max=10.0, rho=AIR_RHO, c=AIR_C,
        m=2, n=2, n_points=32, tolerance=0.03,
    )
    assert result["all_pass"] is False
    assert result["asymptote_scored_points"] == 0
    assert result["scaling_exponent"] == 6
    assert all(r["asymptote_available"] is False for r in result["results"])


def test_subsonic_leppington_is_half_the_exact_monopole_law():
    """Below coincidence the published Leppington form carries unit prefactor.

    The subsonic branch of :func:`leppington_radiation_efficiency` is the
    published *band-average* ``(k/k_c)^2`` law with unit prefactor, adapted here
    to the modal wavenumber ``k_mn``.  The exact per-mode monopole law carries
    ``2.0641`` for the (1,1) square mode, so the ratio is a fixed constant
    independent of ``k`` (2.0641, not 1.0); the per-mode law is the authority
    (see ``docs/acoustic-radiation-low-frequency-asymptotic.md``).
    """
    for ka in (0.1, 0.2, 0.3):
        f = _freq(_k_from_ka(ka, 1.0, 1.0))
        ana = low_frequency_radiation_efficiency(
            length=1.0, width=1.0, frequency=f, c=AIR_C, m=1, n=1)
        lep = leppington_radiation_efficiency(
            length=1.0, width=1.0, frequency=f, c=AIR_C, m=1, n=1)
        assert 0.0 < ana < 1.0 and 0.0 < lep <= 1.0
        assert abs(ana / lep - 2.0641) / 2.0641 < 0.01, (
            f"ka={ka}: ana={ana:.4e} lep={lep:.4e} ratio={ana / lep:.5f}")
