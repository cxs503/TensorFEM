import math

import pytest
import torch

from tensorfem.marine_hydrodynamics import AiryWave
from tensorfem.wave_spectrum import (
    GRAVITY,
    Jonswap,
    PiersonMoskowitz,
    composite_gauss_legendre,
    equivalent_design_wave,
    irregular_sea_realisation,
    rayleigh_exceedance,
    rayleigh_most_probable_extreme,
    short_term_statistics,
)


def test_pierson_moskowitz_moments_match_their_closed_form():
    pm = PiersonMoskowitz(6.0, 10.0)
    for order in (0, 1, 2, 3):
        quadrature = pm.moment(order)
        assert quadrature["converged"]
        exact = pm.closed_form_moment(order)
        assert abs(quadrature["moment"] / exact - 1.0) < 1e-11, order
    with pytest.raises(ValueError, match="diverge"):
        pm.closed_form_moment(4)
    with pytest.raises(ValueError, match="diverge"):
        pm.moment(4)


def test_pierson_moskowitz_peak_sits_at_the_peak_frequency():
    pm = PiersonMoskowitz(4.0, 9.0)
    wp = pm.peak_angular_frequency
    grid = torch.linspace(0.2 * wp, 3.0 * wp, 200001, dtype=torch.float64)
    values = pm.density(grid)
    peak = float(grid[int(torch.argmax(values))])
    assert abs(peak / wp - 1.0) < 2e-5
    # density vanishes below zero frequency instead of raising
    assert float(pm.density(torch.tensor([0.0, -1.0], dtype=torch.float64)).sum()) == 0.0


def test_hs_normalisation_is_exact_for_both_spectra():
    for spectrum in (
        PiersonMoskowitz(3.5, 7.0),
        Jonswap(3.5, 7.0, 2.0),
        Jonswap(12.0, 16.5, 3.3),
    ):
        stats = short_term_statistics(spectrum)
        assert abs(stats.hs - spectrum.significant_wave_height) < 1e-9


def test_jonswap_with_unit_peak_enhancement_reduces_to_pierson_moskowitz():
    width = 2.0 * (2.0 * math.pi / 8.0)
    pm = PiersonMoskowitz(5.0, 8.0)
    jonswap = Jonswap(5.0, 8.0, 1.0)
    assert abs(jonswap.alpha / pm.alpha - 1.0) < 1e-11
    grid = torch.linspace(0.1, width, 1000, dtype=torch.float64)
    assert torch.allclose(jonswap.density(grid), pm.density(grid), rtol=1e-11, atol=0.0)
    a, b = short_term_statistics(pm), short_term_statistics(jonswap)
    for name in ("m0", "m1", "m2"):
        assert abs(getattr(a, name) / getattr(b, name) - 1.0) < 1e-10


def test_jonswap_alpha_matches_the_literature_normalisation():
    """alpha = 5.061 (Hs/Tp^2)^2 (1 - 0.287 ln gamma) is the standard form."""
    for hs, tp, gamma in ((6.0, 10.0, 3.3), (4.0, 8.0, 2.0), (9.0, 13.0, 5.0)):
        alpha = Jonswap(hs, tp, gamma).alpha
        literature = 5.061 * (hs / tp**2) ** 2 * (1.0 - 0.287 * math.log(gamma))
        assert abs(alpha / literature - 1.0) < 0.01, (hs, tp, gamma)
    alpha_pm = PiersonMoskowitz(6.0, 10.0).alpha
    assert abs(alpha_pm / (5.061 * (6.0 / 10.0**2) ** 2) - 1.0) < 0.01


def test_periods_are_consistent_and_physically_ordered():
    for spectrum in (PiersonMoskowitz(6.0, 10.0), Jonswap(6.0, 10.0, 3.3)):
        stats = short_term_statistics(spectrum)
        assert stats.zero_crossing_period == pytest.approx(
            2.0 * math.pi * math.sqrt(stats.m0 / stats.m2)
        )
        assert stats.mean_period == pytest.approx(2.0 * math.pi * stats.m0 / stats.m1)
        # narrow-band ordering Tm > Tz and Tz/Tp in the classical 0.7-0.8 band
        assert stats.mean_period > stats.zero_crossing_period > 0.0
        assert 0.70 <= stats.zero_crossing_period / spectrum.peak_period <= 0.79
    # m4 diverges for PM, so the crest period must be reported as undefined
    assert short_term_statistics(PiersonMoskowitz(6.0, 10.0)).crest_period is None


def test_rayleigh_statistics_have_the_classical_values():
    m0 = 2.25  # Hs = 6 m
    # exactly exp(-2) = 13.53 % of waves exceed Hs in a narrow-band sea
    assert rayleigh_exceedance(6.0, m0) == pytest.approx(math.exp(-2.0), rel=1e-14)
    assert rayleigh_exceedance(0.0, m0) == pytest.approx(1.0)
    assert rayleigh_exceedance(100.0, m0) < 1e-100
    # most probable extreme: exact sample-maximum median, above the asymptotic
    for n_cycles in (100.0, 1000.0, 10000.0):
        exact = rayleigh_most_probable_extreme(n_cycles, m0)
        asymptotic = rayleigh_most_probable_extreme(n_cycles, m0, exact=False)
        assert exact > asymptotic
        assert abs(exact / asymptotic - 1.0) < 0.06
        # Hs normalised: H_mp/Hs -> sqrt(ln N / 2)
        assert abs(exact / math.sqrt(8.0 * m0 * math.log(n_cycles)) - 1.0) < 0.06
    assert rayleigh_most_probable_extreme(1000.0, m0) / 6.0 == pytest.approx(
        math.sqrt(math.log(1000.0) / 2.0), rel=0.06
    )
    with pytest.raises(ValueError):
        rayleigh_most_probable_extreme(1.0, m0)


def test_quadrature_converges_on_a_polynomial():
    value = composite_gauss_legendre(lambda t: t**6, -1.0, 3.0, 4)
    exact = (3.0**7 - (-1.0) ** 7) / 7.0
    assert abs(value / exact - 1.0) < 1e-13
    with pytest.raises(ValueError):
        composite_gauss_legendre(lambda t: t, 0.0, 1.0, 0)


@pytest.mark.parametrize("spectrum", [PiersonMoskowitz(6.0, 10.0), Jonswap(6.0, 10.0, 3.3)])
def test_realised_record_reproduces_spectral_statistics(spectrum):
    record = irregular_sea_realisation(spectrum, n_samples=2**13, seed=7)
    stats = short_term_statistics(spectrum)
    # variance identity: the record is built so that sigma = sqrt(m0) exactly
    assert record["measured_std"] == pytest.approx(math.sqrt(record["truncated_m0"]), rel=1e-12)
    assert record["measured_hs"] / stats.hs - 1.0 == pytest.approx(0.0, abs=5e-3)
    # zero-crossing period is *not* imposed and agrees within the narrow-band error
    assert abs(record["measured_zero_crossing_period"] / stats.zero_crossing_period - 1.0) < 0.06
    assert len(record["wave_heights"]) == record["zero_upcrossings"] - 1
    # mean wave height of a Rayleigh sea: sqrt(2 pi m0)
    assert abs(record["mean_wave_height"] / math.sqrt(2.0 * math.pi * stats.m0) - 1.0) < 0.05
    again = irregular_sea_realisation(spectrum, n_samples=2**13, seed=7)
    assert torch.equal(record["elevation"], again["elevation"])


def test_realisation_fails_closed():
    with pytest.raises(ValueError, match="power of two"):
        irregular_sea_realisation(PiersonMoskowitz(6.0, 10.0), n_samples=1000)
    with pytest.raises(ValueError, match="power of two"):
        irregular_sea_realisation(PiersonMoskowitz(6.0, 10.0), n_samples=8)
    with pytest.raises(TypeError, match="spectrum"):
        irregular_sea_realisation(torch.zeros(4), n_samples=16)
    with pytest.raises(ValueError, match="omega_max_factor"):
        irregular_sea_realisation(PiersonMoskowitz(6.0, 10.0), n_samples=16, omega_max_factor=0.0)


def test_spectrum_parameters_fail_closed():
    for kwargs in (
        {"significant_wave_height": 0.0, "peak_period": 10.0},
        {"significant_wave_height": 6.0, "peak_period": -1.0},
        {"significant_wave_height": float("nan"), "peak_period": 10.0},
        {"significant_wave_height": True, "peak_period": 10.0},
    ):
        with pytest.raises((ValueError, TypeError)):
            PiersonMoskowitz(**kwargs)
    with pytest.raises(ValueError, match="peak_enhancement"):
        Jonswap(6.0, 10.0, 0.5)
    with pytest.raises(ValueError):
        PiersonMoskowitz(6.0, 10.0, gravity=0.0)


def test_equivalent_design_wave_drives_the_regular_wave_load_path():
    spectrum = Jonswap(6.0, 10.0, 3.3)
    wave = equivalent_design_wave(spectrum)
    assert isinstance(wave, AiryWave)
    assert wave.height == pytest.approx(6.0, rel=1e-6)
    assert wave.period == pytest.approx(10.0, rel=1e-12)
    k = wave.wave_number
    assert abs(wave.gravity * k * math.tanh(k * wave.depth) / wave.omega**2 - 1.0) < 1e-12
    crossing = equivalent_design_wave(spectrum, period="zero_crossing")
    assert crossing.period == pytest.approx(short_term_statistics(spectrum).zero_crossing_period)
    with pytest.raises(ValueError, match="peak"):
        equivalent_design_wave(spectrum, period="mean")
    with pytest.raises(ValueError):
        equivalent_design_wave(spectrum, height_ratio=0.0)


def test_density_accepts_scalars_and_tensors_consistently():
    spectrum = Jonswap(6.0, 10.0, 3.3)
    for omega in (0.3, 0.6283185307179586, 1.5, 4.0):
        scalar = spectrum.density(omega)
        tensor = float(spectrum.density(torch.tensor(omega, dtype=torch.float64)))
        assert abs(tensor / scalar - 1.0) < 1e-12
    assert spectrum.density(0.0) == 0.0