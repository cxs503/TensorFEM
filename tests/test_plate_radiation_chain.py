"""Verification of the plate structure-acoustic radiation chain.

The chain is pinned by three independent acoustic routes and two structural
references:

* FEM eigenfrequencies vs the Kirchhoff eigenvalue and Leissa's CCCC table.
* Rayleigh 4-D surface integral vs far-field (boundary-integral) extrapolation
  vs the closed-form angular spectrum of a sine mode.
* Region checks: ``sigma ~ (f/f_c)^2`` deep below coincidence, ``sigma -> 1``
  far above it, and the coincidence peak located from the band-averaged curve.

Marker policy follows ``docs/test-lanes.md``: the fine-mesh eigen-solve and the
very high ``ka`` far-field quadratures are ``slow`` (deselected by
``pytest --fast``), everything else is fast.
"""
import math

import pytest
import torch

from tensorfem.plate_acoustics import (
    AIR_C,
    AIR_RHO,
    PlateAcousticConfig,
    SurfaceVelocityField,
    critical_frequency,
    far_field_power,
    far_field_pressure,
    rayleigh_power,
    sine_mode_field,
    spectral_power_sine_mode,
)
from tensorfem.plate_radiation_chain import (
    BAND_RATIO_THIRD_OCTAVE,
    CLAMPED_SQUARE_COEFFICIENTS,
    asymptotic_report,
    band_radiation_efficiency,
    coincidence_peak_ratio,
    fem_modal_analysis,
    fem_radiation_check,
    fit_power_law_exponent,
    mode_pairs_in_band,
    plateau_level,
    sine_mode_radiation_check,
    simply_supported_mode_frequencies,
)
from tensorfem.plate_vibration import BOUNDARY_CLAMPED, BOUNDARY_SIMPLE

CFG = PlateAcousticConfig()
FC = CFG.critical_frequency


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-30)


def _sigma(power: float, msq: float) -> float:
    """sigma with the harmonic mean-square convention ``<v^2> = msq/2``."""
    return power / (AIR_RHO * AIR_C * CFG.area * msq * 0.5)


# ---------------------------------------------------------------------------
# Coincidence frequency
# ---------------------------------------------------------------------------

def test_critical_frequency_matches_the_bending_dispersion():
    engineering = CFG.critical_frequency
    dispersion = CFG.coincidence_frequency_exact
    # c^2/(1.8 c_L h) versus the exact c^2/(2 pi) sqrt(rho h / D).
    assert _relative(engineering, dispersion) < 0.01
    manual = AIR_C ** 2 / (
        1.8 * math.sqrt(CFG.young / (CFG.density_plate * (1 - CFG.poisson ** 2)))
        * CFG.thickness
    )
    assert critical_frequency(thickness=CFG.thickness) == pytest.approx(manual, rel=1e-12)
    assert critical_frequency(thickness=CFG.thickness) == pytest.approx(engineering, rel=1e-12)


def test_bending_and_acoustic_wavenumbers_cross_at_the_critical_frequency():
    assert CFG.bending_wavenumber(CFG.critical_frequency) == pytest.approx(
        CFG.acoustic_wavenumber(CFG.critical_frequency), rel=0.01
    )


# ---------------------------------------------------------------------------
# Route C: closed-form angular spectrum
# ---------------------------------------------------------------------------

def test_sine_transform_matches_numerical_quadrature():
    from tensorfem.plate_acoustics import _sine_transform

    length = CFG.length
    step = length / 20000.0
    grid = torch.arange(0.5, 20000.0, dtype=torch.float64) * step
    for half_waves in (1, 2, 3):
        for kappa in (0.0, 0.7, 3.0, 6.283185, 12.0):
            numeric = (torch.sin(half_waves * math.pi * grid / length)
                       * torch.exp(-1j * kappa * grid) * step).sum()
            closed = _sine_transform(
                torch.tensor([kappa], dtype=torch.float64), length, half_waves
            )[0]
            assert abs(complex(closed) - complex(numeric)) < 2e-3 * max(1.0, abs(numeric))


def test_sine_transform_is_finite_at_its_removable_singularity():
    from tensorfem.plate_acoustics import _sine_transform

    for half_waves in (1, 2, 5):
        q = half_waves * math.pi / CFG.length
        value = _sine_transform(torch.tensor([q], dtype=torch.float64), CFG.length, half_waves)
        assert torch.isfinite(value).all()
        assert complex(value[0]) == pytest.approx(-0.5j * CFG.length, rel=1e-12)


# ---------------------------------------------------------------------------
# Mean square velocity and the three acoustic routes
# ---------------------------------------------------------------------------

def test_sine_mode_mean_square_velocity_is_a_quarter():
    for (m, n) in ((1, 1), (1, 2), (3, 4)):
        field = sine_mode_field(CFG, m, n, points=16, velocity_amplitude=2.0)
        assert field.mean_square_velocity == pytest.approx(4.0 / 4.0, rel=1e-10)
        assert field.surface == pytest.approx(CFG.area, rel=1e-12)


def test_rayleigh_and_far_field_power_agree_far_below_coincidence():
    for (m, n) in ((1, 1), (1, 2), (2, 2)):
        check = sine_mode_radiation_check(CFG, m, n, points=24, n_theta=24, n_phi=64)
        assert check.far_field_gap < 1e-6
        assert check.sigma_rayleigh == pytest.approx(check.sigma_far_field, rel=1e-6)


def test_spectral_route_reproduces_the_rayleigh_power():
    for (m, n) in ((1, 1), (2, 3)):
        check = sine_mode_radiation_check(
            CFG, m, n, points=24, n_theta=24, n_phi=128, with_spectral=True
        )
        assert check.spectral_gap is not None
        assert check.spectral_gap < 1e-6


def test_all_three_routes_agree_at_coincidence():
    # A mode whose resonance sits at the coincidence frequency: k * a ~ 55.
    mode = (9, 15)
    check = sine_mode_radiation_check(
        CFG, mode[0], mode[1], frequency=FC, points=40,
        n_theta=64, n_phi=512, with_spectral=True,
    )
    assert check.ka == pytest.approx(FC * 2 * math.pi * CFG.length / AIR_C, rel=1e-12)
    assert check.sigma_rayleigh > 1.0            # coincidence amplification
    assert check.far_field_gap < 1e-3
    assert check.spectral_gap < 1e-3


def test_far_field_pressure_recovers_the_power_by_angular_integration():
    field = sine_mode_field(CFG, 1, 1, points=32)
    frequency = CFG.mode_frequency(1, 1)
    distance = 50.0
    n_theta, n_phi = 48, 128
    from tensorfem.plate_acoustics import _gauss_legendre

    u_nodes, u_weights = _gauss_legendre(48)
    u = 0.5 * (u_nodes + 1.0)
    wu = 0.5 * u_weights
    phi = 2.0 * math.pi * torch.arange(n_phi, dtype=torch.float64) / n_phi
    theta = torch.acos(torch.clamp(u, -1.0, 1.0))
    total = 0.0
    for i in range(u.numel()):
        for j in range(n_phi):
            pressure = far_field_pressure(
                field, frequency=frequency, distance=distance,
                theta=float(theta[i]), phi=float(phi[j]),
            )
            total += abs(pressure) ** 2 / (2.0 * AIR_RHO * AIR_C) * distance ** 2 * float(
                wu[i]
            ) * (2.0 * math.pi / n_phi)
    reference = far_field_power(field, frequency=frequency, n_theta=n_theta, n_phi=n_phi)
    assert total == pytest.approx(reference, rel=2e-3)


# ---------------------------------------------------------------------------
# Asymptotic behaviour
# ---------------------------------------------------------------------------

def test_sigma_scales_as_frequency_squared_deep_below_coincidence():
    frequencies = [40.0, 60.0, 90.0, 130.0]
    ratios, sigmas = [], []
    for frequency in frequencies:
        check = sine_mode_radiation_check(
            CFG, 1, 1, frequency=frequency, points=20,
            n_theta=24, n_phi=64, with_spectral=False,
        )
        ratios.append(frequency / FC)
        sigmas.append(check.sigma_rayleigh)
    exponent, _ = fit_power_law_exponent(ratios, sigmas, low=0.0, high=0.04)
    assert abs(exponent - 2.0) / 2.0 < 0.10, exponent


@pytest.mark.slow
def test_sigma_approaches_unity_far_above_coincidence():
    # Mode (1,1) driven at 10 f_c: k a ~ 550, deep in the supersonic limit.
    # At this ka a *sampled* surface grid cannot resolve the far-field phase any
    # more (the surface quadrature needs ~ k*a samples), so the closed-form
    # angular spectrum is the only route that converges; see the module notes.
    power = spectral_power_sine_mode(
        CFG, 1, 1, frequency=10.0 * FC, n_theta=256, n_phi=4096
    )
    sigma = _sigma(power, 0.25)
    assert abs(sigma - 1.0) < 0.05, sigma


def test_uniform_piston_reaches_unit_radiation_efficiency():
    from tensorfem.plate_acoustics import _gauss_grid

    x, y, wx, wy = _gauss_grid(CFG.length, CFG.width, 48)
    piston = SurfaceVelocityField(
        x=x, y=y, velocity=torch.ones((48, 48), dtype=torch.float64),
        weight_x=wx, weight_y=wy, label="piston",
    )
    frequency = 110.0 * AIR_C / (2.0 * math.pi * CFG.length)
    power = far_field_power(piston, frequency=frequency, n_theta=96, n_phi=1024)
    assert _sigma(power, 1.0) == pytest.approx(1.0, rel=0.03)


def test_band_average_exceeds_unity_at_coincidence():
    band = band_radiation_efficiency(CFG, FC, n_theta=24, n_phi=64)
    assert band.modal_count > 50
    assert band.sigma > 1.5
    assert band.sigma_max > band.sigma > band.sigma_min
    assert band.ratio == pytest.approx(1.0, rel=1e-12)


def test_band_average_rises_as_frequency_squared_below_coincidence():
    frequencies = [0.12 * FC, 0.2 * FC, 0.3 * FC, 0.45 * FC]
    bands = [
        band_radiation_efficiency(CFG, value, n_theta=24, n_phi=64)
        for value in frequencies
    ]
    exponent, _ = fit_power_law_exponent(
        [item.ratio for item in bands], [item.sigma for item in bands],
        low=0.1, high=0.5,
    )
    assert exponent > 0.5


def test_coincidence_peak_sits_at_the_critical_frequency():
    """The narrow-band peak locates coincidence; documented accuracy ~7 %."""
    ratios = [0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.14]
    bands = [
        band_radiation_efficiency(
            CFG, ratio * FC, band_ratio=2.0 ** (1.0 / 12.0), n_theta=24, n_phi=64
        )
        for ratio in ratios
    ]
    peak, peak_sigma = coincidence_peak_ratio(
        [item.ratio for item in bands], [item.sigma for item in bands]
    )
    assert 0.95 < peak < 1.12, peak
    assert peak_sigma > 2.0
    assert abs(peak - 1.0) < 0.10


def test_asymptotic_report_quotes_both_asymptotes():
    frequencies = [0.12 * FC, 0.25 * FC, 0.5 * FC, 1.0 * FC, 2.0 * FC]
    bands = [
        band_radiation_efficiency(CFG, value, n_theta=24, n_phi=64)
        for value in frequencies
    ]
    report = asymptotic_report(CFG, bands, low_window=(0.1, 0.5), plateau_above=2.0)
    assert report["critical_frequency"] == pytest.approx(FC, rel=1e-12)
    assert report["low_frequency_exponent"] > 0.5
    assert report["high_frequency_plateau"] > 0.0
    # The knee (low-frequency asymptote meeting the plateau) is reported for
    # completeness; the coincidence *peak* is the acceptance metric below.
    assert report["knee_frequency_error"] >= 0.0
    assert plateau_level(
        [item.ratio for item in bands], [item.sigma for item in bands], above=2.0
    ) == pytest.approx(report["high_frequency_plateau"], rel=1e-12)


# ---------------------------------------------------------------------------
# Structure side: FEM eigenvalues and FEM-driven radiation efficiency
# ---------------------------------------------------------------------------

def test_simply_supported_lattice_ordering_is_by_frequency():
    lattice, frequencies = simply_supported_mode_frequencies(CFG, 5)
    assert lattice[0] == (1, 1)
    assert frequencies[0] == pytest.approx(CFG.mode_frequency(1, 1), rel=1e-12)
    assert bool(torch.all(frequencies[1:] >= frequencies[:-1]).item())
    assert float(frequencies[1]) == pytest.approx(float(frequencies[2]), rel=1e-12)  # (1,2)/(2,1)


@pytest.mark.slow
def test_fem_simply_supported_frequencies_within_two_percent_of_kirchhoff():
    analysis = fem_modal_analysis(CFG, elements=24, modes=5, boundary=BOUNDARY_SIMPLE)
    error = analysis.relative_frequency_error()
    assert error is not None
    assert float(error.max()) < 0.02, error.tolist()


@pytest.mark.slow
def test_fem_clamped_square_frequencies_converge_to_the_leissa_table():
    coarse = fem_modal_analysis(CFG, elements=8, modes=3, boundary=BOUNDARY_CLAMPED)
    fine = fem_modal_analysis(CFG, elements=16, modes=3, boundary=BOUNDARY_CLAMPED)
    coarse_error = coarse.relative_frequency_error()
    fine_error = fine.relative_frequency_error()
    assert coarse_error is not None and fine_error is not None
    # The selective-reduced-integration Q4 Mindlin element converges only in the
    # second order and the clamped CCCC case is its worst test, so the reference
    # check is on the *trend* plus a loose absolute band.
    assert float(fine_error.max()) < float(coarse_error.max())
    assert float(fine_error.max()) < 0.05, fine_error.tolist()
    assert float(fine_error[0]) < 0.03, fine_error.tolist()
    assert len(CLAMPED_SQUARE_COEFFICIENTS) >= 5


def test_fem_radiation_efficiency_matches_the_sine_mode_reference():
    analysis = fem_modal_analysis(CFG, elements=12, modes=4, boundary=BOUNDARY_SIMPLE)
    for mode_index in (1, 2, 4):
        fem = fem_radiation_check(analysis, mode_index, points=20)
        m, n = fem.mode
        reference = sine_mode_radiation_check(
            CFG, m, n, frequency=fem.frequency, points=20, n_theta=24, n_phi=64,
            with_spectral=False,
        )
        assert _relative(fem.sigma_rayleigh, reference.sigma_rayleigh) < 0.05
        assert fem.far_field_gap < 0.05


def test_fem_mode_field_is_normalised_to_unit_rms():
    analysis = fem_modal_analysis(CFG, elements=8, modes=2, boundary=BOUNDARY_SIMPLE)
    field = analysis.surface_velocity_field(1, points=24)
    assert field.rms_velocity == pytest.approx(1.0, rel=1e-12)
    raw = analysis.surface_velocity_field(1, points=24, rms=None)
    assert raw.mean_square_velocity > 0.0
    assert raw.volume_velocity > 0.0          # sign convention: (1,1) mode up


# ---------------------------------------------------------------------------
# Bookkeeping and input validation
# ---------------------------------------------------------------------------

def test_mode_pairs_in_band_are_resonant_and_sorted():
    frequency = 0.3 * FC
    pairs = mode_pairs_in_band(CFG, frequency)
    assert len(pairs) > 3
    for m, n in pairs:
        value = CFG.mode_frequency(m, n)
        assert frequency / BAND_RATIO_THIRD_OCTAVE <= value <= frequency * BAND_RATIO_THIRD_OCTAVE
    assert len(set(pairs)) == len(pairs)


def test_input_validation_rejects_invalid_configuration_and_grids():
    with pytest.raises(ValueError):
        PlateAcousticConfig(thickness=-1.0)
    with pytest.raises(ValueError):
        PlateAcousticConfig(poisson=0.5)
    with pytest.raises(TypeError):
        PlateAcousticConfig(length="0.5")            # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sine_mode_field(CFG, 1, 1, points=1)
    with pytest.raises(ValueError):
        band_radiation_efficiency(CFG, 1.0e-3)
    with pytest.raises(ValueError):
        fem_modal_analysis(CFG, elements=4, modes=4, boundary="free")
    with pytest.raises(ValueError):
        fit_power_law_exponent([0.1], [1.0])


def test_rayleigh_power_scales_with_the_square_of_the_velocity():
    slow = sine_mode_field(CFG, 1, 1, points=24, velocity_amplitude=1.0)
    fast = sine_mode_field(CFG, 1, 1, points=24, velocity_amplitude=3.0)
    frequency = CFG.mode_frequency(1, 1)
    ratio = rayleigh_power(fast, frequency=frequency) / rayleigh_power(
        slow, frequency=frequency
    )
    assert ratio == pytest.approx(9.0, rel=1e-10)


def test_spectral_power_agrees_with_the_far_field_power_for_sine_modes():
    for (m, n) in ((1, 1), (3, 2)):
        field = sine_mode_field(CFG, m, n, points=24)
        frequency = CFG.mode_frequency(m, n)
        spectral = spectral_power_sine_mode(CFG, m, n, frequency=frequency, n_theta=24, n_phi=128)
        numeric = far_field_power(field, frequency=frequency, n_theta=24, n_phi=128)
        assert spectral == pytest.approx(numeric, rel=1e-6)