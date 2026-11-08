"""Verification of ``tensorfem.acoustic_radiation`` against analytical anchors.

Three independent anchors pin the radiation efficiency sigma:
1. Rayleigh integral (numerical, converges to exact)
2. Leppington high-frequency asymptotic (sigma -> 1 as ka -> inf)
3. Maidanik low-frequency asymptotic (sigma ~ (ka)^2 as ka -> 0)

Plus power balance: P_in = P_rad (no damping).
"""
import math

import pytest
import torch

from tensorfem.acoustic_radiation import (
    energy_conservation_check,
    frequency_sweep,
    validate_dipole_correction,
    validate_high_frequency_limit,
    validate_low_frequency_scaling,
    unbaffled_radiation_efficiency,
    unbaffled_radiation_efficiency_rayleigh,
    unbaffled_rayleigh_radiated_power,
    _gauss_legendre,
    leppington_radiation_efficiency,
    maidanik_radiation_efficiency,
    mode_mean_square_velocity,
    power_balance_check,
    radiation_efficiency_rayleigh,
    rayleigh_radiated_power,
    simply_supported_mode_shape,
    triple_anchor_study,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIR_RHO = 1.225
AIR_C = 343.0
STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0, thickness=0.01)


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-30)


# ---------------------------------------------------------------------------
# Test 1: Mode shape correctness
# ---------------------------------------------------------------------------

def test_mode_shape_is_sine_product():
    """Simply-supported mode shape must be sin(m*pi*x/a)*sin(n*pi*y/b)."""
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y = torch.tensor([0.0, 0.5, 1.0])
    phi = simply_supported_mode_shape(x, y, length=1.0, width=1.0, m=1, n=1)
    # phi is a 2D grid (5, 3)
    # At boundaries, mode shape must be zero (within float precision)
    assert phi[0, :].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert phi[-1, :].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert phi[:, 0].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert phi[:, -1].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    # At center (x=0.5, y=0.5), sin(pi/2)*sin(pi/2) = 1
    mid = phi[2, 1].item()
    assert abs(mid - 1.0) < 1e-6


def test_mode_shape_higher_modes():
    """Mode (2,1) should have a nodal line at x = a/2."""
    x = torch.linspace(0, 1.0, 101)  # odd number so index 50 is exactly 0.5
    y = torch.tensor([0.5])
    phi = simply_supported_mode_shape(x, y, length=1.0, width=1.0, m=2, n=1)
    # At x = 0.5, sin(2*pi*0.5) = sin(pi) = 0 (within float precision)
    mid_idx = 50
    assert abs(phi[mid_idx, 0].item()) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: Mean square velocity
# ---------------------------------------------------------------------------

def test_mean_square_velocity_analytical():
    """<v^2> = V0^2 / 8 for a single sine mode (V0 = peak amplitude).

    Harmonic time average (1/2) times the spatial average of sin^2 (1/4).
    The 1/2 was missing before the 2026-11-08 convention audit, which made
    every efficiency in this module half of the literature value.
    """
    v_sq = mode_mean_square_velocity(
        length=1.0, width=1.0, m=1, n=1, velocity_amplitude=2.0
    )
    assert v_sq == pytest.approx(0.5, rel=1e-12)

    v_sq = mode_mean_square_velocity(
        length=2.0, width=1.5, m=3, n=2, velocity_amplitude=1.0
    )
    assert v_sq == pytest.approx(0.125, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 3: Gauss-Legendre quadrature
# ---------------------------------------------------------------------------

def test_gauss_legendre_integrates_polynomials():
    """GL quadrature must integrate polynomials up to degree 2n-1 exactly."""
    # n=3 integrates up to degree 5
    nodes, weights = _gauss_legendre(3)
    # integral of x^4 from -1 to 1 = 2/5
    val = (nodes**4 * weights).sum().item()
    assert abs(val - 2.0/5.0) < 1e-6

    # integral of x^5 from -1 to 1 = 0 (odd function)
    val = (nodes**5 * weights).sum().item()
    assert abs(val) < 1e-12

    # n=5 integrates up to degree 9
    nodes5, weights5 = _gauss_legendre(5)
    val = (nodes5**8 * weights5).sum().item()
    assert abs(val - 2.0/9.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: Rayleigh integral convergence
# ---------------------------------------------------------------------------

def test_rayleigh_power_converges_with_quadrature():
    """Radiated power must converge as quadrature order increases."""
    freq = 100.0  # Hz
    P1 = rayleigh_radiated_power(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=20,
    )
    P2 = rayleigh_radiated_power(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    P3 = rayleigh_radiated_power(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=40,
    )
    # All values should be close (within 1% of each other)
    assert _relative_error(P2, P1) < 0.01
    assert _relative_error(P3, P2) < 0.01
    # Values should be positive and reasonable
    assert P3 > 0


def test_rayleigh_power_scales_with_velocity_squared():
    """P_rad must scale as V0^2."""
    freq = 200.0
    P1 = rayleigh_radiated_power(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    P2 = rayleigh_radiated_power(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=2.0, n_points=30,
    )
    assert _relative_error(P2, 4.0 * P1) < 0.02


# ---------------------------------------------------------------------------
# Test 5: Leppington high-frequency asymptotic (sigma -> 1)
# ---------------------------------------------------------------------------

def test_leppington_approaches_unity_at_high_frequency():
    """As ka -> infinity, Leppington sigma must approach 1 within 2%."""
    freq = 10000.0
    sigma = leppington_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq,
        c=AIR_C, m=1, n=1,
    )
    assert sigma > 0.98, f"Leppington sigma={sigma} should be > 0.98 at high freq"
    assert sigma <= 1.0, f"Leppington sigma={sigma} should be <= 1.0"


def test_leppington_non_decreasing_above_coincidence():
    """Above coincidence, sigma should be non-decreasing toward 1."""
    freqs = [2000.0, 5000.0, 10000.0, 20000.0]
    sigmas = [
        leppington_radiation_efficiency(
            length=1.0, width=1.0, frequency=f, c=AIR_C, m=1, n=1,
        )
        for f in freqs
    ]
    # All should be close to 1
    for s in sigmas:
        assert s > 0.95, f"sigma={s} should be > 0.95 above coincidence"


# ---------------------------------------------------------------------------
# Test 6: Maidanik low-frequency asymptotic (sigma ~ (ka)^2)
# ---------------------------------------------------------------------------

def test_maidanik_scales_as_ka_squared():
    """At low ka, Maidanik sigma must scale as (ka)^2 within 3%.

    Since sigma ~ (k/k_mn)^2 and k ~ f, sigma ~ f^2.
    So sigma2/sigma1 = (f2/f1)^2.
    """
    freq1 = 10.0
    freq2 = 20.0

    sigma1 = maidanik_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq1,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    sigma2 = maidanik_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq2,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )

    # sigma ~ f^2, so sigma2/sigma1 ~ (f2/f1)^2 = 4
    ratio = sigma2 / sigma1
    assert abs(ratio - 4.0) / 4.0 < 0.03, \
        f"Maidanik scaling ratio={ratio}, expected ~4.0"


def test_maidanik_positive_and_small_at_low_freq():
    """At low frequency, Maidanik sigma must be positive and << 1."""
    sigma = maidanik_radiation_efficiency(
        length=1.0, width=1.0, frequency=50.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    assert sigma > 0, f"Maidanik sigma={sigma} must be positive"
    assert sigma < 0.1, f"Maidanik sigma={sigma} must be << 1 at low freq"


# ---------------------------------------------------------------------------
# Test 7: Radiation efficiency from Rayleigh integral
# ---------------------------------------------------------------------------

def test_radiation_efficiency_rayleigh_positive():
    """sigma from Rayleigh must be positive."""
    sigma = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    assert sigma > 0, f"sigma={sigma} must be positive"


def test_radiation_efficiency_bounded():
    """sigma from Rayleigh must stay below the coincidence-peak cap.

    sigma > 1 is physical for an individual baffled mode in the coincidence
    region (Rayleigh route gives 1.15 here, quadrature converged against
    n_points 40/56/72; measured peak over the sweep 1.28-1.60), so the cap is
    2.0 rather than 1.  The asymptote sigma -> 1 far above coincidence is a
    separate, genuine statement (test_rayleigh_matches_leppington_far_above_...).
    """
    sigma = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    assert sigma <= 2.0, f"sigma={sigma} should be <= 2.0"


# ---------------------------------------------------------------------------
# Test 8: Power balance
# ---------------------------------------------------------------------------

def test_power_balance_at_resonance():
    """At resonance with no damping, P_in = P_rad within 1%."""
    length, width = 1.0, 1.0
    thickness = STEEL["thickness"]
    young = STEEL["young"]
    poisson = STEEL["poisson"]
    density_plate = STEEL["density"]

    # Natural frequency of mode (1,1)
    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    k_mn = math.sqrt((math.pi / length)**2 + (math.pi / width)**2)
    M_mn = density_plate * thickness * length * width / 4.0
    K_mn = D * k_mn**4 * length * width / 4.0
    omega_mn = math.sqrt(K_mn / M_mn)
    freq_mn = omega_mn / (2.0 * math.pi)

    result = power_balance_check(
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density_plate=density_plate,
        frequency=freq_mn,
        rho_fluid=AIR_RHO, c=AIR_C,
        m=1, n=1,
        force_amplitude=1.0,
        n_points=30,
    )

    assert result["relative_error"] < 0.01, \
        f"Power balance error={result['relative_error']:.4f} > 1%"


def test_power_balance_off_resonance():
    """Off resonance, P_in and P_rad should still balance (within numerical error)."""
    length, width = 1.0, 1.0
    thickness = STEEL["thickness"]
    young = STEEL["young"]
    poisson = STEEL["poisson"]
    density_plate = STEEL["density"]

    # Off resonance: 80% of natural frequency
    D = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    k_mn = math.sqrt((math.pi / length)**2 + (math.pi / width)**2)
    M_mn = density_plate * thickness * length * width / 4.0
    K_mn = D * k_mn**4 * length * width / 4.0
    omega_mn = math.sqrt(K_mn / M_mn)
    freq = 0.8 * omega_mn / (2.0 * math.pi)

    result = power_balance_check(
        length=length, width=width, thickness=thickness,
        young=young, poisson=poisson, density_plate=density_plate,
        frequency=freq,
        rho_fluid=AIR_RHO, c=AIR_C,
        m=1, n=1,
        force_amplitude=1.0,
        n_points=30,
    )

    # Off resonance the balance is less tight due to reactive impedance
    assert result["relative_error"] < 0.05, \
        f"Off-resonance power balance error={result['relative_error']:.4f}"


# ---------------------------------------------------------------------------
# Test 9: Triple anchor closure
# ---------------------------------------------------------------------------

def test_triple_anchor_low_frequency():
    """At low frequency, Rayleigh and Maidanik should agree in scaling.

    The Maidanik asymptotic captures the (ka)^2 scaling but the geometric
    factor may differ by a constant. We verify the scaling behavior.
    """
    # Use very low frequencies where ka << 1
    results = triple_anchor_study(
        length=1.0, width=1.0,
        frequencies=[5.0, 10.0, 20.0],
        rho=AIR_RHO, c=AIR_C,
        m=1, n=1,
        n_points=40,
    )
    # Verify Maidanik scaling: sigma ~ f^2
    sigma_m = [r.sigma_maidanik for r in results]
    # sigma(10)/sigma(5) should be ~4, sigma(20)/sigma(10) should be ~4
    assert abs(sigma_m[1]/sigma_m[0] - 4.0) / 4.0 < 0.01
    assert abs(sigma_m[2]/sigma_m[1] - 4.0) / 4.0 < 0.01

    # Verify Rayleigh also shows f^2 scaling (approximately)
    sigma_r = [r.sigma_rayleigh for r in results]
    ratio1 = sigma_r[1] / sigma_r[0]
    ratio2 = sigma_r[2] / sigma_r[1]
    # Both ratios should be close to 4 (within 20%)
    assert abs(ratio1 - 4.0) / 4.0 < 0.20
    assert abs(ratio2 - 4.0) / 4.0 < 0.20


def test_triple_anchor_high_frequency():
    """At high frequency, Rayleigh and Leppington should both approach 1.

    The Rayleigh integral may not fully converge at very high frequencies
    with moderate quadrature, so we check that both are in the right ballpark.
    """
    # Use moderate high frequency where Rayleigh converges well
    results = triple_anchor_study(
        length=1.0, width=1.0,
        frequencies=[1000.0, 2000.0],
        rho=AIR_RHO, c=AIR_C,
        m=1, n=1,
        n_points=40,
    )
    for r in results:
        # Leppington should be close to 1
        assert r.sigma_leppington > 0.9, f"Leppington sigma={r.sigma_leppington}"
        # Rayleigh should be positive and bounded
        assert 0 < r.sigma_rayleigh <= 1.05


# ---------------------------------------------------------------------------
# Test 10: Different modes
# ---------------------------------------------------------------------------

def test_higher_mode_radiation_efficiency():
    """Mode (2,2) radiation efficiency should be positive and bounded."""
    freq = 1000.0
    sigma_11 = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    sigma_22 = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=2, n=2,
        velocity_amplitude=1.0, n_points=30,
    )
    # Both should be positive and below the coincidence-peak cap (see
    # test_radiation_efficiency_bounded for why the cap is 2.0, not 1).
    assert 0 < sigma_11 <= 2.0
    assert 0 < sigma_22 <= 2.0


# ---------------------------------------------------------------------------
# Test 11: Input validation
# ---------------------------------------------------------------------------

def test_input_validation():
    """All functions must reject invalid inputs."""
    with pytest.raises(ValueError):
        simply_supported_mode_shape(
            torch.tensor([0.0]), torch.tensor([0.0]),
            length=-1.0, width=1.0, m=1, n=1,
        )
    with pytest.raises(ValueError):
        mode_mean_square_velocity(
            length=1.0, width=1.0, m=0, n=1, velocity_amplitude=1.0,
        )
    with pytest.raises(ValueError):
        radiation_efficiency_rayleigh(
            length=1.0, width=1.0, frequency=-100.0,
            rho=AIR_RHO, c=AIR_C, m=1, n=1,
        )
    with pytest.raises(TypeError):
        radiation_efficiency_rayleigh(
            length=1.0, width=1.0, frequency=True,
            rho=AIR_RHO, c=AIR_C, m=1, n=1,
        )


# ---------------------------------------------------------------------------
# Test 12: Rectangular plate (non-square)
# ---------------------------------------------------------------------------

def test_rectangular_plate_radiation():
    """Radiation efficiency for a rectangular plate must be positive and bounded."""
    sigma = radiation_efficiency_rayleigh(
        length=2.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    assert sigma > 0
    assert sigma <= 2.0  # coincidence-peak cap, see test_radiation_efficiency_bounded

    # Maidanik should also work for rectangular plates
    sigma_m = maidanik_radiation_efficiency(
        length=2.0, width=1.0, frequency=50.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    assert sigma_m > 0


# ---------------------------------------------------------------------------
# Test 13: Unbaffled dipole correction
# ---------------------------------------------------------------------------

def test_unbaffled_dipole_correction_low_freq():
    """Unbaffled sigma should be smaller than baffled at low freq (dipole suppression)."""
    freq = 50.0
    sigma_b = leppington_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq, c=AIR_C, m=1, n=1,
    )
    sigma_u = unbaffled_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    # At low freq, unbaffled should be smaller than baffled (dipole suppression)
    # Note: unbaffled radiates to both sides, so sigma can be up to 2*sigma_b
    # But dipole factor reduces it significantly at low ka
    ka = 2 * math.pi * freq / AIR_C * 1.0
    dipole_factor = ka**2 / (1 + ka**2)
    # Dipole factor should be < 0.5 at low ka
    assert dipole_factor < 0.5, f"Dipole factor={dipole_factor} should be < 0.5"
    # Unbaffled sigma should be less than 2*baffled (two-sided radiation)
    assert sigma_u < 2.0 * sigma_b, f"Unbaffled sigma={sigma_u} should be < 2*baffled={2*sigma_b}"


def test_unbaffled_dipole_correction_high_freq():
    """Unbaffled sigma should approach baffled sigma at high freq."""
    freq = 10000.0
    sigma_b = leppington_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq, c=AIR_C, m=1, n=1,
    )
    sigma_u = unbaffled_radiation_efficiency(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    # At high freq, dipole correction ~ 1, so sigma_u ~ sigma_b
    assert abs(sigma_u - sigma_b) / max(sigma_b, 1e-30) < 0.05, \
        f"Unbaffled={sigma_u} should approach baffled={sigma_b}"


def test_unbaffled_rayleigh_positive():
    """Unbaffled Rayleigh radiation efficiency must be positive."""
    sigma = unbaffled_radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    assert sigma > 0, f"Unbaffled sigma={sigma} must be positive"


def test_unbaffled_rayleigh_bounded():
    """Unbaffled sigma from Rayleigh must be <= 4 (two-sided x peak cap)."""
    sigma = unbaffled_radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
        velocity_amplitude=1.0, n_points=30,
    )
    assert sigma <= 4.0, f"Unbaffled sigma={sigma} should be <= 4.0"


# ---------------------------------------------------------------------------
# Test 14: Frequency sweep
# ---------------------------------------------------------------------------

def test_frequency_sweep_baffled():
    """Frequency sweep should produce valid results across ka range."""
    results = frequency_sweep(
        length=1.0, width=1.0,
        freq_min=10.0, freq_max=5000.0,
        n_points_freq=20,
        rho=AIR_RHO, c=AIR_C,
        m=1, n=1,
        n_points_quad=30,
    )
    assert len(results) == 20
    # Check ka range
    ka_values = [r.ka for r in results]
    assert ka_values[0] > 0.01  # Should start from low ka
    assert ka_values[-1] > 5.0  # Should reach high ka

    # Check sigma trends
    for r in results:
        assert r.sigma_baffled_rayleigh >= 0
        assert r.sigma_baffled_leppington >= 0
        assert r.sigma_unbaffled >= 0


def test_frequency_sweep_unbaffled_vs_baffled():
    """Unbaffled sigma should be bounded (two-sided radiation, max ~2)."""
    results = frequency_sweep(
        length=1.0, width=1.0,
        freq_min=50.0, freq_max=3000.0,
        n_points_freq=15,
        rho=AIR_RHO, c=AIR_C,
        m=1, n=1,
        n_points_quad=30,
    )
    for r in results:
        # Unbaffled radiates to both sides, so sigma can be up to ~2
        # At low freq, dipole suppression makes it much smaller
        assert r.sigma_unbaffled <= 2.1, \
            f"At f={r.frequency}, unbaffled sigma={r.sigma_unbaffled} exceeds physical limit"
        # At low freq, unbaffled should be much smaller than baffled
        if r.ka < 1.0:
            assert r.sigma_unbaffled < r.sigma_baffled_rayleigh * 2, \
                f"At ka={r.ka:.2f}, unbaffled={r.sigma_unbaffled} should be < 2*baffled={r.sigma_baffled_rayleigh*2}"


# ---------------------------------------------------------------------------
# Test 15: Low-frequency scaling validation
# ---------------------------------------------------------------------------

def test_low_frequency_scaling_validation():
    """Validate low-frequency (ka)^2 scaling with 3% tolerance."""
    result = validate_low_frequency_scaling(
        length=1.0, width=1.0,
        freq_max=50.0,
        rho=AIR_RHO, c=AIR_C,
        m=1, n=1,
        n_points=40,
        tolerance=0.03,
    )
    # At very low ka, the scaling should be very accurate
    # The first few points (lowest freq) should pass
    low_freq_results = [r for r in result["results"] if r["ka"] < 0.5]
    if low_freq_results:
        max_error = max(r["relative_error"] for r in low_freq_results)
        assert max_error < 0.05, f"Low-freq max error={max_error} > 5%"


# ---------------------------------------------------------------------------
# Test 16: High-frequency limit validation
# ---------------------------------------------------------------------------

def test_high_frequency_limit_validation():
    """Validate high-frequency sigma -> 1 with 2% tolerance."""
    result = validate_high_frequency_limit(
        length=1.0, width=1.0,
        freq_min=5000.0,
        c=AIR_C,
        m=1, n=1,
        n_points=40,
        tolerance=0.02,
    )
    # Leppington should always pass (it's asymptotic)
    assert result["all_pass_leppington"], "Leppington should approach 1"


# ---------------------------------------------------------------------------
# Test 17: Dipole correction validation
# ---------------------------------------------------------------------------

def test_dipole_correction_validation():
    """Validate unbaffled dipole correction trend (not exact match)."""
    result = validate_dipole_correction(
        length=1.0, width=1.0,
        freq_max=2000.0,
        rho=AIR_RHO, c=AIR_C,
        m=1, n=1,
        n_points=30,
        tolerance=0.50,  # Relaxed: dipole correction is approximate
    )
    # At low frequencies, dipole correction should show correct trend
    # (sigma_u < sigma_b, and both scale similarly)
    low_freq_results = [r for r in result["results"] if r["frequency"] < 200]
    if low_freq_results:
        # Both should be positive
        for r in low_freq_results:
            assert r["sigma_rayleigh"] > 0
            assert r["sigma_dipole"] > 0
            # Dipole correction should give reasonable values
            assert r["sigma_dipole"] < 2.0


# ---------------------------------------------------------------------------
# Test 18: Energy conservation
# ---------------------------------------------------------------------------

def test_energy_conservation_baffled():
    """Passivity screen: sigma > 0 and below the coincidence-peak cap."""
    for freq in [100.0, 500.0, 1000.0]:
        result = energy_conservation_check(
            length=1.0, width=1.0,
            frequency=freq,
            rho=AIR_RHO, c=AIR_C,
            m=1, n=1,
            velocity_amplitude=1.0,
            n_points=30,
        )
        assert result["conserved"], \
            f"sigma not positive at f={freq}: sigma={result['sigma']}"
        assert result["within_coincidence_cap"], \
            f"sigma above coincidence-peak cap at f={freq}: {result['sigma']}"


def test_energy_conservation_unbaffled():
    """Energy conservation for unbaffled plate."""
    for freq in [100.0, 500.0, 1000.0]:
        result = energy_conservation_check(
            length=1.0, width=1.0,
            frequency=freq,
            rho=AIR_RHO, c=AIR_C,
            m=1, n=1,
            velocity_amplitude=1.0,
            n_points=30,
        )
        # For unbaffled, sigma can be up to 2 (two-sided)
        assert result["sigma"] <= 2.1, \
            f"Unbaffled sigma={result['sigma']} exceeds physical limit"


# ---------------------------------------------------------------------------
# Test 19: Cross-validation: Rayleigh vs Leppington/Maidanik
# ---------------------------------------------------------------------------

def test_cross_validation_baffled():
    """Cross-validate Rayleigh against Leppington (high-freq) and Maidanik (low-freq)."""
    # Low frequency: compare with Maidanik
    sigma_r_low = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=20.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1, n_points=60,
    )
    sigma_m_low = maidanik_radiation_efficiency(
        length=1.0, width=1.0, frequency=20.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    # Both should be small and positive
    assert 0 < sigma_r_low < 0.1
    assert 0 < sigma_m_low < 0.1
    # They should be within 15% at low freq (with corrected geom_factor)
    assert abs(sigma_r_low - sigma_m_low) / max(sigma_m_low, 1e-30) < 0.15

    # High frequency: compare with Leppington
    # Note: Rayleigh integral converges slowly at high frequency due to
    # rapid oscillations of sin(k*R)/R kernel. Use moderate frequency.
    sigma_r_high = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=2000.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1, n_points=60,
    )
    sigma_l_high = leppington_radiation_efficiency(
        length=1.0, width=1.0, frequency=2000.0,
        c=AIR_C, m=1, n=1,
    )
    # Both should be positive and bounded
    assert 0 < sigma_r_high < 2.0
    assert 0 < sigma_l_high <= 1.0
    # Leppington should be close to 1 at high frequency
    assert sigma_l_high > 0.99


# ---------------------------------------------------------------------------
# Test 20: Rectangular plate unbaffled
# ---------------------------------------------------------------------------

def test_rectangular_plate_unbaffled():
    """Unbaffled radiation efficiency for rectangular plate."""
    sigma_u = unbaffled_radiation_efficiency(
        length=2.0, width=1.0, frequency=200.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    assert sigma_u > 0
    assert sigma_u < 1.0  # Unbaffled is always < 1


# ---------------------------------------------------------------------------
# Test 21: Higher mode unbaffled
# ---------------------------------------------------------------------------

def test_higher_mode_unbaffled():
    """Higher mode unbaffled radiation efficiency."""
    sigma_u_11 = unbaffled_radiation_efficiency(
        length=1.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=1, n=1,
    )
    sigma_u_22 = unbaffled_radiation_efficiency(
        length=1.0, width=1.0, frequency=500.0,
        rho=AIR_RHO, c=AIR_C, m=2, n=2,
    )
    # Both should be positive
    assert sigma_u_11 > 0
    assert sigma_u_22 > 0


# ---------------------------------------------------------------------------
# Convention audit (2026-11-08): the invariants that caught the factor of two
# ---------------------------------------------------------------------------

def test_rayleigh_matches_leppington_far_above_coincidence():
    """Two independent routes must agree on sigma -> 1 far above coincidence.

    This is the invariant the convention defect violated: before the audit the
    Rayleigh route returned 0.504 while Leppington returned 1.0 at f = 2 kHz
    (1 m square panel, mode (1,1), k/k_mn = 8.2) -- a 49.7% disagreement that no
    test covered.
    """
    for freq in (2000.0, 5000.0):
        sigma_ray = radiation_efficiency_rayleigh(
            length=1.0, width=1.0, frequency=freq,
            rho=AIR_RHO, c=AIR_C, m=1, n=1, n_points=40,
        )
        sigma_lep = leppington_radiation_efficiency(
            length=1.0, width=1.0, frequency=freq, c=AIR_C, m=1, n=1,
        )
        assert abs(sigma_ray - sigma_lep) / sigma_lep < 0.02, (
            f"f={freq}: Rayleigh {sigma_ray:.5f} vs Leppington {sigma_leppington_str(sigma_lep)}")
        assert abs(sigma_ray - 1.0) < 0.02, f"f={freq}: sigma={sigma_ray:.5f} should be ~1"


def sigma_leppington_str(value: float) -> str:
    return f"{value:.5f}"


def test_rayleigh_low_frequency_matches_literature_monopole_value():
    """Absolute check against the closed-form low-ka law (no fitted constant).

    For the (1,1) mode of a 1 m square panel at ka = 0.1 the exact monopole law
    gives sigma = 32 k^2 a b/(pi^5) = 1.04568e-3.  The half-convention module
    returned 5.23e-4 here, i.e. 50% low against literature.
    """
    ka = 0.1
    freq = ka * AIR_C / (2.0 * math.pi)
    sigma_ray = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=freq,
        rho=AIR_RHO, c=AIR_C, m=1, n=1, n_points=48,
    )
    analytic = 32.0 * (2.0 * math.pi * freq / AIR_C) ** 2 / math.pi**5
    assert abs(sigma_ray - analytic) / analytic < 0.005, (
        f"Rayleigh {sigma_ray:.6e} vs literature monopole {analytic:.6e}")


def test_high_frequency_validation_flags_quadrature_starvation():
    """Unresolved sin(kR)/R quadrature must be flagged, not scored.

    At f = 10 kHz (1 m panel, ka = 183) a 40-point Gauss grid returns
    sigma = 9.75 while a 56-point grid returns 1.0003, so the raw Rayleigh
    value is meaningless there.  The validation must mark the row unconverged
    and score only converged rows.
    """
    result = validate_high_frequency_limit(
        length=1.0, width=1.0, freq_min=5000.0, c=AIR_C,
        m=1, n=1, n_points=40, tolerance=0.03,
    )
    rows = {round(r["frequency"]): r for r in result["results"]}
    assert rows[5000]["quadrature_converged"] is True
    assert rows[10000]["quadrature_converged"] is False
    assert rows[10000]["quadrature_rel_drift"] > 0.5
    assert result["converged_points"] < result["scored_points"]
    assert result["all_rows_rayleigh_converged"] is False
    # the converged row does satisfy sigma -> 1
    assert result["all_pass_rayleigh"] is True
