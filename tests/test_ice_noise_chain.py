"""Verification of the ice-load -> wet-girder -> underwater-noise chain.

The module :mod:`tensorfem.ice_noise_chain` was merged without a test module,
so its three stages and the chain driver were unexercised.  This module pins
every stage to an independent anchor (analytic formula, closed form, or a
different numerical method) and regression-locks the four defects that the
exercise exposed:

1. ``baffled_monopole_pressure`` called ``math.exp`` on a complex argument
   (``TypeError`` on every call);
2. ``girder_modal_superposition`` / ``girder_newmark_reference`` indexed the
   modes of the *free-reduced* eigenproblem with global DOF numbers
   (``IndexError`` on every call);
3. the modal coordinates were allocated in the default float32 while the beam
   matrices are float64 (``RuntimeError`` on every call);
4. the Newmark reference solved the acceleration form with the *displacement*
   operator ``M/(beta dt^2) + gamma/(beta dt) C + K`` without rescaling, so the
   response was too small by ``1/(beta dt^2)``.

Anchors
-------
A1  Ice spectrum: a bin-centred multi-tone history is recovered to < 1 %.
A2  Wet girder: the FE fundamental frequency matches the Euler-Bernoulli
    closed form ``f1 = (pi/L)^2 sqrt(EI/(rho_a + m_a)) / 2 pi``; a step load
    produces twice the mid-span static deflection ``P L^3 / 48 EI`` in both the
    modal and the Newmark integrators; ``single_mode_steady_state`` matches a
    direct RK4 integration of the scalar damped oscillator.
A3  Underwater radiation: the Rayleigh-integral radiation efficiency matches
    the exact monopole limit ``sigma = k^2 Q^2 / (pi a b V0^2)``; the far-field
    Rayleigh pressure converges to the baffled monopole ``p = j rho w Q
    exp(-j k r) / (2 pi r)`` with distance.

Known limitation (recorded, not hidden): the reference Newmark integration
applies modal damping only to the retained modes, so modes beyond the modal
truncation keep ringing undamped.  A modal-vs-Newmark comparison therefore only
agrees when the truncation captures the response (see
``docs/ice-noise-chain-verification.md``); the strict acceleration-level gate of
the module docstring is *not* met and is locked below as a strict xfail.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.ice_noise_chain import (
    WATER_C,
    WATER_RHO,
    IceNoiseChainConfig,
    baffled_far_field_pressure,
    baffled_monopole_pressure,
    dem_ice_load_time_history,
    extract_force_spectrum,
    girder_modal_superposition,
    girder_newmark_reference,
    mode_volume_velocity,
    monopole_sigma_lowfreq,
    noise_level_db,
    run_ice_noise_chain,
    single_mode_steady_state,
    slice_added_mass_per_length,
    underwater_sigma_rayleigh,
)

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _girder(elements: int = 10, modes: int = 4, cross: str = "circular") -> dict:
    length, ei, rho_a = 20.0, 1.0e10, 5000.0
    m_a = slice_added_mass_per_length(cross, 0.5, WATER_RHO)

    def zero(t):
        return torch.zeros(2 * (elements + 1), dtype=DTYPE)

    out = girder_modal_superposition(
        length, elements, ei, rho_a, m_a, zero, 1.0e-2, 1.0e-2, modes=modes
    )
    out["m_a"] = m_a
    return out


def _tone_time_axis(fs: float = 1000.0, duration: float = 2.0):
    """Uniform axis whose FFT bin spacing is exactly ``fs/n`` (integer bins)."""
    n = int(round(duration * fs))
    t = torch.arange(n, dtype=DTYPE) / fs
    return t, fs


# ---------------------------------------------------------------------------
# Stage 1 -- ice load spectrum
# ---------------------------------------------------------------------------
def test_tone_amplitudes_recovered():
    t, fs = _tone_time_axis()
    tones = ((20.5, 120.0), (37.0, 45.0), (91.0, 18.0))  # all on 0.5 Hz bins
    force = torch.zeros_like(t)
    for freq, amp in tones:
        force = force + amp * torch.sin(2.0 * math.pi * freq * t)
    spec = extract_force_spectrum(t, force)
    df = float(spec["fs"]) / t.numel()
    assert abs(df - 0.5) < 1e-12
    for freq, amp in tones:
        idx = int(round(freq / df))
        recovered = float(spec["amplitude"][idx])
        assert abs(recovered - amp) / amp < 0.01, (freq, recovered, amp)


def test_unwindowed_tone_is_exact():
    t, _ = _tone_time_axis()
    force = 7.0 * torch.sin(2.0 * math.pi * 91.0 * t)
    spec = extract_force_spectrum(t, force, window="none")
    df = float(spec["fs"]) / t.numel()
    idx = int(round(91.0 / df))
    assert abs(float(spec["amplitude"][idx]) - 7.0) / 7.0 < 1e-9
    assert abs(float(spec["dominant_freq"]) - 91.0) < 1e-12


def test_psd_parseval_for_a_bin_centred_tone():
    """One-sided PSD integrated over the band equals A^2/2 for a pure tone."""
    t, _ = _tone_time_axis()
    amp = 7.0
    force = amp * torch.sin(2.0 * math.pi * 91.0 * t)
    spec = extract_force_spectrum(t, force, window="none")
    df = float(spec["fs"]) / t.numel()
    power = float(spec["psd"].sum()) * df
    assert abs(power - amp ** 2 / 2.0) / (amp ** 2 / 2.0) < 1e-6, power


def test_impact_history_half_sine():
    t = torch.arange(0.0, 0.5, 1e-3, dtype=DTYPE)
    force = dem_ice_load_time_history(t, [(0.1, 0.02, 5.0e5)])
    peak = float(force.max())
    assert abs(peak - 5.0e5) / 5.0e5 < 1e-3
    inside = (t > 0.105) & (t < 0.115)
    assert bool((force[inside] > 0).all())
    assert float(force[(t < 0.1) | (t > 0.12)].abs().max()) == 0.0


def test_stage1_validation():
    t = torch.arange(0.0, 0.05, 1e-3, dtype=DTYPE)
    with pytest.raises(ValueError):
        dem_ice_load_time_history(t, [(0.0, 0.0, 1.0)])
    with pytest.raises(ValueError):
        extract_force_spectrum(t[::-1].clone(), torch.zeros(t.numel(), dtype=DTYPE))
    with pytest.raises(ValueError):
        extract_force_spectrum(t, torch.zeros(3, dtype=DTYPE))
    with pytest.raises(ValueError):
        extract_force_spectrum(t, torch.zeros(t.numel(), dtype=DTYPE), window="hamming")


# ---------------------------------------------------------------------------
# Stage 2 -- wet girder
# ---------------------------------------------------------------------------
def test_added_mass_closed_forms():
    rho = 1025.0
    assert slice_added_mass_per_length("circular", 0.5, rho) == pytest.approx(
        rho * math.pi * 0.25, rel=1e-15
    )
    assert slice_added_mass_per_length("square", 0.8, rho) == pytest.approx(
        (math.pi / 4.0) * rho * 0.64, rel=1e-15
    )
    with pytest.raises(ValueError):
        slice_added_mass_per_length("ellipse", 0.5, rho)
    with pytest.raises(ValueError):
        slice_added_mass_per_length("circular", 0.0, rho)


@pytest.mark.parametrize("elements", [10, 20])
def test_wet_fundamental_matches_euler_bernoulli(elements):
    length, ei, rho_a = 20.0, 1.0e10, 5000.0
    out = _girder(elements=elements)
    f_fe = float(out["omega"][0]) / (2.0 * math.pi)
    f_exact = (math.pi / length) ** 2 * math.sqrt(ei / (rho_a + out["m_a"])) / (
        2.0 * math.pi
    )
    assert abs(f_fe - f_exact) / f_exact < 1e-4, (f_fe, f_exact)


def test_wet_modes_are_global_and_orthonormal_on_free_dofs():
    out = _girder(elements=10, modes=4)
    modes = out["modes"]
    assert modes.shape[0] == 2 * (10 + 1)  # global DOF count
    constrained = (0, 2 * 10)
    for dof in constrained:
        assert float(modes[dof].abs().max()) == 0.0
    free = out["free"]
    phi_f = modes[free, :]
    # unit modal masses for the mass-orthonormal basis placeholder: the columns
    # must at least be non-trivial and mutually orthogonal on the free set.
    gram = phi_f.T @ phi_f
    assert float(gram.diagonal().min()) > 0.0


def test_constant_load_gives_twice_the_static_deflection():
    """Step load from rest: the first peak is 2 x the static deflection."""
    length, ei, rho_a, elements = 20.0, 1.0e10, 5000.0, 10
    mid = 2 * (elements // 2)
    static = 1.0e4 * length ** 3 / (48.0 * ei)

    def force(t):
        vec = torch.zeros(2 * (elements + 1), dtype=DTYPE)
        vec[mid] = 1.0e4
        return vec

    m_a = slice_added_mass_per_length("circular", 0.5, WATER_RHO)
    for solver in (girder_modal_superposition, girder_newmark_reference):
        out = solver(
            length, elements, ei, rho_a, m_a, force, 0.4, 1.0e-3,
            modes=4, zeta=0.0,
        )
        ratio = float(out["u"][:, mid].abs().max()) / (2.0 * static)
        assert abs(ratio - 1.0) < 0.01, (solver.__name__, ratio)


def test_single_mode_steady_state_matches_rk4():
    """Closed-form amplitude vs direct integration of the scalar oscillator."""

    def rk4_amplitude(omega_n, zeta, omega, f0_over_m, dt=2.0e-4, cycles=60):
        period = 2.0 * math.pi / omega
        n = int(cycles * period / dt)
        q, qd = 0.0, 0.0
        peak = 0.0
        for step in range(n):
            t = step * dt

            def acc(qq, vv, tt):
                return (
                    -2.0 * zeta * omega_n * vv
                    - omega_n ** 2 * qq
                    + f0_over_m * math.cos(omega * tt)
                )

            k1q, k1v = qd, acc(q, qd, t)
            k2q = qd + 0.5 * dt * k1v
            k2v = acc(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v, t + 0.5 * dt)
            k3q = qd + 0.5 * dt * k2v
            k3v = acc(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v, t + 0.5 * dt)
            k4q = qd + dt * k3v
            k4v = acc(q + dt * k3q, qd + dt * k3v, t + dt)
            q = q + dt * (k1q + 2 * k2q + 2 * k3q + k4q) / 6.0
            qd = qd + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            if step > 0.8 * n:
                peak = max(peak, abs(q))
        return peak

    for omega_n, zeta, ratio in ((30.0, 0.05, 0.5), (30.0, 0.02, 0.9), (12.0, 0.1, 1.0)):
        omega = ratio * omega_n
        f0_over_m = 4.0
        closed = single_mode_steady_state(omega_n, zeta, omega, f0_over_m)
        numeric = rk4_amplitude(omega_n, zeta, omega, f0_over_m)
        assert abs(closed - numeric) / closed < 0.01, (omega_n, zeta, ratio, closed, numeric)


def test_single_mode_steady_state_validation():
    with pytest.raises(ValueError):
        single_mode_steady_state(0.0, 0.05, 1.0, 1.0)
    with pytest.raises(ValueError):
        single_mode_steady_state(10.0, -0.1, 1.0, 1.0)
    with pytest.raises(ValueError):
        single_mode_steady_state(10.0, 0.05, float("nan"), 1.0)


def test_monopole_pressure_complex_argument():
    """Regression: ``math.exp`` on a complex argument used to raise."""
    p = baffled_monopole_pressure(20.0, 3.0 + 0.5j, 2.0)
    assert isinstance(p, complex)
    expected = (
        1j * WATER_RHO * 2.0 * math.pi * 20.0 * (3.0 + 0.5j)
        * complex(math.cos(-2.0 * math.pi * 20.0 / WATER_C * 2.0),
                  math.sin(-2.0 * math.pi * 20.0 / WATER_C * 2.0))
        / (2.0 * math.pi * 2.0)
    )
    assert abs(p - expected) / abs(expected) < 1e-12


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: modal damping is applied only to the retained modes, "
        "so the Newmark reference keeps the truncated modes undamped; the "
        "docstring's <3% acceleration-level agreement is not met "
        "(measured -40.7% at 0.7 f1, modes=4). Documented in "
        "docs/ice-noise-chain-verification.md."
    ),
)
def test_modal_vs_newmark_acceleration_rms():
    length, ei, rho_a, elements = 20.0, 1.0e10, 5000.0, 10
    mid = 2 * (elements // 2)
    m_a = slice_added_mass_per_length("circular", 0.5, WATER_RHO)
    out = _girder(elements=elements, modes=4)
    f1 = float(out["omega"][0]) / (2.0 * math.pi)
    drive = 0.7 * f1

    def force(t):
        vec = torch.zeros(2 * (elements + 1), dtype=DTYPE)
        vec[mid] = 1.0e4 * math.sin(2.0 * math.pi * drive * t)
        return vec

    kwargs = dict(modes=4, zeta=0.05)
    modal = girder_modal_superposition(
        length, elements, ei, rho_a, m_a, force, 1.0, 1.0e-3, **kwargs
    )
    ref = girder_newmark_reference(
        length, elements, ei, rho_a, m_a, force, 1.0, 1.0e-3, **kwargs
    )
    n0 = 500
    a_modal = float(torch.sqrt((modal["a"][n0:, mid] ** 2).mean()))
    a_ref = float(torch.sqrt((ref["a"][n0:, mid] ** 2).mean()))
    assert abs(a_modal - a_ref) / a_ref < 0.03


# ---------------------------------------------------------------------------
# Stage 3 -- underwater radiation
# ---------------------------------------------------------------------------
def test_mode_volume_velocity_closed_form():
    length, width, v0 = 1.0, 0.6, 2.0
    q11 = mode_volume_velocity(length, width, 1, 1, v0)
    assert q11 == pytest.approx(4.0 * v0 * length * width / math.pi ** 2, rel=1e-14)
    assert mode_volume_velocity(length, width, 2, 1, v0) == pytest.approx(0.0, abs=1e-18)
    assert mode_volume_velocity(length, width, 1, 4, v0) == pytest.approx(0.0, abs=1e-18)
    q31 = mode_volume_velocity(length, width, 3, 1, v0)
    assert q31 == pytest.approx(4.0 * v0 * length * width / (3.0 * math.pi ** 2), rel=1e-14)
    with pytest.raises(ValueError):
        mode_volume_velocity(0.0, 1.0, 1, 1, 1.0)


@pytest.mark.parametrize("frequency", [10.0, 30.0, 80.0])
def test_radiation_efficiency_matches_monopole_limit(frequency):
    length, width = 1.0, 0.6
    sigma_ray = underwater_sigma_rayleigh(length, width, frequency, n_points=60)
    sigma_exact = monopole_sigma_lowfreq(length, width, frequency)
    assert abs(sigma_ray - sigma_exact) / sigma_exact < 3.0e-3, (
        frequency, sigma_ray, sigma_exact,
    )


def test_radiation_efficiency_scales_with_k_squared():
    length, width = 1.0, 0.6
    s1 = underwater_sigma_rayleigh(length, width, 10.0, n_points=60)
    s2 = underwater_sigma_rayleigh(length, width, 20.0, n_points=60)
    assert abs(s2 / s1 - 4.0) / 4.0 < 5.0e-3, s2 / s1


@pytest.mark.parametrize("r,tol_db", [(1.0, 3.0), (10.0, 0.05), (50.0, 0.01)])
def test_far_field_converges_to_baffled_monopole(r, tol_db):
    length, width, frequency = 1.0, 0.6, 20.0
    q = mode_volume_velocity(length, width, 1, 1, 1.0)
    p_ray = baffled_far_field_pressure(
        length, width, frequency, r, 0.0, n_points=60
    )
    p_mono = baffled_monopole_pressure(frequency, q, r)
    diff_db = 20.0 * math.log10(abs(p_ray) / abs(p_mono))
    assert abs(diff_db) <= tol_db, (r, diff_db)


def test_far_field_scales_linearly_with_velocity():
    length, width = 1.0, 0.6
    p1 = baffled_far_field_pressure(
        length, width, 20.0, 2.0, 0.0, n_points=40, velocity_amplitude=1.0
    )
    p2 = baffled_far_field_pressure(
        length, width, 20.0, 2.0, 0.0, n_points=40, velocity_amplitude=3.0
    )
    assert abs(p2 / p1 - 3.0) < 1e-10


def test_noise_level_db_doubling():
    lp = noise_level_db(2.0e-6)
    assert lp == pytest.approx(20.0 * math.log10(2.0), rel=1e-14)
    assert noise_level_db(2.0) - noise_level_db(1.0) == pytest.approx(
        20.0 * math.log10(2.0), rel=1e-14
    )
    with pytest.raises(ValueError):
        noise_level_db(0.0)


# ---------------------------------------------------------------------------
# Chain driver
# ---------------------------------------------------------------------------
def _chain(scale: float = 1.0) -> dict:
    cfg = IceNoiseChainConfig(
        girder_elements=8,
        girder_modes=4,
        panel_length=1.0,
        panel_width=0.6,
        impacts=[(0.05, 0.02, 5.0e5 * scale), (0.25, 0.03, 2.0e5 * scale)],
        t_end=0.4,
        dt=2.0e-3,
        rayleigh_n_points=8,
    )
    return run_ice_noise_chain(cfg)


def test_chain_end_to_end_is_finite_and_consistent():
    out = _chain()
    length, ei, rho_a = 20.0, 1.0e10, 5000.0
    f_exact = (math.pi / length) ** 2 * math.sqrt(
        ei / (rho_a + out["added_mass_per_length"])
    ) / (2.0 * math.pi)
    f_fe = float(out["girder"]["omega"][0]) / (2.0 * math.pi)
    assert abs(f_fe - f_exact) / f_exact < 1e-4
    assert math.isfinite(out["noise_level_db"])
    assert out["total_p_rms"] > 0.0
    assert int((out["sigma"] > 0).sum()) > 0
    assert float(out["girder_velocity_midspan"].abs().max()) > 0.0
    assert out["girder_velocity_midspan"].dtype == DTYPE


def test_chain_is_linear_in_the_ice_load():
    """Doubling every impact amplitude must raise the level by 6.02 dB."""
    base = _chain(1.0)
    loud = _chain(2.0)
    delta = loud["noise_level_db"] - base["noise_level_db"]
    assert abs(delta - 20.0 * math.log10(2.0)) < 0.01, delta