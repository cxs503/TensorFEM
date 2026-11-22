"""Verification of the Cummins radiation-memory module against exact causal pairs.

The reference is the single-pole kernel ``K(t) = c exp(-lam t)`` (and a
sign-changing two-pole kernel with ``B(0) = 0``), whose frequency-domain pair

    B(w) = c lam/(lam^2+w^2),      A(w)-A_inf = -c/(lam^2+w^2)

is closed form.  Every transform below is therefore compared with an analytic
value, never with this module's own output.

Checks (numbered as in docs/marine-controls.md):

V1  B(w) -> K(t) -> B(w) and K(t) -> A(w)                       (transform pair)
V2  Kramers-Kronig gate accepts an exact pair
V3  ... and rejects three independent kinds of inconsistency
V4  Cummins memory force == frequency-domain -A a - B v         (steady harmonic)
V5  radiation dissipation == integral B v^2 dt                  (passivity)
V6  exponential state-space kernel reproduces the direct convolution
V7  free-decay damping ratio == B(wn) / (2 wn (M + A(wn)))
V8  zero damping -> zero kernel, zero memory force

V4-V6 use the *analytic* kernel (not the transform-derived one) so that they
test the time-domain machinery in isolation; the accuracy of the transform is
what V1 measures.
"""
import math

import numpy as np
from scipy.integrate import trapezoid
import pytest

from tensorfem import radiation_memory as rm

C_REF, LAM_REF, A_INF = 2.0, 1.5, 3.0
BAND = np.linspace(0.01, 100.0, 4001)
TIMES = np.arange(0.0, 15.0 + 1e-12, 0.01)
# above this frequency a dt = 0.01 s kernel grid cannot resolve the cosine of the
# inverse transform (w dt > 0.5), so transform residuals are quoted below it
RESOLVED = BAND <= 50.0


def _reference():
    return rm.single_pole_pair(TIMES, BAND, C_REF, LAM_REF, A_INF)


def _exact_kernel(times):
    return C_REF * np.exp(-LAM_REF * times)


def test_kernel_and_inverse_transforms_round_trip():
    ref = _reference()
    kernel = rm.cummins_kernel(BAND, ref["damping"], TIMES, zero_frequency_damping=C_REF / LAM_REF)
    kernel_error = np.max(np.abs(kernel - ref["kernel"])) / C_REF
    assert kernel_error < 3e-2, kernel_error

    damping = rm.damping_from_kernel(TIMES, kernel, BAND)
    damping_error = np.max(
        np.abs(damping - ref["damping"])[RESOLVED]
    ) / ref["damping"].max()
    assert damping_error < 3e-2, damping_error

    added = rm.added_mass_from_kernel(TIMES, kernel, BAND[1:], A_INF)
    dynamic = np.max(np.abs(ref["added_mass"] - A_INF))
    added_error = np.max(np.abs(added - ref["added_mass"][1:])[RESOLVED[1:]]) / dynamic
    assert added_error < 3e-2, added_error

    truncation = rm.kernel_truncation_estimate(BAND, ref["damping"], TIMES)
    assert truncation["bound"] > 0.0


def test_kramers_kronig_gate_accepts_exact_pair():
    ref = _reference()
    gate = rm.kramers_kronig_residual(BAND, ref["damping"], ref["added_mass"], A_INF, zero_frequency_damping=C_REF / LAM_REF)
    assert gate["passed"], gate
    assert gate["residual"] < 2e-2, gate
    assert gate["band"][0] > BAND[0]


def test_kramers_kronig_gate_rejects_inconsistent_pairs():
    ref = _reference()
    scaled = rm.kramers_kronig_residual(
        BAND, 1.05 * ref["damping"], ref["added_mass"], A_INF, zero_frequency_damping=1.05 * C_REF / LAM_REF
    )
    assert scaled["residual"] > 2e-2, scaled
    assert not scaled["passed"]

    other = rm.single_pole_pair(TIMES, BAND, C_REF, 3.0, A_INF)["added_mass"]
    crossed = rm.kramers_kronig_residual(BAND, ref["damping"], other, A_INF, zero_frequency_damping=C_REF / LAM_REF)
    assert crossed["residual"] > 0.5, crossed
    assert not crossed["passed"]

    shifted = rm.kramers_kronig_residual(
        BAND, ref["damping"], ref["added_mass"], A_INF + 0.2, zero_frequency_damping=C_REF / LAM_REF
    )
    assert not shifted["passed"]


def test_cummins_memory_force_matches_frequency_domain():
    omega_0, speed, dt = 2.0, 0.3, 0.01
    times = np.arange(0.0, 40.0 + 1e-12, dt)
    velocity = speed * np.cos(omega_0 * times)
    acceleration = -omega_0 * speed * np.sin(omega_0 * times)
    memory = rm.memory_force(times, _exact_kernel(times), velocity)
    radiation_time = -(A_INF * acceleration + memory)

    a_at = A_INF - C_REF / (LAM_REF ** 2 + omega_0 ** 2)
    b_at = C_REF * LAM_REF / (LAM_REF ** 2 + omega_0 ** 2)
    radiation_frequency = -(a_at * acceleration + b_at * velocity)
    tail = slice(times.size // 2, None)
    error = np.max(np.abs(radiation_time[tail] - radiation_frequency[tail]))
    assert error / np.max(np.abs(radiation_frequency[tail])) < 1e-2, error


def test_radiation_dissipation_energy_balance():
    omega_0, speed = 2.0, 0.3
    periods = 8
    times = np.arange(0.0, periods * 2.0 * math.pi / omega_0 + 1e-12, 0.005)
    velocity = speed * np.cos(omega_0 * times)
    memory = rm.memory_force(times, _exact_kernel(times), velocity)
    dissipated = float(trapezoid(memory * velocity, times))
    b_at = C_REF * LAM_REF / (LAM_REF ** 2 + omega_0 ** 2)
    expected = b_at * float(trapezoid(velocity ** 2, times))
    assert dissipated > 0.0
    assert abs(dissipated - expected) / expected < 2e-2, (dissipated, expected)


def test_state_space_kernel_reproduces_direct_convolution():
    dt = 0.01
    times = np.arange(0.0, 15.0 + 1e-12, dt)
    kernel = _exact_kernel(times)
    fit = rm.fit_exponential_kernel(
        times, kernel, decay_rates=np.geomspace(0.05, 40.0, 12)
    )
    assert fit["relative_error"] < 5e-2, fit["relative_error"]
    model = rm.StateSpaceMemory(fit["coefficients"], -fit["rates"])
    velocity = 0.3 * np.cos(1.7 * times)
    direct = rm.memory_force(times, kernel, velocity)
    recursive = model.force(velocity, dt)
    tail = slice(times.size // 2, None)
    error = np.max(np.abs(recursive[tail] - direct[tail])) / np.max(np.abs(direct[tail]))
    assert error < 5e-2, error


def test_free_decay_damping_ratio():
    c, lam = 1.25, 6.0
    band = np.linspace(1e-3, 60.0, 4001)
    damping = c * lam / (lam * lam + band ** 2)
    added = A_INF - c / (lam * lam + band ** 2)
    mass, omega_n = 1.0, 1.2
    stiffness = omega_n ** 2 * (mass + float(np.interp(omega_n, band, added)))
    zeta_fd = rm.decay_damping_ratio(band, added, damping, mass, omega_n)

    dt, n_lag, n_steps = 0.01, 400, 5000
    lags = np.arange(n_lag) * dt
    result = rm.simulate_cummins(
        mass=mass,
        added_mass_inf=A_INF,
        restoring=stiffness,
        kernel_times=lags,
        kernel=c * np.exp(-lam * lags),
        dt=dt,
        n_steps=n_steps,
        x0=1.0,
    )
    assert result["memory_window"] == pytest.approx((n_lag - 1) * dt)
    x, v, t = result["x"], result["v"], result["t"]
    peaks = np.where((v[:-1] > 0.0) & (v[1:] <= 0.0))[0]
    assert peaks.size >= 8, peaks.size
    keep = np.abs(x[peaks]) > 1e-3
    slope = np.polyfit(t[peaks][keep], np.log(np.abs(x[peaks][keep])), 1)[0]
    zeta_decay = -slope / omega_n
    assert abs(zeta_decay - zeta_fd) / zeta_fd < 1e-1, (zeta_decay, zeta_fd)


def test_zero_damping_gives_zero_kernel_and_no_memory():
    kernel = rm.cummins_kernel(BAND, np.zeros_like(BAND), TIMES)
    assert np.max(np.abs(kernel)) == 0.0
    velocity = 0.3 * np.cos(2.0 * TIMES)
    assert np.max(np.abs(rm.memory_force(TIMES, kernel, velocity))) == 0.0
    # constant added mass: the gate has no dynamic range to normalise against
    with pytest.raises(ValueError):
        rm.kramers_kronig_residual(
            BAND, np.zeros_like(BAND), np.full_like(BAND, A_INF), A_INF
        )


def test_two_pole_pair_gate_accepts_physical_shape():
    a, b, c = 1.0, 6.0, 2.0
    band = np.linspace(0.2, 12.0, 801)
    damping = c * (b * b / (b * b + band ** 2) - a * a / (a * a + band ** 2))
    added = A_INF - c * (b / (b * b + band ** 2) - a / (a * a + band ** 2))
    gate = rm.kramers_kronig_residual(band, damping, added, A_INF, zero_frequency_damping=0.0)
    assert gate["passed"], gate


def test_supplied_origin_matches_full_table_and_unknown_origin_stays_band_limited():
    omega = np.linspace(0.1, 2.0, 17)
    b = 1.0 / (1 + omega**2)
    t = np.array([0.0, 0.5, 2.0])
    expected = rm.cummins_kernel(np.r_[0.0, omega], np.r_[1.0, b], t)
    assert np.array_equal(rm.cummins_kernel(omega, b, t, zero_frequency_damping=1.0), expected)
    truncated = rm.cummins_kernel(omega, b, t)
    assert expected[0] - truncated[0] == pytest.approx(omega[0]*(1 + b[0])/np.pi)
    with pytest.raises(ValueError, match="conflicts"):
        rm.cummins_kernel(np.r_[0.0, omega], np.r_[1.0, b], t, zero_frequency_damping=0.0)
    estimate = rm.kernel_truncation_estimate(omega, b, t, zero_frequency_damping=1.0)
    assert estimate["peak_kernel"] == pytest.approx(expected[0])
    assert estimate["bound"] == pytest.approx(2*b[-1]*omega[-1]/np.pi)


def test_constant_damping_finite_band_principal_value():
    omega = np.linspace(.5, 5., 51)
    at = np.array([.9, 2.7])
    expected = 2. - (np.log((5+at)/(5-at))-np.log((at+.5)/(at-.5)))/(np.pi*at)
    actual = rm.kramers_kronig_added_mass(omega, np.ones_like(omega), 2., at=at)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_conservative_oscillator_preserves_energy():
    dt = .08
    result = rm.simulate_cummins(2., 1., 6., np.array([0.,dt]), np.zeros(2), dt, 500, x0=1.)
    energy = .5*(3.*result["v"]**2 + 6.*result["x"]**2)
    np.testing.assert_allclose(energy, 3., atol=1e-12, rtol=1e-12)


def test_matrix_mass_scalar_added_mass_and_short_memory_window():
    dt = .05
    lags = np.arange(3)*dt
    k = np.array([.6, .5, .4])
    result = rm.simulate_cummins(np.diag([1.,2.]), .3, np.diag([3.,4.]), lags, k, dt, 25,
                                 x0=[.25, -.5], v0=[.1, .2], force=lambda t: [t, -t])
    for j, initial in enumerate([.25, -.5]):
        scalar = rm.simulate_cummins(1.+j, .3, 3.+j, lags, k, dt, 25, x0=initial,
                                     v0=[.1,.2][j], force=lambda t: t*(1-2*j))
        np.testing.assert_allclose(result["x"][:,j], scalar["x"], atol=1e-12)
        for n in range(2,26):
            expected = trapezoid(k*result["v"][n-2:n+1,j][::-1], lags)
            assert result["memory_force"][n,j] == pytest.approx(expected)


def test_recursive_free_motion_uses_one_memory_term_and_starts_at_rest_history():
    dt = .02
    lags = np.arange(7) * dt
    model = rm.StateSpaceMemory([2.], [-1.])
    for _ in range(2):
        sim = rm.simulate_cummins(2., 0., 4., lags, 2*np.exp(-lags), dt, 20,
                                  v0=.5, state_space=model)
        reference, state = [], 0.
        for v in sim["v"][:-1]:
            state = np.exp(-dt)*state + 2*(-np.expm1(-dt))*v
            reference.append(state)
        np.testing.assert_allclose(sim["memory_force"][1:], reference, atol=1e-14)
