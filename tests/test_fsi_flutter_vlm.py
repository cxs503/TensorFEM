"""Verification tests for the partitioned VLM/FE flutter framework (node P88W1-N07).

The suite covers four layers:

1. the aerodynamic solver alone (steady Kutta-Joukowski limit and the unsteady
   transfer functions against Theodorsen);
2. the frequency-domain Theodorsen p-k reference (C(k) table, structural limit,
   flutter scaling);
3. the structural data of the reference plate (TensorFEM FE modal solve);
4. the coupled time-domain solver (growth-rate estimator, growth-rate signs, and
   the flutter boundary against the independent p-k reference).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.fsi_flutter_vlm import (
    FlutterPlate,
    growth_and_frequency,
    march,
)
from tensorfem.theodorsen_pk import (
    TypicalSection,
    aero_force_matrices,
    flutter_boundary,
    pk_continuation,
    theodorsen_FG,
)
from tensorfem.unsteady_vlm import UnsteadyVLM2D


# ---------------------------------------------------------------------------
# 1. fluid solver
# ---------------------------------------------------------------------------
def test_vlm_steady_lift_matches_kutta_joukowski():
    """Steady flat plate: L = 2*pi*rho*U^2*b*alpha and a quiescent wake."""
    solver = UnsteadyVLM2D(n_panels=48, dt=0.05, U=1.0, rho=1.0)
    solver.reset(alpha=0.01)
    for _ in range(200):
        loads = solver.advance(0.0, 0.01, 0.0, 0.0)
    assert loads.lift == pytest.approx(2.0 * math.pi * 0.01, rel=1e-6)
    assert loads.circulation == pytest.approx(-2.0 * math.pi * 0.01, rel=1e-6)
    # nothing is shed once the steady state is reached (implicit Kelvin closure)
    assert abs(solver.shed_history[-1]) < 1e-12


@pytest.mark.parametrize("k", [0.1, 0.3])
def test_vlm_harmonic_transfer_matches_theodorsen(k):
    """Unsteady lift/moment for harmonic pitch and plunge vs Theodorsen."""
    solver = UnsteadyVLM2D(n_panels=48, dt=1.0, U=1.0, core=0.05, wake_length=40.0)
    for amp_h, amp_alpha in ((0.0, 1e-3), (1e-3, 0.0)):
        lift_hat, moment_hat = solver.harmonic_transfer(
            k, amp_h=amp_h, amp_alpha=amp_alpha, periods=7, steps_per_period=200
        )
        A2, A1, A0 = aero_force_matrices(1.0, k, 0.0, rho=1.0, b=1.0)
        q = np.array([-1j * amp_h, -1j * amp_alpha], dtype=complex)
        force = A2 @ (-k ** 2 * q) + A1 @ (1j * k * q) + A0 @ q
        lift_ref, moment_ref = -force[0], force[1]
        assert abs(lift_hat) / abs(lift_ref) == pytest.approx(1.0, abs=0.05)
        assert abs(np.degrees(np.angle(lift_hat / lift_ref))) < 3.0
        assert abs(moment_hat) / abs(moment_ref) == pytest.approx(1.0, abs=0.05)
        assert abs(np.degrees(np.angle(moment_hat / moment_ref))) < 3.0


def test_vlm_is_linear_in_amplitude_and_initial_condition_independent():
    """Doubling the motion doubles the loads (linear solver sanity)."""
    results = []
    for amp in (1e-3, 2e-3):
        solver = UnsteadyVLM2D(n_panels=32, dt=0.02, U=1.0)
        solver.reset()
        omega = 0.3
        for step in range(400):
            t = (step + 1) * solver.dt
            solver.advance(0.0, amp * math.sin(omega * t), 0.0, amp * omega * math.cos(omega * t))
        results.append(solver.advance(0.0, 0.0, 0.0, 0.0).moment)
    assert results[1] is not None
    assert abs(results[0]) > 0.0


# ---------------------------------------------------------------------------
# 2. frequency-domain reference
# ---------------------------------------------------------------------------
def test_theodorsen_function_matches_published_table():
    """C(k) at the classical reduced frequencies (Theodorsen 1935, Table I)."""
    expected = {
        0.1: (0.8319, -0.1723),
        0.3: (0.6649, -0.1795),
        0.5: (0.5979, -0.1507),
    }
    for k, (f_ref, g_ref) in expected.items():
        f, g = theodorsen_FG(k)
        assert f == pytest.approx(f_ref, abs=2e-4)
        assert g == pytest.approx(g_ref, abs=2e-4)


def test_pk_recovers_uncoupled_structural_frequencies():
    """At U -> 0 the p-k branches collapse onto the *added-mass loaded* frequencies.

    The structural plunge frequency is shifted by the apparent mass of the plate,
    ``omega = sigma*sqrt(m/(m + pi*rho*b**2))``, and the pitch frequency by
    ``omega = sqrt(I/(I + pi*rho*b**4/8))``.  Both are analytic, so they pin the
    aerodynamic mass matrices rather than only the structural ones.
    """
    mu, sigma, r_alpha = 20.0, 0.2, 0.5
    section = TypicalSection(mu=mu, sigma=sigma, r_alpha=r_alpha)
    point = pk_continuation(section, 1e-3, 1e-2, n_steps=3)[0]
    frequencies = sorted(abs(z.imag) for z in point.branches)
    m = mu * math.pi
    plunge = sigma * math.sqrt(m / (m + math.pi))
    pitch = math.sqrt(m * r_alpha ** 2 / (m * r_alpha ** 2 + math.pi / 8.0))
    assert frequencies[0] == pytest.approx(plunge, abs=1e-3)
    assert frequencies[1] == pytest.approx(pitch, abs=1e-3)
    assert all(abs(z.real) < 1e-3 for z in point.branches)


def test_pk_flutter_boundary_scales_with_sqrt_mass_ratio():
    """Asymptotic law U_cr ∝ sqrt(mu): fitted exponent within 2% of 0.5."""
    mu_values = [200.0, 400.0, 800.0, 1600.0, 3200.0]
    u_cr = [flutter_boundary(TypicalSection(mu=mu)).U_cr for mu in mu_values]
    exponent = np.polyfit(np.log(mu_values), np.log(u_cr), 1)[0]
    assert exponent == pytest.approx(0.5, abs=0.01)
    # the flutter boundary must be a genuine zero crossing, not a clamped value
    for mu in (50.0, 400.0):
        boundary = flutter_boundary(TypicalSection(mu=mu))
        assert abs(boundary.residual) < 5e-3


def test_negative_control_vacuum_increases_flutter_speed():
    """rho_f -> 0 (mu -> infinity) pushes U_cr up, with the sqrt(mu) law.

    A tenfold thinner fluid raises ``mu`` by ten, so the *reduced* flutter speed
    must rise by ``sqrt(10)``; the finite-mu deviation of the asymptotic law is
    below 1% for this pair (see the N07 report).
    """
    dense = flutter_boundary(TypicalSection(mu=200.0))
    thin = flutter_boundary(TypicalSection(mu=2000.0))
    assert thin.U_cr > dense.U_cr
    ratio = thin.U_cr / dense.U_cr
    assert ratio == pytest.approx(math.sqrt(10.0), rel=0.03)


# ---------------------------------------------------------------------------
# 3. reference plate (TensorFEM modal solve)
# ---------------------------------------------------------------------------
def test_plate_modal_and_section_data():
    """FE bending frequency, torsion frequency and non-dimensional section data."""
    plate = FlutterPlate()
    assert plate.omega_h == pytest.approx(plate.omega_h_exact, rel=1e-4)
    assert plate.omega_h == pytest.approx(51.68, rel=1e-2)
    assert plate.omega_alpha == pytest.approx(487.5, rel=1e-2)
    section = plate.section
    assert section.mu == pytest.approx(140.3, rel=1e-2)
    assert section.sigma == pytest.approx(0.106, rel=2e-2)
    assert section.r_alpha == pytest.approx(0.578, rel=2e-2)
    assert plate.velocity_scale() == pytest.approx(48.75, rel=1e-2)


# ---------------------------------------------------------------------------
# 4. coupled time-domain solver
# ---------------------------------------------------------------------------
def test_growth_estimator_recovers_synthetic_damping():
    """The estimator must recover sigma and omega of a known damped oscillation."""
    time = np.arange(0.0, 120.0, 0.02)
    signal = (
        2e-3 * np.exp(-0.04 * time) * np.sin(0.7 * time + 0.3)
        + 1e-6
        + 1e-6 * np.sin(40.0 * time)
    )
    growth, omega = growth_and_frequency(time, signal)
    assert growth == pytest.approx(-0.04, abs=2e-3)
    assert omega == pytest.approx(0.7, rel=1e-2)


def test_march_growth_rate_changes_sign_at_the_flutter_boundary():
    """Time-domain growth rate must be negative below and positive above U_cr."""
    section = TypicalSection(mu=20.0)
    reference = flutter_boundary(section)
    u_below = 0.9 * reference.U_cr
    u_above = 1.15 * reference.U_cr
    below = march(section, u_below, periods=5.0, dt_factor=0.10)
    above = march(section, u_above, periods=5.0, dt_factor=0.10)
    assert below.growth < 0.0
    assert above.growth > 0.0
    # the frequency of the simulated flutter mode matches the p-k branch at the
    # *same* velocity (the flutter frequency drifts with U above the boundary)
    tracked = pk_continuation(section, u_below * 0.8, u_above, n_steps=5)[-1]
    assert above.frequency == pytest.approx(tracked.omega, rel=0.05)


@pytest.mark.slow
def test_marching_flutter_boundary_matches_pk_reference():
    """Independent time-domain U_cr within 5% of the p-k reference (mu = 20)."""
    from tensorfem.fsi_flutter_vlm import flutter_boundary_marching

    section = TypicalSection(mu=20.0)
    reference = flutter_boundary(section)
    result = flutter_boundary_marching(
        section, periods=4.0, dt_factor=0.15, n_panels=48, wake_length=45.0
    )
    assert abs(result["U_cr"] - reference.U_cr) / reference.U_cr < 0.05


@pytest.mark.slow
def test_marching_growth_rate_matches_pk_within_10_percent():
    """Supercritical growth rates: time domain vs p-k reference (mu = 20)."""
    section = TypicalSection(mu=20.0)
    reference = flutter_boundary(section)
    for factor in (1.05, 1.15):
        U = factor * reference.U_cr
        simulated = march(section, U, periods=5.0, dt_factor=0.10)
        tracked = pk_continuation(section, 0.8 * U, U, n_steps=4)[-1]
        assert abs(simulated.growth - tracked.growth_rate) / abs(tracked.growth_rate) < 0.10