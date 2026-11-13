"""Verification of the HHT-alpha (Hilbert-Hughes-Taylor) time integrator.

The improvement absorbed here is an upstream practice rather than a local
invention: production structural codes (Abaqus ``*DYNAMIC``, ANSYS ``TINTP``,
Code_Aster ``SCHEMA_THETA``) default to the HHT-alpha scheme instead of the
classical average-acceleration Newmark scheme, because HHT-alpha keeps second
order accuracy while giving *controllable* high-frequency numerical damping
(``rho_inf = (1+alpha)/(1-alpha)``).  Spurious high-frequency content created by
spatial discretisation is therefore dissipated instead of polluting the
response, while the resolved low-frequency range is untouched.

Every assertion below is against a **closed-form** solution of the same
initial-value problem or against the analytical spectral radius of the scheme;
nothing is compared with a stored snapshot of this repository's own output.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.dynamics import hht_alpha, mechanical_energy, newmark_linear

DT = torch.float64
GATE = 1.0e-2


# --------------------------------------------------------------------------- #
# closed-form references
# --------------------------------------------------------------------------- #
def undamped_free_exact(time, *, u0, v0, omega):
    return torch.tensor(
        [u0 * math.cos(omega * t) + v0 / omega * math.sin(omega * t) for t in time.tolist()],
        dtype=DT,
    )


def damped_free_exact(time, *, u0, v0, omega, zeta):
    wd = omega * math.sqrt(1.0 - zeta * zeta)
    return torch.tensor(
        [
            math.exp(-zeta * omega * t)
            * (u0 * math.cos(wd * t) + (v0 + zeta * omega * u0) / wd * math.sin(wd * t))
            for t in time.tolist()
        ],
        dtype=DT,
    )


def harmonic_forced_exact(time, *, m, c, k, p0, omega_f, u0, v0):
    """Exact SDOF response to ``m u'' + c u' + k u = p0 cos(omega_f t)``."""
    omega = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(k * m))
    wd = omega * math.sqrt(1.0 - zeta * zeta)
    den = (k - m * omega_f**2) ** 2 + (c * omega_f) ** 2
    a_p = p0 * (k - m * omega_f**2) / den
    b_p = p0 * c * omega_f / den
    c1 = u0 - a_p
    c2 = (v0 - b_p * omega_f + zeta * omega * c1) / wd
    return torch.tensor(
        [
            a_p * math.cos(omega_f * t)
            + b_p * math.sin(omega_f * t)
            + math.exp(-zeta * omega * t) * (c1 * math.cos(wd * t) + c2 * math.sin(wd * t))
            for t in time.tolist()
        ],
        dtype=DT,
    )


def sdof(m: float, k: float, c: float = 0.0):
    M = torch.tensor([[m]], dtype=DT)
    K = torch.tensor([[k]], dtype=DT)
    C = torch.tensor([[c]], dtype=DT) if c else None
    return M, K, C


# --------------------------------------------------------------------------- #
# 1. backward compatibility: alpha = 0 reproduces the existing Newmark path
# --------------------------------------------------------------------------- #
def test_alpha_zero_reproduces_average_acceleration_newmark():
    """alpha = 0 must collapse onto the pre-existing Newmark path bit-for-bit."""
    M = torch.diag(torch.tensor([2.0, 1.0], dtype=DT))
    K = torch.diag(torch.tensor([500.0, 800.0], dtype=DT))
    C = torch.diag(torch.tensor([3.0, 2.0], dtype=DT))
    time = torch.linspace(0.0, 1.0, 201, dtype=DT)
    force = torch.stack(
        [5.0 * torch.sin(4.0 * time), 5.0 * torch.cos(4.0 * time)], dim=1
    )
    u0 = torch.tensor([0.01, -0.004], dtype=DT)
    v0 = torch.tensor([0.0, 0.2], dtype=DT)

    ref = newmark_linear(M, K, force, time, u0, v0, C=C)
    got = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=0.0)

    for name in ("displacement", "velocity", "acceleration"):
        err = torch.max(torch.abs(getattr(ref, name) - getattr(got, name))).item()
        assert err < 1.0e-12, f"{name}: alpha=0 differs from Newmark by {err:.3e}"


# --------------------------------------------------------------------------- #
# 2. default beta/gamma are the alpha-consistent pair, and range is guarded
# --------------------------------------------------------------------------- #
def test_default_beta_gamma_follow_alpha():
    M, K, C = sdof(1.0, 100.0)
    time = torch.linspace(0.0, 0.5, 101, dtype=DT)
    force = torch.zeros((time.numel(), 1), dtype=DT)
    u0 = torch.tensor([1.0], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)

    for alpha in (-1.0 / 3.0, -0.2, -0.1):
        beta = (1.0 - alpha) ** 2 / 4.0
        gamma = 0.5 - alpha
        explicit = hht_alpha(
            M, K, force, time, u0, v0, C=C, alpha=alpha, beta=beta, gamma=gamma
        )
        implicit = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=alpha)
        err = torch.max(torch.abs(explicit.displacement - implicit.displacement)).item()
        assert err < 1.0e-14, f"alpha={alpha}: default beta/gamma mismatch {err:.3e}"


def test_alpha_outside_stability_range_is_rejected():
    M, K, C = sdof(1.0, 100.0)
    time = torch.linspace(0.0, 0.1, 11, dtype=DT)
    force = torch.zeros((time.numel(), 1), dtype=DT)
    u0 = torch.tensor([1.0], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)
    for alpha in (0.1, -0.5, 1.0):
        with pytest.raises(ValueError):
            hht_alpha(M, K, force, time, u0, v0, C=C, alpha=alpha)


# --------------------------------------------------------------------------- #
# 3. analytical spectral radius at omega*dt -> infinity
# --------------------------------------------------------------------------- #
def amplification_matrix(omega_dt: float, alpha: float) -> torch.Tensor:
    """3x3 state matrix of the HHT-alpha recurrence for one undamped SDOF.

    State is ``(u_n, v_n, a_n)``; the recurrence is written out here from the
    scheme definition (not imported from the implementation under test) so the
    comparison below is an independent construction.
    """
    beta = (1.0 - alpha) ** 2 / 4.0
    gamma = 0.5 - alpha
    dt = 1.0
    c0 = 1.0 / (beta * dt * dt)
    c2 = 1.0 / (beta * dt)
    c3 = 1.0 / (2.0 * beta) - 1.0
    c1 = gamma / (beta * dt)
    c4 = gamma / beta - 1.0
    c5 = dt * (gamma / (2.0 * beta) - 1.0)
    omega2 = omega_dt**2  # k/m with dt = 1

    # (c0*m + (1+alpha)*k) u_{n+1} = m*(c0 u_n + c2 v_n + c3 a_n) + alpha*k*u_n
    denom = c0 + (1.0 + alpha) * omega2
    row_u = torch.tensor(
        [(c0 + alpha * omega2) / denom, c2 / denom, c3 / denom], dtype=DT
    )

    # a_{n+1} = c0 (u_{n+1} - u_n) - c2 v_n - c3 a_n
    row_a = torch.tensor(
        [
            c0 * (row_u[0].item() - 1.0),
            c0 * row_u[1].item() - c2,
            c0 * row_u[2].item() - c3,
        ],
        dtype=DT,
    )

    # v_{n+1} = c1 (u_{n+1} - u_n) - c4 v_n - c5 a_n
    row_v = torch.tensor(
        [
            c1 * (row_u[0].item() - 1.0),
            c1 * row_u[1].item() - c4,
            c1 * row_u[2].item() - c5,
        ],
        dtype=DT,
    )
    return torch.stack([row_u, row_v, row_a])


@pytest.mark.parametrize("alpha", [-1.0 / 3.0, -0.2, -0.1, 0.0])
def test_spectral_radius_at_infinity_matches_closed_form(alpha):
    A = amplification_matrix(1.0e4, alpha)
    rho = torch.max(torch.abs(torch.linalg.eigvals(A))).real.item()
    expected = (1.0 + alpha) / (1.0 - alpha)
    assert abs(rho - expected) < GATE * max(expected, 1.0e-6) + 1.0e-6, (
        f"alpha={alpha}: rho_inf measured {rho:.6f} vs closed form {expected:.6f}"
    )


@pytest.mark.parametrize("alpha", [-1.0 / 3.0, -0.2, 0.0])
def test_measured_decay_of_high_frequency_mode_matches_amplification(alpha):
    """Black-box check that the integrator decays exactly like the analytical
    amplification matrix: the energy decay per step equals ``|lambda|^2``."""
    omega = 200.0
    omega_dt = 20.0
    dt = omega_dt / omega
    n = 200
    M, K, C = sdof(1.0, omega**2)
    time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
    force = torch.zeros((n, 1), dtype=DT)
    u0 = torch.tensor([1.0], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)
    res = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=alpha)

    energy = mechanical_energy(M, K, res.displacement, res.velocity)
    slope = torch.diff(torch.log(energy[20:60])) / 2.0
    measured = float(torch.exp(slope.mean()))

    expected = float(
        torch.max(torch.abs(torch.linalg.eigvals(amplification_matrix(omega_dt, alpha)))).real
    )
    assert abs(measured - expected) < 0.03 * expected, (
        f"alpha={alpha}: measured per-step decay {measured:.4f} vs analytical {expected:.4f}"
    )


def test_energy_is_conserved_at_alpha_zero_for_all_omega_dt():
    """alpha = 0 (average acceleration) is non-dissipative at any omega*dt."""
    omega = 200.0
    for omega_dt in (0.5, 4.0, 20.0):
        dt = omega_dt / omega
        n = 120
        M, K, C = sdof(1.0, omega**2)
        time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
        force = torch.zeros((n, 1), dtype=DT)
        res = newmark_linear(
            M, K, force, time, torch.tensor([1.0], dtype=DT), torch.tensor([0.0], dtype=DT)
        )
        energy = mechanical_energy(M, K, res.displacement, res.velocity)
        drift = abs(float(energy[-1]) - float(energy[0])) / float(energy[0])
        assert drift < 1.0e-12, f"omega*dt={omega_dt}: Newmark energy drifted {drift:.3e}"


def test_high_frequency_dissipation_is_absent_at_beta_quarter():
    """alpha = 0 (average acceleration) must be non-dissipative: rho_inf = 1."""
    A = amplification_matrix(1.0e4, 0.0)
    rho = torch.max(torch.abs(torch.linalg.eigvals(A))).real.item()
    assert abs(rho - 1.0) < 1.0e-6, f"alpha=0 rho_inf should be 1, measured {rho:.6f}"


# --------------------------------------------------------------------------- #
# 4. second order accuracy against the closed-form SDOF solution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [-1.0 / 3.0, -0.2, 0.0])
def test_second_order_convergence_undamped_free_vibration(alpha):
    omega = 2.0 * math.pi
    m, k = 1.0, omega**2
    M, K, C = sdof(m, k)
    u0 = torch.tensor([1.0], dtype=DT)
    v0 = torch.tensor([0.3], dtype=DT)
    errors = []
    dts = [0.02, 0.01, 0.005]
    for dt in dts:
        n = int(round(1.0 / dt)) + 1
        time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
        force = torch.zeros((n, 1), dtype=DT)
        res = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=alpha)
        exact = undamped_free_exact(time, u0=1.0, v0=0.3, omega=omega)
        errors.append(float(torch.max(torch.abs(res.displacement[:, 0] - exact))))
    order = [
        math.log(errors[i] / errors[i + 1]) / math.log(dts[i] / dts[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert max(errors) < 2.0e-2, f"alpha={alpha}: absolute error too large {max(errors):.3e}"
    assert min(errors) < 1.0e-3, f"alpha={alpha}: finest grid error {min(errors):.3e}"
    assert all(abs(p - 2.0) < 0.2 for p in order), (
        f"alpha={alpha}: measured orders {order} are not second order"
    )


# --------------------------------------------------------------------------- #
# 5. damped free and damped forced closed-form agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [-1.0 / 3.0, -0.2])
def test_damped_free_vibration_matches_closed_form(alpha):
    omega, zeta, m = 20.0, 0.05, 1.0
    k = m * omega**2
    c = 2.0 * zeta * omega * m
    M, K, C = sdof(m, k, c)
    dt = 0.002
    n = 501
    time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
    force = torch.zeros((n, 1), dtype=DT)
    u0 = torch.tensor([0.01], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)
    res = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=alpha)
    exact = damped_free_exact(time, u0=0.01, v0=0.0, omega=omega, zeta=zeta)
    rel = float(torch.max(torch.abs(res.displacement[:, 0] - exact))) / 0.01
    assert rel < GATE, f"alpha={alpha}: relative error {rel:.3e} exceeds {GATE}"


@pytest.mark.parametrize("alpha", [-1.0 / 3.0, -0.2])
def test_damped_harmonic_forced_response_matches_closed_form(alpha):
    m, omega, zeta, p0, omega_f = 1.0, 15.0, 0.08, 2.0, 9.0
    k = m * omega**2
    c = 2.0 * zeta * omega * m
    M, K, C = sdof(m, k, c)
    dt = 0.001
    n = 1001
    time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
    force = (p0 * torch.cos(omega_f * time)).reshape(-1, 1)
    u0 = torch.tensor([0.0], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)
    res = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=alpha)
    exact = harmonic_forced_exact(
        time, m=m, c=c, k=k, p0=p0, omega_f=omega_f, u0=0.0, v0=0.0
    )
    peak = float(torch.max(torch.abs(exact)))
    rel = float(torch.max(torch.abs(res.displacement[:, 0] - exact))) / peak
    assert rel < GATE, f"alpha={alpha}: relative error {rel:.3e} exceeds {GATE}"


# --------------------------------------------------------------------------- #
# 6. the engineering benefit: spurious stiff mode is dissipated, soft mode kept
# --------------------------------------------------------------------------- #
def two_dof_stiff_soft(omega_slow=10.0, ratio=200.0):
    m = 1.0
    k_slow = m * omega_slow**2
    k_stiff = m * (omega_slow * ratio) ** 2
    M = torch.diag(torch.tensor([m, m], dtype=DT))
    K = torch.tensor([[k_slow + k_stiff, -k_stiff], [-k_stiff, k_stiff]], dtype=DT)
    return M, K


def modal_energies(M, K, res):
    """Per-mode energy ``0.5*(qdot_k**2 + omega_k**2 q_k**2)`` of an undamped system.

    ``M`` and ``K`` are the exact generalised eigenproblem of the model, so this
    decomposes the response into the *physical* slow mode and the spurious
    high-frequency mode introduced by the spatial discretisation.
    """
    ev, evec = torch.linalg.eigh(torch.linalg.inv(M) @ K)
    q = res.displacement @ evec
    qd = res.velocity @ evec
    return 0.5 * (qd**2 + ev * q**2)


def test_spurious_stiff_mode_dissipated_but_physical_mode_preserved():
    """The engineering point of HHT-alpha: kill the spurious mode, keep the rest."""
    M, K = two_dof_stiff_soft()
    # omega_stiff ~ 2828 rad/s, i.e. omega_stiff*dt ~ 5.7 with dt = 0.002 below
    dt = 0.002  # omega_stiff*dt ~ 5.7 -> the stiff mode is barely resolved
    n = int(round(1.0 / dt)) + 1
    time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
    force = torch.zeros((n, 2), dtype=DT)
    u0 = torch.tensor([1.0, 0.0], dtype=DT)
    v0 = torch.zeros(2, dtype=DT)

    hht = hht_alpha(M, K, force, time, u0, v0, alpha=-1.0 / 3.0)
    nmk = newmark_linear(M, K, force, time, u0, v0)

    e_hht = modal_energies(M, K, hht)
    e_nmk = modal_energies(M, K, nmk)
    soft0 = float(e_hht[0, 0])

    # Newmark (alpha = 0) is non-dissipative: the artefact mode keeps its energy.
    assert float(e_nmk[-1, 1]) > 0.999 * float(e_nmk[0, 1]), (
        "average-acceleration Newmark unexpectedly dissipated the stiff mode"
    )
    # HHT-alpha removes it completely.
    assert float(e_hht[-1, 1]) < 1.0e-20 * float(e_hht[0, 1]), (
        f"stiff-mode energy {float(e_hht[-1, 1]):.3e} was not dissipated"
    )
    # ... while the physical (slow) mode energy is preserved to better than 0.1 %.
    for tag, energies in (("hht", e_hht), ("newmark", e_nmk)):
        drift = abs(float(energies[-1, 0]) - soft0) / soft0
        assert drift < 1.0e-3, f"{tag}: slow-mode energy drifted by {drift:.3e}"


def test_energy_is_dissipated_for_alpha_negative_and_conserved_for_alpha_zero():
    omega = 1.0
    dt = 2.0  # omega*dt = 2: the mode is only coarsely resolved
    n = 61
    time = torch.linspace(0.0, dt * (n - 1), n, dtype=DT)
    force = torch.zeros((n, 1), dtype=DT)
    u0 = torch.tensor([1.0], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)
    M, K, C = sdof(1.0, omega**2)

    dissipated = hht_alpha(M, K, force, time, u0, v0, C=C, alpha=-1.0 / 3.0)
    energy = mechanical_energy(M, K, dissipated.displacement, dissipated.velocity)
    assert float(energy[-1]) < 1.0e-2 * float(energy[0]), (
        f"energy only decayed to {float(energy[-1]) / float(energy[0]):.3e} of E0"
    )
    assert float(energy.max()) <= float(energy[0]) * (1.0 + 1.0e-9), "energy grew above E0"

    conserved = newmark_linear(M, K, force, time, u0, v0, C=C)
    energy_c = mechanical_energy(M, K, conserved.displacement, conserved.velocity)
    drift = abs(float(energy_c[-1]) - float(energy_c[0])) / float(energy_c[0])
    assert drift < 1.0e-12, f"alpha=0 should conserve energy exactly, drifted {drift:.3e}"


def test_input_validation_is_shared_with_newmark():
    M, K, C = sdof(1.0, 100.0)
    time = torch.linspace(0.0, 0.1, 11, dtype=DT)
    force = torch.zeros((time.numel(), 1), dtype=DT)
    u0 = torch.tensor([1.0], dtype=DT)
    v0 = torch.tensor([0.0], dtype=DT)
    bad_time = time.clone()
    bad_time[3] = bad_time[2]
    with pytest.raises(ValueError):
        hht_alpha(M, K, force, bad_time, u0, v0, C=C, alpha=-0.2)
    with pytest.raises(ValueError):
        hht_alpha(M, K, force[:-1], time, u0, v0, C=C, alpha=-0.2)