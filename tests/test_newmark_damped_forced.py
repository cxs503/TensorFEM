"""Verification of the damped / loaded paths of ``tensorfem.dynamics.newmark_linear``.

Every assertion is against a **closed-form** solution of the same initial-value
problem, not against a stored snapshot of this repository's own output.  The
motivation is a coverage gap: all pre-existing uses of ``newmark_linear`` pass
``C=None`` and a zero load history, so the damping terms ``c1``/``c4``/``c5`` of
the recurrence never contributed to any assertion before this file.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.dynamics import mechanical_energy, newmark_linear

DT = torch.float64
GATE = 1.0e-2


# --------------------------------------------------------------------------- #
# closed-form references
# --------------------------------------------------------------------------- #
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


def damped_step_exact(time, *, p0, k, omega, zeta):
    wd = omega * math.sqrt(1.0 - zeta * zeta)
    return torch.tensor(
        [
            p0
            / k
            * (
                1.0
                - math.exp(-zeta * omega * t)
                * (math.cos(wd * t) + zeta * omega / wd * math.sin(wd * t))
            )
            for t in time.tolist()
        ],
        dtype=DT,
    )


def rel_inf(fe, exact):
    return float((fe - exact).abs().max()) / float(exact.abs().max())


def _time(omega, steps_per_period, periods):
    period = 2.0 * math.pi / omega
    dt = period / steps_per_period
    return torch.arange(0.0, periods * period + 0.5 * dt, dt, dtype=DT)


def _solve_damped_free(*, omega=7.0, zeta=0.05, u0=0.02, steps_per_period=200,
                       periods=4.0):
    time = _time(omega, steps_per_period, periods)
    mass = torch.tensor([[1.0]], dtype=DT)
    stiffness = mass * omega**2
    damping = 2.0 * zeta * omega * mass
    result = newmark_linear(
        mass, stiffness, torch.zeros((time.numel(), 1), dtype=DT), time,
        torch.tensor([u0], dtype=DT), torch.zeros(1, dtype=DT), C=damping,
    )
    return time, mass, stiffness, damping, result


def recurrence_ignoring_damping(mass, stiffness, force, time, u0, v0):
    """Same recurrence with the damping operator deleted (the old code shape)."""
    dt = float(time[1] - time[0])
    n, ndof = time.numel(), mass.shape[0]
    u = torch.zeros((n, ndof), dtype=DT)
    v = torch.zeros_like(u)
    a = torch.zeros_like(u)
    u[0], v[0] = u0, v0
    a[0] = torch.linalg.solve(mass, force[0] - stiffness @ u0)
    c0, c2, c3 = 1.0 / (0.25 * dt**2), 1.0 / (0.25 * dt), 0.5
    effective = stiffness + c0 * mass
    for i in range(n - 1):
        rhs = force[i + 1] + mass @ (c0 * u[i] + c2 * v[i] + c3 * a[i])
        u[i + 1] = torch.linalg.solve(effective, rhs)
        a[i + 1] = c0 * (u[i + 1] - u[i]) - c2 * v[i] - c3 * a[i]
        v[i + 1] = v[i] + dt * (0.5 * a[i] + 0.5 * a[i + 1])
    return u


# --------------------------------------------------------------------------- #
# damped free vibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("zeta", [0.02, 0.05, 0.2, 0.5])
def test_damped_free_vibration_matches_closed_form(zeta):
    time, _, _, _, result = _solve_damped_free(zeta=zeta)
    exact = damped_free_exact(time, u0=0.02, v0=0.0, omega=7.0, zeta=zeta)
    assert rel_inf(result.displacement[:, 0], exact) < GATE


def test_damping_only_case_isolates_the_damping_coefficients():
    """``K = 0``: only ``c1``, ``c4`` and ``c5`` can produce the decay."""
    mass_value, damping_value, initial_velocity = 2.0, 1.5, 3.0
    time = torch.linspace(0.0, 4.0, 4001, dtype=DT)
    mass = torch.tensor([[mass_value]], dtype=DT)
    damping = torch.tensor([[damping_value]], dtype=DT)
    result = newmark_linear(
        mass, torch.zeros((1, 1), dtype=DT), torch.zeros((time.numel(), 1), dtype=DT),
        time, torch.zeros(1, dtype=DT),
        torch.tensor([initial_velocity], dtype=DT), C=damping,
    )
    decay = damping_value / mass_value
    exact = torch.tensor(
        [mass_value * initial_velocity / damping_value * (1.0 - math.exp(-decay * t))
         for t in time.tolist()],
        dtype=DT,
    )
    assert rel_inf(result.displacement[:, 0], exact) < 1.0e-3

    # negative control: the same problem without the damping terms must be wrong
    broken = recurrence_ignoring_damping(
        mass, torch.zeros((1, 1), dtype=DT), torch.zeros((time.numel(), 1), dtype=DT),
        time, torch.zeros(1, dtype=DT), torch.tensor([initial_velocity], dtype=DT),
    )
    assert rel_inf(broken[:, 0], exact) > 0.3


# --------------------------------------------------------------------------- #
# damped step load
# --------------------------------------------------------------------------- #
def test_damped_step_response_matches_closed_form():
    omega, zeta, p0, k = 50.0, 0.08, 400.0, 2500.0
    time = _time(omega, 200, 6.0)
    mass = torch.tensor([[1.0]], dtype=DT)
    stiffness = torch.tensor([[k]], dtype=DT)
    damping = 2.0 * zeta * omega * mass
    force = torch.full((time.numel(), 1), p0, dtype=DT)
    result = newmark_linear(
        mass, stiffness, force, time, torch.zeros(1, dtype=DT),
        torch.zeros(1, dtype=DT), C=damping,
    )
    exact = damped_step_exact(time, p0=p0, k=k, omega=omega, zeta=zeta)
    assert rel_inf(result.displacement[:, 0], exact) < GATE
    # the loaded path must be exercised: a run with no damping input differs
    undamped = newmark_linear(
        mass, stiffness, force, time, torch.zeros(1, dtype=DT),
        torch.zeros(1, dtype=DT),
    )
    assert rel_inf(undamped.displacement[:, 0], exact) > GATE


def test_damped_step_second_order_convergence():
    omega, zeta, p0, k = 50.0, 0.08, 400.0, 2500.0
    errors = []
    dts = []
    for steps in (50, 100, 200):
        time = _time(omega, steps, 4.0)
        mass = torch.tensor([[1.0]], dtype=DT)
        stiffness = torch.tensor([[k]], dtype=DT)
        damping = 2.0 * zeta * omega * mass
        result = newmark_linear(
            mass, stiffness, torch.full((time.numel(), 1), p0, dtype=DT), time,
            torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT), C=damping,
        )
        errors.append(
            rel_inf(result.displacement[:, 0],
                    damped_step_exact(time, p0=p0, k=k, omega=omega, zeta=zeta))
        )
        dts.append(float(time[1] - time[0]))
    order = math.log(errors[0] / errors[-1]) / math.log(dts[0] / dts[-1])
    assert 1.8 <= order <= 2.3
    assert errors[-1] < errors[0]


# --------------------------------------------------------------------------- #
# harmonic forcing
# --------------------------------------------------------------------------- #
def test_harmonic_forced_steady_state_amplitude_and_phase():
    k, m, zeta, f0, ratio = 2500.0, 1.0, 0.05, 120.0, 0.8
    omega = math.sqrt(k / m)
    drive = ratio * omega
    time = _time(omega, 400, 80.0)
    mass = torch.tensor([[m]], dtype=DT)
    stiffness = torch.tensor([[k]], dtype=DT)
    damping = 2.0 * zeta * omega * mass
    force = (f0 * torch.cos(drive * time)).reshape(-1, 1)
    result = newmark_linear(
        mass, stiffness, force, time, torch.zeros(1, dtype=DT),
        torch.zeros(1, dtype=DT), C=damping,
    )
    drive_period = 2.0 * math.pi / drive
    window = time >= float(time[-1]) - drive_period
    design = torch.stack((torch.cos(drive * time[window]),
                          torch.sin(drive * time[window])), dim=1)
    solution = torch.linalg.lstsq(
        design, result.displacement[window, 0].reshape(-1, 1)
    ).solution
    amplitude = math.hypot(float(solution[0, 0]), float(solution[1, 0]))
    phase = math.atan2(float(solution[1, 0]), float(solution[0, 0]))
    exact_amplitude = (f0 / k) / math.sqrt(
        (1.0 - ratio**2) ** 2 + (2.0 * zeta * ratio) ** 2
    )
    exact_phase = math.atan2(2.0 * zeta * ratio, 1.0 - ratio**2)
    assert abs(amplitude - exact_amplitude) / exact_amplitude < GATE
    assert abs(phase - exact_phase) * 180.0 / math.pi < 0.5


# --------------------------------------------------------------------------- #
# multiple degrees of freedom and proportional damping
# --------------------------------------------------------------------------- #
def test_two_dof_proportional_damping_matches_modal_superposition():
    alpha, beta = 0.4, 2.0e-4
    mass = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=DT)
    stiffness = torch.tensor([[6500.0, -2500.0], [-2500.0, 2500.0]], dtype=DT)
    damping = alpha * mass + beta * stiffness
    p = torch.tensor([300.0, 150.0], dtype=DT)
    lower = torch.linalg.cholesky(mass)
    inverse = torch.linalg.inv(lower)
    eigenvalues, trial = torch.linalg.eigh(inverse @ stiffness @ inverse.T)
    vectors = inverse.T @ trial
    omega = torch.sqrt(eigenvalues)
    period = 2.0 * math.pi / float(omega.min())
    dt = period / 200.0
    time = torch.arange(0.0, 8.0 * period + 0.5 * dt, dt, dtype=DT)
    result = newmark_linear(
        mass, stiffness, p.expand(time.numel(), 2).contiguous(), time,
        torch.zeros(2, dtype=DT), torch.zeros(2, dtype=DT), C=damping,
    )
    modal = []
    for i in range(2):
        wi = float(omega[i])
        modal_zeta = (alpha + beta * wi * wi) / (2.0 * wi)
        modal.append(
            damped_step_exact(time, p0=float(vectors[:, i] @ p), k=wi * wi,
                              omega=wi, zeta=modal_zeta)
        )
    exact = torch.stack(modal, dim=1) @ vectors.T
    assert rel_inf(result.displacement, exact) < GATE


# --------------------------------------------------------------------------- #
# energy bookkeeping
# --------------------------------------------------------------------------- #
def test_energy_ledger_matches_work_minus_dissipation():
    omega, zeta, p0, k = 50.0, 0.1, 400.0, 2500.0
    time = _time(omega, 200, 12.0)
    mass = torch.tensor([[1.0]], dtype=DT)
    stiffness = torch.tensor([[k]], dtype=DT)
    damping = 2.0 * zeta * omega * mass
    force = torch.full((time.numel(), 1), p0, dtype=DT)
    result = newmark_linear(
        mass, stiffness, force, time, torch.zeros(1, dtype=DT),
        torch.zeros(1, dtype=DT), C=damping,
    )
    energy = mechanical_energy(mass, stiffness, result.displacement, result.velocity)
    work_rate = (result.velocity * force).sum(dim=1)
    loss_rate = (result.velocity * (damping @ result.velocity.T).T).sum(dim=1)
    dt = float(time[1] - time[0])

    def cumtrapz(values):
        out = torch.zeros_like(values)
        out[1:] = torch.cumsum(0.5 * (values[1:] + values[:-1]) * dt, dim=0)
        return out

    predicted = energy[0] + cumtrapz(work_rate) - cumtrapz(loss_rate)
    assert float((energy - predicted).abs().max()) / float(energy.abs().max()) < 1.0e-3
    # after twelve periods the damped response sits on the static energy
    static = p0**2 / (2.0 * k)
    assert abs(float(energy[-1]) - static) / static < GATE


def test_damped_free_decay_never_gains_energy():
    _, mass, stiffness, _, result = _solve_damped_free(zeta=0.08, periods=6.0)
    energy = mechanical_energy(mass, stiffness, result.displacement, result.velocity)
    assert torch.all(energy[1:] <= energy[:-1] + 1e-12 * float(energy[0]))
    assert float(energy[-1]) < 0.1 * float(energy[0])


# --------------------------------------------------------------------------- #
# other members of the family, and fail-closed input handling
# --------------------------------------------------------------------------- #
def test_alternate_beta_gamma_member_matches_the_same_reference():
    omega, zeta = 7.0, 0.05
    time = _time(omega, 200, 4.0)
    mass = torch.tensor([[1.0]], dtype=DT)
    stiffness = mass * omega**2
    damping = 2.0 * zeta * omega * mass
    result = newmark_linear(
        mass, stiffness, torch.zeros((time.numel(), 1), dtype=DT), time,
        torch.tensor([0.02], dtype=DT), torch.zeros(1, dtype=DT), C=damping,
        beta=1.0 / 6.0, gamma=0.5,
    )
    exact = damped_free_exact(time, u0=0.02, v0=0.0, omega=omega, zeta=zeta)
    assert rel_inf(result.displacement[:, 0], exact) < GATE


def test_damped_and_loaded_arguments_fail_closed():
    time = torch.linspace(0.0, 1.0, 11, dtype=DT)
    mass = torch.eye(1, dtype=DT)
    zero_force = torch.zeros((time.numel(), 1), dtype=DT)
    with pytest.raises(ValueError):
        newmark_linear(mass, mass, zero_force, time, torch.zeros(1, dtype=DT),
                       torch.zeros(1, dtype=DT), C=torch.zeros((2, 2), dtype=DT))
    with pytest.raises(ValueError):
        newmark_linear(mass, mass, torch.zeros((time.numel(), 2), dtype=DT), time,
                       torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT))
    strained = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0], dtype=DT)
    with pytest.raises(ValueError):
        newmark_linear(mass, mass, torch.zeros((7, 1), dtype=DT), strained,
                       torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT))