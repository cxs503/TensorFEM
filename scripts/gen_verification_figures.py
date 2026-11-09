"""Generate verification report figures for P97-N04."""
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.modal import (
    cantilever_beam_modes,
    simply_supported_beam_modes,
    cantilever_exact_angular_frequency,
    simply_supported_exact_angular_frequency,
)
from tensorfem.dynamics import newmark_linear, mechanical_energy
from tensorfem.plate_vibration import (
    solve_plate_vibration,
    kirchhoff_angular_frequency,
)

# ── Figure 1: Beam frequency convergence ──────────────────────────────────
def fig_beam_convergence():
    L, E, b, h = 5.0, 210e9, 0.1, 0.15
    rho = 7800.0
    EI = E * b * h**3 / 12
    rho_a = rho * b * h

    elements_list = [4, 8, 16, 32, 64]
    modes_to_plot = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes_to_plot:
        omega_exact = cantilever_exact_angular_frequency(L, EI, rho_a, mode)
        f_exact = omega_exact / (2 * math.pi)

        errors = []
        for n_elem in elements_list:
            result = cantilever_beam_modes(L, n_elem, EI, rho_a, modes=mode)
            f_fem = result.frequencies_hz[mode - 1].item()
            errors.append(abs(f_fem - f_exact) / f_exact * 100)

        ax.semilogy(elements_list, errors, 'o-', label=f'Mode {mode} (f_exact={f_exact:.1f} Hz)')

    ax.set_xlabel('Number of elements', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title('Cantilever Beam: Frequency Convergence vs Analytical', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-4, 10)
    fig.tight_layout()
    fig.savefig('artifacts/fig1_beam_convergence.png', dpi=150)
    plt.close(fig)
    print("Figure 1 saved")

# ── Figure 2: SDOF step response vs exact ─────────────────────────────────
def fig_sdof_step_response():
    omega = 10.0
    m = 1.0
    k = m * omega**2
    F0 = 100.0

    M = torch.tensor([[m]], dtype=torch.float64)
    K = torch.tensor([[k]], dtype=torch.float64)

    period = 2 * math.pi / omega
    dt = period / 200
    time = torch.arange(0, 3 * period + dt / 2, dt, dtype=torch.float64)
    force = torch.full((len(time), 1), F0, dtype=torch.float64)

    u0 = torch.tensor([0.0], dtype=torch.float64)
    v0 = torch.tensor([0.0], dtype=torch.float64)

    result = newmark_linear(M, K, force, time, u0, v0)
    u_exact = F0 / k * (1 - torch.cos(omega * time))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    # Time history
    ax1.plot(time.numpy(), result.displacement[:, 0].numpy(), 'b-', linewidth=1.5, label='Newmark (β=1/4)')
    ax1.plot(time.numpy(), u_exact.numpy(), 'r--', linewidth=1.5, label='Exact: $u_0(1-\\cos\\omega t)$')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Displacement (m)', fontsize=12)
    ax1.set_title('SDOF Step Response: Newmark vs Duhamel Integral', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Error
    error = torch.abs(result.displacement[:, 0] - u_exact) / (F0 / k) * 100
    ax2.plot(time.numpy(), error.numpy(), 'g-', linewidth=1)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Relative error (%)', fontsize=12)
    ax2.set_title('Pointwise Relative Error', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.5)

    fig.tight_layout()
    fig.savefig('artifacts/fig2_sdof_step_response.png', dpi=150)
    plt.close(fig)
    print("Figure 2 saved")

# ── Figure 3: Energy conservation over 100 cycles ─────────────────────────
def fig_energy_conservation():
    omega = 5.0
    m = 1.0
    k = m * omega**2
    period = 2 * math.pi / omega
    dt = period / 50
    n_cycles = 100
    time = torch.arange(0, n_cycles * period + dt / 2, dt, dtype=torch.float64)

    M = torch.tensor([[m]], dtype=torch.float64)
    K = torch.tensor([[k]], dtype=torch.float64)
    force = torch.zeros((len(time), 1), dtype=torch.float64)

    u0 = torch.tensor([0.01], dtype=torch.float64)
    v0 = torch.tensor([0.0], dtype=torch.float64)

    result = newmark_linear(M, K, force, time, u0, v0)
    energy = mechanical_energy(M, K, result.displacement, result.velocity)
    E0 = energy[0].item()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    # Energy time history
    ax1.plot(time.numpy(), energy.numpy(), 'b-', linewidth=1)
    ax1.axhline(y=E0, color='r', linestyle='--', linewidth=1, label=f'E₀ = {E0:.6e} J')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Total energy (J)', fontsize=12)
    ax1.set_title(f'Energy Conservation: {n_cycles} Cycles of Undamped Free Vibration', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Energy drift
    drift = (energy - E0) / E0 * 100
    ax2.plot(time.numpy(), drift.numpy(), 'g-', linewidth=1)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Energy drift (%)', fontsize=12)
    ax2.set_title(f'Max drift = {torch.max(torch.abs(drift)).item():.4f}%', fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('artifacts/fig3_energy_conservation.png', dpi=150)
    plt.close(fig)
    print("Figure 3 saved")

# ── Figure 4: Plate frequency convergence ─────────────────────────────────
def fig_plate_convergence():
    a = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    rho = 7800.0

    omega_exact = kirchhoff_angular_frequency(
        length=a, width=a, density=rho, young=E, poisson=nu,
        thickness=t, half_waves_x=1, half_waves_y=1
    )
    f_exact = omega_exact / (2 * math.pi)

    mesh_sizes = [4, 6, 8]
    freqs = []
    for nx in mesh_sizes:
        result = solve_plate_vibration(
            elements_x=nx, elements_y=nx, length=a, width=a,
            young=E, poisson=nu, thickness=t, density=rho,
            boundary="simple", modes=1,
        )
        freqs.append(result.frequencies_hz[0].item())

    errors = [abs(f - f_exact) / f_exact * 100 for f in freqs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(mesh_sizes, errors, 'o-', color='navy', linewidth=2, markersize=8)
    ax.set_xlabel('Mesh size (elements per side)', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title(f'Simply Supported Square Plate: Mode (1,1) Frequency Convergence\n'
                 f'Exact f = {f_exact:.2f} Hz (Navier)', fontsize=13)
    ax.grid(True, alpha=0.3)
    for i, (nx, err) in enumerate(zip(mesh_sizes, errors)):
        ax.annotate(f'{err:.2f}%', (nx, err), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)
    fig.tight_layout()
    fig.savefig('artifacts/fig4_plate_convergence.png', dpi=150)
    plt.close(fig)
    print("Figure 4 saved")

if __name__ == '__main__':
    fig_beam_convergence()
    fig_sdof_step_response()
    fig_energy_conservation()
    fig_plate_convergence()
    print("All figures generated successfully")
