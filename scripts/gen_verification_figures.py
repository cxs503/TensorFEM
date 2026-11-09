"""Generate verification figures for the structural dynamics verification report."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from tensorfem.modal import (
    beam_bending_stiffness, cantilever_beam_modes, simply_supported_beam_modes,
    cantilever_exact_angular_frequency, simply_supported_exact_angular_frequency,
)
from tensorfem.timoshenko_beam_modal import (
    TimoshenkoBeamSection, simply_supported_timoshenko_modes,
    exact_simply_supported_timoshenko_angular_frequencies,
)
from tensorfem.plate_vibration import (
    solve_plate_vibration, kirchhoff_angular_frequency, BOUNDARY_SIMPLE,
)
from tensorfem.plate_uniform_load import ss_uniform_kirchhoff
from tensorfem.structural_verification import (
    solve_cantilever_udl, euler_bernoulli_cantilever_deflection,
    beam_field, cantilever_exact_angular_frequencies,
)
from tensorfem.dynamics import newmark_linear

outdir = '/root/tp-agent/work/m8142/artifacts'

# Figure 1: Cantilever UDL deflection comparison
def plot_cantilever_udl():
    L, EI, q = 1.0, 200.0, 1000.0
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, L, 200)
    w_exact = np.array([euler_bernoulli_cantilever_deflection(xi, L, EI, q) for xi in x])
    ax.plot(x, w_exact * 1e3, 'k-', linewidth=2, label='Analytical (quartic)')
    
    for ne, style in [(2, 'ro--'), (4, 'bs--'), (8, 'g^--')]:
        sol = solve_cantilever_udl(L, EI, q, elements=ne)
        xs, w_fe, _ = beam_field(sol, samples_per_element=8)
        ax.plot(xs, w_fe * 1e3, style, markersize=4, label=f'FE (n={ne})')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Deflection w (mm)')
    ax.set_title('Cantilever Beam under Uniform Load: FE vs Analytical')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig01_cantilever_udl.png', dpi=150)
    plt.close(fig)
    print(f'Saved {outdir}/fig01_cantilever_udl.png')

# Figure 2: Beam frequency convergence
def plot_beam_freq_convergence():
    L, E, b, h = 1.0, 2.1e11, 0.02, 0.02
    I = b * h**3 / 12
    A = b * h
    rho = 7850.0
    rho_a = rho * A
    EI = E * I
    
    fig, ax = plt.subplots(figsize=(8, 5))
    elements_list = [4, 8, 16, 32, 64]
    
    from tensorfem.structural_verification import cantilever_exact_angular_frequencies as sv_freqs
    exact_omegas = sv_freqs(L, EI, rho_a, modes=5)
    for mode in range(1, 6):
        exact_omega = exact_omegas[mode - 1]
        exact_hz = exact_omega / (2 * math.pi)
        errors = []
        for ne in elements_list:
            result = cantilever_beam_modes(L, ne, EI, rho_a, modes=5)
            fe_hz = result.frequencies_hz[mode-1].item()
            err = abs(fe_hz - exact_hz) / exact_hz * 100
            errors.append(err)
        ax.semilogy(elements_list, errors, 'o-', label=f'Mode {mode}')
    
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Cantilever Beam: Frequency Convergence (Euler-Bernoulli)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig02_beam_freq_convergence.png', dpi=150)
    plt.close(fig)
    print(f'Saved {outdir}/fig02_beam_freq_convergence.png')

# Figure 3: Plate frequency comparison
def plot_plate_freq_comparison():
    side, young, poisson, thickness, density = 1.0, 2.1e11, 0.3, 0.01, 7850
    
    fig, ax = plt.subplots(figsize=(8, 5))
    mode_pairs = [(1,1), (1,2), (2,1), (2,2), (1,3), (3,1)]
    mode_labels = ['(1,1)', '(1,2)', '(2,1)', '(2,2)', '(1,3)', '(3,1)']
    
    exact_hz_list = []
    for m, n in mode_pairs:
        omega = kirchhoff_angular_frequency(
            length=side, width=side, density=density,
            young=young, poisson=poisson, thickness=thickness,
            half_waves_x=m, half_waves_y=n,
        )
        exact_hz_list.append(omega / (2 * math.pi))
    
    x = np.arange(len(mode_pairs))
    width = 0.35
    
    fe_8 = []
    fe_12 = []
    for m, n in mode_pairs:
        for nx, lst in [(8, fe_8), (12, fe_12)]:
            result = solve_plate_vibration(
                elements_x=nx, elements_y=nx, length=side, width=side,
                young=young, poisson=poisson, thickness=thickness,
                density=density, boundary=BOUNDARY_SIMPLE, modes=6,
            )
            idx = mode_pairs.index((m, n))
            lst.append(result.frequencies_hz[idx].item())
    
    ax.bar(x - width/2, fe_8, width, label='FE (8×8)', alpha=0.7)
    ax.bar(x + width/2, fe_12, width, label='FE (12×12)', alpha=0.7)
    ax.plot(x, exact_hz_list, 'k*', markersize=12, label='Analytical (Kirchhoff)')
    
    ax.set_xlabel('Mode (m,n)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Simply Supported Square Plate: Natural Frequencies')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig03_plate_frequencies.png', dpi=150)
    plt.close(fig)
    print(f'Saved {outdir}/fig03_plate_frequencies.png')

# Figure 4: Transient response
def plot_transient_response():
    L, E, b, h = 2.0, 2.1e11, 0.05, 0.05
    I = b * h**3 / 12
    A = b * h
    rho = 7850.0
    rho_a = rho * A
    EI = E * I
    
    from tensorfem.modal import simply_supported_beam_modes
    result = simply_supported_beam_modes(L, 32, EI, rho_a, modes=5)
    omega1 = result.angular_frequencies[0].item()
    T1 = 2 * math.pi / omega1
    
    I_0 = 100.0
    M_eff = 0.5 * rho_a * L
    v0_val = I_0 / M_eff
    
    M = torch.tensor([[M_eff]], dtype=torch.float64)
    K = torch.tensor([[M_eff * omega1**2]], dtype=torch.float64)
    dt = T1 / 200
    n_steps = 400
    time = torch.arange(0, n_steps * dt, dt, dtype=torch.float64)
    u0 = torch.tensor([0.0], dtype=torch.float64)
    force = torch.zeros((len(time), 1), dtype=torch.float64)
    
    v0 = torch.tensor([v0_val], dtype=torch.float64)
    transient = newmark_linear(M, K, force, time, u0, v0)
    
    # Analytical solution
    w_peak = v0_val / omega1
    t_analytical = time.numpy()
    u_analytical = w_peak * np.sin(omega1 * t_analytical)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time.numpy() * 1000, transient.displacement[:, 0].numpy() * 1e3, 'b-', linewidth=1.5, label='Newmark (SDOF)')
    ax.plot(t_analytical * 1000, u_analytical * 1e3, 'r--', linewidth=1, label='Analytical')
    ax.axvline(T1/4 * 1000, color='g', linestyle=':', alpha=0.5, label=f'Peak at T₁/4 = {T1/4*1000:.1f} ms')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Displacement (mm)')
    ax.set_title('SDOF Impulse Response: Newmark vs Analytical')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig04_transient_response.png', dpi=150)
    plt.close(fig)
    print(f'Saved {outdir}/fig04_transient_response.png')

# Figure 5: Convergence order
def plot_convergence():
    from tensorfem.structural_verification import beam_l2_relative_error, observed_order
    
    L, EI, q = 1.0, 200.0, 1000.0
    elements_list = [2, 4, 8, 16, 32]
    errors = []
    for ne in elements_list:
        sol = solve_cantilever_udl(L, EI, q, elements=ne)
        l2_err = beam_l2_relative_error(sol, samples_per_element=16)
        errors.append(l2_err)
    
    h = np.array([L / ne for ne in elements_list])
    e = np.array(errors)
    order = observed_order(h, e)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(h, e, 'bo-', markersize=6, linewidth=2, label=f'FE (order={order:.2f})')
    
    # Reference lines
    h_ref = np.array([h[0], h[-1]])
    ax.loglog(h_ref, e[0] * (h_ref/h[0])**4, 'k--', alpha=0.5, label='O(h⁴) reference')
    ax.loglog(h_ref, e[0] * (h_ref/h[0])**2, 'k:', alpha=0.3, label='O(h²) reference')
    
    ax.set_xlabel('Element size h (m)')
    ax.set_ylabel('L₂ relative error')
    ax.set_title('Cantilever UDL: L₂ Convergence (Cubic Hermite Beam)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig05_convergence.png', dpi=150)
    plt.close(fig)
    print(f'Saved {outdir}/fig05_convergence.png')

if __name__ == '__main__':
    import os
    os.makedirs(outdir, exist_ok=True)
    plot_cantilever_udl()
    plot_beam_freq_convergence()
    plot_plate_freq_comparison()
    plot_transient_response()
    plot_convergence()
    print('All figures generated.')
