"""Ice-region ship vibration-noise prediction chain (campaign node IN13-W2-N06).

Physical goal
-------------
Build the complete prediction chain from an ice load time history to the
underwater radiated noise level of an ice-going ship:

    Stage 1  DEM ice load time history  ->  nodal force spectrum (FFT)
    Stage 2  hull girder modal superposition (wet added mass from a
             2-D potential-flow slice)  ->  girder transient response
    Stage 3  underwater radiated noise (baffled hull panel in water,
             Rayleigh / boundary-integral far field)  ->  noise level at 1 m

The chain is *stage-verified*: every stage is pinned to an independent
analytic anchor so that a regression in any stage is caught by a test.

Analytic anchors
----------------
A1. **Ice spectrum.**  A synthetic multi-tone + noise load must have its
    tone amplitudes recovered by the FFT extractor to < 1 %.

A2. **Girder modal superposition.**  The reduced-order modal-superposition
    transient must match a direct Newmark-beta time integration of the full
    (wet) system in acceleration level (RMS) to < 3 %, and the single-mode
    steady state must match the closed-form damped-oscillator solution.
    The wet added mass per unit length uses the exact 2-D incompressible
    potential-flow slice results (circular: ``rho*pi*r^2``; square:
    ``(pi/4)*rho*a^2``).

A3. **Underwater radiation.**  For a baffled simply-supported rectangular
    panel in water:

    * the radiation efficiency ``sigma`` from the Rayleigh integral must
      match the exact low-frequency (monopole) limit

          sigma = k^2 Q^2 / (pi a b V0^2),   Q = int v dS

      (for mode (1,1): ``sigma = 16 k^2 a b / pi^5``) to < 5 % over the
      band ``0.5 fn < f < 2 fn``;

    * the far-field pressure at 1 m from the full Rayleigh field-point
      integral must match the baffled-monopole semi-analytic solution
      ``p = j rho w Q e^{-j k r} / (2 pi r)`` to < 3 dB in the ``ka << 1``
      regime;

    * the power balance ``P_in = P_rad`` (no structural damping) closes.

References
----------
* Leppington, Broadbent & Heron (1982), "The acoustic radiation efficiency
  of rectangular panels", Proc. R. Soc. Lond. A 382, 245-271.
* Maidanik (1962), J. Acoust. Soc. Am. 34, 809-826.
* Riska (1995), model-test programme for ice-going ship noise (secondary
  reference; see report for the data-availability limitation).
* Blevins (1979), *Formulas for Natural Frequency and Mode Shape*.
* Milne-Thomson (1968), *Theoretical Hydrodynamics* (2-D added mass).
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

from tensorfem.acoustic_radiation import (
    radiation_efficiency_rayleigh,
    rayleigh_radiated_power,
)
from tensorfem.modal import solve_modes, uniform_beam_matrices

__all__ = [
    # Stage 1
    "dem_ice_load_time_history",
    "extract_force_spectrum",
    # Stage 2
    "slice_added_mass_per_length",
    "girder_modal_superposition",
    "girder_newmark_reference",
    "single_mode_steady_state",
    # Stage 3
    "mode_volume_velocity",
    "monopole_sigma_lowfreq",
    "baffled_far_field_pressure",
    "baffled_monopole_pressure",
    "noise_level_db",
    "underwater_sigma_rayleigh",
    # Chain
    "IceNoiseChainConfig",
    "run_ice_noise_chain",
]

WATER_RHO = 1025.0
WATER_C = 1500.0
P_REF_UNDERWATER = 1.0e-6  # Pa, 1 micro-Pa reference for underwater acoustics


# ===========================================================================
# Stage 1 -- ice excitation extraction
# ===========================================================================

def dem_ice_load_time_history(
    t: Sequence[float],
    impacts: Iterable[Sequence[float]],
) -> torch.Tensor:
    """Build a DEM-style ice load time history as a sum of half-sine impacts.

    A discrete-element ice contact is modelled as a sequence of local
    crushing pulses.  Each impact is a half-sine pulse

        f_i(t) = A_i * sin(pi (t - t0_i) / T_i),   t in [t0_i, t0_i + T_i]

    which is the standard idealisation of a single ice-floe contact event
    (rise and fall over the contact duration ``T_i`` with peak ``A_i``).

    Parameters
    ----------
    t : sequence of float
        Sample times [s] (uniform spacing assumed).
    impacts : iterable of (t0, duration, amplitude)
        Start time, duration and peak force [N] of each impact.

    Returns
    -------
    torch.Tensor
        Force time history [N], shape ``(len(t),)``.
    """
    t = torch.as_tensor(list(t), dtype=torch.float64)
    force = torch.zeros_like(t)
    for t0, dur, amp in impacts:
        if dur <= 0:
            raise ValueError("impact duration must be positive")
        mask = (t >= t0) & (t <= t0 + dur)
        if bool(mask.any()):
            phase = math.pi * (t[mask] - t0) / dur
            force[mask] += amp * torch.sin(phase)
    return force


def extract_force_spectrum(
    t: Sequence[float],
    force: Sequence[float] | torch.Tensor,
    *,
    window: str = "hanning",
) -> dict:
    """Extract the one-sided amplitude spectrum and PSD of a force history.

    Uses a float64 FFT with an optional Hanning window.  The amplitude
    spectrum is the peak (not RMS) amplitude of each bin, corrected for the
    coherent gain of the window so that a pure tone returns its true peak
    amplitude.  The PSD is the one-sided periodogram (units N^2/Hz) with the
    standard ``2 / (f_s * sum(w^2))`` normalisation (the factor 2 accounts
    for folding the negative frequencies; the DC and Nyquist bins are not
    doubled).

    Returns
    -------
    dict with keys ``freqs``, ``amplitude``, ``psd``, ``dominant_freq``,
    ``dominant_amp``, ``fs``.
    """
    t = torch.as_tensor(list(t), dtype=torch.float64)
    force = torch.as_tensor(list(force), dtype=torch.float64)
    if t.shape != force.shape or t.numel() < 4:
        raise ValueError("t and force must have equal length >= 4")
    if t.numel() > 1:
        dts = torch.diff(t)
        if dts.min() <= 0:
            raise ValueError("t must be strictly increasing")
        if (dts - dts.mean()).abs().max() > 1e-4 * dts.mean():
            raise ValueError("t must be uniformly spaced")
    n = t.numel()
    dt = float(t[1] - t[0])
    fs = 1.0 / dt

    if window == "hanning":
        w = 0.5 * (1.0 - torch.cos(2.0 * math.pi * torch.arange(n) / (n - 1)))
    elif window == "none":
        w = torch.ones(n)
    else:
        raise ValueError("window must be 'hanning' or 'none'")

    x = force * w
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n, d=dt)

    # Coherent gain of the window: a tone exactly on a bin is recovered when
    # dividing by the window sum (Hanning sum = n/2 for this definition).
    w_sum = float(w.sum())
    amplitude = 2.0 * spec.abs() / w_sum
    amplitude[0] = spec.abs()[0] / w_sum  # DC not doubled

    # One-sided PSD: |X|^2 / (fs * sum(w^2)), doubled off DC/Nyquist.
    psd = spec.abs() ** 2 / (fs * float((w * w).sum()))
    if n > 2:
        psd[1:-1] *= 2.0

    k = int(amplitude[1:].argmax().item()) + 1 if n > 2 else 0
    return {
        "freqs": freqs,
        "amplitude": amplitude,
        "psd": psd,
        "dominant_freq": float(freqs[k]),
        "dominant_amp": float(amplitude[k]),
        "fs": fs,
    }


# ===========================================================================
# Stage 2 -- hull girder modal superposition (wet)
# ===========================================================================

def slice_added_mass_per_length(
    cross_section: str,
    size: float,
    rho_water: float = WATER_RHO,
) -> float:
    """2-D incompressible potential-flow added mass per unit length.

    This is the "potential-flow slice" of the task: each girder cross-section
    is a 2-D exterior potential-flow problem and the low-frequency (added)
    mass per unit length has exact closed forms:

    * circular section, radius ``r``:      ``m_a = rho * pi * r^2``
    * square section, side ``a``:          ``m_a = (pi/4) * rho * a^2``

    (Milne-Thomson, *Theoretical Hydrodynamics*; the circular result is the
    classical potential-flow result, the square result follows from the
    conformal-map solution for a square cylinder.)
    """
    cross_section = cross_section.lower()
    size = float(size)
    rho_water = float(rho_water)
    if size <= 0 or rho_water <= 0:
        raise ValueError("size and rho_water must be positive")
    if cross_section == "circular":
        return rho_water * math.pi * size ** 2
    if cross_section == "square":
        return (math.pi / 4.0) * rho_water * size ** 2
    raise ValueError("cross_section must be 'circular' or 'square'")


def _girder_system(
    length: float,
    elements: int,
    EI: float,
    rho_a: float,
    m_a_per_length: float,
):
    """Assemble the wet girder (K, M, M_wet) and the simply-supported BCs."""
    if length <= 0 or elements < 1 or EI <= 0 or rho_a <= 0:
        raise ValueError("length, EI, rho_a must be positive; elements >= 1")
    if m_a_per_length < 0:
        raise ValueError("m_a_per_length must be non-negative")
    K, M = uniform_beam_matrices(length, elements, EI, rho_a)
    M_wet = M + (m_a_per_length / rho_a) * M
    constrained = (0, 2 * elements)  # transverse displacement at both ends
    return K, M, M_wet, constrained


def _girder_free(K: torch.Tensor, constrained: Sequence[int]) -> torch.Tensor:
    ndof = K.shape[0]
    cset = set(int(c) for c in constrained)
    return torch.tensor([i for i in range(ndof) if i not in cset], dtype=torch.long)


def _modal_damping_matrix(
    phi_f: torch.Tensor, omega: torch.Tensor, zeta: float
) -> torch.Tensor:
    """Physical (free-DOF) damping matrix from a modal damping ratio."""
    c_modal = 2.0 * zeta * omega
    return phi_f @ torch.diag(c_modal) @ phi_f.T


def girder_modal_superposition(
    length: float,
    elements: int,
    EI: float,
    rho_a: float,
    m_a_per_length: float,
    force: Callable[[float], torch.Tensor],
    t_end: float,
    dt: float,
    *,
    modes: int = 8,
    zeta: float = 0.01,
) -> dict:
    """Modal-superposition transient of a wet simply-supported girder.

    The wet eigenproblem ``K phi = w^2 M_wet phi`` is solved, the response is
    propagated in the (M_wet-orthonormal) modal coordinates with a 4th-order
    Runge-Kutta integrator, and the physical displacement / acceleration
    fields are reconstructed.

    Parameters
    ----------
    force : callable
        ``force(t) -> (ndof,)`` full-DOF force vector.
    """
    K, M, M_wet, constrained = _girder_system(
        length, elements, EI, rho_a, m_a_per_length
    )
    ndof = K.shape[0]
    free = _girder_free(K, constrained)
    Kf = K[free[:, None], free]
    Mwf = M_wet[free[:, None], free]

    eig = solve_modes(Kf, Mwf, (), modes=modes)
    omega = eig.angular_frequencies
    # ``solve_modes`` was handed the *free-reduced* matrices, so its modes are
    # indexed on the reduced DOF set (0 .. nfree-1) and not on global DOFs; the
    # global basis is rebuilt by scattering them back onto ``free``.
    phi_f = eig.modes  # (nfree, n_modes), M_wet-orthonormal
    n_modes = phi_f.shape[1]
    phi = torch.zeros((ndof, n_modes), dtype=phi_f.dtype)
    phi[free, :] = phi_f

    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError("t_end/dt must be >= 1")
    times = torch.linspace(0.0, t_end, n_steps + 1)

    dtype = omega.dtype
    q = torch.zeros(n_modes, dtype=dtype)
    qd = torch.zeros(n_modes, dtype=dtype)
    u_hist = torch.zeros((n_steps + 1, ndof), dtype=dtype)
    a_hist = torch.zeros((n_steps + 1, ndof), dtype=dtype)

    def accel(qv: torch.Tensor, qdv: torch.Tensor, tnow: float) -> torch.Tensor:
        Ff = force(tnow)[free]
        Q = phi_f.T @ Ff
        return -2.0 * zeta * omega * qdv - (omega ** 2) * qv + Q

    for step in range(n_steps + 1):
        tnow = float(times[step])
        a_modal = accel(q, qd, tnow)
        u_hist[step] = phi @ q
        a_hist[step] = phi @ a_modal
        if step == n_steps:
            break

        def deriv(state: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
            qv = state[:n_modes]
            qdv = state[n_modes:]
            return torch.cat([qdv, accel(qv, qdv, 0.5 * (t0 + t1))])

        s0 = torch.cat([q, qd])
        k1 = deriv(s0, tnow, tnow)
        k2 = deriv(s0 + 0.5 * dt * k1, tnow, tnow + dt)
        k3 = deriv(s0 + 0.5 * dt * k2, tnow, tnow + dt)
        k4 = deriv(s0 + dt * k3, tnow, tnow + dt)
        s_new = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        q, qd = s_new[:n_modes], s_new[n_modes:]

    return {
        "time": times,
        "u": u_hist,
        "a": a_hist,
        "omega": omega,
        "modes": phi,
        "free": free,
        "n_steps": n_steps,
    }


def girder_newmark_reference(
    length: float,
    elements: int,
    EI: float,
    rho_a: float,
    m_a_per_length: float,
    force: Callable[[float], torch.Tensor],
    t_end: float,
    dt: float,
    *,
    modes: int = 8,
    zeta: float = 0.01,
) -> dict:
    """Direct Newmark-beta (average acceleration) reference integration.

    Uses the *same* physical damping as the modal superposition: the damping
    matrix is built as ``C = P diag(2 zeta w_n) P^T`` in the M_wet-orthonormal
    modal basis, so both solvers integrate the identical physical system.
    """
    K, M, M_wet, constrained = _girder_system(
        length, elements, EI, rho_a, m_a_per_length
    )
    ndof = K.shape[0]
    free = _girder_free(K, constrained)
    Kf = K[free[:, None], free]
    Mwf = M_wet[free[:, None], free]

    eig = solve_modes(Kf, Mwf, (), modes=modes)
    omega = eig.angular_frequencies
    # Reduced-system modes (see girder_modal_superposition for the indexing).
    phi_f = eig.modes
    C = _modal_damping_matrix(phi_f, omega, zeta)
    phi = torch.zeros((ndof, phi_f.shape[1]), dtype=phi_f.dtype)
    phi[free, :] = phi_f

    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise ValueError("t_end/dt must be >= 1")
    times = torch.linspace(0.0, t_end, n_steps + 1)

    gamma, beta = 0.5, 0.25
    K_eff = Kf + (gamma / (beta * dt)) * C + (1.0 / (beta * dt ** 2)) * Mwf
    K_eff_inv = torch.linalg.inv(K_eff)

    uf = torch.zeros(free.numel(), dtype=Kf.dtype)
    uf_d = torch.zeros(free.numel(), dtype=Kf.dtype)
    uf_dd = torch.zeros(free.numel(), dtype=Kf.dtype)

    dtype = Kf.dtype
    u_hist = torch.zeros((n_steps + 1, ndof), dtype=dtype)
    a_hist = torch.zeros((n_steps + 1, ndof), dtype=dtype)

    for step in range(n_steps + 1):
        tnow = float(times[step])
        u_hist[step][free] = uf
        a_hist[step][free] = uf_dd
        if step == n_steps:
            break
        F_next = force(tnow + dt)[free]
        rhs = (
            F_next
            - Kf @ (uf + dt * uf_d + dt ** 2 * (0.5 - beta) * uf_dd)
            - C @ (uf_d + dt * (1.0 - gamma) * uf_dd)
        )
        # ``K_eff = M/(beta*dt**2) + gamma/(beta*dt) C + K`` is the Newmark
        # operator scaled by ``1/(beta*dt**2)`` relative to the acceleration
        # form ``M + gamma*dt C + beta*dt**2 K``; the inverse must therefore be
        # rescaled or the acceleration (and the whole response) comes out too
        # small by ``beta*dt**2``.
        uf_dd_new = (K_eff_inv @ rhs) / (beta * dt * dt)
        uf = uf + dt * uf_d + dt ** 2 * ((0.5 - beta) * uf_dd + beta * uf_dd_new)
        uf_d = uf_d + dt * ((1.0 - gamma) * uf_dd + gamma * uf_dd_new)
        uf_dd = uf_dd_new

    return {
        "time": times,
        "u": u_hist,
        "a": a_hist,
        "omega": omega,
        "modes": phi,
        "free": free,
        "n_steps": n_steps,
    }


def single_mode_steady_state(
    omega_n: float,
    zeta: float,
    omega: float,
    f0_over_m: float,
) -> float:
    """Closed-form steady-state amplitude of a damped single mode.

    For ``q'' + 2 zeta w_n q' + w_n^2 q = (F0/M) e^{i w t}`` the steady-state
    amplitude is

        |q| = (F0/M) / sqrt((w_n^2 - w^2)^2 + (2 zeta w_n w)^2).

    This is the exact analytic anchor for the modal superposition of a
    single mode.
    """
    for name, val in (("omega_n", omega_n), ("zeta", zeta),
                     ("omega", omega), ("f0_over_m", f0_over_m)):
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")
    denom = math.sqrt(
        (omega_n ** 2 - omega ** 2) ** 2 + (2.0 * zeta * omega_n * omega) ** 2
    )
    return abs(f0_over_m) / denom


# ===========================================================================
# Stage 3 -- underwater radiated noise (baffled panel in water)
# ===========================================================================

def mode_volume_velocity(
    length: float,
    width: float,
    m: int,
    n: int,
    velocity_amplitude: float,
) -> float:
    """Peak volume velocity ``Q = int v dS`` of a simply-supported mode.

    For ``v = V0 sin(m pi x/a) sin(n pi y/b)``:

        Q = V0 * (a / (m pi)) (1 - (-1)^m) * (b / (n pi)) (1 - (-1)^n).

    Even modes have zero net volume velocity (they radiate as dipoles at
    low frequency); odd-odd modes radiate as monopoles.
    """
    length = float(length)
    width = float(width)
    m = int(m)
    n = int(n)
    if length <= 0 or width <= 0 or m < 1 or n < 1:
        raise ValueError("invalid plate geometry or mode numbers")
    fx = (1.0 - ((-1.0) ** m)) * length / (m * math.pi)
    fy = (1.0 - ((-1.0) ** n)) * width / (n * math.pi)
    return float(velocity_amplitude) * fx * fy


def monopole_sigma_lowfreq(
    length: float,
    width: float,
    frequency: float,
    c: float = WATER_C,
    m: int = 1,
    n: int = 1,
) -> float:
    """Exact ``k -> 0`` (monopole) limit of the baffled-plate efficiency.

    The Rayleigh integral ``P = (rho w / 4 pi) int int v1 v2 sin(kR)/R dS1 dS2``
    has the uniform limit ``sin(kR)/R -> k`` as ``k -> 0``, giving

        P = rho w^2 Q^2 / (4 pi c),
        sigma = P / (rho c S <v^2>) = k^2 Q^2 / (pi a b V0^2),

    with ``<v^2> = V0^2 / 4`` for a single sine mode.  For mode (1,1) this
    reduces to ``sigma = 16 k^2 a b / pi^5``.  This is an *exact* analytic
    result (the low-frequency end of the Leppington 1982 panel family).
    """
    length = float(length)
    width = float(width)
    frequency = float(frequency)
    c = float(c)
    if min(length, width, frequency, c) <= 0:
        raise ValueError("geometry, frequency and c must be positive")
    k = 2.0 * math.pi * frequency / c
    q = mode_volume_velocity(length, width, m, n, 1.0)
    # sigma = k^2 Q^2 / (pi a b V0^2) with V0 = 1 (see docstring derivation).
    return (k ** 2) * q ** 2 / (math.pi * length * width)


def underwater_sigma_rayleigh(
    length: float,
    width: float,
    frequency: float,
    *,
    m: int = 1,
    n: int = 1,
    n_points: int = 40,
    rho: float = WATER_RHO,
    c: float = WATER_C,
) -> float:
    """Radiation efficiency of a baffled panel in water (Rayleigh integral)."""
    return radiation_efficiency_rayleigh(
        length=length,
        width=width,
        frequency=frequency,
        rho=rho,
        c=c,
        m=m,
        n=n,
        velocity_amplitude=1.0,
        n_points=n_points,
    )


def _gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes/weights on [-1, 1] (delegates to the acoustic module)."""
    from tensorfem.acoustic_radiation import _gauss_legendre

    return _gauss_legendre(n)


def baffled_far_field_pressure(
    length: float,
    width: float,
    frequency: float,
    r: float,
    theta: float,
    *,
    rho: float = WATER_RHO,
    c: float = WATER_C,
    m: int = 1,
    n: int = 1,
    velocity_amplitude: float = 1.0,
    n_points: int = 40,
) -> complex:
    """Exact Rayleigh field-point pressure of a baffled panel in water.

    The pressure at a field point ``x_f = r (sin theta, 0, cos theta)`` (theta
    measured from the panel normal) is the single-layer boundary integral

        p(x_f) = (j rho w / 2 pi) int_S v(x) e^{-j k R} / R dS,
        R = |x_f - x|.

    This is the boundary-element (BEM) evaluation of the acoustic field; in
    the far field it reduces to the Fraunhofer form and, for ``ka << 1``, to
    the baffled monopole.  Returns the complex pressure amplitude [Pa].
    """
    length = float(length)
    width = float(width)
    frequency = float(frequency)
    r = float(r)
    theta = float(theta)
    rho = float(rho)
    c = float(c)
    m = int(m)
    n = int(n)
    velocity_amplitude = float(velocity_amplitude)
    n_points = int(n_points)
    if min(length, width, frequency, r, rho, c) <= 0:
        raise ValueError("geometry, frequency, r, rho, c must be positive")
    if m < 1 or n < 1 or n_points < 2:
        raise ValueError("invalid mode numbers or quadrature order")

    omega = 2.0 * math.pi * frequency
    k = omega / c
    nodes, weights = _gauss_legendre(n_points)
    x_pts = 0.5 * length * (nodes + 1.0)
    y_pts = 0.5 * width * (nodes + 1.0)
    wx = 0.5 * length * weights
    wy = 0.5 * width * weights

    xf = r * math.sin(theta)
    zf = r * math.cos(theta)

    total = 0.0 + 0.0j
    for i in range(n_points):
        for j in range(n_points):
            v = velocity_amplitude * math.sin(m * math.pi * x_pts[i] / length) * \
                math.sin(n * math.pi * y_pts[j] / width)
            if v == 0.0:
                continue
            dx = xf - x_pts[i]
            dz = zf
            R = math.sqrt(dx * dx + y_pts[j] ** 2 + dz * dz)
            kernel = cmath.exp(-1j * k * R) / R
            total += v * kernel * wx[i] * wy[j]
    return (1j * rho * omega / (2.0 * math.pi)) * total


def baffled_monopole_pressure(
    frequency: float,
    q_volume: complex,
    r: float,
    *,
    rho: float = WATER_RHO,
    c: float = WATER_C,
) -> complex:
    """Baffled-monopole semi-analytic far-field pressure.

    ``p = j rho w Q e^{-j k r} / (2 pi r)`` -- the half-space (baffled)
    monopole, which is the exact ``ka -> 0`` limit of the full Rayleigh
    field for any panel whose net volume velocity is ``Q``.
    """
    frequency = float(frequency)
    r = float(r)
    rho = float(rho)
    c = float(c)
    if min(frequency, r, rho, c) <= 0:
        raise ValueError("frequency, r, rho, c must be positive")
    omega = 2.0 * math.pi * frequency
    k = omega / c
    return 1j * rho * omega * q_volume * cmath.exp(-1j * k * r) / (2.0 * math.pi * r)


def noise_level_db(
    p_rms: float,
    p_ref: float = P_REF_UNDERWATER,
) -> float:
    """Underwater sound-pressure level ``Lp = 20 log10(p_rms / p_ref)`` [dB]."""
    p_rms = float(p_rms)
    p_ref = float(p_ref)
    if p_rms <= 0 or p_ref <= 0:
        raise ValueError("p_rms and p_ref must be positive")
    return 20.0 * math.log10(p_rms / p_ref)


# ===========================================================================
# Chain driver
# ===========================================================================

@dataclass
class IceNoiseChainConfig:
    """Configuration for :func:`run_ice_noise_chain`.

    Girder (beam) parameters, the panel (plate) parameters, the ice impact
    list and the sampling / solver settings.  All SI units.
    """

    # Girder
    girder_length: float = 20.0
    girder_elements: int = 40
    girder_EI: float = 1.0e10
    girder_rho_a: float = 5000.0  # structural mass per unit length [kg/m]
    girder_cross_section: str = "circular"
    girder_section_size: float = 0.5  # radius (circular) or side (square) [m]
    girder_modes: int = 8
    girder_zeta: float = 0.01

    # Panel (baffled, simply supported, radiating into water)
    panel_length: float = 2.0
    panel_width: float = 1.0
    panel_mode_m: int = 1
    panel_mode_n: int = 1

    # Ice excitation
    impacts: list = field(default_factory=list)  # (t0, duration, amplitude)
    t_end: float = 10.0
    dt: float = 1.0e-3

    # Radiation
    ref_distance: float = 1.0
    rayleigh_n_points: int = 40

    def validate(self) -> None:
        if self.girder_length <= 0 or self.girder_EI <= 0 or self.girder_rho_a <= 0:
            raise ValueError("girder length, EI, rho_a must be positive")
        if self.panel_length <= 0 or self.panel_width <= 0:
            raise ValueError("panel dimensions must be positive")
        if self.t_end <= 0 or self.dt <= 0:
            raise ValueError("t_end and dt must be positive")
        if self.ref_distance <= 0:
            raise ValueError("ref_distance must be positive")


def run_ice_noise_chain(cfg: IceNoiseChainConfig) -> dict:
    """Run the full ice -> girder -> underwater-noise chain.

    Returns a dict with the stage outputs and the final noise level at the
    reference distance.  The panel is driven by the girder velocity at the
    mid-span (the panel attachment point); the panel (1,1) mode velocity
    amplitude at each frequency bin is the girder velocity amplitude there,
    and the radiated noise is energy-summed over the bins.
    """
    cfg.validate()

    # ---- Stage 1: ice load -> spectrum -----------------------------------
    t = torch.arange(0.0, cfg.t_end + 0.5 * cfg.dt, cfg.dt)
    ice_force = dem_ice_load_time_history(t, cfg.impacts)
    spectrum = extract_force_spectrum(t, ice_force)

    # ---- Stage 2: girder modal superposition -----------------------------
    m_a = slice_added_mass_per_length(
        cfg.girder_cross_section, cfg.girder_section_size, WATER_RHO
    )

    def force_fn(tnow: float) -> torch.Tensor:
        # Map the scalar ice force onto the girder mid-span DOF.
        f = torch.zeros(2 * (cfg.girder_elements + 1), dtype=torch.float64)
        f[2 * cfg.girder_elements // 2] = float(ice_force[int(round(tnow / cfg.dt))])
        return f

    girder = girder_modal_superposition(
        cfg.girder_length,
        cfg.girder_elements,
        cfg.girder_EI,
        cfg.girder_rho_a,
        m_a,
        force_fn,
        cfg.t_end,
        cfg.dt,
        modes=cfg.girder_modes,
        zeta=cfg.girder_zeta,
    )

    # Girder velocity at the mid-span (panel attachment), time domain.
    mid_dof = 2 * (cfg.girder_elements // 2)
    u_mid = girder["u"][:, mid_dof]
    v_mid = torch.zeros_like(u_mid)
    v_mid[1:] = (u_mid[1:] - u_mid[:-1]) / cfg.dt

    # ---- Stage 3: per-bin radiation --------------------------------------
    v_spec = extract_force_spectrum(t, v_mid)
    freqs = v_spec["freqs"]
    v_amp = v_spec["amplitude"]

    p_rms_sq = torch.zeros_like(freqs)
    sigma = torch.zeros_like(freqs)
    for idx in range(1, freqs.numel()):
        f = float(freqs[idx])
        if f <= 0 or float(v_amp[idx]) <= 0.0:
            continue
        sigma[idx] = underwater_sigma_rayleigh(
            cfg.panel_length,
            cfg.panel_width,
            f,
            m=cfg.panel_mode_m,
            n=cfg.panel_mode_n,
            n_points=cfg.rayleigh_n_points,
        )
        # Far-field pressure amplitude at the reference distance (theta=0).
        p = baffled_far_field_pressure(
            cfg.panel_length,
            cfg.panel_width,
            f,
            cfg.ref_distance,
            0.0,
            rho=WATER_RHO,
            c=WATER_C,
            m=cfg.panel_mode_m,
            n=cfg.panel_mode_n,
            velocity_amplitude=float(v_amp[idx]),
            n_points=cfg.rayleigh_n_points,
        )
        p_rms = abs(p) / math.sqrt(2.0)
        p_rms_sq[idx] = p_rms ** 2

    total_p_rms = math.sqrt(float(p_rms_sq.sum()))
    lp_total = noise_level_db(total_p_rms)

    return {
        "ice_force": ice_force,
        "spectrum": spectrum,
        "girder": girder,
        "girder_velocity_midspan": v_mid,
        "panel_velocity_spectrum": v_spec,
        "sigma": sigma,
        "p_rms_sq": p_rms_sq,
        "total_p_rms": total_p_rms,
        "noise_level_db": lp_total,
        "added_mass_per_length": m_a,
    }
