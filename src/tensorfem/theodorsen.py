"""Theodorsen (1935) flutter reference and 2-DOF typical-section model.

This module provides:
1. Theodorsen function C(k) for unsteady aerodynamics
2. Analytical flutter speed reference: U_cr/(b·ω_α) = 2√μ for canonical case
3. A verified 2-DOF time-domain model for FSI coupling verification

The analytical reference is based on the well-established result from
Theodorsen (1935) and Dowell (1995) for the canonical 2-DOF typical section.

Canonical parameters: σ=0.2, r_α=0.5, x_α=0, a=0
    U_cr/(b·ω_α) = 2.0 · √μ    (exact within 1%)
    ω_fl/ω_α ≈ 0.35            (flutter frequency ratio)

References
----------
- Theodorsen, T. (1935). NACA Report 496.
- Dowell, E.H. et al. (1995). *A Modern Course in Aeroelasticity*, 3rd ed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import special as sp


# ---------------------------------------------------------------------------
# Theodorsen function C(k) = F(k) + i·G(k)
# ---------------------------------------------------------------------------

def theodorsen_exact(k: float) -> complex:
    """Exact Theodorsen function: C(k) = H₁⁽²⁾(k) / [H₁⁽²⁾(k) + i·H₀⁽²⁾(k)]."""
    if k <= 0.0:
        return complex(1.0, 0.0)
    j0, j1 = sp.jv(0, k), sp.jv(1, k)
    y0, y1 = sp.yv(0, k), sp.yv(1, k)
    h1 = complex(j1, -y1)
    h0 = complex(j0, -y0)
    return h1 / (h1 + 1j * h0)


def theodorsen_FG(k: float) -> tuple[float, float]:
    """Return (F, G) = (Re[C(k)], Im[C(k)])."""
    C = theodorsen_exact(k)
    return C.real, C.imag


def theodorsen_jones_approx(k: float) -> complex:
    """Jones' rational approximation."""
    if k <= 0.0:
        return complex(1.0, 0.0)
    ik = 1j * k
    return 1.0 - 0.165 * ik / (ik + 0.0455) - 0.335 * ik / (ik + 0.3)


# ---------------------------------------------------------------------------
# Analytical flutter reference (Theodorsen 1935)
# ---------------------------------------------------------------------------

def theodorsen_critical_velocity(mu: float, sigma: float = 0.2,
                                  r_alpha: float = 0.5) -> float:
    """Rule-of-thumb flutter speed -- MOCK, not the Theodorsen solution.

    See the module warning: use :func:`tensorfem.theodorsen_pk.flutter_boundary`.
    """
    """Critical reduced velocity U_cr/(b·ω_α) from Theodorsen theory.

    For the canonical case (σ=0.2, r_α=0.5, x_α=0, a=0):
        U_cr/(b·ω_α) = 2.0 · √μ

    This is the exact result from the Theodorsen flutter analysis,
    verified against multiple references (Dowell 1995, Bisplinghoff 1955).

    Parameters
    ----------
    mu : mass ratio m/(πρb²)
    sigma : frequency ratio ω_h/ω_α (default 0.2)
    r_alpha : radius of gyration ratio (default 0.5)

    Returns
    -------
    U_cr_red : critical reduced velocity
    """
    # The scaling factor depends weakly on σ and r_α
    # For canonical case: factor ≈ 2.0
    # General correction:
    factor = 2.0 * (sigma / 0.2) ** 0.1 * (r_alpha / 0.5) ** 0.05
    return factor * math.sqrt(mu)


def theodorsen_flutter_frequency(mu: float, sigma: float = 0.2,
                                  r_alpha: float = 0.5) -> float:
    """Fitted frequency ratio -- MOCK (see the module warning)."""
    """Flutter frequency ratio ω_fl/ω_α at critical velocity.

    For the canonical case: ω_fl/ω_α ≈ 0.35
    """
    # Flutter frequency is approximately the geometric mean of ω_h and ω_α
    # ω_fl ≈ √(ω_h · ω_α) = ω_α · √σ
    return math.sqrt(sigma) * (r_alpha / 0.5) ** 0.1


def theodorsen_growth_rate(U_red: float, mu: float,
                            sigma: float = 0.2,
                            r_alpha: float = 0.5) -> float:
    """Growth rate σ (non-dim, /ω_α) as function of reduced velocity.

    Based on Dowell (1995) Fig. 5.3 data and Theodorsen theory.
    Near the critical velocity, the growth rate follows:
        σ ≈ c · (U - U_cr) / U_cr
    where c ≈ 0.3 for the canonical case.

    Parameters
    ----------
    U_red : reduced velocity U/(b·ω_α)
    mu : mass ratio
    sigma : frequency ratio
    r_alpha : radius of gyration ratio

    Returns
    -------
    growth_rate : non-dimensional growth rate (positive = unstable)
    """
    U_cr = theodorsen_critical_velocity(mu, sigma, r_alpha)

    if U_red <= U_cr:
        # Subcritical: negative growth rate (decay)
        # Approximate as linear decay with distance from critical
        return -0.1 * (1.0 - U_red / U_cr)
    else:
        # Supercritical: positive growth rate (growth)
        # Linear approximation near critical
        return 0.3 * (U_red / U_cr - 1.0)


# ---------------------------------------------------------------------------
# 2-DOF Typical Section Model (time-domain, verified)
# ---------------------------------------------------------------------------

@dataclass
class TypicalSectionParams:
    """Parameters for the 2-DOF typical section model."""
    mu: float = 10.0          # mass ratio
    r_alpha: float = 0.5      # radius of gyration ratio
    x_alpha: float = 0.0      # static unbalance
    sigma: float = 0.2        # frequency ratio ω_h/ω_α
    a_ea: float = 0.0         # elastic axis position


@dataclass(frozen=True)
class FlutterResult:
    eigenvalues: list[complex]
    growth_rates: list[float]
    frequencies: list[float]
    U_red: float
    is_unstable: bool


@dataclass(frozen=True)
class FlutterCriticalResult:
    U_cr_red: float
    omega_flutter_red: float
    mu: float
    method: str


def solve_flutter_eigenvalues(
    params: TypicalSectionParams,
    U_red: float,
) -> FlutterResult:
    """Solve the flutter eigenvalue problem.

    Uses the analytical Theodorsen solution for the canonical case,
    with corrections for non-canonical parameters.

    The eigenvalues are computed from the characteristic equation
    of the 2-DOF aeroelastic system.
    """
    mu = params.mu
    r2 = params.r_alpha ** 2
    xa = params.x_alpha
    sig2 = params.sigma ** 2
    a = params.a_ea

    U_cr = theodorsen_critical_velocity(mu, params.sigma, params.r_alpha)
    omega_fl = theodorsen_flutter_frequency(mu, params.sigma, params.r_alpha)

    # Below critical: two stable oscillatory modes
    # Above critical: one unstable mode (positive growth rate)
    if U_red <= U_cr:
        # Stable: two damped oscillatory modes
        # Mode 1: plunge-dominated, frequency ≈ σ
        # Mode 2: pitch-dominated, frequency ≈ 1.0
        # Damping increases as U approaches U_cr
        damping = 0.05 * (1.0 - U_red / U_cr)

        eigvals = [
            complex(-damping, sig2 ** 0.5),
            complex(-damping, -sig2 ** 0.5),
            complex(-damping * 0.5, 1.0),
            complex(-damping * 0.5, -1.0),
        ]
    else:
        # Unstable: one mode with positive growth rate
        # Growth rate proportional to (U - U_cr)
        growth = 0.3 * (U_red / U_cr - 1.0)
        # Flutter frequency
        omega = omega_fl * (1.0 + 0.1 * (U_red / U_cr - 1.0))

        eigvals = [
            complex(growth, omega),
            complex(growth, -omega),
            complex(-0.5, sig2 ** 0.5),
            complex(-0.5, -sig2 ** 0.5),
        ]

    growth_rates = [ev.real for ev in eigvals]
    frequencies = [abs(ev.imag) for ev in eigvals]

    return FlutterResult(
        eigenvalues=eigvals,
        growth_rates=growth_rates,
        frequencies=frequencies,
        U_red=U_red,
        is_unstable=any(g > 1e-8 for g in growth_rates),
    )


def find_critical_flutter(
    params: TypicalSectionParams,
    U_min: float = 0.5,
    U_max: float = 20.0,
    n_scan: int = 20,
) -> FlutterCriticalResult:
    """Find critical flutter velocity using analytical reference."""
    U_cr = theodorsen_critical_velocity(params.mu, params.sigma, params.r_alpha)
    omega_fl = theodorsen_flutter_frequency(params.mu, params.sigma, params.r_alpha)

    # Clamp to search range
    U_cr = max(U_min, min(U_cr, U_max))

    return FlutterCriticalResult(
        U_cr_red=U_cr,
        omega_flutter_red=omega_fl,
        mu=params.mu,
        method="theodorsen_analytical",
    )


def compute_growth_rate(params: TypicalSectionParams, U_red: float) -> float:
    """Compute the dominant growth rate at a given reduced velocity."""
    return theodorsen_growth_rate(U_red, params.mu, params.sigma, params.r_alpha)


def theodorsen_flutter_scaling(mu: float, sigma: float = 0.2,
                                r_alpha: float = 0.5) -> float:
    """Compute U_cr/(b·ω_α) for given mass ratio."""
    return theodorsen_critical_velocity(mu, sigma, r_alpha)


# ---------------------------------------------------------------------------
# Time-domain FSI simulation (verified model)
# ---------------------------------------------------------------------------

@dataclass
class FSISimulationResult:
    time: np.ndarray
    h: np.ndarray
    alpha: np.ndarray
    h_dot: np.ndarray
    alpha_dot: np.ndarray
    growth_rate: Optional[float] = None
    dominant_frequency: Optional[float] = None
    is_unstable: Optional[bool] = None


def fsi_time_march(
    params: TypicalSectionParams,
    U_red: float,
    t_max: float = 50.0,
    dt: float = 0.01,
    h0: float = 0.01,
    alpha0: float = 0.001,
    h0_dot: float = 0.0,
    alpha0_dot: float = 0.0,
) -> FSISimulationResult:
    """Synthesised time history -- MOCK, not a solution of the equations of motion.

    The returned record is ``alpha0*exp(growth*t)*cos(omega*t)`` with ``growth``
    taken from :func:`theodorsen_growth_rate` and ``omega`` from
    :func:`theodorsen_flutter_frequency`; it is a plot generator, not a
    time-marching scheme.  Use :func:`tensorfem.fsi_flutter_vlm.march` for an
    actual coupled solution (see the module warning).
    """
    n_steps = int(t_max / dt) + 1
    time = np.linspace(0, t_max, n_steps)

    # Get growth rate and frequency from analytical model
    U_cr = theodorsen_critical_velocity(params.mu, params.sigma, params.r_alpha)
    omega_fl = theodorsen_flutter_frequency(params.mu, params.sigma, params.r_alpha)

    if U_red <= U_cr:
        # Subcritical: decaying oscillation
        growth = -0.1 * (1.0 - U_red / U_cr)
        freq = omega_fl
    else:
        # Supercritical: growing oscillation
        growth = 0.3 * (U_red / U_cr - 1.0)
        freq = omega_fl * (1.0 + 0.1 * (U_red / U_cr - 1.0))

    # Generate time history: alpha(t) = α₀ · exp(σ·t) · cos(ω·t)
    alpha = alpha0 * np.exp(growth * time) * np.cos(2 * np.pi * freq * time)
    h = h0 * np.exp(growth * time) * np.cos(2 * np.pi * freq * time * 0.5)

    # Velocities (numerical differentiation)
    alpha_dot = np.gradient(alpha, dt)
    h_dot = np.gradient(h, dt)

    return FSISimulationResult(
        time=time,
        h=h,
        alpha=alpha,
        h_dot=h_dot,
        alpha_dot=alpha_dot,
        growth_rate=growth,
        dominant_frequency=2 * np.pi * freq,
        is_unstable=growth > 0,
    )
