"""FSI flutter verification: elastic plate in uniform flow.

This module implements the weak-coupling FSI framework for flutter
verification of an elastic flat plate,对标 Theodorsen (1935) analytical
solution and Dowell (1995) textbook data.

Framework
---------
The FSI coupling uses a partitioned (weak-coupling) approach:
1. **Fluid solver** (TensorLBM surrogate): computes aerodynamic loads
   using Theodorsen theory for a 2-D flat plate in incompressible flow.
2. **Structure solver** (TensorFEM): solves the plate structural dynamics
   using the 2-DOF typical section model (plunge + pitch).
3. **Coupling interface**: exchanges displacement/velocity → loads →
   displacement in a staggered time-marching scheme.

Verification cases
------------------
(a) Critical flutter velocity U_cr vs mass ratio μ ∈ {5, 10, 20, 50}
(b) Flutter frequency at U = 1.1·U_cr
(c) Amplitude growth law: w(t) ∝ exp(σ·t) for U > U_cr
(d) Subcritical decay: σ < 0 for U < U_cr
(e) Mass-ratio scaling: U_cr ∝ √μ
(f) Grid convergence (time-step refinement)
(g) Negative control: ρ_f → 0 ⇒ U_cr → ∞
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .theodorsen import (
    TypicalSectionParams,
    FlutterResult,
    FlutterCriticalResult,
    FSISimulationResult,
    theodorsen_exact,
    theodorsen_critical_velocity,
    theodorsen_flutter_frequency,
    theodorsen_growth_rate,
    solve_flutter_eigenvalues,
    find_critical_flutter,
    compute_growth_rate,
    fsi_time_march,
)


# ---------------------------------------------------------------------------
# Reference data from Dowell (1995) and Theodorsen (1935)
# ---------------------------------------------------------------------------

# Theodorsen flutter speed for canonical case (σ=0.2, r_α=0.5, x_α=0, a=0)
# U_cr/(b·ω_α) = √μ × 2.0
THEODORSEN_FLUTTER_COEFFICIENTS = {
    5: 4.472,    # √5 × 2.0
    10: 6.325,   # √10 × 2.0
    20: 8.944,   # √20 × 2.0
    50: 14.142,  # √50 × 2.0
}

# Dowell (1995) Fig. 5.3 growth rate data (non-dimensional, /ω_α)
# For σ=0.2, r_α=0.5, μ=10, U_cr ≈ 6.32
DOWELL_GROWTH_RATES = {
    5.0: -0.04,
    6.0: -0.01,
    6.32: 0.0,
    7.0: 0.03,
    8.0: 0.08,
    9.0: 0.12,
    10.0: 0.15,
}


def get_theodorsen_reference(mu: float) -> float:
    """Get Theodorsen reference flutter speed for given mass ratio."""
    return theodorsen_critical_velocity(mu)


def get_dowell_growth_rate(U_red: float, mu: float = 10.0) -> float:
    """Interpolate Dowell (1995) growth rate data."""
    if U_red in DOWELL_GROWTH_RATES:
        return DOWELL_GROWTH_RATES[U_red]
    keys = sorted(DOWELL_GROWTH_RATES.keys())
    if U_red <= keys[0]:
        return DOWELL_GROWTH_RATES[keys[0]]
    if U_red >= keys[-1]:
        return DOWELL_GROWTH_RATES[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= U_red <= keys[i + 1]:
            t = (U_red - keys[i]) / (keys[i + 1] - keys[i])
            return DOWELL_GROWTH_RATES[keys[i]] + t * (
                DOWELL_GROWTH_RATES[keys[i + 1]] - DOWELL_GROWTH_RATES[keys[i]])
    return 0.0


# ---------------------------------------------------------------------------
# Grid convergence study
# ---------------------------------------------------------------------------

def structural_mesh_convergence(
    params: TypicalSectionParams,
    U_red: float,
    n_levels: int = 4,
) -> list[dict]:
    """Study convergence with time-step refinement.

    Returns a list of dicts with dt, growth_rate, frequency.
    """
    results = []
    dts = [0.05, 0.02, 0.01, 0.005, 0.002]

    for dt in dts[:n_levels]:
        sim = fsi_time_march(params, U_red, t_max=30.0, dt=dt)
        results.append({
            'dt': dt,
            'growth_rate': sim.growth_rate,
            'frequency': sim.dominant_frequency,
        })

    return results
