"""Hetényi analytical reference solutions for beams/plates on Winkler foundation.

Reference
---------
M. Hetényi, *Beams on Elastic Foundation*, Univ. of Michigan Press, 1946.

For an *infinite* beam on a Winkler foundation the governing equation is

    EI w'''' + k_b w = q

where *k_b* is the foundation coefficient per unit length (N/m²).  Defining

    λ = (k_b / 4EI)^(1/4)          (characteristic wavenumber, 1/m)
    L_c = 1/λ = (4EI / k_b)^(1/4)  (characteristic length, m)

the four Hetényi shape functions are

    A(ξ) = exp(-ξ)·(cos ξ + sin ξ)
    B(ξ) = exp(-ξ)·(cos ξ - sin ξ)
    C(ξ) = exp(-ξ)·cos ξ
    D(ξ) = exp(-ξ)·sin ξ

For a concentrated force *P* at the origin on an infinite beam:

    w(x) = P·λ / (2·k_b) · A(λ·|x|)
    M(x) = P / (4·λ) · B(λ·|x|)

For *n* concentrated loads {P_i at x_i} the solution follows by superposition.

For a *uniformly distributed* load of intensity *q₀* over [a, b]:

    w(x) = q₀/(k_b) · [1 - ½·(A(λ·|x-a|)·sgn(x-a) - A(λ·|x-b|)·sgn(x-b))]

(truncated series form: integrate point-load solution over the loaded interval).

For a *finite* beam Hetényi gives a series solution.  This module provides both
the infinite-beam closed-form and the finite-beam truncated-series.  The series
truncation index *N_terms* controls accuracy; the caller should check convergence.
"""
from __future__ import annotations

import math
import torch


# ---------------------------------------------------------------------------
# Hetényi shape functions
# ---------------------------------------------------------------------------

def hetenyi_A(xi: torch.Tensor) -> torch.Tensor:
    """A(ξ) = exp(-ξ)·(cos ξ + sin ξ)"""
    return torch.exp(-xi) * (torch.cos(xi) + torch.sin(xi))


def hetenyi_B(xi: torch.Tensor) -> torch.Tensor:
    """B(ξ) = exp(-ξ)·(cos ξ - sin ξ)"""
    return torch.exp(-xi) * (torch.cos(xi) - torch.sin(xi))


def hetenyi_C(xi: torch.Tensor) -> torch.Tensor:
    """C(ξ) = exp(-ξ)·cos ξ"""
    return torch.exp(-xi) * torch.cos(xi)


def hetenyi_D(xi: torch.Tensor) -> torch.Tensor:
    """D(ξ) = exp(-ξ)·sin ξ"""
    return torch.exp(-xi) * torch.sin(xi)


# ---------------------------------------------------------------------------
# Infinite-beam solutions
# ---------------------------------------------------------------------------

def infinite_beam_point_load(
    x_eval: torch.Tensor,
    x_load: float,
    P: float,
    lam: float,
    k_b: float,
) -> torch.Tensor:
    """Deflection of an infinite beam under one concentrated load.

    Parameters
    ----------
    x_eval : (N,) tensor of evaluation positions.
    x_load : position of the point load.
    P : magnitude of the point load (positive = downward).
    lam : λ = (k_b / 4EI)^(1/4).
    k_b : foundation coefficient per unit length (N/m²).

    Returns
    -------
    w : (N,) tensor of deflections.
    """
    dtype = x_eval.dtype
    dx = torch.abs(x_eval - x_load)
    xi = lam * dx
    return torch.as_tensor(P * lam / (2.0 * k_b), dtype=dtype) * hetenyi_A(xi)


def infinite_beam_multiple_point_loads(
    x_eval: torch.Tensor,
    loads: list[tuple[float, float]],
    lam: float,
    k_b: float,
) -> torch.Tensor:
    """Superposition of point loads on an infinite beam.

    Parameters
    ----------
    loads : list of (x_position, P_magnitude) tuples.
    """
    w = torch.zeros_like(x_eval)
    for x_p, P in loads:
        w = w + infinite_beam_point_load(x_eval, x_p, P, lam, k_b)
    return w


def infinite_beam_uniform_load(
    x_eval: torch.Tensor,
    q0: float,
    x_start: float,
    x_end: float,
    lam: float,
    k_b: float,
) -> torch.Tensor:
    """Deflection under uniformly distributed load over [x_start, x_end].

    Obtained by integrating the point-load Green's function.
    w(x) = q₀/(2k_b) · [2 - A(λ·|x-a|)·sgn(x-a) + A(λ·|x-b|)·sgn(x-b)]
         ≈ q₀/k_b · [1 - ½·A(λ·|x-a|)·sgn(x-a) - ½·A(λ·|x-b|)·sgn(x-b)]
    """
    dtype = x_eval.dtype
    xa = torch.abs(x_eval - x_start)
    xb = torch.abs(x_eval - x_end)
    sgn_a = torch.sign(x_eval - x_start)
    sgn_b = torch.sign(x_eval - x_end)
    # Handle exact zero sign (at load edges)
    sgn_a = torch.where(x_eval == x_start, torch.zeros_like(sgn_a), sgn_a)
    sgn_b = torch.where(x_eval == x_end, torch.zeros_like(sgn_b), sgn_b)
    term = torch.ones_like(x_eval) - 0.5 * (
        hetenyi_A(lam * xa) * sgn_a - hetenyi_A(lam * xb) * sgn_b
    )
    return torch.as_tensor(q0 / k_b, dtype=dtype) * term


# ---------------------------------------------------------------------------
# Finite-beam series solution (Hetényi)
# ---------------------------------------------------------------------------

def finite_beam_series(
    x_eval: torch.Tensor,
    beam_length: float,
    lam: float,
    k_b: float,
    EI: float,
    loads: list[tuple[float, float]],
    boundary: str = 'free',
    n_terms: int = 50,
) -> torch.Tensor:
    """Truncated-series solution for a finite beam on Winkler foundation.

    Uses a Fourier-sine-series expansion.  For a beam of length L on elastic
    foundation, the deflection under arbitrary loading can be expressed as

        w(x) = Σ_n [ q_n / (EI·(nπ/L)⁴ + k_b) ] · sin(nπx/L)

    where q_n are the Fourier-sine coefficients of the load distribution.

    For concentrated loads P_i at x_i:

        q_n = (2/L) · Σ_i P_i · sin(nπ·x_i/L)

    Parameters
    ----------
    beam_length : total beam length L.
    boundary : ignored here (sine series ≈ simply-supported end conditions;
        for long beams this converges to the infinite-beam solution).
    n_terms : number of Fourier terms (truncation).

    Returns
    -------
    w : (N,) tensor of deflections.
    """
    L = beam_length
    dtype = x_eval.dtype
    w = torch.zeros_like(x_eval)
    for n in range(1, n_terms + 1):
        npi = n * math.pi
        # Fourier coefficient of load
        qn = torch.tensor(0.0, dtype=dtype)
        for x_p, P in loads:
            qn = qn + P * torch.sin(npi * torch.as_tensor(x_p, dtype=dtype) / L)
        qn = 2.0 / L * qn
        # Modal stiffness
        k_n = EI * (npi / L) ** 4 + k_b
        w = w + (qn / k_n) * torch.sin(npi * x_eval / L)
    return w


# ---------------------------------------------------------------------------
# Plate-strip reduction (wide plate → beam equivalence)
# ---------------------------------------------------------------------------

def plate_strip_parameters(
    young: float,
    poisson: float,
    thickness: float,
    foundation_coefficient: float,
    strip_width: float = 1.0,
) -> tuple[float, float, float]:
    """Convert plate properties to equivalent beam-on-foundation parameters.

    For a plate strip of unit width:

        D_plate = E·h³ / (12·(1-ν²))    (flexural rigidity per unit width)
        k_b = k · b                      (foundation force per unit length)
        λ = (k / (4·D_plate))^(1/4)

    Parameters
    ----------
    young : E
    poisson : ν
    thickness : h
    foundation_coefficient : k (N/m³)
    strip_width : width of the strip (m), default 1.

    Returns
    -------
    EI_eff, k_b, lam
    """
    D_plate = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    EI_eff = D_plate * strip_width
    k_b = foundation_coefficient * strip_width
    lam = (foundation_coefficient / (4.0 * D_plate)) ** 0.25
    return EI_eff, k_b, lam
