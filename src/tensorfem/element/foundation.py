"""Elastic foundation elements for beams and plates on Winkler foundations.

This module provides:

1. **Euler-Bernoulli beam element** with consistent Winkler foundation stiffness
   k_e = ∫ k(x) NᵀN dx using Hermite cubic shape functions.

2. **Consistent distributed load vector** for beam elements: f_e = ∫ q(x) Nᵀ dx.

3. **Q4 Mindlin plate element** Winkler foundation stiffness (delegates to the
   existing ``winkler_foundation`` module but provides a unified interface).

4. **Global assembly** routines for beam and plate foundation systems.

Theory
------
For an Euler-Bernoulli beam on a Winkler foundation, the governing equation is

    EI w'''' + k(x) w = q(x)

where k(x) is the foundation modulus (force/length²) and q(x) is the
distributed load (force/length).  The weak form gives the element foundation
stiffness

    k_e = ∫₀ᴸ k(x) Nᵀ(x) N(x) dx

where N(x) = [N₁(x), N₂(x), N₃(x), N₄(x)] are the Hermite cubic shape
functions and the DOF vector is [w₁, θ₁, w₂, θ₂]ᵀ.

For a uniform foundation k(x) = k₀, the integral can be evaluated exactly:

    k_e = k₀ L/420 × [156   22L    54    -13L ]
                       [22L   4L²    13L   -3L² ]
                       [54    13L    156   -22L ]
                       [-13L  -3L²   -22L   4L² ]

For a plate on a Winkler foundation, the foundation pressure is p = k·w and
the element stiffness is

    k_e = k ∫∫ N_wᵀ N_w dA

where N_w are the shape functions associated with the transverse displacement
DOF at each node.

References
----------
[1] Hetényi, M., *Beams on Elastic Foundation*, Univ. of Michigan Press, 1946.
[2] Cook, R.D. et al., *Concepts and Applications of Finite Element Analysis*,
    4th ed., Wiley, 2001.
[3] Zienkiewicz, O.C. & Taylor, R.L., *The Finite Element Method*, 5th ed.,
    Butterworth-Heinemann, 2000.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hermite cubic shape functions for Euler-Bernoulli beam
# ---------------------------------------------------------------------------

def hermite_shape_functions(
    xi: torch.Tensor,
    L: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Hermite cubic shape functions and their derivatives w.r.t. x.

    The shape functions map the physical coordinate x ∈ [0, L] to the DOF
    vector [w₁, θ₁, w₂, θ₂]ᵀ where θ = dw/dx.

    Parameters
    ----------
    xi : scalar tensor, natural coordinate in [-1, 1].
    L : scalar tensor, element length.

    Returns
    -------
    N : (4,) tensor of shape function values.
    dNdx : (4,) tensor of shape function derivatives w.r.t. physical x.
    """
    # Map from natural coordinate xi ∈ [-1,1] to physical x ∈ [0, L]
    # x = L/2 * (1 + xi), dx/dxi = L/2
    t = 0.5 * (1.0 + xi)  # t ∈ [0, 1], t = x/L

    # Hermite cubics in terms of t = x/L
    N1 = 1.0 - 3.0 * t**2 + 2.0 * t**3
    N2 = L * (t - 2.0 * t**2 + t**3)
    N3 = 3.0 * t**2 - 2.0 * t**3
    N4 = L * (-t**2 + t**3)

    N = torch.stack([N1, N2, N3, N4])

    # Derivatives w.r.t. x: dN/dx = dN/dt * dt/dx = dN/dt / L
    dN1_dt = -6.0 * t + 6.0 * t**2
    dN2_dt = L * (1.0 - 4.0 * t + 3.0 * t**2)
    dN3_dt = 6.0 * t - 6.0 * t**2
    dN4_dt = L * (-2.0 * t + 3.0 * t**2)

    dNdx = torch.stack([dN1_dt, dN2_dt, dN3_dt, dN4_dt]) / L

    return N, dNdx


def euler_bernoulli_bending_stiffness(
    L: float,
    EI: float,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the 4×4 Euler-Bernoulli beam bending stiffness matrix.

    K_b = EI/L³ × [ 12    6L   -12    6L ]
                   [ 6L    4L²  -6L    2L²]
                   [-12   -6L    12   -6L ]
                   [ 6L    2L²  -6L    4L²]

    Parameters
    ----------
    L : element length.
    EI : flexural rigidity.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")
    if EI <= 0.0:
        raise ValueError("EI must be positive")

    c = EI / L**3
    K = torch.tensor([
        [12.0,   6.0*L,  -12.0,  6.0*L],
        [6.0*L,  4.0*L**2, -6.0*L, 2.0*L**2],
        [-12.0, -6.0*L,   12.0, -6.0*L],
        [6.0*L,  2.0*L**2, -6.0*L, 4.0*L**2],
    ], dtype=dtype, device=device)
    return c * K


def euler_bernoulli_foundation_stiffness(
    L: float,
    k: float,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the 4×4 consistent Winkler foundation stiffness for a beam element.

    k_e = ∫₀ᴸ k NᵀN dx

    For uniform k, the exact result is:

    k_e = k·L/420 × [ 156    22L     54    -13L  ]
                     [ 22L    4L²     13L   -3L²  ]
                     [ 54     13L     156   -22L  ]
                     [-13L   -3L²    -22L    4L²  ]

    Parameters
    ----------
    L : element length.
    k : Winkler foundation modulus (force/length², e.g. N/m² for beam).

    Returns
    -------
    ke : (4, 4) tensor.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")
    if k < 0.0:
        raise ValueError("k must be non-negative")

    if k == 0.0:
        return torch.zeros((4, 4), dtype=dtype, device=device)

    c = k * L / 420.0
    ke = torch.tensor([
        [156.0,    22.0*L,    54.0,    -13.0*L],
        [22.0*L,   4.0*L**2,  13.0*L,  -3.0*L**2],
        [54.0,     13.0*L,    156.0,   -22.0*L],
        [-13.0*L,  -3.0*L**2, -22.0*L, 4.0*L**2],
    ], dtype=dtype, device=device)
    return c * ke


def _gauss_legendre(n: int, dtype: torch.dtype = torch.float64,
                    device: Optional[torch.device] = None):
    """Return Gauss-Legendre quadrature points and weights on [-1, 1]."""
    xi_np, w_np = np.polynomial.legendre.leggauss(n)
    xi = torch.tensor(xi_np, dtype=dtype, device=device)
    w = torch.tensor(w_np, dtype=dtype, device=device)
    return xi, w


def euler_bernoulli_foundation_stiffness_general(
    L: float,
    k_func: Callable[[torch.Tensor], torch.Tensor],
    n_gauss: int = 4,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Foundation stiffness for spatially varying k(x) via Gauss quadrature.

    k_e = ∫₀ᴸ k(x) Nᵀ(x) N(x) dx

    Parameters
    ----------
    L : element length.
    k_func : callable k(x) → scalar, foundation modulus as function of position.
    n_gauss : number of Gauss points (default 4, sufficient for cubic×cubic×smooth).

    Returns
    -------
    ke : (4, 4) tensor.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")

    xi_gp, w_gp = _gauss_legendre(n_gauss, dtype, device)

    L_t = torch.tensor(L, dtype=dtype, device=device)
    ke = torch.zeros((4, 4), dtype=dtype, device=device)

    for i in range(n_gauss):
        xi = xi_gp[i]
        w = w_gp[i]
        # Map to physical coordinate
        x = 0.5 * L_t * (1.0 + xi)
        dx_dxi = 0.5 * L_t
        k_val = k_func(x)
        N, _ = hermite_shape_functions(xi, L_t)
        ke = ke + k_val * torch.outer(N, N) * w * dx_dxi

    return ke


def euler_bernoulli_foundation_load(
    L: float,
    q_func: Callable[[torch.Tensor], torch.Tensor],
    n_gauss: int = 4,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Consistent nodal forces from distributed load q(x) on a beam element.

    f_e = ∫₀ᴸ q(x) Nᵀ(x) dx

    Parameters
    ----------
    L : element length.
    q_func : callable q(x) → scalar, distributed load (force/length).
    n_gauss : number of Gauss points.

    Returns
    -------
    fe : (4,) tensor of consistent nodal forces [F1, M1, F2, M2]ᵀ.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")

    xi_gp, w_gp = _gauss_legendre(n_gauss, dtype, device)

    L_t = torch.tensor(L, dtype=dtype, device=device)
    fe = torch.zeros(4, dtype=dtype, device=device)

    for i in range(n_gauss):
        xi = xi_gp[i]
        w = w_gp[i]
        x = 0.5 * L_t * (1.0 + xi)
        dx_dxi = 0.5 * L_t
        q_val = q_func(x)
        N, _ = hermite_shape_functions(xi, L_t)
        fe = fe + q_val * N * w * dx_dxi

    return fe


def euler_bernoulli_uniform_load(
    L: float,
    q0: float,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Consistent nodal forces for uniform load q₀ on a beam element.

    f_e = q₀·L × [1/2, L/12, 1/2, -L/12]ᵀ  (exact for constant q)

    Parameters
    ----------
    L : element length.
    q0 : uniform load intensity (force/length).

    Returns
    -------
    fe : (4,) tensor.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")
    fe = torch.tensor([
        q0 * L / 2.0,
        q0 * L**2 / 12.0,
        q0 * L / 2.0,
        -q0 * L**2 / 12.0,
    ], dtype=dtype, device=device)
    return fe


def assemble_beam_foundation(
    n_elements: int,
    element_length: float,
    foundation_k: float,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Assemble global foundation stiffness for a uniform beam mesh.

    The beam has n_elements elements of equal length, with 2 DOFs per node
    (w, θ). Total DOFs = 2*(n_elements+1).

    Parameters
    ----------
    n_elements : number of beam elements.
    element_length : length of each element.
    foundation_k : Winkler foundation modulus (force/length²).

    Returns
    -------
    K : (ndof, ndof) global foundation stiffness matrix.
    """
    ndof = 2 * (n_elements + 1)
    K = torch.zeros((ndof, ndof), dtype=dtype, device=device)
    ke = euler_bernoulli_foundation_stiffness(element_length, foundation_k, dtype, device)

    for e in range(n_elements):
        # DOF mapping: element e connects nodes e and e+1
        dof_map = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += ke[i, j]

    return K


# ---------------------------------------------------------------------------
# Plate (Q4 Mindlin) Winkler foundation
# ---------------------------------------------------------------------------

def plate_winkler_foundation_stiffness(
    xy: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Return the 12×12 Winkler foundation stiffness for a Q4 Mindlin plate element.

    This is a thin wrapper around the existing ``winkler_foundation`` module
    providing a unified interface under the ``element`` sub-package.

    Parameters
    ----------
    xy : (4, 2) tensor of nodal coordinates (CCW order).
    k : Winkler foundation modulus (force/length³, e.g. N/m³).

    Returns
    -------
    ke : (12, 12) tensor.
    """
    # Import here to avoid circular dependency
    from ..winkler_foundation import q4_winkler_foundation_stiffness
    return q4_winkler_foundation_stiffness(xy, k)


def assemble_plate_winkler_foundation(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    foundation_k: float,
) -> torch.Tensor:
    """Assemble global Winkler foundation stiffness for a plate mesh.

    Parameters
    ----------
    nodes : (N, 2) or (N, 3) tensor of nodal coordinates.
    elements : (E, 4) tensor of element connectivity.
    foundation_k : Winkler modulus (force/length³).

    Returns
    -------
    K : (3*N, 3*N) global foundation stiffness.
    """
    from ..winkler_foundation import assemble_winkler_foundation
    return assemble_winkler_foundation(nodes, elements, foundation_k)
