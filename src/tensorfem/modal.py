"""Euler--Bernoulli beam vibration analysis implemented with PyTorch.

The module is deliberately self-contained so it can also serve as a numerical
reference for future frame elements.  Each node has transverse displacement
and rotation degrees of freedom.

Analytical frequency references
-------------------------------
For a uniform Euler--Bernoulli beam the natural frequencies are

    omega_n = (beta_n L)^2 sqrt(EI / (rho_a L^4))

where ``beta_n L`` are the roots of the characteristic equation determined by
the boundary conditions.  The roots below are taken from R.D. Blevins,
*Formulas for Natural Frequency and Mode Shape*, Van Nostrand Reinhold, 1979,
Table 8-1 (cross-checked against W.T. Thomson & M.D. Dahleh, *Theory of
Vibration with Applications*, 5th ed., Prentice Hall, 1998, Table 9.1-1).

* Cantilever (clamped--free):  ``cos(beta L) cosh(beta L) = -1``
* Simply supported (pinned--pinned):  ``beta_n L = n pi``
* Clamped--clamped:  ``cos(beta L) cosh(beta L) = 1``
* Clamped--pinned:  ``tan(beta L) = tanh(beta L)``
* Free--free:  same roots as clamped--clamped (rigid-body modes at omega=0)
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ModalResult:
    angular_frequencies: torch.Tensor
    frequencies_hz: torch.Tensor
    modes: torch.Tensor


def beam_bending_stiffness(EI: float, length: float, *, dtype=torch.float64) -> torch.Tensor:
    """Return the 4x4 Euler--Bernoulli bending stiffness matrix."""
    L = torch.as_tensor(length, dtype=dtype)
    scale = torch.as_tensor(EI, dtype=dtype) / L**3
    return scale * torch.tensor(
        [[12.0, 6.0 * length, -12.0, 6.0 * length],
         [6.0 * length, 4.0 * length**2, -6.0 * length, 2.0 * length**2],
         [-12.0, -6.0 * length, 12.0, -6.0 * length],
         [6.0 * length, 2.0 * length**2, -6.0 * length, 4.0 * length**2]],
        dtype=dtype,
    )


def beam_consistent_mass(rho_a: float, length: float, *, dtype=torch.float64) -> torch.Tensor:
    """Return the 4x4 consistent bending mass matrix (no rotary inertia)."""
    scale = torch.as_tensor(rho_a * length / 420.0, dtype=dtype)
    return scale * torch.tensor(
        [[156.0, 22.0 * length, 54.0, -13.0 * length],
         [22.0 * length, 4.0 * length**2, 13.0 * length, -3.0 * length**2],
         [54.0, 13.0 * length, 156.0, -22.0 * length],
         [-13.0 * length, -3.0 * length**2, -22.0 * length, 4.0 * length**2]],
        dtype=dtype,
    )


def uniform_beam_matrices(
    length: float, elements: int, EI: float, rho_a: float, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble stiffness and consistent mass matrices for a uniform beam."""
    if length <= 0 or elements < 1 or EI <= 0 or rho_a <= 0:
        raise ValueError("length, elements, EI and rho_a must be positive")
    ndof = 2 * (elements + 1)
    K = torch.zeros((ndof, ndof), dtype=dtype)
    M = torch.zeros_like(K)
    le = length / elements
    ke = beam_bending_stiffness(EI, le, dtype=dtype)
    me = beam_consistent_mass(rho_a, le, dtype=dtype)
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += ke
        M[dofs[:, None], dofs] += me
    return K, M


def solve_modes(
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    constrained_dofs: tuple[int, ...] | list[int],
    modes: int = 6,
) -> ModalResult:
    """Solve ``K phi = omega^2 M phi`` using a symmetric reduction."""
    if stiffness.shape != mass.shape or stiffness.ndim != 2:
        raise ValueError("stiffness and mass must be square matrices of equal size")
    n = stiffness.shape[0]
    constrained = set(constrained_dofs)
    free = torch.tensor([i for i in range(n) if i not in constrained], device=stiffness.device)
    if free.numel() == 0:
        raise ValueError("at least one free degree of freedom is required")
    K = stiffness[free[:, None], free]
    M = mass[free[:, None], free]
    chol = torch.linalg.cholesky(M)
    # A = L^-1 K L^-T is symmetric and has the generalized eigenvalues.
    left = torch.linalg.solve_triangular(chol, K, upper=False)
    A = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    values, vectors = torch.linalg.eigh((A + A.T) * 0.5)
    keep = values > torch.finfo(values.dtype).eps * values.abs().max().clamp_min(1.0)
    values, vectors = values[keep][:modes], vectors[:, keep][:, :modes]
    reduced_modes = torch.linalg.solve_triangular(chol.T, vectors, upper=True)
    full_modes = torch.zeros((n, values.numel()), dtype=stiffness.dtype, device=stiffness.device)
    full_modes[free] = reduced_modes
    omega = torch.sqrt(values)
    return ModalResult(omega, omega / (2.0 * math.pi), full_modes)


# ---------------------------------------------------------------------------
# Convenience solvers for standard boundary conditions
# ---------------------------------------------------------------------------

def cantilever_beam_modes(
    length: float, elements: int, EI: float, rho_a: float, modes: int = 6
) -> ModalResult:
    """Convenience solver for a beam clamped at x=0."""
    K, M = uniform_beam_matrices(length, elements, EI, rho_a)
    return solve_modes(K, M, (0, 1), modes)


def simply_supported_beam_modes(
    length: float, elements: int, EI: float, rho_a: float, modes: int = 6
) -> ModalResult:
    """Convenience solver for a beam pinned at both ends (w=0 at x=0 and x=L)."""
    K, M = uniform_beam_matrices(length, elements, EI, rho_a)
    # Pinned: transverse displacement fixed, rotation free.
    return solve_modes(K, M, (0, 2 * elements), modes)


def clamped_clamped_beam_modes(
    length: float, elements: int, EI: float, rho_a: float, modes: int = 6
) -> ModalResult:
    """Convenience solver for a beam clamped at both ends."""
    K, M = uniform_beam_matrices(length, elements, EI, rho_a)
    return solve_modes(K, M, (0, 1, 2 * elements, 2 * elements + 1), modes)


def clamped_pinned_beam_modes(
    length: float, elements: int, EI: float, rho_a: float, modes: int = 6
) -> ModalResult:
    """Convenience solver for a beam clamped at x=0 and pinned at x=L."""
    K, M = uniform_beam_matrices(length, elements, EI, rho_a)
    return solve_modes(K, M, (0, 1, 2 * elements), modes)


# ---------------------------------------------------------------------------
# Analytical frequency references (Blevins 1979, Thomson & Dahleh 1998)
# ---------------------------------------------------------------------------

#: Cantilever (clamped--free) characteristic roots: cos(beta L) cosh(beta L) = -1.
_CANTILEVER_ROOTS = (
    1.875104068711961,
    4.694091132974174,
    7.854757438237612,
    10.99554073487547,
)

#: Simply supported (pinned--pinned): beta_n L = n pi.
def _ss_root(n: int) -> float:
    return n * math.pi

#: Clamped--clamped characteristic roots: cos(beta L) cosh(beta L) = 1.
_CLAMPED_ROOTS = (
    4.730040744862704,
    7.853204624095838,
    10.99560783800167,
    14.13716549125746,
)

#: Clamped--pinned characteristic roots: tan(beta L) = tanh(beta L).
_CLAMPED_PINNED_ROOTS = (
    3.926602312047917,
    7.068582746011180,
    10.21017612281498,
    13.35177147933869,
)


def _beam_angular_frequency(beta_l: float, EI: float, rho_a: float, length: float) -> float:
    """Compute ``omega = (beta L)^2 sqrt(EI / (rho_a L^4))``."""
    return beta_l ** 2 * math.sqrt(EI / (rho_a * length ** 4))


def cantilever_exact_angular_frequency(
    length: float, EI: float, rho_a: float, mode: int = 1
) -> float:
    """Analytical Euler--Bernoulli cantilever frequency for modes 1--4."""
    if not 1 <= mode <= len(_CANTILEVER_ROOTS):
        raise ValueError("analytical roots are provided for modes 1 through 4")
    return _beam_angular_frequency(_CANTILEVER_ROOTS[mode - 1], EI, rho_a, length)


def simply_supported_exact_angular_frequency(
    length: float, EI: float, rho_a: float, mode: int = 1
) -> float:
    """Analytical Euler--Bernoulli simply-supported beam frequency.

    The roots are exactly ``n pi``, so any mode number is supported.
    """
    if mode < 1:
        raise ValueError("mode must be >= 1")
    return _beam_angular_frequency(_ss_root(mode), EI, rho_a, length)


def clamped_clamped_exact_angular_frequency(
    length: float, EI: float, rho_a: float, mode: int = 1
) -> float:
    """Analytical Euler--Bernoulli clamped--clamped beam frequency for modes 1--4."""
    if not 1 <= mode <= len(_CLAMPED_ROOTS):
        raise ValueError("analytical roots are provided for modes 1 through 4")
    return _beam_angular_frequency(_CLAMPED_ROOTS[mode - 1], EI, rho_a, length)


def clamped_pinned_exact_angular_frequency(
    length: float, EI: float, rho_a: float, mode: int = 1
) -> float:
    """Analytical Euler--Bernoulli clamped--pinned beam frequency for modes 1--4."""
    if not 1 <= mode <= len(_CLAMPED_PINNED_ROOTS):
        raise ValueError("analytical roots are provided for modes 1 through 4")
    return _beam_angular_frequency(_CLAMPED_PINNED_ROOTS[mode - 1], EI, rho_a, length)
