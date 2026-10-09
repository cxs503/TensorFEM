"""Euler--Bernoulli beam vibration analysis implemented with PyTorch.

The module is deliberately self-contained so it can also serve as a numerical
reference for future frame elements.  Each node has transverse displacement
and rotation degrees of freedom.
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


def cantilever_beam_modes(
    length: float, elements: int, EI: float, rho_a: float, modes: int = 6
) -> ModalResult:
    """Convenience solver for a beam clamped at x=0."""
    K, M = uniform_beam_matrices(length, elements, EI, rho_a)
    return solve_modes(K, M, (0, 1), modes)


def cantilever_exact_angular_frequency(length: float, EI: float, rho_a: float, mode: int = 1) -> float:
    """Analytical Euler--Bernoulli cantilever frequency for modes 1--4."""
    beta_l = (1.875104068711961, 4.694091132974174, 7.854757438237612, 10.99554073487547)
    if not 1 <= mode <= len(beta_l):
        raise ValueError("analytical roots are provided for modes 1 through 4")
    return beta_l[mode - 1] ** 2 * math.sqrt(EI / (rho_a * length**4))
