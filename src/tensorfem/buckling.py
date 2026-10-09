"""Linear eigenvalue buckling for Euler--Bernoulli beam-columns."""

from dataclasses import dataclass
import math

import torch

from .modal import beam_bending_stiffness


@dataclass(frozen=True)
class BucklingResult:
    load_factors: torch.Tensor
    modes: torch.Tensor


def beam_geometric_stiffness_unit(length: float, *, dtype=torch.float64) -> torch.Tensor:
    """Consistent geometric stiffness for unit constant compressive force."""
    L = length
    return torch.tensor(
        [[36.0, 3.0 * L, -36.0, 3.0 * L],
         [3.0 * L, 4.0 * L**2, -3.0 * L, -L**2],
         [-36.0, -3.0 * L, 36.0, -3.0 * L],
         [3.0 * L, -L**2, -3.0 * L, 4.0 * L**2]],
        dtype=dtype,
    ) / (30.0 * L)


def uniform_column_matrices(
    length: float, elements: int, EI: float, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble elastic and unit-load geometric stiffness matrices."""
    if length <= 0 or elements < 1 or EI <= 0:
        raise ValueError("length, elements and EI must be positive")
    ndof = 2 * (elements + 1)
    K = torch.zeros((ndof, ndof), dtype=dtype)
    Kg = torch.zeros_like(K)
    le = length / elements
    ke = beam_bending_stiffness(EI, le, dtype=dtype)
    kge = beam_geometric_stiffness_unit(le, dtype=dtype)
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += ke
        Kg[dofs[:, None], dofs] += kge
    return K, Kg


def solve_linear_buckling(
    stiffness: torch.Tensor,
    geometric_unit: torch.Tensor,
    constrained_dofs: tuple[int, ...] | list[int],
    modes: int = 3,
) -> BucklingResult:
    """Solve ``K phi = P_cr Kg(unit) phi`` for positive critical loads."""
    if stiffness.shape != geometric_unit.shape or stiffness.ndim != 2:
        raise ValueError("elastic and geometric matrices must have equal square shape")
    n = stiffness.shape[0]
    constrained = set(constrained_dofs)
    free = torch.tensor([i for i in range(n) if i not in constrained], device=stiffness.device)
    K = stiffness[free[:, None], free]
    G = geometric_unit[free[:, None], free]
    # K is SPD after sufficient supports. Transform using K and solve the
    # reciprocal symmetric problem K^-1/2 G K^-T/2 phi = (1/P) phi.
    chol = torch.linalg.cholesky(K)
    left = torch.linalg.solve_triangular(chol, G, upper=False)
    A = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    mu, vectors = torch.linalg.eigh((A + A.T) * 0.5)
    positive = mu > torch.finfo(mu.dtype).eps * mu.abs().max().clamp_min(1.0)
    mu, vectors = mu[positive], vectors[:, positive]
    order = torch.argsort(1.0 / mu)
    mu, vectors = mu[order][:modes], vectors[:, order][:, :modes]
    reduced = torch.linalg.solve_triangular(chol.T, vectors, upper=True)
    full = torch.zeros((n, mu.numel()), dtype=stiffness.dtype, device=stiffness.device)
    full[free] = reduced
    return BucklingResult(1.0 / mu, full)


def pinned_pinned_column_buckling(length: float, elements: int, EI: float, modes: int = 3) -> BucklingResult:
    """Solve a pin-ended column; transverse displacement is fixed at both ends."""
    K, Kg = uniform_column_matrices(length, elements, EI)
    return solve_linear_buckling(K, Kg, (0, 2 * elements), modes)


def euler_pinned_critical_load(length: float, EI: float, mode: int = 1) -> float:
    """Analytical Euler load for a pin-ended prismatic column."""
    return (mode * math.pi) ** 2 * EI / length**2
