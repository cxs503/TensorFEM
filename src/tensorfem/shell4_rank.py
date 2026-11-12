"""Zero-energy-mode census for the flat four-node shell facet.

A four-node shell element must have exactly six zero-energy modes: the three
rigid translations and the three rigid rotations.  Anything beyond that is a
spurious mechanism (hourglass mode) -- a direction the element does not resist,
so an assembled mesh can deform along it for free and the discrete solution can
converge to something that is not the solution of the shell equations.

The census here separates the two kinds by *dimension*, not eigenvector by
eigenvector: ``eigh`` returns an arbitrary orthonormal basis of the null space,
so individual null vectors are generically mixtures of rigid-body and spurious
modes.  The six rigid-body modes are verified to lie in the null space, then the
null space is projected onto the orthogonal complement of their span; the rank
of that projection is the number of spurious mechanisms and its left singular
vectors are the mechanism shapes.

Only element-level linear algebra is involved, so the census is cheap enough to
run as a guard on every formulation change.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .shell4 import shell4_stiffness

__all__ = ["MechanismCensus", "element_rank_census", "rigid_body_modes"]


@dataclass(frozen=True)
class MechanismCensus:
    """Result of :func:`element_rank_census`."""

    zero_modes: int
    spurious_modes: int
    mechanisms: tuple[torch.Tensor, ...]
    singular_values: torch.Tensor
    rigid_basis_residual: float
    lambda_max: float
    rank_tolerance: float
    first_elastic_eigenvalues: torch.Tensor

    @property
    def is_rank_sufficient(self) -> bool:
        """True when the only zero-energy modes are the six rigid-body ones."""
        return self.spurious_modes == 0


def rigid_body_modes(xyz: torch.Tensor) -> torch.Tensor:
    """Return the six exact rigid-body modes of a facet in the 24-dof layout.

    ``xyz`` is ``(4, 3)``; the dof layout is ``(u_x, u_y, u_z, r_x, r_y, r_z)``
    per node, matching :func:`tensorfem.shell4.shell4_stiffness`.  A rigid
    rotation by ``omega`` displaces ``u = omega x (x - x_c)`` and rotates every
    section by ``omega``.
    """
    if xyz.shape != (4, 3):
        raise ValueError("xyz must have shape (4, 3)")
    dtype, device = xyz.dtype, xyz.device
    basis = torch.zeros((24, 6), dtype=dtype, device=device)
    for k in range(3):  # translations
        basis[k::6, k] = 1.0
    centroid = xyz.mean(0)
    for k in range(3):  # rotations
        axis = torch.zeros(3, dtype=dtype, device=device)
        axis[k] = 1.0
        for node in range(4):
            basis[6 * node:6 * node + 3, 3 + k] = torch.linalg.cross(axis, xyz[node] - centroid)
            basis[6 * node + 3:6 * node + 6, 3 + k] = axis
    return basis


def element_rank_census(
    xyz: torch.Tensor,
    young,
    poisson,
    thickness,
    *,
    plate_shear: str = "sri",
    drilling_factor: float = 1e-6,
    rank_tolerance: float = 1e-12,
    spurious_tolerance: float = 1e-6,
) -> MechanismCensus:
    """Count zero-energy modes of one facet and split them rigid / spurious.

    ``rank_tolerance`` is *relative to* ``lambda_max`` and defaults to ``1e-12``
    -- just above the eigendecomposition noise floor.  It must not be read as a
    physical stiffness threshold: for a thin facet the smallest elastic
    eigenvalue can sit many orders of magnitude below ``lambda_max``.

    ``spurious_tolerance`` is the singular-value cut that separates directions
    inside the orthogonal complement of the rigid-body span (spurious, singular
    value ~1) from directions inside it (rigid, singular value ~0).
    """
    if xyz.shape != (4, 3):
        raise ValueError("xyz must have shape (4, 3)")
    dtype = xyz.dtype
    K = shell4_stiffness(
        xyz,
        young,
        poisson,
        thickness,
        drilling_factor=drilling_factor,
        plate_shear=plate_shear,
    )
    lambda_max = float(K.abs().max())
    if not lambda_max > 0.0:
        raise ValueError("degenerate element: zero stiffness")
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    tol = rank_tolerance * lambda_max
    null_idx = torch.nonzero(evals.abs() <= tol).flatten()
    if null_idx.numel() < 6:
        raise RuntimeError(
            f"null space of dimension {null_idx.numel()} is smaller than the six "
            "rigid-body modes: the element is not rigid-body reproducing"
        )
    null_basis = evecs[:, null_idx]

    rigid = rigid_body_modes(xyz)
    rigid_residual = float(torch.linalg.vector_norm(K @ rigid)) / lambda_max
    q_rigid, _ = torch.linalg.qr(rigid)
    projector = torch.eye(24, dtype=dtype, device=xyz.device) - q_rigid @ q_rigid.T
    residual_basis = projector @ null_basis
    left, singular_values, _ = torch.linalg.svd(residual_basis)
    n_spurious = int((singular_values > spurious_tolerance).sum().item())
    mechanisms = tuple(left[:, k].clone() for k in range(n_spurious))
    return MechanismCensus(
        zero_modes=int(null_idx.numel()),
        spurious_modes=n_spurious,
        mechanisms=mechanisms,
        singular_values=singular_values,
        rigid_basis_residual=rigid_residual,
        lambda_max=lambda_max,
        rank_tolerance=tol,
        first_elastic_eigenvalues=evals[null_idx.numel(): null_idx.numel() + 4].clone(),
    )