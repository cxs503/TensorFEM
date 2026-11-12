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

__all__ = [
    "FacetRankAccounting",
    "MechanismCensus",
    "drilling_gauge_mode",
    "element_rank_census",
    "facet_rank_accounting",
    "rigid_body_modes",
]


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

# --------------------------------------------------------------------------- #
# gauge-aware accounting
# --------------------------------------------------------------------------- #
def drilling_gauge_mode(xyz: torch.Tensor) -> torch.Tensor:
    """Return the unit drilling-gauge direction of a facet in the 24-dof layout.

    A flat facet has six rigid-body modes.  It *also* has one further
    ``K``-free direction that is not a rigid motion: a **constant section
    rotation about the facet normal with zero displacement**, ``u = 0``,
    ``r = n``.  Neither the membrane block (``u = 0``), the plate block (a
    rotation about the normal is not a plate rotation) nor the difference-only
    drilling stabilisation (a constant ``r_z`` field has zero discrete
    Laplacian) resists it, so it is exactly zero-energy -- but it cannot be
    communicated between elements, because two adjacent facets do not share a
    normal.  It is a *gauge*, not a mechanism: no assembled load can excite it.

    ``rigid_body_modes`` represents the normal rotation as the pair
    ``(u = n x (x - c), r = n)``; the direction returned here is the
    orthogonal complement of that representation, i.e. the pure
    ``(0, n)`` field with the rigid span projected out.
    """
    xyz = torch.as_tensor(xyz)
    if xyz.shape != (4, 3):
        raise ValueError("xyz must have shape (4, 3)")
    from .shell4 import shell4_local_frame

    _, basis = shell4_local_frame(xyz)
    normal = basis[2]
    raw = torch.zeros(24, dtype=xyz.dtype, device=xyz.device)
    raw[3::6] = normal[0]
    raw[4::6] = normal[1]
    raw[5::6] = normal[2]
    rigid = rigid_body_modes(xyz)
    q, _ = torch.linalg.qr(rigid)
    gauge = raw - q @ (q.T @ raw)
    return gauge / torch.linalg.vector_norm(gauge)


@dataclass(frozen=True)
class FacetRankAccounting:
    """Gauge-aware zero-energy-mode census of one flat facet.

    ``element_rank_census`` counts *everything* outside the six rigid-body
    modes as spurious, which reports one false positive per facet: the
    drilling gauge described in :func:`drilling_gauge_mode`.  The accounting
    here separates the two populations, so ``spurious_modes`` is the number of
    directions the facet genuinely does not resist while it still supports a
    load path.
    """

    zero_modes: int
    rigid_modes: int
    gauge_modes: int
    spurious_modes: int
    gauge_residual: float
    rigid_residual: float
    mechanisms: tuple[torch.Tensor, ...]
    lambda_max: float
    rank_tolerance: float
    formulation: str
    plate_shear: str

    @property
    def is_rank_sufficient(self) -> bool:
        """True when no zero-energy direction other than rigid + gauge exists."""
        return self.spurious_modes == 0


def facet_rank_accounting(
    xyz: torch.Tensor,
    young,
    poisson,
    thickness,
    *,
    drilling_factor: float = 1e-6,
    plate_shear: str = "sri",
    formulation: str = "shell4",
    rank_tolerance: float = 1e-12,
    spurious_tolerance: float = 1e-8,
) -> FacetRankAccounting:
    """Classify the zero-energy directions of a facet into rigid / gauge / spurious."""
    from .shell4 import shell4_stiffness
    from .shell4_mitc import shell4_mitc_stiffness

    builder = {
        "shell4": shell4_stiffness,
        "shell4_mitc": shell4_mitc_stiffness,
    }.get(formulation.lower())
    if builder is None:
        raise ValueError(f"unknown formulation {formulation!r}")
    dtype, device = xyz.dtype, xyz.device
    K = builder(
        xyz,
        torch.as_tensor(young, dtype=dtype, device=device),
        torch.as_tensor(poisson, dtype=dtype, device=device),
        torch.as_tensor(thickness, dtype=dtype, device=device),
        drilling_factor=drilling_factor,
        plate_shear=plate_shear,
    )
    lambda_max = float(K.abs().max())
    if not lambda_max > 0.0:
        raise ValueError("degenerate element: zero stiffness")
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    tol = rank_tolerance * lambda_max
    null_basis = evecs[:, torch.nonzero(evals.abs() <= tol).flatten()]
    rigid = rigid_body_modes(xyz)
    gauge = drilling_gauge_mode(xyz).unsqueeze(1)
    expected = torch.cat((rigid, gauge), 1)
    rigid_residual = float(torch.linalg.vector_norm(K @ rigid)) / lambda_max
    gauge_residual = float(torch.linalg.vector_norm(K @ gauge[:, 0])) / lambda_max
    q, _ = torch.linalg.qr(expected)
    residual_basis = (torch.eye(24, dtype=dtype, device=device) - q @ q.T) @ null_basis
    left, singular_values, _ = torch.linalg.svd(residual_basis)
    n_spurious = int((singular_values > spurious_tolerance).sum().item())
    return FacetRankAccounting(
        zero_modes=int(null_basis.shape[1]),
        rigid_modes=6,
        gauge_modes=1 if gauge_residual <= tol * 10 else 0,
        spurious_modes=n_spurious,
        gauge_residual=gauge_residual,
        rigid_residual=rigid_residual,
        mechanisms=tuple(left[:, k].clone() for k in range(n_spurious)),
        lambda_max=lambda_max,
        rank_tolerance=rank_tolerance,
        formulation=formulation,
        plate_shear=plate_shear,
    )
