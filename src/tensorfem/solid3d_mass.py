"""Consistent and row-sum lumped mass matrices for the 3D solid path (HEX8 / TET4).

Why this module exists
----------------------
`solid3d` ships fully integrated HEX8 and TET4 stiffness with verified static
benchmarks (bending, internal pressure, torsion, contact), but the 3D continuum
had **no inertia operator at all**: ``grep -n "mass" src/tensorfem/solid3d.py``
returned nothing, and every modal/eigenfrequency verification in this repository
(`modal`, `plate_vibration`, shell families) is one- or two-dimensional.  Any 3D
eigenfrequency or dynamics claim was therefore unsupported by construction.

The mass operator is also the one quantity with *exact* references that cost
almost nothing to check, so it is worth verifying properly rather than assuming:

* total mass must equal ``rho * V`` (patch-level check on the quadrature),
* the six rigid-body modes must reproduce the exact rigid-body mass and inertia
  tensor ``M R^T`` (a global check of shape-function consistency),
* free-free generalised eigenproblems must produce exactly six zero modes,
* the fixed-free rod axial frequencies must converge to
  ``omega_n = (2n-1) pi / (2L) * sqrt(E / rho)`` at the optimal ``O(h^2)`` rate.

Conventions match :mod:`tensorfem.solid3d`: node coordinates are ``(n, 3)``,
connectivity ``(ne, nen)``, DOF ``3 * node + component`` with component order
``(x, y, z)``, and HEX8 local node ordering follows :data:`tensorfem.solid3d._SIGNS`
(``(-,-,-), (+,-,-), (+,+,-), (-,+,-), (-,-,+), (+,-,+), (+,+,+), (-,+,+)``).
"""

from __future__ import annotations

import math

import torch

from .solid3d import _SIGNS

__all__ = [
    "element_type_size",
    "element_consistent_mass",
    "element_lumped_mass",
    "assemble_mass",
    "rigid_body_modes",
    "rigid_body_mass_matrix",
    "box_rigid_body_reference",
    "generalized_modes",
    "rod_axial_angular_frequency",
    "constrained_modulus",
    "axial_free_dofs",
    "axial_mode_index",
]

_GAUSS_1D = {
    1: (0.0, 2.0),
    2: (-1.0 / math.sqrt(3.0), 1.0),
    3: (0.0, 8.0 / 9.0),
}


def element_type_size(element_type: str) -> int:
    """Return the number of nodes per element, rejecting unknown families."""
    nen = {"hex8": 8, "tet4": 4}.get(str(element_type).lower())
    if nen is None:
        raise ValueError("element_type must be 'hex8' or 'tet4'")
    return nen


def _quadrature_points(order: int, dtype, device):
    """Tensor-product Gauss-Legendre points/weights on [-1, 1]^3."""
    if order == 1:
        pts = [(0.0, 2.0)]
    elif order == 2:
        q = 1.0 / math.sqrt(3.0)
        pts = [(-q, 1.0), (q, 1.0)]
    elif order == 3:
        q = math.sqrt(3.0 / 5.0)
        pts = [(-q, 5.0 / 9.0), (0.0, 8.0 / 9.0), (q, 5.0 / 9.0)]
    else:
        raise ValueError("quadrature order must be 1, 2 or 3")
    out = []
    for xi, wx in pts:
        for eta, wy in pts:
            for zeta, wz in pts:
                out.append((xi, eta, zeta, wx * wy * wz))
    del dtype, device
    return out


def _hex8_shapes(xi, eta, zeta, dtype, device):
    """Trilinear shape functions and their natural derivatives."""
    s = torch.tensor(_SIGNS, dtype=dtype, device=device)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    shape = 0.125 * (1.0 + a * xi) * (1.0 + b * eta) * (1.0 + c * zeta)
    deriv = torch.stack(
        (
            0.125 * a * (1.0 + b * eta) * (1.0 + c * zeta),
            0.125 * b * (1.0 + a * xi) * (1.0 + c * zeta),
            0.125 * c * (1.0 + a * xi) * (1.0 + b * eta),
        ),
        1,
    )
    return shape, deriv


def _expand_block(block: torch.Tensor) -> torch.Tensor:
    """Expand an (ne, nen, nen) scalar block matrix to (ne, 3 nen, 3 nen)."""
    ne, nen = block.shape[0], block.shape[1]
    eye = torch.eye(3, dtype=block.dtype, device=block.device).view(1, 1, 3, 1, 3)
    full = block[:, :, None, :, None] * eye
    return full.reshape(ne, 3 * nen, 3 * nen)


def element_consistent_mass(nodes, elements, density, *, element_type: str = "hex8",
                            order: int = 2):
    """Consistent element mass matrices ``M_e = int rho N^T N dV``.

    ``density`` is a scalar or a per-element tensor.  For straight-sided HEX8
    elements (parallelepipeds under an affine map) the 2x2x2 rule is *exact*
    because ``N_i N_j det J`` is at most quadratic per natural coordinate; for
    generally distorted hexes it is a quadrature approximation whose error is
    quantified in ``docs/solid3d-consistent-mass-verification.md``.  TET4 uses
    the closed form ``rho V / 20 * (ones + I)``, which is exact.
    """
    nen = element_type_size(element_type)
    if elements.ndim != 2 or elements.shape[1] != nen:
        raise ValueError(f"connectivity must be (ne, {nen}) for {element_type}")
    x = nodes[elements.long()]
    ne = x.shape[0]
    rho = torch.as_tensor(density, dtype=nodes.dtype, device=nodes.device).reshape(-1)
    if rho.numel() == 1:
        rho = rho.expand(ne)
    elif rho.numel() != ne:
        raise ValueError("density must be a scalar or have one entry per element")

    if nen == 4:
        # Exact linear-tet consistent mass: rho V / 20 * (ones(4, 4) + I).
        e = x[:, 1] - x[:, 0]
        f = x[:, 2] - x[:, 0]
        g = x[:, 3] - x[:, 0]
        det = torch.linalg.det(torch.stack((e, f, g), 1))
        volume = det.abs() / 6.0
        block = (rho * volume / 20.0)[:, None, None] * (
            torch.ones(nen, nen, dtype=nodes.dtype, device=nodes.device)
            + torch.eye(nen, dtype=nodes.dtype, device=nodes.device)
        )[None]
        return _expand_block(block)

    mass = torch.zeros((ne, 3 * nen, 3 * nen), dtype=nodes.dtype, device=nodes.device)
    for xi, eta, zeta, weight in _quadrature_points(order, nodes.dtype, nodes.device):
        shape, deriv = _hex8_shapes(xi, eta, zeta, nodes.dtype, nodes.device)
        jac = torch.einsum("eia,ib->eab", x, deriv)
        det = torch.linalg.det(jac)
        if torch.any(det <= 0.0):
            raise ValueError("HEX8 element has non-positive Jacobian")
        block = (shape[:, None] * shape[None, :])[None] * (rho * weight * det)[:, None, None]
        mass = mass + _expand_block(block)
    return mass


def element_lumped_mass(nodes, elements, density, *, element_type: str = "hex8",
                        order: int = 2):
    """Row-sum lumped element mass (diagonal), returned as (ne, 3 nen, 3 nen)."""
    consistent = element_consistent_mass(
        nodes, elements, density, element_type=element_type, order=order
    )
    diag = consistent.sum(-1)
    return torch.diag_embed(diag)


def assemble_mass(nodes, elements, density, *, element_type: str = "hex8",
                  lumped: bool = False, order: int = 2):
    """Assemble the global mass matrix with ``solid3d`` DOF numbering."""
    nen = element_type_size(element_type)
    builder = element_lumped_mass if lumped else element_consistent_mass
    me = builder(nodes, elements, density, element_type=element_type, order=order)
    ne = me.shape[0]
    ed = torch.stack(tuple(3 * elements.long() + i for i in range(3)), 2).reshape(ne, 3 * nen)
    rows = ed[:, :, None].expand(-1, -1, 3 * nen).reshape(-1)
    cols = ed[:, None, :].expand(-1, 3 * nen, -1).reshape(-1)
    n_dofs = 3 * nodes.shape[0]
    mass = torch.zeros((n_dofs, n_dofs), dtype=nodes.dtype, device=nodes.device)
    return mass.index_put((rows, cols), me.reshape(-1), accumulate=True)


def rigid_body_modes(nodes, *, about=None):
    """Six orthonormal-free rigid modes ``[tx, ty, tz, rx, ry, rz]`` as (6, 3n)."""
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must be (n, 3)")
    ref = torch.zeros(3, dtype=nodes.dtype, device=nodes.device) if about is None else torch.as_tensor(
        about, dtype=nodes.dtype, device=nodes.device
    )
    x = nodes - ref
    n = nodes.shape[0]
    modes = torch.zeros((6, 3 * n), dtype=nodes.dtype, device=nodes.device)
    for i in range(3):
        modes[i, i::3] = 1.0
    # Rotation about the reference point: u = omega x r.
    modes[3, 1::3] = -x[:, 2]
    modes[3, 2::3] = x[:, 1]
    modes[4, 0::3] = x[:, 2]
    modes[4, 2::3] = -x[:, 0]
    modes[5, 0::3] = -x[:, 1]
    modes[5, 1::3] = x[:, 0]
    return modes


def rigid_body_mass_matrix(nodes, mass, *, about=None):
    """``R M R^T`` (6, 6): the mass and inertia the discrete operator implies."""
    modes = rigid_body_modes(nodes, about=about)
    return modes @ mass @ modes.transpose(0, 1)


def box_rigid_body_reference(total_mass: float, lengths, *, dtype=torch.float64):
    """Exact rigid 6x6 mass/inertia matrix of a rectangular block about its centroid."""
    lx, ly, lz = (float(v) for v in lengths)
    m = float(total_mass)
    ixx = m * (ly**2 + lz**2) / 12.0
    iyy = m * (lx**2 + lz**2) / 12.0
    izz = m * (lx**2 + ly**2) / 12.0
    ref = torch.zeros((6, 6), dtype=dtype)
    ref[0, 0] = ref[1, 1] = ref[2, 2] = m
    ref[3, 3], ref[4, 4], ref[5, 5] = ixx, iyy, izz
    return ref


def generalized_modes(stiffness, mass, *, n_modes: int | None = None, drop: int = 0):
    """Solve ``K phi = omega^2 M phi`` by Cholesky whitening.

    Returns ``(omega2, phi)`` with ascending ``omega2`` and mass-normalised modes
    (``phi^T M phi = I``).  ``drop`` skips the first eigenvalues (used to remove
    the six rigid modes of a free-free body before reporting elastic modes).
    """
    if stiffness.shape != mass.shape or stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("stiffness and mass must be square and equally sized")
    chol = torch.linalg.cholesky(mass)
    step = torch.linalg.solve_triangular(chol, stiffness, upper=False)
    whitened = torch.linalg.solve_triangular(chol, step.transpose(0, 1).contiguous(), upper=False)
    whitened = 0.5 * (whitened + whitened.transpose(0, 1))
    values, vectors = torch.linalg.eigh(whitened)
    vectors = torch.linalg.solve_triangular(
        chol.transpose(0, 1).contiguous(), vectors, upper=True
    )
    start = int(drop)
    stop = values.numel() if n_modes is None else min(values.numel(), start + int(n_modes))
    return values[start:stop], vectors[:, start:stop]


def constrained_modulus(young: float, poisson: float) -> float:
    """Uniaxial-strain modulus ``lambda + 2 mu = E (1 - nu) / ((1 + nu)(1 - 2 nu))``.

    Needed whenever the lateral displacement components are suppressed: such a
    column is *not* governed by ``E`` (the 1D rod modulus), because the Poisson
    contraction is blocked.  Using ``E`` there produces a spurious ~16% error for
    ``nu = 0.3`` (see ``docs/solid3d-consistent-mass-verification.md`` section 4).
    """
    return young * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))


def rod_axial_angular_frequency(mode: int, *, young: float, density: float, length: float,
                                modulus: float | None = None) -> float:
    """Exact fixed-free rod axial frequency ``(2n-1) pi / (2 L) * sqrt(M / rho)``.

    ``modulus`` overrides the axial modulus (defaults to ``young``); pass
    :func:`constrained_modulus` for a column whose lateral strain is suppressed.
    """
    if mode < 1:
        raise ValueError("mode must be >= 1")
    axial = float(young) if modulus is None else float(modulus)
    return (2 * mode - 1) * math.pi / (2.0 * length) * math.sqrt(axial / density)


def axial_mode_index(modes: torch.Tensor, nodal_mass: torch.Tensor, *, axis: int = 2) -> int:
    """Index of the mode with the largest axial share of kinetic energy.

    A full 3D slender block puts bending modes below the axial ones; identifying
    the axial mode by its kinetic-energy split is objective and reproducible.
    ``modes`` is (n_dofs, n_modes) mass-normalised, ``nodal_mass`` the diagonal of
    the (lumped) global mass matrix.
    """
    m = nodal_mass.reshape(-1)
    n_nodes = m.numel() // 3
    diag = m.reshape(n_nodes, 3)
    axial = (modes.reshape(n_nodes, 3, -1)[:, axis, :] ** 2 * diag[:, axis][:, None]).sum(0)
    total = sum(
        (modes.reshape(n_nodes, 3, -1)[:, c, :] ** 2 * diag[:, c][:, None]).sum(0)
        for c in range(3)
    )
    return int(torch.argmax(axial / total))


def axial_free_dofs(nodes, *, axis: int = 2, fixed_at: float = 0.0, tol: float = 1e-12):
    """DOFs left free when lateral motion is suppressed and one end is clamped.

    Comparing a 3D solid against the 1D rod requires suppressing the lateral
    components, otherwise the first modes are bending modes of the block and the
    rod formula is not the right reference.  The returned index vector keeps the
    ``axis`` component of every node whose axial coordinate differs from
    ``fixed_at``.
    """
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must be (n, 3)")
    axial = nodes[:, axis]
    keep = axial.abs() > tol if abs(fixed_at) < tol else (axial - fixed_at).abs() > tol
    return (3 * torch.nonzero(keep, as_tuple=True)[0] + axis).long()