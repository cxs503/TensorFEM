"""Added mass for plate–water interaction.

When a plate vibrates in contact with water on one side, the water inertia
contributes an *added mass* that lowers natural frequencies.  Following
classical hydroelasticity the added mass per unit area is

    m_a = c_m * rho_w * L_char

where *c_m* is a dimensionless coefficient (often ≈ 1 for large plates),
*rho_w* is water density and *L_char* a characteristic length of the
vibrating mode (or simply the plate thickness for a uniform approximation).

This module provides two ways of adding the mass:

1. **Consistent** – the added mass is distributed through the element shape
   functions exactly like the structural consistent mass.
2. **Lumped** – the added mass is assigned equally to the transverse
   displacement DOF of each node (diagonal mass matrix).

Both options can be toggled off (``added_mass_per_area = 0``) so that the
original dry-mass matrix is reproduced exactly (bit-aligned).
"""
from __future__ import annotations

import torch


def q4_added_mass_consistent(
    xy: torch.Tensor,
    added_mass_per_area: torch.Tensor | float,
) -> torch.Tensor:
    """Return the 12×12 consistent added-mass matrix for a Q4 plate element.

    The shape functions are the same as in ``plate._shape``.  The mass is
    applied only to the transverse displacement DOFs (indices 0, 3, 6, 9).

    Parameters
    ----------
    xy : (4, 2) tensor
        Nodal physical coordinates in counter-clockwise order.
    added_mass_per_area : scalar
        Added mass per unit area (kg/m²).  Zero returns an identically-zero
        matrix (bit-identical to dry case).

    Returns
    -------
    ma : (12, 12) tensor
        Consistent added-mass matrix.
    """
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    dtype, device = xy.dtype, xy.device
    ma_pa = torch.as_tensor(added_mass_per_area, dtype=dtype, device=device)
    ma = torch.zeros((12, 12), dtype=dtype, device=device)
    if float(ma_pa) == 0.0:
        return ma

    gp = 1.0 / 3.0 ** 0.5
    one = xy.new_tensor(1.0)

    # Accumulate 4×4 w–w block first, then scatter into 12×12.
    ma_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)

    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi = xy.new_tensor(xif)
            eta = xy.new_tensor(etaf)
            N = 0.25 * torch.stack(
                (
                    (one - xi) * (one - eta),
                    (one + xi) * (one - eta),
                    (one + xi) * (one + eta),
                    (one - xi) * (one + eta),
                )
            )
            dN_nat = 0.25 * torch.stack(
                (
                    torch.stack((-(one - eta), -(one - xi))),
                    torch.stack(((one - eta), -(one + xi))),
                    torch.stack(((one + eta), (one + xi))),
                    torch.stack((-(one + eta), (one - xi))),
                )
            )
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)
            if bool((detJ <= 0).item()):
                raise ValueError("plate element has non-positive Jacobian")
            ma_4x4 += N[:, None] * N[None, :] * detJ

    ma_4x4 = ma_4x4 * ma_pa
    ma_full = torch.zeros((12, 12), dtype=dtype, device=device)
    w_dofs = [0, 3, 6, 9]
    for i, di in enumerate(w_dofs):
        for j, dj in enumerate(w_dofs):
            ma_full[di, dj] = ma_4x4[i, j]
    return ma_full


def assemble_added_mass_consistent(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    added_mass_per_area: float,
) -> torch.Tensor:
    """Assemble the global consistent added-mass matrix.

    Parameters
    ----------
    nodes : (N, 2) or (N, 3) tensor
    elements : (E, 4) tensor
    added_mass_per_area : float
        Added mass per unit area (kg/m²).

    Returns
    -------
    Ma : (3N, 3N) tensor
    """
    ndof = 3 * nodes.shape[0]
    if added_mass_per_area == 0.0:
        return torch.zeros((ndof, ndof), dtype=nodes.dtype, device=nodes.device)
    Ma = torch.zeros((ndof, ndof), dtype=nodes.dtype, device=nodes.device)
    for conn in elements:
        xy = nodes[conn, :2]
        me = q4_added_mass_consistent(xy, added_mass_per_area)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        Ma[dofs[:, None], dofs] += me
    return Ma


def assemble_added_mass_lumped(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    added_mass_per_area: float,
) -> torch.Tensor:
    """Assemble a diagonal (lumped) added-mass matrix.

    Each node receives the sum of its element-contributed area times
    *added_mass_per_area*, placed on the transverse-DOF diagonal only.

    Parameters
    ----------
    nodes : (N, 2) or (N, 3) tensor
    elements : (E, 4) tensor
    added_mass_per_area : float

    Returns
    -------
    Ma_diag : (3N,) tensor  (diagonal stored as 1-D vector)
    """
    n_nodes = nodes.shape[0]
    dtype, device = nodes.dtype, nodes.device
    ma_pa = torch.as_tensor(added_mass_per_area, dtype=dtype, device=device)
    nodal_area = torch.zeros(n_nodes, dtype=dtype, device=device)

    for conn in elements:
        xy = nodes[conn, :2]
        # Element area via 2×2 Gauss
        gp = 1.0 / 3.0 ** 0.5
        one = xy.new_tensor(1.0)
        area = torch.zeros((), dtype=dtype, device=device)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                xi = xy.new_tensor(xif)
                eta = xy.new_tensor(etaf)
                dN_nat = 0.25 * torch.stack(
                    (
                        torch.stack((-(one - eta), -(one - xi))),
                        torch.stack(((one - eta), -(one + xi))),
                        torch.stack(((one + eta), (one + xi))),
                        torch.stack((-(one + eta), (one - xi))),
                    )
                )
                J = dN_nat.T @ xy
                area += torch.linalg.det(J)
        # Distribute area equally to 4 nodes (standard lumping)
        for n in conn:
            nodal_area[n] += area / 4.0

    Ma_diag = nodal_area * ma_pa
    # Expand to full 3N diagonal: only w-DOFs (0, 3, 6, …) get the mass
    full_diag = torch.zeros(3 * n_nodes, dtype=dtype, device=device)
    full_diag[0::3] = Ma_diag
    return full_diag
