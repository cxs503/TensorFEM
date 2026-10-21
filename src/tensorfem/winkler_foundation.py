"""Winkler elastic foundation for plate bending.

The Winkler foundation models soil or water support as a bed of independent
springs with pressure proportional to local deflection::

    q = k * w

where *k* is the foundation coefficient (units: force/length³, e.g. N/m³).

For a Q4 Mindlin-Reissner plate element the foundation contribution to the
element stiffness matrix is

    K_w = k * ∫ N^T N  dA

where N is the shape-function matrix acting on the transverse displacement DOF
(DOF index 0, 3, 6, … within each node).  The integral is evaluated with a
2×2 Gauss quadrature consistent with the bending integration in
``plate.q4_mindlin_stiffness``.
"""
from __future__ import annotations

import torch


def q4_winkler_foundation_stiffness(
    xy: torch.Tensor,
    foundation_coefficient: torch.Tensor | float,
) -> torch.Tensor:
    """Return the 12×12 Winkler foundation stiffness for a Q4 plate element.

    Parameters
    ----------
    xy : (4, 2) tensor
        Nodal physical coordinates in counter-clockwise order.
    foundation_coefficient : scalar
        Winkler coefficient *k* (N/m³).  When zero the returned matrix is
        identically zero.

    Returns
    -------
    kw : (12, 12) tensor
        Foundation stiffness matrix.  Only the transverse-displacement DOFs
        (indices 0, 3, 6, 9) receive non-zero entries.
    """
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    dtype, device = xy.dtype, xy.device
    k = torch.as_tensor(foundation_coefficient, dtype=dtype, device=device)
    kw = torch.zeros((12, 12), dtype=dtype, device=device)
    if float(k) == 0.0:
        return kw

    gp = 1.0 / 3.0 ** 0.5
    one = xy.new_tensor(1.0)

    # Accumulate 4×4 w–w block first, then scatter into 12×12.
    kw_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)

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
            kw_4x4 += N[:, None] * N[None, :] * detJ

    kw_4x4 = kw_4x4 * k
    # Place into the 12×12 matrix at transverse-DOF locations (0,3,6,9)
    kw_full = torch.zeros((12, 12), dtype=dtype, device=device)
    w_dofs = [0, 3, 6, 9]
    for i, di in enumerate(w_dofs):
        for j, dj in enumerate(w_dofs):
            kw_full[di, dj] = kw_4x4[i, j]
    return kw_full


def assemble_winkler_foundation(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    foundation_coefficient: float,
) -> torch.Tensor:
    """Assemble the global Winkler foundation stiffness matrix.

    Parameters
    ----------
    nodes : (N, 2) or (N, 3) tensor
        Nodal coordinates (only x, y are used).
    elements : (E, 4) tensor
        Element connectivity (node indices, CCW order).
    foundation_coefficient : float
        Winkler coefficient *k* (N/m³).

    Returns
    -------
    Kw : (3N, 3N) tensor
        Global foundation stiffness matrix.
    """
    if foundation_coefficient == 0.0:
        ndof = 3 * nodes.shape[0]
        return torch.zeros((ndof, ndof), dtype=nodes.dtype, device=nodes.device)

    ndof = 3 * nodes.shape[0]
    Kw = torch.zeros((ndof, ndof), dtype=nodes.dtype, device=nodes.device)
    for conn in elements:
        xy = nodes[conn, :2]
        ke = q4_winkler_foundation_stiffness(xy, foundation_coefficient)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        Kw[dofs[:, None], dofs] += ke
    return Kw
