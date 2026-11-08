"""Kerr (three-parameter) elastic foundation for plate bending.

The Kerr model extends the Winkler foundation by introducing a shear layer
that couples adjacent springs, providing a more realistic representation of
soil-structure interaction.  The foundation pressure is

    q_f = k * w  -  G_p * nabla^2 w

where *k* is the Winkler spring stiffness (N/m³) and *G_p* is the shear
parameter of the intermediate layer (N/m).  In the limit G_p → 0 the Kerr
model reduces to the Winkler model.

For a Q4 Mindlin plate element the Kerr foundation contributes two terms
to the element stiffness:

    K_Kerr = k * ∫ N_w^T N_w dA  +  G_p * ∫ (dN_w/dx)^T (dN_w/dx)
                                              + (dN_w/dy)^T (dN_w/dy) dA

where N_w are the shape functions associated with the transverse displacement
DOF (every 3rd DOF starting at index 0 in the Q4 element).

References
----------
[1] Kerr, A.D., "Elastic and Viscoelastic Foundation Models", J. Appl. Mech.,
    1964.
[2] Vlasov, V.Z. & Leont'ev, N.N., "Beams, Plates and Shells on Elastic
    Foundations", Israel Program for Scientific Translations, 1960.
"""
from __future__ import annotations

import torch

from .plate import _shape


def q4_kerr_foundation_stiffness(
    xy: torch.Tensor,
    winkler_k: float,
    shear_gp: float,
) -> torch.Tensor:
    """Return the 12×12 Kerr foundation stiffness for a Q4 plate element.

    Parameters
    ----------
    xy : (4, 2) tensor
        Nodal physical coordinates in counter-clockwise order.
    winkler_k : float
        Winkler spring coefficient *k* (N/m³).
    shear_gp : float
        Shear layer parameter *G_p* (N/m).

    Returns
    -------
    ke : (12, 12) tensor
        Foundation stiffness.  Only transverse-displacement DOFs
        (indices 0, 3, 6, 9) receive non-zero entries.
    """
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    dtype, device = xy.dtype, xy.device
    ke = torch.zeros((12, 12), dtype=dtype, device=device)

    if winkler_k == 0.0 and shear_gp == 0.0:
        return ke

    gp = 1.0 / 3.0 ** 0.5
    one = xy.new_tensor(1.0)

    # Accumulate 4×4 w–w block
    kw_4x4 = torch.zeros((4, 4), dtype=dtype, device=device)

    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi = xy.new_tensor(xif)
            eta = xy.new_tensor(etaf)
            N, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)
            if bool((detJ <= 0).item()):
                raise ValueError("plate element has non-positive Jacobian")
            Jinv = torch.linalg.inv(J)
            dN = dN_nat @ Jinv.T  # (4, 2): dN[i,:] = [dN_i/dx, dN_i/dy]

            # Winkler part: k * N_i * N_j
            if winkler_k != 0.0:
                kw_4x4 += winkler_k * N[:, None] * N[None, :] * detJ

            # Shear part: G_p * (dN_i/dx * dN_j/dx + dN_i/dy * dN_j/dy)
            if shear_gp != 0.0:
                grad_dot = dN @ dN.T  # (4, 4)
                kw_4x4 += shear_gp * grad_dot * detJ

    # Scatter into 12×12 at transverse DOF locations
    w_dofs = [0, 3, 6, 9]
    for i, di in enumerate(w_dofs):
        for j, dj in enumerate(w_dofs):
            ke[di, dj] = kw_4x4[i, j]
    return ke


def assemble_kerr_foundation(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    winkler_k: float,
    shear_gp: float,
) -> torch.Tensor:
    """Assemble global Kerr foundation stiffness matrix.

    Parameters
    ----------
    nodes : (N, 2) or (N, 3) tensor
    elements : (E, 4) tensor
    winkler_k : float
        Winkler coefficient (N/m³).
    shear_gp : float
        Shear parameter (N/m).

    Returns
    -------
    K : (3N, 3N) tensor
    """
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=nodes.dtype, device=nodes.device)
    for conn in elements:
        xy = nodes[conn, :2]
        ke = q4_kerr_foundation_stiffness(xy, winkler_k, shear_gp)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
    return K
