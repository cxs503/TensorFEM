"""HEX20 verification under mesh distortion, curvature and a Jacobian bug control.

The published HEX20 evidence only covered the regular unit cube: affine patch
test, uniaxial traction and a slender cantilever on a structured mesh, plus a
rank audit on one undistorted brick.  Two questions were therefore open and are
closed here by cheap, closed-form checks:

1. Does the isoparametric HEX20 keep its exactness properties when the elements
   are *distorted* (skewed, graded, curved edges, genuine surface curvature)?
   Isoparametric theory says a linear field must be reproduced exactly for any
   non-degenerate mapping, because ``u(xi) = a + A x(xi)`` and the geometry is
   interpolated by the very same shape functions.  This module measures the
   residual instead of trusting the theorem.
2. Would the existing gates actually *catch* a violation?  A deliberately
   broken stiffness is provided (``center_jacobian=True``: the Jacobian of the
   element centre is reused at all 27 Gauss points, a classic implementation
   bug) and must fail the same patch test that the correct code passes at
   roundoff.

Quantity of interest for the exactness statements is the relative residual
``|u_h - u_exact| / |u_exact|`` on the *interior* nodes with the exact linear
field prescribed on the boundary, plus the relative error of the element
centroid stress and the exact strain-energy identity
``u^T K u = V sigma:epsilon`` (equality for a constant-strain field, any
mapping).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from .hex20 import hex20_shape_gradients, structured_hex20_mesh
from .solid3d import _b, elasticity_matrix_3d

_GAUSS_Q = math.sqrt(3.0 / 5.0)
_GAUSS_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def gauss_rule() -> List[Tuple[float, float, float, float]]:
    """The 3x3x3 rule used by :func:`tensorfem.hex20.hex20_stiffness`."""
    pts = (-_GAUSS_Q, 0.0, _GAUSS_Q)
    out = []
    for i, r in enumerate(pts):
        for j, s in enumerate(pts):
            for k, t in enumerate(pts):
                out.append((r, s, t, _GAUSS_W[i] * _GAUSS_W[j] * _GAUSS_W[k]))
    return out


# --------------------------------------------------------------------------- #
# geometry generators
# --------------------------------------------------------------------------- #
def warp(points: torch.Tensor, amplitude: float) -> torch.Tensor:
    """Quadratic warp ``x + a*(y z, x z, x y)``; straight edges become parabolic."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return torch.stack((x + amplitude * y * z, y + amplitude * x * z, z + amplitude * x * y), 1)


def graded_shear(points: torch.Tensor, amplitude: float) -> torch.Tensor:
    """Graded affine shear ``(x + a y, y + a z / 2, z)``: skewed, graded elements."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return torch.stack((x + amplitude * y, y + amplitude * z / 2, z), 1)


def cylindrical_bend(points: torch.Tensor, radius: float, axis_x: float = 1.0) -> torch.Tensor:
    """Bend a straight block into a genuinely curved (thick) shell sector."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = (x - axis_x) / radius
    rho = radius + z
    return torch.stack((rho * torch.sin(theta) + axis_x, y, rho * torch.cos(theta) - radius), 1)


def boundary_nodes(
    reference: torch.Tensor, length: float, width: float, height: float, tol: float = 1e-9
) -> torch.Tensor:
    """Nodes on the outer box of the *reference* (pre-distortion) configuration."""
    on = (
        (reference[:, 0] <= tol)
        | (reference[:, 0] >= length - tol)
        | (reference[:, 1] <= tol)
        | (reference[:, 1] >= width - tol)
        | (reference[:, 2] <= tol)
        | (reference[:, 2] >= height - tol)
    )
    return torch.where(on)[0]


# --------------------------------------------------------------------------- #
# element kernels (correct one re-uses the production routine)
# --------------------------------------------------------------------------- #
def element_stiffness(x: torch.Tensor, d: torch.Tensor, *, center_jacobian: bool = False):
    """Element stiffness; ``center_jacobian=True`` is the deliberate bug control."""
    from .hex20 import hex20_stiffness

    if not center_jacobian:
        return hex20_stiffness(x, d)
    ke = x.new_zeros((len(x), 60, 60))
    _, dn0 = hex20_shape_gradients(0.0, 0.0, 0.0, like=x)
    jac = torch.einsum("eia,ib->eab", x, dn0)
    inv = torch.linalg.inv(jac)
    det = torch.linalg.det(jac)
    for r, s, t, w in gauss_rule():
        _, dn = hex20_shape_gradients(r, s, t, like=x)
        g = torch.einsum("ib,ebc->eic", dn, inv)
        b = _b(g)
        ke = ke + (b.transpose(1, 2) @ d @ b) * (det * w)[:, None, None]
    g0 = torch.einsum("ib,ebc->eic", dn0, inv)
    return ke, _b(g0)


def reduced_rule() -> List[Tuple[float, float, float, float]]:
    """The 2x2x2 rule -- used only to *rule out* reduced integration of HEX20."""
    q = 1.0 / math.sqrt(3.0)
    return [(r, s, t, 1.0) for r in (-q, q) for s in (-q, q) for t in (-q, q)]


def element_stiffness_reduced(x: torch.Tensor, d: torch.Tensor):
    """2x2x2-integrated HEX20 stiffness: the inadmissible reduced variant."""
    from .hex20 import _gradient

    ke = x.new_zeros((len(x), 60, 60))
    for r, s, t, w in reduced_rule():
        _, dn = hex20_shape_gradients(r, s, t, like=x)
        g, det = _gradient(x, dn)
        b = _b(g)
        ke = ke + (b.transpose(1, 2) @ d @ b) * (det * w)[:, None, None]
    _, dn0 = hex20_shape_gradients(0.0, 0.0, 0.0, like=x)
    g0 = torch.einsum("ib,ebc->eic", dn0, torch.linalg.inv(torch.einsum("eia,ib->eab", x, dn0)))
    return ke, _b(g0)


def element_volume(x: torch.Tensor) -> torch.Tensor:
    """Exact volume: ``det J`` is quadratic, so the 3x3x3 rule integrates it exactly."""
    vol = x.new_zeros(len(x))
    for r, s, t, w in gauss_rule():
        _, dn = hex20_shape_gradients(r, s, t, like=x)
        jac = torch.einsum("eia,ib->eab", x, dn)
        vol = vol + torch.linalg.det(jac) * w
    return vol


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def linear_field(nodes: torch.Tensor, gradient: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Nodal values of ``u(x) = A x + b`` flattened dof-wise."""
    return (nodes @ gradient.T + offset).reshape(-1)


def energy_identity_residual(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    gradient: torch.Tensor,
    young: float = 70e9,
    poisson: float = 0.27,
    *,
    center_jacobian: bool = False,
) -> float:
    """Relative error of ``u^T K u = V sigma:epsilon`` for a constant-strain field."""
    x = nodes[elements.long()]
    d = elasticity_matrix_3d(x.new_full((len(x),), young), x.new_full((len(x),), poisson))
    ke, b = element_stiffness(x, d, center_jacobian=center_jacobian)
    ue = (x @ gradient.T).reshape(len(x), 60)
    lhs = torch.einsum("ei,eij,ej->e", ue, ke, ue)
    strain = torch.einsum("eij,ej->ei", b, ue)
    stress = torch.einsum("eij,ej->ei", d, strain)
    rhs = element_volume(x) * torch.einsum("ei,ei->e", stress, strain)
    return float(torch.max(torch.abs(lhs - rhs) / torch.abs(rhs)))


def patch_residual(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    boundary: torch.Tensor,
    gradient: torch.Tensor,
    offset: torch.Tensor | None = None,
    young: float = 70e9,
    poisson: float = 0.27,
    *,
    center_jacobian: bool = False,
) -> Dict[str, float]:
    """Prescribe an exact linear field on the boundary and solve for the interior.

    Returns the relative displacement error on all dofs (the interior ones carry
    the information) and the relative error of the element-centroid stress.
    """
    if offset is None:
        offset = torch.zeros(3, dtype=nodes.dtype)
    exact = linear_field(nodes, gradient, offset)
    fixed = torch.stack((3 * boundary, 3 * boundary + 1, 3 * boundary + 2), 1).reshape(-1).long()
    x = nodes[elements.long()]
    d = elasticity_matrix_3d(x.new_full((len(x),), young), x.new_full((len(x),), poisson))
    ke, b = element_stiffness(x, d, center_jacobian=center_jacobian)
    n = 3 * len(nodes)
    ed = torch.stack(tuple(3 * elements.long() + i for i in range(3)), 2).reshape(len(x), 60)
    rows = ed[:, :, None].expand(-1, -1, 60).reshape(-1)
    cols = ed[:, None, :].expand(-1, 60, -1).reshape(-1)
    k = nodes.new_zeros((n, n)).index_put((rows, cols), ke.reshape(-1), accumulate=True)
    mask = torch.ones(n, dtype=torch.bool)
    mask[fixed] = False
    free = torch.arange(n)[mask]
    u = nodes.new_zeros(n).index_put((fixed,), exact[fixed])
    rhs = -k[free][:, fixed] @ exact[fixed]
    u = u.index_put((free,), torch.linalg.solve(k[free][:, free], rhs))

    disp = float(torch.linalg.vector_norm(u - exact) / torch.linalg.vector_norm(exact))
    ue = u[ed]
    strain = torch.einsum("eij,ej->ei", b, ue)
    stress = torch.einsum("eij,ej->ei", d, strain)
    ref_strain = (gradient + gradient.T) / 2
    dm = d[0]
    ref = torch.stack(
        (
            dm[0, 0] * ref_strain[0, 0]
            + dm[0, 1] * ref_strain[1, 1]
            + dm[0, 2] * ref_strain[2, 2]
            + 2 * dm[0, 3] * ref_strain[0, 1]
            + 2 * dm[0, 4] * ref_strain[1, 2]
            + 2 * dm[0, 5] * ref_strain[0, 2],
            dm[1, 0] * ref_strain[0, 0]
            + dm[1, 1] * ref_strain[1, 1]
            + dm[1, 2] * ref_strain[2, 2]
            + 2 * dm[1, 3] * ref_strain[0, 1]
            + 2 * dm[1, 4] * ref_strain[1, 2]
            + 2 * dm[1, 5] * ref_strain[0, 2],
            dm[2, 0] * ref_strain[0, 0]
            + dm[2, 1] * ref_strain[1, 1]
            + dm[2, 2] * ref_strain[2, 2]
            + 2 * dm[2, 3] * ref_strain[0, 1]
            + 2 * dm[2, 4] * ref_strain[1, 2]
            + 2 * dm[2, 5] * ref_strain[0, 2],
            dm[3, 0] * ref_strain[0, 0]
            + dm[3, 1] * ref_strain[1, 1]
            + dm[3, 2] * ref_strain[2, 2]
            + 2 * dm[3, 3] * ref_strain[0, 1]
            + 2 * dm[3, 4] * ref_strain[1, 2]
            + 2 * dm[3, 5] * ref_strain[0, 2],
            dm[4, 0] * ref_strain[0, 0]
            + dm[4, 1] * ref_strain[1, 1]
            + dm[4, 2] * ref_strain[2, 2]
            + 2 * dm[4, 3] * ref_strain[0, 1]
            + 2 * dm[4, 4] * ref_strain[1, 2]
            + 2 * dm[4, 5] * ref_strain[0, 2],
            dm[5, 0] * ref_strain[0, 0]
            + dm[5, 1] * ref_strain[1, 1]
            + dm[5, 2] * ref_strain[2, 2]
            + 2 * dm[5, 3] * ref_strain[0, 1]
            + 2 * dm[5, 4] * ref_strain[1, 2]
            + 2 * dm[5, 5] * ref_strain[0, 2],
        )
    )
    stress_err = float(torch.max(torch.abs(stress - ref) / torch.abs(ref).clamp(min=1e-300)))
    return {
        "displacement_relative_error": disp,
        "centroid_stress_relative_error": stress_err,
        "free_dofs": int(len(free)),
        "boundary_nodes": int(len(boundary)),
    }


def rank_audit(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float = 70e9,
    poisson: float = 0.27,
    label: str = "hex20",
):
    """Null-space classification of one distorted brick (6 rigid modes, no hourglass)."""
    from .element_rank_audit import audit_element, rigid_body_basis_solid3d

    x = nodes[elements.long()]
    d = elasticity_matrix_3d(x.new_full((len(x),), young), x.new_full((len(x),), poisson))
    ke, _ = element_stiffness(x, d)
    # A single element: its own nodes define the rigid-body basis.
    return audit_element(label, ke[0], rigid_body_basis_solid3d(x[0]))


def qualification_sweep(
    amplitudes: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
    gradient: torch.Tensor | None = None,
) -> Dict[str, List[Dict[str, float]]]:
    """Patch / energy / rank results over the distortion amplitude sweep."""
    if gradient is None:
        gradient = torch.tensor(
            [[0.012, 0.007, -0.003], [-0.002, -0.009, 0.005], [0.004, 0.006, 0.015]],
            dtype=torch.float64,
        )
    offset = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    out: Dict[str, List[Dict[str, float]]] = {"warp": [], "graded_shear": []}
    for kind in out:
        for a in amplitudes:
            nodes, cells = structured_hex20_mesh(2.0, 1.0, 1.0, 2, 2, 2)
            ref = nodes.clone()
            nodes = warp(nodes, a) if kind == "warp" else graded_shear(nodes, a)
            bnd = boundary_nodes(ref, 2.0, 1.0, 1.0)
            good = patch_residual(nodes, cells, bnd, gradient, offset)
            bad = patch_residual(nodes, cells, bnd, gradient, offset, center_jacobian=True)
            out[kind].append(
                {
                    "amplitude": float(a),
                    "displacement_relative_error": good["displacement_relative_error"],
                    "centroid_stress_relative_error": good["centroid_stress_relative_error"],
                    "energy_identity_relative_error": energy_identity_residual(
                        nodes, cells, gradient
                    ),
                    "control_displacement_relative_error": bad["displacement_relative_error"],
                }
            )
    return out
