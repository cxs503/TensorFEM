"""Four-node flat Mindlin shell assembled from membrane and plate actions.

Each node has global ``(ux, uy, uz, rx, ry, rz)`` degrees of freedom.  The
element constructs an orthonormal local frame, combines a plane-stress Q4
membrane with the selective-integration Mindlin plate, and rotates the result
back to global coordinates.  It is a flat facet element, not a curved-shell
formulation.
"""
from __future__ import annotations

import torch

from .continuum import elasticity_matrix, q4_stiffness
from .plate import q4_mindlin_stiffness


def shell4_local_frame(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local coordinates and row-wise local basis for a planar facet."""
    if xyz.shape != (4, 3):
        raise ValueError("xyz must have shape (4, 3)")
    e1 = xyz[1] - xyz[0]
    e1 = e1 / torch.linalg.vector_norm(e1)
    normal = torch.linalg.cross(e1, xyz[3] - xyz[0], dim=0)
    if bool((torch.linalg.vector_norm(normal) <= torch.finfo(xyz.dtype).eps).item()):
        raise ValueError("degenerate shell element")
    e3 = normal / torch.linalg.vector_norm(normal)
    e2 = torch.linalg.cross(e3, e1, dim=0)
    basis = torch.stack((e1, e2, e3))
    local3 = (xyz - xyz[0]) @ basis.T
    scale = torch.linalg.vector_norm(local3[:, :2], dim=1).max()
    # The out-of-plane residual is a cancellation in the projection, so its
    # floor scales with both the *coordinate* magnitude and the dtype epsilon.
    # A fixed 1e-10 relative bound alone rejects perfectly planar facets in
    # float32 once the model is O(100) long (measured: a R=100 cylinder ring
    # residual 1.6e-7, i.e. exactly float32 eps times the coordinate).
    eps = torch.finfo(xyz.dtype).eps
    tol = 1e-10 * scale + 32.0 * eps * torch.linalg.vector_norm(xyz, dim=1).max()
    if bool((local3[:, 2].abs().max() > tol).item()):
        raise ValueError("shell4 nodes must be coplanar")
    return local3[:, :2], basis


def shell4_stiffness(
    xyz: torch.Tensor,
    young: torch.Tensor | float,
    poisson: torch.Tensor | float,
    thickness: torch.Tensor | float,
    *,
    drilling_factor: float = 1e-6,
    plate_shear: str = "sri",
) -> torch.Tensor:
    """Return the 24x24 global stiffness of a flat four-node shell facet.

    ``plate_shear`` is forwarded to :func:`tensorfem.plate.q4_mindlin_stiffness`;
    ``"mitc"`` selects the assumed natural transverse shear strain field and
    removes the two spurious zero-energy plate mechanisms of the default rule.
    """
    xy, basis = shell4_local_frame(xyz)
    dtype, device = xyz.dtype, xyz.device
    E = torch.as_tensor(young, dtype=dtype, device=device)
    nu = torch.as_tensor(poisson, dtype=dtype, device=device)
    t = torch.as_tensor(thickness, dtype=dtype, device=device)
    if bool((E <= 0).item()) or bool((t <= 0).item()):
        raise ValueError("young and thickness must be positive")
    D = elasticity_matrix(E.reshape(1), nu.reshape(1), "stress")
    km, _ = q4_stiffness(xy.unsqueeze(0), D, t.reshape(1))
    kp = q4_mindlin_stiffness(xy, E, nu, t, shear_treatment=plate_shear)
    kl = torch.zeros((24, 24), dtype=dtype, device=device)
    membrane = torch.stack(tuple(6*torch.arange(4, device=device) + i for i in (0, 1)), 1).reshape(-1)
    # Plate ordering is (w, theta_x, theta_y), with theta_x=ry and theta_y=-rx.
    plate = torch.stack((6*torch.arange(4, device=device)+2,
                         6*torch.arange(4, device=device)+4,
                         6*torch.arange(4, device=device)+3), 1).reshape(-1)
    sign = xyz.new_tensor([1., 1., -1.]).repeat(4)
    kl[membrane[:, None], membrane] += km[0]
    kl[plate[:, None], plate] += sign[:, None] * kp * sign[None, :]
    if drilling_factor:
        # Difference-only drilling stabilization preserves uniform rigid spin.
        area = 0.5*abs(torch.linalg.det(torch.stack((xy[1]-xy[0], xy[3]-xy[0]))))
        kd = drilling_factor * E * t * area
        rz = 6*torch.arange(4, device=device)+5
        lap = xyz.new_tensor([[2.,-1.,0.,-1.],[-1.,2.,-1.,0.],
                              [0.,-1.,2.,-1.],[-1.,0.,-1.,2.]])
        kl[rz[:, None], rz] += kd*lap
    transform = torch.zeros((24, 24), dtype=dtype, device=device)
    for node in range(4):
        transform[6*node:6*node+3, 6*node:6*node+3] = basis
        transform[6*node+3:6*node+6, 6*node+3:6*node+6] = basis
    return transform.T @ kl @ transform

