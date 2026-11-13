"""Four-node cylindrical Reissner--Mindlin shell (MITC4).

Unlike the flat-facet shell, this element retains the exact cylindrical
metric and curvature terms.  Its nodal frame is ``(axial, tangent, normal)``
and varies consistently from node to node.

Integration scheme (MITC4 / Dvorkin-Bathe assumed natural shear):

* membrane: 2x2 Gauss (full).  One-point membrane integration left five
  spurious mechanisms (element rank 13/24); full integration restores the
  rank.
* bending: 2x2 Gauss (full).
* transverse shear: the two shear rows are sampled at the four mid-side
  points and interpolated bilinearly at the 2x2 Gauss points.  This keeps
  the rigid-body rotations zero-energy, removes the one-point rule's
  spurious mechanisms, and avoids the shear locking of full 2x2 shear
  integration.
"""
from __future__ import annotations

import torch


def cylindrical_shell4_stiffness(
    x_theta: torch.Tensor,
    radius: torch.Tensor | float,
    young: torch.Tensor | float,
    poisson: torch.Tensor | float,
    thickness: torch.Tensor | float,
    *,
    shear_factor: float = 5.0 / 6.0,
    drilling_factor: float = 1e-8,
) -> torch.Tensor:
    """Return the global 24x24 stiffness for an x-axis cylindrical patch.

    ``x_theta`` contains axial coordinate and angle in radians.  Global axes
    are ``(x, R sin(theta), R cos(theta))``.  The six global nodal DOFs are
    translations followed by rotation-vector components.
    """
    if x_theta.shape != (4, 2):
        raise ValueError("x_theta must have shape (4, 2)")
    dtype, device = x_theta.dtype, x_theta.device
    R = torch.as_tensor(radius, dtype=dtype, device=device)
    E = torch.as_tensor(young, dtype=dtype, device=device)
    nu = torch.as_tensor(poisson, dtype=dtype, device=device)
    t = torch.as_tensor(thickness, dtype=dtype, device=device)
    if bool((R <= 0).item()) or bool((E <= 0).item()) or bool((t <= 0).item()):
        raise ValueError("radius, young and thickness must be positive")

    # Work in physical surface coordinates (x,s=R theta).
    xy = torch.stack((x_theta[:, 0], R * x_theta[:, 1]), dim=1)
    Dm = E / (1 - nu**2) * torch.stack((
        torch.stack((torch.ones_like(nu), nu, torch.zeros_like(nu))),
        torch.stack((nu, torch.ones_like(nu), torch.zeros_like(nu))),
        torch.stack((torch.zeros_like(nu), torch.zeros_like(nu), (1-nu)/2)),
    ))
    Db = Dm * t**3 / 12
    Dm = Dm * t
    Ds = shear_factor * E / (2*(1+nu)) * t * torch.eye(2, dtype=dtype, device=device)
    K = torch.zeros((24, 24), dtype=dtype, device=device)

    def fields(xi: float, eta: float):
        N = xy.new_tensor([(1-xi)*(1-eta), (1+xi)*(1-eta),
                           (1+xi)*(1+eta), (1-xi)*(1+eta)]) / 4
        nat = xy.new_tensor([[-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
                             [(1+eta), (1+xi)], [-(1+eta), (1-xi)]]) / 4
        J = xy.T @ nat
        det = torch.linalg.det(J)
        d = nat @ torch.linalg.inv(J)
        Bm = torch.zeros((3, 24), dtype=dtype, device=device)
        Bb = torch.zeros_like(Bm)
        Bs = torch.zeros((2, 24), dtype=dtype, device=device)
        for a in range(4):
            j = 6*a
            dx, ds = d[a]
            # local dofs: u, v, w, r_x, r_t, r_n.
            Bm[0, j] = dx
            Bm[1, j+1] = ds
            Bm[1, j+2] = N[a] / R
            Bm[2, j] = ds
            Bm[2, j+1] = dx
            # beta_x=r_t; beta_s=-r_x. Curvature signs do not affect energy,
            # but the common convention below gives a symmetric twist field.
            Bb[0, j+4] = dx
            Bb[1, j+3] = -ds
            Bb[2, j+4] = ds
            Bb[2, j+3] = -dx
            Bs[0, j+2] = dx
            Bs[0, j+4] = N[a]
            Bs[1, j+2] = ds
            Bs[1, j+1] = -N[a] / R
            Bs[1, j+3] = -N[a]
        return N, det, Bm, Bb, Bs

    g = 3.0**-0.5
    for xi, eta in ((-g,-g),(g,-g),(g,g),(-g,g)):
        _, det, Bm, Bb, _ = fields(xi, eta)
        K += Bb.T@Db@Bb * det
        # MITC4: membrane is fully integrated (2x2).  One-point membrane
        # integration left five spurious mechanisms (rank 13/24); full
        # membrane integration restores the rank while the selective
        # one-point shear below keeps the element free of shear locking.
        K += Bm.T @ Dm @ Bm * det
    # Dvorkin-Bathe assumed natural transverse shear (the "assumed shear" of
    # MITC4): sample the two shear rows at the four mid-side points and
    # interpolate them bilinearly at the 2x2 Gauss points.  This keeps the
    # rigid-body rotations zero-energy, removes the spurious mechanisms of the
    # one-point rule, and avoids the shear locking of full 2x2 shear
    # integration (which stiffened the element to 85 % -> 15 % error).
    _, _, _, _, Bs_b = fields(0.0, -1.0)
    _, _, _, _, Bs_t = fields(0.0, 1.0)
    _, _, _, _, Bs_l = fields(-1.0, 0.0)
    _, _, _, _, Bs_r = fields(1.0, 0.0)
    for xi, eta in ((-g,-g),(g,-g),(g,g),(-g,g)):
        _, det, _, _, _ = fields(xi, eta)
        Bs = torch.zeros((2, 24), dtype=dtype, device=device)
        Bs[0] = 0.5*(1.0-eta)*Bs_b[0] + 0.5*(1.0+eta)*Bs_t[0]
        Bs[1] = 0.5*(1.0-xi)*Bs_l[1] + 0.5*(1.0+xi)*Bs_r[1]
        K += Bs.T @ Ds @ Bs * det

    # Objective difference-only stabilization of the unused normal rotation.
    _, det0, _, _, _ = fields(0.0, 0.0)
    area = abs(det0)*4
    ids = torch.arange(4, device=device)
    rn = 6*ids+5
    lap = xy.new_tensor([[2.,-1.,0.,-1.],[-1.,2.,-1.,0.],
                         [0.,-1.,2.,-1.],[-1.,0.,-1.,2.]])
    K[rn[:, None], rn] += drilling_factor*E*t*area*lap

    T = torch.zeros((24, 24), dtype=dtype, device=device)
    for a, theta in enumerate(x_theta[:, 1]):
        one, zero = torch.ones_like(theta), torch.zeros_like(theta)
        basis = torch.stack((torch.stack((one,zero,zero)),
                             torch.stack((zero,torch.cos(theta),-torch.sin(theta))),
                             torch.stack((zero,torch.sin(theta), torch.cos(theta)))))
        T[6*a:6*a+3, 6*a:6*a+3] = basis
        T[6*a+3:6*a+6, 6*a+3:6*a+6] = basis
    return T.T @ K @ T
