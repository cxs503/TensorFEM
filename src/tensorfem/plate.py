"""Four-node Mindlin-Reissner plate element with selective integration.

The element has ``(w, theta_x, theta_y)`` at each node.  Bending is
integrated with a 2x2 Gauss rule and transverse shear at the element centre;
this is the standard selective-reduced-integration (SRI) Q4 formulation used
to avoid shear locking in the thin-plate limit.

``shear_treatment`` selects the transverse-shear field.  ``"sri"`` (default,
unchanged behaviour) is the one-point shear rule; it keeps the bending accuracy
but leaves two zero-energy mechanisms, so the element is rank deficient by two
(see :mod:`tensorfem.element_rank_audit` and
``docs/element-rank-deficiency-audit.md``).  ``"mitc"`` uses the four-node
assumed natural shear strain field (Dvorkin-Bathe) sampled at the four mid-side
points and integrated with the 2x2 rule; the element then has exactly the three
rigid-body mechanisms and no spurious mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlateResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    stiffness: torch.Tensor
    force: torch.Tensor


def _shape(xi: torch.Tensor, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    one = xi.new_tensor(1.0)
    N = 0.25 * torch.stack(((one-xi)*(one-eta), (one+xi)*(one-eta),
                            (one+xi)*(one+eta), (one-xi)*(one+eta)))
    dN = 0.25 * torch.stack((
        torch.stack((-(one-eta), -(one-xi))),
        torch.stack(((one-eta), -(one+xi))),
        torch.stack(((one+eta), (one+xi))),
        torch.stack((-(one+eta), (one-xi))),
    ))
    return N, dN


def q4_mindlin_stiffness(
    xy: torch.Tensor,
    young: torch.Tensor | float,
    poisson: torch.Tensor | float,
    thickness: torch.Tensor | float,
    shear_correction: float = 5.0 / 6.0,
    shear_treatment: str = "sri",
) -> torch.Tensor:
    """Return the 12x12 Mindlin Q4 element stiffness.

    ``shear_treatment="sri"`` (default) is the historical one-point shear rule;
    ``"mitc"`` is the assumed natural transverse shear strain field.  Both use the
    same 2x2 bending rule, so only the shear block changes.
    """
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2) in counter-clockwise order")
    dtype, device = xy.dtype, xy.device
    E = torch.as_tensor(young, dtype=dtype, device=device)
    nu = torch.as_tensor(poisson, dtype=dtype, device=device)
    t = torch.as_tensor(thickness, dtype=dtype, device=device)
    if bool((t <= 0).item()) or bool((nu <= -1).item()) or bool((nu >= .5).item()):
        raise ValueError("invalid plate material or thickness")
    if shear_treatment not in ("sri", "mitc"):
        raise ValueError("shear_treatment must be 'sri' or 'mitc'")
    Db = E*t**3/(12*(1-nu**2))*torch.stack((
        torch.stack((torch.ones_like(nu), nu, torch.zeros_like(nu))),
        torch.stack((nu, torch.ones_like(nu), torch.zeros_like(nu))),
        torch.stack((torch.zeros_like(nu), torch.zeros_like(nu), (1-nu)/2)),
    ))
    Ds = shear_correction * E/(2*(1+nu))*t * torch.eye(2, dtype=dtype, device=device)
    ke = torch.zeros((12, 12), dtype=dtype, device=device)
    gp = 1.0 / 3.0**0.5
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
            _, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)
            if bool((detJ <= 0).item()):
                raise ValueError("plate element has non-positive Jacobian")
            # J has natural-coordinate rows and physical-coordinate columns.
            # Row gradients therefore transform with J^{-T}, not J^{-1}.
            dN = dN_nat @ torch.linalg.inv(J).T
            Bb = torch.zeros((3, 12), dtype=dtype, device=device)
            for i in range(4):
                # kappa_x=d(theta_x)/dx, kappa_y=d(theta_y)/dy
                Bb[0, 3*i+1] = dN[i, 0]
                Bb[1, 3*i+2] = dN[i, 1]
                Bb[2, 3*i+1] = dN[i, 1]
                Bb[2, 3*i+2] = dN[i, 0]
            ke = ke + Bb.T @ Db @ Bb * detJ
    if shear_treatment == "sri":
        # One point (weight 4) for shear: removes thin-plate shear locking.
        xi = eta = xy.new_tensor(0.0)
        N, dN_nat = _shape(xi, eta)
        J = dN_nat.T @ xy
        detJ = torch.linalg.det(J)
        dN = dN_nat @ torch.linalg.inv(J).T
        Bs = torch.zeros((2, 12), dtype=dtype, device=device)
        for i in range(4):
            Bs[0, 3*i] = dN[i, 0]
            Bs[0, 3*i+1] = N[i]
            Bs[1, 3*i] = dN[i, 1]
            Bs[1, 3*i+2] = N[i]
        return ke + Bs.T @ Ds @ Bs * detJ * 4.0
    # Assumed natural shear strain (Dvorkin-Bathe): sample the shear rows at the
    # four mid-side points and interpolate them bilinearly at the 2x2 Gauss
    # points.  This keeps the two rigid-body rotations zero-energy while removing
    # the two spurious mechanisms of the one-point rule.
    def shear_rows(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        N, dN_nat = _shape(xi, eta)
        J = dN_nat.T @ xy
        dN = dN_nat @ torch.linalg.inv(J).T
        rows = torch.zeros((2, 12), dtype=dtype, device=device)
        for i in range(4):
            rows[0, 3*i] = dN[i, 0]
            rows[0, 3*i+1] = N[i]
            rows[1, 3*i] = dN[i, 1]
            rows[1, 3*i+2] = N[i]
        return rows

    zero, one = xy.new_tensor(0.0), xy.new_tensor(1.0)
    bottom = shear_rows(zero, -one)   # gamma_xz at (0,-1)
    top = shear_rows(zero, one)       # gamma_xz at (0,+1)
    left = shear_rows(-one, zero)     # gamma_yz at (-1,0)
    right = shear_rows(one, zero)     # gamma_yz at (+1,0)
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
            _, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)
            if bool((detJ <= 0).item()):
                raise ValueError("plate element has non-positive Jacobian")
            Bs = torch.zeros((2, 12), dtype=dtype, device=device)
            Bs[0] = 0.5*(1.0-eta)*bottom[0] + 0.5*(1.0+eta)*top[0]
            Bs[1] = 0.5*(1.0-xi)*left[1] + 0.5*(1.0+xi)*right[1]
            ke = ke + Bs.T @ Ds @ Bs * detJ
    return ke


def structured_square_mesh(n: int, side: float = 1.0, *, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 1:
        raise ValueError("n must be positive")
    x = torch.linspace(0.0, side, n+1, dtype=dtype)
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elements = []
    for j in range(n):
        for i in range(n):
            n0 = j*(n+1)+i
            elements.append((n0, n0+1, n0+n+2, n0+n+1))
    return nodes, torch.tensor(elements, dtype=torch.long)


def solve_simply_supported_sine_square(
    n: int, *, side: float = 1.0, young: float = 1.0e7,
    poisson: float = 0.3, thickness: float = 0.01, q0: float = 1.0,
    dtype=torch.float64,
) -> PlateResult:
    """Solve a simply-supported square under q=q0 sin(pi*x/a)sin(pi*y/a)."""
    nodes, elems = structured_square_mesh(n, side, dtype=dtype)
    ndof = 3*nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=dtype)
    f = torch.zeros(ndof, dtype=dtype)
    gp = 1.0 / 3.0**0.5
    for conn in elems:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        fe = torch.zeros(12, dtype=dtype)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                xi, eta = nodes.new_tensor(xif), nodes.new_tensor(etaf)
                N, dN_nat = _shape(xi, eta)
                J = dN_nat.T @ xy
                p = N @ xy
                q = q0*torch.sin(torch.pi*p[0]/side)*torch.sin(torch.pi*p[1]/side)
                fe[0::3] += N*q*torch.linalg.det(J)
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        f[dofs] += fe
    tol = side*1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0]-side).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1]-side).abs() < tol)
    edge = xedge | yedge
    # Hard simply-supported Mindlin boundary: w=0 and tangential rotation=0.
    fixed = torch.cat((3*torch.nonzero(edge, as_tuple=False).flatten(),
                       3*torch.nonzero(yedge, as_tuple=False).flatten()+1,
                       3*torch.nonzero(xedge, as_tuple=False).flatten()+2)).unique()
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    return PlateResult(u, K@u-f, K, f)


def kirchhoff_sine_center_deflection(side: float, young: float, poisson: float,
                                      thickness: float, q0: float) -> float:
    """Navier exact thin-plate centre deflection for one sinusoidal load term."""
    D = young*thickness**3/(12*(1-poisson**2))
    return q0*side**4/(4*torch.pi**4*D)


def mindlin_sine_center_deflection(side: float, young: float, poisson: float,
                                   thickness: float, q0: float,
                                   shear_correction: float = 5/6) -> float:
    """Navier exact Mindlin result: bending plus transverse-shear deflection."""
    bending = kirchhoff_sine_center_deflection(side, young, poisson, thickness, q0)
    shear = q0*side**2/(2*torch.pi**2*shear_correction*
                       (young/(2*(1+poisson)))*thickness)
    return bending + shear
