"""Diagnostics that expose the discretisation health of ``cylindrical_shell4``.

The pinched-cylinder / Scordelis-Lo anchors in ``benchmarks/verified`` use
:func:`tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness`.  Those anchors
report "non-monotone convergence", which is normally a sign of a formulation or
integration defect rather than an inherent element limitation.  This module
provides the two checks that decide the question:

``element_nullspace_dimension``
    A 4-node shell element owns 24 DOFs and exactly **6** rigid-body modes.
    Any further zero-energy mode is a spurious mechanism (hourglass mode)
    created by reduced integration.  A rank-deficient element can still match a
    single benchmark value, but its mesh-refinement behaviour is arbitrary.

``rigid_body_patch``
    Energy injected by an exact rigid translation / rotation.  For a consistent
    element this is zero up to the geometric (facet) approximation error, which
    must decay like ``h**2``.

``pinched_cylinder_ladder``
    The convergence ladder used by the anchor, returned as raw numbers so that
    convergence can be judged independently of any acceptance gate.

All routines are pure numpy/torch and run in a few seconds on a laptop.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .cylindrical_shell4 import cylindrical_shell4_stiffness
from .pinched_cylinder import pinched_cylinder_model, solve_pinched_cylinder_linear

RIGID_MODES = 6
__all__ = [
    "RIGID_MODES",
    "element_nullspace_dimension",
    "consistent_integration_nullspace_dimension",
    "rigid_body_patch",
    "pinched_cylinder_ladder",
]


def _element(x0: float, dx: float, dtheta: float) -> torch.Tensor:
    return torch.tensor([[x0, 0.0], [x0 + dx, 0.0],
                         [x0 + dx, dtheta], [x0, dtheta]], dtype=torch.float64)


def element_nullspace_dimension(n_theta: int = 24, radius: float = 300.0) -> int:
    """Number of (near-)zero eigenvalues of one element stiffness matrix.

    ``6`` means the element is rank sufficient.  Anything larger is a
    spurious mechanism.  Section size is picked so that the element is nearly
    square, which is the configuration used by the anchors.
    """
    dx = 25.0
    elem = _element(75.0, dx, (math.pi / 2) / n_theta)
    K = cylindrical_shell4_stiffness(elem, radius, 3e6, 0.3, 3.0).numpy()
    w = np.linalg.eigvalsh(K)
    return int((np.abs(w) <= 1e-9 * np.abs(w).max()).sum())


def consistent_integration_nullspace_dimension(n_theta: int = 24,
                                               radius: float = 300.0) -> int:
    """Nullspace size when membrane **and** shear use 2x2 Gauss integration.

    Used to prove that the spurious mechanisms of the production element come
    from the one-point membrane/shear integration and not from the geometry.
    """
    dx = 25.0
    elem = _element(75.0, dx, (math.pi / 2) / n_theta)
    dtype = elem.dtype
    R = torch.as_tensor(radius, dtype=dtype)
    E, nu, t = torch.as_tensor(3e6, dtype=dtype), torch.as_tensor(0.3, dtype=dtype), torch.as_tensor(3.0, dtype=dtype)
    xy = torch.stack((elem[:, 0], R * elem[:, 1]), dim=1)
    D0 = E / (1 - nu ** 2) * torch.stack((
        torch.stack((torch.ones_like(nu), nu, torch.zeros_like(nu))),
        torch.stack((nu, torch.ones_like(nu), torch.zeros_like(nu))),
        torch.stack((torch.zeros_like(nu), torch.zeros_like(nu), (1 - nu) / 2))))
    Dm, Db = D0 * t, D0 * t ** 3 / 12
    Ds = 5.0 / 6.0 * E / (2 * (1 + nu)) * t * torch.eye(2, dtype=dtype)
    K = torch.zeros((24, 24), dtype=dtype)

    def blocks(xi, eta):
        N = xy.new_tensor([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                           (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4
        nat = xy.new_tensor([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                             [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]]) / 4
        J = xy.T @ nat
        det = torch.linalg.det(J)
        d = nat @ torch.linalg.inv(J)
        Bm = torch.zeros((3, 24), dtype=dtype)
        Bs = torch.zeros((2, 24), dtype=dtype)
        for a in range(4):
            j = 6 * a
            dx_a, ds_a = d[a]
            Bm[0, j] = dx_a
            Bm[1, j + 1] = ds_a
            Bm[1, j + 2] = N[a] / R
            Bm[2, j] = ds_a
            Bm[2, j + 1] = dx_a
            Bs[0, j + 2] = dx_a
            Bs[0, j + 4] = N[a]
            Bs[1, j + 2] = ds_a
            Bs[1, j + 1] = -N[a] / R
            Bs[1, j + 3] = -N[a]
        return det, Bm, Bs

    g = 3.0 ** -0.5
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        det, Bm, Bs = blocks(xi, eta)
        K += (Bm.T @ Dm @ Bm + Bs.T @ Ds @ Bs) * det
    w = np.linalg.eigvalsh(K.numpy())
    return int((np.abs(w) <= 1e-9 * np.abs(w).max()).sum())


def _global_rigid_dofs(elem: torch.Tensor, radius: float,
                       translation=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Exact rigid motion sampled at the nodes, expressed in **global** DOFs."""
    tr, om = np.array(translation, float), np.array(omega, float)
    u = np.zeros(24)
    for a, (x, th) in enumerate(elem.numpy()):
        r = np.array([x, radius * math.sin(th), radius * math.cos(th)])
        u[6 * a:6 * a + 3] = tr + np.cross(om, r)
        u[6 * a + 3:6 * a + 6] = om
    return u


def rigid_body_patch(n_theta: int = 24, radius: float = 300.0) -> dict:
    """Energy of exact rigid modes, relative to a unit membrane-strain energy."""
    scale = 3e6 * 3.0 * (0.1 * radius) ** 2
    dx = 25.0
    elem = _element(75.0, dx, (math.pi / 2) / n_theta)
    K = cylindrical_shell4_stiffness(elem, radius, 3e6, 0.3, 3.0).numpy()
    out = {}
    for label, kw in (("Tx", dict(translation=(1, 0, 0))),
                      ("Ty", dict(translation=(0, 1, 0))),
                      ("Tz", dict(translation=(0, 0, 1))),
                      ("Rx", dict(omega=(1, 0, 0)))):
        u = _global_rigid_dofs(elem, radius, **kw)
        out[label] = float(0.5 * u @ (K @ u) / scale)
    return out


def pinched_cylinder_ladder(mesh_sizes=(2, 4, 6, 8)) -> list:
    """Raw pinched-cylinder displacements / percent errors for each mesh."""
    ref = -1.8248e-5
    out = []
    for n in mesh_sizes:
        res = solve_pinched_cylinder_linear(n)
        out.append({"n": n, "displacement": res.displacement,
                    "error_pct": 100.0 * abs(res.displacement / ref - 1.0)})
    return out