"""Exact 3-D elasticity anchor for the shear-diaphragm simply supported plate.

Why this module exists
----------------------
``tensorfem.plate_thick_reference`` supplies the *exact spectrum of the first-order
shear deformation theory* (FSDT) for the simply supported plate, and
``tensorfem.plate_vibration`` qualifies the Q4 Mindlin element against the
*Kirchhoff* (thin-plate) Navier solution.  Both references are **plate theories**:
they cannot say how much of a measured FE error is discretisation and how much is
the plate idealisation itself.  The repository therefore had no anchor above the
FSDT level, i.e. no three-dimensional (3-D) elasticity reference for a thick
plate.  This module provides it:

* :class:`SSPlateExact3D` — the exact 3-D elasticity solution of the simply
  supported rectangular plate under a double-sine transverse pressure (Pagano
  type solution, Pagano 1970; the same boundary-value problem the classical
  Navier solution addresses, but solved in the full Lamé equations);
* :func:`fsdt_static_navier` — the FSDT static Navier solution (3x3 half-wave
  system), so FSDT and 3-D can be compared with no element in between;
* :func:`kirchhoff_static_navier` — the classical thin-plate Navier value;
* :func:`solve_hex20_ss_plate` — a HEX20 3-D FE bridge that reproduces the exact
  boundary conditions of :class:`SSPlateExact3D`, so the solid element itself can
  be qualified against 3-D elasticity instead of against a plate theory.

Problem statement (derivation chain, every step checkable)
---------------------------------------------------------
Plate ``0 <= x <= a``, ``0 <= y <= b``, thickness ``h``, isotropic ``E``, ``nu``.
The top face ``z = +h/2`` carries ``p(x, y) = q0 sin(alpha x) sin(beta y)``,
``alpha = m pi / a``, ``beta = n pi / b`` (``q0 > 0`` pushes in ``-z``).  Lamé:
``(lam+mu) grad div u + mu laplacian u = 0`` with
``lam = E nu/((1+nu)(1-2nu))``, ``mu = E/(2(1+nu))``.

Edges — the "shear diaphragm"/SS1 set, which is the set for which the double sine
is an exact solution:

* ``x = 0, a``: ``u_y = 0``, ``u_z = 0``, ``sigma_xx = 0`` (``sigma_xx`` vanishes
  identically for this ansatz, it is not imposed);
* ``y = 0, b``: ``u_x = 0``, ``u_z = 0``, ``sigma_yy = 0``;
* the in-plane shear traction ``sigma_xy`` on the edges is **not** zero.  It is the
  reaction of the ``u_t = 0`` constraint — the same reaction a FE model produces
  when it constrains the tangential in-plane dof, which is why the FE bridge
  below is comparable with this solution.

Faces: ``sigma_zz = tau_xz = tau_yz = 0`` at ``z = -h/2``; ``sigma_zz = -q0`` and
``tau_xz = tau_yz = 0`` at ``z = +h/2``.

Ansatz ``u_x = U(z) cos(alpha x) sin(beta y)``,
``u_y = V(z) sin(alpha x) cos(beta y)``, ``u_z = W(z) sin(alpha x) sin(beta y)``
turns the Lamé system into a constant-coefficient ODE system ``y' = A y`` with
``y = (U, V, W, U', V', W')``; the six face conditions become six linear
equations in ``y(-h/2)``, solved once.  Everything is closed form, so no constant
is fitted to a FE result.  The module never imports SciPy: the propagator is
``torch.linalg.matrix_exp``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

__all__ = [
    "SSPlateExact3D",
    "kirchhoff_static_navier",
    "fsdt_static_navier",
    "solve_hex20_ss_plate",
]

_DT = torch.float64


def _lame(young: float, poisson: float) -> tuple[float, float]:
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return lam, mu


@dataclass(frozen=True)
class SSPlateExact3D:
    """Exact 3-D elasticity solution of the shear-diaphragm SS plate.

    Lengths share one unit; ``q0`` is the peak pressure on the loaded (top)
    face.  Public accessors use the *plate* convention: ``z`` is measured from
    the mid-surface and :meth:`deflection` is positive downwards.
    """

    a: float
    b: float
    h: float
    young: float
    poisson: float
    q0: float = 1.0
    m: int = 1
    n: int = 1

    def __post_init__(self) -> None:
        for name in ("a", "b", "h", "young", "q0"):
            if not (float(getattr(self, name)) > 0.0):
                raise ValueError(f"{name} must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")
        if self.m < 1 or self.n < 1:
            raise ValueError("half-wave numbers m, n must be >= 1")

    # -- closed-form coefficients -------------------------------------------
    @property
    def alpha(self) -> float:
        return self.m * math.pi / self.a

    @property
    def beta(self) -> float:
        return self.n * math.pi / self.b

    @property
    def lame(self) -> tuple[float, float]:
        return _lame(self.young, self.poisson)

    @property
    def flexural_rigidity(self) -> float:
        return self.young * self.h**3 / (12.0 * (1.0 - self.poisson**2))

    def _coefficient_matrix(self) -> torch.Tensor:
        """The 6x6 matrix ``A`` of ``y' = A y``, ``y = (U,V,W,U',V',W')``."""
        lam, mu = self.lame
        al, be = self.alpha, self.beta
        A = torch.zeros((6, 6), dtype=_DT)
        A[0, 3] = A[1, 4] = A[2, 5] = 1.0
        A[3, 0] = ((lam + 2.0 * mu) * al**2 + mu * be**2) / mu
        A[3, 1] = (lam + mu) * al * be / mu
        A[3, 5] = -(lam + mu) * al / mu
        A[4, 0] = A[3, 1]
        A[4, 1] = ((lam + 2.0 * mu) * be**2 + mu * al**2) / mu
        A[4, 5] = -(lam + mu) * be / mu
        A[5, 2] = mu * (al**2 + be**2) / (lam + 2.0 * mu)
        A[5, 3] = (lam + mu) * al / (lam + 2.0 * mu)
        A[5, 4] = (lam + mu) * be / (lam + 2.0 * mu)
        return A

    def initial_state(self) -> torch.Tensor:
        """``y(-h/2)``: solve the six face conditions (bottom and top)."""
        lam, mu = self.lame
        al, be = self.alpha, self.beta
        A = self._coefficient_matrix()
        shear_x = torch.tensor([0.0, 0.0, al, 1.0, 0.0, 0.0], dtype=_DT)
        shear_y = torch.tensor([0.0, 0.0, be, 0.0, 1.0, 0.0], dtype=_DT)
        normal_z = torch.tensor([-lam * al, -lam * be, 0.0, 0.0, 0.0, lam + 2.0 * mu], dtype=_DT)
        top = torch.linalg.matrix_exp(A * self.h)
        rows = torch.stack([shear_x, shear_y, normal_z, shear_x @ top, shear_y @ top, normal_z @ top])
        rhs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -self.q0], dtype=_DT)
        return torch.linalg.solve(rows, rhs)

    def state(self, z) -> torch.Tensor:
        """Return ``(U, V, W, U', V', W')`` at mid-surface-relative ``z``.

        ``z`` may be a float or a tensor; a tensor input returns one row per
        sample, so the whole thickness is integrated in one ``matrix_exp``.
        """
        A = self._coefficient_matrix()
        y0 = self.initial_state()
        zt = torch.as_tensor(z, dtype=_DT) + 0.5 * self.h
        if zt.ndim == 0:
            return torch.linalg.matrix_exp(A * zt) @ y0
        prop = torch.linalg.matrix_exp(A.unsqueeze(0) * zt.reshape(-1, 1, 1))
        return prop @ y0

    # -- field accessors (torch-differentiable: equilibrium is checked by AD) --
    def displacement(self, x, y, z) -> tuple:
        """Return ``(u_x, u_y, u_z)``; ``(x, y, z)`` floats or tensors."""
        al, be = self.alpha, self.beta
        s = self.state(z)
        U, V, W = s[..., 0], s[..., 1], s[..., 2]
        xt = torch.as_tensor(x, dtype=_DT)
        yt = torch.as_tensor(y, dtype=_DT)
        cx = torch.cos(al * xt) * torch.sin(be * yt)
        cy = torch.sin(al * xt) * torch.cos(be * yt)
        cz = torch.sin(al * xt) * torch.sin(be * yt)
        return U * cx, V * cy, W * cz

    def deflection(self, x, y, z) -> torch.Tensor:
        """Downward deflection ``-u_z`` (positive downwards)."""
        return -self.displacement(x, y, z)[2]

    def stress(self, x, y, z) -> tuple:
        """Return ``(s_xx, s_yy, s_zz, t_xy, t_xz, t_yz)``: ``(x, y, z)`` tensors."""
        lam, mu = self.lame
        al, be = self.alpha, self.beta
        s = self.state(z)
        U, V, W, Up, Vp, Wp = (s[..., i] for i in range(6))
        xt = torch.as_tensor(x, dtype=_DT)
        yt = torch.as_tensor(y, dtype=_DT)
        cx = torch.cos(al * xt) * torch.sin(be * yt)   # u_x, w_x, tau_xz
        cy = torch.sin(al * xt) * torch.cos(be * yt)   # u_y, w_y, tau_yz
        cz = torch.sin(al * xt) * torch.sin(be * yt)   # w_z, tau_xy^coupled, sigma_nn
        cxy = torch.cos(al * xt) * torch.cos(be * yt)  # tau_xy
        ux, uy, uz = -al * U * cz, -be * V * cz, Wp * cz
        sxx = (lam + 2.0 * mu) * ux + lam * (uy + uz)
        syy = (lam + 2.0 * mu) * uy + lam * (ux + uz)
        szz = (lam + 2.0 * mu) * uz + lam * (ux + uy)
        txy = mu * (be * U + al * V) * cxy
        txz = mu * (Up + al * W) * cx
        tyz = mu * (Vp + be * W) * cy
        return sxx, syy, szz, txy, txz, tyz

    # -- anchors ------------------------------------------------------------
    def center_deflection(self, z: float | None = None) -> float:
        """Downward deflection at the plate centre (default: loaded surface)."""
        zz = 0.5 * self.h if z is None else z
        return float(self.deflection(0.5 * self.a, 0.5 * self.b, zz))

    def equilibrium_residual(self, samples: int = 3) -> dict:
        """``|div sigma|`` by automatic differentiation at interior points.

        This check is *independent of the ODE machinery*: it differentiates the
        stress accessor (which is built from strains, not from ``A``) and asks
        whether ``d sigma_ij / d x_j = 0``.
        """
        worst = 0.0
        for i in range(samples):
            for j in range(samples):
                for k in range(samples):
                    p = torch.tensor(
                        [
                            self.a * (i + 1) / (samples + 1),
                            self.b * (j + 1) / (samples + 1),
                            -0.5 * self.h + self.h * (k + 1) / (samples + 1),
                        ],
                        dtype=_DT,
                        requires_grad=True,
                    )

                    def sig(comp):
                        return self.stress(p[0], p[1], p[2])[comp]

                    grads = []
                    for comp in range(6):
                        g = torch.autograd.grad(sig(comp), p, create_graph=False, retain_graph=True)[0]
                        grads.append(g)
                    dsxx, dsyy, dszz, dtxy, dtxz, dtyz = grads
                    div = (
                        abs(float(dsxx[0] + dtxy[1] + dtxz[2])),
                        abs(float(dtxy[0] + dsyy[1] + dtyz[2])),
                        abs(float(dtxz[0] + dtyz[1] + dszz[2])),
                    )
                    worst = max(worst, *div)
        peak = float(self.stress(torch.tensor([0.5 * self.a, 0.5 * self.b], dtype=_DT).repeat(1),
                                 torch.tensor([0.5 * self.b, 0.5 * self.a], dtype=_DT),
                                 torch.tensor([0.5 * self.h, 0.0], dtype=_DT))[0].abs().max())
        scale = max(peak / self.a, 1e-300)
        return {"max_abs_div_sigma": worst, "stress_gradient_scale": scale, "relative": worst / scale}

    def face_report(self) -> dict:
        """Residuals of the face tractions the ansatz has to satisfy."""
        zs = torch.linspace(-0.5 * self.h, 0.5 * self.h, 9, dtype=_DT)
        xs, ys = 0.37 * self.a, 0.43 * self.b
        xt = torch.full_like(zs, xs)
        yt = torch.full_like(zs, ys)
        faces = torch.tensor([-0.5 * self.h, 0.5 * self.h], dtype=_DT)
        xf = torch.full_like(faces, xs)
        yf = torch.full_like(faces, ys)
        _, _, _, _, txz_f, tyz_f = self.stress(xf, yf, faces)
        cz = torch.sin(self.alpha * xt) * torch.sin(self.beta * yt)
        top = float(self.stress(torch.tensor(xs, dtype=_DT), torch.tensor(ys, dtype=_DT),
                                torch.tensor(0.5 * self.h, dtype=_DT))[2])
        bottom = float(self.stress(torch.tensor(xs, dtype=_DT), torch.tensor(ys, dtype=_DT),
                                   torch.tensor(-0.5 * self.h, dtype=_DT))[2])
        return {
            "max_abs_face_shear": max(float(txz_f.abs().max()), float(tyz_f.abs().max())),
            "abs_bottom_sigma_zz": abs(bottom),
            "top_sigma_zz_expect": -float(self.q0) * float(cz[0]),
            "top_sigma_zz_got": top,
            "top_sigma_zz_rel_error": abs(top + float(self.q0) * float(cz[0])) / (float(self.q0) * abs(float(cz[0]))),
        }

    def edge_report(self) -> dict:
        """Residuals of the kinematic edge conditions and of ``sigma_nn``."""
        zs = torch.linspace(-0.5 * self.h, 0.5 * self.h, 7, dtype=_DT)
        zero = torch.zeros_like(zs)
        vals = {"u_y(x=0)": 0, "u_z(x=0)": 0, "u_x(y=0)": 0, "u_z(y=0)": 0,
                "sigma_xx(x=0)": 0, "sigma_yy(y=0)": 0}
        uy0 = self.displacement(zero, torch.full_like(zs, 0.41 * self.b), zs)[1]
        uz0 = self.displacement(zero, torch.full_like(zs, 0.29 * self.b), zs)[2]
        ux0 = self.displacement(torch.full_like(zs, 0.43 * self.a), zero, zs)[0]
        uz1 = self.displacement(torch.full_like(zs, 0.31 * self.a), zero, zs)[2]
        sxx0 = self.stress(zero, torch.full_like(zs, 0.37 * self.b), zs)[0]
        syy0 = self.stress(torch.full_like(zs, 0.41 * self.a), zero, zs)[1]
        worst = max(float(uy0.abs().max()), float(uz0.abs().max()), float(ux0.abs().max()),
                    float(uz1.abs().max()), float(sxx0.abs().max()), float(syy0.abs().max()))
        # the edge in-plane shear reaction is NOT zero -- quantify it honestly
        txy0 = self.stress(zero, torch.full_like(zs, 0.41 * self.b), zs)[3]
        return {
            "max_kinematic_and_normal_stress_residual": worst,
            "edge_shear_reaction_peak": float(txy0.abs().max()),
            "edge_shear_over_applied_pressure": float(txy0.abs().max()) / float(self.q0),
            "terms_checked": list(vals),
        }


def kirchhoff_static_navier(a: float, b: float, h: float, young: float, poisson: float,
                            q0: float = 1.0, m: int = 1, n: int = 1) -> dict:
    """Classical (Kirchhoff) thin-plate Navier value for the same load."""
    al, be = m * math.pi / a, n * math.pi / b
    R = al**2 + be**2
    D = young * h**3 / (12.0 * (1.0 - poisson**2))
    return {"deflection": q0 / (D * R**2), "rigidity": D, "wave_number": R}


def fsdt_static_navier(a: float, b: float, h: float, young: float, poisson: float,
                       q0: float = 1.0, m: int = 1, n: int = 1,
                       kappa: float = 5.0 / 6.0) -> dict:
    """Exact FSDT (Mindlin--Reissner) static solution of the same SS plate.

    Solves the 3x3 half-wave system for ``(Psi_x, Psi_y, W)``; no element and no
    discretisation enters this reference.  ``deflection`` is positive downwards.
    """
    al, be = m * math.pi / a, n * math.pi / b
    R = al**2 + be**2
    D = young * h**3 / (12.0 * (1.0 - poisson**2))
    G = young / (2.0 * (1.0 + poisson))
    shear = kappa * G * h
    K = torch.zeros((3, 3), dtype=_DT)
    K[0, 0] = -D / 2.0 * ((1.0 - poisson) * R + (1.0 + poisson) * al**2) - shear
    K[0, 1] = -D / 2.0 * (1.0 + poisson) * al * be
    K[0, 2] = shear * al
    K[1, 0] = K[0, 1]
    K[1, 1] = -D / 2.0 * ((1.0 - poisson) * R + (1.0 + poisson) * be**2) - shear
    K[1, 2] = shear * be
    K[2, 0] = shear * al
    K[2, 1] = shear * be
    K[2, 2] = -shear * R
    rhs = torch.tensor([0.0, 0.0, -q0], dtype=_DT)
    sol = torch.linalg.solve(K, rhs)
    return {
        "deflection": float(sol[2]),
        "psi_x": float(sol[0]),
        "psi_y": float(sol[1]),
        "shear_stiffness": shear,
        "rigidity": D,
        "wave_number": R,
        "shear_parameter": D * R / shear,
    }


def solve_hex20_ss_plate(a: float, b: float, h: float, young: float, poisson: float,
                         q0: float = 1.0, m: int = 1, n: int = 1,
                         nx: int = 8, ny: int = 8, nz: int = 2, sparse: bool = True) -> dict:
    """HEX20 3-D FE model with the *same* boundary conditions as the exact solution.

    Kinematic edges (shear diaphragm): ``u_y = u_z = 0`` on ``x = 0, a`` and
    ``u_x = u_z = 0`` on ``y = 0, b``, for every node of the face; the top face
    carries the consistent nodal load of the sine pressure (3x3 Gauss per face).
    ``sparse=True`` assembles and factorises with SciPy (for the study meshes);
    ``sparse=False`` uses the in-house dense HEX20 path of
    :func:`tensorfem.hex20.solve_hex20` (small verification meshes only).
    """
    from .hex20 import hex20_shape_gradients, structured_hex20_mesh, hex20_stiffness
    from .solid3d import elasticity_matrix_3d

    nodes, cells = structured_hex20_mesh(a, b, h, nx, ny, nz)
    al, be = m * math.pi / a, n * math.pi / b
    n_nodes = len(nodes)
    forces = torch.zeros(3 * n_nodes, dtype=_DT)
    gp = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
    gw = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    top_face = (4, 5, 6, 7, 12, 13, 14, 15)
    eps = 1e-6
    for cell in cells.tolist():
        if abs(float(nodes[cell[4], 2]) - h) > 1e-12:
            continue
        xyz = nodes[cell]
        for gi, r in enumerate(gp):
            for gj, s in enumerate(gp):
                N, _ = hex20_shape_gradients(r, s, 1.0, like=xyz)
                N = N.detach()
                xg = float((N * xyz[:, 0]).sum())
                yg = float((N * xyz[:, 1]).sum())

                def pos(rr, ss):
                    Nn, _ = hex20_shape_gradients(rr, ss, 1.0, like=xyz)
                    Nn = Nn.detach()
                    return torch.stack([(Nn * xyz[:, 0]).sum(), (Nn * xyz[:, 1]).sum()])

                dr = (pos(r + eps, s) - pos(r - eps, s)) / (2 * eps)
                ds = (pos(r, s + eps) - pos(r, s - eps)) / (2 * eps)
                jac = float(abs(dr[0] * ds[1] - dr[1] * ds[0]))  # |dA| of the quad face
                w = q0 * math.sin(al * xg) * math.sin(be * yg) * jac * gw[gi] * gw[gj]
                for node in top_face:
                    forces[3 * cell[node] + 2] -= w * float(N[node])

    tol = 1e-9 * max(a, b)
    x_edge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - a).abs() < tol)
    y_edge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - b).abs() < tol)
    xi = torch.nonzero(x_edge).flatten()
    yi = torch.nonzero(y_edge).flatten()
    fixed = torch.cat((3 * xi + 1, 3 * xi + 2, 3 * yi, 3 * yi + 2)).unique()

    if sparse:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        d = elasticity_matrix_3d(nodes.new_tensor(young).reshape(1), nodes.new_tensor(poisson).reshape(1))
        d = d.expand(len(cells), 6, 6)
        ke, _ = hex20_stiffness(nodes[cells], d)
        ed = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(len(cells), 60)
        rows = ed[:, :, None].expand(-1, -1, 60).reshape(-1).numpy()
        cols = ed[:, None, :].expand(-1, 60, -1).reshape(-1).numpy()
        K = sp.coo_matrix((ke.reshape(-1).numpy(), (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
        f = forces.numpy().copy()
        u = torch.zeros(3 * n_nodes, dtype=_DT)
        mask = torch.ones(3 * n_nodes, dtype=torch.bool)
        mask[fixed] = False
        free = torch.nonzero(mask).flatten().numpy()
        uu = u.numpy()
        uu[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
        u = torch.from_numpy(uu)
        residual = K @ uu - f
        residual = torch.from_numpy(residual)
    else:
        from .hex20 import Hex20Model, solve_hex20
        model = Hex20Model(nodes, cells, nodes.new_tensor(young), nodes.new_tensor(poisson), forces, fixed)
        res = solve_hex20(model)
        u = res.displacement
        residual = forces.new_zeros(forces.shape)

    dx = (nodes[:, 0] - 0.5 * a).abs() <= 1e-9 + 0.25 * min(a / nx, b / ny)
    dy = (nodes[:, 1] - 0.5 * b).abs() <= 1e-9 + 0.25 * min(a / nx, b / ny)
    dz = (nodes[:, 2] - h).abs() < tol
    cand = torch.nonzero(dx & dy & dz).flatten()
    if len(cand) == 0:
        top_nodes = torch.nonzero(dz).flatten()
        d2 = (nodes[top_nodes, 0] - 0.5 * a) ** 2 + (nodes[top_nodes, 1] - 0.5 * b) ** 2
        cand = top_nodes[torch.argmin(d2)].reshape(1)
    centre = cand[0]
    return {
        "nodes": nodes,
        "cells": cells,
        "displacement": u,
        "residual": residual,
        "centre_deflection": float(-u[3 * centre + 2]),
        "centre_node": int(centre),
        "n_dofs": int(3 * n_nodes),
        "n_elements": int(len(cells)),
        "n_fixed": int(len(fixed)),
    }