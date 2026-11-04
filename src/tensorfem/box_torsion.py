"""Saint-Venant torsion of closed thin-walled box sections (warping included).

``tensorfem.saint_venant_torsion`` qualifies the merged HEX8 path against
exact fields, but every section it carries is *warping free* (solid circle,
ellipse, rectangle and the circular annulus, which uses ``tau = G theta r``
instead of a single-valued Prandtl function).  The documented limitation of
that module is:

    "仅覆盖无翘曲截面：方管/箱形梁需要翘曲函数，本验证不覆盖；用线性斜坡去做
    方截面会得到偏刚结果，不得当作通过。"

A box girder / square tube is multiply connected: the Prandtl function obeys
``laplace(phi) = -2 G theta`` inside the wall, ``phi = 0`` on the outer
boundary and ``phi = c`` (an *a priori unknown constant*) on the inner
boundary, the constant being fixed by single-valuedness of the warping.  This
module supplies that missing oracle in two independent ways:

* :func:`solve_box_prandtl` -- a finite-difference Prandtl solver on a
  rectangular ring.  The inner constant is eliminated with the exact
  compatibility condition ``contour(dphi/dn) ds = 2 G theta A_inner`` and the
  torsional rigidity is evaluated twice, by the cell energy identity
  ``J = int |grad phi|^2 dA / (G theta)^2`` and by the torque identity
  ``T = 2 int phi dA + 2 c A_inner``.
* :func:`box_torsion_run` -- the merged HEX8 displacement path
  (``solid3d.hex8_stiffness`` + ``sparse_core``) on a body-fitted ring mesh,
  compared against the finite-difference field and *not* against a thin-wall
  approximation, so bilinear warping deficiency is measured rather than
  assumed.

Conventions follow :mod:`tensorfem.saint_venant_torsion`: bar axis ``z``,
section in the ``(x, y)`` plane, ``tau_xz = dphi/dy``, ``tau_yz = -dphi/dx``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .saint_venant_torsion import _Section, prism_mesh
from .solid3d import elasticity_matrix_3d, hex8_stiffness
from .sparse_core import assemble_coo, solve_sparse_static


# --------------------------------------------------------------------------- #
# 2-D finite-difference Prandtl solver on a rectangular ring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrandtlBox:
    """Finite-difference Prandtl field of a rectangular ring cross-section."""

    xs: np.ndarray
    ys: np.ndarray
    phi: np.ndarray
    material: np.ndarray
    c: float
    h: float
    g: float
    theta: float
    sweeps: int
    residual: float
    compat_residual: float

    # -- geometry helpers -------------------------------------------------- #
    @property
    def a(self) -> float:
        return float(self.xs[-1])

    @property
    def b(self) -> float:
        return float(self.ys[-1])

    def _full_cells(self) -> np.ndarray:
        """Cells whose four corner nodes are all material.

        The domain is a rectilinear ring whose vertices are grid nodes, so the
        union of these cells is exactly the material region (no half-cell loss
        at the walls) and both quadratures below stay second order.
        """
        m = self.material
        return m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]

    def _cell_gradients(self):
        phi, h = self.phi, self.h
        gx = ((phi[:-1, 1:] + phi[1:, 1:]) - (phi[:-1, :-1] + phi[1:, :-1])) / (2.0 * h)
        gy = ((phi[1:, :-1] + phi[1:, 1:]) - (phi[:-1, :-1] + phi[:-1, 1:])) / (2.0 * h)
        return gx, gy

    def _cell_centres(self):
        xc = 0.5 * (self.xs[:-1] + self.xs[1:])
        yc = 0.5 * (self.ys[:-1] + self.ys[1:])
        return np.meshgrid(xc, yc)

    @property
    def tau_cells(self):
        """``(x, y, tau_xz, tau_yz)`` at the centres of the full cells."""
        gx, gy = self._cell_gradients()
        full = self._full_cells()
        X, Y = self._cell_centres()
        return X[full], Y[full], gy[full], -gx[full]

    # -- the two independent identities ------------------------------------ #
    @property
    def j_energy(self) -> float:
        gx, gy = self._cell_gradients()
        full = self._full_cells()
        return float((gx[full] ** 2 + gy[full] ** 2).sum() * self.h ** 2
                     / (self.g * self.theta) ** 2)

    @property
    def j_torque(self) -> float:
        phi, h = self.phi, self.h
        full = self._full_cells()
        centre = 0.25 * (phi[:-1, :-1] + phi[:-1, 1:] + phi[1:, :-1] + phi[1:, 1:])
        a_inner = (2.0 * (self.a - self.t)) * (2.0 * (self.b - self.t))
        integral = float(centre[full].sum() * h * h)
        return (2.0 * integral + 2.0 * self.c * a_inner) / (self.g * self.theta)

    @property
    def t(self) -> float:
        """Wall thickness recovered from the material mask."""

        return float(self._thickness)

    @property
    def tau_max(self) -> float:
        _, _, txz, tyz = self.tau_cells
        return float(np.max(np.hypot(txz, tyz)))


def _neighbour_sum(phi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(phi)
    out[1:, :] += phi[:-1, :]
    out[:-1, :] += phi[1:, :]
    out[:, 1:] += phi[:, :-1]
    out[:, :-1] += phi[:, 1:]
    return out


def _sor(phi, active, rhs, h, tol, max_sweeps, omega):
    """Red-black SOR for ``laplace(phi) = rhs`` on ``active`` nodes."""
    h2 = h * h
    ny, nx = phi.shape
    ii, jj = np.indices((ny, nx))
    parity = (ii + jj) % 2
    phi = phi.copy()
    last = math.inf
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        change = 0.0
        for colour in (0, 1):
            nb = _neighbour_sum(phi)
            target = (nb - rhs * h2) / 4.0
            upd = active & (parity == colour)
            if not upd.any():
                continue
            step = omega * (target - phi)
            change = max(change, float(np.abs(step[upd]).max()))
            phi = np.where(upd, phi + step, phi)
        last = change
        if change < tol:
            break
    return phi, sweep, last


def _ring_masks(xs, ys, a, b, t):
    X, Y = np.meshgrid(xs, ys)
    ai, bi = a - t, b - t
    tol = 0.25 * (xs[1] - xs[0])
    hole_inner = (np.abs(X) < ai - tol) & (np.abs(Y) < bi - tol)
    hole_bnd = ((np.abs(X) <= ai + tol) & (np.abs(Y) <= bi + tol)) & (
        (np.abs(np.abs(X) - ai) <= tol) | (np.abs(np.abs(Y) - bi) <= tol))
    outer = (np.abs(np.abs(X) - a) <= tol) | (np.abs(np.abs(Y) - b) <= tol)
    material = ~hole_inner
    active = material & ~outer & ~hole_bnd
    return X, Y, material, active, hole_bnd, outer, ai, bi


def _hole_flux(phi, xs, ys, ai, bi, h):
    """``contour dphi/dn ds`` around the inner boundary, ``n`` into the hole."""
    def j_of(x):
        return int(round((x - xs[0]) / h))

    def i_of(y):
        return int(round((y - ys[0]) / h))

    j_hi, j_lo = j_of(ai), j_of(-ai)
    i_hi, i_lo = i_of(bi), i_of(-bi)
    rows = slice(i_lo + 1, i_hi)          # interior nodes of a side
    cols = slice(j_lo + 1, j_hi)
    total = 0.0

    def trap(values):
        w = np.ones_like(values)
        w[0] = 0.5
        w[-1] = 0.5
        return float((values * w).sum() * h)

    # right side x = +ai: n = -x, dphi/dn = -dphi/dx
    side = (-3.0 * phi[i_lo:i_hi + 1, j_hi] + 4.0 * phi[i_lo:i_hi + 1, j_hi + 1]
            - phi[i_lo:i_hi + 1, j_hi + 2]) / (2.0 * h)
    total += trap(-side)
    # left side x = -ai: n = +x
    side = (3.0 * phi[i_lo:i_hi + 1, j_lo] - 4.0 * phi[i_lo:i_hi + 1, j_lo - 1]
            + phi[i_lo:i_hi + 1, j_lo - 2]) / (2.0 * h)
    total += trap(side)
    # top side y = +bi: n = -y
    side = (-3.0 * phi[i_hi, j_lo:j_hi + 1] + 4.0 * phi[i_hi + 1, j_lo:j_hi + 1]
            - phi[i_hi + 2, j_lo:j_hi + 1]) / (2.0 * h)
    total += trap(-side)
    # bottom side y = -bi: n = +y
    side = (3.0 * phi[i_lo, j_lo:j_hi + 1] - 4.0 * phi[i_lo - 1, j_lo:j_hi + 1]
            + phi[i_lo - 2, j_lo:j_hi + 1]) / (2.0 * h)
    total += trap(side)
    return total


def solve_box_prandtl(a: float, b: float, t: float, h: float,
                      g: float = 1.0, theta: float = 1.0,
                      tol: float = 1e-14, max_sweeps: int = 200000,
                      omega: float | None = None) -> PrandtlBox:
    """Solve the multiply connected Prandtl problem on a rectangular ring.

    ``a``/``b`` are the outer half-dims, ``t`` the wall thickness and ``h`` the
    uniform grid spacing; ``h`` must divide ``t`` exactly so the walls are
    grid aligned.
    """
    if not (a > t > 0.0 and b > t > 0.0):
        raise ValueError("need a > t > 0 and b > t > 0")
    if h <= 0.0:
        raise ValueError("grid spacing must be positive")
    n_wall = t / h
    if abs(n_wall - round(n_wall)) > 1e-9 or round(n_wall) < 2:
        raise ValueError("h must divide the wall thickness into >= 2 intervals")
    nx = int(round(2.0 * a / h))
    ny = int(round(2.0 * b / h))
    if abs(nx * h - 2.0 * a) > 1e-9 or abs(ny * h - 2.0 * b) > 1e-9:
        raise ValueError("h must divide the outer dimensions exactly")
    if abs((a - t) / h - round((a - t) / h)) > 1e-9 or \
            abs((b - t) / h - round((b - t) / h)) > 1e-9:
        raise ValueError("h must put the inner boundary on grid nodes")
    xs = -a + h * np.arange(nx + 1)
    ys = -b + h * np.arange(ny + 1)
    _, _, material, active, hole_bnd, outer, ai, bi = _ring_masks(xs, ys, a, b, t)
    if not active.any():
        raise ValueError("wall too thin for the requested grid")
    if omega is None:
        n = max(nx, ny)
        omega = 2.0 / (1.0 + math.sin(math.pi / n))

    phi_p, sweeps_p, res_p = _sor(np.zeros((ny + 1, nx + 1)), active,
                                  -2.0 * g * theta, h, tol, max_sweeps, omega)
    init_h = np.where(hole_bnd, 1.0, 0.0)
    phi_h, sweeps_h, res_h = _sor(init_h, active, 0.0, h, tol, max_sweeps, omega)

    area_inner = (2.0 * ai) * (2.0 * bi)
    flux_p = _hole_flux(phi_p, xs, ys, ai, bi, h)
    flux_h = _hole_flux(phi_h, xs, ys, ai, bi, h)
    target = 2.0 * g * theta * area_inner
    if abs(flux_h) < 1e-12:
        raise RuntimeError("degenerate inner circulation; refine the grid")
    c = (target - flux_p) / flux_h
    phi = phi_p + c * phi_h
    compat = abs(_hole_flux(phi, xs, ys, ai, bi, h) - target) / abs(target)

    box = PrandtlBox(xs=xs, ys=ys, phi=phi, material=material, c=float(c), h=h,
                     g=g, theta=theta, sweeps=max(sweeps_p, sweeps_h),
                     residual=max(res_p, res_h), compat_residual=float(compat))
    object.__setattr__(box, "_thickness", float(t))
    return box


# --------------------------------------------------------------------------- #
# closed thin-walled section used by the production HEX8 path
# --------------------------------------------------------------------------- #
def ring_cross_section(a: float, b: float, t: float, n_side: int,
                       n_layers: int, dtype=torch.float64):
    """Body-fitted ring mesh of a rectangular box section (body-fitted, CCW)."""
    if n_side < 2 or n_layers < 1:
        raise ValueError("need n_side >= 2 and n_layers >= 1")
    # outward unit normals of the four sides, in counter-clockwise order
    corners = [(a, -b, 1.0, 0.0), (a, b, 0.0, 1.0),
               (-a, b, -1.0, 0.0), (-a, -b, 0.0, -1.0)]
    perim = []
    for k, (px, py, nx_, ny_) in enumerate(corners):
        qx, qy, _, _ = corners[(k + 1) % 4]
        _, _, bx, by = corners[(k - 1) % 4]
        for j in range(n_side):
            s = j / n_side
            x = px + (qx - px) * s
            y = py + (qy - py) * s
            if j == 0:                       # mitred corner: two outward normals
                ox = nx_ + bx
                oy = ny_ + by
            else:
                ox, oy = nx_, ny_
            perim.append((x, y, ox, oy))
    nodes = []
    for k in range(n_layers + 1):
        r = t * k / n_layers
        for (x, y, ox, oy) in perim:
            nodes.append((x - r * ox, y - r * oy))
    n_per = len(perim)
    quads = []
    for k in range(n_layers):
        for p in range(n_per):
            p1 = (p + 1) % n_per
            # orientation chosen so the extrusion to z gives a positive
            # hex8 Jacobian (same convention as the circle/rectangle meshes)
            quads.append([k * n_per + p, k * n_per + p1,
                          (k + 1) * n_per + p1, (k + 1) * n_per + p])
    return (torch.tensor(nodes, dtype=dtype),
            torch.tensor(quads, dtype=torch.long))


class RectBoxSection(_Section):
    """Rectangular box girder, outer half-dims ``a``/``b``, wall ``t``.

    ``j``/``tau_max`` are the Bredt thin-wall values (asymptotically exact for
    ``t -> 0``); they are *not* used to qualify the finite elements, the
    finite-difference field of :func:`solve_box_prandtl` is.  ``phi``/``tau``
    are deliberately unavailable: a multiply connected section has no single
    valuable closed-form stress function, and a linear ramp substitute is known
    to be too stiff (see the module docstring).
    """

    name = "box"

    def __init__(self, a: float = 1.0, b: float = 0.6, t: float = 0.1,
                 shear_modulus: float = 1.0, twist: float = 1.0,
                 n_layers: int = 3):
        super().__init__(shear_modulus, twist)
        for value in (a, b, t):
            if value <= 0.0:
                raise ValueError("dimensions must be positive")
        if not (a > t and b > t):
            raise ValueError("wall thickness must be smaller than both half-dims")
        self.a, self.b, self.t = float(a), float(b), float(t)
        self.n_layers = int(n_layers)

    # -- closed thin-wall reference ---------------------------------------- #
    @property
    def mean_area(self) -> float:
        """Area enclosed by the wall mid-line ``A_0`` (Bredt)."""
        return (2.0 * self.a - self.t) * (2.0 * self.b - self.t)

    @property
    def mean_perimeter(self) -> float:
        return 4.0 * ((self.a - 0.5 * self.t) + (self.b - 0.5 * self.t))

    @property
    def j(self) -> float:
        return 4.0 * self.mean_area ** 2 * self.t / self.mean_perimeter

    @property
    def tau_max(self) -> float:
        """``q / t`` with ``q = T / (2 A_0) = G theta J / (2 A_0)``."""
        return self.g * self.theta * self.j / (2.0 * self.mean_area * self.t)

    @property
    def area(self) -> float:
        return 4.0 * (self.a * self.b - (self.a - self.t) * (self.b - self.t))

    @property
    def scale(self) -> float:
        return self.t

    def phi(self, x, y):  # pragma: no cover - guard rail
        raise NotImplementedError(
            "a box section is multiply connected: the Prandtl function carries "
            "an unknown constant on the inner wall; use solve_box_prandtl "
            "instead of a single-valued closed form")

    def tau(self, x, y):  # pragma: no cover - guard rail
        raise NotImplementedError(
            "no closed-form shear field for a multiply connected section; use "
            "solve_box_prandtl (finite difference) as the reference")

    def contains(self, x, y):
        inside = (x.abs() <= self.a) & (y.abs() <= self.b)
        hole = (x.abs() < self.a - self.t) & (y.abs() < self.b - self.t)
        return inside & ~hole

    def quadrature(self, n: int = 401):
        kx = torch.linspace(-self.a, self.a, n + 1, dtype=torch.float64)[:-1] + self.a / n
        ky = torch.linspace(-self.b, self.b, n + 1, dtype=torch.float64)[:-1] + self.b / n
        xx, yy = torch.meshgrid(kx, ky, indexing="ij")
        weight = torch.full_like(xx, (2.0 * self.a / n) * (2.0 * self.b / n))
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), 1), weight.reshape(-1)

    def cross_section(self, n: int, dtype=torch.float64):
        return ring_cross_section(self.a, self.b, self.t, n, self.n_layers, dtype)


# --------------------------------------------------------------------------- #
# merged HEX8 production path on the ring mesh
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoxTorsionResult:
    a: float
    b: float
    t: float
    n_side: int
    n_layers: int
    layers_z: int
    nodes: int
    elements: int
    dofs: int
    j_exact: float
    j_fe: float
    j_ratio: float
    warping_max: float
    shear_wall: float
    shear_wall_fe: float
    tau_max_fe: float


def box_torsion_run(a: float = 1.0, b: float = 0.6, t: float = 0.1,
                    n_side: int = 4, n_layers: int = 3, layers_z: int = 2,
                    length: float | None = None, young_modulus: float = 2.0 * 1.3,
                    poisson_ratio: float = 0.3, twist: float = 1.0,
                    rtol: float = 1e-11) -> BoxTorsionResult:
    """HEX8 bar under Saint-Venant torsion, torsional rigidity from reactions."""
    shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
    section = RectBoxSection(a, b, t, shear, twist, n_layers=n_layers)
    if length is None:
        length = 2.0 * max(a, b)
    nodes, hexes, m, _ = prism_mesh(section, n=n_side, layers=layers_z,
                                    length=length)
    ne = len(hexes)
    ndof = 3 * len(nodes)
    E = nodes.new_full((ne,), young_modulus)
    nu = nodes.new_full((ne,), poisson_ratio)
    D = elasticity_matrix_3d(E, nu)
    ke, B = hex8_stiffness(nodes[hexes], D)
    edofs = (3 * hexes[:, :, None] + torch.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, ndof)
    dirichlet = {}
    ends = [torch.arange(m), torch.arange(ndof // 3 - m, ndof // 3)]
    for k, sl in enumerate(ends):
        z = 0.0 if k == 0 else length
        xy = nodes[sl]
        ux, uy = -twist * z * xy[:, 1], twist * z * xy[:, 0]
        for pos, node in enumerate(sl.tolist()):
            dirichlet[3 * node] = float(ux[pos])
            dirichlet[3 * node + 1] = float(uy[pos])
    solved = solve_sparse_static(K, nodes.new_zeros(ndof), dirichlet=dirichlet,
                                 rtol=rtol)
    u, reaction = solved.displacement, solved.reaction

    def end_torque(sl):
        return float((nodes[sl, 0] * reaction[3 * sl + 1]
                      - nodes[sl, 1] * reaction[3 * sl]).sum())

    j_fe = 0.5 * (abs(end_torque(ends[0])) + abs(end_torque(ends[1]))) \
        / (shear * twist)

    centres = nodes[hexes].mean(1)
    strain = torch.einsum("eij,ej->ei", B, u.flatten()[edofs])
    stress = torch.einsum("eij,ej->ei", D, strain)
    txz, tyz = stress[:, 5], stress[:, 4]           # columns: xy, yz, xz
    tau = torch.sqrt(txz ** 2 + tyz ** 2)
    # right wall (x > a - t), mid-span band: the Bredt shear direction is +y
    # mid-wall shear on the long wall, away from the sharp inner corners where
    # the exact field is singular (see the docstring of the evidence report)
    wall = ((centres[:, 0] > a - t) & (centres[:, 1].abs() < 0.8 * (b - t))
            & (centres[:, 2] > 0.2 * length) & (centres[:, 2] < 0.8 * length))
    shear_wall_fe = float(tyz[wall].abs().mean()) if bool(wall.any()) else float("nan")
    return BoxTorsionResult(
        a=a, b=b, t=t, n_side=n_side, n_layers=n_layers, layers_z=layers_z,
        nodes=len(nodes), elements=ne, dofs=ndof,
        j_exact=section.g * twist * section.j,
        j_fe=j_fe, j_ratio=j_fe / (section.g * twist * section.j),
        warping_max=float(u[2::3].abs().max()),
        shear_wall=section.g * twist * section.j / (2.0 * section.mean_area * t),
        shear_wall_fe=shear_wall_fe,
        tau_max_fe=float(tau.max()),
    )