"""Saint-Venant torsion of multi-cell closed thin-walled sections (double hull).

``tensorfem.box_torsion`` closed the "no warping section" debt for a **single**
rectangular cell and, in its report (``docs/box-girder-torsion-validation.md``
§5), wrote down what it left open::

    未覆盖的仍是缺口：多闭室箱梁、加劲壁、圆角内角（奇异正则化）需要另立算例。

A ship midship section is a **multi-cell** closed section: deck + double bottom +
two side shells enclose at least two cells, and a centre girder or a
longitudinal bulkhead separates them.  The mechanics of that layout is *not* the
mechanics of one cell counted twice, because the internal wall is shared by both
cells and the shear flow in it is the **difference** of the two cell
circulations.

This module supplies the missing reference in three independent ways:

* :func:`solve_multicell_prandtl` -- a finite-difference Prandtl solver on a
  rectilinear ``n_cell``-hole ring.  Each hole carries its own unknown constant
  ``c_i``; the constants are eliminated with the exact single-valuedness
  conditions ``contour_i(dphi/dn) ds = 2 G theta A_i`` (``n`` into hole ``i``),
  i.e. one linear system of size ``n_cell``.  ``J`` is evaluated **twice**: from
  the cell energy identity ``J = int |grad phi|^2 dA / (G theta)^2`` and from the
  torque identity ``T = 2 int phi dA + 2 sum_i c_i A_i``.
* :func:`bredt_multicell` -- the classical thin-wall multi-cell (Bredt) theory:
  one compatibility equation per cell, ``sum_walls q_w l_w / t_w = 2 G theta A_i``
  with ``q_shared = q_i - q_j``, then ``T = 2 sum_i A_i q_i``.  It is the
  engineering formula the design office uses; this module quantifies where it
  stops being admissible (first order in ``t``, as for one cell).
* :func:`multicell_torsion_fem` -- the merged HEX8 displacement path
  (``solid3d.hex8_stiffness`` + ``sparse_core``) on a body-fitted block mesh of
  the same section, so the element's warping deficiency on a multi-connected
  section is measured instead of assumed.

Conventions follow :mod:`tensorfem.box_torsion`: bar axis ``z``, section in the
``(x, y)`` plane, ``tau_xz = dphi/dy``, ``tau_yz = -dphi/dx``, material
``G = E/(2 (1+nu))``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .solid3d import elasticity_matrix_3d, hex8_stiffness
from .sparse_core import assemble_coo, solve_sparse_static


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RectMultiCellSection:
    """Rectilinear multi-cell section: outer rectangle minus ``n_cell`` holes.

    ``a``/``b`` are the **outer** half-dims, ``wall`` the uniform outer wall
    thickness and ``webs`` a tuple of ``(x_centre, thickness)`` vertical internal
    walls running the full inner height.  Cell ``j`` is the compartment between
    web ``j-1`` and web ``j``.  Corners are square (no fillets), so the inner
    corners are re-entrant and the shear stress is singular there exactly as in
    the single-cell case; ``J`` (an integral) is unaffected, pointwise
    ``tau_max`` is not a usable acceptance quantity.
    """

    a: float
    b: float
    wall: float
    webs: tuple = ()

    def __post_init__(self) -> None:
        if not (self.a > self.wall > 0.0 and self.b > self.wall > 0.0):
            raise ValueError("need a > wall > 0 and b > wall > 0")
        prev = -math.inf
        for x, t in self.webs:
            if t <= 0.0:
                raise ValueError("web thickness must be positive")
            if not (x > prev):
                raise ValueError("webs must be sorted by x and distinct")
            if abs(x) + 0.5 * t >= self.a - self.wall:
                raise ValueError("web overlaps the outer wall")
            prev = x

    # -- topology ---------------------------------------------------------- #
    @property
    def n_cells(self) -> int:
        return len(self.webs) + 1

    def holes(self):
        """``(x0, x1, y0, y1)`` of every compartment (the holes of the domain)."""
        xi, yi = self.a - self.wall, self.b - self.wall
        edges = [(-xi, xi)]
        for x, t in self.webs:
            left = edges[-1]
            edges[-1] = (left[0], x - 0.5 * t)
            edges.append((x + 0.5 * t, xi))
        return tuple((x0, x1, -yi, yi) for x0, x1 in edges)

    def cell_midlines(self):
        """``(x0, x1, y0, y1)`` of every cell on the **wall mid-lines**.

        Bredt theory lives on the mid-line: an outer wall contributes ``a - t/2``
        and an internal web contributes its centre line ``x``.
        """
        xm, ym = self.a - 0.5 * self.wall, self.b - 0.5 * self.wall
        edges = [(-xm, xm)]
        for x, _t in self.webs:
            left = edges[-1]
            edges[-1] = (left[0], x)
            edges.append((x, xm))
        return tuple((x0, x1, -ym, ym) for x0, x1 in edges)

    @property
    def area(self) -> float:
        """Material area of the section."""
        return 4.0 * self.a * self.b - sum(
            (x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in self.holes())

    def grid(self, h: float):
        """Uniform grid that puts **every** wall face on grid nodes."""
        if h <= 0.0:
            raise ValueError("grid spacing must be positive")
        nx = int(round(2.0 * self.a / h))
        ny = int(round(2.0 * self.b / h))
        if abs(nx * h - 2.0 * self.a) > 1e-9 or abs(ny * h - 2.0 * self.b) > 1e-9:
            raise ValueError("h must divide the outer dimensions exactly")
        xs = -self.a + h * np.arange(nx + 1)
        ys = -self.b + h * np.arange(ny + 1)
        for x0, x1, y0, y1 in self.holes():
            for value in (x0, x1):
                if abs((value - xs[0]) / h - round((value - xs[0]) / h)) > 1e-9:
                    raise ValueError("h must put the hole faces on grid nodes")
            for value in (y0, y1):
                if abs((value - ys[0]) / h - round((value - ys[0]) / h)) > 1e-9:
                    raise ValueError("h must put the hole faces on grid nodes")
        return xs, ys

    def contains(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inside = (x.abs() <= self.a) & (y.abs() <= self.b)
        for x0, x1, y0, y1 in self.holes():
            inside &= ~((x > x0) & (x < x1) & (y > y0) & (y < y1))
        return inside


# --------------------------------------------------------------------------- #
# finite-difference Prandtl field, one unknown constant per hole
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MultiCellPrandtl:
    """Finite-difference Prandtl field of a rectilinear multi-cell section."""

    section: RectMultiCellSection
    xs: np.ndarray
    ys: np.ndarray
    phi: np.ndarray
    material: np.ndarray
    constants: np.ndarray
    hole_areas: np.ndarray
    h: float
    g: float
    theta: float
    sweeps: int
    residual: float
    compat_residual: float

    # -- integration helpers ------------------------------------------------ #
    def _full_cells(self) -> np.ndarray:
        """Cells whose four corner nodes are all material (covers the region)."""
        m = self.material
        return m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]

    def _cell_grad(self):
        """Cell-centred ``(dphi/dx, dphi/dy)``: mean of the two edge differences."""
        ph = self.phi
        gx = 0.5 * ((ph[:-1, 1:] - ph[:-1, :-1]) + (ph[1:, 1:] - ph[1:, :-1])) / self.h
        gy = 0.5 * ((ph[1:, :-1] - ph[:-1, :-1]) + (ph[1:, 1:] - ph[:-1, 1:])) / self.h
        return gx, gy

    @property
    def j_energy(self) -> float:
        gx, gy = self._cell_grad()
        full = self._full_cells()
        return float((gx ** 2 + gy ** 2)[full].sum() * self.h ** 2
                     / (self.g * self.theta) ** 2)

    @property
    def j_torque(self) -> float:
        full = self._full_cells()
        phi_c = 0.25 * (self.phi[:-1, :-1] + self.phi[1:, :-1]
                        + self.phi[:-1, 1:] + self.phi[1:, 1:])
        integral = float(phi_c[full].sum() * self.h ** 2)
        torque = 2.0 * integral + 2.0 * float((self.constants * self.hole_areas).sum())
        return torque / (self.g * self.theta)

    @property
    def j(self) -> float:
        """``J`` from the torque identity (the primary of the two)."""
        return self.j_torque

    @property
    def tau_cells(self):
        """``(x, y, tau_xz, tau_yz)`` at the centres of the full cells."""
        gx, gy = self._cell_grad()
        full = self._full_cells()
        xc = 0.5 * (self.xs[:-1] + self.xs[1:])
        yc = 0.5 * (self.ys[:-1] + self.ys[1:])
        X, Y = np.meshgrid(xc, yc)
        return X[full], Y[full], gy[full], -gx[full]

    @property
    def warping_span(self) -> float:
        """``max(phi) - min(phi)`` over the material (warping amplitude)."""
        return float(self.phi[self.material].max() - self.phi[self.material].min())


def _masks(section: RectMultiCellSection, xs: np.ndarray, ys: np.ndarray):
    h = xs[1] - xs[0]
    tol = 0.25 * h
    X, Y = np.meshgrid(xs, ys)
    outer_bnd = (np.abs(np.abs(X) - section.a) <= tol) | \
                (np.abs(np.abs(Y) - section.b) <= tol)
    material = (np.abs(X) <= section.a + tol) & (np.abs(Y) <= section.b + tol)
    hole_bnd = []
    for x0, x1, y0, y1 in section.holes():
        # the compartment is a hole: neither material nor an unknown of the PDE
        material &= ~((X > x0 + tol) & (X < x1 - tol)
                      & (Y > y0 + tol) & (Y < y1 - tol))
        hole_bnd.append((((np.abs(Y - y0) <= tol) | (np.abs(Y - y1) <= tol))
                         & (X >= x0 - tol) & (X <= x1 + tol))
                        | (((np.abs(X - x0) <= tol) | (np.abs(X - x1) <= tol))
                           & (Y >= y0 - tol) & (Y <= y1 + tol)))
    bnd = outer_bnd.copy()
    for mask in hole_bnd:
        bnd |= mask
    active = material & ~bnd
    if not active.any():
        raise ValueError("walls too thin for the requested grid")
    return X, Y, material, active, hole_bnd, outer_bnd


def _sor(phi: np.ndarray, active: np.ndarray, rhs: float, h: float,
         tol: float, max_sweeps: int, omega: float):
    """Red-black SOR on the interior nodes; Dirichlet values sit in ``phi``."""
    inner = active[1:-1, 1:-1]
    ii, jj = np.indices(inner.shape)
    red = inner & ((ii + jj) % 2 == 0)
    black = inner & ((ii + jj) % 2 == 1)
    view = phi[1:-1, 1:-1]
    change = math.inf
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        change = 0.0
        for mask in (red, black):
            # laplace(phi) = rhs  =>  phi = (sum(neighbours) - rhs h^2)/4
            laplace = 0.25 * (phi[:-2, 1:-1] + phi[2:, 1:-1]
                              + phi[1:-1, :-2] + phi[1:-1, 2:] - rhs * h * h)
            step = omega * (laplace - view)
            np.copyto(view, view + np.where(mask, step, 0.0))
            if mask.any():
                change = max(change, float(np.abs(step[mask]).max()))
        if change < tol:
            break
    return phi, sweep, change


def _hole_flux(phi: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               rect, h: float) -> float:
    """``contour dphi/dn ds`` around one hole, ``n`` pointing into the hole."""
    x0, x1, y0, y1 = rect

    def j_of(x):
        return int(round((x - xs[0]) / h))

    def i_of(y):
        return int(round((y - ys[0]) / h))

    j_lo, j_hi = j_of(x0), j_of(x1)
    i_lo, i_hi = i_of(y0), i_of(y1)

    def trap(values):
        w = np.ones_like(values)
        w[0] = 0.5
        w[-1] = 0.5
        return float((values * w).sum() * h)

    rows = slice(i_lo, i_hi + 1)
    cols = slice(j_lo, j_hi + 1)
    total = 0.0
    # right side x = x1: n = -x
    side = (-3.0 * phi[rows, j_hi] + 4.0 * phi[rows, j_hi + 1]
            - phi[rows, j_hi + 2]) / (2.0 * h)
    total += trap(-side)
    # left side x = x0: n = +x
    side = (3.0 * phi[rows, j_lo] - 4.0 * phi[rows, j_lo - 1]
            + phi[rows, j_lo - 2]) / (2.0 * h)
    total += trap(side)
    # top side y = y1: n = -y
    side = (-3.0 * phi[i_hi, cols] + 4.0 * phi[i_hi + 1, cols]
            - phi[i_hi + 2, cols]) / (2.0 * h)
    total += trap(-side)
    # bottom side y = y0: n = +y
    side = (3.0 * phi[i_lo, cols] - 4.0 * phi[i_lo - 1, cols]
            + phi[i_lo - 2, cols]) / (2.0 * h)
    total += trap(side)
    return total


def solve_multicell_prandtl(section: RectMultiCellSection, h: float,
                            g: float = 1.0, theta: float = 1.0,
                            tol: float = 1e-13, max_sweeps: int = 200000,
                            omega: float | None = None) -> MultiCellPrandtl:
    """Solve ``laplace(phi) = -2 G theta`` on the multi-cell ring.

    One Poisson solve plus ``n_cell`` harmonic solves superposed as
    ``phi = phi_p + sum_i c_i phi_i``; the constants follow from the exact
    circulation conditions.  Cost is linear in ``n_cell``.
    """
    xs, ys = section.grid(h)
    _X, _Y, material, active, hole_bnd, _ = _masks(section, xs, ys)
    nx, ny = len(xs) - 1, len(ys) - 1
    if omega is None:
        omega = 2.0 / (1.0 + math.sin(math.pi / max(nx, ny)))

    phi_p, sweeps, res = _sor(np.zeros((ny + 1, nx + 1)), active,
                              -2.0 * g * theta, h, tol, max_sweeps, omega)
    basis = []
    for mask in hole_bnd:
        init = np.where(mask, 1.0, 0.0)
        phi_k, sw, rs = _sor(init, active, 0.0, h, tol, max_sweeps, omega)
        basis.append(phi_k)
        sweeps = max(sweeps, sw)
        res = max(res, rs)

    holes = section.holes()
    areas = np.array([(x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in holes])
    n = len(holes)
    matrix = np.zeros((n, n))
    rhs = np.zeros(n)
    for i, rect in enumerate(holes):
        rhs[i] = 2.0 * g * theta * areas[i] - _hole_flux(phi_p, xs, ys, rect, h)
        for j, field in enumerate(basis):
            matrix[i, j] = _hole_flux(field, xs, ys, rect, h)
    constants = np.linalg.solve(matrix, rhs)
    phi = phi_p.copy()
    for c, field in zip(constants, basis):
        phi = phi + c * field

    worst = 0.0
    for i, rect in enumerate(holes):
        target = 2.0 * g * theta * areas[i]
        worst = max(worst, abs(_hole_flux(phi, xs, ys, rect, h) - target) / target)

    return MultiCellPrandtl(section=section, xs=xs, ys=ys, phi=phi,
                            material=material, constants=constants,
                            hole_areas=areas, h=h, g=g, theta=theta,
                            sweeps=sweeps, residual=float(res),
                            compat_residual=float(worst))


# --------------------------------------------------------------------------- #
# classical thin-wall multi-cell (Bredt) theory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BredtMultiCell:
    """Thin-wall multi-cell solution: cell flows, wall flows and ``J``."""

    section: RectMultiCellSection
    cell_areas: tuple
    cell_flows: tuple
    walls: tuple
    g: float
    theta: float

    @property
    def j(self) -> float:
        return 2.0 * float(sum(a * q for a, q in
                               zip(self.cell_areas, self.cell_flows))) \
            / (self.g * self.theta)

    def wall_flow(self, wall) -> float:
        """Net shear flow in a wall, along the first member cell's direction."""
        flow = 0.0
        for sign, cell in zip(wall["signs"], wall["cells"]):
            flow += sign * self.cell_flows[cell]
        return flow

    def wall_taus(self):
        return tuple(abs(self.wall_flow(w)) / w["thickness"] for w in self.walls)


def bredt_multicell(section: RectMultiCellSection, g: float = 1.0,
                    theta: float = 1.0) -> BredtMultiCell:
    """Classical multi-cell Bredt solution of a rectilinear section.

    One compatibility equation per cell,
    ``sum_walls q_w l_w / t_w = 2 G theta A_i``, with the shared-wall flow taken
    as the difference of the two cell circulations, then
    ``T = 2 sum_i A_i q_i``.  Wall self-torsion and corner effects are neglected,
    so ``j`` is the thin-wall lower bound discussed in the module docstring.
    """
    cells = section.cell_midlines()
    areas = tuple((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in cells)
    heights = tuple(y1 - y0 for _x0, _x1, y0, y1 in cells)
    wall = section.wall
    walls = []
    # outer vertical walls
    walls.append({"label": "outer_left", "length": heights[0], "thickness": wall,
                  "cells": (0,), "signs": (1.0,)})
    walls.append({"label": "outer_right", "length": heights[-1], "thickness": wall,
                  "cells": (section.n_cells - 1,), "signs": (1.0,)})
    # horizontal walls, two per cell
    for j, (x0, x1, _y0, _y1) in enumerate(cells):
        walls.append({"label": f"cell{j}_top", "length": x1 - x0,
                      "thickness": wall, "cells": (j,), "signs": (1.0,)})
        walls.append({"label": f"cell{j}_bottom", "length": x1 - x0,
                      "thickness": wall, "cells": (j,), "signs": (1.0,)})
    # internal webs, shared by two cells (opposite traversal directions)
    for k, (_x, t) in enumerate(section.webs):
        walls.append({"label": f"web{k}", "length": heights[k], "thickness": t,
                      "cells": (k, k + 1), "signs": (1.0, -1.0)})

    n = section.n_cells
    matrix = np.zeros((n, n))
    for w in walls:
        for a_sign, i in zip(w["signs"], w["cells"]):
            for b_sign, j in zip(w["signs"], w["cells"]):
                matrix[i, j] += a_sign * b_sign * w["length"] / w["thickness"]
    rhs = np.array([2.0 * g * theta * A for A in areas])
    flows = np.linalg.solve(matrix, rhs)
    return BredtMultiCell(section=section, cell_areas=areas,
                          cell_flows=tuple(float(q) for q in flows),
                          walls=tuple(walls), g=g, theta=theta)


def wall_band_taus(section: RectMultiCellSection, field: MultiCellPrandtl,
                   half_span: float = 0.5):
    """Mean ``|tau|`` within each wall, mid-span band (singular corners excluded).

    ``half_span`` is the fraction of the wall length kept around mid-span.
    """
    X, Y, txz, tyz = field.tau_cells
    tau = np.sqrt(txz ** 2 + tyz ** 2)
    a, b, t = section.a, section.b, section.wall
    yi = b - t
    out = {}
    band_y = np.abs(Y) < half_span * yi
    out["outer_left"] = float(np.mean(tau[(X < -a + t) & band_y])) \
        if np.any((X < -a + t) & band_y) else float("nan")
    out["outer_right"] = float(np.mean(tau[(X > a - t) & band_y])) \
        if np.any((X > a - t) & band_y) else float("nan")
    for j in range(section.n_cells):
        x0, x1, _y0, _y1 = section.cell_midlines()[j]
        mid = 0.5 * (x0 + x1)
        sel_x = np.abs(X - mid) < half_span * 0.5 * (x1 - x0)
        out[f"cell{j}_top"] = float(np.mean(tau[sel_x & (Y > yi - t)])) \
            if np.any(sel_x & (Y > yi - t)) else float("nan")
        out[f"cell{j}_bottom"] = float(np.mean(tau[sel_x & (Y < -yi + t)])) \
            if np.any(sel_x & (Y < -yi + t)) else float("nan")
    for k, (xw, tw) in enumerate(section.webs):
        sel = (np.abs(X - xw) < 0.5 * tw) & band_y
        out[f"web{k}"] = float(np.mean(tau[sel])) if np.any(sel) else float("nan")
    return out


# --------------------------------------------------------------------------- #
# merged HEX8 production path on a body-fitted block mesh
# --------------------------------------------------------------------------- #
def multicell_cross_section(section: RectMultiCellSection, h: float,
                            dtype=torch.float64):
    """Body-fitted quad mesh of the multi-cell material region.

    The material region is decomposed into five (plus one per web) rectangular
    blocks on a single uniform grid, so the union mesh is conforming without any
    hanging node.  Quads are counter-clockwise in the ``(x, y)`` plane.
    """
    a, b, t = section.a, section.b, section.wall
    yi = b - t
    blocks = [(-a, a, yi, b), (-a, a, -b, -yi), (-a, -a + t, -yi, yi),
              (a - t, a, -yi, yi)]
    for x, tw in section.webs:
        blocks.append((x - 0.5 * tw, x + 0.5 * tw, -yi, yi))

    nodes = []
    index = {}

    def node_id(xv, yv):
        key = (round(float(xv), 9), round(float(yv), 9))
        got = index.get(key)
        if got is None:
            got = len(nodes)
            index[key] = got
            nodes.append((key[0], key[1]))
        return got

    quads = []
    for x0, x1, y0, y1 in blocks:
        nux = (x1 - x0) / h
        nuy = (y1 - y0) / h
        if abs(nux - round(nux)) > 1e-9 or abs(nuy - round(nuy)) > 1e-9:
            raise ValueError("h must divide every block dimension")
        nux, nuy = int(round(nux)), int(round(nuy))
        ids = [[node_id(x0 + i * h, y0 + j * h) for i in range(nux + 1)]
               for j in range(nuy + 1)]
        for j in range(nuy):
            for i in range(nux):
                quads.append((ids[j][i], ids[j][i + 1],
                              ids[j + 1][i + 1], ids[j + 1][i]))
    xy = torch.tensor(nodes, dtype=dtype)
    return xy, torch.tensor(quads, dtype=torch.long)


def _extrude(xy: torch.Tensor, quads: torch.Tensor, length: float, layers: int):
    m = len(xy)
    zs = torch.linspace(0.0, length, layers + 1, dtype=xy.dtype)
    nodes = torch.cat([torch.stack((xy[:, 0], xy[:, 1], z.expand(m)), 1)
                       for z in zs], 0)
    offs = (torch.arange(layers, dtype=torch.long) * m)[:, None]
    hexes = torch.cat([quads + o for o in offs], 0)
    hexes = torch.cat([hexes, hexes + m], 1)
    return nodes, hexes, m


def multicell_torsion_fem(section: RectMultiCellSection, h: float,
                          layers_z: int = 2, length: float | None = None,
                          young_modulus: float = 2.0 * 1.3,
                          poisson_ratio: float = 0.3, twist: float = 1.0,
                          rtol: float = 1e-11) -> dict:
    """HEX8 bar under Saint-Venant torsion on the multi-cell section.

    End faces carry the exact transverse rotation ``u_x = -theta z y``,
    ``u_y = theta z x``; the warping ``u_z`` is free everywhere so the comparison
    tests the element, not a prescribed field.  ``J_fe`` comes from the end
    reactions (``T/(G theta)``), never from a pointwise stress integral.
    """
    if young_modulus <= 0.0 or not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("invalid elastic constants")
    shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
    if length is None:
        length = 2.0 * max(section.a, section.b)
    xy, quads = multicell_cross_section(section, h)
    nodes, hexes, m = _extrude(xy, quads, length, layers_z)
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
        xyz = nodes[sl]
        ux, uy = -twist * z * xyz[:, 1], twist * z * xyz[:, 0]
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
    tau = torch.sqrt(stress[:, 5] ** 2 + stress[:, 4] ** 2)
    return {"j_fe": float(j_fe), "nodes": int(len(nodes)), "elements": int(ne),
            "dofs": int(ndof), "layers_z": int(layers_z),
            "h": float(h), "length": float(length),
            "warping_max": float(u[2::3].abs().max()),
            "tau_max_fe": float(tau.max()),
            "centres": centres.detach().cpu().numpy(),
            "tau": tau.detach().cpu().numpy()}