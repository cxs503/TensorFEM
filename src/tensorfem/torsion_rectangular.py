"""Rectangular (``a != b``) Saint-Venant torsion: 3-D driver plus independent references.

Why this module exists
----------------------
:func:`tensorfem.torsion_bar.solve_torsion_bar` twists a **square** prism only: the
mesh, the end-face weights and the exact reference inside it all use the single
argument ``side``.  The closed-form series next to it
(:func:`tensorfem.torsion_bar.rectangular_torsion_constant`, ...) *is* written for a
general ``a x b`` section, but every published check of it -- including
``docs/saint-vanant-torsion.md`` and ``docs/torsion-boundary-stress.md`` --
evaluates it at ``a == b``.  ``docs/torsion-boundary-stress.md`` §5 records this
as an open item: the aspect-ratio (shape) dependence of both the torsion constant
and the peak shear stress is unverified.

This module closes that item from three directions that do not share code paths:

``rectangular_torsion_bar(a, b, ...)``
    3-D HEX8 driver for an arbitrary ``a x b`` prism (``a >= b``).  Same load case,
    constraints and twist-rate extraction as the square driver -- a least-squares
    slope of the section rotation over ``0.25 L .. 0.75 L`` -- but with distinct
    side lengths and per-element consistent face weights, so the assembled nodal
    forces carry exactly the requested torque.

``prandtl_finite_difference(a, b, ...)``
    Second-order finite-difference solution of the *defining* boundary-value problem
    ``lap(psi) = -2``, ``psi = 0`` on the rectangle, giving ``J = 2 int psi dA`` and
    ``|grad psi|`` at the two side midpoints.  A discretised PDE cannot inherit a
    sign or normalisation mistake from the Fourier series.

``classical_rectangle_coefficients()``
    Hand-book shape factors (Timoshenko & Goodier; Roark, "rectangular sections"),
    normalised as ``J = c1 a b^3`` and ``tau_max = T / (c2 a b^2)``.

The ``(c1, c2)`` normalisation is shared by the closed form via
:func:`shape_factors`, so the three sources can be compared entry by entry.

Conventions (identical to :mod:`tensorfem.torsion_bar`): ``x`` spans the short side
``b``, ``y`` spans the long side ``a``, ``tau_zx = dpsi/dy`` and ``tau_zy = -dpsi/dx``,
so the exact peak sits at the midpoint of a long side, ``(x, y) = (+-b/2, 0)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .solid3d import elasticity_matrix_3d, hex8_stiffness, structured_hex_mesh
from .sparse_core import assemble_coo, solve_sparse_static
from .torsion_bar import (max_shear_stress, rectangular_torsion_constant, torsion_shear_stress)

__all__ = [
    "ClassicalCoefficient",
    "FiniteDifferenceTorsion",
    "RectangularBarResult",
    "SurfacePeak",
    "ShapeFactors",
    "classical_rectangle_coefficients",
    "prandtl_finite_difference",
    "rectangular_torsion_bar",
    "richardson",
    "surface_peak_shear",
    "shape_factors",
    "thin_strip_torsion_constant",
]

_TERMS = 401


def _check_sides(a: float, b: float) -> None:
    if not (a > 0 and b > 0):
        raise ValueError("section sides must be positive")
    if b > a * (1.0 + 1e-12):
        raise ValueError("expects a >= b (long side first)")


# --------------------------------------------------------------------------- 3-D FE


@dataclass(frozen=True)
class RectangularBarResult:
    """Outcome of one ``a x b x length`` twist solve."""

    a: float
    b: float
    segments_short: int
    segments_long: int
    n_elements: int
    n_nodes: int
    n_dofs: int
    applied_torque: float
    twist_rate: float
    exact_twist_rate: float
    measured_torsion_constant: float
    exact_torsion_constant: float
    centroid_shear_l2_error: float
    peak_centroid_fe: float
    peak_series_sampled: float
    end_traction: str = "linear"
    end_traction_residual: float = 0.0

    @property
    def aspect(self) -> float:
        return self.a / self.b

    @property
    def torsion_constant_relative_error(self) -> float:
        return abs(self.measured_torsion_constant / self.exact_torsion_constant - 1.0)


def _hex_local_b(nodes_e: torch.Tensor, xi: float, eta: float, zeta: float) -> torch.Tensor:
    """Strain-displacement matrix of one HEX8 element at a natural coordinate.

    Convention of :func:`tensorfem.solid3d.hex8_stiffness` is kept (row 5 is
    ``d u_x/dz + d u_z/dx``, row 4 is ``d u_y/dz + d u_z/dy``), but evaluated *off*
    the centroid so wall values can be sampled and extrapolated.
    """
    signs = nodes_e.new_tensor(((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0),
                                (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
                                (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)))
    sx, sy, sz = signs[:, 0], signs[:, 1], signs[:, 2]
    dn = torch.stack((sx * (1.0 + sy * eta) * (1.0 + sz * zeta),
                      sy * (1.0 + sx * xi) * (1.0 + sz * zeta),
                      sz * (1.0 + sx * xi) * (1.0 + sy * eta)), 1) / 8.0
    jacobian = torch.einsum("ia,ib->ab", nodes_e, dn)
    if float(torch.linalg.det(jacobian)) <= 0.0:
        raise ValueError("degenerate HEX8 element in natural-coordinate sampling")
    gradient = dn @ torch.linalg.inv(jacobian)
    operator = torch.zeros((6, 24), dtype=nodes_e.dtype, device=nodes_e.device)
    operator[0, 0::3] = gradient[:, 0]
    operator[1, 1::3] = gradient[:, 1]
    operator[2, 2::3] = gradient[:, 2]
    operator[3, 0::3] = gradient[:, 1]
    operator[3, 1::3] = gradient[:, 0]
    operator[4, 1::3] = gradient[:, 2]
    operator[4, 2::3] = gradient[:, 1]
    operator[5, 0::3] = gradient[:, 2]
    operator[5, 2::3] = gradient[:, 0]
    return operator


_GAUSS_2 = 1.0 / 3.0**0.5


def _exact_face_forces(nodes: torch.Tensor, cells: torch.Tensor, *, a: float, b: float,
                       length: float, shear: float, twist_rate: float,
                       terms: int) -> torch.Tensor:
    """Nodal loads equivalent to the *Saint-Venant* shear traction on ``z = length``.

    The original driver spreads a circumferential linear traction ``t = c(-y, x)``.  Its
    resultant torque is right, its pointwise distribution is not the exact traction, so
    it excites a self-equilibrated end disturbance that decays over a length ``~ a``.
    For a slender section (``a`` comparable with the bar length) that disturbance has
    not decayed inside the measuring window -- the failure recorded in
    ``docs/torsion-rectangular-section-verification.md`` §5.1.

    Here the traction is ``(tau_zx, tau_zy) = G theta (dpsi/dy, -dpsi/dx)`` from the
    same closed-form series, integrated with a 2x2 Gauss rule (the ``A_e / 4`` rule used
    for the linear field is exact only for a linear traction).  ``theta`` is the exact
    twist rate ``T / (G J)``, so the load case *is* the Saint-Venant problem.
    """
    corners = (4, 5, 6, 7)
    face_nodes = []
    for cell in cells.tolist():
        if abs(float(nodes[cell[4], 2]) - length) <= 1e-12:
            face_nodes.append([cell[c] for c in corners])
    if not face_nodes:
        raise RuntimeError("no loaded-face elements found at z = length")
    face = torch.tensor(face_nodes, dtype=torch.long)
    xy = nodes[face][:, :, :2]                      # (n_face_elements, 4 corners, 2)

    forces = nodes.new_zeros(3 * len(nodes))
    gauss_points = ((-_GAUSS_2, -_GAUSS_2), (_GAUSS_2, -_GAUSS_2),
                    (_GAUSS_2, _GAUSS_2), (-_GAUSS_2, _GAUSS_2))
    for xi, eta in gauss_points:
        shape = nodes.new_tensor((0.25 * (1.0 - xi) * (1.0 - eta),
                                  0.25 * (1.0 + xi) * (1.0 - eta),
                                  0.25 * (1.0 + xi) * (1.0 + eta),
                                  0.25 * (1.0 - xi) * (1.0 + eta)))
        d_xi = nodes.new_tensor((-0.25 * (1.0 - eta), 0.25 * (1.0 - eta),
                                 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)))
        d_eta = nodes.new_tensor((-0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
                                  0.25 * (1.0 + xi), 0.25 * (1.0 - xi)))
        tangent_xi = torch.einsum("nic,i->nc", xy, d_xi)
        tangent_eta = torch.einsum("nic,i->nc", xy, d_eta)
        jacobian = tangent_xi[:, 0] * tangent_eta[:, 1] - tangent_xi[:, 1] * tangent_eta[:, 0]
        if float(jacobian.min()) <= 0.0:
            raise RuntimeError("loaded face element has non-positive in-plane Jacobian")
        points = torch.einsum("nic,i->nc", xy, shape)
        tx, ty = torsion_shear_stress(points[:, 0] - b / 2.0, points[:, 1] - a / 2.0, a, b,
                                      twist_rate=twist_rate, shear_modulus=shear, terms=terms)
        for corner in range(4):
            weight = shape[corner] * jacobian
            forces.index_add_(0, 3 * face[:, corner], weight * tx)
            forces.index_add_(0, 3 * face[:, corner] + 1, weight * ty)
    return forces


@dataclass(frozen=True)
class _PrismSolve:
    """Internal outcome of one twisted-prism solve, shared by both public entry points."""

    nodes: torch.Tensor
    cells: torch.Tensor
    displacement: torch.Tensor
    centroid_stress: torch.Tensor
    twist_rate: float
    applied_torque: float
    end_traction_residual: float
    n_short: int
    n_long: int
    nz: int
    planes: torch.Tensor
    thetas: torch.Tensor
    length: float

    def twist_rate_in(self, window: tuple[float, float]) -> float:
        """Least-squares section rotation rate over a slice of the bar.

        Exposed so a window-sensitivity study costs one solve instead of one per window.
        """
        inside = (self.planes >= window[0] * self.length) & (self.planes <= window[1] * self.length)
        if int(inside.sum()) < 2:
            raise ValueError("window contains fewer than two node planes")
        design = torch.stack((self.planes[inside],
                              torch.ones(int(inside.sum()), dtype=self.planes.dtype)), 1)
        return float(torch.linalg.lstsq(design, self.thetas[inside]).solution[0])


def _twist_prism(*, a: float, b: float, length: float, segments_short: int,
                 segments_long: int | None, young: float, poisson: float, torque: float,
                 window: tuple[float, float], terms: int, rtol: float,
                 end_traction: str) -> _PrismSolve:
    """Shared solve: mesh, end load, constraints, twist-rate regression, centroid stress."""
    _check_sides(a, b)
    if segments_short < 1:
        raise ValueError("segments_short must be >= 1")
    if end_traction not in ("linear", "exact"):
        raise ValueError("end_traction must be 'linear' or 'exact'")
    n_short = int(segments_short)
    n_long = int(round(segments_short * a / b)) if segments_long is None else int(segments_long)
    if n_long < 1:
        raise ValueError("segments_long must be >= 1")
    if not 0.0 <= window[0] < window[1] <= 1.0:
        raise ValueError("window must be an increasing sub-interval of [0, 1]")

    # Cubic elements, matching the square driver's nz = 4 * segments at length = 4 b:
    # the axial spacing tracks the section spacing instead of being tied to it.
    nz = max(1, int(round(length * n_short / b)))
    nodes, cells = structured_hex_mesh(b, a, length, n_short, n_long, nz)
    n_elements = len(cells)
    modulus = elasticity_matrix_3d(nodes.new_tensor(young).expand(n_elements),
                                   nodes.new_tensor(poisson).expand(n_elements))
    ke, b_matrix = hex8_stiffness(nodes[cells.long()], modulus)
    element_dofs = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(n_elements, 24).long()
    n_dofs = 3 * len(nodes)
    stiffness = assemble_coo(ke, element_dofs, n_dofs, csr=True)

    xc = nodes[:, 0] - b / 2.0
    yc = nodes[:, 1] - a / 2.0
    loading = torch.where(nodes[:, 2] == length)[0]

    shear = young / (2.0 * (1.0 + poisson))
    exact_j = rectangular_torsion_constant(a, b, terms=terms)
    exact_rate = torque / (shear * exact_j)

    if end_traction == "linear":
        # Consistent nodal weights of the loaded face: for a bilinear quad the weight of
        # every corner is A_e / 4, which integrates any linear traction exactly.  A_e is
        # taken per element, so the square path of `torsion_bar._face_weights` is recovered
        # as the special case n_short == n_long with b == a.
        weights = nodes.new_zeros(len(nodes))
        corners = (4, 5, 6, 7)
        for cell in cells.tolist():
            if abs(float(nodes[cell[4], 2]) - length) > 1e-12:
                continue
            face = [cell[c] for c in corners]
            xs, ys = nodes[face, 0], nodes[face, 1]
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            for node in face:
                weights[node] += area / 4.0

        forces = nodes.new_zeros(n_dofs)
        forces[3 * loading] = weights[loading] * (-yc[loading])
        forces[3 * loading + 1] = weights[loading] * xc[loading]
        polar = float(torch.sum(weights[loading] * (xc[loading] ** 2 + yc[loading] ** 2)))
        raw = forces * (torque / polar)
        residual = 0.0
    else:
        raw = _exact_face_forces(nodes, cells, a=a, b=b, length=length, shear=shear,
                                 twist_rate=exact_rate, terms=terms)

    applied = float(torch.sum(xc[loading] * raw[3 * loading + 1]
                              - yc[loading] * raw[3 * loading]))
    if applied <= 0.0:
        raise RuntimeError("end-face traction carries no torque")
    residual = abs(applied / torque - 1.0)
    forces = raw * (torque / applied)
    applied = float(torch.sum(xc[loading] * forces[3 * loading + 1]
                              - yc[loading] * forces[3 * loading]))
    if abs(applied - torque) > 1e-9 * abs(torque):
        raise RuntimeError("end-face traction does not carry the requested torque")

    root = torch.where(nodes[:, 2] == 0)[0]
    fixed = {int(dof): 0.0 for dof in torch.cat((3 * root, 3 * root + 1)).tolist()}
    fixed[3 * int(root[0]) + 2] = 0.0
    displacement = solve_sparse_static(stiffness, forces, dirichlet=fixed, rtol=rtol).displacement

    radius2 = xc**2 + yc**2
    planes = torch.unique(nodes[:, 2])
    thetas = nodes.new_zeros(len(planes))
    for index, z in enumerate(planes.tolist()):
        plane = torch.where(nodes[:, 2] == z)[0]
        thetas[index] = torch.sum(displacement[3 * plane] * (-yc[plane])
                                  + displacement[3 * plane + 1] * xc[plane]) / torch.sum(radius2[plane])
    strain = torch.einsum("eij,ej->ei", b_matrix, displacement[element_dofs])
    centroid_stress = torch.einsum("eij,ej->ei", modulus, strain)
    solve = _PrismSolve(nodes=nodes, cells=cells, displacement=displacement,
                        centroid_stress=centroid_stress, twist_rate=float("nan"),
                        applied_torque=applied, end_traction_residual=residual,
                        n_short=n_short, n_long=n_long, nz=nz, planes=planes, thetas=thetas,
                        length=length)
    twist_rate = solve.twist_rate_in(window)
    return _PrismSolve(nodes=nodes, cells=cells, displacement=displacement,
                       centroid_stress=centroid_stress, twist_rate=twist_rate,
                       applied_torque=applied, end_traction_residual=residual,
                       n_short=n_short, n_long=n_long, nz=nz, planes=planes, thetas=thetas,
                       length=length)


def rectangular_torsion_bar(*, a: float = 2.0, b: float = 1.0, length: float = 4.0,
                            segments_short: int = 3, segments_long: int | None = None,
                            young: float = 1.0e9, poisson: float = 0.3, torque: float = 1.0,
                            window: tuple[float, float] = (0.25, 0.75),
                            terms: int = _TERMS, rtol: float = 1e-12,
                            end_traction: str = "linear") -> RectangularBarResult:
    """Twist an ``a x b x length`` HEX8 prism with a pure end torque and measure it.

    ``segments_long`` defaults to ``round(segments_short * a / b)``, which keeps the
    elements square.  The far face ``z = length`` carries the end traction; the near
    face is clamped in-plane with one axial dof removed so warping stays free.

    ``end_traction="linear"`` is the original load case (numerics unchanged): the
    circumferential field ``t = c(-y, x)``.  ``end_traction="exact"`` spreads the
    Saint-Venant shear tractions themselves, which removes the self-equilibrated end
    disturbance and leaves pure discretisation error -- see
    ``docs/torsion-rectangular-end-traction.md``.
    """
    solve = _twist_prism(a=a, b=b, length=length, segments_short=segments_short,
                         segments_long=segments_long, young=young, poisson=poisson,
                         torque=torque, window=window, terms=terms, rtol=rtol,
                         end_traction=end_traction)
    nodes, cells = solve.nodes, solve.cells
    n_elements = len(cells)
    n_dofs = 3 * len(nodes)
    shear = young / (2.0 * (1.0 + poisson))
    exact_j = rectangular_torsion_constant(a, b, terms=terms)
    twist_rate = solve.twist_rate

    centroids = nodes[cells.long()].mean(1)
    hz = length / solve.nz
    slab = (centroids[:, 2] - length / 2.0).abs() < hz + 1e-12
    cx = centroids[slab, 0] - b / 2.0
    cy = centroids[slab, 1] - a / 2.0
    tx, ty = torsion_shear_stress(cx, cy, a, b, twist_rate=twist_rate, shear_modulus=shear,
                                  terms=terms)
    stress = solve.centroid_stress
    fe = torch.stack((stress[slab, 5], stress[slab, 4]), 1)  # tau_zx, tau_zy
    exact = torch.stack((tx, ty), 1)
    denominator = float(torch.sqrt(torch.sum(exact**2)))
    l2 = float(torch.sqrt(torch.sum((fe - exact) ** 2))) / denominator if denominator > 1e-9 else float("nan")

    peak_fe = float(torch.sqrt(fe[:, 0] ** 2 + fe[:, 1] ** 2).max())
    peak_sampled = float(torch.sqrt(exact[:, 0] ** 2 + exact[:, 1] ** 2).max())

    return RectangularBarResult(
        a=a, b=b, segments_short=solve.n_short, segments_long=solve.n_long,
        n_elements=n_elements, n_nodes=len(nodes), n_dofs=n_dofs,
        applied_torque=solve.applied_torque, twist_rate=twist_rate,
        exact_twist_rate=torque / (shear * exact_j),
        measured_torsion_constant=torque / (shear * twist_rate),
        exact_torsion_constant=exact_j, centroid_shear_l2_error=l2,
        peak_centroid_fe=peak_fe, peak_series_sampled=peak_sampled,
        end_traction=end_traction, end_traction_residual=solve.end_traction_residual,
    )


# ------------------------------------------------- rectangular boundary peak shear


@dataclass(frozen=True)
class SurfacePeak:
    """Wall shear at the two mid-side points of an ``a x b`` section.

    ``exact_*`` are the closed-form peaks ``G theta |grad psi|`` evaluated **at the
    twist rate the FE solve produced**, so a twist-rate error cannot masquerade as a
    stress-recovery error.  ``*_long`` is the long-side midpoint
    ``(|x|, y) = (b/2, 0)`` (the true peak), ``*_short`` the short-side midpoint
    ``(0, |y|) = (0, a/2)``.  Three recovery schemes from the same displacement field:

    ``direct_*``
        strain operator evaluated *on* the wall node for every attached element, then
        averaged (standard nodal post-processing).
    ``gauss_*``
        Barlow-style linear extrapolation to the wall from the two Gauss stations of
        the element adjacent to the wall.
    ``centroid_*``
        raw element-centroid shear half an element inside the wall (strict interior
        lower bound; the quantity the earlier report compared in L2).
    """

    a: float
    b: float
    segments_short: int
    segments_long: int
    n_dofs: int
    end_traction: str
    exact_long: float
    exact_short: float
    direct_long: float
    direct_short: float
    gauss_long: float
    gauss_short: float
    centroid_long: float
    centroid_short: float

    @property
    def aspect(self) -> float:
        return self.a / self.b

    def relative_error(self, scheme: str, side: str = "long") -> float:
        measured = getattr(self, f"{scheme}_{side}")
        exact = getattr(self, f"exact_{side}")
        return abs(measured / exact - 1.0)

    @property
    def long_over_short_exact(self) -> float:
        return self.exact_long / self.exact_short

    @property
    def long_over_short_direct(self) -> float:
        return self.direct_long / self.direct_short


def _element_stress(nodes: torch.Tensor, cells: torch.Tensor, displacement: torch.Tensor,
                    modulus: torch.Tensor, index: int,
                    natural: tuple[float, float, float]) -> torch.Tensor:
    """Stress tensor (6 components) of element ``index`` at one natural coordinate."""
    cell = cells[index]
    element_u = displacement[torch.stack(tuple(3 * cell + i for i in range(3)), 1).reshape(24)]
    operator = _hex_local_b(nodes[cell], *natural)
    return modulus[index] @ (operator @ element_u)


def surface_peak_shear(*, a: float = 3.0, b: float = 1.0, length: float = 6.0,
                       segments_short: int = 4, segments_long: int | None = None,
                       young: float = 1.0e9, poisson: float = 0.3, torque: float = 1.0,
                       window: tuple[float, float] = (0.25, 0.75), terms: int = _TERMS,
                       rtol: float = 1e-12, end_traction: str = "exact") -> SurfacePeak:
    """Recover the wall shear at both mid-side points of a rectangular section.

    The square-section flow (``docs/torsion-boundary-stress.md``) covers ``a == b``
    only; §5.2 of ``docs/torsion-rectangular-section-verification.md`` records the
    rectangular counterpart as an open item.  All three schemes are returned so the
    biased ones stay visible instead of being replaced silently.
    """
    _check_sides(a, b)
    n_long = int(round(segments_short * a / b)) if segments_long is None else int(segments_long)
    if n_long % 2 or int(segments_short) % 2:
        raise ValueError("surface_peak_shear needs even segment counts (a node sits on both symmetry planes)")
    solve = _twist_prism(a=a, b=b, length=length, segments_short=segments_short,
                         segments_long=n_long, young=young, poisson=poisson, torque=torque,
                         window=window, terms=terms, rtol=rtol, end_traction=end_traction)
    nodes, cells, displacement = solve.nodes, solve.cells, solve.displacement
    shear = young / (2.0 * (1.0 + poisson))
    modulus = elasticity_matrix_3d(nodes.new_tensor(young).expand(len(cells)),
                                   nodes.new_tensor(poisson).expand(len(cells)))

    hz = length / solve.nz
    centroids = nodes[cells.long()].mean(1)
    z_mid = float(torch.unique(nodes[:, 2])[int(torch.argmin(
        (torch.unique(nodes[:, 2]) - length / 2.0).abs()))])
    mid_slab = (centroids[:, 2] - z_mid).abs() < hz / 2.0 + 1e-9

    def nearest(target_xy: tuple[float, float]) -> int:
        distance = ((centroids[:, 0] - target_xy[0]) ** 2 + (centroids[:, 1] - target_xy[1]) ** 2)
        distance = torch.where(mid_slab, distance, torch.full_like(distance, float("inf")))
        return int(torch.argmin(distance))

    def magnitude(stress: torch.Tensor) -> float:
        return float(torch.sqrt(stress[5] ** 2 + stress[4] ** 2))

    def gap(scheme_natural: tuple[float, float, float], target_xy: tuple[float, float],
            axis: int) -> float:
        """Barlow linear extrapolation along ``axis`` to the wall from two Gauss stations."""
        index = nearest(target_xy)
        q = _GAUSS_2
        low = list(scheme_natural)
        high = list(scheme_natural)
        low[axis], high[axis] = -q, q
        values = [magnitude(_element_stress(nodes, cells, displacement, modulus, index, tuple(low))),
                  magnitude(_element_stress(nodes, cells, displacement, modulus, index, tuple(high)))]
        return values[1] + (values[1] - values[0]) * (1.0 - q) / (2.0 * q)

    def centroid_value(target_xy: tuple[float, float]) -> float:
        return magnitude(_element_stress(nodes, cells, displacement, modulus, nearest(target_xy),
                                        (0.0, 0.0, 0.0)))

    def direct(target_xy: tuple[float, float], wall: str) -> float:
        """Average the strain operator on the wall node over the attached mid-slab elements."""
        target = nodes.new_tensor((target_xy[0], target_xy[1], z_mid))
        node = int(torch.argmin(torch.linalg.norm(nodes - target, dim=1)))
        corners = nodes.new_tensor(((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0),
                                    (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
                                    (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)))
        total = 0.0
        count = 0
        for index, cell in enumerate(cells.tolist()):
            if node not in cell or not bool(mid_slab[index]):
                continue
            corner = cell.index(node)
            total += magnitude(_element_stress(nodes, cells, displacement, modulus, index,
                                               tuple(float(value) for value in corners[corner])))
            count += 1
        if count == 0:
            raise RuntimeError(f"no mid-slab element touches the {wall}-side wall node")
        return total / count

    factors = shape_factors(a, b, terms=terms)
    exact_long = factors.grad_long_mid * shear * solve.twist_rate
    exact_short = factors.grad_short_mid * shear * solve.twist_rate
    # The *long* sides are x = 0 and x = b (they carry the length ``a``); their midpoint
    # is (b, a/2), which is where the series puts the peak ``(b/2, 0)`` in centred
    # coordinates.  The short sides are y = 0 and y = a, midpoint (b/2, a).
    long_target = (b - b / solve.n_short / 2.0, a / 2.0)
    short_target = (b / 2.0, a - a / solve.n_long / 2.0)
    return SurfacePeak(
        a=a, b=b, segments_short=solve.n_short, segments_long=solve.n_long,
        n_dofs=3 * len(nodes), end_traction=end_traction,
        exact_long=exact_long, exact_short=exact_short,
        direct_long=direct((b, a / 2.0), "long"), direct_short=direct((b / 2.0, a), "short"),
        gauss_long=gap((0.0, 0.0, 0.0), long_target, axis=0),
        gauss_short=gap((0.0, 0.0, 0.0), short_target, axis=1),
        centroid_long=centroid_value(long_target), centroid_short=centroid_value(short_target),
    )

# ------------------------------------------------------- independent PDE reference


@dataclass(frozen=True)
class FiniteDifferenceTorsion:
    """Finite-difference solution of ``lap(psi) = -2`` on an ``a x b`` rectangle."""

    a: float
    b: float
    n_short: int
    n_long: int
    torsion_constant: float   # J = 2 int psi dA
    grad_long_mid: float      # |grad psi| at (+-b/2, 0)  -> peak shear / (G theta)
    grad_short_mid: float     # |grad psi| at (0, +-a/2)

    @property
    def aspect(self) -> float:
        return self.a / self.b

    @property
    def c1(self) -> float:
        return self.torsion_constant / (self.a * self.b**3)

    @property
    def c2(self) -> float:
        """``tau_max = T / (c2 a b^2)``, using ``tau_max = grad_long_mid / J`` at ``T = 1``."""
        return self.torsion_constant / (self.a * self.b**2 * self.grad_long_mid)


def prandtl_finite_difference(a: float, b: float, *, n_short: int = 80) -> FiniteDifferenceTorsion:
    """Solve ``lap(psi) = -2``, ``psi = 0`` on the ``a x b`` boundary by finite differences.

    ``a`` is the long side (``y``), ``b`` the short side (``x``), matching
    :func:`tensorfem.torsion_bar.rectangular_torsion_constant`.  The grid uses
    ``n_short`` cells across ``b`` and ``round(n_short a / b)`` across ``a``; the
    spacing ``hx = b/nx``, ``hy = a/ny`` is kept separate so the discretised domain is
    exactly ``a x b``.  The peak gradients use a second-order one-sided difference that
    exploits ``psi = 0`` on the boundary.
    """
    _check_sides(a, b)
    if n_short < 8:
        raise ValueError("n_short must be >= 8")
    nx = int(n_short)
    ny = max(int(round(n_short * a / b)), n_short)
    # Cells are slightly anisotropic unless a/b is rational with a small denominator;
    # using hx != hy keeps the *domain* exactly a x b, which matters at 1e-3 level.
    hx = b / nx
    hy = a / ny

    index = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    index[1:nx, 1:ny] = np.arange((nx - 1) * (ny - 1)).reshape(nx - 1, ny - 1)

    n = (nx - 1) * (ny - 1)
    flat = index[1:nx, 1:ny].reshape(-1)
    rows, cols, vals = [flat, flat], [flat, flat], [np.full(n, -2.0 / hx**2 - 2.0 / hy**2),
                                                       np.zeros(n)]

    def couple(offsets, value):
        source = index[offsets[0], offsets[1]].reshape(-1)
        mask = source >= 0
        rows.append(flat[mask])
        cols.append(source[mask])
        vals.append(np.full(int(mask.sum()), value))

    couple((np.s_[:nx - 1], np.s_[1:ny]), 1.0 / hx**2)      # i - 1
    couple((np.s_[2:nx + 1], np.s_[1:ny]), 1.0 / hx**2)     # i + 1
    couple((np.s_[1:nx], np.s_[:ny - 1]), 1.0 / hy**2)      # j - 1
    couple((np.s_[1:nx], np.s_[2:ny + 1]), 1.0 / hy**2)     # j + 1

    matrix = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n))
    solution = spla.spsolve(matrix.tocsc(), np.full(n, -2.0))

    psi = np.zeros((nx + 1, ny + 1))
    psi[1:nx, 1:ny] = solution.reshape(nx - 1, ny - 1)

    # J = 2 int psi dA; psi vanishes on the boundary, so the trapezoid rule reduces to
    # the plain interior sum times the cell area.
    j_value = 2.0 * psi.sum() * hx * hy

    def one_sided(u0, u1, u2, step):
        """Second-order boundary derivative where the boundary value ``u0`` is zero."""
        return (-3.0 * u0 + 4.0 * u1 - u2) / (2.0 * step)

    j_mid = ny // 2
    grad_long = abs(-one_sided(psi[nx, j_mid], psi[nx - 1, j_mid], psi[nx - 2, j_mid], hx))
    i_mid = nx // 2
    grad_short = abs(one_sided(psi[i_mid, ny], psi[i_mid, ny - 1], psi[i_mid, ny - 2], hy))
    return FiniteDifferenceTorsion(a=a, b=b, n_short=nx, n_long=ny, torsion_constant=j_value,
                                   grad_long_mid=grad_long, grad_short_mid=grad_short)


def richardson(fine: float, coarse: float, order: float = 2.0) -> float:
    """Richardson extrapolation for a mesh refinement ratio of two."""
    factor = 2.0**order
    return (factor * fine - coarse) / (factor - 1.0)


# ----------------------------------------------------------- closed form & hand-book


@dataclass(frozen=True)
class ShapeFactors:
    """Closed-form shape factors of an ``a x b`` section, in hand-book normalisation."""

    a: float
    b: float
    c1: float              # J = c1 a b^3
    c2: float              # tau_max = T / (c2 a b^2), boundary evaluation
    grad_long_mid: float   # |grad psi| at the long-side midpoint  (= peak / (G theta))
    grad_short_mid: float  # |grad psi| at the short-side midpoint
    peak_location: tuple[float, float]
    peak_scan: float       # interior-scan peak (strict lower bound)

    @property
    def aspect(self) -> float:
        return self.a / self.b

    @property
    def long_over_short(self) -> float:
        return self.grad_long_mid / self.grad_short_mid


def shape_factors(a: float, b: float, *, torque: float = 1.0, samples: int = 401,
                  terms: int = _TERMS) -> ShapeFactors:
    """``(c1, c2)`` and both side-midpoint gradients from the closed-form series.

    ``c2`` uses the series evaluated **on** the boundary (the exact peak); the returned
    ``peak_scan`` is the interior half-cell scan of
    :func:`tensorfem.torsion_bar.max_shear_stress`, which is a strict lower bound and is
    kept as an independent self-consistency check.
    """
    _check_sides(a, b)
    j_value = rectangular_torsion_constant(a, b, terms=terms)

    def gradient(x, y):
        tx, ty = torsion_shear_stress(torch.tensor([x]), torch.tensor([y]), a, b,
                                      twist_rate=1.0, shear_modulus=1.0, terms=terms)
        return float(torch.sqrt(tx**2 + ty**2))

    g_long = gradient(b / 2.0, 0.0)
    g_short = gradient(0.0, a / 2.0)
    peak, location = max_shear_stress(a, b, torque=torque, samples=samples, terms=terms)
    # tau_max = G theta * grad -> with G theta = T / J the hand-book c2 follows
    c2 = j_value / (a * b**2 * g_long)
    return ShapeFactors(a=a, b=b, c1=j_value / (a * b**3), c2=c2, grad_long_mid=g_long,
                        grad_short_mid=g_short, peak_location=location, peak_scan=peak)


@dataclass(frozen=True)
class ClassicalCoefficient:
    """One row of the hand-book shape-factor table for ``a x b``, ``a >= b``."""

    aspect: float
    c1: float
    c2: float

    def torsion_constant(self, b: float) -> float:
        return self.c1 * self.aspect * b * b**3

    def peak_shear(self, b: float, torque: float = 1.0) -> float:
        return torque / (self.c2 * self.aspect * b * b**2)


#: Timoshenko & Goodier / Roark hand-book values, rounded to three decimals.
_CLASSICAL = (
    ClassicalCoefficient(1.0, 0.141, 0.208),
    ClassicalCoefficient(1.2, 0.166, 0.219),
    ClassicalCoefficient(1.5, 0.196, 0.231),
    ClassicalCoefficient(2.0, 0.229, 0.246),
    ClassicalCoefficient(2.5, 0.249, 0.258),
    ClassicalCoefficient(3.0, 0.263, 0.267),
    ClassicalCoefficient(4.0, 0.281, 0.282),
    ClassicalCoefficient(5.0, 0.291, 0.291),
    ClassicalCoefficient(10.0, 0.312, 0.312),
)


def classical_rectangle_coefficients(aspect: float | None = None):
    """Hand-book rows; pass ``aspect`` to fetch the row with that exact ``a / b``."""
    if aspect is None:
        return _CLASSICAL
    for row in _CLASSICAL:
        if abs(row.aspect - aspect) <= 1e-12:
            return row
    raise KeyError(f"no tabulated row for a/b = {aspect}; available: "
                   f"{[row.aspect for row in _CLASSICAL]}")


def thin_strip_torsion_constant(a: float, b: float) -> float:
    """Free thin-strip limit ``J -> a b^3 / 3`` (exact as ``a / b -> inf``)."""
    _check_sides(a, b)
    return a * b**3 / 3.0