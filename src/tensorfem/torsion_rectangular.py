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
    "ShapeFactors",
    "classical_rectangle_coefficients",
    "prandtl_finite_difference",
    "rectangular_torsion_bar",
    "richardson",
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

    @property
    def aspect(self) -> float:
        return self.a / self.b

    @property
    def torsion_constant_relative_error(self) -> float:
        return abs(self.measured_torsion_constant / self.exact_torsion_constant - 1.0)


def rectangular_torsion_bar(*, a: float = 2.0, b: float = 1.0, length: float = 4.0,
                            segments_short: int = 3, segments_long: int | None = None,
                            young: float = 1.0e9, poisson: float = 0.3, torque: float = 1.0,
                            window: tuple[float, float] = (0.25, 0.75),
                            terms: int = _TERMS, rtol: float = 1e-12) -> RectangularBarResult:
    """Twist an ``a x b x length`` HEX8 prism with a pure end torque and measure it.

    ``segments_long`` defaults to ``round(segments_short * a / b)``, which keeps the
    elements square.  The far face ``z = length`` carries the circumferential linear
    traction ``t = c(-y, x)`` integrated with the consistent nodal weights of each
    element face; the near face is clamped in-plane with one axial dof removed so
    warping stays free.
    """
    _check_sides(a, b)
    if segments_short < 1:
        raise ValueError("segments_short must be >= 1")
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
    forces = forces * (torque / polar)
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
    inside = (planes >= window[0] * length) & (planes <= window[1] * length)
    design = torch.stack((planes[inside], torch.ones(int(inside.sum()), dtype=nodes.dtype)), 1)
    twist_rate = float(torch.linalg.lstsq(design, thetas[inside]).solution[0])

    shear = young / (2.0 * (1.0 + poisson))
    exact_j = rectangular_torsion_constant(a, b, terms=terms)
    exact_rate = torque / (shear * exact_j)

    strain = torch.einsum("eij,ej->ei", b_matrix, displacement[element_dofs])
    stress = torch.einsum("eij,ej->ei", modulus, strain)
    centroids = nodes[cells.long()].mean(1)
    hz = length / nz
    slab = (centroids[:, 2] - length / 2.0).abs() < hz + 1e-12
    cx = centroids[slab, 0] - b / 2.0
    cy = centroids[slab, 1] - a / 2.0
    tx, ty = torsion_shear_stress(cx, cy, a, b, twist_rate=twist_rate, shear_modulus=shear,
                                  terms=terms)
    fe = torch.stack((stress[slab, 5], stress[slab, 4]), 1)  # tau_zx, tau_zy
    exact = torch.stack((tx, ty), 1)
    denominator = float(torch.sqrt(torch.sum(exact**2)))
    l2 = float(torch.sqrt(torch.sum((fe - exact) ** 2))) / denominator if denominator > 1e-9 else float("nan")

    peak_fe = float(torch.sqrt(fe[:, 0] ** 2 + fe[:, 1] ** 2).max())
    peak_sampled = float(torch.sqrt(exact[:, 0] ** 2 + exact[:, 1] ** 2).max())

    return RectangularBarResult(
        a=a, b=b, segments_short=n_short, segments_long=n_long, n_elements=n_elements,
        n_nodes=len(nodes), n_dofs=n_dofs, applied_torque=applied, twist_rate=twist_rate,
        exact_twist_rate=exact_rate, measured_torsion_constant=torque / (shear * twist_rate),
        exact_torsion_constant=exact_j, centroid_shear_l2_error=l2,
        peak_centroid_fe=peak_fe, peak_series_sampled=peak_sampled,
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