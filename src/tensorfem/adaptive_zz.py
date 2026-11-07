"""Zienkiewicz-Zhu (ZZ) error indicator and an indicator-driven h-refinement loop.

Why this module exists
----------------------
TensorFEM carried a Gauss-point (Barlow) stress *recovery* (:mod:`tensorfem.stress_recovery`)
but no **error estimator** and no **adaptive mesh refinement**: every convergence study in
the repository is a uniform ``h`` ladder, so a solution whose error is concentrated in a
small part of the domain (a re-entrant corner, a load-introduction point, a local geometric
detail) had to be resolved by refining the whole mesh.  This module supplies the minimal
missing face of that capability:

1. an **element error indicator** built from the ZZ superconvergent patch recovery (SPR),
   ``eta_e^2 = int_{Omega_e} (sigma_h - sigma*)^T C^-1 (sigma_h - sigma*) dOmega``,
2. a **marking** rule (Doerfler / bulk marking) and **quadtree h-refinement** with
   hanging-node elimination by multipoint constraints, and
3. a **closed verification loop** on a problem with an exact solution, so the indicator can
   be scored against the true error: the Williams (1952) eigen-solution of the
   re-entrant-corner singularity on an L-shaped panel, which is the canonical model of a
   ship-structure detail such as a hatch-coaming corner or a bracket toe.

The indicator and the loop are generic (any Q4 mesh plus a displacement field).  The exact
solution is used only by the *verification* entry points, which is what makes the
"estimated versus true error" comparison of :func:`adaptive_study` meaningful.

Physics / benchmark definition
------------------------------
The L-shaped domain is ``Omega = [-H, H]^2 minus [-H, 0] x [-H, 0]`` (``H = HALF_EXTENT``),
a square with the third quadrant removed, so the interior angle at the re-entrant corner (the
origin) is ``2*alpha = 3*pi/2``.  In polar coordinates centred at the corner the symmetric
eigen-solution of the plane problem is the biharmonic Airy function

    phi(r, theta') = r^(lambda+1) [ A cos((lambda+1) theta') + B cos((lambda-1) theta') ]

with ``theta' = theta - pi/4`` measured from the bisector of the 270 degree material angle,
and ``lambda`` the smallest positive root of ``sin(2 alpha lambda) + lambda sin(2 alpha) = 0``
(``lambda = 0.5444837367...`` for ``2 alpha = 3 pi / 2``).  The traction-free conditions on
both notch faces are satisfied identically by the choice of ``B``, so the field is an exact
solution of the homogeneous equilibrium problem (zero body force) on ``Omega``.  Prescribing
its trace on the *outer* square boundary as a Dirichlet condition defines a boundary value
problem whose exact solution is that field, hence the exact strain energy, the exact stress
and the exact energy norm of any discrete error are all available in closed form:

    a(u, u) = int_Omega sigma : C^-1 : sigma dOmega
            = int_0^(2 pi) h(theta) R(theta)^(2 lambda) / (2 lambda) dtheta,

where ``h(theta) = S(theta) : C^-1 : S(theta)`` is the angular energy density of the
unit-radius stress factor ``S`` and ``R(theta)`` is the distance from the corner to the
outer square boundary along the ray ``theta``.  The angular integral is evaluated with
composite Gauss-Legendre panels, i.e. to machine-level accuracy rather than by mesh
quadrature.

Conventions (identical to :mod:`tensorfem.continuum`)
-----------------------------------------------------
* Q4 node order ``(-1,-1) (1,-1) (1,1) (-1,1)`` (counter-clockwise);
* DOF order ``(u, v)`` per node, element vector ``[u0, v0, u1, v1, ...]``;
* parent Jacobian ``J[a, b] = dX_b / dxi_a``, so ``dN/dX = dN/dxi . inv(J)``.

Scope and honesty
-----------------
* Linear elastic, small strain, single material, plane stress/strain, ``h``-adaptive only;
  no ``p``-adaptivity, no goal-oriented (dual-weighted-residual) indicator, no anisotropic
  refinement, no error control for nonlinear paths.
* The ZZ indicator is *not* guaranteed to be asymptotically exact on a graded mesh; the
  effectivity index is measured, not assumed, and a non-positive indicator value is
  rejected instead of silently clipped.
* Hanging nodes are eliminated by multipoint constraints on the *linear* edge
  interpolation, which is exactly conforming for Q4 elements on straight edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .sparse_core import MPC, assemble_coo, solve_sparse_static

# ---------------------------------------------------------------------------
# fixed benchmark parameters (dimensionless, so the study is reproducible)
# ---------------------------------------------------------------------------
HALF_EXTENT = 1.0
YOUNG = 1.0
POISSON = 0.3
PLANE = "strain"
THICKNESS = 1.0
#: interior angle at the re-entrant corner: 270 degrees
CORNER_ANGLE = 1.5 * math.pi
#: nodes farther than this from the corner sit in the analytic part of the field
SMOOTH_RADIUS = 0.25
#: reference value of the Williams exponent for ``2 alpha = 3 pi / 2``
WILLIAMS_LAMBDA = 0.5444837367824639

_INV_SQRT3 = 1.0 / math.sqrt(3.0)
#: 2 x 2 Gauss rule (the superconvergent points of a Q4 element), eta fastest
GAUSS_2X2 = (
    (-_INV_SQRT3, -_INV_SQRT3),
    (_INV_SQRT3, -_INV_SQRT3),
    (_INV_SQRT3, _INV_SQRT3),
    (-_INV_SQRT3, _INV_SQRT3),
)


# ---------------------------------------------------------------------------
# constitutive matrices
# ---------------------------------------------------------------------------
def elasticity_matrix(
    young: float = YOUNG, poisson: float = POISSON, plane: str = PLANE
) -> np.ndarray:
    """Isotropic 3 x 3 constitutive matrix in engineering shear convention."""
    if plane == "stress":
        scale = young / (1.0 - poisson * poisson)
        base = np.array(
            [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]]
        )
        return scale * base
    if plane == "strain":
        scale = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        base = np.array(
            [
                [1.0 - poisson, poisson, 0.0],
                [poisson, 1.0 - poisson, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson)],
            ]
        )
        return scale * base
    raise ValueError("plane must be 'stress' or 'strain'")


def inverse3(matrix: np.ndarray) -> np.ndarray:
    """Analytic 3 x 3 inverse (batched over leading axes).

    ``numpy.linalg.inv``/``solve`` fall back to a per-call LAPACK path on this platform that
    costs milliseconds *per 3 x 3 matrix*, which dominates every estimator evaluation; the
    closed-form adjugate is both faster and exactly invertible to round-off.
    """
    m = np.asarray(matrix, dtype=np.float64)
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    inv = np.empty(m.shape, dtype=np.float64)
    inv[..., 0, 0] = c00
    inv[..., 0, 1] = -(b * i - c * h)
    inv[..., 0, 2] = b * f - c * e
    inv[..., 1, 0] = c01
    inv[..., 1, 1] = a * i - c * g
    inv[..., 1, 2] = -(a * f - c * d)
    inv[..., 2, 0] = c02
    inv[..., 2, 1] = -(a * h - b * g)
    inv[..., 2, 2] = a * e - b * d
    return inv / det[..., None, None]


def compliance_matrix(
    young: float = YOUNG, poisson: float = POISSON, plane: str = PLANE
) -> np.ndarray:
    """``C^-1``: the matrix whose quadratic form is the strain energy density (closed form)."""
    if plane == "stress":
        return (
            np.array(
                [[1.0, -poisson, 0.0], [-poisson, 1.0, 0.0], [0.0, 0.0, 2.0 * (1.0 + poisson)]]
            )
            / young
        )
    if plane == "strain":
        diagonal = (1.0 + poisson) * (1.0 - poisson)
        off = -(1.0 + poisson) * poisson
        return (
            np.array(
                [[diagonal, off, 0.0], [off, diagonal, 0.0], [0.0, 0.0, 2.0 * (1.0 + poisson)]]
            )
            / young
        )
    raise ValueError("plane must be 'stress' or 'strain'")


# ---------------------------------------------------------------------------
# Williams re-entrant corner eigen-solution
# ---------------------------------------------------------------------------
def williams_exponent(
    interior_angle: float = CORNER_ANGLE,
    *,
    lower: float = 1e-6,
    upper: float = 1.0,
    iterations: int = 300,
) -> float:
    """Smallest positive root of ``sin(lambda*2a) + lambda*sin(2a) = 0``.

    ``2a`` is the interior material angle at the corner (``3 pi / 2`` for the L-shape).
    Bisection is used because the residual is monotone between the origin and the first
    root, which makes the bracket deterministic (Newton can jump to a higher eigenvalue).
    """
    alpha = 0.5 * interior_angle

    def residual(lam: float) -> float:
        return math.sin(2.0 * alpha * lam) + lam * math.sin(2.0 * alpha)

    lo, hi = float(lower), float(upper)
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError("no eigenvalue bracket on the requested interval")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def williams_amplitudes(
    lam: float, interior_angle: float = CORNER_ANGLE, amplitude: float = 1.0
) -> tuple[float, float]:
    """``(A, B)`` of the symmetric mode such that both notch faces are traction free."""
    alpha = 0.5 * interior_angle
    a = float(amplitude)
    b = -a * math.cos((lam + 1.0) * alpha) / math.cos((lam - 1.0) * alpha)
    return a, b


def _theta_factors(lam: float, theta, a: float, b: float):
    """Unit-radius polar stress factors and their first angular derivatives."""
    p1, p2 = lam + 1.0, lam - 1.0
    c1, s1 = np.cos(p1 * theta), np.sin(p1 * theta)
    c2, s2 = np.cos(p2 * theta), np.sin(p2 * theta)
    f = a * c1 + b * c2
    f1 = -p1 * a * s1 - p2 * b * s2
    f2 = -p1 * p1 * a * c1 - p2 * p2 * b * c2
    f3 = p1**3 * a * s1 + p2**3 * b * s2
    s_rr = p1 * f + f2
    s_tt = lam * p1 * f
    s_rt = -lam * f1
    d_s_rr = p1 * f1 + f3
    d_s_tt = lam * p1 * f1
    d_s_rt = -lam * f2
    return (s_rr, s_tt, s_rt, d_s_rr, d_s_tt, d_s_rt)


def _polar_to_cartesian(s_rr, s_tt, s_rt, cos_t, sin_t):
    c, s = cos_t, sin_t
    s_xx = c * c * s_rr + s * s * s_tt - 2.0 * c * s * s_rt
    s_yy = s * s * s_rr + c * c * s_tt + 2.0 * c * s * s_rt
    s_xy = c * s * (s_rr - s_tt) + (c * c - s * s) * s_rt
    return s_xx, s_yy, s_xy


def williams_stress(
    points,
    *,
    lam: float | None = None,
    amplitude: float = 1.0,
    interior_angle: float = CORNER_ANGLE,
    origin=(0.0, 0.0),
) -> np.ndarray:
    """Exact Cartesian stress ``(sigma_xx, sigma_yy, sigma_xy)``; ``points`` is ``(n, 2)``.

    The Williams angular functions are written in ``theta' = theta - pi/4`` (measured from
    the bisector of the material angle) while the polar-to-Cartesian rotation uses the
    standard ``theta``; conflating the two is the classic way of getting the tractions on
    the notch faces wrong, so both angles are kept explicitly here.
    """
    if lam is None:
        lam = williams_exponent(interior_angle)
    a, b = williams_amplitudes(lam, interior_angle, amplitude)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    dx = pts[:, 0] - float(origin[0])
    dy = pts[:, 1] - float(origin[1])
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    s_rr, s_tt, s_rt, _, _, _ = _theta_factors(lam, theta - 0.25 * math.pi, a, b)
    s_xx, s_yy, s_xy = _polar_to_cartesian(s_rr, s_tt, s_rt, np.cos(theta), np.sin(theta))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.power(r, lam - 1.0)
    return np.stack((scale * s_xx, scale * s_yy, scale * s_xy), axis=1)


def williams_displacement(
    points,
    *,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    amplitude: float = 1.0,
    interior_angle: float = CORNER_ANGLE,
    origin=(0.0, 0.0),
) -> np.ndarray:
    """Exact Cartesian displacement ``(u_x, u_y)`` of the Williams field.

    Derived from ``u_r = r^lam R(theta')`` and ``u_theta = r^lam T(theta')``, with ``R``
    from the radial strain and ``T`` from the shear strain (no angular integration
    constant is needed because the shear relation is algebraic in ``T``).
    ``tests/test_adaptive_zz.py`` checks the resulting field against the strain-stress
    relation pointwise, which is what makes it safe to use as Dirichlet data.
    """
    if lam is None:
        lam = williams_exponent(interior_angle)
    a, b = williams_amplitudes(lam, interior_angle, amplitude)
    if plane == "stress":
        alpha_c, beta_c = 1.0 / young, -poisson / young
    elif plane == "strain":
        alpha_c = (1.0 - poisson * poisson) / young
        beta_c = -poisson * (1.0 + poisson) / young
    else:
        raise ValueError("plane must be 'stress' or 'strain'")
    shear_c = 2.0 * (1.0 + poisson) / young

    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    dx = pts[:, 0] - float(origin[0])
    dy = pts[:, 1] - float(origin[1])
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    theta_p = theta - 0.25 * math.pi
    s_rr, s_tt, s_rt, d_s_rr, d_s_tt, _ = _theta_factors(lam, theta_p, a, b)

    radial = (alpha_c * s_rr + beta_c * s_tt) / lam
    d_radial = (alpha_c * d_s_rr + beta_c * d_s_tt) / lam
    tangential = (shear_c * s_rt - d_radial) / (lam - 1.0)

    scale = np.power(r, lam)
    u_r = scale * radial
    u_t = scale * tangential
    c, s = np.cos(theta), np.sin(theta)
    return np.stack((u_r * c - u_t * s, u_r * s + u_t * c), axis=1)


def williams_strain(
    points,
    *,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    amplitude: float = 1.0,
    interior_angle: float = CORNER_ANGLE,
) -> np.ndarray:
    """Exact Cartesian strain obtained from the exact stress through ``C^-1``."""
    stress = williams_stress(points, lam=lam, amplitude=amplitude, interior_angle=interior_angle)
    return np.einsum("ij,nj->ni", compliance_matrix(young, poisson, plane), stress)


def _ray_length(theta, half_extent: float = HALF_EXTENT) -> np.ndarray:
    """Distance from the corner to the outer square boundary along ``theta``."""
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    with np.errstate(divide="ignore"):
        rx = half_extent / np.maximum(c, 1e-300)
        ry = half_extent / np.maximum(s, 1e-300)
    return np.minimum(rx, ry)


def _theta_panels(panels: int, order: int):
    """Composite Gauss-Legendre nodes/weights across the material angular span."""
    n, w = np.polynomial.legendre.leggauss(order)
    edges = (
        0.0,
        0.25 * math.pi,
        0.5 * math.pi,
        0.75 * math.pi,
        math.pi,
        1.5 * math.pi,
        1.75 * math.pi,
        2.0 * math.pi,
    )
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        step = (hi - lo) / panels
        for k in range(panels):
            a, b = lo + k * step, lo + (k + 1) * step
            half = 0.5 * (b - a)
            nodes.append(0.5 * (a + b) + half * n)
            weights.append(half * w)
    return np.concatenate(nodes), np.concatenate(weights)


def exact_energy_norm_sq(
    *,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    interior_angle: float = CORNER_ANGLE,
    amplitude: float = 1.0,
    half_extent: float = HALF_EXTENT,
    panels: int = 256,
    order: int = 12,
) -> float:
    """``a(u, u) = int sigma : C^-1 : sigma`` of the Williams field, in closed form.

    Polar integration with the radial power law
    ``int_0^R r^(2 lambda - 1) dr = R^(2 lambda) / (2 lambda)`` integrated out
    analytically; only the smooth angular integral is quadrature, so the result is
    accurate to machine precision (checked by panel doubling in the tests).
    """
    if lam is None:
        lam = williams_exponent(interior_angle)
    a, b = williams_amplitudes(lam, interior_angle, amplitude)
    compliance = compliance_matrix(young, poisson, plane)
    theta, w = _theta_panels(panels, order)
    s_rr, s_tt, s_rt, _, _, _ = _theta_factors(lam, theta - 0.25 * math.pi, a, b)
    s_xx, s_yy, s_xy = _polar_to_cartesian(s_rr, s_tt, s_rt, np.cos(theta), np.sin(theta))
    unit = np.stack((s_xx, s_yy, s_xy), axis=1)
    density = np.einsum("ni,ij,nj->n", unit, compliance, unit)
    radius = _ray_length(theta, half_extent)
    return float(np.sum(w * density * np.power(radius, 2.0 * lam) / (2.0 * lam)))


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadMesh:
    """Quadrilateral mesh: ``nodes`` is ``(n, 2)``, ``elements`` is ``(ne, 4)``."""

    nodes: np.ndarray
    elements: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        elements = np.asarray(self.elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must be (n, 2)")
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("elements must be (ne, 4)")
        if elements.size and (elements.min() < 0 or elements.max() >= len(nodes)):
            raise ValueError("element connectivity out of range")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "elements", elements)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    def coordinates(self) -> np.ndarray:
        return self.nodes[self.elements]

    def centres(self) -> np.ndarray:
        return self.coordinates().mean(axis=1)

    def areas(self) -> np.ndarray:
        p = self.coordinates()
        x, y = p[:, :, 0], p[:, :, 1]
        first = 0.5 * (
            (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        )
        second = 0.5 * (
            (x[:, 2] - x[:, 0]) * (y[:, 3] - y[:, 0]) - (x[:, 3] - x[:, 0]) * (y[:, 2] - y[:, 0])
        )
        return np.abs(first) + np.abs(second)

    def edges(self) -> np.ndarray:
        """``(4*ne, 2)`` sorted node pairs of every element edge."""
        e = self.elements
        quad = np.stack((e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 3]], e[:, [3, 0]]), axis=1)
        return np.sort(quad.reshape(-1, 2), axis=1)


def lshape_mesh(divisions: int = 8, half_extent: float = HALF_EXTENT) -> QuadMesh:
    """Structured Q4 mesh of ``[-H,H]^2`` minus the lower-left (third) quadrant.

    ``divisions`` must be even so that both notch faces coincide with grid lines.
    """
    if divisions < 2 or divisions % 2:
        raise ValueError("divisions must be an even integer >= 2")
    h = 2.0 * half_extent / divisions
    node_ids: dict[tuple[float, float], int] = {}
    nodes: list[tuple[float, float]] = []

    def node_id(x: float, y: float) -> int:
        key = (round(x, 12), round(y, 12))
        if key not in node_ids:
            node_ids[key] = len(nodes)
            nodes.append(key)
        return node_ids[key]

    elements: list[list[int]] = []
    for j in range(divisions):
        y0 = -half_extent + j * h
        for i in range(divisions):
            x0 = -half_extent + i * h
            if x0 + h <= 1e-12 and y0 + h <= 1e-12:  # third quadrant removed
                continue
            elements.append(
                [node_id(x0, y0), node_id(x0 + h, y0), node_id(x0 + h, y0 + h), node_id(x0, y0 + h)]
            )
    return QuadMesh(np.asarray(nodes), np.asarray(elements, dtype=np.int64))


def refine_mesh(mesh: QuadMesh, marked) -> QuadMesh:
    """Quadtree refinement: every marked quad is split into four; midpoints are shared."""
    flagged = np.asarray(marked, dtype=bool)
    if flagged.shape != (mesh.n_elements,):
        raise ValueError("marker must be a boolean array of length n_elements")
    nodes = [(float(x), float(y)) for x, y in mesh.nodes]
    index = {(round(x, 12), round(y, 12)): i for i, (x, y) in enumerate(nodes)}

    def node_id(x: float, y: float) -> int:
        key = (round(x, 12), round(y, 12))
        if key not in index:
            index[key] = len(nodes)
            nodes.append(key)
        return index[key]

    elements: list[list[int]] = []
    for e, corners in enumerate(mesh.elements):
        if not flagged[e]:
            elements.append([int(v) for v in corners])
            continue
        p = mesh.nodes[corners]
        a, b, c, d = (int(v) for v in corners)
        m_ab = node_id(0.5 * (p[0, 0] + p[1, 0]), 0.5 * (p[0, 1] + p[1, 1]))
        m_bc = node_id(0.5 * (p[1, 0] + p[2, 0]), 0.5 * (p[1, 1] + p[2, 1]))
        m_cd = node_id(0.5 * (p[2, 0] + p[3, 0]), 0.5 * (p[2, 1] + p[3, 1]))
        m_da = node_id(0.5 * (p[3, 0] + p[0, 0]), 0.5 * (p[3, 1] + p[0, 1]))
        centre = p.mean(axis=0)
        mid = node_id(float(centre[0]), float(centre[1]))
        elements.extend(
            [
                [a, m_ab, mid, m_da],
                [m_ab, b, m_bc, mid],
                [mid, m_bc, c, m_cd],
                [m_da, mid, m_cd, d],
            ]
        )
    return QuadMesh(np.asarray(nodes), np.asarray(elements, dtype=np.int64))


def boundary_nodes(
    mesh: QuadMesh, half_extent: float = HALF_EXTENT, tolerance: float = 1e-12
) -> np.ndarray:
    """Node ids lying on ``dOmega`` of the L-shaped domain.

    A topological "edge used by one element" test is **wrong** here: after a hanging-node
    refinement the coarse element still holds the full edge while the fine side holds two
    halves, so both halves look unmatched and every hanging node would be misclassified as
    a boundary node (which then collides with the Dirichlet set).  The geometric predicate
    of the known domain is used instead.
    """
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    on_outer = (np.abs(np.abs(x) - half_extent) <= tolerance) | (
        np.abs(np.abs(y) - half_extent) <= tolerance
    )
    on_notch = ((np.abs(x) <= tolerance) & (y <= tolerance)) | (
        (np.abs(y) <= tolerance) & (x <= tolerance)
    )
    return np.nonzero(on_outer | on_notch)[0]


def hanging_nodes(mesh: QuadMesh) -> dict[int, tuple[int, int]]:
    """Map each hanging node to the parent edge ``(a, b)`` whose midpoint it is.

    A hanging node sits at the midpoint of an element edge that does not carry it as a
    corner.  Its value is fixed by the linear edge interpolation ``u = (u_a + u_b) / 2``,
    which is exactly the Q4 restriction on a straight edge.  Because a boundary edge is
    subdivided only when its own element is refined, a boundary node is never a slave --
    which is what lets the exact Dirichlet data be prescribed at every boundary node.
    """
    index = {(round(x, 12), round(y, 12)): i for i, (x, y) in enumerate(mesh.nodes)}
    found: dict[int, tuple[int, int]] = {}
    for corners in mesh.elements:
        corner_set = {int(v) for v in corners}
        for a, b in (
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ):
            pa, pb = mesh.nodes[a], mesh.nodes[b]
            key = (round(0.5 * (pa[0] + pb[0]), 12), round(0.5 * (pa[1] + pb[1]), 12))
            mid = index.get(key)
            if mid is None or mid in corner_set:
                continue
            found[mid] = (int(a), int(b))
    return found


def _flatten_edge_constraints(slaves: dict[int, tuple[int, int]]) -> dict[int, dict[int, float]]:
    """Resolve chained midpoint constraints into flat master coefficients.

    Two levels of refinement can put a node on the edge of an element while that node is
    itself a hanging node of a coarser edge, so the raw constraints must be composed.
    Cycles are rejected (they would silently produce a singular space).
    """
    resolved: dict[int, dict[int, float]] = {}

    def expand(node: int, stack: tuple[int, ...]) -> dict[int, float]:
        if node in resolved:
            return resolved[node]
        if node not in slaves:
            return {node: 1.0}
        if node in stack:
            raise ValueError(f"cyclic hanging-node constraint through node {node}")
        a, b = slaves[node]
        acc: dict[int, float] = {}
        for parent in (a, b):
            for master, coefficient in expand(parent, stack + (node,)).items():
                acc[master] = acc.get(master, 0.0) + 0.5 * coefficient
        resolved[node] = acc
        return acc

    return {node: expand(node, ()) for node in slaves}


def constraint_pairs(mesh: QuadMesh) -> list[tuple[int, dict[int, float]]]:
    """Flat hanging-node constraints as ``(slave_node, {master_node: coefficient})``."""
    return list(_flatten_edge_constraints(hanging_nodes(mesh)).items())


def free_dofs(mesh: QuadMesh, prescribed: dict[int, float]) -> int:
    """Number of unknowns actually solved for (the fair DOF count of an adaptive mesh)."""
    n_mpc = 0
    for slave_node, masters in constraint_pairs(mesh):
        for component in (0, 1):
            if 2 * slave_node + component in prescribed:
                continue
            masters_active = [m for m in masters if 2 * m + component not in prescribed]
            if masters_active:
                n_mpc += 1
    return (
        2 * mesh.n_nodes - len(prescribed) - n_mpc
    )  # ---------------------------------------------------------------------------


# Q4 kernels, assembly and the constrained solve
# ---------------------------------------------------------------------------
def q4_shape(xi: float, eta: float) -> np.ndarray:
    """Bilinear shape functions in the Q4 node order."""
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )


def q4_shape_derivative(xi: float, eta: float) -> np.ndarray:
    """Parametric shape-function derivatives ``dN/dxi``, ``dN/deta`` as ``(4, 2)``."""
    return 0.25 * np.array(
        [
            (-(1.0 - eta), -(1.0 - xi)),
            ((1.0 - eta), -(1.0 + xi)),
            ((1.0 + eta), (1.0 + xi)),
            (-(1.0 + eta), (1.0 - xi)),
        ]
    )


def _strain_matrix(coordinates: np.ndarray, xi: float, eta: float):
    """``(ne, 3, 8)`` strain-displacement matrix and ``(ne,)`` Jacobian determinants."""
    derivative = q4_shape_derivative(xi, eta)  # (4, 2)
    jacobian = np.einsum("eia,ib->eab", coordinates, derivative)  # (ne, 2, 2)
    det = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    if np.any(det <= 0.0):
        raise ValueError("element with non-positive Jacobian")
    inverse = np.empty_like(jacobian)
    inverse[:, 0, 0] = jacobian[:, 1, 1] / det
    inverse[:, 0, 1] = -jacobian[:, 0, 1] / det
    inverse[:, 1, 0] = -jacobian[:, 1, 0] / det
    inverse[:, 1, 1] = jacobian[:, 0, 0] / det
    gradient = np.einsum("ib,ebc->eic", derivative, inverse)
    ne = coordinates.shape[0]
    b = np.zeros((ne, 3, 8))
    b[:, 0, 0::2] = gradient[:, :, 0]
    b[:, 1, 1::2] = gradient[:, :, 1]
    b[:, 2, 0::2] = gradient[:, :, 1]
    b[:, 2, 1::2] = gradient[:, :, 0]
    return b, det


def element_stiffness(mesh: QuadMesh, dmat: np.ndarray, thickness: float = THICKNESS) -> np.ndarray:
    """``(ne, 8, 8)`` fully integrated Q4 stiffness matrices (2 x 2 Gauss)."""
    coordinates = mesh.coordinates()
    ke = np.zeros((mesh.n_elements, 8, 8))
    for xi, eta in GAUSS_2X2:
        b, det = _strain_matrix(coordinates, xi, eta)
        ke += np.einsum("eji,jk,ekl->eil", b, dmat, b) * (det * thickness)[:, None, None]
    return ke


def element_dofs(mesh: QuadMesh) -> np.ndarray:
    """``(ne, 8)`` DOF table in ``[u0, v0, u1, v1, ...]`` order."""
    return (2 * mesh.elements[:, :, None] + np.arange(2)[None, None, :]).reshape(-1, 8)


def assemble(mesh: QuadMesh, dmat: np.ndarray, thickness: float = THICKNESS):
    """Sparse global stiffness matrix and the ``(ne, 8)`` element DOF table."""
    ke = element_stiffness(mesh, dmat, thickness)
    dofs = element_dofs(mesh)
    stiffness = assemble_coo(torch.as_tensor(ke), torch.as_tensor(dofs), 2 * mesh.n_nodes)
    return stiffness, dofs


def _mpc_list(mesh: QuadMesh, prescribed: dict[int, float]) -> list[MPC]:
    """DOF-level multipoint constraints, with prescribed parents folded into the offset."""
    mpc: list[MPC] = []
    for slave_node, masters in constraint_pairs(mesh):
        for component in (0, 1):
            slave = 2 * slave_node + component
            if slave in prescribed:
                raise ValueError("a Dirichlet node cannot also be a hanging node")
            offset = 0.0
            active: list[tuple[int, float]] = []
            for master_node, coefficient in sorted(masters.items()):
                master = 2 * master_node + component
                if master in prescribed:
                    offset += coefficient * prescribed[master]
                else:
                    active.append((master, float(coefficient)))
            if not active:
                prescribed[slave] = offset
                continue
            mpc.append(MPC(slave=slave, masters=tuple(active), offset=offset))
    return mpc


def exact_dirichlet(
    mesh: QuadMesh,
    *,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    amplitude: float = 1.0,
) -> dict[int, float]:
    """Dirichlet data on the outer boundary = trace of the exact Williams field."""
    nodes = boundary_nodes(mesh)
    displacement = williams_displacement(
        mesh.nodes[nodes], lam=lam, young=young, poisson=poisson, plane=plane, amplitude=amplitude
    )
    data: dict[int, float] = {}
    for k, node in enumerate(nodes):
        data[2 * int(node)] = float(displacement[k, 0])
        data[2 * int(node) + 1] = float(displacement[k, 1])
    return data


def solve_displacement(
    mesh: QuadMesh,
    dmat: np.ndarray,
    *,
    dirichlet: dict[int, float] | None = None,
    rtol: float = 1e-12,
    maxiter: int = 40000,
    thickness: float = THICKNESS,
):
    """Solve the constrained statics problem; returns ``(u, K)`` with ``u`` of full length."""
    stiffness, _ = assemble(mesh, dmat, thickness)
    prescribed = dict(dirichlet or {})
    mpc = _mpc_list(mesh, prescribed)
    force = torch.zeros(2 * mesh.n_nodes, dtype=torch.float64)
    result = solve_sparse_static(
        stiffness, force, dirichlet=prescribed, mpcs=tuple(mpc), rtol=rtol, maxiter=maxiter
    )
    return result.displacement.numpy(), stiffness


# ---------------------------------------------------------------------------
# stress field, ZZ superconvergent patch recovery, element indicator
# ---------------------------------------------------------------------------
def element_stress(mesh: QuadMesh, u: np.ndarray, dmat: np.ndarray) -> np.ndarray:
    """``(ne, 3)`` element **stress** ``C : B u`` evaluated at the element centre.

    Every element produced by :func:`lshape_mesh` / :func:`refine_mesh` is a parallelogram,
    so the strain-displacement matrix (and hence the stress) is constant per element; the
    centre evaluation is exact rather than an approximation here.
    """
    b, _ = _strain_matrix(mesh.coordinates(), 0.0, 0.0)
    strain = np.einsum("eij,ej->ei", b, u[element_dofs(mesh)])
    return np.einsum("ij,ej->ei", np.asarray(dmat, dtype=np.float64), strain)


def gauss_point_stresses(mesh: QuadMesh, u: np.ndarray, dmat: np.ndarray):
    """``(ne, 4, 3)`` stresses ``C : B u`` at the 2 x 2 Gauss points plus their coordinates."""
    coordinates = mesh.coordinates()
    ue = u[element_dofs(mesh)]
    stress = np.zeros((mesh.n_elements, 4, 3))
    points = np.zeros((mesh.n_elements, 4, 2))
    constitutive = np.asarray(dmat, dtype=np.float64)
    for k, (xi, eta) in enumerate(GAUSS_2X2):
        b, _ = _strain_matrix(coordinates, xi, eta)
        strain = np.einsum("eij,ej->ei", b, ue)
        stress[:, k, :] = np.einsum("ij,ej->ei", constitutive, strain)
        points[:, k, :] = np.einsum("i,eia->ea", q4_shape(xi, eta), coordinates)
    return stress, points


def barlow_extrapolation() -> np.ndarray:
    """``(4 nodes, 4 Gauss slots)`` linear extrapolation from Gauss points to nodes."""
    plus = 0.5 + 0.5 * math.sqrt(3.0)
    minus = 0.5 - 0.5 * math.sqrt(3.0)
    node_order = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    matrix = np.zeros((4, 4))
    for i, (nx, ny) in enumerate(node_order):
        for k, (gx, gy) in enumerate(GAUSS_2X2):
            row = (plus, minus) if nx < 0.0 else (minus, plus)
            column = (plus, minus) if ny < 0.0 else (minus, plus)
            matrix[i, k] = row[0 if gx < 0.0 else 1] * column[0 if gy < 0.0 else 1]
    return matrix


def spr_recover(mesh: QuadMesh, gauss_stress: np.ndarray, gauss_points: np.ndarray) -> np.ndarray:
    """Nodal superconvergent patch recovery (ZZ): a linear polynomial per nodal patch.

    For every node the Gauss-point stresses of all incident elements are fitted by
    ``sigma(x, y) = a0 + a1 x + a2 y`` in a least-squares sense and the fit is evaluated at
    the node.  The normal equations of all patches are accumulated with a scatter-add, so
    the cost is O(elements) with no Python loop over nodes.  Degenerate patches
    (``det <= 0``, e.g. collinear samples) fall back to area-weighted Barlow extrapolation
    instead of producing a silently garbage fit.
    """
    gs = np.asarray(gauss_stress, dtype=np.float64).reshape(mesh.n_elements, 4, 3)
    gp = np.asarray(gauss_points, dtype=np.float64).reshape(mesh.n_elements, 4, 2)
    rows = mesh.n_elements * 4
    basis = np.concatenate([np.ones((rows, 1)), gp.reshape(rows, 2)], axis=1)
    values = gs.reshape(rows, 3)
    normal = np.einsum("ri,rj->rij", basis, basis)
    right = np.einsum("ri,rk->rik", basis, values)

    node_index = np.repeat(mesh.elements, 4, axis=0).reshape(-1)
    mass = np.zeros((mesh.n_nodes, 3, 3))
    moment = np.zeros((mesh.n_nodes, 3, 3))
    np.add.at(mass, node_index, np.repeat(normal, 4, axis=0))
    np.add.at(moment, node_index, np.repeat(right, 4, axis=0))

    trace = np.trace(mass, axis1=1, axis2=2)
    inverse = inverse3(mass)
    det = np.abs(np.einsum("eij,eij->e", mass, inverse)) ** -1.0
    admissible = np.isfinite(inverse).all(axis=(1, 2)) & (
        det > 1e-12 * np.maximum(trace / 3.0, 1e-300) ** 3
    )

    coefficients = np.zeros((mesh.n_nodes, 3, 3))
    if np.any(admissible):
        coefficients[admissible] = np.einsum(
            "eij,ejk->eik", inverse[admissible], moment[admissible]
        )
    x, y = mesh.nodes[:, 0:1], mesh.nodes[:, 1:2]
    nodal = coefficients[:, 0, :] + x * coefficients[:, 1, :] + y * coefficients[:, 2, :]

    if not np.all(admissible):
        extrapolated = np.einsum("nk,ekc->enc", barlow_extrapolation(), gs)
        weights = mesh.areas()
        accumulated = np.zeros((mesh.n_nodes, 3))
        total = np.zeros(mesh.n_nodes)
        for local in range(4):
            np.add.at(
                accumulated, mesh.elements[:, local], weights[:, None] * extrapolated[:, local, :]
            )
            np.add.at(total, mesh.elements[:, local], weights)
        average = accumulated / np.maximum(total, 1e-300)[:, None]
        nodal[~admissible] = average[~admissible]
    return nodal


def recover_at_gauss(mesh: QuadMesh, nodal_stress: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a continuous nodal field to the 2 x 2 Gauss points."""
    element_nodal = np.asarray(nodal_stress, dtype=np.float64)[mesh.elements]
    recovered = np.zeros((mesh.n_elements, 4, 3))
    for k, (xi, eta) in enumerate(GAUSS_2X2):
        recovered[:, k, :] = np.einsum("i,eic->ec", q4_shape(xi, eta), element_nodal)
    return recovered


def _check_compliance(compliance: np.ndarray) -> np.ndarray:
    """Reject a compliance matrix that cannot define a positive energy density."""
    c = np.asarray(compliance, dtype=np.float64)
    if c.shape != (3, 3):
        raise ValueError("compliance matrix must be 3 x 3")
    scale = max(1.0, float(np.max(np.abs(c))))
    if not np.allclose(c, c.T, atol=1e-12 * scale):
        raise ValueError("compliance matrix must be symmetric")
    leading = (float(c[0, 0]), float(c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]))
    determinant = float(np.einsum("ij,ij->", c, inverse3(c)) ** -1.0)
    if min(leading) <= 0.0 or determinant <= 0.0:
        raise ValueError("compliance matrix must be positive definite")
    return c


def element_indicators(
    mesh: QuadMesh, gauss_stress: np.ndarray, recovered_gauss: np.ndarray, dmat: np.ndarray
) -> np.ndarray:
    """ZZ element error indicators ``eta_e = ||sigma_h - sigma*||_{C^-1, Omega_e}``.

    ``sigma_h`` is sampled at the 2 x 2 Gauss points of the element (its superconvergent
    points) rather than at the element centre: for a bilinear element the stress is
    bilinear, so a single centre value would bias the indicator by O(h).
    """
    compliance = _check_compliance(inverse3(np.asarray(dmat, dtype=np.float64)))
    coordinates = mesh.coordinates()
    stress = np.asarray(gauss_stress, dtype=np.float64).reshape(mesh.n_elements, 4, 3)
    recovered = np.asarray(recovered_gauss, dtype=np.float64).reshape(mesh.n_elements, 4, 3)
    squared = np.zeros(mesh.n_elements)
    for k, (xi, eta) in enumerate(GAUSS_2X2):
        _, det = _strain_matrix(coordinates, xi, eta)
        delta = stress[:, k, :] - recovered[:, k, :]
        squared += np.einsum("ei,ij,ej->e", delta, compliance, delta) * det
    tolerance = 1e-12 * max(1.0, float(squared.max()))
    if np.any(squared < -tolerance):
        raise ValueError("negative element indicator energy: the ZZ density is not admissible")
    return np.sqrt(np.maximum(squared, 0.0))


def validate_indicators(eta, *, name: str = "indicator") -> np.ndarray:
    """Fail-closed gate: rejects negative, non-finite or empty indicator vectors."""
    values = np.asarray(eta, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite entries")
    negative = values < 0.0
    if np.any(negative):
        first = int(np.argmax(negative))
        raise ValueError(
            f"{name} contains {int(negative.sum())} negative entries "
            f"(first={values[first]:.6e}); an indicator is a norm and cannot be negative"
        )
    return values


def dorfler_mark(eta, fraction: float = 0.5) -> np.ndarray:
    """Doerfler (bulk) marking: the smallest set carrying ``fraction`` of the error energy."""
    values = validate_indicators(eta)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    energy = values**2
    total = float(energy.sum())
    if total <= 0.0:
        raise ValueError("all indicators vanish: nothing to mark")
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(energy[order]) / total
    count = int(np.searchsorted(cumulative, fraction) + 1)
    marked = np.zeros(values.size, dtype=bool)
    marked[order[:count]] = True
    return marked


def effectivity_index(estimated: float, exact: float) -> float:
    """``theta = estimated / exact`` energy norm ratio (asymptotic exactness score)."""
    exact_value = float(exact)
    if not np.isfinite(exact_value) or exact_value <= 0.0:
        raise ValueError("the exact energy norm must be finite and positive")
    return float(estimated) / exact_value


# ---------------------------------------------------------------------------
# exact error by graded quadrature (independent of the estimator)
# ---------------------------------------------------------------------------
def _element_measure(nsub: int, order: int):
    """Parent-space quadrature coordinates and subcell weights (``nsub x nsub`` subcells)."""
    n, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-1.0, 1.0, nsub + 1)
    local, weight = [], []
    for s in range(nsub):
        cx, hx = 0.5 * (edges[s] + edges[s + 1]), 0.5 * (edges[s + 1] - edges[s])
        for t in range(nsub):
            cy, hy = 0.5 * (edges[t] + edges[t + 1]), 0.5 * (edges[t + 1] - edges[t])
            for xi, wx in zip(cx + hx * n, w):
                for eta, wy in zip(cy + hy * n, w):
                    local.append((xi, eta))
                    weight.append(hx * hy * wx * wy)
    return np.asarray(local), np.asarray(weight)


def true_element_error_sq(
    mesh: QuadMesh,
    u: np.ndarray,
    dmat: np.ndarray,
    *,
    corner=(0.0, 0.0),
    near_factor: float = 4.0,
    max_subcells: int = 8,
    order: int = 3,
    lam: float | None = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Per-element **true** energy-norm error squared, by graded Gauss quadrature.

    ``sigma_h`` is evaluated at every quadrature point through ``B(x) u_e`` (a bilinear
    element does not have a constant stress), and the exact field is inserted at the same
    points.  The number of subcells per element is graded towards the corner so that the
    ``r^(2 lambda - 2)`` energy density of the singular element is resolved
    (``max_subcells`` caps the cost).  Nothing here is shared with the estimator: this is
    an independent measurement of the quantity the ZZ indicator estimates.
    """
    compliance = inverse3(np.asarray(dmat, dtype=np.float64))
    coordinates = mesh.coordinates()
    ue = np.asarray(u, dtype=np.float64)[element_dofs(mesh)]
    centres = mesh.centres()
    size = np.sqrt(mesh.areas())
    distance = np.hypot(centres[:, 0] - float(corner[0]), centres[:, 1] - float(corner[1]))
    ratio = near_factor * size / np.maximum(distance, 0.25 * size)
    nsub = np.clip(np.ceil(ratio).astype(int), 1, int(max_subcells))
    squared = np.zeros(mesh.n_elements)
    for count in np.unique(nsub):
        ids = np.nonzero(nsub == count)[0]
        local, weight = _element_measure(int(count), order)
        element_coordinates = coordinates[ids]
        total = np.zeros(len(ids))
        for q, (xi, eta) in enumerate(local):
            b, det = _strain_matrix(element_coordinates, xi, eta)
            shape = q4_shape(xi, eta)
            points = np.einsum("i,eia->ea", shape, element_coordinates)
            exact = np.einsum(
                "ij,ej->ei", compliance, williams_stress(points, lam=lam, amplitude=amplitude)
            )
            delta = np.einsum("eij,ej->ei", b, ue[ids]) - exact
            total += np.einsum("ei,ij,ej->e", delta, dmat, delta) * (weight[q] * det)
        squared[ids] = total
    return squared


# ---------------------------------------------------------------------------
# one refinement level
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelResult:
    """Everything measured on one mesh: solution, estimator, and independent truth."""

    n_elements: int
    n_nodes: int
    dofs: int
    hanging: int
    min_element_size: float
    indicators: np.ndarray
    estimated_norm: float
    true_norm: float
    exact_energy_norm: float
    fe_energy_norm: float
    effectivity: float
    nodal_stress: np.ndarray
    recovery_error: float
    recovery_error_far: float
    element_stress: np.ndarray
    displacement: np.ndarray
    strain_energy: float

    def record(self) -> dict:
        """JSON-serialisable summary (no arrays)."""
        return {
            "n_elements": self.n_elements,
            "n_nodes": self.n_nodes,
            "dofs": self.dofs,
            "hanging_nodes": self.hanging,
            "min_element_size": self.min_element_size,
            "estimated_energy_norm": self.estimated_norm,
            "true_energy_norm": self.true_norm,
            "exact_energy_norm": self.exact_energy_norm,
            "fe_energy_norm": self.fe_energy_norm,
            "effectivity": self.effectivity,
            "recovery_nodal_error": self.recovery_error,
            "recovery_nodal_error_smooth": self.recovery_error_far,
            "strain_energy": self.strain_energy,
        }


def solve_level(
    mesh: QuadMesh,
    *,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    amplitude: float = 1.0,
    thickness: float = THICKNESS,
    rtol: float = 1e-12,
    max_subcells: int = 8,
) -> LevelResult:
    """Solve one mesh with the exact Dirichlet data and measure estimator + true error."""
    if lam is None:
        lam = williams_exponent()
    dmat = elasticity_matrix(young, poisson, plane)
    prescribed = exact_dirichlet(
        mesh, lam=lam, young=young, poisson=poisson, plane=plane, amplitude=amplitude
    )
    displacement, stiffness = solve_displacement(
        mesh, dmat, dirichlet=prescribed, rtol=rtol, thickness=thickness
    )
    stress = element_stress(mesh, displacement, dmat)
    gauss_stress, gauss_points = gauss_point_stresses(mesh, displacement, dmat)
    nodal = spr_recover(mesh, gauss_stress, gauss_points)
    recovered = recover_at_gauss(mesh, nodal)
    eta = validate_indicators(element_indicators(mesh, gauss_stress, recovered, dmat))
    estimated = float(np.sqrt(np.sum(eta**2)))

    true_squared = true_element_error_sq(
        mesh, displacement, dmat, lam=lam, amplitude=amplitude, max_subcells=max_subcells
    )
    true_norm = float(np.sqrt(np.sum(true_squared)))

    exact_energy = exact_energy_norm_sq(
        lam=lam, young=young, poisson=poisson, plane=plane, amplitude=amplitude
    )
    u_t = torch.as_tensor(displacement, dtype=torch.float64)
    fe_energy = float(torch.dot(u_t, torch.sparse.mm(stiffness, u_t[:, None])[:, 0]))

    exact_nodal = williams_stress(mesh.nodes, lam=lam, amplitude=amplitude)
    radius = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    usable = radius > 1e-9  # the exact stress is singular at r = 0
    difference = (nodal - exact_nodal)[usable]
    reference = exact_nodal[usable]
    recovery_error = float(np.linalg.norm(difference) / np.linalg.norm(reference))
    smooth = radius > SMOOTH_RADIUS  # away from the corner the field is analytic
    recovery_error_far = float(
        np.linalg.norm((nodal - exact_nodal)[smooth]) / np.linalg.norm(exact_nodal[smooth])
    )

    return LevelResult(
        n_elements=mesh.n_elements,
        n_nodes=mesh.n_nodes,
        dofs=free_dofs(mesh, prescribed),
        hanging=len(hanging_nodes(mesh)),
        min_element_size=float(np.sqrt(mesh.areas()).min()),
        indicators=eta,
        estimated_norm=estimated,
        true_norm=true_norm,
        exact_energy_norm=float(np.sqrt(exact_energy)),
        fe_energy_norm=float(np.sqrt(max(fe_energy, 0.0))),
        effectivity=effectivity_index(estimated, true_norm),
        nodal_stress=nodal,
        recovery_error=recovery_error,
        recovery_error_far=recovery_error_far,
        element_stress=stress,
        displacement=displacement,
        strain_energy=0.5 * fe_energy,
    )


# ---------------------------------------------------------------------------
# the closed adaptive loop and the uniform reference ladder
# ---------------------------------------------------------------------------
def adaptive_study(
    *,
    divisions: int = 8,
    rounds: int = 2,
    fraction: float = 0.5,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    max_subcells: int = 8,
) -> dict:
    """Indicator-driven refinement loop: ``rounds`` adaptive steps from one start mesh.

    Level 0 is the starting mesh; every following level refines the elements selected by
    Doerfler bulk marking of the previous level's ZZ indicators.  All reported numbers are
    measured, and the true error is computed independently of the estimator.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    mesh = lshape_mesh(divisions)
    levels: list[LevelResult] = []
    meshes: list[QuadMesh] = [mesh]
    markings: list[dict] = []
    for step in range(rounds + 1):
        result = solve_level(
            mesh, lam=lam, young=young, poisson=poisson, plane=plane, max_subcells=max_subcells
        )
        levels.append(result)
        if step == rounds:
            break
        marked = dorfler_mark(result.indicators, fraction)
        centres = mesh.centres()
        distance = np.hypot(centres[:, 0], centres[:, 1])
        markings.append(
            {
                "step": step + 1,
                "marked_elements": int(marked.sum()),
                "marked_fraction": float(marked.mean()),
                "error_energy_captured": float(
                    np.sum(result.indicators[marked] ** 2) / np.sum(result.indicators**2)
                ),
                "max_marked_distance_to_corner": float(distance[marked].max()),
                "median_marked_distance_to_corner": float(np.median(distance[marked])),
                "min_marked_distance_to_corner": float(distance[marked].min()),
            }
        )
        mesh = refine_mesh(mesh, marked)
        meshes.append(mesh)
    return {
        "problem": "williams_lshape_reentrant_corner",
        "lam": float(williams_exponent() if lam is None else lam),
        "divisions": divisions,
        "rounds": rounds,
        "marking_fraction": fraction,
        "levels": [result.record() for result in levels],
        "markings": markings,
        "meshes": meshes,
        "results": levels,
    }


def uniform_study(
    divisions_list,
    *,
    lam: float | None = None,
    young: float = YOUNG,
    poisson: float = POISSON,
    plane: str = PLANE,
    max_subcells: int = 8,
) -> dict:
    """Uniform refinement ladder (no adaptivity) used as the efficiency reference."""
    levels: list[LevelResult] = []
    meshes: list[QuadMesh] = []
    for divisions in divisions_list:
        mesh = lshape_mesh(int(divisions))
        levels.append(
            solve_level(
                mesh, lam=lam, young=young, poisson=poisson, plane=plane, max_subcells=max_subcells
            )
        )
        meshes.append(mesh)
    return {
        "problem": "williams_lshape_reentrant_corner_uniform",
        "divisions": [int(d) for d in divisions_list],
        "levels": [result.record() for result in levels],
        "meshes": meshes,
        "results": levels,
    }


def convergence_rate(dofs, errors) -> float:
    """Least-squares slope of ``log(error)`` against ``log(dofs)``."""
    x = np.log(np.asarray(dofs, dtype=np.float64))
    y = np.log(np.asarray(errors, dtype=np.float64))
    if x.size < 2:
        raise ValueError("at least two levels are needed for a rate")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def matched_dof_table(adaptive: dict, uniform: dict) -> list[dict]:
    """Pair every adaptive level with the cheapest uniform level of at least the same DOFs.

    Equal unknown count is the only fair basis for an adaptivity gain: the adaptive mesh is
    allowed *at least* as much work as the uniform one it is compared against, so the
    reported ratio is a conservative lower bound on the gain.
    """
    uniform_dofs = np.array([level["dofs"] for level in uniform["levels"]], dtype=float)
    uniform_error = np.array([level["true_energy_norm"] for level in uniform["levels"]])
    uniform_divisions = np.array([int(d) for d in uniform["divisions"]])
    rows: list[dict] = []
    for index, level in enumerate(adaptive["levels"]):
        candidates = np.nonzero(uniform_dofs >= level["dofs"])[0]
        row = {
            "adaptive_level": index,
            "adaptive_mesh": "start" if index == 0 else f"adaptive-{index}",
            "adaptive_dofs": level["dofs"],
            "adaptive_error": level["true_energy_norm"],
            "adaptive_effectivity": level["effectivity"],
        }
        if candidates.size:
            pick = int(candidates[np.argmin(uniform_dofs[candidates])])
            row.update(
                {
                    "uniform_divisions": int(uniform_divisions[pick]),
                    "uniform_dofs": int(uniform_dofs[pick]),
                    "uniform_error": float(uniform_error[pick]),
                    "uniform_over_adaptive_dof_ratio": float(uniform_dofs[pick] / level["dofs"]),
                    "uniform_over_adaptive_error_ratio": float(
                        uniform_error[pick] / level["true_energy_norm"]
                    ),
                }
            )
        rows.append(row)
    return rows


def equal_error_table(adaptive: dict, uniform: dict) -> list[dict]:
    """How many uniform DOFs are needed to reach the accuracy of each adaptive level.

    The uniform ladder is interpolated in ``log(error)`` against ``log(dofs)`` (the
    measured asymptotic trend) to answer the practical question: *for the same accuracy,
    how much work does a uniform mesh need?*
    """
    dofs = np.array([level["dofs"] for level in uniform["levels"]], dtype=float)
    errors = np.array([level["true_energy_norm"] for level in uniform["levels"]])
    rows: list[dict] = []
    for index, level in enumerate(adaptive["levels"]):
        target = level["true_energy_norm"]
        row = {"adaptive_level": index, "adaptive_dofs": level["dofs"], "adaptive_error": target}
        inside = np.nonzero((target < errors) & (target > errors.min()))[0]
        if inside.size:
            upper = int(inside.min())  # errors is decreasing
            lower = upper - 1
            # log-log interpolation between two neighbouring uniform levels
            x1, x2 = np.log(dofs[lower]), np.log(dofs[upper])
            y1, y2 = np.log(errors[lower]), np.log(errors[upper])
            slope = (y2 - y1) / (x2 - x1)
            needed = float(np.exp(x1 + (np.log(target) - y1) / slope))
            row.update(
                {
                    "uniform_dofs_for_same_error": needed,
                    "uniform_dofs_ratio": needed / level["dofs"],
                    "uniform_local_rate": float(slope),
                }
            )
        rows.append(row)
    return rows


def study(
    *,
    divisions: int = 8,
    rounds: int = 2,
    fraction: float = 0.5,
    uniform_divisions=(8, 16, 24, 32, 48, 64),
    max_subcells: int = 8,
) -> dict:
    """Full adaptive-versus-uniform comparison with the acceptance numbers in one payload."""
    adaptive = adaptive_study(
        divisions=divisions, rounds=rounds, fraction=fraction, max_subcells=max_subcells
    )
    uniform = uniform_study(uniform_divisions, max_subcells=max_subcells)
    table = matched_dof_table(adaptive, uniform)
    equal = equal_error_table(adaptive, uniform)
    comparison = [row for row in table if "uniform_error" in row]
    return {
        "schema": "tensorfem.adaptive_zz.study/v1",
        "problem": adaptive["problem"],
        "williams_lambda": adaptive["lam"],
        "parameters": {
            "divisions": divisions,
            "rounds": rounds,
            "marking": "doerfler",
            "marking_fraction": fraction,
            "young": YOUNG,
            "poisson": POISSON,
            "plane": PLANE,
            "dirichlet": "exact williams trace on the outer square boundary",
            "notch_faces": "traction free (natural)",
        },
        "adaptive": {
            "levels": adaptive["levels"],
            "markings": adaptive["markings"],
            "rate": convergence_rate(
                [level["dofs"] for level in adaptive["levels"]],
                [level["true_energy_norm"] for level in adaptive["levels"]],
            ),
        },
        "uniform": {
            "divisions": uniform["divisions"],
            "levels": uniform["levels"],
            "rate": convergence_rate(
                [level["dofs"] for level in uniform["levels"]],
                [level["true_energy_norm"] for level in uniform["levels"]],
            ),
        },
        "matched_dof_table": table,
        "equal_error_table": equal,
        "criteria": {
            "two_rounds_executed": len(adaptive["levels"]) == rounds + 1,
            "final_adaptive_error": adaptive["levels"][-1]["true_energy_norm"],
            "final_adaptive_dofs": adaptive["levels"][-1]["dofs"],
            "final_uniform_error_at_matched_dofs": (
                comparison[-1]["uniform_error"] if comparison else None
            ),
            "error_ratio_uniform_over_adaptive": (
                comparison[-1]["uniform_over_adaptive_error_ratio"] if comparison else None
            ),
            "dof_ratio_uniform_over_adaptive": (
                comparison[-1]["uniform_over_adaptive_dof_ratio"] if comparison else None
            ),
            "adaptive_rate": convergence_rate(
                [level["dofs"] for level in adaptive["levels"]],
                [level["true_energy_norm"] for level in adaptive["levels"]],
            ),
            "uniform_rate": convergence_rate(
                [level["dofs"] for level in uniform["levels"]],
                [level["true_energy_norm"] for level in uniform["levels"]],
            ),
            "theory_uniform_rate": -0.5 * adaptive["lam"],
            "effectivity_range": [
                min(level["effectivity"] for level in adaptive["levels"]),
                max(level["effectivity"] for level in adaptive["levels"]),
            ],
            "final_effectivity": adaptive["levels"][-1]["effectivity"],
            "recovery_errors_smooth": [
                level["recovery_nodal_error_smooth"] for level in adaptive["levels"]
            ],
            "uniform_dofs_for_same_error": (
                equal[-1].get("uniform_dofs_for_same_error") if equal else None
            ),
            "dof_saving_at_equal_error": (equal[-1].get("uniform_dofs_ratio") if equal else None),
            "exact_energy_norm": adaptive["levels"][-1]["exact_energy_norm"],
            "fe_energy_norm": adaptive["levels"][-1]["fe_energy_norm"],
        },
        "_adaptive_internal": adaptive,
        "_uniform_internal": uniform,
    }
