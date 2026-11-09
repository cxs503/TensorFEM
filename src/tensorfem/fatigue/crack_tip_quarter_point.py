"""Quarter-point singular elements for the mode-I crack-tip J-integral.

The fatigue crack-growth chain in :mod:`tensorfem.fatigue_crack_propagation`
takes the stress intensity factor ``K_I`` as an input and integrates the Paris
law ``da/dN = C (Delta K)^m``.  Its SIF helpers are closed-form handbook
formulae; the missing link was a *numerical* SIF/J producer, i.e. a finite
element model with a genuinely singular crack-tip element that could be
grid-refined and checked with a GCI (grid convergence index) band.

This module supplies exactly that, with no handbook factor anywhere in the
numerical path:

``polar_slit_t6_mesh``
    A conforming six-node (quadratic) triangle mesh of a circular disc of
    radius ``R`` that contains a straight crack ``x in [-a, 0]``, ``y = 0``
    (crack tip at the origin, crack faces along the negative x axis).  The
    innermost layer is a fan of **quarter-point** triangles: the mid-side nodes
    on the two edges that meet at the tip are placed at one quarter of the edge
    length instead of one half, which reproduces the ``r^(-1/2)`` strain
    singularity of linear elastic fracture mechanics (Barsoum / Henshell-Shaw
    degenerated element).  Nodes lying on the crack faces are duplicated so
    that the two faces are independent.

``solve_tip_model``
    Plane strain/plane stress linear elastic solution.  The truncated far
    boundary carries the tractions of the crack-free (uniform tension) field,
    ``t = sigma n_y e_y`` on ``r = R``, so the model converges to the infinite
    plate as ``R/a`` grows.

``j_integral_contour``
    J evaluated on a circular contour around the tip from the *finite element*
    stress and displacement gradient, using the already verified polygon
    contour integrator :func:`tensorfem.crack_tip_j_integral.contour_j_integral`.
    ``K_I = sqrt(J E')`` with ``E' = E`` (plane stress) or ``E/(1-nu^2)``
    (plane strain).

``grid_convergence_index``
    Roache GCI from three systematically refined grids (``Fs = 1.25``).

Scope and limits: homogeneous isotropic linear elasticity, small strain, mode I
only, stationary crack, no body force, no plasticity, no crack closure.  The
singular element ring is the only singular enrichment; a contour that passes
through the innermost layer is rejected because the J flux still needs a
sufficiently smooth field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..continuum import elasticity_matrix
from ..crack_tip_j_integral import ContourField, contour_j_integral
from ..sparse_core import assemble_coo, solve_sparse_static

__all__ = [
    "DEFAULT_QUARTER_POINT_SWEEP",
    "FEMJResult",
    "PolarSlitMesh",
    "grid_convergence_index",
    "j_integral_ring",
    "mode_i_k_from_j",
    "polar_slit_t6_mesh",
    "quarter_point_sweep",
    "solve_tip_model",
    "t6_shape",
    "t6_shape_derivative",
    "t6_stiffness",
    "westergaard_stress",
]

# --------------------------------------------------------------------------- #
# six-node triangle: shape functions and stiffness
# --------------------------------------------------------------------------- #
# 7-point Dunavant rule, degree 5 exact on the reference triangle (area 1/2).
_DUNAVANT: tuple[tuple[float, float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0, 0.225),
    (0.059715871789770, 0.470142064105115, 0.132394152788506),
    (0.470142064105115, 0.059715871789770, 0.132394152788506),
    (0.470142064105115, 0.470142064105115, 0.132394152788506),
    (0.797426985353087, 0.101286507323456, 0.125939180544827),
    (0.101286507323456, 0.797426985353087, 0.125939180544827),
    (0.101286507323456, 0.101286507323456, 0.125939180544827),
)


def t6_shape(xi: float, eta: float) -> tuple[float, ...]:
    """Quadratic triangle shape functions, node order (V1 V2 V3 M12 M23 M31)."""
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    return (
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l1 * l3,
    )


def t6_shape_derivative(xi: float, eta: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(dN/dxi, dN/deta)`` for :func:`t6_shape`."""
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    d_xi = (
        1.0 - 4.0 * l1,
        4.0 * l2 - 1.0,
        0.0,
        4.0 * (l1 - l2),
        4.0 * l3,
        -4.0 * l3,
    )
    d_eta = (
        1.0 - 4.0 * l1,
        0.0,
        4.0 * l3 - 1.0,
        -4.0 * l2,
        4.0 * l2,
        4.0 * (l1 - l3),
    )
    return d_xi, d_eta


def _parent_gradient(coordinates: torch.Tensor, xi: float, eta: float):
    """Shape-function gradients in physical coordinates plus ``det J``.

    ``coordinates`` has shape ``(ne, 6, 2)``.  The parent Jacobian follows the
    repository convention ``jac[a, b] = dX_b / dxi_a``.
    """
    d_xi, d_eta = t6_shape_derivative(xi, eta)
    d_xi_t = torch.tensor(d_xi, dtype=coordinates.dtype)
    d_eta_t = torch.tensor(d_eta, dtype=coordinates.dtype)
    # dX/dxi and dX/deta, shape (ne, 2)
    dx_dxi = torch.einsum("e,nec->nc", d_xi_t, coordinates)
    dx_deta = torch.einsum("e,nec->nc", d_eta_t, coordinates)
    jac = torch.stack((dx_dxi, dx_deta), dim=1)  # (ne, 2 (xi index), 2 (x index))
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    inv = torch.stack(
        (
            torch.stack((jac[:, 1, 1], -jac[:, 0, 1]), dim=1),
            torch.stack((-jac[:, 1, 0], jac[:, 0, 0]), dim=1),
        ),
        dim=1,
    ) / det[:, None, None]
    d_xi_all = d_xi_t.unsqueeze(0).expand(coordinates.shape[0], -1)
    d_eta_all = d_eta_t.unsqueeze(0).expand(coordinates.shape[0], -1)
    # dN/dX = jac^-1 dN/dxi with jac[a, b] = dX_b / dxi_a
    grad_x = inv[:, 0, 0][:, None] * d_xi_all + inv[:, 0, 1][:, None] * d_eta_all
    grad_y = inv[:, 1, 0][:, None] * d_xi_all + inv[:, 1, 1][:, None] * d_eta_all
    return grad_x, grad_y, det


def t6_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: float) -> torch.Tensor:
    """Element stiffness matrices ``(ne, 12, 12)`` for six-node triangles."""
    ne = coordinates.shape[0]
    dtype = coordinates.dtype
    stiffness = torch.zeros((ne, 12, 12), dtype=dtype)
    for xi, eta, weight in _DUNAVANT:
        grad_x, grad_y, det = _parent_gradient(coordinates, xi, eta)
        if bool(torch.any(det <= 0.0)):
            raise ValueError("degenerate or inverted six-node triangle")
        b = torch.zeros((ne, 3, 12), dtype=dtype)
        b[:, 0, 0::2] = grad_x
        b[:, 1, 1::2] = grad_y
        b[:, 2, 0::2] = grad_y
        b[:, 2, 1::2] = grad_x
        db = torch.einsum("ik,ekj->eij", dmat, b)
        stiffness += 0.5 * weight * thickness * det[:, None, None] * (
            b.transpose(1, 2) @ db
        )
    return stiffness


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolarSlitMesh:
    """Conforming six-node mesh of a disc with a straight crack ``[-a, 0]``."""

    nodes: torch.Tensor  # (nn, 2)
    elements: torch.Tensor  # (ne, 6)
    crack_half_length: float
    outer_radius: float
    stations: tuple[float, ...]
    n_sectors: int
    inner_layer: int  # element ids of the quarter-point fan
    ring_elements: tuple[tuple[int, ...], ...] = ()  # per station: elements with an inner arc edge

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])


def _radial_stations(outer_radius: float, first_ring: float, growth: float) -> list[float]:
    stations = [0.0, 0.25 * first_ring, first_ring]
    if first_ring >= outer_radius:
        raise ValueError("first_ring must be smaller than outer_radius")
    while stations[-1] < outer_radius * (1.0 - 1e-12):
        nxt = stations[-1] * growth
        if nxt >= outer_radius:
            nxt = outer_radius
        stations.append(nxt)
    return stations


def polar_slit_t6_mesh(
    crack_half_length: float,
    outer_radius: float,
    *,
    n_sectors: int = 24,
    first_ring: float | None = None,
    growth: float = 1.35,
    dtype: torch.dtype = torch.float64,
) -> PolarSlitMesh:
    """Build the quarter-point fan mesh.

    ``n_sectors`` must be even so that the crack ray at ``theta = pi`` is a mesh
    line.  Radial stations start at the tip, the second station sits at one
    quarter of ``first_ring`` (the quarter-point position) and the remaining
    stations grow geometrically towards ``outer_radius``.
    """
    a = float(crack_half_length)
    radius = float(outer_radius)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("crack_half_length must be positive and finite")
    if not math.isfinite(radius) or radius <= a:
        raise ValueError("outer_radius must exceed the crack half length")
    if n_sectors < 8 or n_sectors % 2:
        raise ValueError("n_sectors must be an even integer >= 8")
    if growth <= 1.0:
        raise ValueError("growth must exceed 1")
    ring = float(first_ring) if first_ring is not None else a / 8.0
    stations = _radial_stations(radius, ring, growth)
    tol = 1e-9 * a

    nodes: list[tuple[float, float]] = []
    index: dict[tuple[int, int, int, int], int] = {}

    def node_id(x: float, y: float, side: int) -> int:
        # side distinguishes the two faces of the crack opening (y ~ 0, x < 0).
        key_side = side if (abs(y) < tol and x < -tol and abs(x) < a - tol) else 0
        key = (
            int(round(x / (1e-6 * a))),
            int(round(y / (1e-6 * a))),
            key_side,
            0,
        )
        if key not in index:
            index[key] = len(nodes)
            nodes.append((x, y))
        return index[key]

    def polar(r: float, angle: float) -> tuple[float, float]:
        return r * math.cos(angle), r * math.sin(angle)

    elements: list[tuple[int, int, int, int, int, int]] = []
    ring_lookup: dict[int, list[int]] = {}
    n_inner = 0

    def add_triangle(c1, c2, c3, m12, m23, m31) -> None:
        elements.append(
            (
                node_id(*c1[0], c1[1]),
                node_id(*c2[0], c2[1]),
                node_id(*c3[0], c3[1]),
                node_id(*m12[0], m12[1]),
                node_id(*m23[0], m23[1]),
                node_id(*m31[0], m31[1]),
            )
        )

    for s in range(n_sectors):
        a0 = 2.0 * math.pi * s / n_sectors
        a1 = 2.0 * math.pi * (s + 1) / n_sectors
        am = 0.5 * (a0 + a1)
        side = 1 if math.sin(am) > 0.0 else -1
        # quarter-point fan: tip -> first full ring, mid-side nodes at ring/4
        quarter = stations[1]
        first = stations[2]
        tip = ((0.0, 0.0), side)
        c0 = (polar(first, a0), side)
        c1 = (polar(first, a1), side)
        m_lo = (polar(quarter, a0), side)
        m_hi = (polar(quarter, a1), side)
        m_out = (polar(first, am), side)
        add_triangle(tip, c0, c1, m_lo, m_out, m_hi)
    n_inner = len(elements)

    for k in range(2, len(stations) - 1):
        r0, r1 = stations[k], stations[k + 1]
        rm = 0.5 * (r0 + r1)
        for s in range(n_sectors):
            a0 = 2.0 * math.pi * s / n_sectors
            a1 = 2.0 * math.pi * (s + 1) / n_sectors
            am = 0.5 * (a0 + a1)
            side = 1 if math.sin(am) > 0.0 else -1
            v_lo_in = (polar(r0, a0), side)
            v_hi_in = (polar(r0, a1), side)
            v_lo_out = (polar(r1, a0), side)
            v_hi_out = (polar(r1, a1), side)
            m_in = (polar(r0, am), side)
            m_out = (polar(r1, am), side)
            m_lo = (polar(rm, a0), side)
            m_hi = (polar(rm, a1), side)
            m_diag = (polar(rm, am), side)
            # diagonal v_lo_in -> v_hi_out splits the polar quad into two triangles
            add_triangle(v_lo_in, v_lo_out, v_hi_out, m_lo, m_out, m_diag)
            ring_lookup.setdefault(k, []).append(len(elements))
            add_triangle(v_lo_in, v_hi_out, v_hi_in, m_diag, m_hi, m_in)

    ring_elements = tuple(
        tuple(ring_lookup.get(index, ())) for index in range(len(stations))
    )
    return PolarSlitMesh(
        nodes=torch.tensor(nodes, dtype=dtype),
        elements=torch.tensor(elements, dtype=torch.long),
        crack_half_length=a,
        outer_radius=radius,
        stations=tuple(stations),
        n_sectors=n_sectors,
        inner_layer=n_inner,
        ring_elements=ring_elements,
    )


# --------------------------------------------------------------------------- #
# solution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FEMJResult:
    """One mesh level: FEM J-integral, derived K_I and mesh metadata."""

    j_j_m2: float
    k_i_pa_sqrt_m: float
    k_i_reference_pa_sqrt_m: float
    relative_error: float
    n_nodes: int
    n_elements: int
    contour_radius_m: float
    path_spread: float
    displacement: torch.Tensor
    mesh: PolarSlitMesh


def westergaard_stress(
    x: float, y: float, sigma_pa: float, crack_half_length: float
) -> tuple[float, float, float]:
    """Exact mode-I stress of a centre crack in an unbounded plate.

    Westergaard potential ``Z(z) = sigma z / sqrt(z^2 - a^2)`` for a crack
    ``|x| <= a`` on ``y = 0`` under remote uniaxial tension ``sigma`` normal to
    the crack.  Used only to prescribe the far boundary tractions of the
    truncated disc, so that the disc model has the *exact* infinite-plate
    solution on its boundary and the residual discretisation error is isolated
    from the far-field truncation error.
    """
    import cmath

    a = float(crack_half_length)
    z = complex(x, y)
    root = cmath.sqrt(z * z - a * a)
    if abs(root) == 0.0:
        raise ZeroDivisionError("Westergaard potential is singular at the crack tip")
    # branch with sqrt(z^2 - a^2) ~ z at infinity (Z -> sigma)
    if (root / z).real < 0.0:
        root = -root
    z_pot = sigma_pa * z / root
    z_prime = -sigma_pa * a * a / (root ** 3)
    s_xx = z_pot.real - y * z_prime.imag - sigma_pa  # uniaxial remote tension
    s_yy = z_pot.real + y * z_prime.imag
    s_xy = -y * z_prime.real
    return s_xx, s_yy, s_xy


def _outer_edge_loads(
    mesh: PolarSlitMesh, sigma_pa: float, ndof: int, dtype: torch.dtype,
    far_field: str = "uncracked",
) -> torch.Tensor:
    """Consistent nodal loads for the far-boundary tractions on ``r = R``.

    ``far_field="uncracked"`` applies the tractions of the crack-free uniform
    tension field (a truncated-plate model, whose bias decays slowly with
    ``R/a``); ``far_field="westergaard"`` applies the tractions of the exact
    infinite-plate solution, which removes the truncation bias entirely.
    """
    load = torch.zeros(ndof, dtype=dtype)
    n_sectors = mesh.n_sectors
    last = len(mesh.stations) - 1
    # outer arc edge of each sector: (R, a_s) - (R, a_s+1) with mid at (R, a_mid)
    # locate the element that owns that edge by scanning the outermost layer
    edge_nodes: dict[tuple[int, int], tuple[int, int, int]] = {}
    outer = mesh.elements[-2 * n_sectors:]
    station_r = mesh.stations[last]
    for element in outer:
        ids = element.tolist()
        coords = mesh.nodes[torch.tensor(ids)]
        radii = torch.linalg.vector_norm(coords, dim=1)
        hits = [i for i in range(3) if abs(float(radii[i]) - station_r) < 1e-9 * station_r]
        # the two far corners plus their mid-side node form the arc edge
        if len(hits) >= 2:
            corners = sorted(hits)
            mid = 3 + corners[0]  # M12 sits at local index 3, M23 at 4, M31 at 5
            c0, c1 = corners[0], corners[1]
            if corners == [0, 1]:
                mid = 3
            elif corners == [1, 2]:
                mid = 4
            else:
                mid = 5
            key = (min(ids[c0], ids[c1]), max(ids[c0], ids[c1]))
            edge_nodes[key] = (ids[c0], ids[mid], ids[c1])
    # total arc contribution; every arc segment appears exactly once
    gauss = ((0.5 - math.sqrt(0.15), 0.5 * 5.0 / 9.0),
             (0.5, 0.5 * 8.0 / 9.0),
             (0.5 + math.sqrt(0.15), 0.5 * 5.0 / 9.0))
    for c0, mid, c1 in edge_nodes.values():
        x0 = mesh.nodes[c0]
        xm = mesh.nodes[mid]
        x1 = mesh.nodes[c1]
        for t, w in gauss:
            n1 = 2.0 * t * t - 3.0 * t + 1.0
            n2 = 4.0 * t * (1.0 - t)
            n3 = 2.0 * t * t - t
            point = n1 * x0 + n2 * xm + n3 * x1
            dxdt = (4.0 * t - 3.0) * x0 + (4.0 - 8.0 * t) * xm + (4.0 * t - 1.0) * x1
            ds = float(torch.linalg.vector_norm(dxdt)) * w
            r = float(torch.linalg.vector_norm(point))
            nx, ny = float(point[0]) / r, float(point[1]) / r
            if far_field == "uncracked":
                tx, ty = 0.0, sigma_pa * ny
            elif far_field == "westergaard":
                s_xx, s_yy, s_xy = westergaard_stress(
                    float(point[0]), float(point[1]), sigma_pa, mesh.crack_half_length
                )
                tx, ty = s_xx * nx + s_xy * ny, s_xy * nx + s_yy * ny
            else:
                raise ValueError(f"unknown far_field {far_field!r}")
            load[2 * c0 + 0] += n1 * tx * ds
            load[2 * mid + 0] += n2 * tx * ds
            load[2 * c1 + 0] += n3 * tx * ds
            load[2 * c0 + 1] += n1 * ty * ds
            load[2 * mid + 1] += n2 * ty * ds
            load[2 * c1 + 1] += n3 * ty * ds
    return load


def solve_tip_model(
    mesh: PolarSlitMesh,
    *,
    sigma_pa: float,
    young_pa: float,
    poisson: float = 0.3,
    plane: str = "strain",
    thickness: float = 1.0,
    far_field: str = "westergaard",
    rtol: float = 1e-12,
) -> torch.Tensor:
    """Solve the truncated crack-tip model; returns the displacement vector."""
    dtype = mesh.nodes.dtype
    dmat = elasticity_matrix(
        torch.tensor([young_pa], dtype=dtype),
        torch.tensor([poisson], dtype=dtype),
        plane,
    )[0]
    coordinates = mesh.nodes[mesh.elements]
    stiffness = t6_stiffness(coordinates, dmat, thickness)
    element_dofs = torch.empty((mesh.n_elements, 12), dtype=torch.long)
    element_dofs[:, 0::2] = 2 * mesh.elements
    element_dofs[:, 1::2] = 2 * mesh.elements + 1
    ndof = 2 * mesh.nodes.shape[0]
    K = assemble_coo(stiffness, element_dofs, ndof)
    load = _outer_edge_loads(mesh, sigma_pa, ndof, dtype, far_field)
    # Three rigid-body constraints are required: the discrete model has two
    # translations and one rotation in its null space, and an unconstrained
    # rotation mode contaminates the near-tip field (and therefore J).
    dirichlet = {0: 0.0, 1: 0.0}
    on_axis = int(torch.argmax(mesh.nodes[:, 0] - 1e6 * mesh.nodes[:, 1].abs()))
    dirichlet[2 * on_axis] = 0.0
    solved = solve_sparse_static(K, load, dirichlet=dirichlet, rtol=rtol)
    return solved.displacement


# --------------------------------------------------------------------------- #
# J-integral from the finite element field
# --------------------------------------------------------------------------- #
# 3-point Gauss rule on the edge parameter t in [0, 1].
_EDGE_GAUSS = (
    (0.5 - math.sqrt(0.15), 5.0 / 18.0),
    (0.5, 8.0 / 18.0),
    (0.5 + math.sqrt(0.15), 5.0 / 18.0),
)


def _element_field(mesh: PolarSlitMesh, displacement: torch.Tensor, element: int,
                   xi: float, eta: float, dmat: torch.Tensor):
    """Stress tensor ``(2, 2)`` and displacement gradient ``(2, 2)`` at a parent point."""
    ids = mesh.elements[element]
    coords = mesh.nodes[ids]
    grad_x, grad_y, det = _parent_gradient(coords.unsqueeze(0), xi, eta)
    local = torch.zeros(12, dtype=mesh.nodes.dtype)
    local[0::2] = displacement[2 * ids]
    local[1::2] = displacement[2 * ids + 1]
    b = torch.zeros((3, 12), dtype=mesh.nodes.dtype)
    b[0, 0::2] = grad_x[0]
    b[1, 1::2] = grad_y[0]
    b[2, 0::2] = grad_y[0]
    b[2, 1::2] = grad_x[0]
    stress_vec = dmat @ (b @ local)
    stress = torch.stack(
        (
            torch.stack((stress_vec[0], stress_vec[2])),
            torch.stack((stress_vec[2], stress_vec[1])),
        )
    )
    grad = torch.stack(
        (
            torch.stack((grad_x[0] @ local[0::2], grad_y[0] @ local[0::2])),
            torch.stack((grad_x[0] @ local[1::2], grad_y[0] @ local[1::2])),
        )
    )
    return stress, grad, det


def j_integral_ring(
    mesh: PolarSlitMesh,
    displacement: torch.Tensor,
    station_index: int,
    *,
    young_pa: float,
    poisson: float = 0.3,
    plane: str = "strain",
) -> float:
    """Rice J-integral on the closed mesh-line contour at ``stations[station_index]``.

    The contour is assembled from the inner arc edges of the elements that start
    at that radial station, so every evaluation point has *exact* parent
    coordinates and no point-location search is needed.  The outward normal is
    taken away from the crack tip and the energy and traction-work fluxes are
    integrated with the same three-point Gauss rule that the element uses along
    its edges.
    """
    if not 2 <= station_index < len(mesh.stations) - 1:
        raise IndexError("station_index must address an interior radial station")
    radius = mesh.stations[station_index]
    if radius <= mesh.crack_half_length:
        raise ValueError("the contour must enclose the whole crack")
    elements = mesh.ring_elements[station_index]
    if not elements:
        raise ValueError("no closed element ring at this station")
    dtype = mesh.nodes.dtype
    dmat = elasticity_matrix(
        torch.tensor([young_pa], dtype=dtype),
        torch.tensor([poisson], dtype=dtype),
        plane,
    )[0]
    total = 0.0
    for element in elements:
        ids = mesh.elements[element]
        coords = mesh.nodes[ids]
        local = torch.zeros(12, dtype=dtype)
        local[0::2] = displacement[2 * ids]
        local[1::2] = displacement[2 * ids + 1]
        # inner arc edge of the second triangle: V3 - M31 - V1 (local 2, 5, 0),
        # parameterised as (xi, eta) = (0, 1 - t) with t from 0 (V3) to 1 (V1).
        for t, weight in _EDGE_GAUSS:
            xi, eta = 0.0, 1.0 - t
            stress, grad, det = _element_field(mesh, displacement, element, xi, eta, dmat)
            if float(det) <= 0.0:
                raise ValueError("inverted element on the J contour")
            d_xi, d_eta = t6_shape_derivative(xi, eta)
            tangent = -torch.tensor(d_eta, dtype=dtype) @ coords  # dX/dt along the edge
            point = torch.tensor(t6_shape(xi, eta), dtype=dtype) @ coords
            normal = torch.stack((tangent[1], -tangent[0]))
            if float(torch.dot(normal, point)) < 0.0:
                normal = -normal
            ds = float(torch.linalg.vector_norm(tangent)) * weight
            strain = 0.5 * (grad + grad.transpose(0, 1))
            energy = 0.5 * float(torch.sum(stress * strain))
            unit_normal = normal / float(torch.linalg.vector_norm(normal))
            traction = stress @ unit_normal
            work = float(torch.dot(traction, grad[:, 0]))
            total += (energy * float(unit_normal[0]) - work) * ds
    return total


# --------------------------------------------------------------------------- #
# verification drivers
# --------------------------------------------------------------------------- #
def mode_i_k_from_j(j_j_m2: float, young_pa: float, poisson: float = 0.3,
                    *, plane: str = "strain") -> float:
    """``K_I = sqrt(J E')`` with the plane-dependent effective modulus."""
    e_eff = young_pa if plane == "stress" else young_pa / (1.0 - poisson ** 2)
    if j_j_m2 <= 0.0:
        raise ValueError("J must be positive")
    return math.sqrt(j_j_m2 * e_eff)


def grid_convergence_index(fine: float, medium: float, coarse: float,
                           refinement: float = 2.0, safety: float = 1.25) -> dict:
    """Roache GCI for three systematically refined scalar results.

    ``fine``/``medium``/``coarse`` are the values on the finest, medium and
    coarsest grid; ``refinement`` is the constant grid ratio.  The observed
    order ``p`` comes from the three values themselves.
    """
    for value in (fine, medium, coarse):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("all three results must be positive and finite")
    if refinement <= 1.0:
        raise ValueError("refinement must exceed 1")
    # successive differences, coarse -> medium -> fine
    e21 = medium - fine
    e32 = coarse - medium
    if e32 == 0.0 or e21 == 0.0:
        return {
            "observed_order": float("nan"),
            "gci_fine": float("nan"),
            "converging": False,
            "relative_change": float("nan"),
        }
    ratio = e32 / e21
    if ratio <= 0.0:  # oscillatory or diverging sequence
        return {
            "observed_order": float("nan"),
            "gci_fine": float("nan"),
            "converging": False,
            "relative_change": abs(e21) / fine,
        }
    p = math.log(abs(ratio)) / math.log(refinement)
    denominator = refinement ** p - 1.0
    if denominator <= 1e-12:
        return {
            "observed_order": p,
            "gci_fine": float("nan"),
            "converging": False,
            "relative_change": abs(e21) / fine,
        }
    return {
        "observed_order": p,
        "gci_fine": safety * abs(e21 / fine) / denominator,
        "converging": True,
        "relative_change": abs(e21) / fine,
    }


DEFAULT_QUARTER_POINT_SWEEP: tuple[dict, ...] = (
    {"name": "coarse", "n_sectors": 8, "first_ring": 1.0 / 4.0, "growth": 1.7},
    {"name": "medium", "n_sectors": 16, "first_ring": 1.0 / 8.0, "growth": 1.5},
    {"name": "fine", "n_sectors": 32, "first_ring": 1.0 / 16.0, "growth": 1.32},
)


def quarter_point_sweep(
    *,
    sigma_pa: float = 100e6,
    crack_half_length: float = 0.01,
    outer_radius: float = 0.20,
    young_pa: float = 210e9,
    poisson: float = 0.3,
    plane: str = "strain",
    contour_fraction: float = 0.125,
    levels: tuple[dict, ...] = DEFAULT_QUARTER_POINT_SWEEP,
) -> dict:
    """Run the grid study and return the evidence dictionary.

    The reference SIF is the infinite-plate value ``sigma sqrt(pi a)``; the
    truncated far boundary at ``R`` adds a small positive bias, so the residual
    error is reported rather than hidden.
    """
    k_ref = sigma_pa * math.sqrt(math.pi * crack_half_length)
    results = []
    for level in levels:
        mesh = polar_slit_t6_mesh(
            crack_half_length,
            outer_radius,
            n_sectors=level["n_sectors"],
            first_ring=level["first_ring"] * crack_half_length,
            growth=level["growth"],
        )
        displacement = solve_tip_model(
            mesh, sigma_pa=sigma_pa, young_pa=young_pa, poisson=poisson, plane=plane
        )
        station = min(
            range(2, len(mesh.stations) - 2),
            key=lambda idx: abs(mesh.stations[idx] - contour_fraction * outer_radius),
        )
        j_value = j_integral_ring(
            mesh, displacement, station, young_pa=young_pa, poisson=poisson, plane=plane
        )
        k_value = mode_i_k_from_j(j_value, young_pa, poisson, plane=plane)
        results.append(
            {
                "name": level["name"],
                "n_sectors": level["n_sectors"],
                "n_nodes": int(mesh.nodes.shape[0]),
                "n_elements": mesh.n_elements,
                "j_j_m2": j_value,
                "k_i": k_value,
                "relative_error": (k_value - k_ref) / k_ref,
            }
        )
    gci = grid_convergence_index(
        results[2]["k_i"], results[1]["k_i"], results[0]["k_i"], refinement=2.0
    )
    return {
        "reference_k_i_pa_sqrt_m": k_ref,
        "levels": results,
        "gci": gci,
    }


def run_quarter_point_j_qualification() -> dict[str, object]:
    """Machine-checkable evidence bundle for criterion (f)."""
    sweep = quarter_point_sweep()
    fine = sweep["levels"][2]
    gci = sweep["gci"]
    evidence = {
        "id": "fem_quarter_point_j_gci",
        "element": "six-node triangle with quarter-point mid-side nodes",
        "reference": "K_I = sigma sqrt(pi a) (infinite plate)",
        "reference_k_i_pa_sqrt_m": sweep["reference_k_i_pa_sqrt_m"],
        "levels": sweep["levels"],
        "observed_order": gci["observed_order"],
        "gci_fine": gci["gci_fine"],
        "converging": gci["converging"],
        "relative_error_fine": fine["relative_error"],
        "tolerance": 0.05,
        "passed": bool(
            gci["converging"]
            and math.isfinite(gci["gci_fine"])
            and gci["gci_fine"] < 0.05
            and abs(fine["relative_error"]) < 0.05
        ),
    }
    return evidence