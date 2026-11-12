"""Single-edge-notched tension (SENT) plate: numerical ``K_I`` from a
quarter-point quadratic-triangle mesh and the contour J-integral.

Why this module exists
----------------------
:mod:`tensorfem.fatigue.crack_tip_quarter_point` supplies a numerical SIF
producer for a *crack in a disc*, i.e. the infinite-plate idealisation, and its
regression gate on the recovered ``K_I`` is deliberately loose (a factor-two
band) because the truncated circular far boundary biases the result.  The case
pool anchor ``CASE-FEM-JINTEGRAL-EDGE-CRACK`` asks for the complementary
configuration: an **edge crack in a finite-width strip**, where the free surface
behind the crack mouth and the finite width both act, and where a handbook
reference with sub-percent accuracy exists.  This module provides that
configuration end to end:

``sent_t6_mesh``
    A conforming six-node (quadratic) triangle mesh of the *upper half* of the
    SENT specimen.  The crack lies on the symmetry plane ``y = 0`` and runs from
    the free edge ``x = -a`` to the tip at the origin; the material occupies
    ``x in [-a, W-a]``, ``y in [0, H]``.  The mesh is a ray fan from the crack
    tip: ray ``j`` leaves the tip at ``theta_j`` and terminates on the
    rectangular outer boundary, and the radial stations are the *same
    normalised fractions* ``s_k`` on every ray, so a fixed station index is a
    closed, tip-encircling mesh line (used as the J contour) at every angle.
    The innermost ring places its mid-side nodes at one quarter of the edge
    length (Barsoum / Henshell-Shaw degenerated element), which reproduces the
    ``r^(-1/2)`` strain singularity of LEFM.  ``quarter_point=False`` keeps the
    mid-side nodes at the midpoint and is kept as a *control*: an isoparametric
    (non-singular) mesh must **not** recover the handbook stress intensity
    factor.

``solve_sent``
    Plane stress / plane strain linear elastic solution.  Uniform tension
    ``sigma`` is applied to the loaded edge ``y = H``; the free edge ``x = -a``,
    the crack face and the far edge ``x = W - a`` are natural (traction-free)
    boundaries; ``u_y = 0`` is imposed on the symmetry plane ahead of the crack.
    The half model is justified because the SENT geometry *and* the loading are
    symmetric about ``y = 0``.  The global vertical reaction is reported so the
    discrete equilibrium can be audited independently of the J integral.

``j_integral_subring``
    Rice's J-integral on the mesh line at a given station, evaluated from the
    *finite element* stress and displacement gradient with the three-point
    Gauss rule along the quadratic mesh edges, so every evaluation point has
    exact parent coordinates and no point-location search is needed.  Only the
    upper half of the contour is integrated and the result is doubled: on the
    full contour the two crack faces are traction-free and contribute nothing,
    while the upper and lower arcs are mirror images (``n_x``, ``W`` and
    ``t.u_,x`` are all mirror invariant).

``tada_sent_shape_factor`` / ``sent_k_reference``
    The handbook reference, ``K_I = sigma sqrt(pi a) F(a/W)`` with
    ``F = 1.12 - 0.231 alpha + 10.55 alpha^2 - 21.72 alpha^3 + 30.39 alpha^4``
    (Tada, Paris & Irwin, *The Stress Analysis of Cracks Handbook*, 3rd ed.,
    single-edge-notch tension, ``H/W >= 1``, ``alpha <= 0.6``, stated accuracy
    +-0.5 %).

``sent_convergence`` / ``sent_sweep`` / ``sent_gci``
    The graded verification drivers that produce the evidence dictionary used by
    ``benchmarks/verified/fem_sent_j_integral/run.py``.

Scope and limits: homogeneous isotropic linear elasticity, small strain, mode I
only, stationary crack, no body force, no plasticity, no crack closure, no
cohesive traction on the crack faces.  The singular element ring is the only
enrichment; a contour that passes through the innermost ring is rejected because
the J flux needs a field that is at least partly resolved.  The finite height
``H`` truncates the strip, so the residual ``H`` sensitivity is *measured* by
:func:`sent_sweep` rather than assumed away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .continuum import elasticity_matrix
from .sparse_core import assemble_coo, solve_sparse_static
from .fatigue.crack_tip_quarter_point import (
    grid_convergence_index,
    mode_i_k_from_j,
    t6_shape,
    t6_shape_derivative,
    t6_stiffness,
)

__all__ = [
    "DEFAULT_SENT_LEVELS",
    "SentMesh",
    "SentSolve",
    "sent_t6_mesh",
    "solve_sent",
    "element_field_at",
    "edge_geometry",
    "edge_parent",
    "j_integral_subring",
    "tada_sent_shape_factor",
    "sent_k_reference",
    "sent_k_from_j",
    "sent_convergence",
    "sent_sweep",
    "sent_gci",
]

# 3-point Gauss rule on the edge parameter t in [0, 1] (exact for degree 5).
_EDGE_GAUSS: tuple[tuple[float, float], ...] = (
    (0.5 - math.sqrt(0.15), 5.0 / 18.0),
    (0.5, 8.0 / 18.0),
    (0.5 + math.sqrt(0.15), 5.0 / 18.0),
)

# local midside node of each corner edge of the six-node triangle
_EDGE_MIDSIDE = (3, 4, 5)


# --------------------------------------------------------------------------- #
# handbook reference
# --------------------------------------------------------------------------- #
def tada_sent_shape_factor(alpha: float) -> float:
    """Geometry factor ``F(a/W)`` of the single-edge-notch tension specimen.

    ``K_I = sigma sqrt(pi a) F(a/W)`` with the polynomial fit of Tada, Paris &
    Irwin (3rd ed.), quoted for ``alpha = a/W <= 0.6`` and ``H/W >= 1`` with a
    stated accuracy of +-0.5 %.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha = a/W must be positive and finite")
    if alpha > 0.6:
        raise ValueError("the Tada fit is quoted for a/W <= 0.6")
    return (
        1.12
        - 0.231 * alpha
        + 10.55 * alpha ** 2
        - 21.72 * alpha ** 3
        + 30.39 * alpha ** 4
    )


def sent_k_reference(sigma_pa: float, crack_length_m: float, width_m: float) -> float:
    """Handbook mode-I SIF of the SENT specimen (Pa sqrt(m))."""
    if crack_length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("crack length and width must be positive")
    return (
        float(sigma_pa)
        * math.sqrt(math.pi * float(crack_length_m))
        * tada_sent_shape_factor(float(crack_length_m) / float(width_m))
    )


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SentMesh:
    """Conforming six-node mesh of the upper half of a SENT specimen."""

    nodes: torch.Tensor  # (nn, 2)
    elements: torch.Tensor  # (ne, 6), node order (V1 V2 V3 M12 M23 M31)
    stations: tuple[float, ...]  # normalised radial fractions, stations[0] == 0
    angles: tuple[float, ...]  # ray angles, angles[0] == 0, angles[-1] == pi
    ray_lengths: tuple[float, ...]  # tip -> outer-boundary distance per ray
    ring: tuple[tuple[int, ...], ...]  # [sector][station] -> node id
    arc_edge: tuple[tuple[tuple[int, int], ...], ...]
    # [sector][station] -> (element, local edge index) of the arc, outer side
    top_edge: tuple[tuple[int, int], ...]  # (element, local edge) on the loaded edge
    node_station: tuple[int, ...]  # radial index of every node (-1 = mid-side)
    width: float
    crack_length: float
    height: float
    quarter_point: bool

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_sectors(self) -> int:
        return len(self.angles) - 1


def _ray_exit(angle: float, crack_length: float, width: float, height: float) -> float:
    """Distance from the tip to the rectangular boundary along ``angle``."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    candidates: list[float] = []
    if sin_a > 1e-14:
        candidates.append(height / sin_a)
    if cos_a > 1e-14:
        candidates.append((width - crack_length) / cos_a)
    if cos_a < -1e-14:
        candidates.append(-crack_length / cos_a)
    if not candidates:
        raise ValueError(f"ray angle {angle} leaves the tip parallel to the free edge")
    return min(candidates)


def _normalised_stations(first_ring: float, growth: float) -> list[float]:
    stations = [0.0, 0.25 * first_ring, first_ring]
    while stations[-1] < 1.0 - 1e-12:
        nxt = stations[-1] * growth
        if nxt >= 1.0:
            nxt = 1.0
        stations.append(nxt)
    return stations


def sent_t6_mesh(
    width: float,
    crack_length: float,
    height: float,
    *,
    n_sectors: int = 40,
    first_ring: float = 0.05,
    growth: float = 1.45,
    quarter_point: bool = True,
    dtype: torch.dtype = torch.float64,
) -> SentMesh:
    """Build the quarter-point ray-fan mesh of the upper half of a SENT plate.

    ``width`` is the specimen width ``W`` (free edge ``x = -a`` to far edge
    ``x = W - a``), ``crack_length`` is ``a`` and ``height`` is the half height
    ``H``.  ``n_sectors`` counts the angular sectors over ``theta in [0, pi]``;
    rays through the two top corners are inserted so the rectangular outer
    boundary is reproduced without corner-cutting chords.
    """
    width = float(width)
    a = float(crack_length)
    height = float(height)
    if not (0.0 < a < width):
        raise ValueError("crack_length must lie inside the specimen width")
    if height <= 0.0:
        raise ValueError("height must be positive")
    if n_sectors < 8:
        raise ValueError("n_sectors must be at least 8")
    if not (0.0 < first_ring < 0.5):
        raise ValueError("first_ring must be a fraction in (0, 0.5)")
    if growth <= 1.0:
        raise ValueError("growth must exceed 1")

    corner_angles = [
        math.atan2(height, width - a),  # top-right corner
        math.atan2(height, -a),  # top-left corner
    ]
    angles = sorted(
        {round(math.pi * j / n_sectors, 12) for j in range(n_sectors + 1)}
        | {round(angle, 12) for angle in corner_angles}
    )
    if angles[0] != 0.0:
        raise AssertionError("the fan must start on the symmetry ray theta = 0")
    angles[-1] = math.pi
    ray_lengths = [_ray_exit(angle, a, width, height) for angle in angles]
    stations = _normalised_stations(first_ring, growth)

    nodes: list[tuple[float, float]] = []
    index: dict[tuple[int, int], int] = {}

    def node_id(x: float, y: float) -> int:
        key = (int(round(x / (1e-9 * width))), int(round(y / (1e-9 * width))))
        if key not in index:
            index[key] = len(nodes)
            nodes.append((float(x), float(y)))
        return index[key]

    n_rays = len(angles)
    ring: list[list[int]] = [[-1] * len(stations) for _ in range(n_rays)]
    for k, s in enumerate(stations):
        for j, angle in enumerate(angles):
            r = s * ray_lengths[j]
            ring[j][k] = node_id(r * math.cos(angle), r * math.sin(angle))

    def at_fraction(p: int, q: int, frac: float) -> int:
        x = (1.0 - frac) * nodes[p][0] + frac * nodes[q][0]
        y = (1.0 - frac) * nodes[p][1] + frac * nodes[q][1]
        return node_id(x, y)

    arc_mid: dict[tuple[int, int], int] = {}
    for k in range(1, len(stations)):
        for j in range(n_rays - 1):
            arc_mid[(j, k)] = at_fraction(ring[j][k], ring[j + 1][k], 0.5)
    ray_mid: dict[tuple[int, int], int] = {}
    for j in range(n_rays):
        for k in range(len(stations) - 1):
            frac = 0.25 if (k == 0 and quarter_point) else 0.5
            ray_mid[(j, k)] = at_fraction(ring[j][k], ring[j][k + 1], frac)
    diag_mid: dict[tuple[int, int], int] = {}
    for j in range(n_rays - 1):
        for k in range(1, len(stations) - 1):
            diag_mid[(j, k)] = at_fraction(ring[j][k], ring[j + 1][k + 1], 0.5)

    raw: list[tuple[int, ...]] = []
    band_element: list[list[int]] = [[-1] * (len(stations) - 1) for _ in range(n_rays - 1)]
    for j in range(n_rays - 1):
        tip = ring[j][0]
        if tip != ring[j + 1][0]:
            raise AssertionError("the tip node must be shared by every sector")
        raw.append(
            (
                tip,
                ring[j][1],
                ring[j + 1][1],
                ray_mid[(j, 0)],
                arc_mid[(j, 1)],
                ray_mid[(j + 1, 0)],
            )
        )
        band_element[j][0] = len(raw) - 1
        for k in range(1, len(stations) - 1):
            a_id, b_id = ring[j][k], ring[j + 1][k]
            c_id, d_id = ring[j + 1][k + 1], ring[j][k + 1]
            raw.append(
                (
                    a_id,
                    b_id,
                    c_id,
                    arc_mid[(j, k)],
                    ray_mid[(j + 1, k)],
                    diag_mid[(j, k)],
                )
            )
            band_element[j][k] = len(raw) - 1
            raw.append(
                (
                    a_id,
                    c_id,
                    d_id,
                    diag_mid[(j, k)],
                    arc_mid[(j, k + 1)],
                    ray_mid[(j, k)],
                )
            )

    node_tensor = torch.tensor(nodes, dtype=dtype)
    elements = torch.tensor(raw, dtype=torch.long)
    # enforce a positive Jacobian: swap V2/V3 and their mid-side nodes when the
    # corner ordering comes out clockwise
    corner = node_tensor[elements[:, :3]]
    e1 = corner[:, 1] - corner[:, 0]
    e2 = corner[:, 2] - corner[:, 0]
    signed_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flipped = signed_area < 0.0
    if bool(flipped.any()):
        elements[flipped] = elements[flipped][:, [0, 2, 1, 5, 4, 3]]

    # node -> radial index (mid-side nodes get -1)
    node_station = [-1] * node_tensor.shape[0]
    for j in range(n_rays):
        for k in range(len(stations)):
            node_station[ring[j][k]] = k

    # corner-edge lookup: unordered node pair -> [(element, local edge index)]
    edge_lookup: dict[tuple[int, int], list[tuple[int, int]]] = {}
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    for element, ids in enumerate(elements.tolist()):
        for edge_index, (p, q) in enumerate(edge_pairs):
            key = (min(ids[p], ids[q]), max(ids[p], ids[q]))
            edge_lookup.setdefault(key, []).append((element, edge_index))

    def outer_edge(j: int, k: int) -> tuple[int, int]:
        """Element/edge of the arc at ``stations[k]`` on the outer side."""
        p, q = ring[j][k], ring[j + 1][k]
        candidates = edge_lookup[(min(p, q), max(p, q))]

        def mean_station(candidate: tuple[int, int]) -> float:
            ids = elements[candidate[0]][:3].tolist()
            known = [node_station[i] for i in ids if node_station[i] >= 0]
            return sum(known) / len(known)

        return max(candidates, key=mean_station)

    arc_edge = tuple(
        tuple(outer_edge(j, k) if k >= 1 else (-1, -1) for k in range(len(stations)))
        for j in range(n_rays - 1)
    )
    top_tol = 1e-9 * max(height, width)
    top_edge = tuple(
        sorted(
            {
                (element, edge_index)
                for element, ids in enumerate(elements.tolist())
                for edge_index, (p, q) in enumerate(edge_pairs)
                if abs(nodes[ids[p]][1] - height) < top_tol
                and abs(nodes[ids[q]][1] - height) < top_tol
            }
        )
    )
    if not top_edge:
        raise AssertionError("no element edge was found on the loaded edge y = H")

    return SentMesh(
        nodes=node_tensor,
        elements=elements,
        stations=tuple(stations),
        angles=tuple(angles),
        ray_lengths=tuple(ray_lengths),
        ring=tuple(tuple(row) for row in ring),
        arc_edge=arc_edge,
        top_edge=top_edge,
        node_station=tuple(node_station),
        width=width,
        crack_length=a,
        height=height,
        quarter_point=bool(quarter_point),
    )


# --------------------------------------------------------------------------- #
# element kinematics
# --------------------------------------------------------------------------- #
def edge_geometry(edge_index: int, t: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Shape functions and ``dN/dt`` on local corner edge ``edge_index``.

    ``edge_index`` 0 is ``V1 -> V2``, 1 is ``V2 -> V3`` and 2 is ``V3 -> V1``;
    ``t`` runs from 0 to 1 along the edge.
    """
    if edge_index == 0:
        xi, eta = t, 0.0
        d_xi, d_eta = t6_shape_derivative(xi, eta)
        shape = t6_shape(xi, eta)
        return shape, d_xi
    if edge_index == 1:
        xi, eta = 1.0 - t, t
        d_xi, d_eta = t6_shape_derivative(xi, eta)
        shape = t6_shape(xi, eta)
        return shape, tuple(-a + b for a, b in zip(d_xi, d_eta))
    if edge_index == 2:
        xi, eta = 0.0, 1.0 - t
        d_xi, d_eta = t6_shape_derivative(xi, eta)
        shape = t6_shape(xi, eta)
        return shape, tuple(-b for b in d_eta)
    raise ValueError("edge_index must be 0, 1 or 2")


def element_field_at(
    mesh: SentMesh,
    displacement: torch.Tensor,
    element: int,
    xi: float,
    eta: float,
    dmat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Stress ``(2, 2)``, displacement gradient ``(2, 2)`` and ``det J`` at a parent point.

    The Jacobian convention is the repository-wide one, ``jac[a, b] = dX_b/dxi_a``,
    matching :func:`tensorfem.fatigue.crack_tip_quarter_point.t6_stiffness`.
    """
    ids = mesh.elements[element]
    coords = mesh.nodes[ids]
    d_xi, d_eta = t6_shape_derivative(xi, eta)
    d_xi_t = torch.tensor(d_xi, dtype=coords.dtype)
    d_eta_t = torch.tensor(d_eta, dtype=coords.dtype)
    dx_dxi = d_xi_t @ coords
    dx_deta = d_eta_t @ coords
    jac = torch.stack((dx_dxi, dx_deta), dim=0)  # row a = dX_b/dxi_a
    det = float(jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0])
    if det <= 0.0:
        raise ValueError("inverted six-node triangle on the contour")
    inv = torch.stack(
        (
            torch.stack((jac[1, 1], -jac[0, 1])),
            torch.stack((-jac[1, 0], jac[0, 0])),
        )
    ) / det
    grad_x = inv[0, 0] * d_xi_t + inv[0, 1] * d_eta_t
    grad_y = inv[1, 0] * d_xi_t + inv[1, 1] * d_eta_t
    local = torch.zeros(12, dtype=coords.dtype)
    local[0::2] = displacement[2 * ids]
    local[1::2] = displacement[2 * ids + 1]
    b = torch.zeros((3, 12), dtype=coords.dtype)
    b[0, 0::2] = grad_x
    b[1, 1::2] = grad_y
    b[2, 0::2] = grad_y
    b[2, 1::2] = grad_x
    stress_vec = dmat @ (b @ local)
    stress = torch.stack(
        (
            torch.stack((stress_vec[0], stress_vec[2])),
            torch.stack((stress_vec[2], stress_vec[1])),
        )
    )
    grad = torch.stack(
        (
            torch.stack((grad_x @ local[0::2], grad_y @ local[0::2])),
            torch.stack((grad_x @ local[1::2], grad_y @ local[1::2])),
        )
    )
    return stress, grad, det


# --------------------------------------------------------------------------- #
# boundary conditions and solve
# --------------------------------------------------------------------------- #
def _loaded_edge_loads(mesh: SentMesh, sigma_pa: float) -> torch.Tensor:
    """Consistent nodal loads of the uniform tension on ``y = H``."""
    dtype = mesh.nodes.dtype
    load = torch.zeros(2 * mesh.n_nodes, dtype=dtype)
    applied = 0.0
    for element, edge_index in mesh.top_edge:
        ids = mesh.elements[element]
        coords = mesh.nodes[ids]
        for t, weight in _EDGE_GAUSS:
            shape, d_dt = edge_geometry(edge_index, t)
            tangent = torch.tensor(d_dt, dtype=dtype) @ coords
            ds = float(torch.linalg.vector_norm(tangent)) * weight
            for local, value in enumerate(shape):
                load[2 * ids[local] + 1] += value * sigma_pa * ds
            applied += float(sum(shape)) * sigma_pa * ds
    expected = sigma_pa * mesh.width
    if abs(applied - expected) > 1e-8 * expected + 1e-12:
        raise AssertionError(
            f"applied edge load {applied!r} does not integrate to sigma*W = {expected!r}"
        )
    return load


def symmetry_dofs(mesh: SentMesh) -> dict[int, float]:
    """Zero ``u_y`` on the symmetry plane ahead of the crack plus one ``u_x`` tie."""
    tol = 1e-9 * max(mesh.height, mesh.width)
    on_axis = (mesh.nodes[:, 1].abs() < tol) & (mesh.nodes[:, 0] >= -tol)
    dirichlet: dict[int, float] = {}
    for node in torch.nonzero(on_axis).flatten().tolist():
        dirichlet[2 * node + 1] = 0.0
    tip = int(torch.argmin(torch.linalg.vector_norm(mesh.nodes, dim=1)))
    dirichlet[2 * tip] = 0.0
    return dirichlet


@dataclass(frozen=True)
class SentSolve:
    """Solved state of the SENT half model."""

    displacement: torch.Tensor
    applied_load_n: float
    reaction_n: float

    @property
    def reaction_ratio(self) -> float:
        return self.reaction_n / self.applied_load_n


def solve_sent(
    mesh: SentMesh,
    *,
    sigma_pa: float,
    young_pa: float,
    poisson: float = 0.3,
    plane: str = "stress",
    thickness: float = 1.0,
    rtol: float = 1e-12,
) -> SentSolve:
    """Solve the SENT half model and return displacement plus the global reaction."""
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
    K = assemble_coo(stiffness, element_dofs, 2 * mesh.n_nodes).to_sparse_coo()
    load = _loaded_edge_loads(mesh, sigma_pa)
    dirichlet = symmetry_dofs(mesh)
    solved = solve_sparse_static(K, load, dirichlet=dirichlet, rtol=rtol)
    internal = torch.sparse.mm(K.coalesce(), solved.displacement.reshape(-1, 1)).flatten()
    reaction = sum(
        float(internal[dof]) for dof in dirichlet if dof % 2 == 1
    )
    return SentSolve(
        displacement=solved.displacement,
        applied_load_n=float(load.sum()),
        reaction_n=abs(reaction),
    )


# --------------------------------------------------------------------------- #
# J integral on a mesh line of the fan
# --------------------------------------------------------------------------- #
def j_integral_subring(
    mesh: SentMesh,
    displacement: torch.Tensor,
    station_index: int,
    *,
    young_pa: float,
    poisson: float = 0.3,
    plane: str = "stress",
) -> dict:
    """Rice J-integral on the fan mesh line at ``stations[station_index]``.

    Returns the full-plate ``J`` (twice the integrated upper arc) plus the two
    flux components and the contour perimeter, so the balance can be audited.
    """
    if not 1 <= station_index < len(mesh.stations) - 1:
        raise IndexError("station_index must address a station strictly inside the fan")
    dtype = mesh.nodes.dtype
    dmat = elasticity_matrix(
        torch.tensor([young_pa], dtype=dtype),
        torch.tensor([poisson], dtype=dtype),
        plane,
    )[0]
    energy_total = 0.0
    work_total = 0.0
    perimeter = 0.0
    for j in range(len(mesh.angles) - 1):
        element, edge_index = mesh.arc_edge[j][station_index]
        if element < 0:
            raise ValueError("no element band at this station")
        ids = mesh.elements[element]
        coords = mesh.nodes[ids]
        for t, weight in _EDGE_GAUSS:
            shape, d_dt = edge_geometry(edge_index, t)
            xi, eta = edge_parent(edge_index, t)
            stress, grad, det = element_field_at(mesh, displacement, element, xi, eta, dmat)
            if det <= 0.0:
                raise ValueError("inverted element on the J contour")
            tangent = torch.tensor(d_dt, dtype=dtype) @ coords
            point = torch.tensor(shape, dtype=dtype) @ coords
            normal = torch.stack((tangent[1], -tangent[0]))
            if float(torch.dot(normal, point)) < 0.0:
                normal = -normal
            unit_normal = normal / float(torch.linalg.vector_norm(normal))
            ds = float(torch.linalg.vector_norm(tangent)) * weight
            strain = 0.5 * (grad + grad.transpose(0, 1))
            energy = 0.5 * float(torch.sum(stress * strain))
            traction = stress @ unit_normal
            work = float(torch.dot(traction, grad[:, 0]))
            energy_total += energy * float(unit_normal[0]) * ds
            work_total += work * ds
            perimeter += ds
    j_upper = energy_total - work_total
    return {
        "station_index": int(station_index),
        "station_fraction": float(mesh.stations[station_index]),
        "j_upper_j_m2": j_upper,
        "j_j_m2": 2.0 * j_upper,
        "energy_flux_j_m2": 2.0 * energy_total,
        "work_flux_j_m2": 2.0 * work_total,
        "contour_perimeter_m": perimeter,
    }


def edge_parent(edge_index: int, t: float) -> tuple[float, float]:
    """Parent coordinates of the point at parameter ``t`` of a local edge."""
    if edge_index == 0:
        return t, 0.0
    if edge_index == 1:
        return 1.0 - t, t
    if edge_index == 2:
        return 0.0, 1.0 - t
    raise ValueError("edge_index must be 0, 1 or 2")


def sent_k_from_j(j_j_m2: float, young_pa: float, poisson: float = 0.3,
                  *, plane: str = "stress") -> float:
    """``K_I = sqrt(J E')`` in Pa sqrt(m) (thin wrapper over the shipped helper)."""
    return mode_i_k_from_j(j_j_m2, young_pa, poisson, plane=plane)


# --------------------------------------------------------------------------- #
# verification drivers
# --------------------------------------------------------------------------- #
DEFAULT_SENT_LEVELS: tuple[dict, ...] = (
    {"name": "coarse", "n_sectors": 24, "first_ring": 0.08, "growth": 1.6},
    {"name": "medium", "n_sectors": 48, "first_ring": 0.05, "growth": 1.45},
    {"name": "fine", "n_sectors": 96, "first_ring": 0.03, "growth": 1.32},
)


def _contour_record(
    mesh: SentMesh,
    displacement: torch.Tensor,
    fraction: float,
    *,
    sigma_pa: float,
    young_pa: float,
    poisson: float,
    plane: str,
    k_reference: float,
    crack_length: float,
) -> dict:
    station = min(
        range(1, len(mesh.stations) - 1),
        key=lambda idx: abs(mesh.stations[idx] - fraction),
    )
    j = j_integral_subring(
        mesh, displacement, station, young_pa=young_pa, poisson=poisson, plane=plane
    )
    k = sent_k_from_j(j["j_j_m2"], young_pa, poisson, plane=plane)
    return {
        "requested_fraction": float(fraction),
        "station_fraction": j["station_fraction"],
        "j_j_m2": j["j_j_m2"],
        "k_pa_sqrt_m": k,
        "shape_factor": k / (sigma_pa * math.sqrt(math.pi * crack_length)),
        "rel_error_vs_tada": (k - k_reference) / k_reference,
    }


def sent_convergence(
    *,
    width_m: float = 0.10,
    crack_ratios: tuple[float, ...] = (0.2, 0.35, 0.5),
    height_ratio: float = 2.0,
    sigma_pa: float = 100e6,
    young_pa: float = 210e9,
    poisson: float = 0.3,
    plane: str = "stress",
    contour_fractions: tuple[float, ...] = (0.10, 0.15, 0.20),
    levels: tuple[dict, ...] = DEFAULT_SENT_LEVELS,
    quarter_point: bool = True,
) -> dict:
    """Run the refinement ladder for every ``a/W`` and return the evidence dictionary."""
    levels_out = []
    for level in levels:
        per_ratio = []
        for alpha in crack_ratios:
            a = alpha * width_m
            mesh = sent_t6_mesh(
                width_m,
                a,
                height_ratio * width_m,
                n_sectors=level["n_sectors"],
                first_ring=level["first_ring"],
                growth=level["growth"],
                quarter_point=quarter_point,
            )
            solved = solve_sent(
                mesh, sigma_pa=sigma_pa, young_pa=young_pa, poisson=poisson, plane=plane
            )
            k_ref = sent_k_reference(sigma_pa, a, width_m)
            contours = [
                _contour_record(
                    mesh,
                    solved.displacement,
                    fraction,
                    sigma_pa=sigma_pa,
                    young_pa=young_pa,
                    poisson=poisson,
                    plane=plane,
                    k_reference=k_ref,
                    crack_length=a,
                )
                for fraction in contour_fractions
            ]
            per_ratio.append(
                {
                    "a_over_w": float(alpha),
                    "crack_length_m": a,
                    "shape_factor_tada": tada_sent_shape_factor(alpha),
                    "k_reference_pa_sqrt_m": k_ref,
                    "n_nodes": mesh.n_nodes,
                    "n_elements": mesh.n_elements,
                    "n_stations": len(mesh.stations),
                    "contours": contours,
                    "reaction_ratio": solved.reaction_ratio,
                }
            )
        levels_out.append({"level": level["name"], **level, "cases": per_ratio})
    return {
        "width_m": width_m,
        "height_over_width": height_ratio,
        "sigma_pa": sigma_pa,
        "young_pa": young_pa,
        "poisson": poisson,
        "plane": plane,
        "quarter_point": quarter_point,
        "contour_fractions": list(contour_fractions),
        "levels": levels_out,
    }


def sent_sweep(
    *,
    width_m: float = 0.10,
    crack_ratios: tuple[float, ...] = (0.2, 0.35, 0.5),
    height_ratios: tuple[float, ...] = (1.5, 2.0, 3.0),
    sigma_pa: float = 100e6,
    young_pa: float = 210e9,
    poisson: float = 0.3,
    plane: str = "stress",
    contour_fraction: float = 0.15,
    level: dict | None = None,
) -> dict:
    """Height sensitivity of the recovered factor at one fixed mesh level."""
    level = dict(DEFAULT_SENT_LEVELS[1] if level is None else level)
    rows = []
    for height_ratio in height_ratios:
        per_ratio = []
        for alpha in crack_ratios:
            a = alpha * width_m
            mesh = sent_t6_mesh(
                width_m,
                a,
                height_ratio * width_m,
                n_sectors=level["n_sectors"],
                first_ring=level["first_ring"],
                growth=level["growth"],
            )
            solved = solve_sent(
                mesh, sigma_pa=sigma_pa, young_pa=young_pa, poisson=poisson, plane=plane
            )
            record = _contour_record(
                mesh,
                solved.displacement,
                contour_fraction,
                sigma_pa=sigma_pa,
                young_pa=young_pa,
                poisson=poisson,
                plane=plane,
                k_reference=sent_k_reference(sigma_pa, a, width_m),
                crack_length=a,
            )
            per_ratio.append(
                {
                    "a_over_w": float(alpha),
                    "shape_factor": record["shape_factor"],
                    "rel_error_vs_tada": record["rel_error_vs_tada"],
                }
            )
        rows.append({"height_over_width": float(height_ratio), "cases": per_ratio})
    return {
        "level": level,
        "contour_fraction": contour_fraction,
        "rows": rows,
    }


def sent_gci(
    records: list[dict],
    *,
    alpha: float,
    refinement: float = 2.0,
    contour_fraction: float = 0.15,
) -> dict:
    """Roache GCI of the three finest levels for one ``a/W``."""
    values = []
    for record in records:
        for case in record["cases"]:
            if abs(case["a_over_w"] - alpha) < 1e-12:
                best = min(
                    case["contours"],
                    key=lambda c: abs(c["requested_fraction"] - contour_fraction),
                )
                values.append(best["shape_factor"])
    if len(values) < 3:
        raise ValueError("need at least three levels")
    return grid_convergence_index(
        values[-1], values[-2], values[-3], refinement=refinement
    )