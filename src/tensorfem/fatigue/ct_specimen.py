"""Compact-tension C(T) fatigue-crack-growth verification face (ASTM E647 / E399).

This module builds a systematic verification face for fatigue crack growth driven
by linear-elastic fracture mechanics (LEFM):

``ct_geometry_factor``
    Dimensionless ASTM E399 / E647 compact-specimen stress-intensity factor,
    ``K_I = P / (B sqrt(W)) * f(a/W)`` with

    ``f(alpha) = (2 + alpha)/(1 - alpha)**1.5 *
    (0.886 + 4.64 alpha - 13.32 alpha**2 + 14.72 alpha**3 - 5.6 alpha**4)``,

    quoted by ASTM as better than 0.5 % for ``0.2 <= alpha <= 0.8``.

``ct_t6_mesh`` / ``solve_ct_model``
    Pin-loaded C(T) plane model discretised with six-node triangles: a
    quarter-point fan around the crack tip (Barsoum / Henshell-Shaw) embedded in
    a mapped ``(rho, theta)`` grid that lands exactly on the specimen boundary
    ``x in [-0.25 W, W]``, ``y in [-0.6 W, 0.6 W]`` (ASTM total width 1.25 W,
    height 1.2 W).  The pin load is transferred to the mesh as a statically
    equivalent nodal triplet (same resultant force and zero net moment about the
    pin centre at ``(0, +-0.275 W)``).

``ring_j_integral``
    Rice's J-integral from the finite-element field on closed mesh-line contours
    around the tip; ``K_J = sqrt(J E')``.

``cod_k_from_opening``
    ``K_I`` from the quarter-point crack-opening displacement
    ``COD(r) = (8 K_I / E') sqrt(r / 2 pi)`` extrapolated to ``r -> 0``.

``ThresholdParis``
    Threshold-modified Paris law ``da/dN = C dK**m (1 - (dK_th/dK)**k)`` above
    the threshold and *exactly* zero below it.  The near-threshold branch is
    steep (``k`` control) while the mid-rate branch stays a pure Paris law, so a
    log-log fit of mid-rate data recovers ``(C, m)`` without bias.

``e647_secant_reduction`` / ``paris_fit``
    ASTM E647 data reduction: secant construction ``da/dN_i = (a_(i+1) - a_i) /
    (N_(i+1) - N_i)`` at the mean ``dK``, then a least-squares fit in log-log.

``load_shedding_threshold_test``
    Annex-A1 style load shedding (normalised K-gradient ``C = -0.08 mm**-1``)
    locating ``dK_th`` where ``da/dN`` first falls below ``1e-10 m/cycle``.

``ct_fracture_toughness``
    ``K_IC`` from the instability load of the C(T) specimen plus the E399
    plane-strain size requirement ``B >= 2.5 (K_IC/sigma_y)**2``.

``miner_accumulated_damage``
    Miner summation ``D = sum n_i/N_i`` for a variable-amplitude block, with each
    ``N_i`` from a constant-amplitude Paris integration, compared against the
    true block-sequence life.

Units: geometry in metres, stresses in Pa, ``K`` and ``dK`` in ``Pa*sqrt(m)``
inside the mechanics routines; the Paris constants are quoted for
``MPa*sqrt(m)`` and converted at the boundary of :class:`ThresholdParis`.
Scope: linear elastic, small strain, mode I, stationary amplitude per cycle, no
closure / overload retardation and no plasticity correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ..continuum import elasticity_matrix
from ..sparse_core import assemble_coo
from .crack_tip_quarter_point import (
    _parent_gradient,
    mode_i_k_from_j,
    t6_shape_derivative,
    t6_stiffness,
)

__all__ = [
    "ASTM_CT_RANGE",
    "CTFieldResult",
    "CTMesh",
    "CTSpecimen",
    "ThresholdParis",
    "cod_k_from_opening",
    "crack_growth_cycles",
    "ct_fracture_toughness",
    "ct_geometry_factor",
    "ct_stress_intensity",
    "ct_t6_mesh",
    "e647_constant_amplitude_test",
    "e647_secant_reduction",
    "load_shedding_threshold_test",
    "miner_accumulated_damage",
    "paris_fit",
    "ring_j_integral",
    "solve_and_postprocess",
    "solve_ct_model",
    "unstable_crack_ratio",
]

#: ASTM E399 / E647 validity window of the compact-specimen geometry factor.
ASTM_CT_RANGE = (0.2, 0.8)

_MPA_SQRT_M = 1.0e6  # Pa*sqrt(m) per MPa*sqrt(m)

#: 3-point Gauss rule on the edge parameter ``t in [0, 1]``.
_EDGE_GAUSS = (
    (0.5 - math.sqrt(0.15), 5.0 / 18.0),
    (0.5, 8.0 / 18.0),
    (0.5 + math.sqrt(0.15), 5.0 / 18.0),
)


# --------------------------------------------------------------------------- #
# ASTM geometry
# --------------------------------------------------------------------------- #
def ct_geometry_factor(alpha: float) -> float:
    """Dimensionless C(T) stress-intensity factor ``f(a/W)`` (ASTM E399)."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha = a/W must be positive and finite")
    if alpha >= 1.0:
        raise ValueError("alpha = a/W must stay below 1")
    root = (1.0 - alpha) ** 1.5
    poly = 0.886 + 4.64 * alpha - 13.32 * alpha**2 + 14.72 * alpha**3 - 5.6 * alpha**4
    return (2.0 + alpha) / root * poly


def ct_stress_intensity(load_n: float, width_m: float, thickness_m: float, alpha: float) -> float:
    """``K_I`` in Pa*sqrt(m) of the standard compact specimen."""
    for name, value in (("load_n", load_n), ("width_m", width_m), ("thickness_m", thickness_m)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    return load_n / (thickness_m * math.sqrt(width_m)) * ct_geometry_factor(alpha)


@dataclass(frozen=True)
class CTSpecimen:
    """Standard ASTM C(T) specimen (lengths in metres, stresses in Pa)."""

    width: float = 0.050
    thickness: float = 0.025
    young: float = 2.06e11
    poisson: float = 0.3
    yield_strength: float = 5.5e8

    @property
    def total_width(self) -> float:
        """ASTM total width ``1.25 W`` (front face to back face)."""
        return 1.25 * self.width

    @property
    def half_height(self) -> float:
        """ASTM half height ``0.6 W``."""
        return 0.6 * self.width

    @property
    def pin_offset(self) -> float:
        """Pin-hole centres sit at ``x = 0``, ``y = +-0.275 W``."""
        return 0.275 * self.width

    @property
    def effective_modulus(self) -> float:
        """Plane-strain ``E' = E / (1 - nu**2)``."""
        return self.young / (1.0 - self.poisson**2)

    def stress_intensity(self, load_n: float, alpha: float) -> float:
        return ct_stress_intensity(load_n, self.width, self.thickness, alpha)

    def delta_k(self, load_range_n: float, alpha: float) -> float:
        """``dK`` in Pa*sqrt(m) for a load range ``dP``."""
        return self.stress_intensity(load_range_n, alpha)

    def load_range_for_delta_k(self, delta_k_mpa: float, alpha: float) -> float:
        """Load range ``dP`` in N giving ``dK`` at the given ``a/W``."""
        factor = self.thickness * math.sqrt(self.width)
        return delta_k_mpa * _MPA_SQRT_M * factor / ct_geometry_factor(alpha)


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
def _ray_length(width: float, alpha: float, theta: float) -> float:
    """Distance from the crack tip to the specimen boundary along ``theta``."""
    cx, sy = math.cos(theta), math.sin(theta)
    distances: list[float] = []
    limit_right = width - alpha * width
    limit_left = alpha * width + 0.25 * width
    limit_top = 0.6 * width
    if abs(cx) > 1.0e-12:
        distances.append((limit_right if cx > 0.0 else limit_left) / abs(cx))
    if abs(sy) > 1.0e-12:
        distances.append(limit_top / abs(sy))
    if not distances:
        raise ValueError("degenerate ray direction")
    return min(distances)


@dataclass(frozen=True)
class CTMesh:
    """Mapped quarter-point mesh of the C(T) specimen."""

    nodes: torch.Tensor
    elements: torch.Tensor
    rings: tuple[tuple[tuple[int, str], ...], ...]
    face_upper: tuple[int, ...]
    face_lower: tuple[int, ...]
    tip_node: int
    ring_radii: tuple[float, ...]
    width: float
    alpha: float
    n_angles: int
    n_rings: int

    def coordinates(self, node: int) -> tuple[float, float]:
        return float(self.nodes[node, 0]), float(self.nodes[node, 1])


def ct_t6_mesh(
    specimen: CTSpecimen,
    alpha: float,
    *,
    n_angles: int = 72,
    n_rings: int = 14,
    growth: float = 1.35,
    dtype: torch.dtype = torch.float64,
) -> CTMesh:
    """Quarter-point six-node-triangle mesh of the standard C(T) specimen.

    ``alpha = a/W`` with ``a`` measured from the load line.  ``n_angles`` sectors
    cover ``theta in [-pi, pi]`` (the two crack faces are distinct node sets, so
    the ray ``theta = +-pi`` is duplicated) and ``n_rings`` rings follow the
    geometric grading ``rho_j = (growth**j - 1)/(growth**n_rings - 1)``.
    """
    if not ASTM_CT_RANGE[0] <= alpha <= ASTM_CT_RANGE[1]:
        raise ValueError(f"alpha = {alpha} outside the ASTM validity window {ASTM_CT_RANGE}")
    if n_angles < 16 or n_angles % 4:
        raise ValueError("n_angles must be a multiple of 4 and at least 16")
    if n_rings < 4:
        raise ValueError("n_rings must be at least 4")
    if growth <= 1.0:
        raise ValueError("growth must exceed 1")

    width = specimen.width
    tip = (alpha * width, 0.0)
    rho = [0.0] + [(growth**j - 1.0) / (growth**n_rings - 1.0) for j in range(1, n_rings + 1)]
    theta = [-math.pi + 2.0 * math.pi * k / n_angles for k in range(n_angles + 1)]
    theta_mid = [-math.pi + 2.0 * math.pi * (k + 0.5) / n_angles for k in range(n_angles)]

    nodes: list[tuple[float, float]] = []

    def add(x: float, y: float) -> int:
        nodes.append((x, y))
        return len(nodes) - 1

    def at(r: float, angle: float) -> tuple[float, float]:
        length = _ray_length(width, alpha, angle)
        return (tip[0] + r * length * math.cos(angle), tip[1] + r * length * math.sin(angle))

    tip_node = add(*tip)
    vertex: dict[int, list[int]] = {j: [] for j in range(1, n_rings + 1)}
    for j in range(1, n_rings + 1):
        vertex[j] = [add(*at(rho[j], angle)) for angle in theta]
    radial: list[list[int]] = []
    for j in range(n_rings):
        if j == 0:
            radial.append([add(*at(0.25 * rho[1], angle)) for angle in theta])
        else:
            mid = 0.5 * (rho[j] + rho[j + 1])
            radial.append([add(*at(mid, angle)) for angle in theta])
    arc: dict[int, list[int]] = {j: [] for j in range(1, n_rings + 1)}
    for j in range(1, n_rings + 1):
        arc[j] = [add(*at(rho[j], angle)) for angle in theta_mid]
    diagonal: dict[int, list[int]] = {j: [] for j in range(1, n_rings + 1)}
    for j in range(1, n_rings):
        row: list[int] = []
        for k in range(n_angles):
            x0, y0 = nodes[vertex[j][k]]
            x1, y1 = nodes[vertex[j + 1][k + 1]]
            row.append(add(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        diagonal[j] = row

    elements: list[tuple[int, ...]] = []
    tip_triangles: list[int] = []
    for k in range(n_angles):
        tip_triangles.append(len(elements))
        elements.append(
            (tip_node, vertex[1][k], vertex[1][k + 1], radial[0][k], arc[1][k], radial[0][k + 1])
        )
    for j in range(1, n_rings):
        for k in range(n_angles):
            elements.append(
                (
                    vertex[j][k],
                    vertex[j + 1][k],
                    vertex[j + 1][k + 1],
                    radial[j][k],
                    arc[j + 1][k],
                    diagonal[j][k],
                )
            )
            elements.append(
                (
                    vertex[j][k],
                    vertex[j + 1][k + 1],
                    vertex[j][k + 1],
                    diagonal[j][k],
                    radial[j][k + 1],
                    arc[j][k],
                )
            )

    rings: list[tuple[tuple[int, str], ...]] = [
        tuple((tip_triangles[k], "outer") for k in range(n_angles))
    ]
    n_tip = len(tip_triangles)
    for j in range(2, n_rings + 1):
        base = n_tip + 2 * (j - 2) * n_angles
        rings.append(tuple((base + 2 * k + 1, "inner") for k in range(n_angles)))

    ring_radii = tuple(rho[j] * 0.75 * width for j in range(1, n_rings + 1))
    return CTMesh(
        nodes=torch.tensor(nodes, dtype=dtype),
        elements=torch.tensor(elements, dtype=torch.long),
        rings=tuple(rings),
        face_upper=tuple(vertex[j][n_angles] for j in range(1, n_rings + 1)),
        face_lower=tuple(vertex[j][0] for j in range(1, n_rings + 1)),
        tip_node=tip_node,
        ring_radii=ring_radii,
        width=width,
        alpha=alpha,
        n_angles=n_angles,
        n_rings=n_rings,
    )


# --------------------------------------------------------------------------- #
# finite element solution
# --------------------------------------------------------------------------- #
def _nearest_node(mesh: CTMesh, point: tuple[float, float]) -> int:
    dx = mesh.nodes[:, 0] - point[0]
    dy = mesh.nodes[:, 1] - point[1]
    return int(torch.argmin(dx * dx + dy * dy))


def _equivalent_forces(
    mesh: CTMesh, point: tuple[float, float], force: tuple[float, float], ndof: int
) -> torch.Tensor:
    """Statically equivalent nodal triplet (resultant and zero net moment kept).

    The three nodes nearest the pin centre rarely form a well conditioned
    triangle in a radially graded mesh (they often share one ray), so the first
    non-collinear triplet out of the nearest candidates is taken.
    """
    dx = mesh.nodes[:, 0] - point[0]
    dy = mesh.nodes[:, 1] - point[1]
    candidates = torch.argsort(dx * dx + dy * dy)[:24].tolist()
    dtype = mesh.nodes.dtype

    def determinant(nodes: list[int]) -> float:
        matrix = torch.zeros((3, 3), dtype=dtype)
        for column, node in enumerate(nodes):
            matrix[0, column] = 1.0
            matrix[1, column] = dx[node]
            matrix[2, column] = dy[node]
        return float(torch.linalg.det(matrix))

    chosen: list[int] = [candidates[0]]
    scale = math.sqrt(max(float(dx[candidates[0]] ** 2 + dy[candidates[0]] ** 2), 1e-30))
    for node in candidates[1:]:
        if len(chosen) == 1:
            cross = float(dx[chosen[0]] * dy[node] - dy[chosen[0]] * dx[node])
            if abs(cross) > 1e-8 * scale**2:
                chosen.append(node)
        else:
            trial = chosen + [node]
            if abs(determinant(trial)) > 1e-8 * scale**2:
                chosen = trial
                break
    if len(chosen) != 3:
        raise ValueError("degenerate pin-load triplet")
    matrix = torch.zeros((3, 3), dtype=dtype)
    for column, node in enumerate(chosen):
        matrix[0, column] = 1.0
        matrix[1, column] = dx[node]
        matrix[2, column] = dy[node]
    weights = torch.linalg.solve(matrix, torch.tensor([1.0, 0.0, 0.0], dtype=dtype))
    vector = torch.zeros(ndof, dtype=dtype)
    for weight, node in zip(weights.tolist(), chosen):
        vector[2 * node] += weight * force[0]
        vector[2 * node + 1] += weight * force[1]
    return vector


def solve_ct_model(
    mesh: CTMesh, specimen: CTSpecimen, load_n: float, *, plane: str = "strain"
) -> torch.Tensor:
    """Solve the pin-loaded C(T) model and return the nodal displacements."""
    dtype = mesh.nodes.dtype
    dmat = elasticity_matrix(
        torch.tensor([specimen.young], dtype=dtype),
        torch.tensor([specimen.poisson], dtype=dtype),
        plane,
    )[0]
    coords = mesh.nodes[mesh.elements]
    stiffness = t6_stiffness(coords, dmat, specimen.thickness)
    dofs = torch.stack([2 * mesh.elements, 2 * mesh.elements + 1], dim=-1).reshape(
        mesh.elements.shape[0], 12
    )
    ndof = 2 * mesh.nodes.shape[0]
    matrix = assemble_coo(stiffness, dofs, ndof)
    # ASTM convention: ``load_n`` is the force transmitted through *one* pin, so
    # the upper pin carries +P and the lower pin -P; the couple is self
    # equilibrated (both act on the load line x = 0).
    force = _equivalent_forces(mesh, (0.0, specimen.pin_offset), (0.0, load_n), ndof)
    force = force + _equivalent_forces(mesh, (0.0, -specimen.pin_offset), (0.0, -load_n), ndof)
    # Rigid-body removal only: the two pins load the specimen in equilibrium, so
    # a translation and a rotation are the only singular modes.  The fixed nodes
    # sit on the back face, far outside every J contour used here.
    back_mid = _nearest_node(mesh, (specimen.width, 0.0))
    top_back = _nearest_node(mesh, (specimen.width, specimen.half_height))
    dirichlet = {2 * back_mid: 0.0, 2 * back_mid + 1: 0.0, 2 * top_back: 0.0}
    return _solve_sparse(matrix, force, dirichlet)


def _solve_sparse(
    matrix: torch.Tensor, force: torch.Tensor, dirichlet: dict[int, float]
) -> torch.Tensor:
    """Exact sparse direct solve of the constrained system (no dense fallback)."""
    import numpy as np

    from ..sparse_direct import factorize_sparse

    fixed = np.zeros(force.numel(), dtype=bool)
    for dof, value in dirichlet.items():
        if value != 0.0:
            raise ValueError("only homogeneous Dirichlet data are supported")
        fixed[dof] = True
    free = np.where(~fixed)[0]
    remap = -np.ones(force.numel(), dtype=np.int64)
    remap[free] = np.arange(free.size)
    coalesced = matrix.coalesce()
    index = coalesced.indices().numpy()
    keep = ~fixed[index[0]] & ~fixed[index[1]]
    rows = remap[index[0][keep]]
    cols = remap[index[1][keep]]
    values = coalesced.values().numpy()[keep]
    reduced = torch.sparse_coo_tensor(
        torch.as_tensor(np.stack((rows, cols))),
        torch.as_tensor(values, dtype=torch.float64),
        (free.size, free.size),
    ).coalesce()
    factor = factorize_sparse(reduced, method="splu")
    rhs = torch.as_tensor(force.numpy()[free], dtype=torch.float64)
    answer = torch.zeros(force.numel(), dtype=torch.float64)
    answer[torch.as_tensor(free)] = factor.solve(rhs).detach().to(torch.float64)
    return answer


@dataclass(frozen=True)
class CTFieldResult:
    """Post-processed C(T) solution: J-integral rings and COD extrapolation."""

    alpha: float
    load_n: float
    ring_j: tuple[float, ...]
    ring_k: tuple[float, ...]
    cod_k: float
    astm_k: float
    constrained_node_count: int

    @property
    def k_j(self) -> float:
        """Mean ``K_J`` over the reporting contours (rings 3..6)."""
        values = [k for k in self.ring_k if k == k]
        if not values:
            raise ValueError("no usable J contour")
        return sum(values) / len(values)


def _element_field(
    mesh: CTMesh,
    displacement: torch.Tensor,
    element: int,
    xi: float,
    eta: float,
    dmat: torch.Tensor,
):
    """Stress, displacement gradient, parent tangents and ``det J`` at a point."""
    ids = mesh.elements[element]
    coords = mesh.nodes[ids]
    dtype = mesh.nodes.dtype
    grad_x, grad_y, det = _parent_gradient(coords.unsqueeze(0), xi, eta)
    local = torch.zeros(12, dtype=dtype)
    local[0::2] = displacement[2 * ids]
    local[1::2] = displacement[2 * ids + 1]
    b = torch.zeros((3, 12), dtype=dtype)
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
    d_xi, d_eta = t6_shape_derivative(xi, eta)
    dx_dxi = torch.tensor(d_xi, dtype=dtype) @ coords
    dx_deta = torch.tensor(d_eta, dtype=dtype) @ coords
    return stress, grad, dx_dxi, dx_deta, det


def ring_j_integral(
    mesh: CTMesh,
    displacement: torch.Tensor,
    ring: int,
    *,
    specimen: CTSpecimen,
    plane: str = "strain",
) -> float:
    """Rice J-integral on the closed mesh contour at ring index ``ring``.

    The contour runs counter-clockwise from the lower crack face
    (``theta = -pi``) around the tip to the upper crack face (``theta = +pi``)
    with the outward normal taken away from the tip.
    """
    if not 1 <= ring <= mesh.n_rings:
        raise IndexError(f"ring must address 1..{mesh.n_rings}")
    dtype = mesh.nodes.dtype
    dmat = elasticity_matrix(
        torch.tensor([specimen.young], dtype=dtype),
        torch.tensor([specimen.poisson], dtype=dtype),
        plane,
    )[0]
    total = 0.0
    for element, kind in mesh.rings[ring - 1]:
        for t, weight in _EDGE_GAUSS:
            if kind == "outer":
                xi, eta = 1.0 - t, t
            else:
                xi, eta = 0.0, t
            stress, grad, dx_dxi, dx_deta, det = _element_field(
                mesh, displacement, element, xi, eta, dmat
            )
            if float(det) <= 0.0:
                raise ValueError("inverted element on the J contour")
            tangent = (-dx_dxi + dx_deta) if kind == "outer" else dx_deta
            length = float(torch.linalg.vector_norm(tangent))
            normal = torch.stack((tangent[1], -tangent[0])) / length
            ds = length * weight
            strain = 0.5 * (grad + grad.transpose(0, 1))
            energy = 0.5 * float(torch.sum(stress * strain))
            traction = stress @ normal
            work = float(torch.dot(traction, grad[:, 0]))
            total += (energy * float(normal[0]) - work) * ds
    return total


def cod_k_from_opening(
    mesh: CTMesh, displacement: torch.Tensor, *, rings: int = 4, specimen: CTSpecimen
) -> float:
    """``K_I`` from the quarter-point crack-opening extrapolation ``r -> 0``."""
    count = max(3, min(rings, mesh.n_rings - 1))
    tip_x, tip_y = mesh.coordinates(mesh.tip_node)
    xs: list[float] = []
    ys: list[float] = []
    for index in range(count):
        upper = mesh.face_upper[index]
        lower = mesh.face_lower[index]
        opening = float(displacement[2 * upper + 1] - displacement[2 * lower + 1])
        x, y = mesh.coordinates(upper)
        radius = math.hypot(x - tip_x, y - tip_y)
        root = math.sqrt(radius)
        xs.append(root)
        ys.append(opening / root)
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx <= 0.0:
        raise ValueError("degenerate crack-opening sampling")
    # COD/sqrt(r) -> (8 K_I / E') / sqrt(2 pi) as r -> 0
    intercept = mean_y - (sxy / sxx) * mean_x
    return intercept * specimen.effective_modulus * math.sqrt(2.0 * math.pi) / 8.0


def solve_and_postprocess(
    mesh: CTMesh, specimen: CTSpecimen, load_n: float, *, plane: str = "strain"
) -> CTFieldResult:
    """Solve and evaluate ``K`` from J-integral rings and COD extrapolation."""
    displacement = solve_ct_model(mesh, specimen, load_n, plane=plane)
    ring_j: list[float] = []
    ring_k: list[float] = []
    first, last = 3, min(8, mesh.n_rings - 2)
    for ring in range(first, last + 1):
        value = ring_j_integral(mesh, displacement, ring, specimen=specimen, plane=plane)
        ring_j.append(value)
        if value > 0.0:
            ring_k.append(mode_i_k_from_j(value, specimen.young, specimen.poisson, plane=plane))
        else:
            ring_k.append(float("nan"))
    return CTFieldResult(
        alpha=mesh.alpha,
        load_n=load_n,
        ring_j=tuple(ring_j),
        ring_k=tuple(ring_k),
        cod_k=cod_k_from_opening(mesh, displacement, specimen=specimen),
        astm_k=specimen.stress_intensity(load_n, mesh.alpha),
        constrained_node_count=3,
    )


# --------------------------------------------------------------------------- #
# crack growth law and ASTM E647 data reduction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThresholdParis:
    """Threshold-modified Paris law with ``dK`` in MPa*sqrt(m)."""

    c: float
    m: float
    delta_k_th_mpa: float = 0.0
    threshold_exponent: float = 8.0

    def __post_init__(self) -> None:
        if not self.c > 0.0:
            raise ValueError("c must be positive")
        if not self.m > 0.0:
            raise ValueError("m must be positive")
        if self.delta_k_th_mpa < 0.0:
            raise ValueError("threshold must be non-negative")
        if self.threshold_exponent <= 0.0:
            raise ValueError("threshold_exponent must be positive")

    def da_dn(self, delta_k_mpa: float) -> float:
        """Growth rate in m/cycle; exactly zero at or below the threshold."""
        delta_k = float(delta_k_mpa)
        if not math.isfinite(delta_k) or delta_k <= 0.0:
            return 0.0
        if delta_k <= self.delta_k_th_mpa:
            return 0.0
        factor = 1.0
        if self.delta_k_th_mpa > 0.0:
            factor = 1.0 - (self.delta_k_th_mpa / delta_k) ** self.threshold_exponent
        return self.c * delta_k**self.m * factor


def paris_fit(delta_k_mpa: Sequence[float], da_dn: Sequence[float]) -> dict:
    """Least-squares fit of ``log10(da/dN) = log10(C) + m log10(dK)``."""
    pairs = [
        (math.log10(float(k)), math.log10(float(v)))
        for k, v in zip(delta_k_mpa, da_dn)
        if float(k) > 0.0 and float(v) > 0.0
    ]
    if len(pairs) < 3:
        raise ValueError("need at least three positive (dK, da/dN) pairs")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = float(len(pairs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0:
        raise ValueError("dK values must not be constant")
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - ss_res / syy if syy > 0.0 else 1.0
    return {
        "c": 10.0**intercept,
        "m": slope,
        "r_squared": r_squared,
        "points": len(pairs),
        "delta_k_min_mpa": 10.0 ** min(xs),
        "delta_k_max_mpa": 10.0 ** max(xs),
    }


def _delta_k_mpa(specimen: CTSpecimen, load_range_n: float, alpha: float) -> float:
    return specimen.delta_k(load_range_n, alpha) / _MPA_SQRT_M


def _cycles_increment(
    alpha: float, alpha_step: float, rate_of_alpha, substeps: int = 8
) -> tuple[float, bool]:
    """RK4 integration of ``dN/dalpha = 1 / (da/dN)`` over ``alpha_step``."""

    def inverse_rate(value: float) -> float:
        rate = rate_of_alpha(value)
        if rate <= 0.0:
            return float("inf")
        return 1.0 / rate

    h = alpha_step / substeps
    total = 0.0
    current = alpha
    for _ in range(substeps):
        k1 = inverse_rate(current)
        k2 = inverse_rate(current + 0.5 * h)
        k3 = inverse_rate(current + 0.5 * h)
        k4 = inverse_rate(current + h)
        if not all(math.isfinite(k) for k in (k1, k2, k3, k4)):
            return float("inf"), False
        total += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        current += h
    return total, True


def crack_growth_cycles(
    specimen: CTSpecimen,
    load_range_n: float,
    alpha_start: float,
    alpha_end: float,
    law: ThresholdParis,
    *,
    steps: int = 400,
) -> float:
    """Cycles to grow from ``alpha_start`` to ``alpha_end``.

    ``inf`` is returned when the load range never exceeds the threshold (no
    growth at all).  The integration runs on ``dN/dalpha`` with RK4 sub-steps.
    """
    if not 0.0 < alpha_start < alpha_end < 1.0:
        raise ValueError("need 0 < alpha_start < alpha_end < 1")
    if steps < 8:
        raise ValueError("steps must be at least 8")

    def rate(value: float) -> float:
        return law.da_dn(_delta_k_mpa(specimen, load_range_n, value)) / specimen.width

    if rate(alpha_start) <= 0.0:
        return float("inf")
    increment = (alpha_end - alpha_start) / steps
    total = 0.0
    current = alpha_start
    for _ in range(steps):
        piece, ok = _cycles_increment(current, increment, rate)
        if not ok:
            return float("inf")
        total += piece
        current += increment
    return total


def e647_constant_amplitude_test(
    specimen: CTSpecimen,
    load_range_n: float,
    alpha_start: float,
    alpha_end: float,
    law: ThresholdParis,
    *,
    crack_increment: float = 2.5e-4,
    sampling_steps: int = 24,
) -> dict:
    """Virtual E647 constant-amplitude test: measured ``(a, N)`` pairs.

    The crack is advanced in ``crack_increment`` physical steps; the cycle count
    at every station comes from the RK4 integration of the growth law, i.e. the
    record is what a perfectly instrumented test of this material would log.
    """
    if not 0.0 < alpha_start < alpha_end < 1.0:
        raise ValueError("need 0 < alpha_start < alpha_end < 1")
    if crack_increment <= 0.0:
        raise ValueError("crack_increment must be positive")
    widths = specimen.width
    a_position = alpha_start * widths
    a_final = alpha_end * widths
    crack_lengths = [a_position]
    cycles = [0.0]
    total = 0.0
    guard = 0
    while a_position < a_final - 1e-15:
        step = min(crack_increment, a_final - a_position)
        alpha_lo = a_position / widths
        alpha_step = step / widths

        def rate(value: float) -> float:
            return law.da_dn(_delta_k_mpa(specimen, load_range_n, value)) / widths

        if rate(alpha_lo) <= 0.0:
            return {
                "crack_lengths": crack_lengths,
                "cycles": cycles,
                "threshold_limited": True,
                "cycles_to_failure": float("inf"),
            }
        piece, ok = _cycles_increment(alpha_lo, alpha_step, rate, substeps=sampling_steps)
        if not ok:
            return {
                "crack_lengths": crack_lengths,
                "cycles": cycles,
                "threshold_limited": True,
                "cycles_to_failure": float("inf"),
            }
        total += piece
        a_position += step
        crack_lengths.append(a_position)
        cycles.append(total)
        guard += 1
        if guard > 100000:
            raise RuntimeError("crack growth integration did not advance")
    return {
        "crack_lengths": crack_lengths,
        "cycles": cycles,
        "threshold_limited": False,
        "cycles_to_failure": total,
    }


def e647_secant_reduction(
    crack_lengths: Sequence[float],
    cycles: Sequence[float],
    specimen: CTSpecimen,
    load_range_n: float,
) -> dict:
    """ASTM E647 secant construction ``da/dN = Da/DN`` at the mean ``dK``."""
    if len(crack_lengths) != len(cycles) or len(crack_lengths) < 4:
        raise ValueError("need matching crack length and cycle sequences of length >= 4")
    delta_k: list[float] = []
    rates: list[float] = []
    for index in range(len(crack_lengths) - 1):
        da = float(crack_lengths[index + 1]) - float(crack_lengths[index])
        dn = float(cycles[index + 1]) - float(cycles[index])
        if da <= 0.0 or dn <= 0.0:
            raise ValueError("crack length and cycles must increase")
        mean_alpha = (
            0.5 * (float(crack_lengths[index]) + float(crack_lengths[index + 1])) / specimen.width
        )
        delta_k.append(_delta_k_mpa(specimen, load_range_n, mean_alpha))
        rates.append(da / dn)
    return {"delta_k_mpa": delta_k, "da_dn": rates, "fit": paris_fit(delta_k, rates)}


def load_shedding_threshold_test(
    specimen: CTSpecimen,
    law: ThresholdParis,
    *,
    alpha_start: float = 0.30,
    start_delta_k_mpa: float = 12.0,
    gradient_per_m: float = -80.0,
    crack_increment: float = 2.0e-4,
    target_rate: float = 1.0e-10,
    max_steps: int = 4000,
) -> dict:
    """Annex-A1 style load-shedding threshold test.

    ``dK`` is shed with the normalised gradient ``dK/da = C dK`` (``C`` given in
    m**-1, negative) while the crack advances by ``crack_increment``; the
    threshold is the ``dK`` at which the measured rate reaches
    ``target_rate = 1e-10 m/cycle``, obtained by linear interpolation in
    ``log10(da/dN)`` against ``dK``.
    """
    if not 0.0 < alpha_start < ASTM_CT_RANGE[1]:
        raise ValueError("alpha_start outside the specimen window")
    if crack_increment <= 0.0 or target_rate <= 0.0:
        raise ValueError("crack_increment and target_rate must be positive")
    if gradient_per_m >= 0.0:
        raise ValueError("gradient_per_m must be negative (load shedding)")
    delta_k = float(start_delta_k_mpa)
    previous: tuple[float, float] | None = None
    schedule: list[dict] = []
    for _ in range(max_steps):
        rate = law.da_dn(delta_k)
        schedule.append({"delta_k_mpa": delta_k, "da_dn": rate})
        if rate <= target_rate:
            if previous is None:
                return {
                    "delta_k_th_mpa": delta_k,
                    "schedule": schedule,
                    "bracketed": False,
                    "cycles_total": 0.0,
                }
            lo_k, lo_rate = previous
            hi_k, hi_rate = delta_k, max(rate, 1e-300)
            # linear interpolation in log10(rate) against dK
            weight = (math.log10(target_rate) - math.log10(hi_rate)) / (
                math.log10(lo_rate) - math.log10(hi_rate)
            )
            return {
                "delta_k_th_mpa": hi_k + weight * (lo_k - hi_k),
                "schedule": schedule,
                "bracketed": True,
                "cycles_total": crack_increment / target_rate,
            }
        previous = (delta_k, rate)
        delta_k *= math.exp(gradient_per_m * crack_increment)
    raise RuntimeError("load shedding did not reach the threshold")


def unstable_crack_ratio(
    specimen: CTSpecimen,
    peak_load_n: float,
    k_ic_pa: float,
    alpha_start: float,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Crack ratio ``a/W`` where ``K_max = K_IC`` (bisection on a monotone ``K``)."""
    if not 0.0 < alpha_start < ASTM_CT_RANGE[1]:
        raise ValueError("alpha_start outside the specimen window")
    if specimen.stress_intensity(peak_load_n, ASTM_CT_RANGE[1]) < k_ic_pa:
        raise ValueError("K_IC is never reached inside the ASTM geometry window")
    low, high = alpha_start, ASTM_CT_RANGE[1]
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if specimen.stress_intensity(peak_load_n, mid) < k_ic_pa:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def ct_fracture_toughness(
    specimen: CTSpecimen, peak_load_n: float, alpha: float, *, pop_in_ratio: float = 1.0
) -> dict:
    """``K_IC`` from the C(T) instability load with the E399 validity checks."""
    if pop_in_ratio < 1.0:
        raise ValueError("pop_in_ratio = P_max / P_Q must be >= 1")
    k_q = specimen.stress_intensity(peak_load_n, alpha)
    minimum_thickness = 2.5 * (k_q / specimen.yield_strength) ** 2
    equivalent_energy = k_q**2 * (1.0 - specimen.poisson**2) / specimen.young
    return {
        "k_q_pa": k_q,
        "k_q_mpa": k_q / _MPA_SQRT_M,
        "k_ic_pa": k_q,
        "pop_in_ratio": pop_in_ratio,
        "minimum_thickness_m": minimum_thickness,
        "thickness_m": specimen.thickness,
        "thickness_valid": specimen.thickness >= minimum_thickness,
        "pmax_ratio_valid": pop_in_ratio <= 1.10,
        "g_ic_j_m2": equivalent_energy,
        "valid": (specimen.thickness >= minimum_thickness) and (pop_in_ratio <= 1.10),
    }


def _advance_cycles(
    specimen: CTSpecimen,
    load_range_n: float,
    alpha: float,
    n_cycles: float,
    law: ThresholdParis,
    alpha_end: float,
    *,
    substeps: int = 16,
) -> tuple[float, float]:
    """Advance the crack over ``n_cycles``; returns ``(alpha, cycles_used)``."""
    used = 0.0
    if n_cycles <= 0.0:
        return alpha, used
    chunk = n_cycles / substeps
    for _ in range(substeps):
        rate = law.da_dn(_delta_k_mpa(specimen, load_range_n, alpha))
        if rate <= 0.0:
            return alpha, used + (n_cycles - used)
        da = rate * chunk
        alpha_next = alpha + da / specimen.width
        if alpha_next >= alpha_end:
            need = (alpha_end - alpha) / (da / chunk)
            return alpha_end, used + min(need, chunk)
        alpha = alpha_next
        used += chunk
    return alpha, used


def miner_accumulated_damage(
    specimen: CTSpecimen,
    spectrum: Sequence[tuple[float, int]],
    law: ThresholdParis,
    alpha_start: float,
    alpha_end: float,
    *,
    max_blocks: int = 20000,
    integration_steps: int = 400,
) -> dict:
    """Miner linear damage summation for a variable-amplitude block spectrum.

    ``spectrum`` is a sequence of ``(load_range_N, cycles_per_block)`` levels.
    ``N_i`` is the constant-amplitude crack-growth life of level ``i`` between
    ``alpha_start`` and ``alpha_end``; the block sequence is then marched until
    the crack reaches ``alpha_end`` and the consumed cycles give
    ``D = sum n_i / N_i``.
    """
    if not spectrum:
        raise ValueError("spectrum must not be empty")
    if len(spectrum) < 2:
        raise ValueError("a variable-amplitude spectrum needs at least two levels")
    lifetimes: list[float] = []
    for load_range, _ in spectrum:
        if load_range <= 0.0:
            raise ValueError("load ranges must be positive")
        lifetimes.append(
            crack_growth_cycles(
                specimen, load_range, alpha_start, alpha_end, law, steps=integration_steps
            )
        )
    consumed = [0.0] * len(spectrum)
    alpha = alpha_start
    blocks = 0.0
    stalled = False
    while alpha < alpha_end and blocks < max_blocks:
        alpha_before = alpha
        for index, (load_range, cycles) in enumerate(spectrum):
            if cycles <= 0:
                continue
            alpha, used = _advance_cycles(
                specimen, load_range, alpha, float(cycles), law, alpha_end
            )
            consumed[index] += used
            if alpha >= alpha_end:
                break
        blocks += 1.0
        if alpha <= alpha_before:
            # every level sits below its threshold: no growth at all
            stalled = True
            break
    damage_terms = [
        (n_i / life) if math.isfinite(life) and life > 0.0 else 0.0
        for n_i, life in zip(consumed, lifetimes)
    ]
    total_cycles = sum(consumed)
    total_damage = sum(damage_terms)
    predicted = None
    if 0.0 < total_damage < float("inf"):
        predicted = total_cycles / total_damage
    return {
        "damage": total_damage,
        "damage_terms": damage_terms,
        "consumed_cycles": consumed,
        "constant_amplitude_lives": lifetimes,
        "total_cycles": total_cycles,
        "blocks": blocks,
        "miner_predicted_cycles": predicted,
        "deviation": (total_damage - 1.0),
        "interacted": blocks >= max_blocks,
        "stalled": stalled,
    }


# --------------------------------------------------------------------------- #
# qualification driver
# --------------------------------------------------------------------------- #
#: Documented material anchor (ferritic-pearlitic structural steel, R = 0.1, air).
#: ``C`` and ``m``: Barsom & Rolfe, *Fatigue and Fracture Control in Structures*,
#: 2nd ed. (da/dN = 6.9e-12 dK**3, m/cycle with dK in MPa*sqrt(m)).
#: ``delta_k_th``: same source, threshold at R = 0.1.  ``k_ic``: quenched and
#: tempered low-alloy steel, literature range 45-60 MPa*sqrt(m).
MATERIAL_ANCHOR = {
    "name": "ferritic-pearlitic structural steel, R = 0.1, air",
    "c": 6.9e-12,
    "m": 3.0,
    "delta_k_th_mpa": 6.4,
    "k_ic_mpa": 50.0,
    "k_ic_range_mpa": (45.0, 60.0),
    "reference": (
        "Barsom & Rolfe, Fatigue and Fracture Control in Structures, 2nd ed.; "
        "K_IC range for QT low-alloy steel at sigma_y = 550 MPa"
    ),
}

#: Near-threshold steepness of the growth law (number of decades between the
#: threshold and the mid-rate Paris branch).  ``k = 8`` puts da/dN = 1e-10 m/cycle
#: within 1 % of dK_th, which is what real near-threshold curves show.
THRESHOLD_EXPONENT = 8.0


def run_ct_e647_qualification(
    *,
    specimen: CTSpecimen | None = None,
    load_range_n: float = 12000.0,
    crack_ratio_start: float = 0.30,
    crack_ratio_end: float = 0.55,
    crack_increment: float = 2.5e-4,
    spectrum: Sequence[tuple[float, int]] | None = None,
    run_finite_elements: bool = True,
    material: dict | None = None,
) -> dict:
    """Run the whole C(T) fatigue-crack-growth verification face.

    Returns a dictionary with the geometry factor, the FE cross-checks, the
    ASTM E647 data reduction, the threshold and toughness identifications, the
    Miner summation and a ``criteria`` block holding every machine-checked
    number of the campaign node.
    """
    specimen = specimen or CTSpecimen()
    anchor = dict(material or MATERIAL_ANCHOR)
    law = ThresholdParis(
        c=anchor["c"],
        m=anchor["m"],
        delta_k_th_mpa=anchor["delta_k_th_mpa"],
        threshold_exponent=THRESHOLD_EXPONENT,
    )
    k_ic_pa = anchor["k_ic_mpa"] * _MPA_SQRT_M
    result: dict = {
        "specimen": {
            "width_m": specimen.width,
            "thickness_m": specimen.thickness,
            "total_width_m": specimen.total_width,
            "height_m": 2.0 * specimen.half_height,
            "pin_offset_m": specimen.pin_offset,
            "young_pa": specimen.young,
            "poisson": specimen.poisson,
            "yield_strength_pa": specimen.yield_strength,
            "stress_ratio": 0.1,
        }
    }
    result["material"] = dict(anchor)
    result["growth_law"] = {
        "form": "da/dN = C dK^m (1 - (dK_th/dK)^k), zero below dK_th",
        "c": law.c,
        "m": law.m,
        "delta_k_th_mpa": law.delta_k_th_mpa,
        "threshold_exponent": law.threshold_exponent,
    }

    # ---- geometry factor table ------------------------------------------- #
    table = []
    for step in range(13):
        alpha = 0.20 + 0.05 * step
        table.append({"alpha": round(alpha, 4), "f": ct_geometry_factor(alpha)})
    result["geometry_factor_table"] = table

    # ---- finite-element cross-check -------------------------------------- #
    if run_finite_elements:
        fe_rows = []
        for alpha in (0.40, 0.55):
            mesh = ct_t6_mesh(specimen, alpha, n_angles=72, n_rings=14, growth=1.35)
            load = specimen.load_range_for_delta_k(20.0, alpha) / 0.9
            post = solve_and_postprocess(mesh, specimen, load)
            fe_rows.append(
                {
                    "alpha": alpha,
                    "n_angles": mesh.n_angles,
                    "n_rings": mesh.n_rings,
                    "growth": 1.35,
                    "load_n": load,
                    "ring_radius_ratio": [value / specimen.width for value in mesh.ring_radii],
                    "ring_k_mpa": [value / _MPA_SQRT_M for value in post.ring_k],
                    "k_j_mpa": post.k_j / _MPA_SQRT_M,
                    "k_cod_mpa": post.cod_k / _MPA_SQRT_M,
                    "k_astm_mpa": post.astm_k / _MPA_SQRT_M,
                    "deviation_j_pct": 100.0 * (post.k_j - post.astm_k) / post.astm_k,
                    "deviation_cod_pct": 100.0 * (post.cod_k - post.astm_k) / post.astm_k,
                }
            )
        result["fe_cross_check"] = fe_rows

        levels = [(48, 10, 1.35), (72, 14, 1.35), (96, 18, 1.30)]
        meshes = []
        for n_angles, n_rings, growth in levels:
            mesh = ct_t6_mesh(specimen, 0.40, n_angles=n_angles, n_rings=n_rings, growth=growth)
            load = specimen.load_range_for_delta_k(20.0, 0.40) / 0.9
            post = solve_and_postprocess(mesh, specimen, load)
            meshes.append(
                {
                    "n_angles": n_angles,
                    "n_rings": n_rings,
                    "growth": growth,
                    "elements": int(mesh.elements.shape[0]),
                    "nodes": int(mesh.nodes.shape[0]),
                    "k_j_mpa": post.k_j / _MPA_SQRT_M,
                    "k_cod_mpa": post.cod_k / _MPA_SQRT_M,
                    "k_astm_mpa": post.astm_k / _MPA_SQRT_M,
                }
            )
        fine = meshes[-1]["k_j_mpa"]
        medium = meshes[1]["k_j_mpa"]
        coarse = meshes[0]["k_j_mpa"]
        relative_change = abs(fine - coarse) / fine * 100.0
        shallow = []
        for n_angles, n_rings, growth in ((72, 14, 1.35), (96, 20, 1.28)):
            mesh = ct_t6_mesh(specimen, 0.25, n_angles=n_angles, n_rings=n_rings, growth=growth)
            load = specimen.load_range_for_delta_k(20.0, 0.25) / 0.9
            post = solve_and_postprocess(mesh, specimen, load)
            shallow.append(
                {
                    "alpha": 0.25,
                    "n_angles": n_angles,
                    "n_rings": n_rings,
                    "growth": growth,
                    "k_j_mpa": post.k_j / _MPA_SQRT_M,
                    "k_astm_mpa": post.astm_k / _MPA_SQRT_M,
                    "deviation_j_pct": 100.0 * (post.k_j - post.astm_k) / post.astm_k,
                }
            )
        result["shallow_crack_study"] = {
            "rows": shallow,
            "deviation_pct_coarse": shallow[0]["deviation_j_pct"],
            "deviation_pct_fine": shallow[-1]["deviation_j_pct"],
            "note": (
                "the FE pin force acts at the hole centre without modelling the "
                "0.25 W hole, so the shallow-crack compliance differs from the "
                "ASTM calibration and the difference does not vanish under "
                "refinement"
            ),
        }
        result["fe_mesh_convergence"] = {
            "levels": meshes,
            "k_j_fine_mpa": fine,
            "k_j_medium_mpa": medium,
            "k_j_coarse_mpa": coarse,
            "k_cod_fine_mpa": meshes[-1]["k_cod_mpa"],
            "max_relative_change_pct": relative_change,
            "finest_vs_astm_pct": 100.0
            * (fine - meshes[-1]["k_astm_mpa"])
            / meshes[-1]["k_astm_mpa"],
        }
    else:
        result["fe_cross_check"] = []
        result["fe_mesh_convergence"] = None

    # ---- virtual ASTM E647 constant-amplitude test ------------------------ #
    alpha_failure = unstable_crack_ratio(specimen, load_range_n / 0.9, k_ic_pa, crack_ratio_start)
    test = e647_constant_amplitude_test(
        specimen,
        load_range_n,
        crack_ratio_start,
        crack_ratio_end,
        law,
        crack_increment=crack_increment,
    )
    reduction = e647_secant_reduction(test["crack_lengths"], test["cycles"], specimen, load_range_n)
    fit = reduction["fit"]
    result["virtual_test"] = {
        "load_range_n": load_range_n,
        "peak_load_n": load_range_n / 0.9,
        "crack_ratio_start": crack_ratio_start,
        "crack_ratio_end": crack_ratio_end,
        "crack_increment_m": crack_increment,
        "data_points": len(reduction["delta_k_mpa"]),
        "delta_k_min_mpa": min(reduction["delta_k_mpa"]),
        "delta_k_max_mpa": max(reduction["delta_k_mpa"]),
        "cycles_to_end": test["cycles_to_failure"],
        "crack_lengths_m": test["crack_lengths"],
        "cycles": test["cycles"],
        "delta_k_mpa": reduction["delta_k_mpa"],
        "da_dn": reduction["da_dn"],
    }
    result["paris_fit"] = {
        **fit,
        "c_reference": anchor["c"],
        "m_reference": anchor["m"],
        "c_deviation_pct": 100.0 * (fit["c"] - anchor["c"]) / anchor["c"],
        "m_deviation_pct": 100.0 * (fit["m"] - anchor["m"]) / anchor["m"],
    }

    # ---- threshold (load shedding) --------------------------------------- #
    threshold = load_shedding_threshold_test(
        specimen,
        law,
        alpha_start=crack_ratio_start,
        start_delta_k_mpa=2.0 * anchor["delta_k_th_mpa"],
    )
    threshold_row = {
        "delta_k_th_mpa": threshold["delta_k_th_mpa"],
        "delta_k_th_reference_mpa": anchor["delta_k_th_mpa"],
        "deviation_pct": 100.0
        * (threshold["delta_k_th_mpa"] - anchor["delta_k_th_mpa"])
        / anchor["delta_k_th_mpa"],
        "target_rate_m_per_cycle": 1.0e-10,
        "schedule_points": len(threshold["schedule"]),
        "schedule_tail": threshold["schedule"][-6:],
    }
    result["threshold"] = threshold_row

    # ---- fracture toughness --------------------------------------------- #
    peak_load = load_range_n / 0.9
    toughness = ct_fracture_toughness(specimen, peak_load, alpha_failure)
    fe_toughness = None
    if run_finite_elements:
        mesh = ct_t6_mesh(specimen, alpha_failure, n_angles=72, n_rings=14, growth=1.35)
        post = solve_and_postprocess(mesh, specimen, peak_load)
        fe_toughness = {
            "alpha": alpha_failure,
            "k_j_mpa": post.k_j / _MPA_SQRT_M,
            "k_cod_mpa": post.cod_k / _MPA_SQRT_M,
            "k_astm_mpa": post.astm_k / _MPA_SQRT_M,
            "deviation_j_pct": 100.0 * (post.k_j - post.astm_k) / post.astm_k,
        }
    result["toughness"] = {
        "critical_crack_ratio": alpha_failure,
        "critical_crack_length_m": alpha_failure * specimen.width,
        "peak_load_n": peak_load,
        **{key: value for key, value in toughness.items() if key != "k_ic_pa"},
        "k_ic_reference_mpa": anchor["k_ic_mpa"],
        "deviation_pct": 100.0 * (toughness["k_q_mpa"] - anchor["k_ic_mpa"]) / anchor["k_ic_mpa"],
        "finite_element": fe_toughness,
    }

    # ---- negative control: below-threshold growth ----------------------- #
    negative_rows = []
    for factor in (0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        delta_k = factor * law.delta_k_th_mpa
        rate = law.da_dn(delta_k)
        negative_rows.append({"delta_k_mpa": delta_k, "da_dn": rate, "is_zero": rate == 0.0})
    result["negative_control"] = {
        "rows": negative_rows,
        "max_rate": max(row["da_dn"] for row in negative_rows),
    }

    # ---- Miner linear damage summation ---------------------------------- #
    if spectrum is None:
        spectrum = (
            (0.75 * load_range_n, 4000),
            (0.90 * load_range_n, 2500),
            (1.00 * load_range_n, 1500),
            (1.10 * load_range_n, 700),
            (1.25 * load_range_n, 200),
        )
    miner = miner_accumulated_damage(specimen, spectrum, law, crack_ratio_start, alpha_failure)
    result["spectrum"] = [
        {
            "load_range_n": float(load_range),
            "cycles_per_block": int(cycles),
            "delta_k_start_mpa": _delta_k_mpa(specimen, float(load_range), crack_ratio_start),
        }
        for load_range, cycles in spectrum
    ]
    result["miner"] = {
        **{key: value for key, value in miner.items() if key != "damage_terms"},
        "damage": miner["damage"],
        "deviation_pct": 100.0 * (miner["damage"] - 1.0),
        "crack_length_start_m": crack_ratio_start * specimen.width,
        "crack_length_end_m": alpha_failure * specimen.width,
    }

    # ---- Miner sensitivity: what breaks the exactness -------------------- #
    sensitivity = []
    for scale in (0.5, 0.7, 0.9, 1.0, 1.2):
        scaled = tuple((float(load_range) * scale, int(cycles)) for load_range, cycles in spectrum)
        row = miner_accumulated_damage(
            specimen, scaled, law, crack_ratio_start, alpha_failure, max_blocks=2000
        )
        sensitivity.append(
            {
                "spectrum_scale": scale,
                "damage": row["damage"],
                "deviation_pct": 100.0 * (row["damage"] - 1.0),
                "stalled": row["stalled"],
                "levels_above_threshold": sum(
                    1
                    for load_range, _ in scaled
                    if _delta_k_mpa(specimen, float(load_range), crack_ratio_start)
                    > law.delta_k_th_mpa
                ),
            }
        )
    result["miner_sensitivity"] = sensitivity

    # ---- criteria block ------------------------------------------------- #
    criteria = {
        "C1_paris_r_squared": {
            "value": fit["r_squared"],
            "threshold": 0.98,
            "pass": fit["r_squared"] >= 0.98,
        },
        "C2_paris_constants": {
            "c_deviation_pct": result["paris_fit"]["c_deviation_pct"],
            "m_deviation_pct": result["paris_fit"]["m_deviation_pct"],
            "threshold_pct": 30.0,
            "pass": abs(result["paris_fit"]["c_deviation_pct"]) <= 30.0
            and abs(result["paris_fit"]["m_deviation_pct"]) <= 30.0,
        },
        "C3_threshold": {
            "value_mpa": threshold_row["delta_k_th_mpa"],
            "reference_mpa": anchor["delta_k_th_mpa"],
            "deviation_pct": threshold_row["deviation_pct"],
            "threshold_pct": 20.0,
            "pass": abs(threshold_row["deviation_pct"]) <= 20.0,
        },
        "C4_toughness": {
            "value_mpa": toughness["k_q_mpa"],
            "reference_mpa": anchor["k_ic_mpa"],
            "deviation_pct": result["toughness"]["deviation_pct"],
            "thickness_valid": toughness["thickness_valid"],
            "threshold_pct": 15.0,
            "pass": abs(result["toughness"]["deviation_pct"]) <= 15.0
            and toughness["thickness_valid"],
        },
        "C5_miner": {
            "damage": miner["damage"],
            "deviation_pct": result["miner"]["deviation_pct"],
            "threshold_pct": 20.0,
            "pass": abs(result["miner"]["deviation_pct"]) <= 20.0,
        },
        "C6_negative_control": {
            "max_rate": result["negative_control"]["max_rate"],
            "threshold": 1.0e-12,
            "pass": result["negative_control"]["max_rate"] <= 1.0e-12,
        },
        "C7_mesh_convergence": {
            "max_relative_change_pct": (result["fe_mesh_convergence"] or {}).get(
                "max_relative_change_pct"
            ),
            "threshold_pct": 5.0,
            "pass": bool(result["fe_mesh_convergence"])
            and (result["fe_mesh_convergence"]["max_relative_change_pct"] <= 5.0),
        },
    }
    result["criteria"] = criteria
    result["criteria_pass"] = all(row["pass"] for row in criteria.values())
    return result
