"""Quadratic (Q8 serendipity) plane element for plane stress and plane strain.

The production plane continuum path (``tensorfem.continuum``) only carried the
constant-strain triangle and the bilinear Q4: both are O(h) in stress, so a stress
qualification against a 3 % pointwise gate needs very fine meshes.  This module adds
the standard 8-node serendipity element with 3x3 Gauss integration and the textbook
Barlow/Zienkiewicz-Zhu stress recovery, so that quadratic fields (bending-dominated
membranes, curved boundaries) converge at O(h^2).

Conventions (identical to ``tensorfem.continuum.q4_stiffness``):

* node order ``(-1,-1) (1,-1) (1,1) (-1,1) (0,-1) (1,0) (0,1) (-1,0)``;
* DOF order ``(u, v)`` per node, element vector ``[u0, v0, u1, v1, ...]``;
* parent Jacobian ``jac[a, b] = dX_b / dxi_a`` so that ``dN/dX = dN/dxi @ inv(jac)``.

Every routine accepts a batch of elements (``ne`` leading dimension).  Nothing here
mutates the existing plane path; ``solve_q8_plane`` assembles with
``sparse_core.assemble_coo`` and solves with ``sparse_core.solve_sparse_static``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .continuum import _b_matrix, elasticity_matrix
from .sparse_core import assemble_coo, solve_sparse_static

# 3x3 Gauss rule: exact for products of degree <= 5 in each direction.
_GAUSS_1D = (-math.sqrt(0.6), 0.0, math.sqrt(0.6))
_WEIGHTS_1D = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)

#: 9 integration points ordered with the second parent coordinate fastest.
GAUSS_POINTS_3X3 = tuple(
    (a, b) for a in _GAUSS_1D for b in _GAUSS_1D
)
GAUSS_WEIGHTS_3X3 = tuple(
    (wa, wb) for wa in _WEIGHTS_1D for wb in _WEIGHTS_1D
)

#: Node order of the serendipity element in parent coordinates.
NODE_ORDER = (
    (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
    (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
)

#: Number of spurious-mechanism-free rigid body modes expected from one element.
RIGID_BODY_MODES = 3


def q8_shape(xi: float, eta: float) -> torch.Tensor:
    """Serendipity shape functions at one parent point, in ``NODE_ORDER``."""
    return torch.tensor(
        (
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        ),
        dtype=torch.float64,
    )


def q8_shape_derivative(xi: float, eta: float) -> torch.Tensor:
    """Parent derivatives ``[dN/dxi, dN/deta]`` at one parent point (sympy-verified)."""
    return torch.tensor(
        (
            (-(eta + 2 * xi) * (eta - 1) / 4.0, -(2 * eta + xi) * (xi - 1) / 4.0),
            ((eta - 1) * (eta - 2 * xi) / 4.0, (2 * eta - xi) * (xi + 1) / 4.0),
            ((eta + 1) * (eta + 2 * xi) / 4.0, (2 * eta + xi) * (xi + 1) / 4.0),
            ((2 * xi - eta) * (eta + 1) / 4.0, (xi - 2 * eta) * (xi - 1) / 4.0),
            (xi * (eta - 1), (xi * xi - 1.0) / 2.0),
            ((1.0 - eta * eta) / 2.0, -eta * (xi + 1.0)),
            (-xi * (eta + 1.0), (1.0 - xi * xi) / 2.0),
            ((eta * eta - 1.0) / 2.0, eta * (xi - 1.0)),
        ),
        dtype=torch.float64,
    )


def q8_strain_matrix(xi: float, eta: float, coordinates: torch.Tensor):
    """Strain-displacement matrix and Jacobian determinant of a batch of Q8 elements.

    ``coordinates`` has shape ``(ne, 8, 2)``; the returned matrix has shape
    ``(ne, 3, 16)`` with the engineering shear row last.
    """
    natural = q8_shape_derivative(xi, eta).to(dtype=coordinates.dtype, device=coordinates.device)
    jacobian = torch.einsum("eia,ib->eab", coordinates, natural)
    det = torch.linalg.det(jacobian)
    if bool(torch.any(det <= 0.0)):
        raise ValueError("Q8 element has non-positive Jacobian")
    derivative = torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jacobian))
    return _b_matrix(derivative), det


def q8_element_area(coordinates: torch.Tensor) -> torch.Tensor:
    """Exact (rule-integrated) area of each element in the batch."""
    area = torch.zeros(len(coordinates), dtype=coordinates.dtype, device=coordinates.device)
    for (xi, eta), (wa, wb) in zip(GAUSS_POINTS_3X3, GAUSS_WEIGHTS_3X3):
        _, det = q8_strain_matrix(xi, eta, coordinates)
        area = area + det * wa * wb
    return area


def q8_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor):
    """Element stiffness ``(ne, 16, 16)`` and the Gauss-point strain matrices.

    The strain matrices are returned in the ``GAUSS_POINTS_3X3`` order so callers can
    recover superconvergent stresses without re-evaluating the element.
    """
    ne = len(coordinates)
    stiffness = torch.zeros(
        (ne, 16, 16), dtype=coordinates.dtype, device=coordinates.device
    )
    matrices = []
    for (xi, eta), (wa, wb) in zip(GAUSS_POINTS_3X3, GAUSS_WEIGHTS_3X3):
        matrix, det = q8_strain_matrix(xi, eta, coordinates)
        matrices.append(matrix)
        stiffness = stiffness + (
            matrix.transpose(1, 2) @ dmat @ matrix
        ) * (det * thickness * wa * wb)[:, None, None]
    return stiffness, torch.stack(matrices, 1)


def q8_element_dofs(elements: torch.Tensor) -> torch.Tensor:
    """Global DOF indices ``(ne, 16)`` for ``(u, v)`` interleaved nodal ordering."""
    dofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), 2)
    return dofs.reshape(len(elements), 16).long()


# --------------------------------------------------------------------------- #
# stress recovery
# --------------------------------------------------------------------------- #
def _extrapolation_operator(dtype, device) -> torch.Tensor:
    """Least-squares map from the 9 Gauss-point stresses to the 8 element nodes.

    The Q8 stress field is bilinear-in-parent-per-component for a quadratic
    displacement field, so a 6-term basis ``{1, xi, eta, xi*eta, xi^2, eta^2}`` is fitted
    to the 9 superconvergent Gauss values and evaluated at the nodes.
    """
    basis = []
    for xi, eta in GAUSS_POINTS_3X3:
        basis.append((1.0, xi, eta, xi * eta, xi * xi, eta * eta))
    design = torch.tensor(basis, dtype=dtype, device=device)
    nodes = []
    for xi, eta in NODE_ORDER:
        nodes.append((1.0, xi, eta, xi * eta, xi * xi, eta * eta))
    targets = torch.tensor(nodes, dtype=dtype, device=device)
    # design is (9, 6) full column rank -> normal equations are well conditioned.
    gram = design.transpose(0, 1) @ design
    operator = torch.linalg.solve(gram, design.transpose(0, 1))  # (6, 9)
    return targets @ operator  # (8, 9)


def q8_recovered_nodal_stress(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    gauss_stress: torch.Tensor,
) -> torch.Tensor:
    """Area-weighted nodal stress field ``(nnodes, 3)`` from Gauss-point stresses.

    ``gauss_stress`` has shape ``(ne, 9, 3)`` in ``GAUSS_POINTS_3X3`` order.
    """
    operator = _extrapolation_operator(gauss_stress.dtype, gauss_stress.device)
    extrapolated = torch.einsum("nk,ekc->enc", operator, gauss_stress)
    areas = q8_element_area(nodes[elements.long()])
    nodal = torch.zeros(
        (len(nodes), 3), dtype=gauss_stress.dtype, device=gauss_stress.device
    )
    weight = torch.zeros(len(nodes), dtype=gauss_stress.dtype, device=gauss_stress.device)
    nodal.index_put_((elements.reshape(-1),), (extrapolated * areas[:, None, None]).reshape(-1, 3), accumulate=True)
    weight.index_put_((elements.reshape(-1),), areas[:, None].expand(-1, 8).reshape(-1), accumulate=True)
    nonzero = weight > 0.0
    nodal[nonzero] = nodal[nonzero] / weight[nonzero, None]
    return nodal


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Plane von Mises equivalent stress from ``(..., 3)`` components."""
    sxx, syy, sxy = stress[..., 0], stress[..., 1], stress[..., 2]
    return torch.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Q8PlaneResult:
    """Solution of a Q8 plane problem."""

    displacement: torch.Tensor
    reaction: torch.Tensor
    gauss_stress: torch.Tensor
    nodal_stress: torch.Tensor
    centre_stress: torch.Tensor
    strain_energy: torch.Tensor
    stiffness: torch.Tensor
    relative_residual: float


def q8_stress_at(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    displacement: torch.Tensor,
    element: int,
    xi: float,
    eta: float,
    *,
    young: float = 1.0,
    poisson: float = 0.3,
    plane: str = "stress",
) -> torch.Tensor:
    """Stress components of element ``element`` at one parent point.

    The value is single-valued inside an element; on an interface the two incident
    elements may disagree, so probes must be declared interior.
    """
    coordinates = nodes[elements.long()][element]
    dmat = elasticity_matrix(
        torch.tensor([young], dtype=nodes.dtype, device=nodes.device),
        torch.tensor([poisson], dtype=nodes.dtype, device=nodes.device),
        plane,
    )[0]
    matrix, _ = q8_strain_matrix(xi, eta, coordinates.unsqueeze(0))
    connectivity = elements.long()[element]
    local = torch.zeros(16, dtype=nodes.dtype, device=nodes.device)
    local[0::2] = displacement[2 * connectivity]
    local[1::2] = displacement[2 * connectivity + 1]
    return dmat @ matrix[0] @ local


def solve_q8_plane(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    *,
    young: float = 1.0,
    poisson: float = 0.3,
    thickness: float = 1.0,
    plane: str = "stress",
    forces: torch.Tensor | None = None,
    dirichlet: dict[int, float] | None = None,
    rtol: float = 1e-12,
    maxiter: int | None = None,
) -> Q8PlaneResult:
    """Assemble and solve a Q8 plane problem with sparse CG."""
    elements = elements.long()
    ne = len(elements)
    coordinates = nodes[elements]
    dtype, device = nodes.dtype, nodes.device
    dmat = elasticity_matrix(
        torch.tensor([young], dtype=dtype, device=device),
        torch.tensor([poisson], dtype=dtype, device=device),
        plane,
    )[0]
    thick = torch.tensor(thickness, dtype=dtype, device=device)
    stiffness, matrices = q8_stiffness(coordinates, dmat, thick)
    dofs = q8_element_dofs(elements)
    ndof = 2 * len(nodes)
    K = assemble_coo(stiffness, dofs, ndof)
    load = (
        torch.zeros(ndof, dtype=dtype, device=device)
        if forces is None
        else forces.to(dtype=dtype, device=device)
    )
    solved = solve_sparse_static(
        K, load, dirichlet=dict(dirichlet or {}), rtol=rtol, maxiter=maxiter
    )
    displacement = solved.displacement
    local = torch.zeros((ne, 16), dtype=dtype, device=device)
    local[:, 0::2] = displacement[2 * elements]
    local[:, 1::2] = displacement[2 * elements + 1]
    gauss_stress = torch.einsum("ekij,ej->eki", matrices, local)  # (ne, 9, 3) with D applied below
    gauss_stress = torch.einsum("ij,ekj->eki", dmat, gauss_stress)
    centre = torch.zeros((ne, 3), dtype=dtype, device=device)
    for xi, eta in ((0.0, 0.0),):
        matrix, _ = q8_strain_matrix(xi, eta, coordinates)
        centre = (dmat @ matrix @ local.unsqueeze(-1)).squeeze(-1)
    result = Q8PlaneResult(
        displacement=displacement,
        reaction=K @ displacement - load,
        gauss_stress=gauss_stress,
        nodal_stress=q8_recovered_nodal_stress(nodes, elements, gauss_stress),
        centre_stress=centre,
        strain_energy=0.5 * displacement @ K @ displacement,
        stiffness=K,
        relative_residual=float(solved.iterative.relative_residual),
    )
    return result


# --------------------------------------------------------------------------- #
# mesh helpers
# --------------------------------------------------------------------------- #
def q8_mesh(nodes: torch.Tensor, cells: torch.Tensor):
    """Upgrade a Q4 connectivity to Q8 by inserting shared mid-edge nodes.

    ``nodes``/``cells`` describe a Q4 mesh (``cells`` shape ``(ne, 4)``,
    counter-clockwise).  Mid-edge nodes are placed at the straight midpoint of each
    Q4 edge and shared between the incident cells, so the Q8 element geometry is the
    quadratic completion of the Q4 mesh.
    """
    nodes = nodes.detach()
    cells = cells.long()
    dtype, device = nodes.dtype, nodes.device
    extra: list[torch.Tensor] = []
    index: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in index:
            index[key] = len(nodes) + len(extra)
            extra.append(0.5 * (nodes[a] + nodes[b]))
        return index[key]

    quad_cells = []
    for cell in cells.tolist():
        n0, n1, n2, n3 = cell
        quad_cells.append(
            (n0, n1, n2, n3,
             midpoint(n0, n1), midpoint(n1, n2), midpoint(n2, n3), midpoint(n3, n0))
        )
    connectivity = torch.tensor(quad_cells, dtype=torch.long, device=device)
    if extra:
        nodes = torch.cat((nodes, torch.stack(extra).to(dtype=dtype, device=device)), 0)
    return nodes, connectivity


def mapped_q4_grid(corners: torch.Tensor, nx: int, ny: int):
    """Bilinear unit-square grid mapped to ``corners`` (same layout as the Cook mesh)."""
    from .continuum_benchmarks import mapped_q4_mesh

    return mapped_q4_mesh(corners, nx, ny)


def polar_q8_mesh(inner: float, outer: float, n_theta: int, n_radial: int, *, dtype=torch.float64):
    """Annulus Q8 mesh whose circumferential mid-edge nodes sit exactly on the circles.

    ``n_theta`` angular cells and ``n_radial`` radial layers.  The first parent
    coordinate is radial and the second tangential, so every element is positively
    oriented in the Cartesian plane, and the circles are represented by quadratic
    edges rather than polygons.
    """
    if n_theta < 4 or n_radial < 1:
        raise ValueError("need n_theta >= 4 and n_radial >= 1")
    steps = 2 * n_theta
    lookup: dict[tuple[int, int], int] = {}
    coordinates: list[torch.Tensor] = []

    def node(radial_index: int, angle_index: int) -> int:
        radial_index = max(0, min(2 * n_radial, radial_index))
        angle_index %= steps
        key = (radial_index, angle_index)
        if key not in lookup:
            radius = inner + (outer - inner) * radial_index / (2 * n_radial)
            theta = 2.0 * math.pi * angle_index / steps
            lookup[key] = len(coordinates)
            coordinates.append(
                torch.tensor(
                    (radius * math.cos(theta), radius * math.sin(theta)), dtype=dtype
                )
            )
        return lookup[key]

    cells = []
    for j in range(n_theta):
        for i in range(n_radial):
            cells.append(
                (
                    node(2 * i, 2 * j),
                    node(2 * i + 2, 2 * j),
                    node(2 * i + 2, 2 * j + 2),
                    node(2 * i, 2 * j + 2),
                    node(2 * i + 1, 2 * j),
                    node(2 * i + 2, 2 * j + 1),
                    node(2 * i + 1, 2 * j + 2),
                    node(2 * i, 2 * j + 1),
                )
            )
    return torch.stack(coordinates), torch.tensor(cells, dtype=torch.long)
