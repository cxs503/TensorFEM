"""Gauss-point (Barlow) stress recovery for Q4 plane-stress fields — production path.

Why this module exists
----------------------
The production benchmark payload ``additional_benchmark_fields.cook_fields`` stores a
**constant-per-element** von Mises field (one evaluation of ``C B u`` per element).  That
field is ``O(h)`` accurate and, on a structured grid, is *discontinuous* along element
interfaces — so a stress probe that lands on an interface has no unique value and the
pointwise Cook criterion stayed ``WITHHELD`` for protocol reasons, not because the
discretisation was wrong (see ``docs/cook-probe-protocol-audit.md``).

The textbook remedy, used by every post-processor, is the Barlow/Zienkiewicz recovery:

1. evaluate the stress at the ``2 x 2`` Gauss points (the superconvergent points of a
   Q4 element),
2. extrapolate linearly, per element, to the four nodes (tensor product of the 1-D
   two-point extrapolation ``(0.5 +/- 1/(2 sqrt 3))``),
3. average the extrapolated values at shared nodes with element-area weights.

The result is a **continuous, single-valued** nodal field, so a probe on an element
interface is now well posed, and the interpolation error drops from ``O(h)`` to
``O(h^2)``.

Scope / honesty
---------------
* This module is measurement/post-processing only: it does not change the assembly or
  the solver, and it does not touch the fail-closed flags of the production payload
  (``stress_accuracy_qualified`` stays ``False`` for the raw element-centre protocol).
Union API (PR #79 merge)
------------------------
This module carries two numerically equivalent implementations that were developed in
parallel by two agents and are now published together, because both are in production
use:

* the **Torch** kernels (``q4_gauss_stresses``, ``recover_nodal_stress``,
  ``interpolate_nodal_field``, ``locate_point``, ``probe_errors``, ``normalized_error``)
  consumed by ``additional_benchmark_fields`` and
  ``scripts/cook_recovery_production_evidence.py``;
* the **NumPy** kernels (``gauss_point_stresses``, ``nodal_recovered_stress``,
  ``interpolate_nodal_stress``, ``containing_elements``, ``recovery_error``) consumed by
  ``cook_probe_protocol`` and ``cook_recovery_qualification``.

``q4_shape`` and ``element_areas`` exist in both families and are unified by dispatch:
``q4_shape`` returns a NumPy array unless a ``dtype``/``device`` is supplied (Torch), and
``element_areas`` follows the type of its ``nodes`` argument.  ``EXTRAPOLATION_1D`` is a
2x2 ``numpy.ndarray``, readable by both index styles.  The two families are cross-checked
against each other by ``scripts/cook_recovery_production_evidence.py`` and by
``tests/test_stress_recovery.py``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# NumPy reference kernels (the audit path).  Imported under private aliases so the
# public Torch kernels of the same name below keep their signatures.
from .cook_stress_reference import (
    plane_stress_matrix as _reference_plane_stress_matrix,
    q4_strain_matrix as _reference_q4_strain_matrix,
)

__all__ = [
    "EXTRAPOLATION_1D",
    "GAUSS",
    "GAUSS_ORDER",
    "NODE_ORDER",
    "containing_elements",
    "element_areas",
    "extrapolation_matrix",
    "gauss_point_stresses",
    "interpolate_nodal_field",
    "interpolate_nodal_stress",
    "local_coordinates",
    "locate_point",
    "nodal_recovered_stress",
    "normalized_error",
    "probe_errors",
    "q4_extrapolation_matrix",
    "q4_gauss_stresses",
    "q4_shape",
    "q4_shape_derivative",
    "q4_strain_matrix",
    "recover_nodal_stress",
    "recovery_error",
    "von_mises",
]

GAUSS = 1.0 / math.sqrt(3.0)

# 1-D linear extrapolation from the two Gauss points (orders -g, +g) to the nodes (-1, +1).
_A = 0.5 + 0.5 / GAUSS
_B = 0.5 - 0.5 / GAUSS
# 2x2 ndarray: the Torch path indexes a row and then an entry, the NumPy path
# indexes ``[row, column]``; both read the same numbers.
EXTRAPOLATION_1D = np.array(((_A, _B), (_B, _A)), dtype=np.float64)

# Q4 node order used throughout TensorFEM and in ``cook_stress_reference``.
NODE_ORDER = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

# 2 x 2 Gauss slot order (eta fastest), matching ``q4_gauss_stresses``.
GAUSS_ORDER = ((-GAUSS, -GAUSS), (GAUSS, -GAUSS), (GAUSS, GAUSS), (-GAUSS, GAUSS))


def q4_extrapolation_matrix(dtype=torch.float64, device=None) -> torch.Tensor:
    """``(4 nodes, 4 Gauss slots)`` Barlow extrapolation matrix for a Q4 element."""
    matrix = torch.zeros((4, 4), dtype=dtype, device=device)
    for node, (nx, ny) in enumerate(NODE_ORDER):
        for slot, (gx, gy) in enumerate(GAUSS_ORDER):
            row = EXTRAPOLATION_1D[0 if nx < 0 else 1]
            column = EXTRAPOLATION_1D[0 if ny < 0 else 1]
            matrix[node, slot] = row[0 if gx < 0 else 1] * column[0 if gy < 0 else 1]
    return matrix


def q4_shape_derivative(xi: float, eta: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Parametric derivatives ``dN/dxi``, ``dN/deta`` in the Q4 node order."""
    dxi = (
        -0.25 * (1 - eta),
        0.25 * (1 - eta),
        0.25 * (1 + eta),
        -0.25 * (1 + eta),
    )
    deta = (
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
        0.25 * (1 + xi),
        0.25 * (1 - xi),
    )
    return dxi, deta


def q4_shape(xi: float, eta: float, dtype=None, device=None):
    """Bilinear shape functions in the Q4 node order.

    Union of two call conventions: with ``dtype``/``device`` (or a Torch ``dtype`` passed
    positionally) a ``torch.Tensor`` is returned -- the production Torch path; with no
    dtype/device a ``numpy.ndarray`` is returned -- the audit path used by
    ``cook_probe_protocol`` and ``cook_recovery_qualification``.
    """
    values = (
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    )
    if dtype is None and device is None:
        return np.array(values, dtype=np.float64)
    return torch.tensor(
        values, dtype=torch.float64 if dtype is None else dtype, device=device
    )


def q4_strain_matrix(xi: float, eta: float, coordinates: torch.Tensor) -> torch.Tensor:
    """``(3, 8)`` plane-stress strain-displacement matrix at parent point ``(xi, eta)``.

    ``coordinates`` is ``(ne, 4, 2)`` or ``(4, 2)``; a single-element input returns a
    ``(3, 8)`` matrix, a batch input returns ``(ne, 3, 8)``.
    """
    single = coordinates.ndim == 2
    xy = coordinates.unsqueeze(0) if single else coordinates
    dxi, deta = q4_shape_derivative(xi, eta)
    dshape = torch.tensor((dxi, deta), dtype=xy.dtype, device=xy.device)  # (2 parent, 4 nodes)
    # J[a, b] = d x_b / d xi_a
    jacobian = torch.einsum("ai,eib->eab", dshape, xy)
    inverse = torch.linalg.inv(jacobian)  # inverse[a, b] = d xi_a / d x_b
    # Chain rule reads dN/dx = derivative @ inv(J).T, i.e. dN_i/dx_b = sum_a dN_i/dxi_a inv(J)[b, a]:
    # applying inv(J) without the transpose is only correct for a symmetric Jacobian (rectangles),
    # not for the tapered Cook map.  Same convention as ``cook_stress_reference._strain_matrix``.
    gradients = torch.einsum("ai,eba->eib", dshape, inverse)  # (ne, 4 nodes, 2 physical)
    dNdx = gradients[..., 0]
    dNdy = gradients[..., 1]
    shape = (len(xy), 3, 8)
    matrix = torch.zeros(shape, dtype=xy.dtype, device=xy.device)
    matrix[:, 0, 0::2] = dNdx
    matrix[:, 1, 1::2] = dNdy
    matrix[:, 2, 0::2] = dNdy
    matrix[:, 2, 1::2] = dNdx
    return matrix[0] if single else matrix


def plane_stress_matrix(young: float, poisson: float, dtype=torch.float64, device=None) -> torch.Tensor:
    """Isotropic plane-stress elasticity matrix ``(3, 3)``."""
    factor = young / (1.0 - poisson * poisson)
    return torch.tensor(
        (
            (factor, factor * poisson, 0.0),
            (factor * poisson, factor, 0.0),
            (0.0, 0.0, factor * 0.5 * (1.0 - poisson)),
        ),
        dtype=dtype,
        device=device,
    )


def q4_gauss_stresses(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    displacement: torch.Tensor,
    young: float,
    poisson: float,
) -> torch.Tensor:
    """Stress at the ``2 x 2`` Gauss points, shaped ``(ne, 4 Gauss slots, 3)``.

    Slot order is :data:`GAUSS_ORDER` (eta fastest).  ``displacement`` may be given as a
    flat ``(2*nnodes,)`` vector or as ``(nnodes, 2)``.
    """
    coordinates = nodes[elements]  # (ne, 4, 2)
    u = displacement.reshape(-1, 2)
    matrix = plane_stress_matrix(young, poisson, dtype=nodes.dtype, device=nodes.device)
    element_u = u[elements].reshape(len(elements), 8)
    stresses = []
    for xi, eta in GAUSS_ORDER:
        strain_matrix = q4_strain_matrix(xi, eta, coordinates)  # (ne, 3, 8)
        strain = torch.einsum("eij,ej->ei", strain_matrix, element_u)
        stresses.append(strain @ matrix.T)
    return torch.stack(stresses, dim=1)


def element_areas(nodes, elements):
    """Unsigned area of each Q4 element.

    Type driven: Torch inputs use the two-triangle cross-product sum ``(0,1,2)+(0,2,3)``,
    NumPy inputs use the shoelace rule on the node loop.  Both are exact for a convex
    bilinear quad (the only case either implementation is used with) and each preserves
    the numbers of the implementation it comes from.
    """
    if not isinstance(nodes, torch.Tensor):
        corners = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
        x, y = corners[:, :, 0], corners[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        )
    coordinates = nodes[elements]
    total = torch.zeros(len(elements), dtype=nodes.dtype, device=nodes.device)
    for a, b, c in ((0, 1, 2), (0, 2, 3)):
        ab = coordinates[:, b] - coordinates[:, a]
        ac = coordinates[:, c] - coordinates[:, a]
        total = total + 0.5 * torch.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    return total


def recover_nodal_stress(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    gauss_stress: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Barlow extrapolation to nodes plus weighted nodal averaging, ``(nnodes, 3)``.

    ``weights`` defaults to the element area, which is the standard choice for a
    structured patch: it makes the averaged nodal value exact for a constant stress
    field and keeps the recovery conservative at boundaries.
    """
    extrapolation = q4_extrapolation_matrix(dtype=nodes.dtype, device=nodes.device)
    coordinates = nodes[elements]
    nodal = torch.zeros((len(nodes), 3), dtype=nodes.dtype, device=nodes.device)
    weight = torch.zeros(len(nodes), dtype=nodes.dtype, device=nodes.device)
    if weights is None:
        weights = element_areas(nodes, elements)
    extrapolated = torch.einsum("ng,egk->enk", extrapolation, gauss_stress)  # (ne, 4 nodes, 3)
    for local in range(4):
        index = elements[:, local]
        nodal.index_add_(0, index, weights[:, None] * extrapolated[:, local, :])
        weight.index_add_(0, index, weights)
    return nodal / weight.clamp_min(1e-300)[:, None]


def locate_point(nodes: torch.Tensor, elements: torch.Tensor, point) -> list[dict]:
    """All elements whose closed parent square contains ``point`` (bracketed coordinates).

    Bounding-box filtering keeps the Newton inversion to the few candidate elements, so a
    probe query costs O(ne) vectorised comparisons plus a handful of scalar solves.
    """
    coordinates = nodes[elements]
    target = torch.as_tensor(point, dtype=nodes.dtype, device=nodes.device)
    tolerance = 1e-9 * (1.0 + float(torch.max(torch.abs(coordinates))))
    lower = coordinates.min(dim=1).values - tolerance
    upper = coordinates.max(dim=1).values + tolerance
    candidates = torch.nonzero(
        torch.all((target >= lower) & (target <= upper), dim=1)
    ).reshape(-1).tolist()
    hits = []
    for element in candidates:
        xi, eta = _local_coordinates(target, coordinates[element])
        if abs(xi) <= 1.0 + 1e-9 and abs(eta) <= 1.0 + 1e-9:
            hits.append(
                {
                    "element": int(element),
                    "local": (float(xi), float(eta)),
                    "on_boundary": bool(
                        abs(abs(xi) - 1.0) < 1e-9 or abs(abs(eta) - 1.0) < 1e-9
                    ),
                }
            )
    return hits


def _local_coordinates(point: torch.Tensor, coordinates: torch.Tensor, tolerance: float = 1e-14):
    """Newton inversion of the bilinear element map (same recursion as the audit module)."""
    xi = torch.zeros((), dtype=coordinates.dtype, device=coordinates.device)
    eta = torch.zeros((), dtype=coordinates.dtype, device=coordinates.device)
    for _ in range(60):
        shape = q4_shape(float(xi), float(eta), dtype=coordinates.dtype, device=coordinates.device)
        residual = point - coordinates.T @ shape
        if float(torch.linalg.vector_norm(residual)) < tolerance:
            break
        dxi, deta = q4_shape_derivative(float(xi), float(eta))
        derivative = torch.tensor(
            (dxi, deta), dtype=coordinates.dtype, device=coordinates.device
        )
        jacobian = coordinates.T @ derivative.T
        delta = torch.linalg.solve(jacobian, residual)
        xi = xi + delta[0]
        eta = eta + delta[1]
    return xi, eta


def von_mises(stress) -> float:
    """Plane-stress von Mises equivalent stress."""
    sxx, syy, sxy = (float(value) for value in stress[:3])
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def interpolate_nodal_field(
    nodes: torch.Tensor, elements: torch.Tensor, field: torch.Tensor, point
) -> dict:
    """Bilinear interpolation of a recovered nodal field at a physical ``point``.

    For a *recovered* nodal field the value on an element interface is single valued by
    construction; ``spread`` reports the exact size of any residual ambiguity (it must be
    at machine-zero, and that is asserted by the test suite).
    """
    hits = locate_point(nodes, elements, point)
    if not hits:
        raise ValueError(f"point {point} is not contained in any element")
    values = []
    for hit in hits:
        shape = q4_shape(float(hit["local"][0]), float(hit["local"][1]), dtype=field.dtype, device=field.device)
        values.append(shape @ field[elements[hit["element"]]])
    stress = values[0]
    spread = max(
        (float(torch.max(torch.abs(value - stress))) for value in values[1:]), default=0.0
    )
    return {
        "point": [float(component) for component in torch.as_tensor(point)],
        "stress": [float(component) for component in stress],
        "von_mises": von_mises(stress),
        "containing_elements": len(hits),
        "on_element_boundary": any(hit["on_boundary"] for hit in hits),
        "spread": spread,
    }


def normalized_error(value: float, reference: float, scale: float) -> float:
    """Relative error with the 5 %-of-scale floor used by ``cook_stress_reference``."""
    return abs(value - reference) / max(abs(reference), 0.05 * scale)


def probe_errors(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    field: torch.Tensor,
    points: list[dict],
    scale: float,
) -> dict:
    """Recovered-field von Mises error at every reference point, plus the worst site.

    ``points`` entries must carry ``point`` (physical coordinates) and
    ``von_mises_reference``.
    """
    sites = []
    for entry in points:
        sample = interpolate_nodal_field(nodes, elements, field, entry["point"])
        sites.append(
            {
                "point": sample["point"],
                "von_mises": sample["von_mises"],
                "von_mises_reference": float(entry["von_mises_reference"]),
                "error": normalized_error(
                    sample["von_mises"], float(entry["von_mises_reference"]), scale
                ),
                "on_element_boundary": sample["on_element_boundary"],
                "spread": sample["spread"],
            }
        )
    return {
        "scale": float(scale),
        "sites": sites,
        "worst_site_error": max(site["error"] for site in sites),
        "worst_boundary_spread": max(site["spread"] for site in sites),
    }


# ---------------------------------------------------------------------------
# NumPy audit API (kept verbatim from the m3402 side of the PR #79 merge).
# ---------------------------------------------------------------------------

def _element_stress(connectivity, coordinates, displacement, xi, eta) -> np.ndarray:
    matrix, _ = _reference_q4_strain_matrix(xi, eta, coordinates[connectivity])
    return _reference_plane_stress_matrix() @ matrix @ np.asarray(displacement, dtype=np.float64).reshape(
        -1, 2
    )[connectivity].ravel()


def extrapolation_matrix() -> np.ndarray:
    """4x4 matrix mapping the four Gauss-point stresses to the four nodal values."""
    matrix = np.zeros((4, 4), dtype=np.float64)
    for node, (nx, ny) in enumerate(NODE_ORDER):
        for slot, (px, py) in enumerate(GAUSS_ORDER):
            matrix[node, slot] = (
                EXTRAPOLATION_1D[0 if nx < 0 else 1, 0 if px < 0 else 1]
                * EXTRAPOLATION_1D[0 if ny < 0 else 1, 0 if py < 0 else 1]
            )
    return matrix


def local_coordinates(
    point, coordinates, tolerance: float = 1e-12, iterations: int = 60
) -> tuple[float, float]:
    """Newton inversion of the bilinear element map; returns parent coordinates."""
    target = np.asarray(point, dtype=np.float64)
    corners = np.asarray(coordinates, dtype=np.float64)
    xi = eta = 0.0
    for _ in range(iterations):
        shape = q4_shape(xi, eta)
        residual = target - corners.T @ shape
        if float(np.linalg.norm(residual)) < tolerance:
            break
        derivative = 0.25 * np.array(
            (
                (-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)),
                (-(1 - xi), -(1 + xi), 1 + xi, 1 - xi),
            )
        )
        jacobian = corners.T @ derivative.T
        xi, eta = (xi, eta) + np.linalg.solve(jacobian, residual)
    return float(xi), float(eta)


def containing_elements(nodes, elements, point, tolerance: float = 1e-9) -> list[dict]:
    """All elements whose closed parent square contains ``point``.

    A boundary point yields two entries (edge) or four (corner); each entry is one
    admissible one-sided evaluation site of a discontinuous field.
    """
    coordinates = np.asarray(nodes, dtype=np.float64)
    connectivity = np.asarray(elements, dtype=np.int64)
    target = np.asarray(point, dtype=np.float64)
    corners = coordinates[connectivity]
    lower = corners.min(axis=1) - tolerance
    upper = corners.max(axis=1) + tolerance
    inside_box = np.all((target >= lower) & (target <= upper), axis=1)
    hits = []
    for index in np.flatnonzero(inside_box):
        xi, eta = local_coordinates(target, corners[index])
        if abs(xi) <= 1.0 + 1e-9 and abs(eta) <= 1.0 + 1e-9:
            hits.append(
                {
                    "element": int(index),
                    "local": [xi, eta],
                    "nodes": connectivity[index].tolist(),
                    "on_boundary": bool(
                        abs(abs(xi) - 1.0) < 1e-9 or abs(abs(eta) - 1.0) < 1e-9
                    ),
                }
            )
    return hits


def gauss_point_stresses(nodes, elements, displacement) -> np.ndarray:
    """Stress at the four 2x2 Gauss points of every element, shape ``(ne, 4, 3)``."""
    coordinates = np.asarray(nodes, dtype=np.float64)
    connectivity = np.asarray(elements, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    out = np.empty((len(connectivity), 4, 3), dtype=np.float64)
    for slot, (xi, eta) in enumerate(GAUSS_ORDER):
        for element, conn in enumerate(connectivity):
            out[element, slot] = _element_stress(conn, coordinates, displacement, xi, eta)
    return out


def nodal_recovered_stress(nodes, elements, displacement) -> np.ndarray:
    """Gauss-point extrapolation plus area-weighted nodal averaging.

    Returns an ``(nnodes, 3)`` single-valued stress field: stresses are evaluated at
    the superconvergent 2x2 Gauss points, extrapolated linearly to the element nodes
    and averaged over the incident elements with element area as weight.
    """
    coordinates = np.asarray(nodes, dtype=np.float64)
    connectivity = np.asarray(elements, dtype=np.int64)
    gauss = gauss_point_stresses(nodes, elements, displacement)
    matrix = extrapolation_matrix()
    nodal = np.zeros((len(coordinates), 3), dtype=np.float64)
    weight = np.zeros(len(coordinates), dtype=np.float64)
    areas = element_areas(nodes, elements)
    for element, conn in enumerate(connectivity):
        extrapolated = matrix @ gauss[element]  # (4 nodes, 3)
        for local, node in enumerate(conn):
            nodal[node] += areas[element] * extrapolated[local]
            weight[node] += areas[element]
    nonzero = weight > 0.0
    if not np.all(nonzero):
        raise ValueError("recovery left nodes without incident elements")
    return nodal / weight[:, None]


def interpolate_nodal_stress(nodes, elements, field, point) -> dict:
    """Bilinear interpolation of a nodal field at ``point`` (well posed everywhere).

    A nodal field is single valued, so on an element boundary the result does not
    depend on the element chosen; the first containing element is used and every
    alternative one-sided interpolation is reported as a spread for auditability.
    """
    hits = containing_elements(nodes, elements, point)
    if not hits:
        raise ValueError(f"point {np.asarray(point, dtype=np.float64).tolist()} is outside the mesh")
    values = np.asarray(field, dtype=np.float64)
    hit = hits[0]
    shape = q4_shape(hit["local"][0], hit["local"][1])
    stress = shape @ values[hit["nodes"]]
    alternates = []
    for other in hits[1:]:
        other_shape = q4_shape(other["local"][0], other["local"][1])
        alternates.append(abs(von_mises(other_shape @ values[other["nodes"]]) - von_mises(stress)))
    return {
        "point": [float(p) for p in np.asarray(point, dtype=np.float64)],
        "element": hit["element"],
        "local": hit["local"],
        "stress": [float(s) for s in stress],
        "von_mises": von_mises(stress),
        "candidate_count": len(hits),
        "alternate_element_spread": float(max(alternates)) if alternates else 0.0,
    }


def recovery_error(value: float, reference: float, scale: float) -> float:
    """Relative error with the reference-scaled 5 % floor used by the Cook audit.

    ``|value - reference| / max(|reference|, 0.05 * scale)``: small reference values
    cannot inflate the error, which keeps a near-zero component from dominating a
    global comparison.
    """
    return abs(float(value) - float(reference)) / max(
        abs(float(reference)), 0.05 * float(scale)
    )
