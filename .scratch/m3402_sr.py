"""Production stress recovery for bilinear (Q4) plane-stress/strain fields.

A constant-per-element stress field is only first-order accurate and is discontinuous
across element interfaces, so it cannot be sampled pointwise: at a shared node the
value depends on which incident element is selected.  The textbook remedy (Barlow
1976; Zienkiewicz & Zhu 1987) is to evaluate the stress at the 2x2 Gauss points --
where the field is superconvergent -- extrapolate linearly to the element nodes and
average the nodal contributions from all incident elements, weighted by element area.
The resulting *nodal* field is single valued and second-order accurate.

This module is the production implementation of that recovery.  It is the single
source of truth: the measurement-only audit in
:mod:`tensorfem.cook_probe_protocol` re-exports these functions instead of carrying
its own copy, so an audit number and a production number can never drift apart.

Scope: small-strain linear elasticity, 4-node bilinear quadrilaterals, plane stress
through :func:`tensorfem.cook_stress_reference.plane_stress_matrix`.  Nothing here
smooths the field globally; the recovery is local and bounded, it does not remove a
genuine stress concentration, and it does not repair a singular peak.
"""

from __future__ import annotations

import math

import numpy as np

from .cook_stress_reference import plane_stress_matrix, q4_strain_matrix

__all__ = [
    "GAUSS",
    "EXTRAPOLATION_1D",
    "extrapolation_matrix",
    "q4_shape",
    "local_coordinates",
    "containing_elements",
    "von_mises",
    "element_areas",
    "gauss_point_stresses",
    "nodal_recovered_stress",
    "interpolate_nodal_stress",
    "recovery_error",
]

GAUSS = 1.0 / math.sqrt(3.0)
#: 1-D linear extrapolation from the two Gauss points (parent -g, +g) to the nodes (-1, +1).
EXTRAPOLATION_1D = np.array(
    [[0.5 + 0.5 / GAUSS, 0.5 - 0.5 / GAUSS], [0.5 - 0.5 / GAUSS, 0.5 + 0.5 / GAUSS]],
    dtype=np.float64,
)

#: Q4 node order used throughout: counter-clockwise from the lower-left corner.
NODE_ORDER = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
#: Gauss slot order matching ``NODE_ORDER``: eta varies fastest.
GAUSS_ORDER = ((-GAUSS, -GAUSS), (GAUSS, -GAUSS), (GAUSS, GAUSS), (-GAUSS, GAUSS))


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


def q4_shape(xi: float, eta: float) -> np.ndarray:
    """Bilinear shape functions for the Q4 node order above."""
    return 0.25 * np.array(
        (
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ),
        dtype=np.float64,
    )


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


def von_mises(stress) -> float:
    """Plane-stress von Mises equivalent stress."""
    sxx, syy, sxy = float(stress[0]), float(stress[1]), float(stress[2])
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def element_areas(nodes, elements) -> np.ndarray:
    """Q4 areas via the shoelace rule on the counter-clockwise node loop."""
    corners = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    x, y = corners[:, :, 0], corners[:, :, 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    )


def _element_stress(connectivity, coordinates, displacement, xi, eta) -> np.ndarray:
    matrix, _ = q4_strain_matrix(xi, eta, coordinates[connectivity])
    return plane_stress_matrix() @ matrix @ np.asarray(displacement, dtype=np.float64).reshape(
        -1, 2
    )[connectivity].ravel()


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