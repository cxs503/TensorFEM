"""Probe-protocol audit for the Cook membrane Q4 stress field.

The Cook membrane stress qualification (``scripts/audit_cook_membrane_stress.py``)
compares the production Q4 stress at five material points against an independent Q8
Richardson-extrapolated reference and reports a residual von Mises error above the
3 % gate (5.02 % at n=64).  This module asks whether that residual is a property of
the discretisation or of the measurement protocol.

Two protocol defects are quantified here:

1. **Ill-posed probe sites.**  Three of the five canonical sample points sit on the
   parametric mid-lines ``xi=0.5`` / ``eta=0.25, 0.5, 0.75``.  On the structured
   ``n x n`` Q4 grid these points land exactly on element corner nodes whenever ``n``
   is a multiple of four, where the Q4 strain field is discontinuous: the sampled
   value depends on which of the (up to four) incident elements the floor/ceil
   tie-break selects.  ``one_sided_values`` returns every admissible one-sided value
   so that the spread can be measured instead of being hidden behind one tie-break.

2. **Recovery underestimation.**  A constant-per-element stress is O(h) accurate;
   the standard remedy is Gauss-point (Barlow) extrapolation to nodes followed by
   nodal averaging and re-interpolation.  ``nodal_recovered_stress`` implements that
   recovery without touching the production assembly or solver.

Nothing here mutates production paths; the module is measurement-only and every
returned number is derived from the production displacement field passed in.
"""

from __future__ import annotations

import math

import numpy as np

from .cook_stress_reference import (
    inverse_map,
    map_point,
    plane_stress_matrix,
    q4_strain_matrix,
)

GAUSS = 1.0 / math.sqrt(3.0)
# 1-D linear extrapolation from the two Gauss points (order -g, +g) to the nodes (-1, +1).
_A = 0.5 + 0.5 / GAUSS  # 1.3660254
_B = 0.5 - 0.5 / GAUSS  # -0.3660254
EXTRAPOLATION_1D = np.array([[_A, _B], [_B, _A]], dtype=np.float64)

# 2-D tensor-product extrapolation ordering matches the Q4 node order below.
_NODE_ORDER = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def q4_shape(xi: float, eta: float) -> np.ndarray:
    """Bilinear shape functions for the Q4 node order used throughout the audit."""
    return 0.25 * np.array(
        (
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ),
        dtype=np.float64,
    )


def _local_coordinates(point: np.ndarray, coordinates: np.ndarray, tolerance: float = 1e-12):
    """Newton inversion of the element bilinear map; returns parent coords and Jacobian."""
    xi = eta = 0.0
    for _ in range(60):
        shape = q4_shape(xi, eta)
        residual = point - coordinates.T @ shape
        if float(np.linalg.norm(residual)) < tolerance:
            break
        derivative = 0.25 * np.array(
            (
                (-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)),
                (-(1 - xi), -(1 + xi), 1 + xi, 1 - xi),
            )
        )
        jacobian = coordinates.T @ derivative.T
        xi, eta = (xi, eta) + np.linalg.solve(jacobian, residual)
    return float(xi), float(eta)


def containing_elements(nodes, elements, point, tolerance: float = 1e-9) -> list[dict]:
    """All elements that contain ``point`` in their closed parent square.

    A point that lands on an element boundary yields two entries (edge) or four
    entries (corner).  Each entry is one admissible one-sided evaluation site, which
    is exactly what a discontinuous Q4 stress field requires to be reported honestly.
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
        xi, eta = _local_coordinates(target, corners[index])
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


def _stress_from_displacement(connectivity, coordinates, displacement, xi, eta) -> np.ndarray:
    matrix, _ = q4_strain_matrix(xi, eta, coordinates[connectivity])
    return plane_stress_matrix() @ matrix @ np.asarray(displacement).reshape(-1, 2)[
        connectivity
    ].ravel()


def von_mises(stress) -> float:
    sxx, syy, sxy = float(stress[0]), float(stress[1]), float(stress[2])
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def one_sided_values(nodes, elements, displacement, point) -> dict:
    """Every one-sided Q4 stress value admissible at ``point``.

    ``spread_relative`` is the max pairwise von Mises difference normalised by the
    mean: it is the *irreducible* ambiguity of a pointwise probe at a discontinuous
    site and must be small before any error figure can be interpreted as accuracy.
    """
    values = []
    for hit in containing_elements(nodes, elements, point):
        stress = _stress_from_displacement(
            hit["nodes"],
            np.asarray(nodes, dtype=np.float64),
            displacement,
            hit["local"][0],
            hit["local"][1],
        )
        values.append(
            {
                "element": hit["element"],
                "local": hit["local"],
                "stress": [float(s) for s in stress],
                "von_mises": von_mises(stress),
            }
        )
    if not values:
        raise ValueError(f"point {point} is not contained in any element")
    magnitudes = [entry["von_mises"] for entry in values]
    mean = float(np.mean(magnitudes))
    spread = (max(magnitudes) - min(magnitudes)) / abs(mean) if len(magnitudes) > 1 else 0.0
    return {
        "point": [float(p) for p in np.asarray(point, dtype=np.float64)],
        "candidates": values,
        "candidate_count": len(values),
        "von_mises_min": min(magnitudes),
        "von_mises_max": max(magnitudes),
        "spread_relative": float(spread),
        "well_posed": bool(len(magnitudes) == 1 or spread < 1e-9),
    }


def element_areas(nodes, elements) -> np.ndarray:
    """Areas of the Q4 quadrilaterals (shoelace on the counter-clockwise node loop)."""
    corners = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    x, y = corners[:, :, 0], corners[:, :, 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    )


def gauss_point_stresses(nodes, elements, displacement) -> np.ndarray:
    """Stress at the four 2x2 Gauss points of every element, shape (ne, 4, 3)."""
    coordinates = np.asarray(nodes, dtype=np.float64)
    connectivity = np.asarray(elements, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    matrix = plane_stress_matrix()
    out = np.empty((len(connectivity), 4, 3), dtype=np.float64)
    for slot, (xi, eta) in enumerate(
        ((-GAUSS, -GAUSS), (GAUSS, -GAUSS), (GAUSS, GAUSS), (-GAUSS, GAUSS))
    ):
        for element, conn in enumerate(connectivity):
            out[element, slot] = _stress_from_displacement(
                conn, coordinates, displacement, xi, eta
            )
    return out


def nodal_recovered_stress(nodes, elements, displacement) -> np.ndarray:
    """Gauss-point extrapolation to nodes plus area-weighted nodal averaging.

    Returns an (nnodes, 3) smoothed stress field.  The recovery is the textbook
    Barlow/Zienkiewicz-Zhu construction for bilinear quadrilaterals: stresses are
    superconvergent at the 2x2 Gauss points, linearly extrapolated to the element
    nodes, and averaged over all incident elements weighted by element area.
    """
    coordinates = np.asarray(nodes, dtype=np.float64)
    connectivity = np.asarray(elements, dtype=np.int64)
    gauss = gauss_point_stresses(nodes, elements, displacement)
    # (4 gauss) -> (4 nodes) tensor-product linear extrapolation.
    # gauss slot order above is (--, +-, ++, -+), i.e. eta fastest.
    extrapolation_matrix = np.zeros((4, 4))
    pairs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    for node, (nx, ny) in enumerate(_NODE_ORDER):
        for gauss_slot, (px, py) in enumerate(pairs):
            extrapolation_matrix[node, gauss_slot] = EXTRAPOLATION_1D[
                0 if nx < 0 else 1, 0 if px < 0 else 1
            ] * EXTRAPOLATION_1D[0 if ny < 0 else 1, 0 if py < 0 else 1]
    nodal = np.zeros((len(coordinates), 3), dtype=np.float64)
    weight = np.zeros(len(coordinates), dtype=np.float64)
    areas = element_areas(nodes, elements)
    for element, conn in enumerate(connectivity):
        extrapolated = extrapolation_matrix @ gauss[element]  # (4 nodes, 3)
        for local, node in enumerate(conn):
            nodal[node] += areas[element] * extrapolated[local]
            weight[node] += areas[element]
    nonzero = weight > 0.0
    nodal[nonzero] /= weight[nonzero, None]
    return nodal


def interpolate_nodal_stress(nodes, elements, field, point) -> dict:
    """Bilinear interpolation of a nodal field at ``point`` (well posed everywhere).

    On an element boundary the nodal field is by construction single valued, so the
    interpolated result does not depend on the element chosen; the first containing
    element is therefore used and the choice is recorded.
    """
    hits = containing_elements(nodes, elements, point)
    if not hits:
        raise ValueError(f"point {point} is not contained in any element")
    hit = hits[0]
    shape = q4_shape(hit["local"][0], hit["local"][1])
    values = np.asarray(field, dtype=np.float64)[hit["nodes"]]
    stress = shape @ values
    return {
        "point": [float(p) for p in np.asarray(point, dtype=np.float64)],
        "element": hit["element"],
        "local": hit["local"],
        "stress": [float(s) for s in stress],
        "von_mises": von_mises(stress),
        "alternate_element_spread": (
            float(
                max(
                    abs(von_mises(shape_alt @ np.asarray(field, dtype=np.float64)[alt["nodes"]])
                        - von_mises(stress))
                    for alt, shape_alt in [
                        (h, q4_shape(h["local"][0], h["local"][1])) for h in hits[1:]
                    ]
                )
            )
            if len(hits) > 1
            else 0.0
        ),
    }


def normalized_error(value: float, reference: float, scale: float) -> float:
    """Same floor convention as ``cook_stress_reference.normalized_error`` (5 % of scale)."""
    return abs(value - reference) / max(abs(reference), 0.05 * scale)


def canonical_points(reference: dict) -> list[dict]:
    """The five reference sample points with their parametric and physical coordinates."""
    return [
        {
            "xi": point["xi"],
            "eta": point["eta"],
            "point": point["point"],
            "von_mises_reference": point["von_mises"]["extrapolated"],
            "component_reference": {
                key: point[key]["extrapolated"] for key in ("sxx", "syy", "sxy")
            },
        }
        for point in reference["points"]
    ]


def probe_grid_contact(divisions: int, xi: float, eta: float, tolerance: float = 1e-9) -> dict:
    """Whether a parametric probe lands exactly on a structured ``divisions`` grid node.

    A Q4 grid node is hit when ``xi*divisions`` and ``eta*divisions`` are both
    integers.  ``xi*divisions`` alone puts the probe on an element edge.
    """
    scaled = (xi * divisions, eta * divisions)
    on_edge = tuple(
        bool(abs(value - round(value)) < tolerance) for value in scaled
    )
    return {
        "divisions": divisions,
        "scaled": [float(scaled[0]), float(scaled[1])],
        "on_element_edge": on_edge,
        "on_grid_node": bool(on_edge[0] and on_edge[1]),
    }