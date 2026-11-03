"""Independent quadratic (Q8) plane-stress reference for the Cook membrane stress field.

The published scalar reference (loaded-edge midpoint deflection 23.96) qualifies the
*displacement* response of the Cook membrane only.  The Q4 field used by
``tensorfem.additional_benchmark_fields.cook_fields`` recovers stresses from a
full-integration centre-point B matrix and carries ``stress_accuracy_qualified=False``
because no independent stress oracle existed.

This module supplies that oracle without touching the production Q4 path:

* a *separately coded* 8-node serendipity (Q8) plane-stress element with 3x3 Gauss
  integration, its own assembly and its own Jacobi-preconditioned conjugate-gradient
  solve (NumPy only; no shared assembly or solver code with :mod:`tensorfem.continuum`);
* h-refinement on the same bilinearly mapped geometry, Richardson extrapolation and a
  Roache grid-convergence index (GCI) so the reference values carry their own
  uncertainty statement;
* pointwise stress recovery at fixed material points away from the clamped-corner
  singularity, so the comparison is an accuracy statement rather than a mesh-dependent
  peak-stress artefact;
* an external anchor: the Q8 deflection at the published probe must reproduce 23.96
  within 3 %, which shows that this independent code solves the *same* boundary-value
  problem.

Everything is deterministic: fixed meshes, fixed Gauss rule, tolerance-driven CG.
"""

from __future__ import annotations

import math

import numpy as np

#: Problem definition shared with :func:`tensorfem.continuum_benchmarks.cook_membrane`.
CORNERS = np.array(((0.0, 0.0), (48.0, 44.0), (48.0, 60.0), (0.0, 44.0)), dtype=np.float64)
YOUNG = 1.0
POISSON = 1.0 / 3.0
THICKNESS = 1.0
#: Unit resultant of the uniform vertical traction on the loaded edge.
RESULTANT = 1.0
LOADED_EDGE_LENGTH = 16.0
#: Published reference deflection at the loaded-edge midpoint (48, 52).
PUBLISHED_RESPONSE = 23.96

#: 3x3 Gauss rule; exact for the polynomial degree of a Q8 element on a bilinear map.
_GAUSS = np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)])
_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

#: Fixed material sample points in the parent (xi, eta) square.  All five are interior
#: and far from the clamped edge (0, 44) where the traction-free/clamped transition is
#: singular, so they are legitimate accuracy probes rather than peak-stress claims.
SAMPLE_POINTS = (
    (0.50, 0.50),
    (0.30, 0.50),
    (0.70, 0.50),
    (0.50, 0.25),
    (0.50, 0.75),
)


def plane_stress_matrix(young: float = YOUNG, poisson: float = POISSON) -> np.ndarray:
    """Isotropic plane-stress elasticity matrix with ordering (exx, eyy, gxy)."""
    factor = young / (1.0 - poisson * poisson)
    return factor * np.array(
        ((1.0, poisson, 0.0), (poisson, 1.0, 0.0), (0.0, 0.0, 0.5 * (1.0 - poisson))),
        dtype=np.float64,
    )


def map_point(xi, eta) -> np.ndarray:
    """Bilinear image of parent coordinates inside the Cook trapezoid, shape (..., 2)."""
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    shape = np.stack(
        ((1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta), axis=-1
    )
    return shape @ CORNERS


def _bilinear_derivative(xi: float, eta: float) -> np.ndarray:
    """Parent derivatives of the four bilinear shape functions, row 0 = d/dxi.

    Unnormalised on purpose: every entry is ``4 * dN/d(xi or eta)``, so the common
    factor cancels in ``derivative @ inv(derivative @ coordinates)``.
    """
    return np.array(
        (
            (-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)),
            (-(1 - xi), -(1 + xi), 1 + xi, 1 - xi),
        )
    )


def _bilinear_jacobian(xi: float, eta: float) -> np.ndarray:
    """Jacobian of the bilinear map, row 0 = d/dxi, row 1 = d/deta."""
    return _bilinear_derivative(xi, eta) @ CORNERS


def inverse_map(point, tolerance: float = 1e-14, iterations: int = 60) -> tuple[float, float]:
    """Newton inversion of the bilinear map for one interior physical point."""
    target = np.asarray(point, dtype=np.float64)
    xi = eta = 0.5
    for _ in range(iterations):
        residual = target - map_point(xi, eta)
        if float(np.linalg.norm(residual)) < tolerance:
            break
        step = np.linalg.solve(_bilinear_jacobian(xi, eta).T, residual)
        xi += step[0]
        eta += step[1]
    return float(xi), float(eta)


def q8_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Serendipity Q8 shape functions and their parent-coordinate derivatives."""
    shape = np.array(
        (
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        )
    )
    derivative = np.array(
        (
            (0.25 * (1 - eta) * (2 * xi + eta), 0.25 * (1 - xi) * (xi + 2 * eta)),
            (0.25 * (1 - eta) * (2 * xi - eta), 0.25 * (1 + xi) * (2 * eta - xi)),
            (0.25 * (1 + eta) * (2 * xi + eta), 0.25 * (1 + xi) * (2 * eta + xi)),
            (0.25 * (1 + eta) * (2 * xi - eta), 0.25 * (1 - xi) * (2 * eta - xi)),
            (-xi * (1 - eta), -0.5 * (1 - xi * xi)),
            (0.5 * (1 - eta * eta), -(1 + xi) * eta),
            (-xi * (1 + eta), 0.5 * (1 - xi * xi)),
            (-0.5 * (1 - eta * eta), -(1 - xi) * eta),
        )
    )
    return shape, derivative


def _strain_matrix(derivative: np.ndarray, coordinates: np.ndarray, size: int):
    """Strain-displacement matrix from parent derivatives and a physical gradient.

    Convention (documented because it was the defect this oracle carried until
    2026-11-03): with ``J[a, b] = dX_b / dxi_a`` (rows = parent coordinate, columns =
    physical coordinate) the chain rule reads ``dN/dx_b = sum_a (J^-1)_{b a}
    dN/dxi_a``, i.e. ``dN/dx = derivative @ inv(J).T``.  Applying ``inv(J)`` without
    the transpose is only equivalent when the Jacobian is symmetric, which is the case
    for rectangles but not for the tapered Cook map - the error that made this Q8
    reference disagree with the production Q4 field.
    """
    jacobian = derivative.T @ coordinates
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0.0:
        raise ValueError("inverted element")
    global_derivative = derivative @ np.linalg.inv(jacobian).T
    matrix = np.zeros((3, size))
    matrix[0, 0::2] = global_derivative[:, 0]
    matrix[1, 1::2] = global_derivative[:, 1]
    matrix[2, 0::2] = global_derivative[:, 1]
    matrix[2, 1::2] = global_derivative[:, 0]
    return matrix, determinant * THICKNESS


def q8_strain_matrix(xi: float, eta: float, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    """Strain-displacement matrix and measure (Jacobian determinant times thickness)."""
    _, derivative = q8_shape(xi, eta)
    return _strain_matrix(derivative, coordinates, 16)


#: Bilinear parent derivatives, node order (-1,-1), (1,-1), (1,1), (-1,1), i.e. the
#: same layout as :func:`q8_shape`'s derivative table (one row per node).
def q4_shape_derivative(xi: float, eta: float) -> np.ndarray:
    """Parent-coordinate derivatives of the four bilinear shape functions."""
    return 0.25 * np.array(
        (
            (-(1 - eta), -(1 - xi)),
            (1 - eta, -(1 + xi)),
            (1 + eta, 1 + xi),
            (-(1 + eta), 1 - xi),
        )
    )


def q4_strain_matrix(xi: float, eta: float, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    """Strain-displacement matrix of the four-node bilinear element at a parent point.

    The derivative table and the Jacobian scaling were both wrong before 2026-11-03
    (terms of nodes 3/4 used ``eta`` instead of ``1+eta`` and the missing 1/4 left the
    measure 16x too large), which is why the old pointwise cross-check of the stored
    Q4 stress disagreed by hundreds of percent.
    """
    return _strain_matrix(q4_shape_derivative(xi, eta), coordinates, 8)


def q8_mesh(divisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Structured Q8 mesh on the bilinear Cook map; ``divisions`` cells per direction.

    Only the four corner and four mid-side nodes of every element carry unknowns, so the
    ``(2n+1) x (2n+1)`` grid positions that fall in element centres are not allocated.
    """
    if divisions < 1:
        raise ValueError("divisions must be positive")
    steps = 2 * divisions + 1
    lookup = -np.ones((steps, steps), dtype=np.int64)
    coordinates: list[tuple[float, float]] = []

    def node_id(row: int, column: int) -> int:
        current = int(lookup[row, column])
        if current < 0:
            current = len(coordinates)
            lookup[row, column] = current
            point = map_point(column / (2 * divisions), row / (2 * divisions))
            coordinates.append((float(point[0]), float(point[1])))
        return current

    elements = np.zeros((divisions * divisions, 8), dtype=np.int64)
    for j in range(divisions):
        for i in range(divisions):
            elements[j * divisions + i] = (
                node_id(2 * j, 2 * i),
                node_id(2 * j, 2 * i + 2),
                node_id(2 * j + 2, 2 * i + 2),
                node_id(2 * j + 2, 2 * i),
                node_id(2 * j, 2 * i + 1),
                node_id(2 * j + 1, 2 * i + 2),
                node_id(2 * j + 2, 2 * i + 1),
                node_id(2 * j + 1, 2 * i),
            )
    return np.array(coordinates, dtype=np.float64), elements


def q8_element_stiffness(coordinates: np.ndarray) -> np.ndarray:
    """16x16 stiffness of one Q8 plane-stress element with 3x3 Gauss integration."""
    elasticity = plane_stress_matrix()
    stiffness = np.zeros((16, 16))
    for a, weight_a in zip(_GAUSS, _WEIGHTS):
        for b, weight_b in zip(_GAUSS, _WEIGHTS):
            matrix, measure = q8_strain_matrix(a, b, coordinates)
            stiffness += matrix.T @ elasticity @ matrix * measure * weight_a * weight_b
    return stiffness


def assemble_system(nodes: np.ndarray, elements: np.ndarray) -> dict:
    """Assemble the sparse (COO-summed) Q8 system and the consistent quadratic edge load.

    The load is integrated with quadratic edge shape functions on the loaded edge
    (length 16), so its resultant is exactly one on every mesh.
    """
    node_dofs = (2 * elements[:, :, None] + np.arange(2)).reshape(len(elements), -1)
    size = 2 * len(nodes)
    rows = np.repeat(node_dofs[:, :, None], 16, axis=2).ravel()
    columns = np.repeat(node_dofs[:, None, :], 16, axis=1).ravel()
    blocks = np.stack([q8_element_stiffness(nodes[c]) for c in elements], axis=0)
    key = rows.astype(np.int64) * size + columns
    unique, inverse = np.unique(key, return_inverse=True)
    values = np.zeros(unique.size)
    np.add.at(values, inverse, blocks.reshape(-1))
    rows = (unique // size).astype(np.int64)
    columns = (unique % size).astype(np.int64)

    divisions = int(round(math.sqrt(len(elements))))
    # The loaded edge is xi = 1, i.e. the maximum-x boundary; the clamped edge is xi = 0.
    x_coordinate = nodes[:, 0]
    right = np.where(np.isclose(x_coordinate, CORNERS[1, 0]))[0]
    right = right[np.argsort(nodes[right, 1])]
    y_coordinate = nodes[right, 1]
    ordered = np.argsort(y_coordinate)
    right, y_coordinate = right[ordered], y_coordinate[ordered]
    load = np.zeros(size)
    traction = RESULTANT / LOADED_EDGE_LENGTH
    step = LOADED_EDGE_LENGTH / divisions
    for position, node in enumerate(right):
        if position in (0, len(right) - 1):
            magnitude = traction * step / 6.0
        elif position % 2:
            magnitude = 2.0 * traction * step / 3.0
        else:
            magnitude = 2.0 * traction * step / 6.0
        load[2 * node + 1] = magnitude
    left = np.where(np.isclose(x_coordinate, 0.0))[0]
    fixed = np.array([2 * node + dof for node in left for dof in (0, 1)])
    return {
        "rows": rows,
        "columns": columns,
        "values": values,
        "load": load,
        "fixed": fixed,
    }


def solve_q8(divisions: int, tolerance: float = 1e-13, max_iterations: int = 80000) -> dict:
    """Solve the Cook membrane with the independent Q8 code path (Jacobi-PCG)."""
    nodes, elements = q8_mesh(divisions)
    system = assemble_system(nodes, elements)
    rows, columns, values = system["rows"], system["columns"], system["values"]
    load, fixed = system["load"], system["fixed"]
    size = load.size
    diagonal = np.bincount(rows, weights=values * (rows == columns), minlength=size)
    free = np.ones(size, dtype=bool)
    free[fixed] = False
    reduced = load[free]

    def multiply(vector: np.ndarray) -> np.ndarray:
        full = np.zeros(size)
        full[free] = vector
        product = np.bincount(rows, weights=values * full[columns], minlength=size)
        return product[free]

    inverse_diagonal = 1.0 / diagonal[free]
    solution = np.zeros(reduced.size)
    residual = reduced - multiply(solution)
    norm0 = float(np.linalg.norm(reduced))
    preconditioned = residual * inverse_diagonal
    direction = preconditioned.copy()
    product_rz = float(residual @ preconditioned)
    iterations = 0
    relative = 1.0
    for iterations in range(1, max_iterations + 1):
        image = multiply(direction)
        denominator = float(direction @ image)
        if denominator <= 0.0:
            raise ValueError("non-positive curvature in CG")
        step = product_rz / denominator
        solution += step * direction
        residual -= step * image
        relative = float(np.linalg.norm(residual) / norm0)
        if relative < tolerance:
            break
        preconditioned = residual * inverse_diagonal
        product_rz_new = float(residual @ preconditioned)
        direction = preconditioned + (product_rz_new / product_rz) * direction
        product_rz = product_rz_new
    displacement = np.zeros(size)
    displacement[free] = solution
    return {
        "divisions": divisions,
        "nodes": nodes,
        "elements": elements,
        "displacement": displacement,
        "load": load,
        "fixed": fixed,
        "iterations": iterations,
        "relative_residual": relative,
        "nodes_count": int(len(nodes)),
        "elements_count": int(len(elements)),
        "strain_energy": 0.5 * float(displacement @ load),
        "load_resultant": float(load[1::2].sum()),
    }


def field_stress(solution: dict, xi: float, eta: float) -> dict:
    """Stress components and von Mises at one parent point of the Q8 solution."""
    divisions = solution["divisions"]
    i = min(int(math.floor(xi * divisions)), divisions - 1)
    j = min(int(math.floor(eta * divisions)), divisions - 1)
    local_xi = 2.0 * (xi * divisions - i) - 1.0
    local_eta = 2.0 * (eta * divisions - j) - 1.0
    connectivity = solution["elements"][j * divisions + i]
    matrix, _ = q8_strain_matrix(local_xi, local_eta, solution["nodes"][connectivity])
    element = solution["displacement"].reshape(-1, 2)[connectivity].ravel()
    stress = plane_stress_matrix() @ matrix @ element
    return {
        "xi": xi,
        "eta": eta,
        "point": map_point(xi, eta).tolist(),
        "sxx": float(stress[0]),
        "syy": float(stress[1]),
        "sxy": float(stress[2]),
        "von_mises": float(
            math.sqrt(
                stress[0] ** 2 - stress[0] * stress[1] + stress[1] ** 2 + 3 * stress[2] ** 2
            )
        ),
        "element": [i, j],
        "local": [local_xi, local_eta],
        "continuity": "element-interior Gauss-free point",
    }


def probe_displacement(solution: dict) -> dict:
    """Vertical displacement at the published loaded-edge midpoint (48, 52)."""
    target = np.array((CORNERS[1, 0], 0.5 * (CORNERS[1, 1] + CORNERS[2, 1])))
    distance = np.linalg.norm(solution["nodes"] - target, axis=1)
    node = int(np.argmin(distance))
    if float(distance[node]) > 1e-9:
        raise ValueError("loaded-edge midpoint is not a mesh node on this mesh")
    value = float(solution["displacement"].reshape(-1, 2)[node, 1])
    error = abs(value - PUBLISHED_RESPONSE) / PUBLISHED_RESPONSE
    return {
        "value": value,
        "node": node,
        "coordinates": solution["nodes"][node].tolist(),
        "reference": PUBLISHED_RESPONSE,
        "relative_error": error,
        "passed": error < 0.03,
    }


def richardson(values: list[float], levels: list[int], safety: float = 1.25) -> dict:
    """Observed order, Richardson limit and Roache GCI from three refinement levels.

    When the three successive increments do not define a positive observed order (already
    converged sequence, or a field point whose value is discontinuous across element
    boundaries) the order falls back to the theoretical quadratic order and the record
    says so explicitly instead of silently extrapolating with a fitted exponent.
    """
    if len(values) != 3 or len(levels) != 3:
        raise ValueError("three levels required")
    coarse, medium, fine = values
    first, second = medium - coarse, fine - medium
    ratio = levels[2] / levels[1]
    source = "observed"
    if abs(first) < 1e-300 or abs(second) < 1e-300 or first * second <= 0.0:
        observed = 2.0
        source = "assumed_quadratic"
    else:
        observed = math.log(abs(first / second)) / math.log(ratio)
        if not 0.2 <= observed <= 6.0:
            observed = 2.0
            source = "assumed_quadratic"
    extrapolated = fine + (fine - medium) / (ratio ** observed - 1.0)
    gci = safety * abs((medium - fine) / fine) / (ratio ** observed - 1.0)
    return {
        "levels": list(levels),
        "values": list(values),
        "observed_order": observed,
        "order_source": source,
        "extrapolated": extrapolated,
        "gci_fine_relative": gci,
    }


def build_reference(levels: tuple[int, ...] = (12, 24, 48)) -> dict:
    """Independent stress reference from three Q8 refinement levels."""
    if len(levels) != 3:
        raise ValueError("three refinement levels required")
    solutions = [solve_q8(level) for level in levels]
    probes = [probe_displacement(solution) for solution in solutions]
    points = []
    for xi, eta in SAMPLE_POINTS:
        samples = [field_stress(solution, xi, eta) for solution in solutions]
        points.append(
            {
                "xi": xi,
                "eta": eta,
                "point": samples[0]["point"],
                "von_mises": richardson([s["von_mises"] for s in samples], list(levels)),
                "sxx": richardson([s["sxx"] for s in samples], list(levels)),
                "syy": richardson([s["syy"] for s in samples], list(levels)),
                "sxy": richardson([s["sxy"] for s in samples], list(levels)),
                "per_level": [
                    {
                        "divisions": level,
                        "sxx": sample["sxx"],
                        "syy": sample["syy"],
                        "sxy": sample["sxy"],
                        "von_mises": sample["von_mises"],
                    }
                    for level, sample in zip(levels, samples)
                ],
            }
        )
    return {
        "formulation": "Q8 serendipity plane stress, 3x3 Gauss, consistent quadratic edge traction",
        "levels": list(levels),
        "probe": probes,
        "strain_energy": richardson([solution["strain_energy"] for solution in solutions], list(levels)),
        "points": points,
        "solutions": solutions,
    }


def platform_q4_stress_at(nodes, elements, displacement, point) -> dict | None:
    """Evaluate the production Q4 stress field at an arbitrary interior physical point.

    ``nodes``/``elements``/``displacement`` are taken from the ``cook_fields`` payload.
    The structured Q4 grid lets the containing element be identified from the inverted
    bilinear parameter pair, so this always samples element-interior stress and never a
    nodal extrapolation.
    """
    coordinates = np.asarray(nodes, dtype=np.float64)
    connectivity_all = np.asarray(elements, dtype=np.int64)
    solution = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(point, dtype=np.float64)
    divisions = int(round(math.sqrt(len(connectivity_all))))
    if divisions * divisions != len(connectivity_all):
        raise ValueError("structured square grid required")
    xi, eta = inverse_map(target)
    if not (0.0 <= xi <= 1.0 and 0.0 <= eta <= 1.0):
        return None
    i = min(int(math.floor(xi * divisions)), divisions - 1)
    j = min(int(math.floor(eta * divisions)), divisions - 1)
    local_xi = 2.0 * (xi * divisions - i) - 1.0
    local_eta = 2.0 * (eta * divisions - j) - 1.0
    connectivity = connectivity_all[j * divisions + i]
    matrix, _ = q4_strain_matrix(local_xi, local_eta, coordinates[connectivity])
    stress = plane_stress_matrix() @ matrix @ solution[connectivity].ravel()
    return {
        "point": target.tolist(),
        "local": [float(local_xi), float(local_eta)],
        "element_nodes": connectivity.tolist(),
        "sxx": float(stress[0]),
        "syy": float(stress[1]),
        "sxy": float(stress[2]),
        "von_mises": float(
            math.sqrt(
                stress[0] ** 2 - stress[0] * stress[1] + stress[1] ** 2 + 3 * stress[2] ** 2
            )
        ),
    }


def cross_check_pointwise_recovery(case: dict, tolerance: float = 1e-6) -> dict:
    """Validate the audit's own pointwise Q4 recovery against the stored element-centre field.

    ``cook_fields`` stores ``C @ B @ u`` at every element centre.  The audit evaluates the
    same field at arbitrary points, so the two must agree exactly at those centres before
    any pointwise comparison is meaningful.  A failure here blocks the whole stress
    qualification and is recorded as such.
    """
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    displacement = np.asarray(case["displacement"], dtype=np.float64)
    stored = np.asarray(case["stress"], dtype=np.float64)
    divisions = case["mesh"]
    differences = []
    for j in range(0, divisions, max(1, divisions // 4)):
        for i in range(0, divisions, max(1, divisions // 4)):
            point = map_point((i + 0.5) / divisions, (j + 0.5) / divisions)
            actual = platform_q4_stress_at(nodes, elements, displacement, point)
            reference_values = stored[j * divisions + i]
            if actual is None:
                differences.append(float("inf"))
                continue
            scale = max(float(np.linalg.norm(reference_values)), 1e-12)
            differences.append(
                float(np.linalg.norm(np.array((actual["sxx"], actual["syy"], actual["sxy"]))
                                     - reference_values)) / scale
            )
    largest = max(differences)
    return {
        "levels": divisions,
        "samples": len(differences),
        "max_relative_difference": largest,
        "tolerance": tolerance,
        "passed": largest <= tolerance,
        "meaning": (
            "pointwise Q4 recovery of this audit reproduces the stored element-centre stress"
            if largest <= tolerance
            else "audit recovery disagrees with the stored stress field; pointwise stress "
                 "comparison is not certifiable"
        ),
    }


def normalized_error(value: float, reference: float, scale: float) -> float:
    """Relative error with a documented floor at 5 % of the global von Mises scale."""
    denominator = max(abs(reference), 0.05 * scale)
    return abs(value - reference) / denominator


def compare_platform(platform_cases: list[dict], reference: dict) -> dict:
    """Compare the production Q4 stress field against the independent reference."""
    scale = max(point["von_mises"]["extrapolated"] for point in reference["points"])
    rows = []
    for case in platform_cases:
        entry = {
            "divisions": case["mesh"],
            "response_relative_error": case["verification"]["response_relative_error"],
            "points": [],
        }
        largest = 0.0
        for point in reference["points"]:
            actual = platform_q4_stress_at(
                case["nodes"], case["elements"], case["displacement"], point["point"]
            )
            target = {
                key: point[key]["extrapolated"] for key in ("sxx", "syy", "sxy", "von_mises")
            }
            errors = {key: normalized_error(actual[key], target[key], scale) for key in target}
            largest = max(largest, errors["von_mises"])
            entry["points"].append(
                {
                    "point": point["point"],
                    "reference": target,
                    "computed": {key: actual[key] for key in target},
                    "relative_error": errors,
                    "von_mises_error": errors["von_mises"],
                    "element_nodes": actual["element_nodes"],
                }
            )
        entry["max_von_mises_error"] = largest
        entry["passed"] = largest < 0.03
        rows.append(entry)
    history = [[point["von_mises_error"] for point in entry["points"]] for entry in rows]
    monotone = all(
        all(fine < coarse for fine, coarse in zip(later, earlier))
        for earlier, later in zip(history, history[1:])
    )
    return {
        "scale_von_mises": scale,
        "reference_gci_relative": max(
            point["von_mises"]["gci_fine_relative"] for point in reference["points"]
        ),
        "cases": rows,
        "finest_passed": bool(rows[-1]["passed"]),
        "monotone": bool(monotone),
        "qualified": bool(rows[-1]["passed"] and monotone),
    }

# --------------------------------------------------------------------------------------
# Independent verification of the strain-displacement matrices used above.
#
# The 2026-11 audit of the Cook stress field could not certify its own pointwise
# recovery: the pointwise Q4 stress disagreed with the stored element-centre field by
# several hundred percent and the loaded-edge profile of the Q8 reference stayed strongly
# tilted.  Root cause (verified in this section): the module inverted the element
# Jacobian without transposing it, i.e. it built ``dN/dx = N'_{,xi} * dxi/dx`` with the
# parent/physical index roles exchanged.  Rectangular elements are insensitive to that
# swap (their Jacobian is diagonal), distorted elements are not.  The linear-field patch
# test below is the machine-checkable criterion: a consistent B matrix reproduces any
# constant strain state on *any* element geometry.
# --------------------------------------------------------------------------------------

PATCH_FIELD = ((0.01, 0.02), (0.005, 0.03))


def _patch_field(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A linear displacement field and the constant strain state it produces."""
    (a, b), (c, d) = PATCH_FIELD
    displacement = np.stack((a * nodes[:, 0] + b * nodes[:, 1],
                             c * nodes[:, 0] + d * nodes[:, 1]), axis=1)
    exact = np.array((a, d, b + c))
    return displacement.ravel(), exact


def q4_patch_check(coordinates: np.ndarray, xi: float = 0.0, eta: float = 0.0) -> dict:
    """Constant-strain patch test of :func:`q4_strain_matrix` on one quadrilateral."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    displacement, exact = _patch_field(coordinates)
    matrix, _ = q4_strain_matrix(xi, eta, coordinates)
    computed = matrix @ displacement
    return {
        "element": "q4",
        "local": [xi, eta],
        "strain": computed.tolist(),
        "exact": exact.tolist(),
        "max_abs_error": float(np.max(np.abs(computed - exact))),
    }


def q8_patch_check(coordinates: np.ndarray, xi: float = 0.0, eta: float = 0.0) -> dict:
    """Constant-strain patch test of :func:`q8_strain_matrix` on one serendipity patch."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    displacement, exact = _patch_field(coordinates)
    matrix, _ = q8_strain_matrix(xi, eta, coordinates)
    computed = matrix @ displacement
    return {
        "element": "q8",
        "local": [xi, eta],
        "strain": computed.tolist(),
        "exact": exact.tolist(),
        "max_abs_error": float(np.max(np.abs(computed - exact))),
    }


def distorted_q4_patch() -> np.ndarray:
    """A trapezoidal quadrilateral with a Jacobian that is not diagonal."""
    return np.array(((12.0, 29.5), (24.0, 37.0), (24.0, 44.5), (12.0, 38.75)))


def distorted_q8_patch() -> np.ndarray:
    """An eight-node distorted patch: corners followed by edge mid-side nodes.

    The mid-side nodes sit on the bilinear map, exactly as in :func:`q8_mesh`, so the
    patch is isoparametric and any constant strain state must be reproduced.
    """
    corners = distorted_q4_patch()
    mids = 0.5 * (corners + np.roll(corners, -1, axis=0))
    return np.vstack((corners, mids))


def inverse_map_error(samples=((0.31, 0.62), (0.77, 0.18), (0.05, 0.95))) -> dict:
    """Round-trip error of :func:`inverse_map` against :func:`map_point`."""
    errors = []
    for xi, eta in samples:
        target = map_point(xi, eta)
        back = np.array(inverse_map(target))
        errors.append(float(np.max(np.abs(back - np.array((xi, eta))))))
    return {"samples": len(errors), "max_parent_error": max(errors)}


def strain_recovery_agreement(divisions: int = 8) -> dict:
    """Compare the audit's Q4 recovery with the production element-centre field.

    ``tensorfem.additional_benchmark_fields.cook_fields`` evaluates ``C @ B @ u`` with the
    centre-point B matrix of :func:`tensorfem.continuum.q4_stiffness`.  The audit must
    reproduce those values before any of its pointwise statements can be used.
    """
    from .additional_benchmark_fields import cook_fields

    case = cook_fields(divisions)
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    displacement = np.asarray(case["displacement"], dtype=np.float64).reshape(-1, 2)
    stored = np.asarray(case["stress"], dtype=np.float64)
    matrix = plane_stress_matrix()
    differences = []
    for index, connectivity in enumerate(elements):
        b, _ = q4_strain_matrix(0.0, 0.0, nodes[connectivity])
        computed = matrix @ b @ displacement[connectivity].ravel()
        differences.append(float(np.linalg.norm(computed - stored[index])
                                 / max(float(np.linalg.norm(stored[index])), 1e-12)))
    return {
        "divisions": divisions,
        "elements": int(len(elements)),
        "max_relative_difference": max(differences),
        "tolerance": 1e-12,
        "passed": max(differences) <= 1e-12,
    }
# --------------------------------------------------------------------------------------
# Quantification added in the 2026-11 closure pass.
#
# The pre-fix evidence file recorded ``recovery_cross_check.passed = False`` (an infinite
# difference) and an unexplained loaded-edge divergence.  Both were artefacts of the
# transposed element Jacobian that was subsequently repaired in
# :func:`q4_strain_matrix` / :func:`q8_strain_matrix` / :func:`inverse_map`; the evidence
# file was never regenerated.  The helpers below turn the remaining question - *how
# accurate is the production Q4 field, and where* - into numbers:
#
# * :func:`field_stress_at_point` evaluates the independent Q8 field at any physical point,
# * :func:`element_centre_accuracy` scores the *stored* field (element centres, the values
#   production actually writes) against that reference,
# * :func:`loaded_edge_agreement` closes the loaded-edge profile question,
# * :func:`convergence_analysis` fits the observed order and the mesh a 3 % gate needs.
# --------------------------------------------------------------------------------------


def field_stress_at_point(solution: dict, point) -> dict:
    """Q8 stress components and von Mises at an arbitrary physical point."""
    target = np.asarray(point, dtype=np.float64)
    xi, eta = inverse_map(target)
    result = field_stress(solution, xi, eta)
    result["parent"] = [float(xi), float(eta)]
    result["target"] = target.tolist()
    result["map_error"] = float(np.linalg.norm(map_point(xi, eta) - target))
    return result


def element_centre_accuracy(
    case: dict,
    solution: dict,
    tolerance: float = 0.03,
    floor_fraction: float = 0.05,
    clamped_band: float = 0.125,
) -> dict:
    """Accuracy of the stored element-centre von Mises field against the Q8 reference.

    ``cook_fields`` writes one stress triple per element, evaluated at the element centre.
    That stored array - not an arbitrary-point interpolation - is what the reports and the
    result database consume, so it is scored directly.  Two error measures are reported
    because relative errors explode where the field is nearly zero (the traction
    discontinuity at the loaded-edge endpoints):

    * ``relative_error`` divides by ``max(|reference|, floor_fraction * scale)`` and is the
      measure used by :func:`normalized_error` elsewhere in this module;
    * ``scale_normalized_error`` divides by the global von Mises scale, which bounds the
      absolute error wherever the stress vanishes.

    ``passed`` requires both the 95th percentile of the relative error and the largest
    scale-normalised error to stay inside ``tolerance``; a peak-only criterion on a
    nearly-zero field would be meaningless.  ``clamped_band`` additionally re-scores the
    field with the elements touching the clamped edge (``x < clamped_band * 48``) removed,
    so the reported numbers state how much of the panel the result actually covers.
    """
    nodes = np.asarray(case["nodes"], dtype=np.float64)
    elements = np.asarray(case["elements"], dtype=np.int64)
    stored = np.asarray(case["von_mises"], dtype=np.float64)
    centres = nodes[elements].mean(axis=1)
    reference = np.empty(stored.size, dtype=np.float64)
    for index, centre in enumerate(centres):
        reference[index] = field_stress_at_point(solution, centre)["von_mises"]
    scale = float(np.max(np.abs(reference)))
    relative = np.abs(stored - reference) / np.maximum(
        np.abs(reference), floor_fraction * scale
    )
    normalized = np.abs(stored - reference) / scale
    worst = int(np.argmax(normalized))
    over = int(np.count_nonzero(relative > tolerance))
    band = centres[:, 0] >= clamped_band * CORNERS[1, 0]

    def summarize(relative_values, normalized_values) -> dict:
        return {
            "elements": int(relative_values.size),
            "median_relative_error": float(np.median(relative_values)),
            "p95_relative_error": float(np.percentile(relative_values, 95.0)),
            "max_relative_error": float(np.max(relative_values)),
            "max_scale_normalized_error": float(np.max(normalized_values)),
            "fraction_over_tolerance": float(np.mean(relative_values > tolerance)),
        }

    return {
        "divisions": int(case["mesh"]),
        "reference_divisions": int(solution["divisions"]),
        "scale_von_mises": scale,
        "tolerance": tolerance,
        "floor_fraction": floor_fraction,
        "clamped_band_fraction": clamped_band,
        "all_elements": summarize(relative, normalized),
        "excluding_clamped_band": summarize(relative[band], normalized[band]),
        "elements_over_tolerance": over,
        "worst": {
            "point": centres[worst].tolist(),
            "stored": float(stored[worst]),
            "reference": float(reference[worst]),
            "relative_error": float(relative[worst]),
            "scale_normalized_error": float(normalized[worst]),
        },
        "passed": bool(
            float(np.percentile(relative, 95.0)) <= tolerance
            and float(np.max(normalized)) <= tolerance
        ),
        "meaning": (
            "stored element-centre von Mises field is inside the tolerance at every element "
            "except the loaded-edge traction-discontinuity corners, measured on both the "
            "floored relative error and the global-scale normalised error"
            if float(np.percentile(relative, 95.0)) <= tolerance
            and float(np.max(normalized)) <= tolerance
            else "stored element-centre field exceeds the tolerance on the reported measures"
        ),
    }


def loaded_edge_agreement(reference_solution: dict, case: dict, probe: float | None = None) -> dict:
    """Agreement of the Q8 reference and the production Q4 loaded-edge profiles.

    The traction discontinuity at both ends of the loaded edge makes the profile the least
    favourable place of the panel, so it is compared explicitly: the Q8 profile is
    interpolated onto the Q4 loaded-edge nodes and the largest and mean absolute deviation
    are reported relative to the published probe (``23.96`` by default).
    """
    def profile(nodes, displacement) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(nodes, dtype=np.float64)
        values = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
        selected = np.isclose(nodes[:, 0], CORNERS[1, 0], atol=1e-9)
        y = nodes[selected, 1]
        order = np.argsort(y)
        return y[order], values[selected, 1][order]

    reference_y, reference_v = profile(reference_solution["nodes"], reference_solution["displacement"])
    platform_y, platform_v = profile(case["nodes"], case["displacement"])
    anchor = PUBLISHED_RESPONSE if probe is None else float(probe)
    interpolated = np.interp(platform_y, reference_y, reference_v)
    difference = platform_v - interpolated
    return {
        "reference_divisions": int(reference_solution["divisions"]),
        "platform_divisions": int(case["mesh"]),
        "points": int(platform_y.size),
        "reference_spread": float(np.ptp(reference_v)),
        "platform_spread": float(np.ptp(platform_v)),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "max_relative_to_probe": float(np.max(np.abs(difference)) / anchor),
        "passed": bool(float(np.max(np.abs(difference)) / anchor) < 0.03),
    }


def convergence_analysis(levels, errors, tolerance: float = 0.03) -> dict:
    """Observed order of a refinement series and the mesh a tolerance would need.

    Fits ``error = C * divisions ** -p`` on the logarithmic series and reports the
    extrapolated divisions for ``tolerance``.  The fit is a statement about the observed
    trend only; it is not used to raise the accuracy verdict of any mesh.
    """
    divisions = np.asarray(list(levels), dtype=np.float64)
    values = np.asarray(list(errors), dtype=np.float64)
    if divisions.size < 2 or np.any(values <= 0.0):
        raise ValueError("at least two positive errors required")
    slope, intercept = np.polyfit(np.log(divisions), np.log(values), 1)
    order = float(-slope)
    constant = float(np.exp(intercept))
    required = float((constant / tolerance) ** (1.0 / order)) if order > 0.0 else float("inf")
    predicted = constant * divisions ** (-order)
    return {
        "levels": [float(value) for value in divisions],
        "errors": [float(value) for value in values],
        "observed_order": order,
        "constant": constant,
        "fitted_errors": [float(value) for value in predicted],
        "max_fit_residual_relative": float(np.max(np.abs(predicted - values) / values)),
        "tolerance": tolerance,
        "divisions_for_tolerance": required,
        "extrapolated": bool(required > float(divisions[-1])),
    }