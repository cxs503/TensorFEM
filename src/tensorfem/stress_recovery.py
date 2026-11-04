"""Production stress recovery for bilinear quadrilateral (Q4) continuum fields.

``tensorfem.continuum.q4_stiffness`` returns a single *element-centre* B matrix, so
``additional_benchmark_fields.cook_fields`` stores ``C @ B(0,0) @ u``: a constant
stress per element, which is only first-order accurate.  The Cook membrane audit
(``docs/cook-membrane-pointwise-protocol.md``) showed that this first-order field
leaves a 3.687 % pointwise von Mises residual at n=128 and cannot be certified
against the 3 % gate, while the standard Barlow / Zienkiewicz–Zhu construction -
stresses at the 2x2 Gauss points extrapolated to the element nodes and averaged -
drops the same residual to 0.12 % at n=64.  That recovery lived only in the audit
module; this module is the production implementation.

Conventions (identical to ``continuum.q4_stiffness``):

* node order ``(-1,-1) (1,-1) (1,1) (-1,1)`` counter-clockwise;
* engineering shear strain ``gamma_xy`` in the third strain slot, so the third
  stress component is ``tau_xy``;
* plane-stress ``dmat`` from :func:`tensorfem.continuum.elasticity_matrix`.

Everything here is pure tensor algebra: no assembly, no solver, no global state.
"""

from __future__ import annotations

import math

import torch

GAUSS = 1.0 / math.sqrt(3.0)
# Gauss slot order matches the tensor-product order of the extrapolation matrix.
GAUSS_POINTS = ((-GAUSS, -GAUSS), (GAUSS, -GAUSS), (GAUSS, GAUSS), (-GAUSS, GAUSS))
NODE_ORDER = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
_A = 0.5 + 0.5 / GAUSS  # 1.366025...
_B = 0.5 - 0.5 / GAUSS  # -0.366025...


def _b_matrix(derivatives: torch.Tensor) -> torch.Tensor:
    """Strain-displacement matrix from shape derivatives (ne, 4 nodes, 2 coords)."""
    b = derivatives.new_zeros((len(derivatives), 3, 8))
    b[:, 0, 0::2] = derivatives[:, :, 0]
    b[:, 1, 1::2] = derivatives[:, :, 1]
    b[:, 2, 0::2] = derivatives[:, :, 1]
    b[:, 2, 1::2] = derivatives[:, :, 0]
    return b


def _natural_derivatives(xi: float, eta: float, like: torch.Tensor) -> torch.Tensor:
    """dN/dxi, dN/deta for the four nodes, shape (4, 2)."""
    return like.new_tensor(
        [
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)],
        ]
    ) / 4


def _jacobian(coordinates: torch.Tensor, natural: torch.Tensor) -> torch.Tensor:
    """Jacobian (ne, 2, 2) of the bilinear map at the given natural derivatives."""
    return torch.einsum("eia,ib->eab", coordinates, natural)


def extrapolation_matrix(like: torch.Tensor | None = None) -> torch.Tensor:
    """4x4 matrix mapping the four 2x2 Gauss stresses to the four element nodes.

    Linear extrapolation from ``-/+ 1/sqrt(3)`` to ``-/+ 1`` is exact for the
    bilinear-plus-constant field that a Q4 Gauss point carries, so this is the
    Barlow extrapolation for the tensor-product element.
    """
    dtype = None if like is None else like.dtype
    device = None if like is None else like.device
    one_d = torch.tensor([[_A, _B], [_B, _A]], dtype=dtype, device=device)
    out = torch.zeros((4, 4), dtype=dtype, device=device)
    for node, (nx, ny) in enumerate(NODE_ORDER):
        for slot, (px, py) in enumerate(GAUSS_POINTS):
            out[node, slot] = one_d[0 if nx < 0 else 1, 0 if px < 0 else 1] * one_d[
                0 if ny < 0 else 1, 0 if py < 0 else 1
            ]
    return out


def q4_gauss_stresses(
    coordinates: torch.Tensor, dmat: torch.Tensor, displacement: torch.Tensor
) -> torch.Tensor:
    """Stress at the four 2x2 Gauss points of every element, shape ``(ne, 4, 3)``.

    ``displacement`` is the element displacement vector, shape ``(ne, 8)``.
    """
    if coordinates.shape[-1] != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape (nelements, 4, 2)")
    if displacement.shape[:2] != (coordinates.shape[0], 8):
        raise ValueError("displacement must have shape (nelements, 8)")
    out = displacement.new_empty((coordinates.shape[0], 4, 3))
    for slot, (xi, eta) in enumerate(GAUSS_POINTS):
        natural = _natural_derivatives(xi, eta, coordinates)
        jac = _jacobian(coordinates, natural)
        det = torch.linalg.det(jac)
        if bool(torch.any(det <= 0)):
            raise ValueError("Q4 element has non-positive Jacobian")
        derivatives = torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jac))
        b = _b_matrix(derivatives)
        out[:, slot] = (dmat @ b @ displacement.unsqueeze(-1)).squeeze(-1)
    return out


def q4_element_areas(coordinates: torch.Tensor) -> torch.Tensor:
    """Shoelace area of each quadrilateral (clockwise node loops are rejected)."""
    x, y = coordinates[:, :, 0], coordinates[:, :, 1]
    area = 0.5 * (
        x * torch.roll(y, -1, dims=1) - torch.roll(x, -1, dims=1) * y
    ).sum(dim=1)
    return area.abs()


def recover_nodal_stress(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    dmat: torch.Tensor,
    displacement: torch.Tensor,
) -> torch.Tensor:
    """Gauss-point extrapolation plus area-weighted nodal averaging, ``(nnodes, 3)``.

    ``displacement`` is the nodal vector, shape ``(nnodes, 2)`` or ``(2*nnodes,)``.
    The returned nodal field is *single valued*: every incident element contributes
    to a node, so interpolation is well posed even on element boundaries, unlike the
    discontinuous element-centre field.
    """
    coords = nodes[elements]
    flat = displacement.reshape(-1, 2)
    element_u = flat[elements].reshape(len(elements), 8)
    gauss = q4_gauss_stresses(coords, dmat, element_u)
    nodal_values = torch.einsum("ns,esc->enc", extrapolation_matrix(coords), gauss)
    areas = q4_element_areas(coords)
    numerator = torch.zeros((len(nodes), 3), dtype=nodes.dtype, device=nodes.device)
    weight = torch.zeros((len(nodes),), dtype=nodes.dtype, device=nodes.device)
    for local in range(4):
        numerator.index_add_(0, elements[:, local], areas.unsqueeze(-1) * nodal_values[:, local])
        weight.index_add_(0, elements[:, local], areas)
    if bool(torch.any(weight <= 0)):
        raise ValueError("every node must belong to at least one element")
    return numerator / weight.unsqueeze(-1)


def q4_shape(xi: float, eta: float, like: torch.Tensor | None = None) -> torch.Tensor:
    """Bilinear shape functions for the node order used by the Q4 element."""
    return (
        torch.stack(
            (
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            )
        )
        if like is None
        else like.new_tensor(
            (
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            )
        )
    )


def locate_point(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    point,
    tolerance: float = 1e-11,
    max_iterations: int = 80,
) -> dict:
    """Find every element containing ``point`` and its parent coordinates.

    The bilinear map inverts to a quadratic, so a bare Newton iteration started at
    the parent-centre origin does not converge globally on a skewed element (that
    defect is documented in ``docs/cook-membrane-pointwise-protocol.md``).  This
    routine therefore runs multi-start Newton - centre origin plus the four parent
    corners - verifies the mapped residual, and raises instead of silently returning
    a wrong element.
    """
    target = torch.as_tensor(point, dtype=nodes.dtype, device=nodes.device).reshape(2)
    corners = nodes[elements]
    lower = corners.min(dim=1).values - 1e-9
    upper = corners.max(dim=1).values + 1e-9
    inside_box = ((target >= lower) & (target <= upper)).all(dim=1)
    hits: list[dict] = []
    for index in torch.nonzero(inside_box, as_tuple=False).reshape(-1).tolist():
        xy = corners[index]
        best = None
        for start in ((0.0, 0.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            xi, eta = start
            for _ in range(max_iterations):
                shape = q4_shape(xi, eta, xy)
                residual = target - shape @ xy
                if float(torch.linalg.vector_norm(residual)) < tolerance:
                    break
                jac = _jacobian(xy.unsqueeze(0), _natural_derivatives(xi, eta, xy))[0]
                try:
                    step = torch.linalg.solve(jac, residual)
                except RuntimeError as error:  # singular element
                    raise ValueError(f"singular Q4 Jacobian in element {index}") from error
                xi, eta = xi + float(step[0]), eta + float(step[1])
            else:
                continue
            if abs(xi) <= 1.0 + 1e-9 and abs(eta) <= 1.0 + 1e-9 and best is None:
                best = (xi, eta)
        if best is not None:
            hits.append(
                {
                    "element": int(index),
                    "local": [float(best[0]), float(best[1])],
                    "nodes": elements[index].tolist(),
                    "on_boundary": bool(
                        abs(abs(best[0]) - 1.0) < 1e-9 or abs(abs(best[1]) - 1.0) < 1e-9
                    ),
                }
            )
    if not hits:
        raise ValueError(f"point {list(map(float, target))} is not contained in any element")
    return {"point": [float(v) for v in target], "hits": hits}


def interpolate_nodal_stress(
    nodes: torch.Tensor, elements: torch.Tensor, nodal_stress: torch.Tensor, point
) -> dict:
    """Bilinearly interpolate a recovered nodal field at ``point``.

    A recovered nodal field is single valued by construction, so the choice among
    containing elements must not change the answer.  ``element_spread`` reports the
    largest von Mises difference between the candidate elements; a non-zero value on
    a field that claims to be nodal is a recovery defect and is surfaced, not hidden.
    """
    located = locate_point(nodes, elements, point)
    values = []
    for hit in located["hits"]:
        shape = q4_shape(hit["local"][0], hit["local"][1], nodes)
        stress = shape @ nodal_stress[torch.as_tensor(hit["nodes"], dtype=torch.long)]
        values.append(
            {
                "element": hit["element"],
                "local": hit["local"],
                "on_boundary": hit["on_boundary"],
                "stress": [float(v) for v in stress],
                "von_mises": von_mises(torch.as_tensor(stress)),
            }
        )
    reference = values[0]["von_mises"]
    spread = max(abs(entry["von_mises"] - reference) for entry in values)
    return {
        "point": located["point"],
        "element": values[0]["element"],
        "local": values[0]["local"],
        "candidate_count": len(values),
        "stress": values[0]["stress"],
        "von_mises": reference,
        "element_spread": float(spread),
        "single_valued": bool(spread < 1e-9),
    }


def von_mises(stress: torch.Tensor) -> float:
    """Plane-stress von Mises equivalent of ``(sxx, syy, tau_xy)``."""
    sxx, syy, sxy = float(stress[0]), float(stress[1]), float(stress[2])
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def relative_error(value: float, reference: float, scale: float, floor_fraction: float = 0.05) -> float:
    """Relative error with the audit's 5 %-of-scale floor (stable near zero stress)."""
    return abs(value - reference) / max(abs(reference), floor_fraction * abs(scale))


def recovery_verdict(
    errors,
    gate: float = 0.03,
    require_monotone: bool = True,
    gate_tolerance: float = 0.0,
) -> dict:
    """Fail-closed verdict for a recovered-stress accuracy ladder.

    ``errors`` is the per-level worst-site relative error, coarsest level first.
    The verdict is ``True`` only when every level is finite and non-negative, the
    finest level is strictly below ``gate - gate_tolerance``, and (by default) the
    ladder decreases monotonically.  Missing, NaN or ``withheld`` input is a
    failure, never a pass.
    """
    values = [float(value) for value in errors]
    if not values:
        raise ValueError("an accuracy verdict needs at least one level")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("accuracy ladder contains a non-finite or negative error")
    monotone = all(b < a for a, b in zip(values, values[1:]))
    qualified = (values[-1] is not None) and (values[-1] < gate - gate_tolerance)
    if require_monotone and not monotone:
        qualified = False
    return {
        "levels": len(values),
        "finest": values[-1],
        "gate": float(gate),
        "monotone": bool(monotone),
        "require_monotone": bool(require_monotone),
        "qualified": bool(qualified),
    }