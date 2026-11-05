"""Distorted- and curved-geometry qualification of the serendipity HEX20 brick.

``docs/element-rank-deficiency-audit.md`` §6 recorded an explicit open item: the
20-node brick was audited only on the *regular* unit cube, so neither the
distorted brick nor the curved-geometry (bulged midside node) brick had
zero-energy-mode evidence.  This module closes that item and adds the
geometry-degradation benchmarks that ``docs/hex20-validation.md`` named as the
gate for promoting HEX20 out of its provisional status:

* **rank sufficiency under distortion** — a single free element must still have
  exactly the six rigid-body zero modes (no spurious hourglass mode), for a
  deterministic corner-distortion pattern and for a curved (bulged) brick whose
  midside nodes leave the chord midpoints;
* **affine patch on a warped conforming mesh** — a quadratic isoparametric
  mapping represents an affine displacement field exactly even when every
  element is geometrically distorted, so a prescribed-boundary solve on a warped
  mesh must reproduce the field to round-off;
* **warped cantilever A/B** — the documented HEX20/HEX8 cantilever is re-run on
  meshes whose *interior* nodes are warped (boundary nodes untouched, so the
  boundary-value problem of the reference beam is unchanged) and both the
  absolute Timoshenko error and the distortion-induced change of the tip
  deflection are reported.

The end-face shear is assembled as a genuinely consistent load: the element's own
shape functions are integrated over the loaded parent face with the exact face
Jacobian, so the total applied load equals the nominal load for every mesh and
both element types.  This is also a self-check — at ``warp = 0`` and
``ny = nz = 1`` it must reproduce the documented HEX20 cantilever error.

All quantities are deterministic; nothing here samples random distortion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .element_rank_audit import RankRecord, audit_element, rigid_body_basis_solid3d
from .hex20 import (
    Hex20Model,
    hex20_shape_gradients,
    hex20_stiffness,
    solve_hex20,
    structured_hex20_mesh,
)
from .solid3d import SolidModel, elasticity_matrix_3d, solve_solid, structured_hex_mesh

__all__ = [
    "DistortionRecord",
    "distorted_brick",
    "curved_brick",
    "brick_jacobian_extremes",
    "rank_probe",
    "distortion_sweep",
    "warped_hex20_mesh",
    "affine_patch_error",
    "consistent_end_shear",
    "warped_cantilever",
    "run_audit",
]

# Unit-cube corner order of ``hex20``; the distortion pattern is a fixed
# deterministic offset per corner, normalised to unit maximum magnitude.
_CORNER_ORDER = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)
_CORNER_PATTERN = (
    (0.0, 0.0, 0.0),
    (0.10, 0.62, -0.35),
    (0.55, 0.10, 0.20),
    (0.40, 0.05, 0.62),
    (0.15, 0.30, 0.75),
    (0.05, 0.85, 0.95),
    (-0.35, 0.10, 0.30),
    (0.20, 0.75, -0.45),
)
_EDGE_ORDER = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_CORNER_PARENT = (
    (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
)

_GAUSS_1D = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GAUSS_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def _normalised_pattern(dtype, device):
    pattern = torch.tensor(_CORNER_PATTERN, dtype=dtype, device=device)
    return pattern / torch.linalg.vector_norm(pattern, dim=1).max()


def distorted_brick(amplitude: float = 0.10, *, curvature: float = 0.0, dtype=torch.float64):
    """Return a single 20-node brick on the unit cube with deterministic distortion.

    ``amplitude`` scales a fixed corner-offset pattern (0 reproduces the regular
    cube bit for bit).  ``curvature`` additionally moves every midside node
    radially away from the brick centroid, which makes the faces genuinely
    curved instead of merely warped, so geometry and displacement interpolations
    are both quadratic.
    """
    x = torch.tensor(_CORNER_ORDER, dtype=dtype)
    corners = x + amplitude * _normalised_pattern(dtype, x.device)
    mids = torch.stack([(corners[a] + corners[b]) / 2 for a, b in _EDGE_ORDER])
    if curvature:
        centre = corners.mean(0)
        mids = centre + (1.0 + curvature) * (mids - centre)
    return torch.cat((corners, mids), 0)


def curved_brick(curvature: float = 0.04, *, dtype=torch.float64):
    """A brick with straight-edge-consistent corners but bulged midside nodes."""
    return distorted_brick(0.0, curvature=curvature, dtype=dtype)


def brick_jacobian_extremes(nodes: torch.Tensor) -> tuple[float, float]:
    """Return ``(min, max)`` Jacobian determinant at the 3x3x3 Gauss points."""
    x = nodes.reshape(1, -1, 3)
    determinants = []
    for r in _GAUSS_1D:
        for s in _GAUSS_1D:
            for t in _GAUSS_1D:
                _, dn = hex20_shape_gradients(r, s, t, like=x)
                j = torch.einsum("eia,ib->eab", x, dn)
                determinants.append(torch.linalg.det(j))
    values = torch.cat(determinants)
    return float(values.min()), float(values.max())


def rank_probe(nodes: torch.Tensor, *, young: float = 200.0e9, poisson: float = 0.3) -> RankRecord:
    """Classify the null space of a single free HEX20 element."""
    d = elasticity_matrix_3d(
        torch.tensor([young], dtype=nodes.dtype), torch.tensor([poisson], dtype=nodes.dtype)
    )
    k, _ = hex20_stiffness(nodes.unsqueeze(0), d)
    return audit_element("hex20", k[0], rigid_body_basis_solid3d(nodes))


@dataclass(frozen=True)
class DistortionRecord:
    """One (amplitude, curvature) point of the distortion sweep."""

    amplitude: float
    curvature: float
    null_dim: int
    spurious: int
    first_positive_ratio: float
    min_jacobian: float
    max_jacobian: float
    truncated: bool


def distortion_sweep(
    amplitudes=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25),
    curvatures=(0.0, 0.01, 0.02, 0.04, 0.06),
    *,
    young: float = 200.0e9,
    poisson: float = 0.3,
) -> list[DistortionRecord]:
    """Sweep corner distortion against midside curvature.

    A combination whose geometry is degenerate (non-positive Jacobian) is
    reported as ``truncated`` rather than silently skipped: the brick must fail
    closed there, and the sweep documents where that boundary lies.
    """
    rows: list[DistortionRecord] = []
    for amplitude in amplitudes:
        for curvature in curvatures:
            nodes = distorted_brick(amplitude, curvature=curvature)
            try:
                lo, hi = brick_jacobian_extremes(nodes)
                record = rank_probe(nodes, young=young, poisson=poisson)
            except ValueError:
                rows.append(
                    DistortionRecord(
                        amplitude=amplitude,
                        curvature=curvature,
                        null_dim=-1,
                        spurious=-1,
                        first_positive_ratio=float("nan"),
                        min_jacobian=float("nan"),
                        max_jacobian=float("nan"),
                        truncated=True,
                    )
                )
                continue
            rows.append(
                DistortionRecord(
                    amplitude=amplitude,
                    curvature=curvature,
                    null_dim=record.null_dim,
                    spurious=record.spurious,
                    first_positive_ratio=record.first_positive_ratio,
                    min_jacobian=lo,
                    max_jacobian=hi,
                    truncated=False,
                )
            )
    return rows


def _warp_nodes(nodes: torch.Tensor, length: float, width: float, height: float, warp: float) -> torch.Tensor:
    """Displace strictly interior nodes by a smooth field vanishing on the boundary."""
    if not warp:
        return nodes
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    inside = (
        (x > 1e-12) & (x < length - 1e-12)
        & (y > 1e-12) & (y < width - 1e-12)
        & (z > 1e-12) & (z < height - 1e-12)
    )
    blob = (
        torch.sin(math.pi * x / length)
        * torch.sin(math.pi * y / width)
        * torch.sin(math.pi * z / height)
    )
    warped = nodes.clone()
    warped[inside, 1] += warp * blob[inside]
    warped[inside, 2] -= 0.5 * warp * blob[inside]
    return warped


def warped_hex20_mesh(
    length: float = 2.0,
    width: float = 1.0,
    height: float = 0.8,
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
    *,
    warp: float = 0.0,
    dtype=torch.float64,
):
    """Structured HEX20 mesh whose strictly interior nodes are displaced."""
    nodes, cells = structured_hex20_mesh(length, width, height, nx, ny, nz, dtype=dtype)
    return _warp_nodes(nodes, length, width, height, warp), cells


def affine_patch_error(
    amplitude: float = 0.012,
    *,
    warp: float = 0.05,
    length: float = 2.0,
    width: float = 1.0,
    height: float = 0.8,
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
) -> float:
    """Relative displacement error of a prescribed-boundary affine patch solve.

    A quadratic isoparametric mapping reproduces an affine field exactly, hence
    a *warped* conforming mesh must still be exact to round-off.  Any non-zero
    value here is a discretisation defect, not a convergence issue.
    """
    nodes, cells = warped_hex20_mesh(length, width, height, nx, ny, nz, warp=warp)
    dtype = nodes.dtype
    a = amplitude * torch.tensor(
        [[1.0, 0.6, -0.25], [-0.2, -0.75, 0.4], [0.35, 0.5, 1.25]], dtype=dtype
    )
    exact = (nodes @ a.T + torch.tensor([0.1, -0.2, 0.05], dtype=dtype)).reshape(-1)
    boundary = (
        (nodes[:, 0] <= 1e-12) | (nodes[:, 0] >= length - 1e-12)
        | (nodes[:, 1] <= 1e-12) | (nodes[:, 1] >= width - 1e-12)
        | (nodes[:, 2] <= 1e-12) | (nodes[:, 2] >= height - 1e-12)
    )
    ids = torch.where(boundary)[0]
    fixed = torch.stack((3 * ids, 3 * ids + 1, 3 * ids + 2), 1).reshape(-1)
    zero = nodes.new_zeros(3 * len(nodes))
    result = solve_hex20(
        Hex20Model(
            nodes, cells, nodes.new_tensor(70e9), nodes.new_tensor(0.27), zero, fixed, exact[fixed]
        )
    )
    return float(
        torch.linalg.vector_norm(result.displacement - exact) / torch.linalg.vector_norm(exact)
    )


def _hex8_shape(r, s, t, parent, like):
    """Trilinear HEX8 shape values and natural derivatives for one parent point."""
    r = torch.as_tensor(r, dtype=like.dtype)
    s = torch.as_tensor(s, dtype=like.dtype)
    t = torch.as_tensor(t, dtype=like.dtype)
    values, derivs = [], []
    for (ri, si, ti) in parent:
        a, b, c = 1.0 + ri * r, 1.0 + si * s, 1.0 + ti * t
        values.append(a * b * c / 8.0)
        derivs.append(((ri * b * c) / 8.0, (si * a * c) / 8.0, (ti * a * b) / 8.0))
    return torch.stack(values), torch.tensor(derivs, dtype=like.dtype)


def consistent_end_shear(
    nodes: torch.Tensor,
    cells: torch.Tensor,
    *,
    element: str,
    end_axis: int,
    total_load: float,
    area: float,
    load_axis: int | None = None,
) -> torch.Tensor:
    """Consistent nodal load for a uniform traction on the ``end_axis`` face.

    The traction is integrated with each loaded element's own shape functions
    and the exact face Jacobian, so the assembled load sums to ``total_load``
    for any mesh density, both element types and warped faces.
    """
    if element == "hex20":
        parent = [torch.tensor(_CORNER_PARENT[i], dtype=nodes.dtype) for i in range(8)]
        parent += [0.5 * (parent[a] + parent[b]) for a, b in _EDGE_ORDER]
        parent = torch.stack(parent)
        shape_fn = hex20_shape_gradients
    elif element == "hex8":
        parent = torch.tensor(_CORNER_PARENT, dtype=nodes.dtype)
        shape_fn = None
    else:
        raise ValueError("element must be 'hex20' or 'hex8'")

    load_axis = end_axis if load_axis is None else load_axis
    traction = total_load / area
    end_value = float(nodes[:, end_axis].max())
    forces = nodes.new_zeros(3 * len(nodes))
    s_axis, t_axis = [a for a in range(3) if a != end_axis]
    for cell in cells.tolist():
        coords = nodes[cell]
        face_local = torch.where(parent[:, end_axis] > 0)[0]
        if not len(face_local):
            continue
        # only the parent face lying on the loaded end plane carries traction
        if abs(coords[face_local, end_axis].mean() - end_value) > 1e-9 * max(1.0, abs(end_value)):
            continue
        for i, r in enumerate(_GAUSS_1D):
            for j, s in enumerate(_GAUSS_1D):
                point = [0.0, 0.0, 0.0]
                point[end_axis] = 1.0
                point[s_axis] = r
                point[t_axis] = s
                if shape_fn is None:
                    n, dn = _hex8_shape(point[0], point[1], point[2], _CORNER_PARENT, coords)
                else:
                    n, dn = shape_fn(point[0], point[1], point[2], like=coords)
                x_s = torch.einsum("a,ai->i", dn[face_local, s_axis], coords[face_local])
                x_t = torch.einsum("a,ai->i", dn[face_local, t_axis], coords[face_local])
                jac = torch.linalg.vector_norm(torch.linalg.cross(x_s, x_t))
                weight = _GAUSS_W[i] * _GAUSS_W[j] * jac
                for a in face_local.tolist():
                    forces[3 * cell[a] + load_axis] += n[a] * traction * weight
    return forces


def _timoshenko_reference(
    load: float, length: float, width: float, height: float, young: float, poisson: float
) -> float:
    inertia = width * height**3 / 12.0
    shear = young / (2.0 * (1.0 + poisson))
    return (
        load * length**3 / (3.0 * young * inertia)
        + load * length / ((5.0 / 6.0) * shear * width * height)
    )


def warped_cantilever(
    nx: int = 8,
    *,
    warp: float = 0.0,
    element: str = "hex20",
    ny: int = 2,
    nz: int = 2,
) -> dict:
    """Cantilever A/B on a warped mesh, scored against the Timoshenko beam.

    ``warp = 0`` reproduces the documented benchmark; ``warp > 0`` moves only
    strictly interior nodes, so the boundary-value problem is identical and the
    tip-deflection change isolates the effect of element distortion.
    """
    young, poisson, length, width, height, load = 210e9, 0.3, 10.0, 1.0, 1.0, -1e6
    if element == "hex20":
        nodes, cells = structured_hex20_mesh(length, width, height, nx, ny, nz)
    elif element == "hex8":
        nodes, cells = structured_hex_mesh(length, width, height, nx, ny, nz)
    else:
        raise ValueError("element must be 'hex20' or 'hex8'")
    nodes = _warp_nodes(nodes, length, width, height, warp)

    forces = consistent_end_shear(
        nodes, cells, element=element, end_axis=0, total_load=load,
        area=width * height, load_axis=2,
    )
    applied = float(-forces.sum())
    right = torch.where(nodes[:, 0] == length)[0]
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((3 * left, 3 * left + 1, 3 * left + 2), 1).reshape(-1)
    if element == "hex20":
        result = solve_hex20(
            Hex20Model(
                nodes, cells, nodes.new_tensor(young), nodes.new_tensor(poisson), forces, fixed
            )
        )
    else:
        result = solve_solid(
            SolidModel(
                nodes, cells, nodes.new_tensor(young), nodes.new_tensor(poisson), forces, fixed, "hex8"
            )
        )
    computed = float(result.displacement[3 * right + 2].mean())
    reference = _timoshenko_reference(applied, length, width, height, young, poisson)
    return {
        "element": element,
        "mesh": f"{nx}x{ny}x{nz}",
        "warp": warp,
        "applied_load": applied,
        "computed": computed,
        "reference": reference,
        "error_percent": abs((abs(computed) - reference) / reference) * 100.0,
    }


def run_audit(*, include_cantilever: bool = True) -> dict:
    """Collect every scalar quoted in ``docs/hex20-distortion-robustness.md``."""
    sweep = [
        {
            "amplitude": row.amplitude,
            "curvature": row.curvature,
            "null_dim": row.null_dim,
            "spurious": row.spurious,
            "first_positive_ratio": row.first_positive_ratio,
            "min_jacobian": row.min_jacobian,
            "max_jacobian": row.max_jacobian,
            "truncated": row.truncated,
        }
        for row in distortion_sweep()
    ]
    payload: dict = {
        "regular_cube_min_jacobian": brick_jacobian_extremes(distorted_brick(0.0))[0],
        "distortion_sweep": sweep,
        "affine_patch": [
            {"warp": warp, "relative_error": affine_patch_error(warp=warp)}
            for warp in (0.0, 0.01, 0.05, 0.10)
        ],
    }
    if include_cantilever:
        payload["cantilever"] = [
            warped_cantilever(8, warp=warp, element=element)
            for element in ("hex20", "hex8")
            for warp in (0.0, 0.01, 0.02, 0.04)
        ]
    return payload