"""Small-sliding, frictionless node-to-segment contact assembly.

The contact normal and projection weights are evaluated in the reference
configuration.  The resulting affine gap constraints can be passed directly
to :func:`tensorfem.contact.solve_rigid_plane_contact`.  This is a verified
small-sliding kernel, not a finite-sliding surface search algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NodeSegmentPair:
    slave: int
    master_a: int
    master_b: int


@dataclass(frozen=True)
class ContactConstraints:
    matrix: torch.Tensor
    initial_gap: torch.Tensor
    projection: torch.Tensor
    normal: torch.Tensor


def assemble_node_segment_constraints(
    coordinates: torch.Tensor,
    pairs: tuple[NodeSegmentPair, ...] | list[NodeSegmentPair],
) -> ContactConstraints:
    """Assemble ``gap = initial_gap + matrix @ displacement >= 0``.

    Each slave point is projected onto its master segment.  Projections must
    lie on the closed segment; degenerate segments and repeated node indices
    are rejected.  Two translational degrees of freedom per node are assumed.
    Master segment orientation defines the normal ``[-t_y, t_x]``.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (nnode, 2)")
    nnode = coordinates.shape[0]
    rows, gaps, xis, normals = [], [], [], []
    for pair in pairs:
        ids = (pair.slave, pair.master_a, pair.master_b)
        if len(set(ids)) != 3 or min(ids) < 0 or max(ids) >= nnode:
            raise ValueError("contact pair must contain three distinct valid nodes")
        xs, xa, xb = (coordinates[i] for i in ids)
        edge = xb - xa
        length2 = torch.dot(edge, edge)
        if float(length2) <= torch.finfo(coordinates.dtype).eps:
            raise ValueError("master segment is degenerate")
        xi = torch.dot(xs - xa, edge) / length2
        tol = 100 * torch.finfo(coordinates.dtype).eps
        if float(xi) < -tol or float(xi) > 1.0 + tol:
            raise ValueError("slave projection lies outside master segment")
        xi = torch.clamp(xi, 0.0, 1.0)
        tangent = edge / torch.sqrt(length2)
        normal = torch.stack((-tangent[1], tangent[0]))
        projected = (1.0 - xi) * xa + xi * xb
        row = torch.zeros(2 * nnode, dtype=coordinates.dtype, device=coordinates.device)
        row[2 * pair.slave:2 * pair.slave + 2] = normal
        row[2 * pair.master_a:2 * pair.master_a + 2] = -(1.0 - xi) * normal
        row[2 * pair.master_b:2 * pair.master_b + 2] = -xi * normal
        rows.append(row)
        gaps.append(torch.dot(xs - projected, normal))
        xis.append(xi)
        normals.append(normal)
    if not rows:
        return ContactConstraints(
            coordinates.new_zeros((0, 2 * nnode)), coordinates.new_zeros((0,)),
            coordinates.new_zeros((0,)), coordinates.new_zeros((0, 2)))
    return ContactConstraints(torch.stack(rows), torch.stack(gaps),
                              torch.stack(xis), torch.stack(normals))


def independent_active_reactions(
    stiffness: torch.Tensor, load: torch.Tensor, matrix: torch.Tensor,
    initial_gap: torch.Tensor,
) -> torch.Tensor:
    """Closed-form reactions assuming all supplied constraints are active.

    This Schur-complement expression is kept independent of the KKT solver and
    is useful as a multi-contact benchmark oracle.
    """
    kinv_f = torch.linalg.solve(stiffness, load)
    kinv_ct = torch.linalg.solve(stiffness, matrix.T)
    return torch.linalg.solve(matrix @ kinv_ct, -(initial_gap + matrix @ kinv_f))
