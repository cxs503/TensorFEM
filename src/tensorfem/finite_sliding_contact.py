"""Finite-sliding contact of a 2-D node with a rigid polyline.

This module is deliberately a local contact kernel: geometry is searched in
the *current* configuration at every call and Coulomb friction is integrated
with an elastic predictor / plastic corrector.  It does not implement mortar,
self-contact, or a global nonlinear finite-element solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolylineProjection:
    point: torch.Tensor
    tangent: torch.Tensor
    normal: torch.Tensor
    coordinate: torch.Tensor
    gap: torch.Tensor
    segment: int
    distance: torch.Tensor


@dataclass(frozen=True)
class FrictionState:
    normal_multiplier: torch.Tensor
    elastic_slip: torch.Tensor
    dissipated_energy: torch.Tensor
    projection: torch.Tensor
    segment: int
    active: bool
    sticking: bool


@dataclass(frozen=True)
class ContactUpdate:
    traction: torch.Tensor
    normal_traction: torch.Tensor
    tangential_traction: torch.Tensor
    gap: torch.Tensor
    projection: PolylineProjection
    state: FrictionState
    stored_energy: torch.Tensor
    dissipation_increment: torch.Tensor


def _check_polyline(vertices: torch.Tensor) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 2:
        raise ValueError("vertices must have shape (n>=2, 2)")
    lengths2 = torch.sum((vertices[1:] - vertices[:-1]) ** 2, dim=1)
    if bool(torch.any(lengths2 <= torch.finfo(vertices.dtype).eps)):
        raise ValueError("polyline contains a degenerate segment")


def project_point_to_polyline(point: torch.Tensor, vertices: torch.Tensor) -> PolylineProjection:
    """Return the closest projection onto all closed polyline segments.

    Vertex order orients every segment and its outward normal is ``[-ty,tx]``.
    The signed gap is measured along that normal.  End-point projections are
    included, which permits a node to move from one segment to the next.
    """
    _check_polyline(vertices)
    if point.shape != (2,) or point.dtype != vertices.dtype:
        raise ValueError("point must have shape (2,) and match vertices dtype")
    a, edge = vertices[:-1], vertices[1:] - vertices[:-1]
    length2 = torch.sum(edge * edge, dim=1)
    xi = torch.clamp(torch.sum((point - a) * edge, dim=1) / length2, 0.0, 1.0)
    projected = a + xi[:, None] * edge
    distance2 = torch.sum((point - projected) ** 2, dim=1)
    segment = int(torch.argmin(distance2))
    tangent = edge[segment] / torch.sqrt(length2[segment])
    normal = torch.stack((-tangent[1], tangent[0]))
    closest = projected[segment]
    return PolylineProjection(
        closest, tangent, normal, xi[segment],
        torch.dot(point - closest, normal), segment,
        torch.sqrt(distance2[segment]),
    )


def initial_friction_state(point: torch.Tensor, vertices: torch.Tensor) -> FrictionState:
    """Construct a zero-traction history state at the current projection."""
    projection = project_point_to_polyline(point, vertices)
    zero = point.new_zeros(())
    return FrictionState(zero, zero, zero, projection.point, projection.segment, False, True)


def update_node_polyline_contact(
    point: torch.Tensor,
    vertices: torch.Tensor,
    state: FrictionState,
    *,
    normal_penalty: float | torch.Tensor,
    tangential_penalty: float | torch.Tensor,
    friction: float | torch.Tensor,
    relative_tangential_increment: torch.Tensor | float | None = None,
) -> ContactUpdate:
    """Update finite-sliding contact and Coulomb friction for one increment.

    Normal contact uses the robust penalty law
    ``lambda=kn*max(-gap,0)``.  Tangential traction uses a return
    mapping with yield function ``abs(tau)-mu*lambda``.  If an explicit
    relative tangential increment is omitted, motion of the closest point is
    projected onto the current tangent.  Supplying it is preferred when the
    master surface also moves.
    """
    projection = project_point_to_polyline(point, vertices)
    kn = torch.as_tensor(normal_penalty, dtype=point.dtype, device=point.device)
    kt = torch.as_tensor(tangential_penalty, dtype=point.dtype, device=point.device)
    mu = torch.as_tensor(friction, dtype=point.dtype, device=point.device)
    if bool(kn <= 0) or bool(kt <= 0) or bool(mu < 0):
        raise ValueError("penalties must be positive and friction non-negative")

    lam = kn * torch.clamp(-projection.gap, min=0.0)
    if relative_tangential_increment is None:
        ds = torch.dot(projection.point - state.projection, projection.tangent)
    else:
        ds = torch.as_tensor(relative_tangential_increment,
                             dtype=point.dtype, device=point.device)
    zero = point.new_zeros(())
    if bool(lam <= torch.finfo(point.dtype).eps):
        new_state = FrictionState(zero, zero, state.dissipated_energy,
                                  projection.point, projection.segment, False, True)
        return ContactUpdate(torch.zeros_like(point), zero, zero, projection.gap,
                             projection, new_state, zero, zero)

    slip_trial = state.elastic_slip + ds
    tau_trial = -kt * slip_trial
    limit = mu * lam
    sticking = bool(torch.abs(tau_trial) <= limit)
    if sticking:
        tau, elastic_slip, plastic_increment = tau_trial, slip_trial, zero
    else:
        tau = -limit * torch.sign(slip_trial)
        elastic_slip = -tau / kt
        plastic_increment = slip_trial - elastic_slip
    dissipation = torch.abs(tau * plastic_increment)
    total_dissipation = state.dissipated_energy + dissipation
    traction = lam * projection.normal + tau * projection.tangent
    stored = 0.5 * lam**2 / kn + 0.5 * kt * elastic_slip**2
    new_state = FrictionState(lam, elastic_slip, total_dissipation,
                              projection.point, projection.segment, True, sticking)
    return ContactUpdate(traction, lam, tau, projection.gap, projection,
                         new_state, stored, dissipation)
