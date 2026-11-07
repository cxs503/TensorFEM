"""Two-sided frictionless mortar *segment* integration for 3-D facet contact.

:mod:`tensorfem.mortar_contact3d` integrates the penalty potential at the
quadrature points of the slave facet itself, i.e. it is a one-sided facet
quadrature rule that ignores how much of a slave facet actually overlaps a
given master facet.  This module builds the real mortar segments: every slave
facet is clipped against each neighbouring master facet in the master facet
plane, the resulting convex polygon is triangulated, and the penalty potential

    Pi_c = 1/2 * eps_n * \\int_{Gamma_c} < -g_n >_+^2 dA

is integrated with a three-point Gauss rule on every segment triangle.  The
normal traction is distributed to *both* surfaces with the respective facet
shape functions, so two non-matching interface discretisations exchange force
without any node-on-node pairing.

Status of the formulation
-------------------------
* primal penalty mortar, frictionless, small relative sliding;
* segment topology, master normal and slave-side projection weights are frozen
  during one linearisation (the geometry and the active set are refreshed at
  every assembly: "updated mortar"), so the returned residual/tangent pair is
  the exact derivative pair of the frozen potential;
* no dual Lagrange multipliers, no friction, no finite sliding (roadmap);
* facets must be planar to the extent that clipping in the master facet plane
  is meaningful; warped QUAD4 master facets are integrated as two flat
  triangles.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

_TRI_RULE = ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
"""Barycentric abscissae of the three-point triangle rule (weights 1/3 of area)."""


@dataclass(frozen=True)
class MortarSegmentSamples:
    """Per-quadrature-point interface data at the supplied configuration."""

    points: torch.Tensor
    normals: torch.Tensor
    gaps: torch.Tensor
    normal_traction: torch.Tensor
    areas: torch.Tensor
    slave_faces: torch.Tensor
    master_faces: torch.Tensor
    total_normal_force: torch.Tensor
    integrated_area: torch.Tensor
    contact_area: torch.Tensor
    maximum_penetration: torch.Tensor


@dataclass(frozen=True)
class MortarSegmentAssembly:
    """Energy-consistent two-sided mortar segment contribution.

    ``residual`` is the gradient of the discrete penalty potential ordered as
    all slave degrees of freedom followed by all master degrees of freedom;
    physical nodal contact forces are therefore ``-residual``.
    """

    residual: torch.Tensor
    tangent: torch.Tensor
    slave_forces: torch.Tensor
    master_forces: torch.Tensor
    segments: int
    integrated_area: torch.Tensor
    maximum_penetration: torch.Tensor
    penalty_energy: torch.Tensor


def _facet_triangles(face) -> list[tuple[int, int, int]]:
    f = [int(i) for i in face]
    if len(f) == 3:
        return [(f[0], f[1], f[2])]
    return [(f[0], f[1], f[2]), (f[0], f[2], f[3])]


def _clip_halfplane(poly, p, q):
    """Sutherland-Hodgman clip of ``poly`` against the left side of p->q."""
    out = []
    n = len(poly)
    ex, ey = q[0] - p[0], q[1] - p[1]

    def side(x):
        return ex * (x[1] - p[1]) - ey * (x[0] - p[0])

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        sa, sb = side(a), side(b)
        if sa >= 0.0:
            out.append(a)
        if (sa > 0.0 > sb) or (sa < 0.0 < sb):
            t = sa / (sa - sb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _polygon_area(poly) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _facet_weights(nodes, point):
    """Shape weights of ``point`` projected into the plane of facet ``nodes``.

    ``nodes`` is a list of 3-D tensors ordered like the facet connectivity;
    ``point`` is a 3-D tensor on the master plane.  For a TRI3 facet the
    in-plane coordinates are recovered exactly, for a QUAD4 facet the bilinear
    inverse is obtained by Newton iteration.  Both surfaces of an interface are
    planar in the supported scope, for which this projection is the exact
    mortar pairing.
    """
    n = len(nodes)
    origin = nodes[0]
    a = nodes[1] - origin
    length = float(torch.linalg.vector_norm(a))
    if length <= torch.finfo(nodes[0].dtype).eps:
        raise ValueError("degenerate facet")
    e1 = a / length
    nrm = torch.linalg.cross(a, nodes[2] - origin)
    area = float(torch.linalg.vector_norm(nrm))
    if area <= torch.finfo(nodes[0].dtype).eps:
        raise ValueError("degenerate facet")
    e2 = torch.linalg.cross(nrm / area, e1)
    q = [float((nodes[i] - origin) @ e1) for i in range(n)]
    r = [float((nodes[i] - origin) @ e2) for i in range(n)]
    xq = (float((point - origin) @ e1), float((point - origin) @ e2))
    tol = 1e-8
    if n == 3:
        det = q[1] * r[2] - q[2] * r[1]
        if abs(det) <= 1e-14:
            raise ValueError("degenerate facet")
        xi = (xq[0] * r[2] - q[2] * xq[1]) / det
        eta = (q[1] * xq[1] - xq[0] * r[1]) / det
        if xi < -tol or eta < -tol or xi + eta > 1.0 + tol:
            raise ValueError("quadrature point outside slave facet")
        return [1.0 - xi - eta, xi, eta]
    if n != 4:
        raise ValueError("facets must be TRI3 or QUAD4")
    xi = eta = 0.0
    for _ in range(60):
        s = [
            (1 - xi) * (1 - eta) / 4,
            (1 + xi) * (1 - eta) / 4,
            (1 + xi) * (1 + eta) / 4,
            (1 - xi) * (1 + eta) / 4,
        ]
        fx = sum(s[i] * q[i] for i in range(4)) - xq[0]
        fy = sum(s[i] * r[i] for i in range(4)) - xq[1]
        d = (
            (-(1 - eta) / 4, -(1 - xi) / 4),
            ((1 - eta) / 4, -(1 + xi) / 4),
            ((1 + eta) / 4, (1 + xi) / 4),
            (-(1 + eta) / 4, (1 - xi) / 4),
        )
        j11 = sum(d[i][0] * q[i] for i in range(4))
        j12 = sum(d[i][1] * q[i] for i in range(4))
        j21 = sum(d[i][0] * r[i] for i in range(4))
        j22 = sum(d[i][1] * r[i] for i in range(4))
        det = j11 * j22 - j12 * j21
        if abs(det) <= 1e-14:
            raise ValueError("degenerate facet")
        dxi = (fx * j22 - fy * j12) / det
        deta = (-fx * j21 + fy * j11) / det
        xi -= dxi
        eta -= deta
        if abs(dxi) <= 1e-14 and abs(deta) <= 1e-14:
            break
    if abs(xi) > 1.0 + tol or abs(eta) > 1.0 + tol:
        raise ValueError("quadrature point outside slave facet")
    return [
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4,
    ]


def _default_search_distance(slave_coords, slave_faces):
    """Search radius used when the caller passes ``search_distance=None``.

    Two percent of the mean slave facet edge length.  The candidate radius must
    exceed the penalty penetration depth, otherwise a penetrating pair would be
    dropped from the segment set (the axis-aligned boxes of a penetrating slave
    facet are displaced, not overlapping).
    """
    lengths = []
    for face in slave_faces:
        pts = [slave_coords[int(i)] for i in face]
        for k in range(len(pts)):
            lengths.append(float(torch.linalg.vector_norm(pts[(k + 1) % len(pts)] - pts[k])))
    positive = [x for x in lengths if x > 0.0]
    if not positive:
        raise ValueError("degenerate slave surface")
    return 0.02 * sum(positive) / len(positive)


def _candidate_pairs(slave_coords, slave_faces, master_coords, master_faces, search_distance):
    if search_distance is None:
        search_distance = _default_search_distance(slave_coords, slave_faces)
    if search_distance < 0.0:
        raise ValueError("search_distance must be non-negative")
    sb = [(slave_coords[f].min(0).values, slave_coords[f].max(0).values) for f in slave_faces]
    mb = [(master_coords[f].min(0).values, master_coords[f].max(0).values) for f in master_faces]
    pairs = []
    for i in range(len(slave_faces)):
        for j in range(len(master_faces)):
            lo = torch.maximum(sb[i][0], mb[j][0])
            hi = torch.minimum(sb[i][1], mb[j][1])
            if float(torch.linalg.vector_norm(torch.clamp(lo - hi, min=0.0))) <= search_distance:
                pairs.append((i, j))
    return pairs


@dataclass(frozen=True)
class _Segments:
    slave_index: torch.Tensor
    slave_weight: torch.Tensor
    master_index: torch.Tensor
    master_weight: torch.Tensor
    normal: torch.Tensor
    weight: torch.Tensor
    point: torch.Tensor
    slave_face: torch.Tensor
    master_face: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.weight.numel())


def _empty_segments(dtype, s_width, m_width):
    zeros = torch.zeros((0, 3), dtype=dtype)
    return _Segments(
        torch.zeros((0, s_width), dtype=torch.long),
        torch.zeros((0, s_width), dtype=dtype),
        torch.zeros((0, m_width), dtype=torch.long),
        torch.zeros((0, m_width), dtype=dtype),
        zeros,
        torch.zeros(0, dtype=dtype),
        zeros,
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
    )


def _build_segments(slave_coords, slave_faces, master_coords, master_faces, search_distance):
    """Clip every slave facet against its master facets (frozen topology).

    A master QUAD4 is clipped by its four edges and integrated with the
    bilinear shape functions (no internal triangulation), so the segment rule of
    a uniform traction field is symmetric; a master TRI3 is clipped by its three
    edges.  The master normal follows the right-hand rule of the connectivity.
    """
    dtype = slave_coords.dtype
    s_width = max(int(len(f)) for f in slave_faces)
    m_width = max(int(len(f)) for f in master_faces)
    records = dict(
        slave_index=[],
        slave_weight=[],
        master_index=[],
        master_weight=[],
        normal=[],
        weight=[],
        point=[],
        slave_face=[],
        master_face=[],
    )
    eps = 1e-13
    for si, sj in _candidate_pairs(
        slave_coords, slave_faces, master_coords, master_faces, search_distance
    ):
        snodes = [int(v) for v in slave_faces[si]]
        mnodes = [int(v) for v in master_faces[sj]]
        spts = [slave_coords[i] for i in snodes]
        mpts = [master_coords[i] for i in mnodes]
        a = mpts[0]
        edge = mpts[1] - a
        length = float(torch.linalg.vector_norm(edge))
        if length <= eps:
            raise ValueError("degenerate master facet")
        e1v = edge / length
        nrm = torch.linalg.cross(edge, mpts[2] - a)
        twice_area = float(torch.linalg.vector_norm(nrm))
        if twice_area <= eps:
            raise ValueError("degenerate master facet")
        n = nrm / twice_area
        e2v = torch.linalg.cross(n, e1v)
        m2 = [(float((p - a) @ e1v), float((p - a) @ e2v)) for p in mpts]
        orientation = 1.0 if _polygon_area(m2) > 0.0 else -1.0
        poly = [(float((p - a) @ e1v), float((p - a) @ e2v)) for p in spts]
        if _polygon_area(poly) < 0.0:
            poly = poly[::-1]
        clip = poly
        for k in range(len(mnodes)):
            p2, q2 = m2[k], m2[(k + 1) % len(mnodes)]
            if orientation < 0.0:
                p2, q2 = q2, p2
            clip = _clip_halfplane(clip, p2, q2)
            if len(clip) < 3:
                break
        if len(clip) < 3:
            continue
        for t in range(1, len(clip) - 1):
            v0, v1, v2 = clip[0], clip[t], clip[t + 1]
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]))
            if area <= eps:
                continue
            for l1, l2 in _TRI_RULE:
                l0 = 1.0 - l1 - l2
                x2 = (l0 * v0[0] + l1 * v1[0] + l2 * v2[0], l0 * v0[1] + l1 * v1[1] + l2 * v2[1])
                point = a + x2[0] * e1v + x2[1] * e2v
                mw = _facet_weights(mpts, point)
                sw = _facet_weights(spts, point)
                records["slave_index"].append(snodes + [snodes[0]] * (s_width - len(snodes)))
                records["slave_weight"].append(sw + [0.0] * (s_width - len(sw)))
                records["master_index"].append(mnodes + [mnodes[0]] * (m_width - len(mnodes)))
                records["master_weight"].append(mw + [0.0] * (m_width - len(mw)))
                records["normal"].append([float(x) for x in n])
                records["weight"].append(area / 3.0)
                records["point"].append([float(x) for x in point])
                records["slave_face"].append(si)
                records["master_face"].append(int(sj))
    if not records["weight"]:
        return _empty_segments(dtype, s_width, m_width)
    return _Segments(
        torch.tensor(records["slave_index"], dtype=torch.long),
        torch.tensor(records["slave_weight"], dtype=dtype),
        torch.tensor(records["master_index"], dtype=torch.long),
        torch.tensor(records["master_weight"], dtype=dtype),
        torch.tensor(records["normal"], dtype=dtype),
        torch.tensor(records["weight"], dtype=dtype),
        torch.tensor(records["point"], dtype=dtype),
        torch.tensor(records["slave_face"], dtype=torch.long),
        torch.tensor(records["master_face"], dtype=torch.long),
    )


def _gaps_and_potential(seg, slave_coords, master_coords, penalty):
    if seg.count == 0:
        zero = slave_coords.new_zeros(())
        return zero, slave_coords.new_zeros(0), slave_coords.new_zeros(0)
    xs = (seg.slave_weight[..., None] * slave_coords[seg.slave_index]).sum(1)
    xm = (seg.master_weight[..., None] * master_coords[seg.master_index]).sum(1)
    gaps = ((xs - xm) * seg.normal).sum(1)
    penetration = torch.clamp(-gaps, min=0.0)
    energy = (0.5 * penalty * penetration**2 * seg.weight).sum()
    return energy, gaps, penetration


def _validate_surfaces(slave_reference, slave_faces, master_reference, master_faces):
    for name, coords in (("slave", slave_reference), ("master", master_reference)):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{name} surface coordinates must have shape (n,3)")
    for name, faces in (("slave", slave_faces), ("master", master_faces)):
        if faces.ndim != 2 or faces.shape[1] not in (3, 4) or faces.dtype != torch.long:
            raise ValueError(f"invalid {name} connectivity")
        if int(faces.max()) >= (
            slave_reference.shape[0] if name == "slave" else master_reference.shape[0]
        ):
            raise ValueError(f"{name} connectivity refers to undefined nodes")


def sample_mortar_segment_contact(
    slave_coords: torch.Tensor,
    slave_faces: torch.Tensor,
    master_coords: torch.Tensor,
    master_faces: torch.Tensor,
    *,
    normal_penalty: float,
    search_distance: float | None = None,
    slave_reference: torch.Tensor | None = None,
    master_reference: torch.Tensor | None = None,
) -> MortarSegmentSamples:
    """Integrate contact tractions at the mortar segment quadrature points.

    Passing the *reference* surfaces builds the segment topology, normals and
    area measure from them while the gap is evaluated on the supplied current
    coordinates.  That is the measure used by
    :func:`assemble_mortar_segment_contact` (a total-Lagrangian, reference-area
    convention), so sampled tractions and assembled nodal forces agree to
    machine precision; without references the current configuration is used for
    both (a purely current-area diagnostic).
    """
    if normal_penalty <= 0:
        raise ValueError("normal_penalty must be positive")
    _validate_surfaces(slave_coords, slave_faces, master_coords, master_faces)
    sref = slave_coords if slave_reference is None else slave_reference
    mref = master_coords if master_reference is None else master_reference
    if sref.shape != slave_coords.shape or mref.shape != master_coords.shape:
        raise ValueError("reference coordinates must match the current coordinates")
    seg = _build_segments(sref, slave_faces, mref, master_faces, search_distance)
    _, gaps, penetration = _gaps_and_potential(
        seg, slave_coords, master_coords, float(normal_penalty)
    )
    traction = float(normal_penalty) * penetration
    total = (traction * seg.weight).sum() if seg.count else slave_coords.new_zeros(())
    closed = (penetration > 0).to(seg.weight.dtype) if seg.count else seg.weight
    return MortarSegmentSamples(
        seg.point,
        seg.normal,
        gaps,
        traction,
        seg.weight,
        seg.slave_face,
        seg.master_face,
        total,
        (seg.weight.sum() if seg.count else slave_coords.new_zeros(())),
        ((closed * seg.weight).sum() if seg.count else slave_coords.new_zeros(())),
        (penetration.max() if seg.count else slave_coords.new_zeros(())),
    )


def assemble_mortar_segment_contact(
    slave_reference: torch.Tensor,
    slave_faces: torch.Tensor,
    master_reference: torch.Tensor,
    master_faces: torch.Tensor,
    slave_displacement: torch.Tensor,
    master_displacement: torch.Tensor,
    *,
    normal_penalty: float,
    tangent: bool = True,
    search_distance: float | None = None,
    check_initial_penetration: bool = True,
    initial_penetration_tol: float = 1e-10,
) -> MortarSegmentAssembly:
    """Assemble the frictionless mortar segment residual and tangent.

    ``normal_penalty`` must be strictly positive.  When
    ``check_initial_penetration`` is set the reference configuration may not be
    interpenetrating by more than ``initial_penetration_tol``; an already
    closed/overlapping input (typical self-contact modelling error) is rejected
    instead of being silently penalised.

    The master facet normal follows the right-hand rule of the master
    connectivity and must point towards the slave surface, so that
    ``gap = (x_slave - x_master) . n_master`` is positive when the surfaces are
    separated.
    """
    if normal_penalty <= 0:
        raise ValueError("normal_penalty must be positive")
    if (
        slave_reference.shape != slave_displacement.shape
        or master_reference.shape != master_displacement.shape
    ):
        raise ValueError("reference coordinates and displacements must match")
    if (
        slave_faces.device != slave_reference.device
        or master_faces.device != master_reference.device
    ):
        raise ValueError("connectivity and coordinates must share a device")
    _validate_surfaces(slave_reference, slave_faces, master_reference, master_faces)
    seg = _build_segments(
        slave_reference, slave_faces, master_reference, master_faces, search_distance
    )
    if seg.count and search_distance is not None:
        _, gaps, _ = _gaps_and_potential(
            seg, slave_reference, master_reference, float(normal_penalty)
        )
        if float(gaps.max()) > search_distance:
            raise ValueError("search_distance is smaller than a detected facet gap")
    if seg.count and check_initial_penetration:
        _, gaps, _ = _gaps_and_potential(
            seg, slave_reference, master_reference, float(normal_penalty)
        )
        worst = float(gaps.min())
        if worst < -abs(initial_penetration_tol):
            raise ValueError("initial interpenetration exceeds tolerance")
    ns = slave_reference.shape[0]
    q = (
        torch.cat((slave_displacement.reshape(-1), master_displacement.reshape(-1)))
        .detach()
        .requires_grad_(True)
    )

    def potential(dofs: torch.Tensor) -> torch.Tensor:
        slave = slave_reference + dofs[: 3 * ns].reshape_as(slave_reference)
        master = master_reference + dofs[3 * ns :].reshape_as(master_reference)
        energy, _, _ = _gaps_and_potential(seg, slave, master, float(normal_penalty))
        return energy

    energy = potential(q)
    residual = torch.autograd.grad(energy, q, create_graph=tangent)[0]
    matrix = (
        torch.autograd.functional.hessian(potential, q)
        if tangent
        else q.new_zeros((q.numel(), q.numel()))
    )
    current_slave = slave_reference + slave_displacement
    current_master = master_reference + master_displacement
    _, gaps, penetration = _gaps_and_potential(
        seg, current_slave, current_master, float(normal_penalty)
    )
    physical = -residual.detach()
    area = seg.weight.sum() if seg.count else slave_reference.new_zeros(())
    worst = penetration.max() if seg.count else slave_reference.new_zeros(())
    return MortarSegmentAssembly(
        residual.detach(),
        matrix.detach(),
        physical[: 3 * ns].reshape_as(slave_reference),
        physical[3 * ns :].reshape_as(master_reference),
        seg.count,
        area,
        worst,
        energy.detach(),
    )
