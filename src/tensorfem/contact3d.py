"""Finite-sliding node-to-facet contact in three dimensions.

This is a local contact kernel for oriented TRI3 and planar/warped QUAD4
facets.  It deliberately does not claim mortar contact, continuous Hertz
contact, or a global contact Newton solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class FacetProjection:
    point: torch.Tensor
    normal: torch.Tensor
    weights: torch.Tensor
    gap: torch.Tensor
    distance: torch.Tensor
    face: int
    triangle: int


@dataclass(frozen=True)
class Contact3DState:
    normal_multiplier: torch.Tensor
    elastic_slip: torch.Tensor
    dissipated_energy: torch.Tensor
    projection: torch.Tensor
    face: int
    active: bool
    sticking: bool


@dataclass(frozen=True)
class Contact3DUpdate:
    slave_force: torch.Tensor
    master_forces: torch.Tensor
    normal_force: torch.Tensor
    tangential_force: torch.Tensor
    projection: FacetProjection
    state: Contact3DState
    stored_energy: torch.Tensor
    dissipation_increment: torch.Tensor


def _closest_triangle(p: torch.Tensor, tri: torch.Tensor):
    """Closest point and barycentric weights (Ericson region algorithm)."""
    a, b, c = tri
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = torch.dot(ab, ap), torch.dot(ac, ap)
    if bool(d1 <= 0) and bool(d2 <= 0): return a, p.new_tensor([1., 0., 0.])
    bp = p-b; d3, d4 = torch.dot(ab, bp), torch.dot(ac, bp)
    if bool(d3 >= 0) and bool(d4 <= d3): return b, p.new_tensor([0., 1., 0.])
    vc = d1*d4-d3*d2
    if bool(vc <= 0) and bool(d1 >= 0) and bool(d3 <= 0):
        v=d1/(d1-d3); return a+v*ab, torch.stack((1-v, v, v*0))
    cp=p-c; d5, d6=torch.dot(ab, cp), torch.dot(ac, cp)
    if bool(d6 >= 0) and bool(d5 <= d6): return c, p.new_tensor([0., 0., 1.])
    vb=d5*d2-d1*d6
    if bool(vb <= 0) and bool(d2 >= 0) and bool(d6 <= 0):
        w=d2/(d2-d6); return a+w*ac, torch.stack((1-w, w*0, w))
    va=d3*d6-d5*d4
    if bool(va <= 0) and bool(d4-d3 >= 0) and bool(d5-d6 >= 0):
        w=(d4-d3)/((d4-d3)+(d5-d6)); return b+w*(c-b), torch.stack((w*0, 1-w, w))
    den=1/(va+vb+vc); v=vb*den; w=vc*den
    return a+ab*v+ac*w, torch.stack((1-v-w, v, w))


def _triangles(n: int):
    if n == 3: return ((0, 1, 2),)
    if n == 4: return ((0, 1, 2), (0, 2, 3))
    raise ValueError("facets must be TRI3 or QUAD4")


def project_point_to_facets(point: torch.Tensor, vertices: torch.Tensor,
                            faces: torch.Tensor, *,
                            excluded_faces: torch.Tensor | None = None) -> FacetProjection:
    """Globally search current facet coordinates for the closest projection."""
    if point.shape != (3,) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("point/vertices must have shapes (3,) and (n,3)")
    if faces.ndim != 2 or faces.shape[1] not in (3, 4) or faces.dtype != torch.long:
        raise ValueError("faces must be a long TRI3 or QUAD4 connectivity array")
    excluded = set([] if excluded_faces is None else excluded_faces.tolist())
    best = None
    for fi in range(faces.shape[0]):
        if fi in excluded: continue
        fv = vertices[faces[fi]]
        for ti, ids in enumerate(_triangles(faces.shape[1])):
            tri=fv[list(ids)]; cross=torch.linalg.cross(tri[1]-tri[0], tri[2]-tri[0])
            norm=torch.linalg.vector_norm(cross)
            if bool(norm <= torch.finfo(vertices.dtype).eps):
                raise ValueError("facet contains a degenerate triangle")
            q, bary=_closest_triangle(point, tri); dist=torch.linalg.vector_norm(point-q)
            if best is None or bool(dist < best[0]):
                weights=point.new_zeros(faces.shape[1]); weights[list(ids)]=bary
                n=cross/norm
                best=(dist, FacetProjection(q,n,weights,torch.dot(point-q,n),dist,fi,ti))
    if best is None: raise ValueError("no contact candidate remains after filtering")
    return best[1]


def initial_contact_state(point: torch.Tensor, vertices: torch.Tensor,
                          faces: torch.Tensor, **kwargs) -> Contact3DState:
    pr=project_point_to_facets(point,vertices,faces,**kwargs); z=point.new_zeros(())
    return Contact3DState(z,torch.zeros_like(point),z,pr.point,pr.face,False,True)


def update_node_facet_contact(
    point: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor,
    state: Contact3DState, *, normal_penalty: float,
    tangential_penalty: float, friction: float,
    normal_method: Literal["penalty", "augmented_lagrangian"] = "penalty",
    relative_increment: torch.Tensor | None = None,
    excluded_faces: torch.Tensor | None = None,
) -> Contact3DUpdate:
    """Return slave/master nodal forces and committed local history."""
    pr=project_point_to_facets(point,vertices,faces,excluded_faces=excluded_faces)
    kn=point.new_tensor(normal_penalty); kt=point.new_tensor(tangential_penalty); mu=point.new_tensor(friction)
    if bool(kn <= 0) or bool(kt <= 0) or bool(mu < 0): raise ValueError("invalid contact parameters")
    if normal_method not in ("penalty","augmented_lagrangian"): raise ValueError("invalid normal method")
    old=state.normal_multiplier if normal_method == "augmented_lagrangian" else point.new_zeros(())
    lam=torch.clamp(old-kn*pr.gap,min=0.)
    zero=point.new_zeros(())
    if bool(lam <= torch.finfo(point.dtype).eps):
        ns=Contact3DState(zero,torch.zeros_like(point),state.dissipated_energy,pr.point,pr.face,False,True)
        return Contact3DUpdate(torch.zeros_like(point),torch.zeros_like(vertices),zero,torch.zeros_like(point),pr,ns,zero,zero)
    dx=(pr.point-state.projection) if relative_increment is None else relative_increment
    ds=dx-torch.dot(dx,pr.normal)*pr.normal
    elastic_old=state.elastic_slip-torch.dot(state.elastic_slip,pr.normal)*pr.normal
    trial=elastic_old+ds; tau_trial=-kt*trial; mag=torch.linalg.vector_norm(tau_trial); limit=mu*lam
    sticking=bool(mag <= limit) or bool(mag <= torch.finfo(point.dtype).eps)
    if sticking: tau=tau_trial; elastic=trial; plastic=torch.zeros_like(point)
    else:
        tau=-limit*trial/torch.linalg.vector_norm(trial); elastic=-tau/kt; plastic=trial-elastic
    diss=torch.abs(torch.dot(tau,plastic)); slave=lam*pr.normal+tau
    mf=torch.zeros_like(vertices); mf[faces[pr.face]]=-pr.weights[:,None]*slave
    stored=.5*(lam-old)**2/kn+.5*kt*torch.dot(elastic,elastic)
    ns=Contact3DState(lam,elastic,state.dissipated_energy+diss,pr.point,pr.face,True,sticking)
    return Contact3DUpdate(slave,mf,lam,tau,pr,ns,stored,diss)


def update_contact_nodes(points: torch.Tensor, vertices: torch.Tensor,
                         faces: torch.Tensor, states: list[Contact3DState], **kwargs):
    """Update multiple slave nodes and assemble action/reaction forces."""
    if len(states) != points.shape[0]: raise ValueError("one state is required per slave node")
    updates=[update_node_facet_contact(p,vertices,faces,s,**kwargs) for p,s in zip(points,states)]
    return updates, torch.stack([u.slave_force for u in updates]), sum((u.master_forces for u in updates),torch.zeros_like(vertices))
