"""Edge--edge CCD, unified events, spatial hash and Coulomb impulses."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .self_contact_ccd import CCDEvent,earliest_self_contact


class AmbiguousCoplanarContactError(RuntimeError):
    """Raised for collinear/coplanar contact without a unique contact normal."""


@dataclass(frozen=True)
class EdgeEdgeEvent:
    edge_a: int
    edge_b: int
    toi: float
    coordinate_a: torch.Tensor
    coordinate_b: torch.Tensor
    point_a: torch.Tensor
    point_b: torch.Tensor
    normal: torch.Tensor


@dataclass(frozen=True)
class UnifiedEvent:
    kind: str
    toi: float
    side_a: torch.Tensor
    weights_a: torch.Tensor
    side_b: torch.Tensor
    weights_b: torch.Tensor
    normal: torch.Tensor


@dataclass(frozen=True)
class ImpactResult:
    impulses: torch.Tensor
    normal_impulse: torch.Tensor
    friction_impulse: torch.Tensor
    kinetic_before: torch.Tensor
    kinetic_after: torch.Tensor


def _closest_segments(a0,a1,b0,b1):
    """Closest points on two segments and their local coordinates."""
    u=a1-a0; v=b1-b0; w=a0-b0
    aa=torch.dot(u,u); bb=torch.dot(u,v); cc=torch.dot(v,v); dd=torch.dot(u,w); ee=torch.dot(v,w)
    eps=torch.finfo(a0.dtype).eps*100
    if bool(aa<=eps) or bool(cc<=eps): raise ValueError("degenerate collision edge")
    den=aa*cc-bb*bb
    s=a0.new_zeros(()) if bool(den<=eps*aa*cc) else torch.clamp((bb*ee-cc*dd)/den,0.,1.)
    t=torch.clamp((bb*s+ee)/cc,0.,1.)
    s=torch.clamp((bb*t-dd)/aa,0.,1.)
    return a0+s*u,b0+t*v,s,t


def edge_edge_ccd(a_start,a_end,b_start,b_end,*,thickness=0.,time_tolerance=1e-10,
                  distance_tolerance=1e-10,max_iterations=200):
    """Conservative CCD for two linearly moving line segments."""
    if a_start.shape!=(2,3) or b_start.shape!=(2,3) or min(thickness,time_tolerance,distance_tolerance)<0:
        raise ValueError("invalid edge CCD input")
    va=a_end-a_start; vb=b_end-b_start
    speed=float(torch.linalg.vector_norm(va,dim=1).max()+torch.linalg.vector_norm(vb,dim=1).max())
    scale=max(float(torch.linalg.vector_norm(torch.cat((a_start.flatten(),b_start.flatten())))),1.)
    spatial=max(distance_tolerance*scale,torch.finfo(a_start.dtype).eps*100)
    time=0.
    for _ in range(max_iterations):
        a=a_start+time*va; b=b_start+time*vb
        pa,pb,s,t=_closest_segments(a[0],a[1],b[0],b[1]); delta=pa-pb; distance=float(torch.linalg.vector_norm(delta))
        if distance<=thickness+spatial:
            if distance>spatial*.1: normal=delta/torch.linalg.vector_norm(delta)
            else:
                cross=torch.linalg.cross(a[1]-a[0],b[1]-b[0]); cn=torch.linalg.vector_norm(cross)
                if bool(cn<=spatial):
                    raise AmbiguousCoplanarContactError("parallel/collinear edge contact has no unique normal")
                normal=cross/cn
                rel=(torch.stack((1-s,s))@va)-(torch.stack((1-t,t))@vb)
                if bool(torch.dot(rel,normal)>0): normal=-normal
            return EdgeEdgeEvent(-1,-1,time,s,t,pa,pb,normal)
        if speed<=spatial: return None
        time += max(.9*(distance-thickness)/speed,time_tolerance)
        if time>1+time_tolerance: return None
        time=min(time,1.)
    raise RuntimeError("edge CCD conservative advancement did not converge")


def spatial_hash_edge_pairs(start,end,edges,*,cell_size: float,thickness=0.):
    """Deterministic swept-AABB spatial hash for nonincident edge pairs."""
    if cell_size<=0: raise ValueError("cell_size must be positive")
    buckets={}
    for ei,edge in enumerate(edges):
        x=torch.cat((start[edge],end[edge]),dim=0)
        lo=torch.floor((x.min(0).values-thickness)/cell_size).to(torch.long)
        hi=torch.floor((x.max(0).values+thickness)/cell_size).to(torch.long)
        for i in range(int(lo[0]),int(hi[0])+1):
            for j in range(int(lo[1]),int(hi[1])+1):
                for k in range(int(lo[2]),int(hi[2])+1): buckets.setdefault((i,j,k),[]).append(ei)
    pairs=set()
    for key in sorted(buckets):
        ids=sorted(set(buckets[key]))
        for ii,a in enumerate(ids):
            for b in ids[ii+1:]:
                if set(edges[a].tolist())&set(edges[b].tolist()): continue
                pairs.add((a,b))
    return tuple(sorted(pairs))


def earliest_edge_contacts(start,end,edges,*,cell_size=1.,thickness=0.,time_tolerance=1e-10):
    found=[]
    for ea,eb in spatial_hash_edge_pairs(start,end,edges,cell_size=cell_size,thickness=thickness):
        hit=edge_edge_ccd(start[edges[ea]],end[edges[ea]],start[edges[eb]],end[edges[eb]],
                          thickness=thickness,time_tolerance=time_tolerance)
        if hit is not None:
            found.append(EdgeEdgeEvent(ea,eb,hit.toi,hit.coordinate_a,hit.coordinate_b,
                                       hit.point_a,hit.point_b,hit.normal))
    if not found:return ()
    first=min(x.toi for x in found)
    return tuple(x for x in found if abs(x.toi-first)<=max(time_tolerance,1e-8))


def earliest_unified_contacts(start,end,faces,edges,*,cell_size=1.,thickness=0.,time_tolerance=1e-10):
    events=[]
    for e in earliest_self_contact(start,end,faces,thickness=thickness,time_tolerance=time_tolerance):
        events.append(UnifiedEvent("vertex-face",e.toi,start.new_tensor([e.vertex],dtype=torch.long),
            start.new_ones(1),faces[e.face],e.barycentric,e.normal))
    for e in earliest_edge_contacts(start,end,edges,cell_size=cell_size,thickness=thickness,time_tolerance=time_tolerance):
        events.append(UnifiedEvent("edge-edge",e.toi,edges[e.edge_a],torch.stack((1-e.coordinate_a,e.coordinate_a)),
            edges[e.edge_b],torch.stack((1-e.coordinate_b,e.coordinate_b)),e.normal))
    if not events:return ()
    first=min(e.toi for e in events)
    # Deterministic de-duplication by contact stencil and kind.
    unique={}
    for e in events:
        if abs(e.toi-first)>max(time_tolerance,1e-8):continue
        key=(e.kind,tuple(e.side_a.tolist()),tuple(e.side_b.tolist()))
        unique[key]=e
    return tuple(unique[k] for k in sorted(unique))


def coulomb_impact(event: UnifiedEvent,velocities,masses,*,friction: float,restitution=0.):
    """Apply one mass-lumped Coulomb impact impulse; kinetic energy is audited."""
    if friction<0 or not 0<=restitution<=1 or bool(torch.any(masses<=0)): raise ValueError("invalid impact data")
    va=event.weights_a@velocities[event.side_a]; vb=event.weights_b@velocities[event.side_b]
    rel=va-vb; vn=torch.dot(rel,event.normal)
    den=torch.sum(event.weights_a**2/masses[event.side_a])+torch.sum(event.weights_b**2/masses[event.side_b])
    jn=torch.clamp(-(1+restitution)*vn,min=0.)/den
    vt=rel-vn*event.normal; jt_trial=-vt/den; jt_norm=torch.linalg.vector_norm(jt_trial)
    limit=friction*jn
    jt=jt_trial if bool(jt_norm<=limit) or bool(jt_norm<=torch.finfo(velocities.dtype).eps) else jt_trial*(limit/jt_norm)
    impulse=jn*event.normal+jt; assembled=torch.zeros_like(velocities)
    assembled[event.side_a]+=event.weights_a[:,None]*impulse
    assembled[event.side_b]-=event.weights_b[:,None]*impulse
    after=velocities+assembled/masses[:,None]
    kb=.5*torch.sum(masses[:,None]*velocities**2); ka=.5*torch.sum(masses[:,None]*after**2)
    if bool(ka>kb*(1+1e-12)+1e-12): raise RuntimeError("impact update increased kinetic energy")
    return ImpactResult(assembled,jn*event.normal,jt,kb,ka)
