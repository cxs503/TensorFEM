"""Conservative vertex--moving-triangle CCD and transactional pair history."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .contact3d import _closest_triangle


@dataclass(frozen=True)
class CCDEvent:
    vertex: int
    face: int
    toi: float
    point: torch.Tensor
    projection: torch.Tensor
    barycentric: torch.Tensor
    normal: torch.Tensor


@dataclass(frozen=True)
class DynamicPairHistory:
    vertex: int
    face: int
    age: float
    tangential_slip: torch.Tensor
    last_toi: float


@dataclass(frozen=True)
class DynamicSelfContactState:
    pairs: tuple[DynamicPairHistory,...]


@dataclass(frozen=True)
class DynamicSelfContactUpdate:
    events: tuple[CCDEvent,...]
    accepted_fraction: float
    impact_positions: torch.Tensor
    impulses: torch.Tensor
    state: DynamicSelfContactState


def vertex_triangle_ccd(p0,p1,t0,t1,*,thickness=0.,time_tolerance=1e-10,
                        distance_tolerance=1e-10,max_iterations=200):
    """Conservative advancement for linear vertex and triangle trajectories."""
    if p0.shape!=(3,) or t0.shape!=(3,3) or min(thickness,time_tolerance,distance_tolerance)<0:
        raise ValueError("invalid CCD input")
    for tri in (t0,t1):
        if bool(torch.linalg.vector_norm(torch.linalg.cross(tri[1]-tri[0],tri[2]-tri[0]))
                <=torch.finfo(p0.dtype).eps):
            raise ValueError("triangle is degenerate at a step endpoint")
    vp=p1-p0; vt=t1-t0
    speed=float(torch.linalg.vector_norm(vp)+torch.linalg.vector_norm(vt,dim=1).max())
    scale=max(float(torch.linalg.vector_norm(torch.cat((p0,t0.flatten())))),1.)
    spatial=max(distance_tolerance*scale,torch.finfo(p0.dtype).eps*100)
    time=0.
    for _ in range(max_iterations):
        p=p0+time*vp; tri=t0+time*vt; q,bary=_closest_triangle(p,tri)
        delta=p-q; distance=float(torch.linalg.vector_norm(delta))
        if distance<=thickness+spatial:
            cross=torch.linalg.cross(tri[1]-tri[0],tri[2]-tri[0]); cn=torch.linalg.vector_norm(cross)
            if bool(cn<=torch.finfo(p.dtype).eps): raise ValueError("triangle degenerates during CCD")
            normal=delta/torch.linalg.vector_norm(delta) if distance>spatial*.1 else cross/cn
            return time,p,q,bary,normal
        if speed<=spatial: return None
        advance=.9*(distance-thickness)/speed
        if advance<time_tolerance: advance=time_tolerance
        time += advance
        if time>1+time_tolerance: return None
        time=min(time,1.)
    raise RuntimeError("CCD conservative advancement did not converge")


def swept_vertex_face_candidates(start,end,faces,*,thickness=0.):
    """Swept-AABB broad phase, excluding incident vertex/face pairs."""
    lo=torch.minimum(start,end); hi=torch.maximum(start,end)
    pairs=[]
    for vi in range(len(start)):
        vlo=lo[vi]-thickness; vhi=hi[vi]+thickness
        for fi,face in enumerate(faces):
            if vi in face.tolist(): continue
            flo=lo[face].min(0).values-thickness; fhi=hi[face].max(0).values+thickness
            if bool(torch.all(torch.maximum(vlo,flo)<=torch.minimum(vhi,fhi))): pairs.append((vi,fi))
    return tuple(pairs)


def earliest_self_contact(start,end,faces,*,thickness=0.,time_tolerance=1e-10):
    """Return all vertex/face events at the earliest detected time."""
    found=[]
    for vi,fi in swept_vertex_face_candidates(start,end,faces,thickness=thickness):
        hit=vertex_triangle_ccd(start[vi],end[vi],start[faces[fi]],end[faces[fi]],
                                thickness=thickness,time_tolerance=time_tolerance)
        if hit is not None:
            toi,p,q,b,n=hit; found.append(CCDEvent(vi,fi,toi,p,q,b,n))
    if not found: return ()
    first=min(e.toi for e in found)
    return tuple(e for e in found if abs(e.toi-first)<=max(time_tolerance,1e-8))


def update_dynamic_self_contact(start,end,faces,state,*,dt: float,masses=None,
                                thickness=0.,time_tolerance=1e-10):
    """Detect earliest impact, assemble conservative impulses and trial history.

    The accepted configuration is clipped to the earliest TOI. The caller may
    commit ``result.state`` or discard it and retain ``state`` on step rollback.
    """
    if dt<=0 or start.shape!=end.shape or start.ndim!=2 or start.shape[1]!=3:
        raise ValueError("invalid dynamic contact step")
    events=earliest_self_contact(start,end,faces,thickness=thickness,time_tolerance=time_tolerance)
    if not events:
        return DynamicSelfContactUpdate((),1.,end.clone(),torch.zeros_like(start),DynamicSelfContactState(()))
    toi=min(e.toi for e in events); impact=start+toi*(end-start); velocity=(end-start)/dt
    mass=torch.ones(len(start),dtype=start.dtype,device=start.device) if masses is None else masses
    if mass.shape!=(len(start),) or bool(torch.any(mass<=0)): raise ValueError("masses must be positive nodal values")
    impulses=torch.zeros_like(start); old={(p.vertex,p.face):p for p in state.pairs}; histories=[]
    for event in events:
        face=faces[event.face]; vm=event.barycentric@velocity[face]
        rel=velocity[event.vertex]-vm; vn=torch.dot(rel,event.normal)
        # Only closing motion receives an impulse. Restitution is zero.
        magnitude=torch.clamp(-vn,min=0.)/(1/mass[event.vertex]+torch.sum(event.barycentric**2/mass[face]))
        impulse=magnitude*event.normal
        impulses[event.vertex]+=impulse; impulses[face]-=event.barycentric[:,None]*impulse
        key=(event.vertex,event.face); previous=old.get(key)
        age=(0. if previous is None else previous.age)+toi*dt
        tangential=rel-vn*event.normal
        prior=torch.zeros(3,dtype=start.dtype,device=start.device) if previous is None else previous.tangential_slip
        histories.append(DynamicPairHistory(*key,age,prior+tangential*toi*dt,event.toi))
    return DynamicSelfContactUpdate(events,toi,impact,impulses,DynamicSelfContactState(tuple(histories)))
