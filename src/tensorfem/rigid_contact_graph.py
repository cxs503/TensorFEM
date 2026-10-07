"""Rigid-body contact islands with rotational effective mass and warm PGS."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RigidContact:
    key: tuple
    body_a: int
    body_b: int                 # -1 denotes fixed environment
    point: torch.Tensor
    normal: torch.Tensor        # points from B toward A


@dataclass(frozen=True)
class RigidImpulseHistory:
    key: tuple
    normal_impulse: torch.Tensor
    friction_impulse: torch.Tensor


@dataclass(frozen=True)
class RigidContactState:
    histories: tuple[RigidImpulseHistory,...]


@dataclass(frozen=True)
class RigidContactResult:
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    state: RigidContactState
    islands: tuple[tuple[int,...],...]
    iterations: int
    complementarity_residual: torch.Tensor
    cone_residual: torch.Tensor
    kinetic_before: torch.Tensor
    kinetic_after: torch.Tensor
    converged: bool


def contact_islands(n_bodies: int,contacts):
    """Deterministic union-find decomposition; fixed ground joins no islands."""
    parent=list(range(n_bodies))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            if ra>rb:ra,rb=rb,ra
            parent[rb]=ra
    active=set()
    for c in contacts:
        if not 0<=c.body_a<n_bodies or c.body_b>=n_bodies: raise ValueError("invalid body index")
        active.add(c.body_a)
        if c.body_b>=0: active.add(c.body_b); union(c.body_a,c.body_b)
    groups={}
    for b in sorted(active):groups.setdefault(find(b),[]).append(b)
    return tuple(tuple(v) for _,v in sorted(groups.items()))


def _skew(r):
    z=r.new_zeros(())
    return torch.stack((torch.stack((z,-r[2],r[1])),torch.stack((r[2],z,-r[0])),torch.stack((-r[1],r[0],z))))


def _point_velocity(body,point,com,v,w):
    if body<0:return torch.zeros(3,dtype=v.dtype,device=v.device)
    return v[body]+torch.linalg.cross(w[body],point-com[body])


def _delassus(c,com,mass,inertia_inv):
    eye=torch.eye(3,dtype=com.dtype,device=com.device); W=eye/mass[c.body_a]
    ra=c.point-com[c.body_a]; S=_skew(ra); W=W-S@inertia_inv[c.body_a]@S
    if c.body_b>=0:
        rb=c.point-com[c.body_b]; S=_skew(rb); W=W+eye/mass[c.body_b]-S@inertia_inv[c.body_b]@S
    return W


def _apply(c,impulse,com,mass,inertia_inv,v,w):
    ra=c.point-com[c.body_a]; v[c.body_a]+=impulse/mass[c.body_a]
    w[c.body_a]+=inertia_inv[c.body_a]@torch.linalg.cross(ra,impulse)
    if c.body_b>=0:
        rb=c.point-com[c.body_b]; v[c.body_b]-=impulse/mass[c.body_b]
        w[c.body_b]-=inertia_inv[c.body_b]@torch.linalg.cross(rb,impulse)


def _tangent_basis(n):
    axis=n.new_tensor([1.,0.,0.]) if abs(float(n[0]))<.8 else n.new_tensor([0.,1.,0.])
    t1=torch.linalg.cross(n,axis); t1=t1/torch.linalg.vector_norm(t1)
    return torch.stack((t1,torch.linalg.cross(n,t1)),dim=1)


def solve_rigid_contact_graph(com,mass,inertia_world,linear_velocity,angular_velocity,
                              contacts,state,*,friction: float,restitution=0.,
                              tolerance=1e-10,max_iterations=200):
    """Solve all rigid contacts island-by-island using deterministic warm PGS."""
    n=len(com)
    if mass.shape!=(n,) or inertia_world.shape!=(n,3,3) or bool(torch.any(mass<=0)):
        raise ValueError("invalid rigid body properties")
    if friction<0 or not 0<=restitution<=1 or tolerance<=0 or max_iterations<1:
        raise ValueError("invalid contact solver parameter")
    ordered=tuple(sorted(contacts,key=lambda c:c.key))
    if len({c.key for c in ordered})!=len(ordered):raise ValueError("duplicate contact key")
    for c in ordered:
        nn=torch.linalg.vector_norm(c.normal)
        if bool(abs(nn-1)>1e-8):raise ValueError("contact normal must be unit length")
    invI=torch.linalg.inv(inertia_world); v=linear_velocity.clone(); w=angular_velocity.clone()
    initial={(c.key):torch.dot(_point_velocity(c.body_a,c.point,com,v,w)-_point_velocity(c.body_b,c.point,com,v,w),c.normal) for c in ordered}
    old={h.key:h for h in state.histories}; current={}
    for c in ordered:
        h=old.get(c.key); z=com.new_zeros(()); jn=z if h is None else torch.clamp(h.normal_impulse,min=0.)
        jt=torch.zeros(3,dtype=com.dtype,device=com.device) if h is None else h.friction_impulse.clone()
        mag=torch.linalg.vector_norm(jt); limit=friction*jn
        if bool(mag>limit) and bool(mag>0):jt*=limit/mag
        _apply(c,jn*c.normal+jt,com,mass,invI,v,w); current[c.key]=[jn,jt]
    islands=contact_islands(n,ordered); body_island={b:i for i,g in enumerate(islands) for b in g}
    groups=[[] for _ in islands]
    for c in ordered:groups[body_island[c.body_a]].append(c)
    converged=True; max_used=0
    for group in groups:
        island_done=False
        for iteration in range(1,max_iterations+1):
            maximum=0.
            for c in group:
                W=_delassus(c,com,mass,invI); jn,jt=current[c.key]
                rel=_point_velocity(c.body_a,c.point,com,v,w)-_point_velocity(c.body_b,c.point,com,v,w)
                vn=torch.dot(rel,c.normal); target=-restitution*min(float(initial[c.key]),0.)
                new_jn=torch.clamp(jn-(vn-target)/torch.dot(c.normal,W@c.normal),min=0.); djn=new_jn-jn
                _apply(c,djn*c.normal,com,mass,invI,v,w)
                rel=_point_velocity(c.body_a,c.point,com,v,w)-_point_velocity(c.body_b,c.point,com,v,w)
                T=_tangent_basis(c.normal); vt=T.T@rel; WT=T.T@W@T
                trial2=T.T@jt-torch.linalg.solve(WT,vt); trial=T@trial2
                mag=torch.linalg.vector_norm(trial); limit=friction*new_jn
                new_jt=trial if bool(mag<=limit) or bool(mag==0) else trial*(limit/mag)
                _apply(c,new_jt-jt,com,mass,invI,v,w); current[c.key]=[new_jn,new_jt]
                maximum=max(maximum,float(torch.abs(djn)),float(torch.linalg.vector_norm(new_jt-jt)))
            if maximum<=tolerance:island_done=True;break
        max_used=max(max_used,iteration); converged &= island_done
    comp=com.new_zeros(()); cone=comp.clone(); histories=[]
    for c in ordered:
        jn,jt=current[c.key]; rel=_point_velocity(c.body_a,c.point,com,v,w)-_point_velocity(c.body_b,c.point,com,v,w)
        target=-restitution*min(float(initial[c.key]),0.); gapv=torch.dot(rel,c.normal)-target
        comp=torch.maximum(comp,torch.maximum(torch.clamp(-gapv,min=0.),torch.abs(jn*torch.clamp(gapv,min=0.))))
        cone=torch.maximum(cone,torch.clamp(torch.linalg.vector_norm(jt)-friction*jn,min=0.))
        histories.append(RigidImpulseHistory(c.key,jn,jt))
    kb=.5*torch.sum(mass[:,None]*linear_velocity**2)+.5*sum(torch.dot(angular_velocity[i],inertia_world[i]@angular_velocity[i]) for i in range(n))
    ka=.5*torch.sum(mass[:,None]*v**2)+.5*sum(torch.dot(w[i],inertia_world[i]@w[i]) for i in range(n))
    if bool(ka>kb*(1+1e-9)+1e-11):raise RuntimeError("rigid contact increased kinetic energy")
    return RigidContactResult(v,w,RigidContactState(tuple(histories)),islands,max_used,comp,cone,kb,ka,converged)
