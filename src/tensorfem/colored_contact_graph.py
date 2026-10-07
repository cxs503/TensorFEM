"""Deterministically colored, vectorised frictionless rigid contact PGS."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .rigid_contact_graph import contact_islands


@dataclass(frozen=True)
class ColoredContactData:
    keys: tuple[tuple,...]
    body_a: torch.Tensor
    body_b: torch.Tensor
    points: torch.Tensor
    normals: torch.Tensor
    colors: tuple[torch.Tensor,...]
    islands: tuple[tuple[int,...],...]


@dataclass(frozen=True)
class ColoredContactState:
    keys: tuple[tuple,...]
    normal_impulses: torch.Tensor


@dataclass(frozen=True)
class ColoredContactResult:
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    state: ColoredContactState
    iterations: int
    complementarity_residual: torch.Tensor
    kinetic_before: torch.Tensor
    kinetic_after: torch.Tensor
    converged: bool


def prepare_colored_contacts(n_bodies,contacts,*,device=None,dtype=torch.float64):
    """Stable greedy coloring of constraints that share dynamic bodies."""
    ordered=tuple(sorted(contacts,key=lambda c:c.key))
    if len({c.key for c in ordered})!=len(ordered):raise ValueError("duplicate contact key")
    assigned=[]; used_by_body=[set() for _ in range(n_bodies)]
    for c in ordered:
        if not 0<=c.body_a<n_bodies or c.body_b>=n_bodies:raise ValueError("invalid body index")
        forbidden=set(used_by_body[c.body_a])
        if c.body_b>=0:forbidden|=used_by_body[c.body_b]
        color=0
        while color in forbidden:color+=1
        assigned.append(color); used_by_body[c.body_a].add(color)
        if c.body_b>=0:used_by_body[c.body_b].add(color)
    groups=tuple(torch.tensor([i for i,c in enumerate(assigned) if c==color],dtype=torch.long,device=device)
                 for color in range(max(assigned,default=-1)+1))
    return ColoredContactData(tuple(c.key for c in ordered),
        torch.tensor([c.body_a for c in ordered],dtype=torch.long,device=device),
        torch.tensor([c.body_b for c in ordered],dtype=torch.long,device=device),
        torch.stack([c.point.to(device=device,dtype=dtype) for c in ordered]) if ordered else torch.empty((0,3),device=device,dtype=dtype),
        torch.stack([c.normal.to(device=device,dtype=dtype) for c in ordered]) if ordered else torch.empty((0,3),device=device,dtype=dtype),
        groups,contact_islands(n_bodies,ordered))


def _point_velocity(indices,points,com,v,w):
    r=points-com[indices]
    return v[indices]+torch.linalg.cross(w[indices],r,dim=1),r


def solve_colored_contacts(com,mass,inertia_world,linear_velocity,angular_velocity,data,
                           state=None,*,restitution=0.,tolerance=1e-10,max_iterations=500):
    """Color-sequential, within-color vectorised normal impulse iterations."""
    if not 0<=restitution<=1 or tolerance<=0 or max_iterations<1:raise ValueError("invalid solver parameter")
    if bool(torch.any(mass<=0)):raise ValueError("masses must be positive")
    if data.body_a.device!=com.device:raise ValueError("contact data and bodies must share device")
    invI=torch.linalg.inv(inertia_world); v=linear_velocity.clone(); w=angular_velocity.clone()
    m=len(data.keys)
    if state is None:j=torch.zeros(m,dtype=com.dtype,device=com.device)
    else:
        if state.keys!=data.keys or state.normal_impulses.shape!=(m,):raise ValueError("warm state topology mismatch")
        j=torch.clamp(state.normal_impulses.to(device=com.device,dtype=com.dtype),min=0.).clone()
    # Initial closing velocities define restitution targets.
    va,_=_point_velocity(data.body_a,data.points,com,v,w)
    mask=data.body_b>=0; vb=torch.zeros_like(va)
    if bool(mask.any()):vb[mask],_=_point_velocity(data.body_b[mask],data.points[mask],com,v,w)
    vn0=torch.sum((va-vb)*data.normals,dim=1); target=-restitution*torch.clamp(vn0,max=0.)

    def apply(ids,delta):
        a=data.body_a[ids]; p=data.points[ids]; n=data.normals[ids]; impulse=delta[:,None]*n
        v.index_add_(0,a,impulse/mass[a,None])
        ra=p-com[a]; dwa=torch.einsum('nij,nj->ni',invI[a],torch.linalg.cross(ra,impulse,dim=1)); w.index_add_(0,a,dwa)
        valid=data.body_b[ids]>=0
        if bool(valid.any()):
            b=data.body_b[ids][valid]; imp=-impulse[valid]; v.index_add_(0,b,imp/mass[b,None])
            rb=p[valid]-com[b]; dwb=torch.einsum('nij,nj->ni',invI[b],torch.linalg.cross(rb,imp,dim=1)); w.index_add_(0,b,dwb)
    # Warm start is applied colorwise (no write conflicts inside a color).
    for ids in data.colors:apply(ids,j[ids])
    converged=False
    for iteration in range(1,max_iterations+1):
        maximum=0.
        for ids in data.colors:
            a=data.body_a[ids]; p=data.points[ids]; n=data.normals[ids]
            va,ra=_point_velocity(a,p,com,v,w); rel=va; eff=1/mass[a]
            crossa=torch.linalg.cross(ra,n,dim=1)
            eff+=torch.sum(crossa*torch.einsum('nij,nj->ni',invI[a],crossa),dim=1)
            valid=data.body_b[ids]>=0
            if bool(valid.any()):
                b=data.body_b[ids][valid]; vb,rb=_point_velocity(b,p[valid],com,v,w); rel=rel.clone(); rel[valid]-=vb
                crossb=torch.linalg.cross(rb,n[valid],dim=1)
                eff=eff.clone(); eff[valid]+=1/mass[b]+torch.sum(crossb*torch.einsum('nij,nj->ni',invI[b],crossb),dim=1)
            vn=torch.sum(rel*n,dim=1); new=torch.clamp(j[ids]-(vn-target[ids])/eff,min=0.); delta=new-j[ids]
            apply(ids,delta); j[ids]=new; maximum=max(maximum,float(torch.max(torch.abs(delta))))
        if maximum<=tolerance:converged=True;break
    va,_=_point_velocity(data.body_a,data.points,com,v,w); vb=torch.zeros_like(va)
    if bool(mask.any()):vb[mask],_=_point_velocity(data.body_b[mask],data.points[mask],com,v,w)
    gapv=torch.sum((va-vb)*data.normals,dim=1)-target
    comp=torch.max(torch.maximum(torch.clamp(-gapv,min=0.),torch.abs(j*torch.clamp(gapv,min=0.)))) if m else com.new_zeros(())
    kb=.5*torch.sum(mass[:,None]*linear_velocity**2)+.5*torch.sum(angular_velocity*torch.einsum('nij,nj->ni',inertia_world,angular_velocity))
    ka=.5*torch.sum(mass[:,None]*v**2)+.5*torch.sum(w*torch.einsum('nij,nj->ni',inertia_world,w))
    if bool(ka>kb*(1+1e-9)+1e-11):raise RuntimeError("colored contact increased energy")
    return ColoredContactResult(v,w,ColoredContactState(data.keys,j),iteration,comp,kb,ka,converged)
