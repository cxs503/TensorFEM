"""Color-batched rigid Coulomb contact with rotational tangent blocks."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .colored_contact_graph import ColoredContactData


@dataclass(frozen=True)
class ColoredFrictionState:
    keys: tuple[tuple,...]
    normal_impulses: torch.Tensor
    friction_impulses: torch.Tensor
    sticking: torch.Tensor


@dataclass(frozen=True)
class ColoredFrictionResult:
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    state: ColoredFrictionState
    iterations: int
    complementarity_residual: torch.Tensor
    cone_residual: torch.Tensor
    kinetic_before: torch.Tensor
    kinetic_after: torch.Tensor
    dissipated_energy: torch.Tensor
    converged: bool


def _point(indices,points,com,v,w):
    r=points-com[indices]
    return v[indices]+torch.linalg.cross(w[indices],r,dim=1),r


def _basis(n):
    x=torch.tensor([1.,0.,0.],dtype=n.dtype,device=n.device).expand_as(n)
    y=torch.tensor([0.,1.,0.],dtype=n.dtype,device=n.device).expand_as(n)
    axis=torch.where((torch.abs(n[:,0])<.8)[:,None],x,y)
    t1=torch.linalg.cross(n,axis,dim=1); t1=t1/torch.linalg.vector_norm(t1,dim=1)[:,None]
    return torch.stack((t1,torch.linalg.cross(n,t1,dim=1)),dim=2)


def solve_colored_frictional_contacts(com,mass,inertia_world,linear_velocity,angular_velocity,
                                      data: ColoredContactData,state=None,*,friction: float,
                                      restitution=0.,tolerance=1e-10,max_iterations=500):
    """Color-sequential, within-color vectorized Coulomb PGS."""
    if friction<0 or not 0<=restitution<=1 or tolerance<=0 or max_iterations<1:
        raise ValueError("invalid frictional solver parameter")
    if bool(torch.any(mass<=0)) or data.body_a.device!=com.device:raise ValueError("invalid mass or device")
    m=len(data.keys); invI=torch.linalg.inv(inertia_world); v=linear_velocity.clone(); w=angular_velocity.clone()
    if state is None:
        jn=torch.zeros(m,dtype=com.dtype,device=com.device); jt=torch.zeros((m,3),dtype=com.dtype,device=com.device)
    else:
        if state.keys!=data.keys or state.normal_impulses.shape!=(m,) or state.friction_impulses.shape!=(m,3):
            raise ValueError("warm state topology mismatch")
        jn=torch.clamp(state.normal_impulses.to(com),min=0.).clone(); jt=state.friction_impulses.to(com).clone()
        mag=torch.linalg.vector_norm(jt,dim=1); limit=friction*jn
        scale=torch.where(mag>limit,limit/torch.clamp(mag,min=torch.finfo(com.dtype).eps),torch.ones_like(mag)); jt*=scale[:,None]
    va,_=_point(data.body_a,data.points,com,v,w); valid_all=data.body_b>=0; vb=torch.zeros_like(va)
    if bool(valid_all.any()):vb[valid_all],_=_point(data.body_b[valid_all],data.points[valid_all],com,v,w)
    vn0=torch.sum((va-vb)*data.normals,dim=1); target=-restitution*torch.clamp(vn0,max=0.)

    def apply(ids,impulse):
        a=data.body_a[ids]; p=data.points[ids]
        v.index_add_(0,a,impulse/mass[a,None]); ra=p-com[a]
        w.index_add_(0,a,torch.einsum('nij,nj->ni',invI[a],torch.linalg.cross(ra,impulse,dim=1)))
        ok=data.body_b[ids]>=0
        if bool(ok.any()):
            b=data.body_b[ids][ok]; imp=-impulse[ok]; v.index_add_(0,b,imp/mass[b,None]); rb=p[ok]-com[b]
            w.index_add_(0,b,torch.einsum('nij,nj->ni',invI[b],torch.linalg.cross(rb,imp,dim=1)))
    for ids in data.colors:apply(ids,jn[ids,None]*data.normals[ids]+jt[ids])
    converged=False
    for iteration in range(1,max_iterations+1):
        maximum=0.
        for ids in data.colors:
            a=data.body_a[ids]; p=data.points[ids]; n=data.normals[ids]; va,ra=_point(a,p,com,v,w); rel=va
            cn=torch.linalg.cross(ra,n,dim=1); eff=1/mass[a]+torch.sum(cn*torch.einsum('nij,nj->ni',invI[a],cn),dim=1)
            ok=data.body_b[ids]>=0
            if bool(ok.any()):
                b=data.body_b[ids][ok]; vb,rb=_point(b,p[ok],com,v,w); rel=rel.clone();rel[ok]-=vb
                cb=torch.linalg.cross(rb,n[ok],dim=1);eff=eff.clone();eff[ok]+=1/mass[b]+torch.sum(cb*torch.einsum('nij,nj->ni',invI[b],cb),dim=1)
            vn=torch.sum(rel*n,dim=1); newn=torch.clamp(jn[ids]-(vn-target[ids])/eff,min=0.); dn=newn-jn[ids]
            apply(ids,dn[:,None]*n)
            va,ra=_point(a,p,com,v,w); rel=va
            if bool(ok.any()):vb,rb=_point(data.body_b[ids][ok],p[ok],com,v,w);rel=rel.clone();rel[ok]-=vb
            T=_basis(n); vt=torch.einsum('nki,nk->ni',T,rel)
            ca=torch.linalg.cross(ra[:,None,:],T.transpose(1,2),dim=2)
            WT=torch.eye(2,dtype=com.dtype,device=com.device)[None]*((1/mass[a])[:,None,None])
            WT=WT+torch.einsum('nki,nij,nlj->nkl',ca,invI[a],ca)
            if bool(ok.any()):
                cb=torch.linalg.cross(rb[:,None,:],T[ok].transpose(1,2),dim=2)
                add=torch.eye(2,dtype=com.dtype,device=com.device)[None]*((1/mass[data.body_b[ids][ok]])[:,None,None])
                add=add+torch.einsum('nki,nij,nlj->nkl',cb,invI[data.body_b[ids][ok]],cb)
                WT=WT.clone();WT[ok]+=add
            old2=torch.einsum('nki,nk->ni',T,jt[ids]); trial2=old2-torch.linalg.solve(WT,vt[:,:,None]).squeeze(2)
            trial=torch.einsum('nki,ni->nk',T,trial2); mag=torch.linalg.vector_norm(trial,dim=1); limit=friction*newn
            scale=torch.where(mag>limit,limit/torch.clamp(mag,min=torch.finfo(com.dtype).eps),torch.ones_like(mag)); newt=trial*scale[:,None]
            dt=newt-jt[ids]; apply(ids,dt); jn[ids]=newn;jt[ids]=newt
            maximum=max(maximum,float(torch.max(torch.abs(dn))),float(torch.max(torch.linalg.vector_norm(dt,dim=1))))
        if maximum<=tolerance:converged=True;break
    va,_=_point(data.body_a,data.points,com,v,w);vb=torch.zeros_like(va)
    if bool(valid_all.any()):vb[valid_all],_=_point(data.body_b[valid_all],data.points[valid_all],com,v,w)
    gapv=torch.sum((va-vb)*data.normals,dim=1)-target
    comp=torch.max(torch.maximum(torch.clamp(-gapv,min=0.),torch.abs(jn*torch.clamp(gapv,min=0.)))) if m else com.new_zeros(())
    cone=torch.max(torch.clamp(torch.linalg.vector_norm(jt,dim=1)-friction*jn,min=0.)) if m else com.new_zeros(())
    sticking=torch.linalg.vector_norm(jt,dim=1)<friction*jn-tolerance
    kb=.5*torch.sum(mass[:,None]*linear_velocity**2)+.5*torch.sum(angular_velocity*torch.einsum('nij,nj->ni',inertia_world,angular_velocity))
    ka=.5*torch.sum(mass[:,None]*v**2)+.5*torch.sum(w*torch.einsum('nij,nj->ni',inertia_world,w))
    if bool(ka>kb*(1+1e-9)+1e-11):raise RuntimeError("colored frictional contact increased energy")
    return ColoredFrictionResult(v,w,ColoredFrictionState(data.keys,jn,jt,sticking),iteration,comp,cone,kb,ka,kb-ka,converged)
