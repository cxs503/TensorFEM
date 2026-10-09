"""Recorded XY wrench replay on a linear XYZ shell; no motion feedback.

A resultant does not determine a pressure field. Area-weighted minimum-norm
redistribution preserves its wrench, but the resulting stresses are illustrative.
The dense midpoint solver is a CPU verification tool for small linear models.
"""
from __future__ import annotations
import math
import hashlib
import torch
from .surface_coupling import PlanarEmbedding, _time

D = torch.float64


def embed_wrench(record, prefix, embedding):
    """Transform polar force and axial moment about the original origin."""
    if prefix not in ('fluid', 'ice_contact', 'other', 'total'):
        raise ValueError('unknown load contribution')
    values=[record[f'{prefix}_{name}'] for name in
            ('fx_n','fy_n','fz_n','mx_nm','my_nm','mz_nm')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('invalid SI wrench')
    f=torch.tensor(values[:3],dtype=D);m=torch.tensor(values[3:],dtype=D)
    if abs(float(f[2]))>1e-12 or bool((m[:2].abs()>1e-12).any()):
        raise ValueError('wrench is not planar XY')
    force=embedding.vectors(f[:2].reshape(1,2))[0]
    normal=torch.linalg.cross(embedding.axes[:,0],embedding.axes[:,1])
    moment=normal*m[2]+torch.linalg.cross(embedding.origin,force)
    return torch.cat((force,moment))


def distribute_wrench(surface, wrench, sample_mask):
    """Minimize sum |F_i|² / A_i subject to all six wrench constraints."""
    surface._check_integrity()
    w=torch.as_tensor(wrench,dtype=D)
    mask=torch.as_tensor(sample_mask)
    if w.shape!=(6,) or not bool(torch.isfinite(w).all()) or mask.dtype!=torch.bool or mask.shape!=(len(surface.areas),) or not bool(mask.any()):
        raise ValueError('invalid wrench or surface selection')
    p=surface.positions[mask];area=surface.areas[mask]
    A=torch.zeros((6,3*len(p)),dtype=D)
    for i,(x,y,z) in enumerate(p):
        A[:3,3*i:3*i+3]=torch.eye(3,dtype=D)
        A[3:,3*i:3*i+3]=torch.tensor([[0.,-z,y],[z,0.,-x],[-y,x,0.]],dtype=D)
    weights=area.repeat_interleave(3)
    gram=(A*weights)@A.T
    if int(torch.linalg.matrix_rank(gram))!=6:
        raise ValueError('selected surface cannot carry all six wrench components')
    force=(weights*(A.T@torch.linalg.solve(gram,w))).reshape(-1,3)
    residual=torch.linalg.vector_norm(A@force.flatten()-w)
    if float(residual)>1e-9*max(1.,float(torch.linalg.vector_norm(w))):
        raise ValueError('wrench redistribution failed')
    result=torch.zeros_like(surface.positions);result[mask]=force
    return result


class LinearMidpointIntegrator:
    """Constant interval forces, implicit midpoint, fixed linear geometry.

    Exact discrete energy work identity for a symmetric linear K. This dense
    solve is deliberately bounded to 6000 DOFs, not a scalable production path.
    """
    def __init__(self,mass,stiffness,dt_s):
        self.mass=torch.as_tensor(mass,dtype=D).clone()
        self.K=stiffness.to_dense().to(D) if stiffness.layout!=torch.strided else stiffness.to(D).clone()
        self.dt=_time(dt_s)
        n=len(self.mass)
        if self.dt<=0 or n>6000 or self.mass.ndim!=1 or self.K.shape!=(n,n) or not bool(torch.isfinite(self.mass).all() and torch.isfinite(self.K).all()) or not bool((self.mass>0).all()):
            raise ValueError('invalid or oversized linear model')
        if not torch.allclose(self.K,self.K.T,rtol=1e-10,atol=1e-6):
            raise ValueError('stiffness must be symmetric')
        self.K=(self.K+self.K.T)/2
        digest=hashlib.sha256(self.mass.contiguous().numpy().tobytes())
        digest.update(self.K.contiguous().numpy().tobytes())
        self.model_sha256=digest.hexdigest()
        self.factor=torch.linalg.cholesky(torch.diag(self.mass)+self.dt**2*self.K/4)
        self.q=torch.zeros(n,dtype=D);self.v=self.q.clone();self.work_J=0.;self.time_s=0.

    def step(self,force):
        f=torch.as_tensor(force,dtype=D)
        if f.shape!=self.q.shape or not bool(torch.isfinite(f).all()):raise ValueError('invalid force')
        rhs=self.mass*self.v+self.dt/2*(f-self.K@self.q)
        midpoint=torch.cholesky_solve(rhs[:,None],self.factor).flatten()
        delta=self.dt*midpoint
        self.q+=delta;self.v=2*midpoint-self.v
        self.work_J+=float(f@delta);self.time_s+=self.dt

    def energy(self):
        return float(.5*self.mass@(self.v*self.v)+.5*self.q@(self.K@self.q))

    def snapshot(self):
        return {'model_sha256':self.model_sha256,'time_s':self.time_s,'q':self.q.tolist(),'v':self.v.tolist(),'work_J':self.work_J,'dt_s':self.dt}

    def restore(self,data):
        if not isinstance(data,dict) or set(data)!=set(self.snapshot()) or data['dt_s']!=self.dt or data['model_sha256']!=self.model_sha256:raise ValueError('incompatible restart')
        q=torch.as_tensor(data['q'],dtype=D);v=torch.as_tensor(data['v'],dtype=D)
        if q.shape!=self.q.shape or v.shape!=self.v.shape or not bool(torch.isfinite(q).all() and torch.isfinite(v).all()):raise ValueError('invalid restart')
        time=_time(data['time_s']);work=data['work_J']
        if isinstance(work,bool) or not isinstance(work,(float,int)) or not math.isfinite(work):raise ValueError('invalid restart work')
        self.q=q.clone();self.v=v.clone();self.time_s=time;self.work_J=float(work)
