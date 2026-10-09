"""Validate actual FVM pressure/viscous wall forces before Q4 point transfer.

This is a one-way interface fixture; the explicit pressure/viscous sampling
clocks are preserved. No flexible response or physical CFD qualification.
"""
from dataclasses import dataclass
import math
import torch
from .surface_coupling import D

@dataclass(frozen=True)
class WallFacetLoads:
    points_m: torch.Tensor
    pressure_force_N: torch.Tensor
    viscous_force_N: torch.Tensor
    total_force_N: torch.Tensor
    time_s: float
    viscous_time_s: float
    config: dict


def read_wall_loads(data):
    if not isinstance(data,dict) or data.get('schema')!='tensorfvm.actual-wall-facets/1' or data.get('physical_accuracy_qualified') is not False:
        raise ValueError('invalid or falsely qualified FVM wall source')
    if data['units'].get('length')!='m' or data['units'].get('time')!='s' or data['units'].get('mass')!='kg' or not data['force_owner'].startswith('fluid-only'):
        raise ValueError('explicit SI units and single fluid force owner required')
    time=data['time_s'];visc=data['viscous_evaluation_time_s'];pressure=data['pressure_evaluation_time_s'];dt=data['config']['time_step']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (time,visc,pressure,dt)) or dt<=0 or time<0 or visc<0 or abs(time-pressure)>1e-12 or abs(time-visc-dt)>1e-12:
        raise ValueError('invalid or stale staggered wall load clocks')
    points=[];pf=[];vf=[];total=[]
    for facet in data['facets']:
        p=torch.as_tensor(facet['points_m'],dtype=D);n=torch.as_tensor(facet['normal_fluid'],dtype=D)
        a=torch.as_tensor(facet['area_m2'],dtype=D);pressure=torch.as_tensor(facet['pressure_pa'],dtype=D)
        tau=torch.as_tensor(facet['viscous_traction_on_body_pa'],dtype=D)
        fp=torch.as_tensor(facet['pressure_force_on_body_N'],dtype=D);fv=torch.as_tensor(facet['viscous_force_on_body_N'],dtype=D);f=torch.as_tensor(facet['total_force_on_body_N'],dtype=D)
        if p.numel()==0:
            if any(t.numel()!=0 for t in (n,pressure,tau,fp,fv,f)) or not bool(torch.isfinite(a).all()) or bool((a<=0).any()):raise ValueError('inconsistent empty wall facet axis')
            continue
        if a.ndim==0:a=a.expand(len(p))
        if p.ndim!=2 or p.shape[1]!=3 or any(t.shape!=p.shape for t in (n,tau,fp,fv,f)) or a.shape!=(len(p),) or pressure.shape!=(len(p),):raise ValueError('invalid wall facet field dimensions')
        if any(not bool(torch.isfinite(t).all()) for t in (p,n,a,pressure,tau,fp,fv,f)) or bool((a<=0).any()) or not torch.allclose(n.norm(dim=1),torch.ones(len(p),dtype=D),rtol=0,atol=1e-12):raise ValueError('invalid wall geometry or field')
        for actual,expected in [(fp,pressure[:,None]*n*a[:,None]),(fv,tau*a[:,None]),(f,fp+fv)]:
            if not torch.allclose(actual,expected,rtol=1e-12,atol=1e-12):raise ValueError('wall force does not match pressure/viscous traction')
        points.append(p);pf.append(fp);vf.append(fv);total.append(f)
    if not points or sum(len(p) for p in points)==0:raise ValueError('no actual wall force points')
    xyz=torch.cat(points);f=torch.cat(total)
    if not torch.allclose(f.sum(0),torch.tensor(data['force_on_body_N'],dtype=D),rtol=1e-12,atol=1e-12) or not torch.allclose(torch.linalg.cross(xyz,f).sum(0),torch.tensor(data['moment_on_body_about_origin_Nm'],dtype=D),rtol=1e-12,atol=1e-12):raise ValueError('source wall force/moment resultant inconsistent')
    return WallFacetLoads(xyz.clone(),torch.cat(pf).clone(),torch.cat(vf).clone(),f.clone(),float(time),float(visc),dict(data['config']))
