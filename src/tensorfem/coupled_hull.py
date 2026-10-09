"""Vehicle-only linear SUBOFF shell receiver for external coupling loads.

No FEM ice is assembled. This explicit small-strain adapter is an integration
foundation; finite-rotation shell dynamics and coupled wet accuracy are open.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch

from .surface_coupling import QuadSurfaceExchange, SurfaceLoad, _time
from .suboff_geometry import inward_midsurface, audit_closed_quads
from .suboff_appendages import audit_appended_mesh
from .suboff_ice_collision import _area, _ids, _sparse, _stress, conservative_step_bound
from .suboff_ice_simulation import _operators
from .spherical_shell import projected_shell4_stiffness

D=torch.float64


@dataclass(frozen=True)
class HullCouplingConfig:
    young: float=210e9
    poisson: float=.3
    density: float=7850.
    hull_thickness: float=.003
    appendage_thickness: float=.003
    total_mass: float=700.
    time_safety: float=.55

    def validate(self):
        for name,value in asdict(self).items():
            if isinstance(value,bool) or not math.isfinite(value) or (name!='poisson' and value<=0):
                raise ValueError(f'invalid {name}')
        if not -1<self.poisson<.5 or self.time_safety>1:
            raise ValueError('invalid Poisson ratio or time safety')


class LinearHullReceiver:
    def __init__(self,geometry,config=HullCouplingConfig()):
        config.validate();self.config=config
        self.geometry_sha256=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        schema=geometry.get('schema')
        if schema not in ('tensorfem.suboff-appended-shell/1','tensorfem.suboff-outer-geometry/1'):
            raise ValueError('unsupported SUBOFF geometry')
        appended=schema=='tensorfem.suboff-appended-shell/1'
        self.outer=torch.tensor(geometry['nodes'],dtype=D)
        self.elements=torch.tensor(geometry['elements'],dtype=torch.long)
        bn=geometry['body_node_count'] if appended else len(self.outer)
        be=geometry['body_element_count'] if appended else len(self.elements)
        self.labels=torch.tensor(geometry['element_component_ids'] if appended else [0]*be)
        self.audit=(audit_appended_mesh(self.outer,self.elements,bn,be,geometry['components']) if appended else audit_closed_quads(self.outer,self.elements))
        if not self.audit['passed']:raise ValueError('invalid geometry')
        self.nodes=self.outer.clone()
        self.nodes[:bn]=inward_midsurface(self.outer[:bn],self.elements[:be],config.hull_thickness)[0]
        mass=torch.zeros((len(self.nodes),6),dtype=D);ke=[];materials=[]
        self.thickness=torch.where(self.labels==0,config.hull_thickness,config.appendage_thickness).to(D)
        for el,t in zip(self.elements,self.thickness):
            area=_area(self.nodes[el]);m=config.density*float(t)*area/4
            mass[el,:3]+=m;mass[el,3:]+=m*float(t)**2/12
            ke.append(projected_shell4_stiffness(self.nodes[el],config.young,config.poisson,t))
            materials.append((config.young,config.poisson,float(t),config.density))
        self.steel_mass=float(mass[:,2].sum())
        if config.total_mass<self.steel_mass:raise ValueError('declared mass below steel mass')
        mass[:,:3]*=config.total_mass/self.steel_mass
        self.mass=mass.reshape(-1);self.ids=_ids(self.elements);ke=torch.stack(ke)
        shell=_sparse([self.ids],[ke],len(self.mass))
        self.dt_bound=conservative_step_bound(shell,self.mass,torch.ones(len(self.mass),dtype=torch.bool))*config.time_safety
        self.shell=shell.to_sparse_csr()
        self.stress_operator=_operators(self.nodes,self.elements,materials)
        self.q=torch.zeros_like(self.mass);self.v=self.q.clone();self.time_s=0.
        self.work_J=0.;self.impulse=torch.zeros(3,dtype=D)

    def surface(self):
        return QuadSurfaceExchange(self.nodes+self.q.reshape(-1,6)[:,:3],self.elements,
                                   face_offset_m=self.thickness/2)

    def advance(self,surface,load:SurfaceLoad,dt):
        if not math.isfinite(dt) or dt<=0 or dt>self.dt_bound*(1+1e-12):
            raise ValueError('invalid or unstable hull timestep')
        if (not isinstance(surface,QuadSurfaceExchange) or
                not torch.equal(surface.nodes,self.nodes+self.q.reshape(-1,6)[:,:3]) or
                not torch.equal(surface.elements,self.elements) or
                not torch.equal(surface.face_offset_m,self.thickness/2)):
            raise ValueError('stale hull surface')
        surface.validate_load(load,expected_time_s=self.time_s)
        f=load.generalized_force.reshape(-1)
        half=self.v+.5*dt*(f-torch.mv(self.shell,self.q))/self.mass
        q=self.q+dt*half
        v=half+.5*dt*(f-torch.mv(self.shell,q))/self.mass
        if not bool(torch.isfinite(q).all() and torch.isfinite(v).all()):
            raise RuntimeError('nonfinite hull response')
        self.work_J+=float(f@(q-self.q));self.impulse+=dt*load.generalized_force[:,:3].sum(0)
        self.q=q;self.v=v;self.time_s+=dt

    def diagnostics(self):
        _,vm=_stress({'ids':self.ids,'stress_operator':self.stress_operator},self.q)
        kinetic=float(.5*self.mass@(self.v*self.v))
        stored=float(.5*self.q@torch.mv(self.shell,self.q))
        momentum=(self.mass.reshape(-1,6)[:,:3]*self.v.reshape(-1,6)[:,:3]).sum(0)
        return {'time_s':self.time_s,'kinetic_J':kinetic,'strain_J':stored,
                'external_work_J':self.work_J,'energy_residual_J':kinetic+stored-self.work_J,
                'linear_momentum_kg_m_s':momentum.tolist(),'external_impulse_N_s':self.impulse.tolist(),
                'momentum_residual_kg_m_s':float(torch.linalg.vector_norm(momentum-self.impulse)),
                'maximum_gauss_von_mises_Pa':float(vm.max()),
                'maximum_rotation_rad':float(self.q.reshape(-1,6)[:,3:].abs().max()),
                'physical_accuracy_qualified':False,'ice_elements':0}

    def snapshot(self):
        return {'schema':'tensorfem.hull-coupling-restart/1','geometry_sha256':self.geometry_sha256,
                'config':asdict(self.config),'time_s':self.time_s,'q':self.q.tolist(),'v':self.v.tolist(),
                'external_work_J':self.work_J,'external_impulse_N_s':self.impulse.tolist()}

    def restore(self,data):
        expected=set(self.snapshot())
        if not isinstance(data,dict) or set(data)!=expected or data['schema']!='tensorfem.hull-coupling-restart/1' or data['geometry_sha256']!=self.geometry_sha256 or data['config']!=asdict(self.config):
            raise ValueError('incompatible hull restart')
        time=_time(data['time_s']);q=torch.tensor(data['q'],dtype=D);v=torch.tensor(data['v'],dtype=D)
        impulse=torch.tensor(data['external_impulse_N_s'],dtype=D);work=data['external_work_J']
        if q.shape!=self.q.shape or v.shape!=self.v.shape or impulse.shape!=(3,) or not bool(torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(impulse).all()) or isinstance(work,bool) or not math.isfinite(work):
            raise ValueError('invalid hull restart state')
        self.q=q;self.v=v;self.time_s=time;self.work_J=float(work);self.impulse=impulse
