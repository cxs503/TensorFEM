"""Executable fluid/rigid translation feedback with a Q4 motion carrier.

A verification fixture, not a flexible SUBOFF or icebreaking solver. Angular
motion is constrained and no angular momentum or surface pressure is certified.
"""
from __future__ import annotations
import math
import torch
from tensorlbm.feedback_moving_disk import MovingDiskFeedback
from tensorlbm.local_moving_boundary_2d import LocalMovingDisk
from tensorlbm.d2q9 import C
from .point_surface_coupling import PointSurfaceExchange
from .surface_coupling import PlanarEmbedding,_time,D

class _LocalWithoutReservoir(LocalMovingDisk):
    def step(self):
        # Adapter compatibility: there is no reservoir operation in this model.
        entry=super().step()
        return {**entry,'reservoir_momentum':[0.,0.]}

class LocalDiskFeedback(MovingDiskFeedback):
    def __init__(self,*,local_radius=4,**config):
        super().__init__(**config)
        self.solver=_LocalWithoutReservoir(local_radius=local_radius,**self.solver.config)
        self.local_radius=local_radius

    def advance(self,**kwargs):
        record=super().advance(**kwargs)
        record.update(boundary='halfway disk with finite-neighborhood mass transfer',
                      global_mass_rescale=False,local_transfer_radius_cells=self.local_radius)
        return record

    def snapshot(self):
        return {'schema':'tensorfem.local-live-disk/1','config':self.config,
                'local_radius':self.local_radius,'solver':self.solver.snapshot(),'history':self.history}

    @classmethod
    def restore(cls,data):
        if not isinstance(data,dict) or set(data)!={'schema','config','local_radius','solver','history'} or data['schema']!='tensorfem.local-live-disk/1':
            raise ValueError('invalid local feedback restart')
        obj=cls(local_radius=data['local_radius'],**data['config'])
        restored=_LocalWithoutReservoir.restore(data['solver'])
        if restored.config!=obj.solver.config or restored.local_radius!=obj.local_radius or len(data['history'])!=restored.time:
            raise ValueError('local exchange configuration or clock mismatch')
        obj.solver=restored;obj.history=data['history'];return obj

def fluid_kinetic_J(fluid):
    f=fluid.solver.f;rho=f.sum(0)
    p=torch.einsum('iyx,ia->ayx',f,C.double())
    c=fluid.config
    return float((.5*p.square().sum(0)/rho.clamp_min(1e-30)).sum())*c['density_kg_m3']*c['thickness_m']*c['dx_m']**4/c['dt_s']**2

class RigidFluidFeedback:
    """Free two-dimensional translation, fluid pose updated every real step.

    The structural mass is four equal Q4 carrier-node masses restricted to
    uniform translation. Force is the body impulse / dt at the disk center.
    This is a rigid translation subspace, not a solved flexible shell response.
    """
    def __init__(self,*,backend='local',body_mass_kg=5.,initial_velocity_m_s=(0.,.03),**fluid_config):
        if backend not in ('local','global'):raise ValueError('unknown fluid backend')
        if isinstance(body_mass_kg,bool) or not isinstance(body_mass_kg,(float,int)) or not math.isfinite(body_mass_kg) or body_mass_kg<=0:raise ValueError('positive body mass required')
        self.velocity=torch.as_tensor(initial_velocity_m_s,dtype=D).clone()
        if self.velocity.shape!=(2,) or not bool(torch.isfinite(self.velocity).all()):raise ValueError('finite initial velocity required')
        self.backend=backend;self.mass=float(body_mass_kg)
        self.fluid=(LocalDiskFeedback if backend=='local' else MovingDiskFeedback)(**fluid_config)
        self.embedding=PlanarEmbedding([0.,0.,0.],[[1.,0.],[0.,0.],[0.,1.]])
        self.initial_center=torch.tensor(self.fluid.center_m,dtype=D)
        x,y=self.initial_center
        self.reference_nodes=torch.tensor([[x-.06,0.,y-.05],[x+.06,0.,y-.05],[x+.06,0.,y+.05],[x-.06,0.,y+.05]],dtype=D)
        self.initial_momentum=torch.tensor(self.fluid.fluid_momentum_Ns(),dtype=D)+self.mass*self.velocity
        self.initial_kinetic_J=fluid_kinetic_J(self.fluid)+float(.5*self.mass*(self.velocity**2).sum())
        self.reservoir_impulse=torch.zeros(2,dtype=D);self.history=[]

    def step(self):
        f=self.fluid;clock=f.time_s;dt=f.config['dt_s']
        center=torch.tensor(f.center_m,dtype=D);shift=self.embedding.vectors((center-self.initial_center)[None])[0]
        nodes=self.reference_nodes+shift
        carrier=PointSurfaceExchange(nodes,[[0,1,2,3]],self.embedding.positions(center[None]),maximum_offset_m=.001)
        nv=torch.zeros((4,6),dtype=D);nv[:,:3]=self.embedding.vectors(self.velocity[None])[0]
        mapped=self.embedding.project_load(carrier.velocities(nv))[0]
        before_ke=float(.5*self.mass*(self.velocity**2).sum());before_velocity=self.velocity.clone()
        record=f.advance(time_s=clock,center_m=f.center_m,velocity_m_s=mapped.tolist())
        impulse=torch.tensor(record['impulse_on_body_Ns'],dtype=D)
        load=carrier.point_loads(self.embedding.vectors((impulse/dt)[None]),time_s=clock,source='TensorLBM')
        mapping=carrier.audit(load,nv,expected_time_s=clock)
        # Project carrier nodal forces onto the uniform structural translations.
        generalized_impulse=load.generalized_force[:,:3].sum(0)*dt
        actual_impulse=self.embedding.project_load(generalized_impulse[None])[0]
        self.velocity+=actual_impulse/self.mass
        self.reservoir_impulse+=torch.tensor(record['reservoir_impulse_on_fluid_Ns'],dtype=D)
        body_ke=float(.5*self.mass*(self.velocity**2).sum());fluid_ke=fluid_kinetic_J(f)
        residual=torch.tensor(f.fluid_momentum_Ns(),dtype=D)+self.mass*self.velocity-self.initial_momentum-self.reservoir_impulse
        midpoint_work=float(impulse@(.5*(before_velocity+self.velocity)))
        entry={'exchange':record,'body_velocity_m_s':self.velocity.tolist(),
               'carrier_mapping':mapping,'motion_feedback_error_m_s':float((mapped-before_velocity).abs().max()),
               'fluid_kinetic_J':fluid_ke,'body_kinetic_J':body_ke,'body_midpoint_impulse_work_J':midpoint_work,
               'body_energy_identity_error_J':body_ke-before_ke-midpoint_work,
               'staggered_body_work_defect_J':body_ke-before_ke-record['body_interface_work_J'],
               'total_kinetic_change_J':body_ke+fluid_ke-self.initial_kinetic_J,
               'total_momentum_residual_Ns':residual.tolist(),'global_reservoir_impulse_Ns':self.reservoir_impulse.tolist(),
               'physical_accuracy_qualified':False,'flexible_shell_response':False}
        self.history.append(entry);return entry

    def snapshot(self):
        return {'schema':'tensorfem.rigid-fluid-feedback/1','backend':self.backend,'body_mass_kg':self.mass,
                'velocity_m_s':self.velocity.tolist(),'initial_center_m':self.initial_center.tolist(),
                'initial_momentum_Ns':self.initial_momentum.tolist(),'initial_kinetic_J':self.initial_kinetic_J,
                'reservoir_impulse_Ns':self.reservoir_impulse.tolist(),'history':self.history,'fluid':self.fluid.snapshot()}

    @classmethod
    def restore(cls,data):
        expected={'schema','backend','body_mass_kg','velocity_m_s','initial_center_m','initial_momentum_Ns','initial_kinetic_J','reservoir_impulse_Ns','history','fluid'}
        if not isinstance(data,dict) or set(data)!=expected or data['schema']!='tensorfem.rigid-fluid-feedback/1':raise ValueError('invalid rigid feedback restart')
        obj=cls(backend=data['backend'],body_mass_kg=data['body_mass_kg'],initial_velocity_m_s=data['velocity_m_s'],**data['fluid']['config'])
        obj.fluid=(LocalDiskFeedback if obj.backend=='local' else MovingDiskFeedback).restore(data['fluid'])
        if len(data['history'])!=obj.fluid.solver.time:raise ValueError('body/fluid history clock mismatch')
        if data['initial_center_m']!=obj.initial_center.tolist():raise ValueError('initial geometry registration mismatch')
        if data['history']:
            last=data['history'][-1]
            if last['body_velocity_m_s']!=data['velocity_m_s'] or last['exchange']!=obj.fluid.history[-1]:raise ValueError('body/fluid exchange state mismatch')
            first=torch.tensor(data['history'][0]['exchange']['velocity_held_m_s'],dtype=D)
            if not torch.allclose(torch.tensor(data['initial_momentum_Ns'],dtype=D),obj.mass*first,atol=1e-12,rtol=0):raise ValueError('initial momentum baseline mismatch')
            expected_energy=float(.5*obj.mass*(first**2).sum())
            if abs(data['initial_kinetic_J']-expected_energy)>1e-12:raise ValueError('initial energy baseline mismatch')
        for key,field in [('initial_center_m','initial_center'),('initial_momentum_Ns','initial_momentum'),('reservoir_impulse_Ns','reservoir_impulse')]:
            value=torch.as_tensor(data[key],dtype=D)
            if value.shape!=(2,) or not bool(torch.isfinite(value).all()):raise ValueError('invalid restart vector')
            setattr(obj,field,value.clone())
        if not math.isfinite(data['initial_kinetic_J']) or data['initial_kinetic_J']<0:raise ValueError('invalid energy baseline')
        obj.initial_kinetic_J=data['initial_kinetic_J'];obj.history=data['history']
        # The immutable carrier geometry is rebuilt at the original registration.
        x,y=obj.initial_center
        obj.reference_nodes=torch.tensor([[x-.06,0.,y-.05],[x+.06,0.,y-.05],[x+.06,0.,y+.05],[x-.06,0.,y+.05]],dtype=D)
        return obj
