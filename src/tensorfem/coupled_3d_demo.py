"""Auditable 3-D geometric/material/contact nonlinear demonstration.

The model is a corotational bar whose free node moves along global X and can
contact an oriented rigid TRI3 plane.  It combines finite rotation/stretch,
incremental 1-D plasticity and the real 3-D contact kernel in one equilibrium
equation.  It is intentionally a verification problem, not a general solid
or surface-to-surface contact solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .contact3d import Contact3DState, initial_contact_state, update_node_facet_contact
from .nonlinear_step import StepState
from .plasticity import Plastic1DState, update_bilinear_1d


@dataclass(frozen=True)
class Coupled3DModel:
    fixed_point: torch.Tensor
    initial_free_point: torch.Tensor
    master_vertices: torch.Tensor
    master_faces: torch.Tensor
    area: float
    young: float
    yield_stress: float
    hardening: float
    normal_penalty: float
    reference_force: float

    def __post_init__(self):
        if self.fixed_point.shape != (3,) or self.initial_free_point.shape != (3,):
            raise ValueError("bar points must be three-dimensional")
        if min(self.area,self.young,self.yield_stress,self.normal_penalty) <= 0 or self.hardening < 0:
            raise ValueError("invalid constitutive/contact parameter")


@dataclass(frozen=True)
class Coupled3DState:
    plastic: Plastic1DState
    contact: Contact3DState


@dataclass(frozen=True)
class Coupled3DResponse:
    current_point: torch.Tensor
    stretch: torch.Tensor
    strain: torch.Tensor
    stress: torch.Tensor
    bar_force_x: torch.Tensor
    contact_force_x: torch.Tensor
    external_force: torch.Tensor
    residual: torch.Tensor


def default_coupled_3d_model(*, dtype=torch.float64, device=None) -> Coupled3DModel:
    """Reference inclined-bar model; the master normal points toward -X."""
    p0=torch.tensor([0.,0.,0.],dtype=dtype,device=device)
    p1=torch.tensor([.8,.6,.3],dtype=dtype,device=device)
    xp=.804
    master=torch.tensor([[xp,-2.,-2.],[xp,0.,2.],[xp,2.,-2.]],dtype=dtype,device=device)
    return Coupled3DModel(p0,p1,master,torch.tensor([[0,1,2]],dtype=torch.long,device=device),
                          .01,200000.,250.,10000.,500.,8.)


def coupled_3d_problem(model: Coupled3DModel):
    dtype=model.fixed_point.dtype; device=model.fixed_point.device
    z=torch.zeros((),dtype=dtype,device=device)
    plastic=Plastic1DState(z.clone(),z.clone())
    contact=initial_contact_state(model.initial_free_point,model.master_vertices,model.master_faces)
    initial=StepState(0.,torch.zeros(1,dtype=dtype,device=device),Coupled3DState(plastic,contact))
    reference_length=torch.linalg.vector_norm(model.initial_free_point-model.fixed_point)

    def evaluate(u: torch.Tensor, factor: float, committed: Coupled3DState):
        x=model.initial_free_point+torch.stack((u[0],u[0]*0,u[0]*0))
        vector=x-model.fixed_point; length=torch.linalg.vector_norm(vector)
        strain=(length-reference_length)/reference_length
        stress,mt,plastic_trial=update_bilinear_1d(
            strain,model.young,model.yield_stress,model.hardening,committed.plastic)
        direction=vector/length
        bar_x=model.area*stress*direction[0]
        contact=update_node_facet_contact(
            x,model.master_vertices,model.master_faces,committed.contact,
            normal_penalty=model.normal_penalty,tangential_penalty=model.normal_penalty,
            friction=0.,relative_increment=torch.zeros(3,dtype=dtype,device=device))
        internal=bar_x-contact.slave_force[0]
        residual=u.new_tensor([factor*model.reference_force])-internal.reshape(1)
        qx=direction[0]
        geometric=model.area*stress*(1-qx*qx)/length
        material=model.area*mt*qx*qx/reference_length
        active=contact.state.active
        tangent=material+geometric+(model.normal_penalty if active else 0.)
        return residual,tangent.reshape(1,1),Coupled3DState(plastic_trial,contact.state)
    return evaluate,initial


def coupled_3d_response(model: Coupled3DModel, state: StepState) -> Coupled3DResponse:
    u=state.displacement[0]; x=model.initial_free_point+u.new_tensor([1.,0.,0.])*u
    L0=torch.linalg.vector_norm(model.initial_free_point-model.fixed_point)
    vector=x-model.fixed_point; length=torch.linalg.vector_norm(vector); strain=(length-L0)/L0
    stress=model.young*(strain-state.material_state.plastic.plastic_strain)
    bar=model.area*stress*vector[0]/length
    contact=update_node_facet_contact(x,model.master_vertices,model.master_faces,
        state.material_state.contact,normal_penalty=model.normal_penalty,
        tangential_penalty=model.normal_penalty,friction=0.,
        relative_increment=torch.zeros(3,dtype=x.dtype,device=x.device))
    external=x.new_tensor(state.load_factor*model.reference_force)
    residual=external+contact.slave_force[0]-bar
    return Coupled3DResponse(x,length/L0,strain,stress,bar,contact.slave_force[0],external,residual)


def monotonic_reference_displacement(model: Coupled3DModel, force: float,
                                     *, tolerance: float=1e-14) -> float:
    """Independent scalar bisection of the closed-form monotonic force law."""
    if force < 0: raise ValueError("reference is defined for positive monotonic loading")
    x0=model.initial_free_point.detach().cpu(); L=float(torch.linalg.vector_norm(x0-model.fixed_point.detach().cpu()))
    plane=float(model.master_vertices[0,0].detach().cpu()); clearance=plane-float(x0[0])
    Et=model.young*model.hardening/(model.young+model.hardening) if model.hardening else 0.
    ey=model.yield_stress/model.young
    def applied(u):
        vec=x0.clone(); vec[0]+=u; length=float(torch.linalg.vector_norm(vec)); strain=(length-L)/L
        stress=model.young*strain if strain <= ey else model.yield_stress+Et*(strain-ey)
        return model.area*stress*float(vec[0])/length+model.normal_penalty*max(u-clearance,0.)
    lo,hi=0.,max(.1,clearance*2)
    while applied(hi)<force: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if applied(mid)<force: lo=mid
        else: hi=mid
        if hi-lo<tolerance: break
    return (lo+hi)/2
