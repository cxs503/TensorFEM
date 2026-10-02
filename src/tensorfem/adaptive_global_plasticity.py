"""Global TET4 Newton integration of adaptive multiplicative finite plasticity."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from .finite_strain_elasticity import FiniteStrainTet4Model
from .nonlinear_step import IncrementRecord,StepState,solve_adaptive
from .nonproportional_plasticity import AdaptiveFiniteJ2State,integrate_adaptive,virgin_adaptive_state
from .solid3d import _gradient
Tensor=torch.Tensor

@dataclass(frozen=True)
class AdaptiveGlobalState:
    points: tuple[AdaptiveFiniteJ2State,...]
    cumulative_dissipation: Tensor
    @classmethod
    def virgin(cls,model):
        pts=tuple(virgin_adaptive_state(dtype=model.reference_nodes.dtype,device=model.reference_nodes.device) for _ in model.elements)
        return cls(pts,torch.zeros(len(pts),dtype=model.reference_nodes.dtype,device=model.reference_nodes.device))

@dataclass(frozen=True)
class AdaptiveGlobalResult:
    load_factor: float; displacement: Tensor; reaction: Tensor; first_piola: Tensor
    state: AdaptiveGlobalState; increment_dissipation: Tensor
    increments: tuple[IncrementRecord,...]; tangent_material_evaluations: int

def _kinematics(model):
    x=model.reference_nodes[model.elements]; dn=x.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    g,d=_gradient(x,dn,"adaptive finite-plastic TET4")
    edofs=torch.stack(tuple(3*model.elements+i for i in range(3)),2).reshape(-1,12).long()
    return g,d/6,edofs

def assemble_adaptive_global(model: FiniteStrainTet4Model,displacement: Tensor,committed: AdaptiveGlobalState,
    *,yield_stress,hardening,material_rtol=2e-5,tangent=True,difference_step=2e-5):
    """Assemble adaptive stress and O(h^4) Richardson residual tangent."""
    g,v,edofs=_kinematics(model)
    if len(committed.points)!=len(edofs): raise ValueError("integration-point state count mismatch")
    internal=torch.zeros_like(displacement); K=displacement.new_zeros((model.n_dofs,model.n_dofs))
    Ps=[]; points=[]; added=[]
    def response(e,ue):
        F=torch.eye(3,dtype=ue.dtype,device=ue.device)+ue.reshape(4,3).T@g[e]
        r=integrate_adaptive(F,committed.points[e],model.young,model.poisson,yield_stress,hardening,rtol=material_rtol)
        return (r.first_piola@g[e].T).T.reshape(-1)*v[e],r
    evaluations=0
    for e,dofs in enumerate(edofs):
        ue=displacement[dofs]; fe,r=response(e,ue); evaluations+=1; internal.index_add_(0,dofs,fe)
        if tangent:
            coarse=torch.empty((12,12),dtype=ue.dtype,device=ue.device); fine=torch.empty_like(coarse)
            for j in range(12):
                scale=max(1.,abs(float(ue[j])))
                for matrix,h in ((coarse,difference_step*scale),(fine,.5*difference_step*scale)):
                    up=ue.clone(); um=ue.clone(); up[j]+=h; um[j]-=h
                    matrix[:,j]=(response(e,up)[0]-response(e,um)[0])/(2*h); evaluations+=2
            ke=(4*fine-coarse)/3
            K.index_put_((dofs[:,None],dofs[None,:]),ke,accumulate=True)
        Ps.append(r.first_piola); points.append(r.state); added.append(r.dissipation*v[e])
    added_t=torch.stack(added); trial=AdaptiveGlobalState(tuple(points),committed.cumulative_dissipation+added_t)
    return internal,K,torch.stack(Ps),trial,added_t,evaluations

def solve_adaptive_global_path(model,reference_force,targets: Sequence[float],*,yield_stress,hardening,
    initial: AdaptiveGlobalResult|None=None,initial_increment=.08,minimum_increment=1e-4,
    maximum_increment=.2,tolerance=1e-8,max_iterations=18,material_rtol=2e-5):
    """Force-controlled cyclic path with committed material rollback/restart."""
    force=reference_force.to(model.reference_nodes); fixed=model.fixed_dofs.to(device=force.device,dtype=torch.long)
    mask=torch.ones(model.n_dofs,dtype=torch.bool,device=force.device); mask[fixed]=False
    free=torch.arange(model.n_dofs,device=force.device)[mask]
    factor=0. if initial is None else initial.load_factor
    u=torch.zeros_like(force) if initial is None else initial.displacement.clone()
    state=AdaptiveGlobalState.virgin(model) if initial is None else initial.state
    outputs=[]
    for value in targets:
        target=float(value); start=float(factor); delta=target-start; before=state.cumulative_dissipation.clone()
        def evaluate(q,progress,base):
            trial_u=u.clone().index_copy(0,free,q)
            fi,kt,_,trial,_,_=assemble_adaptive_global(model,trial_u,base,yield_stress=yield_stress,
                hardening=hardening,material_rtol=material_rtol)
            return (start+delta*progress)*force[free]-fi[free],kt[free][:,free],trial
        advanced=solve_adaptive(evaluate,StepState(0.,u[free].clone(),state),target_factor=1.,
            initial_increment=initial_increment,minimum_increment=minimum_increment,maximum_increment=maximum_increment,
            tolerance=tolerance,max_iterations=max_iterations)
        u=u.clone().index_copy(0,free,advanced.displacement); state=advanced.material_state; factor=target
        fi,_,P,_,_,evals=assemble_adaptive_global(model,u,state,yield_stress=yield_stress,hardening=hardening,
                                                   material_rtol=material_rtol,tangent=False)
        outputs.append(AdaptiveGlobalResult(factor,u.clone(),fi-factor*force,P,state,
            state.cumulative_dissipation-before,tuple(advanced.history),49*len(model.elements)))
    return tuple(outputs)
