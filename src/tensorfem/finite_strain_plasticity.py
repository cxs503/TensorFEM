"""Multiplicative finite-strain logarithmic J2 plasticity for TET4.

The model stores ``Fp`` explicitly, uses ``F=Fe Fp`` and an exponential update
with a traceless associative direction.  Global tangents are auditable centred
algorithmic differences; no claim of a closed-form consistent tangent is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from .finite_strain_elasticity import FiniteStrainTet4Model
from .nonlinear_step import IncrementRecord,StepState,solve_adaptive
from .solid3d import _gradient

Tensor=torch.Tensor

@dataclass(frozen=True)
class MultiplicativeJ2State:
    plastic_gradient: Tensor
    alpha: Tensor

@dataclass(frozen=True)
class FinitePlasticState:
    points: tuple[MultiplicativeJ2State,...]
    @classmethod
    def virgin(cls,model):
        I=torch.eye(3,dtype=model.reference_nodes.dtype,device=model.reference_nodes.device)
        z=torch.zeros((),dtype=I.dtype,device=I.device)
        return cls(tuple(MultiplicativeJ2State(I.clone(),z.clone()) for _ in model.elements))

@dataclass(frozen=True)
class FinitePlasticResult:
    load_factor: float; displacement: Tensor; reaction: Tensor; first_piola: Tensor
    state: FinitePlasticState; dissipation: Tensor; increments: tuple[IncrementRecord,...]

def _sym_log(A):
    values,vectors=torch.linalg.eigh(.5*(A+A.T))
    if bool(torch.any(values<=torch.finfo(A.dtype).eps)): raise ValueError("elastic metric is not positive definite")
    return (vectors*torch.log(values))@vectors.T

def _sym_sqrt(A):
    values,vectors=torch.linalg.eigh(.5*(A+A.T))
    if bool(torch.any(values<=0)): raise ValueError("elastic metric is not positive definite")
    return (vectors*torch.sqrt(values))@vectors.T

def update_multiplicative_j2(F: Tensor, young: float, poisson: float,
                             yield_stress: float, hardening: float,
                             committed: MultiplicativeJ2State):
    """Backward radial return in elastic logarithmic strain space.

    Returns first Piola stress, trial state, plastic dissipation and yield value.
    The update is exact for coaxial proportional paths and objective under a
    superposed spatial rigid rotation.
    """
    if F.shape!=(3,3) or committed.plastic_gradient.shape!=(3,3): raise ValueError("F and Fp must be 3x3")
    J=torch.linalg.det(F)
    if bool(J<=torch.finfo(F.dtype).eps): raise ValueError("total deformation gradient is inverted")
    G=young/(2*(1+poisson)); K=young/(3*(1-2*poisson)); I=torch.eye(3,dtype=F.dtype,device=F.device)
    Fe=F@torch.linalg.inv(committed.plastic_gradient); Ce=Fe.T@Fe
    Ee=.5*_sym_log(Ce); dev=Ee-torch.trace(Ee)/3*I
    Mdev=2*G*dev; q=torch.sqrt(1.5*torch.sum(Mdev*Mdev))
    f=q-(yield_stress+hardening*committed.alpha)
    if bool(f>0):
        dgamma=f/(3*G+hardening); N=1.5*Mdev/q
        Fp=torch.matrix_exp(dgamma*N)@committed.plastic_gradient
        alpha=committed.alpha+dgamma
        trial=MultiplicativeJ2State(Fp,alpha)
        dissipation=dgamma*(yield_stress+hardening*(committed.alpha+.5*dgamma))
    else:
        trial=committed; dgamma=torch.zeros_like(q); dissipation=torch.zeros_like(q)
    Fe=F@torch.linalg.inv(trial.plastic_gradient); Ce=Fe.T@Fe; Ee=.5*_sym_log(Ce)
    M=K*torch.trace(Ee)*I+2*G*(Ee-torch.trace(Ee)/3*I)
    Ue=_sym_sqrt(Ce); Re=Fe@torch.linalg.inv(Ue); tau=Re@M@Re.T
    P=tau@torch.linalg.inv(F).T
    final_q=torch.sqrt(1.5*torch.sum((M-torch.trace(M)/3*I)**2))
    yield_value=final_q-(yield_stress+hardening*trial.alpha)
    return P,trial,dissipation,yield_value

def _kinematics(model):
    x=model.reference_nodes[model.elements]; dn=x.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    g,d=_gradient(x,dn,"finite-plastic TET4")
    edofs=torch.stack(tuple(3*model.elements+i for i in range(3)),2).reshape(-1,12).long()
    return g,d/6,edofs

def assemble_finite_plastic(model: FiniteStrainTet4Model, displacement: Tensor,
                            committed: FinitePlasticState, *, yield_stress: float,
                            hardening: float, tangent=True, difference_step=2e-7):
    """Assemble residual and centred algorithmic tangent from committed state."""
    g,v,edofs=_kinematics(model)
    if len(committed.points)!=len(edofs): raise ValueError("integration-point state count mismatch")
    internal=torch.zeros_like(displacement); K=displacement.new_zeros((model.n_dofs,model.n_dofs))
    Ps=[]; states=[]; dissipations=[]
    def element(e,ue):
        F=torch.eye(3,dtype=ue.dtype,device=ue.device)+ue.reshape(4,3).T@g[e]
        P,state,diss,_=update_multiplicative_j2(F,model.young,model.poisson,yield_stress,hardening,committed.points[e])
        return (P@g[e].T).T.reshape(-1)*v[e],P,state,diss
    for e,dofs in enumerate(edofs):
        ue=displacement[dofs]; fe,P,state,diss=element(e,ue); internal.index_add_(0,dofs,fe)
        if tangent:
            ke=torch.empty((12,12),dtype=ue.dtype,device=ue.device)
            for j in range(12):
                h=difference_step*max(1.,abs(float(ue[j])))
                up=ue.clone(); um=ue.clone(); up[j]+=h; um[j]-=h
                ke[:,j]=(element(e,up)[0]-element(e,um)[0])/(2*h)
            K.index_put_((dofs[:,None],dofs[None,:]),ke,accumulate=True)
        Ps.append(P); states.append(state); dissipations.append(diss*v[e])
    return internal,K,torch.stack(Ps),FinitePlasticState(tuple(states)),torch.stack(dissipations)

def solve_finite_plastic_path(model,reference_force,targets: Sequence[float],*,yield_stress,
    hardening,initial: FinitePlasticResult|None=None,initial_increment=.05,minimum_increment=1e-4,
    maximum_increment=.15,tolerance=1e-8,max_iterations=20):
    force=reference_force.to(model.reference_nodes); fixed=model.fixed_dofs.to(device=force.device,dtype=torch.long)
    mask=torch.ones(model.n_dofs,dtype=torch.bool,device=force.device); mask[fixed]=False
    free=torch.arange(model.n_dofs,device=force.device)[mask]
    factor=0. if initial is None else initial.load_factor
    u=torch.zeros_like(force) if initial is None else initial.displacement.clone(); state=FinitePlasticState.virgin(model) if initial is None else initial.state
    out=[]
    for value in targets:
        target=float(value); start=float(factor); delta=target-start
        def evaluate(q,progress,base):
            trial_u=u.clone().index_copy(0,free,q)
            fi,kt,_,trial,_=assemble_finite_plastic(model,trial_u,base,yield_stress=yield_stress,hardening=hardening)
            return (start+delta*progress)*force[free]-fi[free],kt[free][:,free],trial
        advanced=solve_adaptive(evaluate,StepState(0.,u[free].clone(),state),target_factor=1.,initial_increment=initial_increment,
            minimum_increment=minimum_increment,maximum_increment=maximum_increment,tolerance=tolerance,max_iterations=max_iterations)
        u=u.clone().index_copy(0,free,advanced.displacement); state=advanced.material_state; factor=target
        fi,_,P,trial,diss=assemble_finite_plastic(model,u,state,yield_stress=yield_stress,hardening=hardening,tangent=False)
        state=trial; out.append(FinitePlasticResult(factor,u.clone(),fi-factor*force,P,state,diss,tuple(advanced.history)))
    return tuple(out)
