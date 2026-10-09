"""Total-Lagrangian finite-strain TET4 hyperelasticity.

This is a genuine finite-deformation elastic foundation.  It intentionally
does not expose the small-strain J2 state as finite-strain plasticity: no
multiplicative plastic deformation gradient is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .nonlinear_step import IncrementRecord, StepState, solve_adaptive
from .solid3d import _gradient

Tensor = torch.Tensor


@dataclass(frozen=True)
class FiniteStrainTet4Model:
    reference_nodes: Tensor
    elements: Tensor
    young: float
    poisson: float
    fixed_dofs: Tensor

    def __post_init__(self):
        if self.reference_nodes.ndim != 2 or self.reference_nodes.shape[1] != 3:
            raise ValueError("reference_nodes must have shape [n,3]")
        if self.elements.dtype != torch.long or self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise TypeError("TET4 elements must be a [ne,4] long tensor")
        if self.young <= 0 or not (-1. < self.poisson < .5):
            raise ValueError("invalid compressible elastic constants")

    @property
    def n_dofs(self): return 3*self.reference_nodes.shape[0]

    @property
    def lame(self):
        mu=self.young/(2*(1+self.poisson))
        lam=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson))
        return lam,mu


@dataclass(frozen=True)
class FiniteElasticResult:
    load_factor: float
    displacement: Tensor
    reaction: Tensor
    first_piola: Tensor
    jacobian: Tensor
    strain_energy: Tensor
    increments: tuple[IncrementRecord,...]


def neo_hookean_response(F: Tensor, young: float, poisson: float
                         ) -> tuple[Tensor,Tensor,Tensor]:
    """Energy, first Piola stress and Cauchy stress for compressible NH."""
    if F.shape[-2:] != (3,3): raise ValueError("F must end in shape [3,3]")
    J=torch.linalg.det(F)
    if bool(torch.any(J <= torch.finfo(F.dtype).eps)):
        raise ValueError("deformation gradient is inverted or singular")
    mu=young/(2*(1+poisson)); lam=young*poisson/((1+poisson)*(1-2*poisson))
    logJ=torch.log(J); FinvT=torch.linalg.inv(F).transpose(-1,-2)
    I1=torch.sum(F*F,dim=(-2,-1))
    energy=.5*mu*(I1-3)-mu*logJ+.5*lam*logJ**2
    P=mu*(F-FinvT)+lam*logJ[...,None,None]*FinvT
    sigma=(P@F.transpose(-1,-2))/J[...,None,None]
    return energy,P,sigma


def _reference_kinematics(model):
    if model.elements.numel()==0: raise ValueError("at least one element is required")
    if int(model.elements.min())<0 or int(model.elements.max())>=model.reference_nodes.shape[0]:
        raise IndexError("element node outside model")
    x=model.reference_nodes[model.elements]
    dn=x.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    gradients,det=_gradient(x,dn,"finite-strain TET4")
    edofs=torch.stack(tuple(3*model.elements+i for i in range(3)),2).reshape(-1,12)
    return gradients,det/6,edofs.long()


def assemble_finite_tet4(model: FiniteStrainTet4Model, displacement: Tensor,
                         *, tangent: bool=True):
    """Assemble TL internal force and exact AD material/geometric tangent."""
    if displacement.shape != (model.n_dofs,): raise ValueError("displacement size differs from model")
    gradients,volumes,edofs=_reference_kinematics(model)
    internal=torch.zeros_like(displacement); tang=displacement.new_zeros((model.n_dofs,model.n_dofs))
    stresses=[]; jacobians=[]; energies=[]
    for e,dofs in enumerate(edofs):
        ue=displacement[dofs]
        def force_energy(local):
            u=local.reshape(4,3)
            F=torch.eye(3,dtype=u.dtype,device=u.device)+u.T@gradients[e]
            psi,P,_=neo_hookean_response(F,model.young,model.poisson)
            return (P@gradients[e].T).T.reshape(-1)*volumes[e],psi,F
        fe,psi,F=force_energy(ue)
        _,P,_=neo_hookean_response(F,model.young,model.poisson)
        internal.index_add_(0,dofs,fe)
        if tangent:
            ke=torch.autograd.functional.jacobian(lambda q: force_energy(q)[0],ue,
                                                   create_graph=False)
            tang.index_put_((dofs[:,None],dofs[None,:]),ke,accumulate=True)
        stresses.append(P); jacobians.append(torch.linalg.det(F)); energies.append(psi*volumes[e])
    return internal,tang,torch.stack(stresses),torch.stack(jacobians),torch.stack(energies)


def solve_finite_elastic_path(model: FiniteStrainTet4Model, reference_force: Tensor,
                              targets: Sequence[float], *, initial: FiniteElasticResult|None=None,
                              initial_increment=.1, minimum_increment=1e-5,
                              maximum_increment=.25, tolerance=1e-9,max_iterations=25
                              ) -> tuple[FiniteElasticResult,...]:
    """Adaptive proportional finite-deformation equilibrium with restart."""
    _reference_kinematics(model)
    force=reference_force.to(model.reference_nodes)
    if force.shape!=(model.n_dofs,): raise ValueError("reference_force size differs from model")
    fixed=model.fixed_dofs.to(device=force.device,dtype=torch.long)
    if fixed.numel()!=torch.unique(fixed).numel(): raise ValueError("fixed_dofs must be unique")
    if fixed.numel() and (int(fixed.min())<0 or int(fixed.max())>=model.n_dofs):
        raise IndexError("fixed DOF outside model")
    mask=torch.ones(model.n_dofs,dtype=torch.bool,device=force.device); mask[fixed]=False
    free=torch.arange(model.n_dofs,device=force.device)[mask]
    if free.numel()==0: raise ValueError("model has no free DOFs")
    factor=0. if initial is None else float(initial.load_factor)
    u=torch.zeros_like(force) if initial is None else initial.displacement.to(force).clone()
    outputs=[]
    for value in targets:
        target=float(value); start=factor; delta=target-start
        if delta==0: histories=()
        else:
            def evaluate(q,progress,state):
                trial=u.clone().index_copy(0,free,q)
                internal,k,_,_,_=assemble_finite_tet4(model,trial)
                return (start+delta*progress)*force[free]-internal[free],k[free][:,free],state
            advanced=solve_adaptive(evaluate,StepState(0.,u[free].clone(),None),target_factor=1.,
                initial_increment=initial_increment,minimum_increment=minimum_increment,
                maximum_increment=maximum_increment,tolerance=tolerance,max_iterations=max_iterations)
            u=u.clone().index_copy(0,free,advanced.displacement); histories=tuple(advanced.history)
        factor=target
        internal,_,P,J,energy=assemble_finite_tet4(model,u,tangent=False)
        outputs.append(FiniteElasticResult(factor,u.clone(),internal-factor*force,P,J,energy,histories))
    return tuple(outputs)
