"""Adaptive non-proportional integration for multiplicative finite J2.

Path segments interpolate the total deformation gradient.  A full step and two
half steps estimate local state error; rejected segments recurse transactionally.
The tangent uses Richardson-extrapolated centred algorithmic differences.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .finite_strain_plasticity import MultiplicativeJ2State,update_multiplicative_j2

Tensor=torch.Tensor

@dataclass(frozen=True)
class AdaptiveFiniteJ2State:
    plastic: MultiplicativeJ2State
    deformation_gradient: Tensor

@dataclass(frozen=True)
class AdaptiveUpdate:
    first_piola: Tensor
    state: AdaptiveFiniteJ2State
    dissipation: Tensor
    yield_value: Tensor
    accepted_substeps: int
    estimated_error: float

def virgin_adaptive_state(*,dtype=torch.float64,device=None):
    I=torch.eye(3,dtype=dtype,device=device)
    return AdaptiveFiniteJ2State(MultiplicativeJ2State(I.clone(),torch.zeros((),dtype=dtype,device=device)),I)

def _state_error(a,b):
    fp=torch.linalg.vector_norm(a.plastic_gradient-b.plastic_gradient)/max(float(torch.linalg.vector_norm(b.plastic_gradient)),1.)
    alpha=torch.abs(a.alpha-b.alpha)/max(abs(float(b.alpha)),1.)
    return float(torch.maximum(fp,alpha))

def integrate_adaptive(F_target: Tensor,state: AdaptiveFiniteJ2State,young: float,poisson: float,
    yield_stress: float,hardening: float,*,rtol=1e-6,max_depth=12) -> AdaptiveUpdate:
    """Integrate one non-proportional segment with local state-error control."""
    if rtol<=0 or max_depth<0: raise ValueError("rtol must be positive and max_depth nonnegative")
    F0=state.deformation_gradient
    if F_target.shape!=(3,3) or F0.shape!=(3,3): raise ValueError("deformation gradients must be 3x3")
    def advance(Fa,Fb,base,depth):
        Pfull,sfull,dfull,yfull=update_multiplicative_j2(Fb,young,poisson,yield_stress,hardening,base)
        Fm=.5*(Fa+Fb)
        _,sm,d1,_=update_multiplicative_j2(Fm,young,poisson,yield_stress,hardening,base)
        Phalf,shalf,d2,yhalf=update_multiplicative_j2(Fb,young,poisson,yield_stress,hardening,sm)
        error=_state_error(sfull,shalf)
        if error<=rtol:
            return Phalf,shalf,d1+d2,yhalf,2,error
        if depth>=max_depth:
            raise RuntimeError(f"finite-plastic substepping tolerance not met: error={error:g}")
        _,sl,dl,_,nl,el=advance(Fa,Fm,base,depth+1)
        Pr,sr,dr,yr,nr,er=advance(Fm,Fb,sl,depth+1)
        return Pr,sr,dl+dr,yr,nl+nr,max(el,er)
    P,plastic,diss,y,n,error=advance(F0,F_target,state.plastic,0)
    return AdaptiveUpdate(P,AdaptiveFiniteJ2State(plastic,F_target.clone()),diss,y,n,error)

def integrate_path(path,state,young,poisson,yield_stress,hardening,**kwargs):
    """Commit an ordered deformation history; order is intentionally material."""
    results=[]
    for F in path:
        result=integrate_adaptive(F,state,young,poisson,yield_stress,hardening,**kwargs)
        results.append(result); state=result.state
    return tuple(results)

def richardson_algorithmic_tangent(F_target,state,young,poisson,yield_stress,hardening,
                                   *,step=2e-5,rtol=1e-7,max_depth=12):
    """Return dP/dF using O(h^4) Richardson-centred differences."""
    if step<=0: raise ValueError("step must be positive")
    def central(h):
        out=F_target.new_empty((3,3,3,3))
        for k in range(3):
            for l in range(3):
                plus=F_target.clone(); minus=F_target.clone(); plus[k,l]+=h; minus[k,l]-=h
                pp=integrate_adaptive(plus,state,young,poisson,yield_stress,hardening,rtol=rtol,max_depth=max_depth).first_piola
                pm=integrate_adaptive(minus,state,young,poisson,yield_stress,hardening,rtol=rtol,max_depth=max_depth).first_piola
                out[:,:,k,l]=(pp-pm)/(2*h)
        return out
    coarse=central(step); fine=central(step/2)
    return (4*fine-coarse)/3

def fixed_substep_reference(F_target,state,young,poisson,yield_stress,hardening,nsteps=1024):
    """Independent high-resolution reference using uniform path substeps."""
    if nsteps<1: raise ValueError("nsteps must be positive")
    F0=state.deformation_gradient; plastic=state.plastic; diss=F0.new_zeros(())
    for i in range(1,nsteps+1):
        F=F0+(F_target-F0)*(i/nsteps)
        P,plastic,d,y=update_multiplicative_j2(F,young,poisson,yield_stress,hardening,plastic); diss+=d
    return AdaptiveUpdate(P,AdaptiveFiniteJ2State(plastic,F_target.clone()),diss,y,nsteps,0.)
