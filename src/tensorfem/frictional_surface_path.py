"""Complete Newton path for two-compliant frictional surface patches."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .frictional_mortar import (
    FrictionalMortarAssembly, FrictionalMortarState,
    assemble_frictional_mortar, initial_frictional_mortar_state,
)
from .surface_surface_contact3d import SurfacePatchModel


@dataclass(frozen=True)
class FrictionalSurfaceState:
    slave_displacement: torch.Tensor
    master_displacement: torch.Tensor
    contact: FrictionalMortarState


@dataclass(frozen=True)
class FrictionalSurfaceLoad:
    normal_pressure: float
    shear_pressure: float


@dataclass(frozen=True)
class FrictionalSurfaceStep:
    state: FrictionalSurfaceState
    load: FrictionalSurfaceLoad
    iterations: int
    residual_norm: float
    assembly: FrictionalMortarAssembly


def initial_frictional_surface_state(model: SurfacePatchModel):
    us=torch.zeros_like(model.slave);um=torch.zeros_like(model.master)
    contact=initial_frictional_mortar_state(model.slave,model.slave_faces,
        model.master,model.master_faces)
    return FrictionalSurfaceState(us,um,contact)


def _system(model: SurfacePatchModel, state: FrictionalSurfaceState,
            committed: FrictionalSurfaceState, load: FrictionalSurfaceLoad, *,
            tangential_penalty: float, friction: float, tangent: bool=True):
    contact=assemble_frictional_mortar(model.slave,model.slave_faces,
        model.master,model.master_faces,state.slave_displacement,
        state.master_displacement,committed.contact,
        normal_penalty=model.normal_penalty,tangential_penalty=tangential_penalty,
        friction=friction,tangent=tangent)
    ns=model.slave.numel();u=torch.cat((state.slave_displacement.reshape(-1),
                                       state.master_displacement.reshape(-1)))
    stiffness=torch.zeros_like(u)
    stiffness[:ns].reshape(-1,3)[:]=model.slave_foundation*model.slave_weights[:,None]
    stiffness[ns:].reshape(-1,3)[:]=model.master_foundation*model.master_weights[:,None]
    external=torch.zeros_like(u)
    slave_load=torch.tensor([load.shear_pressure,0.,-load.normal_pressure],dtype=u.dtype)
    master_load=-slave_load
    external[:ns].reshape(-1,3)[:]=model.slave_weights[:,None]*slave_load
    external[ns:].reshape(-1,3)[:]=model.master_weights[:,None]*master_load
    residual=stiffness*u-external+contact.residual
    matrix=contact.tangent+torch.diag(stiffness) if tangent else contact.tangent
    return contact,residual,matrix,external,stiffness


def solve_frictional_surface_path(model: SurfacePatchModel,
    loads: Sequence[FrictionalSurfaceLoad], *, tangential_penalty: float,
    friction: float, initial_state: FrictionalSurfaceState|None=None,
    tolerance: float=1.e-9, max_iterations: int=30):
    """Solve load steps and commit Coulomb history only after convergence."""
    if tangential_penalty<=0 or friction<0:
        raise ValueError("invalid friction parameters")
    committed=initial_frictional_surface_state(model) if initial_state is None else initial_state
    steps=[]
    for load in loads:
        if (not math.isfinite(load.normal_pressure) or load.normal_pressure<0
                or not math.isfinite(load.shear_pressure)):
            raise ValueError("loads must be finite and normal pressure nonnegative")
        trial=FrictionalSurfaceState(committed.slave_displacement.clone(),
            committed.master_displacement.clone(),committed.contact)
        converged=False;norm=float("inf")
        for iteration in range(1,max_iterations+1):
            assembly,residual,matrix,external,_=_system(model,trial,committed,load,
                tangential_penalty=tangential_penalty,friction=friction)
            norm=float(torch.linalg.vector_norm(residual));scale=max(1.,float(torch.linalg.vector_norm(external)))
            if norm<=tolerance*scale: converged=True;break
            delta=torch.linalg.solve(matrix,-residual)
            joined=torch.cat((trial.slave_displacement.reshape(-1),trial.master_displacement.reshape(-1)))+delta
            ns=model.slave.numel()
            trial=FrictionalSurfaceState(joined[:ns].reshape_as(model.slave),
                joined[ns:].reshape_as(model.master),committed.contact)
        if not converged:
            raise RuntimeError("frictional surface Newton solve failed; committed state unchanged")
        final,residual,_,_,_=_system(model,trial,committed,load,
            tangential_penalty=tangential_penalty,friction=friction,tangent=False)
        committed=FrictionalSurfaceState(trial.slave_displacement,
            trial.master_displacement,final.result.state)
        steps.append(FrictionalSurfaceStep(committed,load,iteration,
            float(torch.linalg.vector_norm(residual)),final))
    return tuple(steps)


def run_frictional_surface_path_qualification(model: SurfacePatchModel):
    """Run stick-to-slip, balance and rollback gates on a supplied patch."""
    mu=.3;kt=5.e4
    loads=(FrictionalSurfaceLoad(250.,0.),FrictionalSurfaceLoad(250.,2.),
           FrictionalSurfaceLoad(250.,80.))
    initial=initial_frictional_surface_state(model)
    steps=solve_frictional_surface_path(model,loads,tangential_penalty=kt,friction=mu,
                                         tolerance=2.e-9)
    last=steps[-1];result=last.assembly.result
    normal=float(torch.linalg.vector_norm(result.normal_resultant).detach())
    tangential=float(torch.linalg.vector_norm(result.tangential_resultant).detach())
    coulomb_error=abs(tangential/(mu*normal)-1)
    sticking=[all(float(p.dissipated_energy_density.detach())==0 for p in s.state.contact.points)
              for s in steps]
    force_imbalance=float(torch.linalg.vector_norm(
        result.slave_forces.sum(0)+result.master_forces.sum(0)).detach())
    # Nodal equilibrium includes foundation reactions and external actions;
    # its moment is the appropriate global balance for a penalty friction couple.
    _,residual,_,_,_=_system(model,last.state,steps[-2].state,last.load,
        tangential_penalty=kt,friction=mu,tangent=False)
    current=torch.cat((model.slave+last.state.slave_displacement,
                       model.master+last.state.master_displacement))
    residual_moment=float(torch.linalg.vector_norm(
        torch.linalg.cross(current,residual.reshape(-1,3)).sum(0)).detach())
    before_s=initial.slave_displacement.clone();before_history=initial.contact.points[0].elastic_slip.clone()
    failed=False
    try: solve_frictional_surface_path(model,[loads[-1]],tangential_penalty=kt,
        friction=mu,initial_state=initial,max_iterations=1,tolerance=1.e-14)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,before_s)
        and torch.equal(initial.contact.points[0].elastic_slip,before_history))
    passed=(sticking[0] and sticking[1] and not sticking[2]
        and result.dissipation_increment>0 and coulomb_error<.03
        and force_imbalance<1.e-9 and residual_moment<1.e-7 and rollback)
    if not passed: raise AssertionError("frictional surface path qualification failed")
    return {"schema":"tensorfem.frictional-surface-path-qualification/1.0",
        "stick_history":sticking,"normal_resultant":normal,
        "tangential_resultant":tangential,"coulomb_relative_error":coulomb_error,
        "force_imbalance":force_imbalance,"equilibrium_moment_residual":residual_moment,
        "rollback_exact":rollback,"general_surface_to_surface":"blocked",
        "remaining_blockers":["curved friction path mesh convergence",
            "symmetric two-pass friction history","finite-strain contact"],
        "passed":True}
