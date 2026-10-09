"""Complete Newton path for two-compliant frictional surface patches."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .frictional_mortar import (
    FrictionalMortarAssembly, FrictionalMortarState,
    SymmetricFrictionalMortarAssembly, SymmetricFrictionalMortarState,
    assemble_frictional_mortar, assemble_symmetric_frictional_mortar,
    initial_frictional_mortar_state, initial_symmetric_frictional_mortar_state,
)
from .surface_surface_contact3d import SurfacePatchModel, _reflected_role_exchange


@dataclass(frozen=True)
class FrictionalSurfaceState:
    slave_displacement: torch.Tensor
    master_displacement: torch.Tensor
    contact: FrictionalMortarState | SymmetricFrictionalMortarState


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
    assembly: FrictionalMortarAssembly | SymmetricFrictionalMortarAssembly


def initial_frictional_surface_state(model: SurfacePatchModel):
    us=torch.zeros_like(model.slave);um=torch.zeros_like(model.master)
    contact=(initial_symmetric_frictional_mortar_state(model.slave,model.slave_faces,
        model.master,model.master_faces) if model.two_pass else
        initial_frictional_mortar_state(model.slave,model.slave_faces,
            model.master,model.master_faces))
    return FrictionalSurfaceState(us,um,contact)


def _system(model: SurfacePatchModel, state: FrictionalSurfaceState,
            committed: FrictionalSurfaceState, load: FrictionalSurfaceLoad, *,
            tangential_penalty: float, friction: float, tangent: bool=True):
    if model.two_pass:
        if not isinstance(committed.contact,SymmetricFrictionalMortarState):
            raise ValueError("two-pass model requires symmetric friction history")
        contact=assemble_symmetric_frictional_mortar(model.slave,model.slave_faces,
            model.master,model.master_faces,state.slave_displacement,
            state.master_displacement,committed.contact,
            normal_penalty=model.normal_penalty,tangential_penalty=tangential_penalty,
            friction=friction,tangent=tangent)
    else:
        if not isinstance(committed.contact,FrictionalMortarState):
            raise ValueError("one-pass model requires one-pass friction history")
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
    area_vector=u.new_zeros(3)
    for face in model.master_faces:
        tri=model.master[face[:3]]
        area_vector=area_vector+torch.linalg.cross(tri[1]-tri[0],tri[2]-tri[0])
    normal=area_vector/torch.linalg.vector_norm(area_vector)
    shear_direction=u.new_tensor([1.,0.,0.]);shear_direction=shear_direction-torch.dot(shear_direction,normal)*normal
    shear_direction=shear_direction/torch.linalg.vector_norm(shear_direction)
    slave_load=load.shear_pressure*shear_direction-load.normal_pressure*normal
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
           FrictionalSurfaceLoad(250.,250.))
    initial=initial_frictional_surface_state(model)
    steps=solve_frictional_surface_path(model,loads,tangential_penalty=kt,friction=mu,
                                         tolerance=2.e-9)
    last=steps[-1];result=last.assembly.result
    normal=float(torch.linalg.vector_norm(result.normal_resultant).detach())
    tangential=float(torch.linalg.vector_norm(result.tangential_resultant).detach())
    coulomb_error=abs(tangential/(mu*normal)-1)
    def points(contact):
        return (contact.forward.points+contact.reverse.points
                if isinstance(contact,SymmetricFrictionalMortarState) else contact.points)
    sticking=[all(float(p.dissipated_energy_density.detach())==0 for p in points(s.state.contact))
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
    before_s=initial.slave_displacement.clone();before_history=points(initial.contact)[0].elastic_slip.clone()
    failed=False
    try: solve_frictional_surface_path(model,[loads[-1]],tangential_penalty=kt,
        friction=mu,initial_state=initial,max_iterations=1,tolerance=1.e-14)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,before_s)
        and torch.equal(points(initial.contact)[0].elastic_slip,before_history))
    energy_partition_error=0.
    if isinstance(result.state,SymmetricFrictionalMortarState):
        expected=.5*(float(result.forward.dissipation_increment)
                     +float(result.reverse.dissipation_increment))
        energy_partition_error=abs(float(result.dissipation_increment)-expected)/max(1.,abs(expected))
    exchanged_model=_reflected_role_exchange(model)
    exchanged_loads=tuple(FrictionalSurfaceLoad(item.normal_pressure,-item.shear_pressure)
                          for item in loads)
    exchanged=solve_frictional_surface_path(exchanged_model,exchanged_loads,
        tangential_penalty=kt,friction=mu,tolerance=2.e-9)[-1].assembly.result
    interchange=max(abs(float(torch.linalg.vector_norm(exchanged.normal_resultant))/normal-1),
        abs(float(torch.linalg.vector_norm(exchanged.tangential_resultant))/tangential-1))
    passed=(sticking[0] and sticking[1] and not sticking[2]
        and result.dissipation_increment>0 and coulomb_error<.03
        and force_imbalance<1.e-9 and residual_moment<1.e-7 and rollback
        and energy_partition_error<1.e-12 and interchange<.03)
    if not passed: raise AssertionError("frictional surface path qualification failed: "
        f"stick={sticking}, coulomb={coulomb_error}, force={force_imbalance}, "
        f"moment={residual_moment}, rollback={rollback}, energy={energy_partition_error}, "
        f"interchange={interchange}")
    return {"schema":"tensorfem.frictional-surface-path-qualification/1.0",
        "stick_history":sticking,"normal_resultant":normal,
        "tangential_resultant":tangential,"coulomb_relative_error":coulomb_error,
        "force_imbalance":force_imbalance,"equilibrium_moment_residual":residual_moment,
        "rollback_exact":rollback,"two_pass_energy_partition_error":energy_partition_error,
        "master_slave_interchange_relative_error":interchange,
        "general_surface_to_surface":"blocked",
        "symmetric_two_pass":model.two_pass,
        "remaining_blockers":["curved friction path mesh convergence",
            "finite-strain contact","self-contact"],
        "passed":True}


def run_frictional_surface_mesh_qualification():
    """Run the opt-in 2/3, 3/4 and 4/5 curved friction qualification."""
    radius,gap,foundation,penalty=4.,.005,2.e4,1.e6
    applied,mu,kt=250.,.3,5.e4
    indentation=2*applied/foundation-gap
    effective=1/(1/penalty+2/foundation)
    normal_oracle=math.pi*effective*radius*indentation**2
    loads=(FrictionalSurfaceLoad(applied,0.),FrictionalSurfaceLoad(applied,2.),
           FrictionalSurfaceLoad(applied,250.))
    rows=[]; finest=None; finest_model=None
    for slave_cells,master_cells in ((2,3),(3,4),(4,5)):
        from .surface_surface_contact3d import build_parabolic_surface_model
        model=build_parabolic_surface_model(cells=slave_cells,master_cells=master_cells,
            radius=radius,clearance=gap,foundation=foundation,normal_penalty=penalty)
        steps=solve_frictional_surface_path(model,loads,tangential_penalty=kt,
                                             friction=mu,tolerance=2.e-9)
        normal_step=steps[0].assembly.result;sliding=steps[-1].assembly.result
        normal=float(torch.linalg.vector_norm(normal_step.normal_resultant))
        final_normal=float(torch.linalg.vector_norm(sliding.normal_resultant))
        tangential=float(torch.linalg.vector_norm(sliding.tangential_resultant))
        points=sliding.state.forward.points+sliding.state.reverse.points
        row={"slave_cells":slave_cells,"master_cells":master_cells,
            "normal_load":normal,"normal_oracle":normal_oracle,
            "normal_relative_error":abs(normal/normal_oracle-1),
            "coulomb_relative_error":abs(tangential/(mu*final_normal)-1),
            "dissipation_increment":float(sliding.dissipation_increment),
            "sliding_points":sum(float(p.dissipated_energy_density)>0 for p in points),
            "iterations":[step.iterations for step in steps],
            "force_imbalance":float(torch.linalg.vector_norm(
                sliding.slave_forces.sum(0)+sliding.master_forces.sum(0)))}
        rows.append(row);finest=steps;finest_model=model
    assert finest is not None and finest_model is not None
    exchanged_model=_reflected_role_exchange(finest_model)
    exchanged_loads=tuple(FrictionalSurfaceLoad(item.normal_pressure,-item.shear_pressure)
                          for item in loads)
    exchanged=solve_frictional_surface_path(exchanged_model,exchanged_loads,
        tangential_penalty=kt,friction=mu,tolerance=2.e-9)
    original_result=finest[-1].assembly.result;exchanged_result=exchanged[-1].assembly.result
    interchange=max(abs(float(torch.linalg.vector_norm(exchanged_result.normal_resultant))
                         /float(torch.linalg.vector_norm(original_result.normal_resultant))-1),
        abs(float(torch.linalg.vector_norm(exchanged_result.tangential_resultant))
            /float(torch.linalg.vector_norm(original_result.tangential_resultant))-1))
    initial=initial_frictional_surface_state(finest_model);before=initial.slave_displacement.clone()
    old=initial.contact.forward.points[0].elastic_slip.clone();failed=False
    try: solve_frictional_surface_path(finest_model,[loads[-1]],tangential_penalty=kt,
        friction=mu,initial_state=initial,max_iterations=1,tolerance=1.e-14)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,before)
                  and torch.equal(initial.contact.forward.points[0].elastic_slip,old))
    errors=[row["normal_relative_error"] for row in rows]
    passed=(errors[-1]<.03 and errors[-1]<errors[-2]<errors[-3]
        and max(row["coulomb_relative_error"] for row in rows)<.03
        and all(row["dissipation_increment"]>0 and row["sliding_points"]>0 for row in rows)
        and max(row["force_imbalance"] for row in rows)<1.e-9
        and interchange<.03 and rollback)
    if not passed:
        raise AssertionError("frictional curved mesh qualification failed: "
            f"normal_errors={errors}, coulomb={[r['coulomb_relative_error'] for r in rows]}, "
            f"interchange={interchange}, rollback={rollback}")
    return {"schema":"tensorfem.frictional-surface-mesh-qualification/1.0",
        "normal_oracle":"F=pi*keff*R*delta^2, keff^-1=kn^-1+ks^-1+km^-1",
        "mesh_sequence":rows,"master_slave_interchange_relative_error":interchange,
        "rollback_exact":rollback,
        "objectivity_evidence":"symmetric frictional Mortar finite-rotation covariance gate",
        "general_surface_to_surface":"qualified_curved_frictional_small_strain_subset",
        "remaining_boundaries":["finite-strain contact","self-contact","impact"],
        "passed":True}
