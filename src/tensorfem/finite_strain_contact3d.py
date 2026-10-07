"""Minimal double-deformable finite-strain solid/contact Newton closure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import hashlib
import json
import torch

from .finite_strain_elasticity import FiniteStrainTet4Model, assemble_finite_tet4
from .mortar_contact3d import MortarContactAssembly, assemble_mortar_contact
from .frictional_mortar import (
    SymmetricFrictionalMortarAssembly, SymmetricFrictionalMortarState,
    assemble_symmetric_frictional_mortar,
    initial_symmetric_frictional_mortar_state,
)


@dataclass(frozen=True)
class FiniteStrainContactModel:
    solid: FiniteStrainTet4Model
    slave_nodes: torch.Tensor
    slave_faces: torch.Tensor
    master_nodes: torch.Tensor
    master_faces: torch.Tensor
    normal_penalty: float


@dataclass(frozen=True)
class FiniteStrainContactStep:
    displacement: torch.Tensor
    load: torch.Tensor
    iterations: int
    residual_norm: float
    contact: MortarContactAssembly
    strain_energy: torch.Tensor
    jacobian: torch.Tensor


@dataclass(frozen=True)
class FiniteStrainFrictionState:
    displacement: torch.Tensor
    contact: SymmetricFrictionalMortarState


@dataclass(frozen=True)
class FiniteStrainFrictionStep:
    state: FiniteStrainFrictionState
    load: torch.Tensor
    iterations: int
    residual_norm: float
    contact: SymmetricFrictionalMortarAssembly
    strain_energy: torch.Tensor
    jacobian: torch.Tensor


def build_two_block_contact(*, clearance: float=.02, young: float=1.e3,
                            poisson: float=.3, normal_penalty: float=1.e5):
    """Build two one-cube TET4 blocks with opposed deformable QUAD4 faces."""
    if clearance<=0 or normal_penalty<=0: raise ValueError("clearance and penalty must be positive")
    nodes=[]
    for z0,z1 in ((-1.,0.),(clearance,1.+clearance)):
        nodes.extend((x,y,z) for x in (0.,1.) for z in (z0,z1) for y in (0.,1.))
    x=torch.tensor(nodes,dtype=torch.float64)
    pattern=((0,6,4,7),(0,2,6,7),(0,3,2,7),(0,1,3,7),(0,5,1,7),(0,4,5,7))
    elements=[]
    for offset in (0,8): elements.extend(tuple(offset+i for i in tet) for tet in pattern)
    fixed=[]
    for node,(_,_,z) in enumerate(nodes):
        if z in (-1.,1.+clearance): fixed.extend((3*node,3*node+1,3*node+2))
    solid=FiniteStrainTet4Model(x,torch.tensor(elements,dtype=torch.long),young,poisson,
        torch.tensor(fixed,dtype=torch.long))
    lookup={(float(a),float(b),float(c)):i for i,(a,b,c) in enumerate(nodes)}
    master=torch.tensor([lookup[(0.,0.,0.)],lookup[(1.,0.,0.)],lookup[(1.,1.,0.)],lookup[(0.,1.,0.)]])
    slave=torch.tensor([lookup[(0.,0.,clearance)],lookup[(0.,1.,clearance)],
                        lookup[(1.,1.,clearance)],lookup[(1.,0.,clearance)]])
    face=torch.tensor([[0,1,2,3]],dtype=torch.long)
    return FiniteStrainContactModel(solid,slave,face,master,face,normal_penalty)


def build_curved_nonmatching_two_block_contact(*, master_cells: int,
    slave_cells: int, clearance: float=.02, radius: float=8.,young: float=1.e3,
    poisson: float=.3,normal_penalty: float=1.e5,lateral_size: float=1.,
    block_depth: float=1.,center_grading: float=0.,vertical_cells: int=1,
    vertical_grading: float=1., symmetry_planes: bool=False):
    """Build independently meshed TET4 blocks with a shallow curved slave face."""
    for name,value in (("master_cells",master_cells),("slave_cells",slave_cells)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f"{name} must be a positive integer")
    if (not math.isfinite(clearance) or min(radius,normal_penalty,lateral_size,block_depth)<=0):
        raise ValueError("clearance must be finite and radius/penalty positive")
    if not math.isfinite(center_grading) or center_grading<0:
        raise ValueError("center_grading must be finite and nonnegative")
    if isinstance(vertical_cells,bool) or not isinstance(vertical_cells,int) or vertical_cells<1:
        raise ValueError("vertical_cells must be a positive integer")
    if not math.isfinite(vertical_grading) or vertical_grading<1:
        raise ValueError("vertical_grading must be at least one")
    nodes=[];elements=[];fixed=[];surface_nodes=[];surface_faces=[]
    pattern=((0,6,4,7),(0,2,6,7),(0,3,2,7),(0,1,3,7),(0,5,1,7),(0,4,5,7))

    def block(cells: int, *, upper: bool):
        offset=len(nodes);index={}
        for i in range(cells+1):
            for k in range(vertical_cells+1):
                for j in range(cells+1):
                    def coordinate(index):
                        s=2*index/cells-1
                        mapped=(s if center_grading==0 else
                                math.sinh(center_grading*s)/math.sinh(center_grading))
                        return lateral_size*(mapped+1)/2
                    x=coordinate(i);y=coordinate(j)
                    curve=((x-lateral_size/2)**2+(y-lateral_size/2)**2)/(2*radius) if upper else 0.
                    t=k/vertical_cells
                    z=(clearance+curve+block_depth*t**vertical_grading if upper else
                       -block_depth+block_depth*(1-(1-t)**vertical_grading))
                    index[i,k,j]=len(nodes);nodes.append((x,y,z))
                    if (upper and k==vertical_cells) or (not upper and k==0):
                        fixed.extend((3*(len(nodes)-1),3*(len(nodes)-1)+1,3*(len(nodes)-1)+2))
        for k in range(vertical_cells):
          for i in range(cells):
            for j in range(cells):
                local=(index[i,k,j],index[i,k,j+1],index[i,k+1,j],index[i,k+1,j+1],
                       index[i+1,k,j],index[i+1,k,j+1],index[i+1,k+1,j],index[i+1,k+1,j+1])
                elements.extend(tuple(local[q] for q in tet) for tet in pattern)
        layer=0 if upper else vertical_cells
        snodes=[index[i,layer,j] for i in range(cells+1) for j in range(cells+1)]
        lookup={node:n for n,node in enumerate(snodes)};faces=[]
        for i in range(cells):
            for j in range(cells):
                ids=(index[i,layer,j],index[i+1,layer,j],index[i+1,layer,j+1],index[i,layer,j+1])
                faces.append(tuple(lookup[q] for q in ids))
        return torch.tensor(snodes),torch.tensor(faces,dtype=torch.long)

    master_nodes,master_faces=block(master_cells,upper=False)
    slave_nodes,slave_faces=block(slave_cells,upper=True)
    # x>=0,y>=0 is a quarter-domain. Reflection planes constrain the normal
    # displacement on every solid node lying on x=0 or y=0.
    if symmetry_planes:
        tol=max(1.e-12,lateral_size*1.e-12)
        for node,(x,y,_z) in enumerate(nodes):
            if abs(x)<=tol: fixed.append(3*node)
            if abs(y)<=tol: fixed.append(3*node+1)
    solid=FiniteStrainTet4Model(torch.tensor(nodes,dtype=torch.float64),
        torch.tensor(elements,dtype=torch.long),young,poisson,
        torch.tensor(sorted(set(fixed)),dtype=torch.long))
    # Fail immediately if the structured tetrahedralisation or curvature inverted.
    assemble_finite_tet4(solid,torch.zeros(solid.n_dofs,dtype=torch.float64),tangent=False)
    return FiniteStrainContactModel(solid,slave_nodes,slave_faces,
        master_nodes,master_faces,normal_penalty)


def _contact(model: FiniteStrainContactModel,u: torch.Tensor,*,tangent: bool=True):
    us=u.reshape(-1,3)[model.slave_nodes]
    um=u.reshape(-1,3)[model.master_nodes]
    return assemble_mortar_contact(model.solid.reference_nodes[model.slave_nodes],
        model.slave_faces,model.solid.reference_nodes[model.master_nodes],model.master_faces,
        us,um,normal_penalty=model.normal_penalty,tangent=tangent)


def _assemble(model,u,external,*,tangent=True):
    internal,k,_,jacobian,energy=assemble_finite_tet4(model.solid,u,tangent=tangent)
    contact=_contact(model,u,tangent=tangent)
    ids=torch.cat(tuple(3*nodes[:,None]+torch.arange(3) for nodes in
        (model.slave_nodes,model.master_nodes))).reshape(-1)
    residual=internal-external
    residual=residual.index_add(0,ids,contact.residual)
    if tangent:
        k=k.clone();k.index_put_((ids[:,None],ids[None,:]),contact.tangent,accumulate=True)
    return residual,k,contact,energy,jacobian


def solve_finite_strain_contact_path(model: FiniteStrainContactModel,
    loads: Sequence[torch.Tensor], *, initial_displacement: torch.Tensor|None=None,
    tolerance: float=1.e-9,max_iterations: int=30):
    """Solve finite-strain solid/contact increments and commit only convergence."""
    ndof=model.solid.n_dofs
    committed=torch.zeros(ndof,dtype=model.solid.reference_nodes.dtype) if initial_displacement is None else initial_displacement.clone()
    if committed.shape!=(ndof,): raise ValueError("initial displacement size mismatch")
    mask=torch.ones(ndof,dtype=torch.bool);mask[model.solid.fixed_dofs]=False
    free=torch.nonzero(mask).flatten();steps=[]
    for load_index,load in enumerate(loads):
        if load.shape!=(ndof,): raise ValueError("load size mismatch")
        trial=committed.clone();converged=False;norm=float("inf")
        for iteration in range(1,max_iterations+1):
            residual,k,contact,energy,jacobian=_assemble(model,trial,load)
            norm=float(torch.linalg.vector_norm(residual[free]));scale=max(1.,float(torch.linalg.vector_norm(load[free])))
            if norm<=tolerance*scale: converged=True;break
            kfree=k[free][:,free]
            delta=torch.linalg.solve(kfree,-residual[free]);accepted=False
            line_residuals=[]
            for power in range(9):
                candidate=trial.clone();candidate[free]+=(.5**power)*delta
                try: candidate_residual,_,_,_,_=_assemble(model,candidate,load,tangent=False)
                except ValueError: continue
                if float(torch.linalg.vector_norm(candidate_residual[free]))<norm:
                    trial=candidate;accepted=True;break
            if not accepted: break
        if not converged:
            raise RuntimeError("finite-strain contact Newton failed; committed displacement unchanged")
        residual,_,contact,energy,jacobian=_assemble(model,trial,load,tangent=False)
        committed=trial
        steps.append(FiniteStrainContactStep(trial.clone(),load.clone(),iteration,
            float(torch.linalg.vector_norm(residual[free])),contact,energy.detach().clone(),jacobian.detach().clone()))
    return tuple(steps)


def balanced_face_load(model: FiniteStrainContactModel,*,normal: float,shear: float=0.):
    """Return equal/opposite nominal tractions on the two contact faces."""
    load=torch.zeros(model.solid.n_dofs,dtype=model.solid.reference_nodes.dtype)
    direction=torch.tensor([shear,0.,-normal],dtype=load.dtype)
    def weights(nodes,faces):
        x=model.solid.reference_nodes[nodes];w=torch.zeros(len(nodes),dtype=load.dtype)
        for face in faces:
            q=x[face];projected=.5*abs(float(torch.linalg.cross(q[1]-q[0],q[2]-q[0])[2]))
            projected+=.5*abs(float(torch.linalg.cross(q[2]-q[0],q[3]-q[0])[2]))
            w[face]+=projected/4
        return w
    sw=weights(model.slave_nodes,model.slave_faces);mw=weights(model.master_nodes,model.master_faces)
    load.reshape(-1,3)[model.slave_nodes]+=sw[:,None]*direction
    load.reshape(-1,3)[model.master_nodes]-=mw[:,None]*direction
    return load


def initial_finite_strain_friction_state(model: FiniteStrainContactModel):
    slave=model.solid.reference_nodes[model.slave_nodes]
    master=model.solid.reference_nodes[model.master_nodes]
    history=initial_symmetric_frictional_mortar_state(
        slave,model.slave_faces,master,model.master_faces)
    return FiniteStrainFrictionState(torch.zeros(model.solid.n_dofs,dtype=slave.dtype),history)


def _assemble_frictional(model,state,committed,external,*,tangential_penalty,friction,tangent=True):
    u=state.displacement
    internal,k,_,jacobian,energy=assemble_finite_tet4(model.solid,u,tangent=tangent)
    us=u.reshape(-1,3)[model.slave_nodes];um=u.reshape(-1,3)[model.master_nodes]
    contact=assemble_symmetric_frictional_mortar(
        model.solid.reference_nodes[model.slave_nodes],model.slave_faces,
        model.solid.reference_nodes[model.master_nodes],model.master_faces,
        us,um,committed.contact,normal_penalty=model.normal_penalty,
        tangential_penalty=tangential_penalty,friction=friction,tangent=tangent)
    ids=torch.cat(tuple(3*nodes[:,None]+torch.arange(3) for nodes in
        (model.slave_nodes,model.master_nodes))).reshape(-1)
    residual=(internal-external).index_add(0,ids,contact.residual)
    if tangent:
        k=k.clone();k.index_put_((ids[:,None],ids[None,:]),contact.tangent,accumulate=True)
    return residual,k,contact,energy,jacobian


def solve_finite_strain_friction_path(model: FiniteStrainContactModel,
    loads: Sequence[torch.Tensor], *, tangential_penalty: float, friction: float,
    initial_state: FiniteStrainFrictionState|None=None,tolerance: float=1.e-9,
    max_iterations: int=30,initial_trial_displacement: torch.Tensor|None=None):
    """Solve and transactionally commit finite-strain symmetric friction."""
    if tangential_penalty<=0 or friction<0: raise ValueError("invalid friction parameters")
    committed=initial_finite_strain_friction_state(model) if initial_state is None else initial_state
    fixed=model.solid.fixed_dofs;mask=torch.ones(model.solid.n_dofs,dtype=torch.bool);mask[fixed]=False
    free=torch.nonzero(mask).flatten();steps=[]
    for load_index,load in enumerate(loads):
        if load.shape!=(model.solid.n_dofs,): raise ValueError("load size mismatch")
        guess=(initial_trial_displacement if load_index==0 and initial_trial_displacement is not None
               else committed.displacement)
        if guess.shape!=(model.solid.n_dofs,): raise ValueError("initial trial displacement size mismatch")
        trial=FiniteStrainFrictionState(guess.detach().clone(),committed.contact)
        converged=False;norm=float("inf")
        for iteration in range(1,max_iterations+1):
            residual,k,contact,energy,jacobian=_assemble_frictional(model,trial,committed,load,
                tangential_penalty=tangential_penalty,friction=friction)
            norm=float(torch.linalg.vector_norm(residual[free]));scale=max(1.,float(torch.linalg.vector_norm(load[free])))
            if norm<=tolerance*scale: converged=True;break
            kfree=k[free][:,free]
            delta=torch.linalg.solve(kfree,-residual[free]);accepted=False
            line_residuals=[]
            for power in range(9):
                candidate=trial.displacement.clone();candidate[free]+=(.5**power)*delta
                candidate_state=FiniteStrainFrictionState(candidate,committed.contact)
                try: r,_,_,_,_=_assemble_frictional(model,candidate_state,committed,load,
                    tangential_penalty=tangential_penalty,friction=friction,tangent=False)
                except ValueError: continue
                candidate_norm=float(torch.linalg.vector_norm(r[free]));line_residuals.append(candidate_norm)
                if candidate_norm<norm:
                    trial=candidate_state;accepted=True;break
            if not accepted: break
        if not converged:
            singular=torch.linalg.svdvals(kfree)
            result=contact.result
            normal=float(torch.linalg.vector_norm(result.normal_resultant))
            tangential=float(torch.linalg.vector_norm(result.tangential_resultant))
            ratio=tangential/max(friction*normal,torch.finfo(residual.dtype).eps)
            raise RuntimeError(
                "finite-strain friction Newton failed at load index "
                f"{load_index} (free-load norm={float(torch.linalg.vector_norm(load[free])):.6g}, "
                f"last residual={norm:.6g}, line residuals={line_residuals}, "
                f"active forward/reverse={result.forward.active_points}/{result.reverse.active_points}, "
                f"Coulomb ratio={ratio:.6g}, tangent sigma_min={float(singular[-1]):.6g}, "
                f"condition={float(singular[0]/singular[-1]):.6g}); committed state unchanged")
        residual,_,contact,energy,jacobian=_assemble_frictional(model,trial,committed,load,
            tangential_penalty=tangential_penalty,friction=friction,tangent=False)
        committed=FiniteStrainFrictionState(trial.displacement,contact.result.state)
        steps.append(FiniteStrainFrictionStep(committed,load.clone(),iteration,
            float(torch.linalg.vector_norm(residual[free])),contact,
            energy.detach().clone(),jacobian.detach().clone()))
    return tuple(steps)


def solve_finite_strain_friction_adaptive_path(model: FiniteStrainContactModel,
    loads: Sequence[torch.Tensor], *, tangential_penalty: float, friction: float,
    initial_state: FiniteStrainFrictionState|None=None,tolerance: float=1.e-9,
    max_iterations: int=30,max_subdivisions: int=7):
    """Advance requested loads with transactional bisection at active-set changes.

    Intermediate loads are committed only after full Newton convergence.  A
    target that still fails after ``max_subdivisions`` leaves the last accepted
    state intact and raises; requested target loads are never reduced.
    """
    if max_subdivisions<0: raise ValueError("max_subdivisions must be nonnegative")
    committed=initial_finite_strain_friction_state(model) if initial_state is None else initial_state
    previous=torch.zeros(model.solid.n_dofs,dtype=model.solid.reference_nodes.dtype)
    older_load=None;older_displacement=None
    accepted=[]

    def advance(left_load,target,depth):
        nonlocal committed,older_load,older_displacement
        guess=committed.displacement
        if older_load is not None and older_displacement is not None:
            past=left_load-older_load;future=target-left_load
            denominator=float(torch.dot(past,past))
            if denominator>torch.finfo(past.dtype).eps:
                factor=float(torch.dot(future,past))/denominator
                candidate=committed.displacement+factor*(committed.displacement-older_displacement)
                if bool(torch.isfinite(candidate).all()):
                    try:
                        _,_,_,jacobian,_=assemble_finite_tet4(model.solid,candidate,tangent=False)
                        if bool(torch.min(jacobian)>0): guess=candidate
                    except ValueError:
                        pass
        try:
            step=solve_finite_strain_friction_path(model,[target],
                tangential_penalty=tangential_penalty,friction=friction,
                initial_state=committed,tolerance=tolerance,
                max_iterations=max_iterations,initial_trial_displacement=guess)[0]
            older_load=left_load.clone();older_displacement=committed.displacement.clone()
            committed=step.state;accepted.append(step);return
        except RuntimeError:
            if depth>=max_subdivisions: raise
        middle=.5*(left_load+target)
        advance(left_load,middle,depth+1)
        advance(middle,target,depth+1)

    for target in loads:
        advance(previous,target,0)
        previous=target.clone()
    return tuple(accepted)


def run_curved_finite_strain_friction_qualification():
    """Run 1/2 and 2/3 real-solid curved friction qualification paths."""
    mu=.3;kt=2.e4
    rows=[];finest=None;finest_model=None;finest_loads=None
    for master_cells,slave_cells in ((1,2),(2,3)):
        model=build_curved_nonmatching_two_block_contact(
            master_cells=master_cells,slave_cells=slave_cells)
        loads=[balanced_face_load(model,normal=p) for p in (5.,10.,15.,20.,25.)]
        loads.extend(balanced_face_load(model,normal=25.,shear=s)
                     for s in (2.,8.,16.))
        steps=solve_finite_strain_friction_adaptive_path(model,loads,
            tangential_penalty=kt,friction=mu,tolerance=3.e-9)
        result=steps[-1].contact.result
        normal=float(torch.linalg.vector_norm(result.normal_resultant))
        tangential=float(torch.linalg.vector_norm(result.tangential_resultant))
        rows.append({"master_cells":master_cells,"slave_cells":slave_cells,
            "nodes":len(model.solid.reference_nodes),"elements":len(model.solid.elements),
            "normal_resultant":normal,"tangential_resultant":tangential,
            "coulomb_relative_error":abs(tangential/(mu*normal)-1),
            "interface_force_imbalance":float(torch.linalg.vector_norm(
                result.slave_forces.sum(0)+result.master_forces.sum(0))),
            "minimum_jacobian":float(torch.min(steps[-1].jacobian)),
            "dissipation_increment":float(result.dissipation_increment),
            "residual_norm":steps[-1].residual_norm,
            "iterations":[step.iterations for step in steps]})
        finest,finest_model,finest_loads=steps,model,loads
    assert finest is not None and finest_model is not None and finest_loads is not None
    swapped=FiniteStrainContactModel(finest_model.solid,finest_model.master_nodes,
        torch.flip(finest_model.master_faces,[1]),finest_model.slave_nodes,
        torch.flip(finest_model.slave_faces,[1]),finest_model.normal_penalty)
    swapped_steps=solve_finite_strain_friction_adaptive_path(swapped,finest_loads,
        tangential_penalty=kt,friction=mu,tolerance=3.e-9)
    a=finest[-1].contact.result;b=swapped_steps[-1].contact.result
    interchange=max(abs(float(torch.linalg.vector_norm(b.normal_resultant))
                         /float(torch.linalg.vector_norm(a.normal_resultant))-1),
        abs(float(torch.linalg.vector_norm(b.tangential_resultant))
            /float(torch.linalg.vector_norm(a.tangential_resultant))-1))
    initial=initial_finite_strain_friction_state(finest_model);before=initial.displacement.clone()
    old=initial.contact.forward.points[0].elastic_slip.clone();failed=False
    try: solve_finite_strain_friction_path(finest_model,[finest_loads[-1]],
        tangential_penalty=kt,friction=mu,initial_state=initial,max_iterations=1,
        tolerance=1.e-14)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.displacement,before)
        and torch.equal(initial.contact.forward.points[0].elastic_slip,old))
    passed=(max(row["coulomb_relative_error"] for row in rows)<.03
        and max(row["interface_force_imbalance"] for row in rows)<1.e-9
        and min(row["minimum_jacobian"] for row in rows)>0
        and all(row["dissipation_increment"]>0 for row in rows)
        and max(row["residual_norm"] for row in rows)<1.e-7
        and interchange<.03 and rollback)
    report={"schema":"tensorfem.curved-finite-strain-friction-qualification/1.0",
        "mesh_sequence":rows,"master_slave_interchange_relative_error":interchange,
        "rollback_exact":rollback,
        "oracles":["Coulomb |T|=mu*N","interface action-reaction",
            "positive det(F)","nonnegative weighted dissipation"],
        "objectivity_evidence":"Neo-Hookean and symmetric Mortar finite-rotation covariance gates",
        "general_composite_assessment":("qualified curved nonmatching double-deformable "
            "finite-strain friction subset; self-contact and impact excluded") if passed else "blocked",
        "remaining_boundaries":["self-contact","impact","production segmentation"],
        "passed":passed}
    canonical=json.dumps(report,sort_keys=True,separators=(",",":"))
    report["evidence_sha256"]=hashlib.sha256(canonical.encode()).hexdigest()
    if not passed: raise AssertionError(f"curved finite-strain friction qualification failed: {report}")
    return report
