"""Incremental frictional mortar and two-sided self-contact foundations."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .contact3d import project_point_to_facets
from .mortar_contact3d import _quadrature,self_contact_candidates


@dataclass(frozen=True)
class MortarPointState:
    elastic_slip: torch.Tensor
    dissipated_energy_density: torch.Tensor
    projection: torch.Tensor
    master_face: int
    slave_point: torch.Tensor | None = None
    master_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class FrictionalMortarState:
    topology: tuple[int,...]
    points: tuple[MortarPointState,...]


@dataclass(frozen=True)
class FrictionalMortarResult:
    slave_forces: torch.Tensor
    master_forces: torch.Tensor
    state: FrictionalMortarState
    normal_resultant: torch.Tensor
    tangential_resultant: torch.Tensor
    stored_energy: torch.Tensor
    dissipation_increment: torch.Tensor
    active_points: int


@dataclass(frozen=True)
class FrictionalMortarAssembly:
    """Two-sided residual and frozen-branch algorithmic tangent."""
    residual: torch.Tensor
    tangent: torch.Tensor
    result: FrictionalMortarResult


def _detached_result(result: FrictionalMortarResult) -> FrictionalMortarResult:
    """Deeply detach a trial result before it crosses the assembly boundary."""
    def tensor(value: torch.Tensor | None):
        return None if value is None else value.detach().clone()

    points=tuple(MortarPointState(
        tensor(point.elastic_slip), tensor(point.dissipated_energy_density),
        tensor(point.projection), point.master_face, tensor(point.slave_point),
        tensor(point.master_weights),
    ) for point in result.state.points)
    state=FrictionalMortarState(result.state.topology,points)
    return FrictionalMortarResult(
        tensor(result.slave_forces),tensor(result.master_forces),state,
        tensor(result.normal_resultant),tensor(result.tangential_resultant),
        tensor(result.stored_energy),tensor(result.dissipation_increment),
        result.active_points)


def _topology(slave_faces,master_faces):
    return (slave_faces.shape[1],*slave_faces.detach().cpu().flatten().tolist(),
            -1,master_faces.shape[1],*master_faces.detach().cpu().flatten().tolist())


def initial_frictional_mortar_state(slave_vertices,slave_faces,master_vertices,master_faces):
    points=[]
    for face in slave_faces:
        for point,_,_ in _quadrature(slave_vertices[face]):
            p=project_point_to_facets(point,master_vertices,master_faces)
            points.append(MortarPointState(torch.zeros_like(point),point.new_zeros(()),p.point,p.face,point,p.weights))
    return FrictionalMortarState(_topology(slave_faces,master_faces),tuple(points))


def update_frictional_mortar(slave_vertices,slave_faces,master_vertices,master_faces,state,
                             *,normal_penalty: float,tangential_penalty: float,
                             friction: float,relative_increments: torch.Tensor | None=None):
    """One-pass penalty mortar update with Coulomb integration-point history."""
    if min(normal_penalty,tangential_penalty)<=0 or friction<0: raise ValueError("invalid contact parameters")
    if state.topology!=_topology(slave_faces,master_faces): raise ValueError("mortar state topology mismatch")
    if relative_increments is not None and relative_increments.shape!=slave_vertices.shape:
        raise ValueError("relative increments must match slave vertices")
    sf=torch.zeros_like(slave_vertices); mf=torch.zeros_like(master_vertices)
    normal=slave_vertices.new_zeros(3); tangential=normal.clone(); stored=normal.new_zeros(()); diss=stored.clone()
    new=[]; active=0; qid=0
    for face in slave_faces:
        x=slave_vertices[face]
        for point,N,da in _quadrature(x):
            old=state.points[qid]; qid+=1
            p=project_point_to_facets(point,master_vertices,master_faces)
            pressure=normal_penalty*torch.clamp(-p.gap,min=0.)
            if bool(pressure<=torch.finfo(point.dtype).eps):
                new.append(MortarPointState(torch.zeros_like(point),old.dissipated_energy_density,p.point,p.face,point,p.weights)); continue
            active+=1
            if relative_increments is None:
                old_slave=point if old.slave_point is None else old.slave_point
                if old.master_weights is None:
                    master_motion=p.point-old.projection
                else:
                    material_point=old.master_weights@master_vertices[master_faces[old.master_face]]
                    master_motion=material_point-old.projection
                inc=(point-old_slave)-master_motion
            else:
                inc=N@relative_increments[face]
            slip_inc=inc-torch.dot(inc,p.normal)*p.normal
            old_slip=old.elastic_slip-torch.dot(old.elastic_slip,p.normal)*p.normal
            trial=old_slip+slip_inc; tau_trial=-tangential_penalty*trial
            mag=torch.linalg.vector_norm(tau_trial); limit=friction*pressure
            if bool(mag<=limit) or bool(mag<=torch.finfo(point.dtype).eps):
                tau=tau_trial; elastic=trial; plastic=torch.zeros_like(point)
            else:
                tau=-limit*trial/torch.linalg.vector_norm(trial); elastic=-tau/tangential_penalty; plastic=trial-elastic
            di=torch.abs(torch.dot(tau,plastic)); traction=pressure*p.normal+tau
            sf[face]+=N[:,None]*traction*da
            mf[master_faces[p.face]]-=p.weights[:,None]*traction*da
            normal+=pressure*p.normal*da; tangential+=tau*da
            stored+=(.5*pressure**2/normal_penalty+.5*tangential_penalty*torch.dot(elastic,elastic))*da
            diss+=di*da
            new.append(MortarPointState(elastic,old.dissipated_energy_density+di,p.point,p.face,point,p.weights))
    return FrictionalMortarResult(sf,mf,FrictionalMortarState(state.topology,tuple(new)),
                                  normal,tangential,stored,diss,active)


def assemble_frictional_mortar(
    slave_reference: torch.Tensor, slave_faces: torch.Tensor,
    master_reference: torch.Tensor, master_faces: torch.Tensor,
    slave_displacement: torch.Tensor, master_displacement: torch.Tensor,
    committed: FrictionalMortarState, *, normal_penalty: float,
    tangential_penalty: float, friction: float, tangent: bool = True,
) -> FrictionalMortarAssembly:
    """Assemble frictional contact with an exact frozen-return-branch tangent.

    Search, active-set and stick/slip decisions are deliberately held fixed
    during one linearisation. Within that branch, current projection, normal,
    pressure and Coulomb return mapping stay in the autograd graph. The trial
    history is returned but the caller-owned committed history is immutable.
    """
    if slave_reference.shape != slave_displacement.shape or master_reference.shape != master_displacement.shape:
        raise ValueError("reference coordinates and displacements must match")
    ns=slave_reference.numel()
    q=torch.cat((slave_displacement.reshape(-1),master_displacement.reshape(-1))).detach().requires_grad_(True)

    def evaluate(dofs: torch.Tensor):
        slave=slave_reference+dofs[:ns].reshape_as(slave_reference)
        master=master_reference+dofs[ns:].reshape_as(master_reference)
        return update_frictional_mortar(slave,slave_faces,master,master_faces,committed,
            normal_penalty=normal_penalty,tangential_penalty=tangential_penalty,
            friction=friction)

    def residual_only(dofs: torch.Tensor):
        result=evaluate(dofs)
        return -torch.cat((result.slave_forces.reshape(-1),result.master_forces.reshape(-1)))

    result=evaluate(q)
    residual=residual_only(q)
    matrix=(torch.autograd.functional.jacobian(residual_only,q,create_graph=False)
            if tangent else q.new_zeros((q.numel(),q.numel())))
    return FrictionalMortarAssembly(
        residual.detach().clone(),matrix.detach().clone(),_detached_result(result))


@dataclass(frozen=True)
class SelfContactResult:
    forces: torch.Tensor
    candidate_pairs: tuple[tuple[int,int],...]
    active_pairs: tuple[tuple[int,int],...]
    maximum_penetration: torch.Tensor
    penalty_energy: torch.Tensor


def update_self_contact(vertices,faces,*,clearance: float,normal_penalty: float):
    """Assemble frictionless two-sided forces once per nonadjacent facet pair."""
    if clearance<=0 or normal_penalty<=0: raise ValueError("clearance and penalty must be positive")
    pairs=self_contact_candidates(vertices,faces,search_distance=clearance)
    force=torch.zeros_like(vertices); active=[]; maxpen=vertices.new_zeros(()); energy=maxpen.clone()
    for si,mi in pairs:
        master_face=faces[mi:mi+1]
        pair_active=False
        for point,N,da in _quadrature(vertices[faces[si]]):
            p=project_point_to_facets(point,vertices,master_face)
            distance=torch.linalg.vector_norm(point-p.point)
            penetration=torch.clamp(point.new_tensor(clearance)-distance,min=0.)
            maxpen=torch.maximum(maxpen,penetration)
            if bool(penetration<=torch.finfo(point.dtype).eps): continue
            pair_active=True
            if bool(distance<=torch.finfo(point.dtype).eps): normal=p.normal
            else: normal=(point-p.point)/distance
            traction=normal_penalty*penetration*normal
            force[faces[si]]+=N[:,None]*traction*da
            force[faces[mi]]-=p.weights[:,None]*traction*da
            energy+=.5*normal_penalty*penetration**2*da
        if pair_active: active.append((si,mi))
    return SelfContactResult(force,pairs,tuple(active),maxpen,energy)
