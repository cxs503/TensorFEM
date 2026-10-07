"""Experimental corotational nonlinear analysis of general curved Shell4 meshes."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import torch

from .corotational_shell import _proper_fit_rotation, _rotation_vector
from .shell_consistent import rotation_matrix_from_vector
from .spherical_shell import projected_shell4_stiffness


def general_shell_energy(reference: torch.Tensor, dofs: torch.Tensor, young, poisson,
                         thickness, *, drilling_factor=1e-6):
    """Objective quadratic-material energy for one general curved facet."""
    q=dofs.reshape(4,6); current=reference+q[:,:3]
    rotations=torch.stack([rotation_matrix_from_vector(v) for v in q[:,3:]])
    Rc=_proper_fit_rotation(reference,current)
    translations=(current-current.mean(0))@Rc-(reference-reference.mean(0))
    relative=Rc.T.unsqueeze(0)@rotations
    rv=torch.stack([_rotation_vector(Q) for Q in relative])
    deform=torch.cat((translations,rv),1).reshape(-1)
    K=projected_shell4_stiffness(reference,young,poisson,thickness,
                                 drilling_factor=drilling_factor)
    return .5*torch.dot(deform,K@deform)


def general_shell_force_tangent(reference,dofs,young,poisson,thickness,*,drilling_factor=1e-6):
    flat=dofs.reshape(-1)
    if flat.numel()!=24: raise ValueError("general shell element requires 24 DOFs")
    flat=flat.detach().clone().requires_grad_(True)
    def fn(v): return general_shell_energy(reference,v,young,poisson,thickness,
                                            drilling_factor=drilling_factor)
    energy=fn(flat); force=torch.autograd.grad(energy,flat,create_graph=True)[0]
    tangent=torch.stack([torch.autograd.grad(v,flat,retain_graph=True)[0] for v in force])
    return energy.detach(),force.detach(),tangent.detach()


@dataclass(frozen=True)
class GeneralShellMesh:
    nodes: torch.Tensor
    elements: torch.Tensor
    young: float
    poisson: float
    thickness: float
    drilling_factor: float=1e-6
    def validate(self):
        if self.nodes.ndim!=2 or self.nodes.shape[1]!=3 or self.elements.ndim!=2 or self.elements.shape[1]!=4:
            raise ValueError("nodes/elements must have shapes (n,3)/(m,4)")
        if self.elements.numel()==0 or bool(torch.any(self.elements<0)) or int(self.elements.max())>=len(self.nodes):
            raise ValueError("invalid shell connectivity")


@dataclass(frozen=True)
class GeneralShellResult:
    dofs: torch.Tensor
    reaction: torch.Tensor
    load_factor: float
    path: tuple[tuple[float,float],...]
    converged: bool


def _ids(conn): return torch.stack(tuple(6*conn+k for k in range(6)),1).reshape(-1)


def assemble_general_shell(mesh,dofs,*,tangent=True):
    mesh.validate(); ndof=6*len(mesh.nodes)
    if dofs.shape!=(ndof,): raise ValueError("wrong global DOF vector length")
    energy=dofs.new_zeros(()); force=torch.zeros_like(dofs)
    K=torch.zeros((ndof,ndof),dtype=dofs.dtype,device=dofs.device) if tangent else None
    for conn in mesh.elements:
        ids=_ids(conn); ref=mesh.nodes[conn]; local=dofs[ids]
        if tangent:
            e,f,k=general_shell_force_tangent(ref,local,mesh.young,mesh.poisson,
                                               mesh.thickness,drilling_factor=mesh.drilling_factor)
            K[ids[:,None],ids]+=k
        else:
            local=local.detach().clone().requires_grad_(True)
            e=general_shell_energy(ref,local,mesh.young,mesh.poisson,mesh.thickness,
                                   drilling_factor=mesh.drilling_factor)
            f=torch.autograd.grad(e,local)[0].detach();e=e.detach()
        energy+=e;force[ids]+=f
    return energy,force,K


def _write(path,dofs,factor):
    Path(path).write_text(json.dumps({"schema":"tensorfem.general-shell.v1","load_factor":factor,
                                      "dofs":dofs.detach().cpu().tolist()},indent=2)+"\n")


def solve_general_shell(mesh: GeneralShellMesh, loads: torch.Tensor,
                        prescribed: Mapping[int,float], *, probe_dof: int,
                        increment=.25, minimum_increment=1e-3, max_iterations=12,
                        tolerance=1e-7, checkpoint: str|Path|None=None,
                        restart: str|Path|None=None):
    """Adaptive full-Newton step with backtracking, rollback and restart."""
    mesh.validate();ndof=6*len(mesh.nodes)
    if loads.shape!=(ndof,) or not torch.all(torch.isfinite(loads)): raise ValueError("invalid loads")
    fixed=torch.tensor(sorted(prescribed),dtype=torch.long,device=loads.device)
    mask=torch.ones(ndof,dtype=torch.bool,device=loads.device);mask[fixed]=False;free=torch.nonzero(mask).flatten()
    values=loads.new_tensor([prescribed[int(i)] for i in fixed.tolist()])
    if restart:
        payload=json.loads(Path(restart).read_text())
        if payload.get("schema")!="tensorfem.general-shell.v1": raise ValueError("invalid checkpoint")
        q=torch.tensor(payload["dofs"],dtype=loads.dtype,device=loads.device);factor=float(payload["load_factor"])
    else:q=torch.zeros_like(loads);factor=0.
    step=min(increment,1-factor);path=[]
    while factor<1-1e-14:
        target=min(1.,factor+step);trial=q.clone();trial[fixed]=target*values;ok=False
        for _iteration in range(max_iterations):
            energy,internal,_=assemble_general_shell(mesh,trial,tangent=False);res=internal-target*loads
            norm=float(torch.linalg.vector_norm(res[free]).detach())
            if norm<=tolerance+1e-8*max(float(torch.linalg.vector_norm((target*loads)[free])),1.):ok=True;break
            energy,internal,K=assemble_general_shell(mesh,trial,tangent=True);res=internal-target*loads
            try:delta=torch.linalg.solve(K[free[:,None],free],-res[free])
            except torch.linalg.LinAlgError:break
            potential=float((energy-target*torch.dot(loads,trial)).detach());slope=float(torch.dot(res[free],delta).detach());alpha=1.
            for _ in range(10):
                candidate=trial.clone();candidate[free]+=alpha*delta;candidate[fixed]=target*values
                e,_,_=assemble_general_shell(mesh,candidate,tangent=False)
                if float((e-target*torch.dot(loads,candidate)).detach())<=potential+1e-4*alpha*slope:
                    trial=candidate;break
                alpha*=.5
            else:break
        if ok:
            q=trial;factor=target;path.append((factor,float(q[probe_dof])))
            if checkpoint:_write(checkpoint,q,factor)
            step=min(step*1.5,.5,1-factor) if factor<1 else step
        else:
            step*=.5
            if step<minimum_increment:return GeneralShellResult(q,torch.zeros_like(q),factor,tuple(path),False)
    _,internal,_=assemble_general_shell(mesh,q,tangent=False)
    return GeneralShellResult(q,internal-loads,factor,tuple(path),True)
