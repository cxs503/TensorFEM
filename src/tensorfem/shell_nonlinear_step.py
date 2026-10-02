"""Global incremental Newton step for experimental corotational shell meshes."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import torch

from .shell_consistent import consistent_internal_force_tangent, shell_energy


@dataclass(frozen=True)
class CylindricalShellMesh:
    parameters: torch.Tensor       # (nnode,2): axial x and theta
    elements: torch.Tensor         # (nelem,4)
    radius: float
    young: float
    poisson: float
    thickness: float

    def validate(self) -> None:
        if self.parameters.ndim!=2 or self.parameters.shape[1]!=2:
            raise ValueError("parameters must have shape (nnode,2)")
        if self.elements.ndim!=2 or self.elements.shape[1]!=4:
            raise ValueError("elements must have shape (nelem,4)")
        if self.elements.numel()==0 or self.parameters.shape[0]<4:
            raise ValueError("shell mesh must be non-empty")
        if bool(torch.any(self.elements<0)) or int(self.elements.max())>=len(self.parameters):
            raise ValueError("element connectivity is out of range")


@dataclass(frozen=True)
class ShellIncrement:
    load_factor: float
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class ShellStepResult:
    dofs: torch.Tensor
    reaction: torch.Tensor
    load_factor: float
    increments: tuple[ShellIncrement,...]
    converged: bool


def _element_dofs(nodes: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(6*nodes+k for k in range(6)),dim=1).reshape(-1)


def assemble_shell(mesh: CylindricalShellMesh, dofs: torch.Tensor,
                   *, tangent: bool = True):
    """Assemble energy, internal force and optionally the exact tangent."""
    mesh.validate(); ndof=6*len(mesh.parameters)
    if dofs.shape!=(ndof,): raise ValueError(f"dofs must have shape ({ndof},)")
    energy=dofs.new_zeros(()); internal=torch.zeros_like(dofs)
    K=torch.zeros((ndof,ndof),dtype=dofs.dtype,device=dofs.device) if tangent else None
    for conn in mesh.elements:
        ids=_element_dofs(conn); p=mesh.parameters[conn]; q=dofs[ids]
        if tangent:
            result=consistent_internal_force_tangent(p,q,mesh.radius,mesh.young,
                                                      mesh.poisson,mesh.thickness)
            energy=energy+result.energy; internal[ids]+=result.internal_force
            K[ids[:,None],ids]+=result.tangent
        else:
            local=q.detach().clone().requires_grad_(True)
            value=shell_energy(p,local,mesh.radius,mesh.young,mesh.poisson,mesh.thickness)
            force=torch.autograd.grad(value,local)[0]
            energy=energy+value.detach(); internal[ids]+=force.detach()
    return energy,internal,K


def _prescribed_vector(ndof: int, prescribed: Mapping[int,float], like: torch.Tensor):
    fixed=torch.tensor(sorted(prescribed),dtype=torch.long,device=like.device)
    if any(i<0 or i>=ndof for i in prescribed): raise ValueError("prescribed DOF out of range")
    values=like.new_tensor([prescribed[int(i)] for i in fixed.tolist()])
    mask=torch.ones(ndof,dtype=torch.bool,device=like.device); mask[fixed]=False
    return fixed,values,torch.nonzero(mask).flatten()


def write_shell_checkpoint(path: str | Path, dofs: torch.Tensor, load_factor: float) -> None:
    payload={"schema":"tensorfem.shell-step.v1","load_factor":load_factor,
             "dofs":dofs.detach().cpu().tolist()}
    Path(path).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


def read_shell_checkpoint(path: str | Path, *, dtype=torch.float64, device=None):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema")!="tensorfem.shell-step.v1": raise ValueError("invalid shell checkpoint schema")
    factor=float(payload["load_factor"])
    if not 0<=factor<=1: raise ValueError("invalid checkpoint load factor")
    return torch.tensor(payload["dofs"],dtype=dtype,device=device),factor


def solve_shell_step(mesh: CylindricalShellMesh, loads: torch.Tensor,
                     prescribed: Mapping[int,float], *,
                     initial_increment: float = .25, minimum_increment: float = 1e-4,
                     maximum_increment: float = .5, max_iterations: int = 15,
                     relative_tolerance: float = 1e-8, absolute_tolerance: float = 1e-8,
                     line_search_steps: int = 10, checkpoint: str | Path | None = None,
                     restart: str | Path | None = None) -> ShellStepResult:
    """Solve a proportional-load shell step with rollback and adaptive increments."""
    mesh.validate(); ndof=6*len(mesh.parameters)
    if loads.shape!=(ndof,) or not torch.all(torch.isfinite(loads)):
        raise ValueError("loads must be a finite vector with one value per DOF")
    if not 0<minimum_increment<=initial_increment<=maximum_increment<=1:
        raise ValueError("invalid increment bounds")
    fixed,values,free=_prescribed_vector(ndof,prescribed,loads)
    if len(free)==0: raise ValueError("at least one free DOF is required")
    if restart:
        committed,factor=read_shell_checkpoint(restart,dtype=loads.dtype,device=loads.device)
        if committed.shape!=(ndof,): raise ValueError("checkpoint DOF count mismatch")
    else:
        committed=torch.zeros_like(loads); factor=0.
    increment=min(initial_increment,1-factor); history=[]
    while factor < 1-1e-14:
        target=min(1.,factor+increment); trial=committed.clone(); trial[fixed]=target*values
        accepted=False; last_norm=float("inf")
        for iteration in range(1,max_iterations+1):
            # Residual-only evaluation is much cheaper than a 24x24 element
            # Hessian and avoids constructing a tangent after convergence.
            energy,internal,_=assemble_shell(mesh,trial,tangent=False)
            residual=internal-target*loads; rf=residual[free]
            last_norm=float(torch.linalg.vector_norm(rf).detach())
            scale=max(float(torch.linalg.vector_norm((target*loads)[free]).detach()),1.)
            if last_norm <= absolute_tolerance+relative_tolerance*scale:
                accepted=True; break
            energy,internal,K=assemble_shell(mesh,trial,tangent=True)
            residual=internal-target*loads; rf=residual[free]
            try: delta=torch.linalg.solve(K[free[:,None],free],-rf)
            except torch.linalg.LinAlgError: break
            potential=float((energy-target*torch.dot(loads,trial)).detach())
            slope=float(torch.dot(rf,delta).detach()); alpha=1.
            for _ in range(line_search_steps):
                candidate=trial.clone(); candidate[free]+=alpha*delta; candidate[fixed]=target*values
                e2,_,_=assemble_shell(mesh,candidate,tangent=False)
                p2=float((e2-target*torch.dot(loads,candidate)).detach())
                if p2 <= potential+1e-4*alpha*slope: trial=candidate; break
                alpha*=.5
            else: break
        if accepted:
            committed=trial; factor=target
            history.append(ShellIncrement(factor,iteration,last_norm))
            if checkpoint: write_shell_checkpoint(checkpoint,committed,factor)
            increment=min(maximum_increment,increment*(1.5 if iteration<=5 else 1.))
            increment=min(increment,1-factor) if factor<1 else increment
        else:
            increment*=.5
            if increment < minimum_increment:
                return ShellStepResult(committed,torch.zeros_like(loads),factor,tuple(history),False)
    _,internal,_=assemble_shell(mesh,committed,tangent=False)
    return ShellStepResult(committed,internal-loads,factor,tuple(history),True)
