"""Crisfield-style spherical arc-length continuation with fail-closed steps."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable
import torch


@dataclass(frozen=True)
class ArcLengthPoint:
    step: int
    load_factor: float
    displacement: torch.Tensor
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class ArcLengthResult:
    points: tuple[ArcLengthPoint,...]
    converged: bool
    step_size: float


@dataclass(frozen=True)
class ArcLengthProblem:
    """Nonlinear equilibrium ``internal(u) - lambda*reference_load = 0``."""
    internal_tangent: Callable[[torch.Tensor],tuple[torch.Tensor,torch.Tensor]]
    reference_load: torch.Tensor
    residual_tangent: Callable[[torch.Tensor,float],tuple[torch.Tensor,torch.Tensor,torch.Tensor]]|None=None


def general_shell_arc_problem(mesh, reference_load: torch.Tensor, fixed_dofs):
    """Reduce a zero-prescribed general shell mesh to an arc-length problem."""
    from .general_shell_nonlinear import assemble_general_shell
    ndof=6*len(mesh.nodes)
    if reference_load.shape!=(ndof,): raise ValueError("wrong shell load length")
    fixed=torch.tensor(sorted(set(int(i) for i in fixed_dofs)),dtype=torch.long,
                       device=reference_load.device)
    if bool(torch.any(fixed<0)) or (len(fixed) and int(fixed.max())>=ndof):
        raise ValueError("fixed shell DOF out of range")
    mask=torch.ones(ndof,dtype=torch.bool,device=reference_load.device);mask[fixed]=False
    free=torch.nonzero(mask).flatten()
    def response(reduced):
        q=torch.zeros(ndof,dtype=reduced.dtype,device=reduced.device);q[free]=reduced
        _,internal,K=assemble_general_shell(mesh,q,tangent=True)
        return internal[free],K[free[:,None],free]
    return ArcLengthProblem(response,reference_load[free]),free


def _write_checkpoint(path,u,load,direction,step_size,completed):
    Path(path).write_text(json.dumps({"schema":"tensorfem.arc-length.v1",
        "displacement":u.detach().cpu().tolist(),"load_factor":load,
        "direction":direction.detach().cpu().tolist(),"step_size":step_size,
        "completed_steps":completed},indent=2)+"\n")


def solve_arc_length(problem: ArcLengthProblem, initial: torch.Tensor, *, steps: int,
                     step_size: float, load_scale: float,
                     tolerance: float=1e-10, max_iterations: int=15,
                     minimum_step: float=1e-5, maximum_step: float|None=None,
                     checkpoint: str|Path|None=None, restart: str|Path|None=None,
                     branch_switch: str|None=None, branch_mode_fraction: float=.25,
                     branch_sign: int=1, critical_eigenvalue_ratio: float=1e-6,
                     diagnostics: list[dict[str, object]]|None=None):
    """Trace equilibrium through limit points using an augmented Newton solve."""
    if steps<1 or step_size<=0 or load_scale<=0: raise ValueError("invalid arc-length controls")
    if branch_switch not in (None,"critical_mode"):
        raise ValueError("branch_switch must be None or 'critical_mode'")
    if not 0<branch_mode_fraction<=1 or branch_sign not in (-1,1) or critical_eigenvalue_ratio<=0:
        raise ValueError("invalid branch-switch controls")
    f=problem.reference_load
    if initial.shape!=f.shape or not torch.all(torch.isfinite(f)): raise ValueError("invalid initial/load vector")
    maximum_step=maximum_step or step_size
    if restart:
        p=json.loads(Path(restart).read_text())
        if p.get("schema")!="tensorfem.arc-length.v1": raise ValueError("invalid arc-length checkpoint")
        u=torch.tensor(p["displacement"],dtype=initial.dtype,device=initial.device)
        load=float(p["load_factor"]);previous=torch.tensor(p["direction"],dtype=initial.dtype,device=initial.device)
        ds=float(p["step_size"]);offset=int(p["completed_steps"])
    else:
        u=initial.clone();load=0.;previous=None;ds=step_size;offset=0
    points=[];accepted=0
    while accepted<steps:
        committed_u=u.clone();committed_load=load
        if problem.residual_tangent is None:
            internal,K=problem.internal_tangent(u);residual=internal-load*f;load_derivative=-f
        else:
            residual,K,load_derivative=problem.residual_tangent(u,load)
        predictor_rhs=f if problem.residual_tangent is None else -load_derivative
        try:duhat=torch.linalg.solve(K,predictor_rhs)
        except torch.linalg.LinAlgError:
            if branch_switch!="critical_mode":
                return ArcLengthResult(tuple(points),False,ds)
            # At an exact bifurcation both the load tangent and structural
            # tangent may vanish along the emerging branch.  The minimum-norm
            # particular solution plus the critical homogeneous mode is the
            # appropriate branch-switch predictor.
            duhat=torch.linalg.pinv(K)@predictor_rhs
        tangent=torch.cat((duhat,duhat.new_tensor([1.])))
        sign=1.
        if previous is not None:
            weighted_dot=torch.dot(tangent[:-1],previous[:-1])+load_scale**2*tangent[-1]*previous[-1]
            sign=1. if float(weighted_dot.detach())>=0 else -1.
        direction_arc=torch.cat((duhat,duhat.new_tensor([load_scale])))
        predictor=sign*direction_arc/torch.linalg.vector_norm(direction_arc)
        mode_diagnostic=None
        if branch_switch=="critical_mode":
            from .branch_switch import perturbed_arc_predictor,smallest_eligible_symmetric_mode
            critical=smallest_eligible_symmetric_mode(K)
            activated=critical.relative_eigenvalue<=critical_eigenvalue_ratio
            mode_diagnostic={"eigenvalue":critical.eigenvalue,
                "relative_eigenvalue":critical.relative_eigenvalue,
                "activated":activated,"sign":branch_sign}
            if activated:
                mode=torch.cat((critical.vector,critical.vector.new_zeros(1)))
                predictor=perturbed_arc_predictor(predictor,mode,
                    mode_fraction=branch_mode_fraction,sign=branch_sign)
        du=ds*predictor[:-1];dl=float(ds*predictor[-1]/load_scale)
        trial_u=u+du;trial_load=load+dl;ok=False;last=float("inf")
        for iteration in range(1,max_iterations+1):
            if problem.residual_tangent is None:
                internal,K=problem.internal_tangent(trial_u);res=internal-trial_load*f;load_derivative=-f
            else:res,K,load_derivative=problem.residual_tangent(trial_u,trial_load)
            Du=trial_u-committed_u;Dl=trial_load-committed_load
            constraint=torch.dot(Du,Du)+(load_scale*Dl)**2-ds**2
            last=float(torch.linalg.vector_norm(torch.cat((res,constraint.reshape(1)))).detach())
            if float(torch.linalg.vector_norm(res).detach())<=tolerance and abs(float(constraint.detach()))<=tolerance*max(ds,1.):
                ok=True;break
            A=torch.zeros((len(u)+1,len(u)+1),dtype=u.dtype,device=u.device)
            A[:-1,:-1]=K;A[:-1,-1]=load_derivative;A[-1,:-1]=2*Du;A[-1,-1]=2*load_scale**2*Dl
            rhs=-torch.cat((res,constraint.reshape(1)))
            try:correction=torch.linalg.solve(A,rhs)
            except torch.linalg.LinAlgError:break
            trial_u=trial_u+correction[:-1];trial_load+=float(correction[-1].detach())
        if ok:
            u=trial_u;load=trial_load;accepted+=1
            direction=torch.cat((u-committed_u,u.new_tensor([load-committed_load])))
            previous=direction
            points.append(ArcLengthPoint(offset+accepted,load,u.clone(),iteration,last))
            if iteration<=4:ds=min(maximum_step,ds*1.25)
            elif iteration>8:ds=max(minimum_step,ds*.75)
            if checkpoint:_write_checkpoint(checkpoint,u,load,previous,ds,offset+accepted)
            if diagnostics is not None:
                diagnostics.append({"step":offset+accepted,"reason":"accepted",
                                    "critical_mode":mode_diagnostic})
        else:
            if diagnostics is not None:
                diagnostics.append({"step":offset+accepted+1,"reason":"rejected",
                                    "critical_mode":mode_diagnostic})
            u=committed_u;load=committed_load;ds*=.5
            if ds<minimum_step:return ArcLengthResult(tuple(points),False,ds)
    return ArcLengthResult(tuple(points),True,ds)
