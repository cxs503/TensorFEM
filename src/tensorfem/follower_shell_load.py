"""Configuration-dependent Shell4 pressure loads and consistent load tangent."""
from __future__ import annotations
import math
import torch


def follower_pressure_force(reference: torch.Tensor, dofs: torch.Tensor,
                            pressure: torch.Tensor|float=1.) -> torch.Tensor:
    """Consistent 24-DOF pressure force on the current bilinear surface."""
    if reference.shape!=(4,3) or dofs.reshape(-1).numel()!=24:
        raise ValueError("reference/dofs must describe one four-node shell")
    q=dofs.reshape(4,6);x=reference+q[:,:3]
    p=torch.as_tensor(pressure,dtype=x.dtype,device=x.device)
    force=torch.zeros((4,6),dtype=x.dtype,device=x.device);g=1/math.sqrt(3)
    for xi,eta in ((-g,-g),(g,-g),(g,g),(-g,g)):
        N=x.new_tensor([(1-xi)*(1-eta),(1+xi)*(1-eta),
                        (1+xi)*(1+eta),(1-xi)*(1+eta)])/4
        nat=x.new_tensor([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],
                          [(1+eta),(1+xi)],[-(1+eta),(1-xi)]])/4
        tangent=x.T@nat;area_vector=torch.linalg.cross(tangent[:,0],tangent[:,1],dim=0)
        force[:,:3]+=N[:,None]*p*area_vector
    return force.reshape(-1)


def follower_pressure_force_tangent(reference,dofs,pressure=1.):
    """Return force and exact ``d(force)/d(dofs)`` from one expression."""
    q=dofs.reshape(-1).detach().clone().requires_grad_(True)
    force=follower_pressure_force(reference,q,pressure)
    rows=[torch.autograd.grad(v,q,retain_graph=True)[0] for v in force]
    return force.detach(),torch.stack(rows).detach()


def assemble_follower_pressure(mesh,dofs,pressures=1.,*,tangent=True):
    ndof=6*len(mesh.nodes);force=torch.zeros(ndof,dtype=dofs.dtype,device=dofs.device)
    K=torch.zeros((ndof,ndof),dtype=dofs.dtype,device=dofs.device) if tangent else None
    values=torch.as_tensor(pressures,dtype=dofs.dtype,device=dofs.device)
    if values.ndim==0:values=values.expand(len(mesh.elements))
    if values.shape!=(len(mesh.elements),):raise ValueError("one pressure per element is required")
    for e,(conn,p) in enumerate(zip(mesh.elements,values)):
        ids=torch.stack(tuple(6*conn+k for k in range(6)),1).reshape(-1)
        if tangent:f,k=follower_pressure_force_tangent(mesh.nodes[conn],dofs[ids],p);K[ids[:,None],ids]+=k
        else:f=follower_pressure_force(mesh.nodes[conn],dofs[ids],p).detach()
        force[ids]+=f
    return force,K


def general_shell_pressure_arc_problem(mesh,pressures,fixed_dofs):
    """Create a follower-pressure residual/tangent for Crisfield continuation."""
    from .arc_length import ArcLengthProblem
    from .general_shell_nonlinear import assemble_general_shell
    mesh.validate();ndof=6*len(mesh.nodes);fixed=torch.tensor(sorted(set(int(i) for i in fixed_dofs)),dtype=torch.long,device=mesh.nodes.device)
    if bool(torch.any(fixed<0)) or (len(fixed) and int(fixed.max())>=ndof):
        raise ValueError("fixed shell DOF out of range")
    mask=torch.ones(ndof,dtype=torch.bool,device=mesh.nodes.device);mask[fixed]=False;free=torch.nonzero(mask).flatten()
    def residual(reduced,load):
        q=torch.zeros(ndof,dtype=reduced.dtype,device=reduced.device);q[free]=reduced
        _,internal,Kint=assemble_general_shell(mesh,q,tangent=True)
        external,Kload=assemble_follower_pressure(mesh,q,pressures,tangent=True)
        return internal[free]-load*external[free],(Kint-load*Kload)[free[:,None],free],-external[free]
    return ArcLengthProblem(lambda u: (_ for _ in ()).throw(RuntimeError("load-dependent residual required")),
                            torch.zeros(len(free),dtype=mesh.nodes.dtype,device=mesh.nodes.device),residual),free
