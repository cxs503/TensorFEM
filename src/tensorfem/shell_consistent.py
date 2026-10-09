"""Energy-consistent residual and tangent for the corotational shell."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .corotational_shell import corotational_cylindrical_shell4, cylindrical_nodes


@dataclass(frozen=True)
class ShellTangentResult:
    energy: torch.Tensor
    internal_force: torch.Tensor
    tangent: torch.Tensor


def rotation_matrix_from_vector(vector: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map with a nonsingular zero-angle limit."""
    if vector.shape != (3,): raise ValueError("rotation vector must have shape (3,)")
    x,y,z=vector; zero=torch.zeros_like(x)
    W=torch.stack((torch.stack((zero,-z,y)),torch.stack((z,zero,-x)),torch.stack((-y,x,zero))))
    theta2=torch.dot(vector,vector)
    # Explicit analytic series avoids the undefined second derivative of
    # ||vector|| at the origin. The branch is fixed from a detached scalar.
    if float(theta2.detach()) < 1e-8:
        a=1-theta2/6+theta2**2/120
        b=.5-theta2/24+theta2**2/720
    else:
        theta=torch.sqrt(theta2)
        a=torch.sin(theta)/theta
        b=(1-torch.cos(theta))/theta2
    I=torch.eye(3,dtype=vector.dtype,device=vector.device)
    return I+a*W+b*(W@W)


def shell_energy(x_theta: torch.Tensor, dofs: torch.Tensor, radius, young, poisson,
                 thickness) -> torch.Tensor:
    """Corotational energy for 4x(translation, global rotation-vector) DOFs."""
    if dofs.shape not in ((24,),(4,6)): raise ValueError("dofs must have shape (24,) or (4,6)")
    q=dofs.reshape(4,6)
    reference=cylindrical_nodes(x_theta,radius)
    rotations=torch.stack([rotation_matrix_from_vector(v) for v in q[:,3:]])
    return corotational_cylindrical_shell4(x_theta,reference+q[:,:3],rotations,
                                            radius,young,poisson,thickness).energy


def consistent_internal_force_tangent(x_theta: torch.Tensor, dofs: torch.Tensor,
                                      radius, young, poisson, thickness,
                                      *, create_graph: bool = False) -> ShellTangentResult:
    """Return energy, exact autograd residual and material/geometric tangent.

    The Hessian differentiates through rigid-frame extraction, rotation
    exponentials and the shell energy; no independently approximated tangent is
    used. The caller supplies a non-degenerate cylindrical patch.
    """
    flat=dofs.reshape(-1)
    if flat.numel()!=24: raise ValueError("shell element requires 24 DOFs")
    if not flat.requires_grad: flat=flat.detach().clone().requires_grad_(True)
    def energy(q): return shell_energy(x_theta,q,radius,young,poisson,thickness)
    value=energy(flat)
    force=torch.autograd.grad(value,flat,create_graph=True)[0]
    rows=[]
    for component in force:
        rows.append(torch.autograd.grad(component,flat,retain_graph=True,
                                        create_graph=create_graph)[0])
    tangent=torch.stack(rows)
    if not create_graph:
        value=value.detach(); force=force.detach(); tangent=tangent.detach()
    return ShellTangentResult(value,force,tangent)
