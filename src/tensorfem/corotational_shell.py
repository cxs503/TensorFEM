"""Objective corotational kinematics for a four-node cylindrical shell.

This module is intentionally separate from the experimental linear shell.  It
removes a best-fit proper rigid rotation before the verified small-strain
cylindrical stiffness is evaluated.  It therefore supplies an exact rigid-body
gate without claiming a general large-strain shell formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .cylindrical_shell4 import cylindrical_shell4_stiffness


@dataclass(frozen=True)
class CorotationalState:
    rotation: torch.Tensor
    generalized_deformation: torch.Tensor
    energy: torch.Tensor


def cylindrical_nodes(x_theta: torch.Tensor, radius: torch.Tensor | float) -> torch.Tensor:
    """Map cylindrical parameters to reference Cartesian coordinates."""
    R=torch.as_tensor(radius,dtype=x_theta.dtype,device=x_theta.device)
    return torch.stack((x_theta[:,0],R*torch.sin(x_theta[:,1]),
                        R*torch.cos(x_theta[:,1])),dim=1)


def _proper_fit_rotation(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Return column-vector rotation mapping centred reference to current."""
    X=reference-reference.mean(0); x=current-current.mean(0)
    if float(torch.linalg.matrix_norm(X)) <= torch.finfo(X.dtype).eps:
        raise ValueError("reference patch has zero spatial extent")
    U,_,Vh=torch.linalg.svd(X.T@x)
    correction=torch.eye(3,dtype=X.dtype,device=X.device)
    correction[-1,-1]=torch.sign(torch.linalg.det(U@Vh))
    row_rotation=U@correction@Vh
    return row_rotation.T


def _rotation_vector(Q: torch.Tensor) -> torch.Tensor:
    """Principal logarithm of a proper 3-D rotation, including near-pi cases."""
    trace=torch.trace(Q)
    cosine=torch.clamp((trace-1)/2,-1.,1.)
    angle=torch.acos(cosine)
    skew=torch.stack((Q[2,1]-Q[1,2],Q[0,2]-Q[2,0],Q[1,0]-Q[0,1]))
    angle_value=float(angle.detach())
    if angle_value < 1e-7:
        return .5*skew
    if abs(math.pi-angle_value) < 1e-5:
        # Symmetric part determines the axis when sin(angle) is ill-conditioned.
        diag=torch.clamp((torch.diagonal(Q)+1)/2,min=0)
        axis=torch.sqrt(diag)
        k=int(torch.argmax(axis))
        if float(axis[k].detach()) <= 1e-12: raise ValueError("cannot determine pi-rotation axis")
        if k == 0:
            axis[1]=(Q[0,1]+Q[1,0])/(4*axis[0]); axis[2]=(Q[0,2]+Q[2,0])/(4*axis[0])
        elif k == 1:
            axis[0]=(Q[0,1]+Q[1,0])/(4*axis[1]); axis[2]=(Q[1,2]+Q[2,1])/(4*axis[1])
        else:
            axis[0]=(Q[0,2]+Q[2,0])/(4*axis[2]); axis[1]=(Q[1,2]+Q[2,1])/(4*axis[2])
        return angle*axis/torch.linalg.vector_norm(axis)
    return angle/(2*torch.sin(angle))*skew


def _validate_rotations(rotations: torch.Tensor, tolerance: float) -> None:
    if rotations.shape != (4,3,3): raise ValueError("nodal_rotations must have shape (4,3,3)")
    eye=torch.eye(3,dtype=rotations.dtype,device=rotations.device)
    defect=torch.amax(torch.abs(rotations.transpose(1,2)@rotations-eye))
    determinants=torch.linalg.det(rotations)
    if float(defect.detach()) > tolerance or bool(torch.any(determinants.detach() <= 0)):
        raise ValueError("nodal_rotations must be proper orthogonal matrices")


def corotational_cylindrical_shell4(
    x_theta: torch.Tensor,
    current_nodes: torch.Tensor,
    nodal_rotations: torch.Tensor,
    radius: torch.Tensor | float,
    young: torch.Tensor | float,
    poisson: torch.Tensor | float,
    thickness: torch.Tensor | float,
    *,
    rotation_tolerance: float = 1e-8,
) -> CorotationalState:
    """Evaluate objective element energy for current positions/orientations.

    ``nodal_rotations[a]`` is the total proper rotation of node ``a`` from the
    reference global frame.  The returned generalized deformation is expressed
    in the reference frame after removal of the best-fit rigid motion.
    """
    if x_theta.shape != (4,2) or current_nodes.shape != (4,3):
        raise ValueError("x_theta/current_nodes must have shapes (4,2)/(4,3)")
    if current_nodes.dtype != x_theta.dtype or current_nodes.device != x_theta.device:
        raise ValueError("geometry tensors must share dtype and device")
    _validate_rotations(nodal_rotations,rotation_tolerance)
    X=cylindrical_nodes(x_theta,radius)
    Rc=_proper_fit_rotation(X,current_nodes)
    # Remove translations and the corotational frame. Centroid subtraction is
    # essential: it makes the measure exactly insensitive to global shifts.
    local_current=(current_nodes-current_nodes.mean(0))@Rc
    local_reference=X-X.mean(0)
    translations=local_current-local_reference
    relative=Rc.T.unsqueeze(0)@nodal_rotations
    rotation_vectors=torch.stack([_rotation_vector(Q) for Q in relative])
    q=torch.cat((translations,rotation_vectors),dim=1).reshape(-1)
    K=cylindrical_shell4_stiffness(x_theta,radius,young,poisson,thickness)
    energy=.5*torch.dot(q,K@q)
    return CorotationalState(Rc,q,energy)


def axis_angle(axis: torch.Tensor, angle: float | torch.Tensor) -> torch.Tensor:
    """Construct a proper rotation matrix for examples and verification."""
    axis=axis/torch.linalg.vector_norm(axis)
    a=torch.as_tensor(angle,dtype=axis.dtype,device=axis.device)
    x,y,z=axis; zero=torch.zeros_like(x)
    W=torch.stack((torch.stack((zero,-z,y)),torch.stack((z,zero,-x)),torch.stack((-y,x,zero))))
    I=torch.eye(3,dtype=axis.dtype,device=axis.device)
    return I+torch.sin(a)*W+(1-torch.cos(a))*(W@W)
