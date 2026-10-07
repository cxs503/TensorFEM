"""Rotation-controlled large-rotation pure-bending shell benchmark."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .general_shell_nonlinear import GeneralShellMesh, solve_general_shell


@dataclass(frozen=True)
class LargeRotationResult:
    elements: int
    result: object
    tip: torch.Tensor
    exact_tip: torch.Tensor
    applied_moment: float
    exact_moment: float
    @property
    def tip_error(self):
        return float(torch.linalg.vector_norm(self.tip-self.exact_tip)/
                     torch.linalg.vector_norm(self.exact_tip))
    @property
    def moment_error(self): return abs(self.applied_moment/self.exact_moment-1.)


def pure_bending_shell(n: int, *, angle: float=math.pi/2) -> LargeRotationResult:
    """Bend a 10x1x0.1 shell strip into a circular arc by end rotation.

    Cross-section rotations are prescribed linearly in arc coordinate. This is
    a displacement-controlled form of the standard large-rotation cantilever,
    avoiding load-control difficulties at rotations approaching a full turn.
    """
    if n<1: raise ValueError("n must be positive")
    D=torch.float64;L=10.;width=1.;thickness=.1;E=1.2e6
    xs=torch.linspace(0,L,n+1,dtype=D);ys=torch.tensor([-width/2,width/2],dtype=D)
    gx,gy=torch.meshgrid(xs,ys,indexing="ij")
    nodes=torch.stack((gx,gy,torch.zeros_like(gx)),-1).reshape(-1,3)
    elements=torch.tensor([[2*i,2*i+2,2*i+3,2*i+1] for i in range(n)])
    mesh=GeneralShellMesh(nodes,elements,E,0.,thickness)
    loads=torch.zeros(6*len(nodes),dtype=D);prescribed={6*a+k:0. for a in (0,1) for k in range(6)}
    for i in range(1,n+1):
        for a in (2*i,2*i+1): prescribed[6*a+4]=-angle*i/n
    probe=6*(2*n)+2
    result=solve_general_shell(mesh,loads,prescribed,probe_dof=probe,increment=.5,
                               max_iterations=15,tolerance=1e-5)
    if not result.converged: raise RuntimeError("large-rotation shell did not converge")
    current=nodes+result.dofs.reshape(-1,6)[:,:3];tip=current[-2:].mean(0)
    radius=L/angle;exact=nodes.new_tensor([radius*math.sin(angle),0.,radius*(1-math.cos(angle))])
    I=width*thickness**3/12;exact_moment=E*I*angle/L
    end_rot=torch.tensor([6*(2*n)+4,6*(2*n+1)+4])
    moment=-float(result.reaction[end_rot].sum())
    return LargeRotationResult(n,result,tip,exact,moment,exact_moment)
