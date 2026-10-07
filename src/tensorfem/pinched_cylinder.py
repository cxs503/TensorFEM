"""MacNeal--Harder pinched-cylinder benchmark for the cylindrical shell."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .cylindrical_shell4 import cylindrical_shell4_stiffness
from .shell_nonlinear_step import CylindricalShellMesh, solve_shell_step


REFERENCE_DISPLACEMENT=-1.8248e-5


@dataclass(frozen=True)
class PinchedCylinderResult:
    mesh_size: int
    displacement: float
    reference: float=REFERENCE_DISPLACEMENT
    @property
    def relative_error(self): return abs(self.displacement/self.reference-1.)


def pinched_cylinder_model(n: int, *, dtype=torch.float64, device=None):
    """Return the symmetry octant, load vector, constraints and probe DOF.

    The full cylinder has length 600 and a pair of diametrically opposed unit
    pinching loads at midspan. Reflection symmetry leaves x=[0,300] and
    theta=[0,pi/2]; the corner load is therefore one quarter of a unit.
    """
    if n<2: raise ValueError("n must be at least two")
    dev=torch.device(device or "cpu"); R=300.
    xs=torch.linspace(0.,300.,n+1,dtype=dtype,device=dev)
    ts=torch.linspace(0.,math.pi/2,n+1,dtype=dtype,device=dev)
    gx,gt=torch.meshgrid(xs,ts,indexing="ij")
    parameters=torch.stack((gx,gt),-1).reshape(-1,2)
    conn=[]
    for i in range(n):
        for j in range(n):
            a=i*(n+1)+j; conn.append((a,a+n+1,a+n+2,a+1))
    elements=torch.tensor(conn,dtype=torch.long,device=dev)
    mesh=CylindricalShellMesh(parameters,elements,R,3e6,.3,3.)
    loads=torch.zeros(6*len(parameters),dtype=dtype,device=dev); loads[2]=-.25
    fixed: dict[int,float]={}
    # Midspan reflection plane x=0: normal displacement and in-plane rotations.
    for j in range(n+1):
        a=j
        for k in (0,4,5): fixed[6*a+k]=0.
    # Circumferential reflection planes y=0 and z=0.
    for i in range(n+1):
        a=i*(n+1)
        for k in (1,3,5): fixed[6*a+k]=0.
        a=i*(n+1)+n
        for k in (2,3,4): fixed[6*a+k]=0.
    # Rigid end diaphragm: cross-section translations are restrained.
    for j in range(n+1):
        a=n*(n+1)+j; fixed[6*a+1]=0.; fixed[6*a+2]=0.
    return mesh,loads,fixed,2


def solve_pinched_cylinder_linear(n: int) -> PinchedCylinderResult:
    mesh,loads,fixed,probe=pinched_cylinder_model(n)
    ndof=len(loads); K=torch.zeros((ndof,ndof),dtype=loads.dtype,device=loads.device)
    for conn in mesh.elements:
        ids=torch.stack(tuple(6*conn+k for k in range(6)),1).reshape(-1)
        ke=cylindrical_shell4_stiffness(mesh.parameters[conn],mesh.radius,mesh.young,
                                        mesh.poisson,mesh.thickness)
        K[ids[:,None],ids]+=ke
    constrained=torch.tensor(sorted(fixed),dtype=torch.long,device=loads.device)
    mask=torch.ones(ndof,dtype=torch.bool,device=loads.device); mask[constrained]=False
    free=torch.nonzero(mask).flatten(); q=torch.zeros_like(loads)
    q[free]=torch.linalg.solve(K[free[:,None],free],loads[free])
    return PinchedCylinderResult(n,float(q[probe]))


def solve_pinched_cylinder_nonlinear(n: int=3) -> PinchedCylinderResult:
    mesh,loads,fixed,probe=pinched_cylinder_model(n)
    result=solve_shell_step(mesh,loads,fixed,initial_increment=1.,maximum_increment=1.,
                            relative_tolerance=1e-8,absolute_tolerance=1e-4)
    if not result.converged: raise RuntimeError("pinched-cylinder nonlinear step did not converge")
    return PinchedCylinderResult(n,float(result.dofs[probe]))
