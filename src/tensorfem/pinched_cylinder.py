"""MacNeal--Harder pinched-cylinder benchmark for the cylindrical shell."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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


def pinched_cylinder_model(n: int, *, n_theta: int | None = None, axial_u_clamped: bool = False,
                           dtype=torch.float64, device=None):
    """Return the symmetry octant, load vector, constraints and probe DOF.

    The full cylinder has length 600 and a pair of diametrically opposed unit
    pinching loads at midspan. Reflection symmetry leaves x=[0,300] and
    theta=[0,pi/2]; the corner load is therefore one quarter of a unit.

    ``n`` is the number of divisions in the axial direction.  ``n_theta``
    (default ``n``) is the number of divisions in the circumferential
    direction, so an anisotropic ``n x n_theta`` mesh can be requested; the
    default reproduces the standard isotropic octant lattice.

    ``axial_u_clamped`` additionally restrains the axial displacement on the end
    ring.  A rigid diaphragm does not resist axial translation, so the benchmark
    model leaves ``u`` free; the flag exists to quantify that modelling choice
    (~16 % stiffening at the load point) in the sensitivity study.
    """
    nt=n if n_theta is None else n_theta
    if n<2 or nt<2: raise ValueError("n and n_theta must be at least two")
    dev=torch.device(device or "cpu"); R=300.
    xs=torch.linspace(0.,300.,n+1,dtype=dtype,device=dev)
    ts=torch.linspace(0.,math.pi/2,nt+1,dtype=dtype,device=dev)
    gx,gt=torch.meshgrid(xs,ts,indexing="ij")
    parameters=torch.stack((gx,gt),-1).reshape(-1,2)
    conn=[]
    for i in range(n):
        for j in range(nt):
            a=i*(nt+1)+j; conn.append((a,a+nt+1,a+nt+2,a+1))
    elements=torch.tensor(conn,dtype=torch.long,device=dev)
    mesh=CylindricalShellMesh(parameters,elements,R,3e6,.3,3.)
    loads=torch.zeros(6*len(parameters),dtype=dtype,device=dev); loads[2]=-.25
    fixed: dict[int,float]={}
    # Midspan reflection plane x=0: normal displacement and in-plane rotations.
    for j in range(nt+1):
        a=j
        for k in (0,4,5): fixed[6*a+k]=0.
    # Circumferential reflection planes y=0 and z=0.
    for i in range(n+1):
        a=i*(nt+1)
        for k in (1,3,5): fixed[6*a+k]=0.
        a=i*(nt+1)+nt
        for k in (2,3,4): fixed[6*a+k]=0.
    # Rigid end diaphragm: cross-section translations are restrained.
    for j in range(nt+1):
        a=n*(nt+1)+j; fixed[6*a+1]=0.; fixed[6*a+2]=0.
        if axial_u_clamped: fixed[6*a+0]=0.
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


def pinched_cylinder_sparse_system(n: int, *, n_theta: int | None = None,
                                   axial_u_clamped: bool = False,
                                   shear_factor: float = 5./6.,
                                   drilling_factor: float = 1e-8,
                                   dtype=torch.float64):
    """Return ``(K, loads, free, probe, mesh)`` with ``K`` a CSR sparse operator.

    Same element matrices as :func:`solve_pinched_cylinder_linear`; only the
    global storage changes.  Dense assembly costs ``O(N^4)`` memory in the mesh
    parameter ``N`` (346 MB at N=32, >5 GB at N=64), so the fine rungs of a
    convergence ladder are only reachable through this sparse path.  The two
    paths agree to solver round-off at the meshes both can reach.
    """
    mesh,loads,fixed,probe=pinched_cylinder_model(n,n_theta=n_theta,
                                                 axial_u_clamped=axial_u_clamped,dtype=dtype)
    ndof=loads.numel(); rows=[]; cols=[]; vals=[]
    for conn in mesh.elements:
        ids=torch.stack(tuple(6*conn+k for k in range(6)),1).reshape(-1).numpy()
        ke=cylindrical_shell4_stiffness(mesh.parameters[conn],mesh.radius,mesh.young,
                                        mesh.poisson,mesh.thickness,
                                        shear_factor=shear_factor,
                                        drilling_factor=drilling_factor)
        rows.append(np.repeat(ids,ids.size)); cols.append(np.tile(ids,ids.size))
        vals.append(ke.reshape(-1).numpy())
    K=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                    shape=(ndof,ndof)).tocsr()
    constrained=np.fromiter(sorted(fixed),dtype=np.int64,count=len(fixed))
    free=np.setdiff1d(np.arange(ndof),constrained)
    return K,loads.numpy().copy(),free,probe,mesh


def solve_pinched_cylinder_sparse(n: int, *, n_theta: int | None = None,
                                  axial_u_clamped: bool = False,
                                  shear_factor: float = 5./6.,
                                  drilling_factor: float = 1e-8,
                                  full: bool = False):
    """Sparse-LU solution of the octant model; ``full=True`` also returns DOFs."""
    K,loads,free,probe,mesh=pinched_cylinder_sparse_system(
        n,n_theta=n_theta,axial_u_clamped=axial_u_clamped,
        shear_factor=shear_factor,drilling_factor=drilling_factor)
    q=np.zeros_like(loads); q[free]=spla.spsolve(K[free][:,free].tocsc(),loads[free])
    result=PinchedCylinderResult(n,float(q[probe]))
    return (result,q,mesh) if full else result


def pinched_cylinder_radial_field(n: int, *, n_theta: int | None = None, **kwargs):
    """Radial displacement field ``w_r = v sin(theta) + w cos(theta)``.

    Returns ``(x, theta, radial, dofs)`` with the mesh node arrays; the
    rotation of the nodal frame with the circumference is why the radial
    component is a combination of the two global transverse DOFs.
    """
    result,q,mesh=solve_pinched_cylinder_sparse(n,n_theta=n_theta,full=True,**kwargs)
    theta=mesh.parameters[:,1].numpy(); x=mesh.parameters[:,0].numpy()
    idx=6*np.arange(len(x))
    radial=q[idx+1]*np.sin(theta)+q[idx+2]*np.cos(theta)
    return x,theta,radial,q


def solve_pinched_cylinder_nonlinear(n: int=3) -> PinchedCylinderResult:
    mesh,loads,fixed,probe=pinched_cylinder_model(n)
    result=solve_shell_step(mesh,loads,fixed,initial_increment=1.,maximum_increment=1.,
                            relative_tolerance=1e-8,absolute_tolerance=1e-4)
    if not result.converged: raise RuntimeError("pinched-cylinder nonlinear step did not converge")
    return PinchedCylinderResult(n,float(result.dofs[probe]))