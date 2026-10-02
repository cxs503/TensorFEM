"""Axisymmetric deformable-half-space Hertz contact experiment.

The elastic body is discretised by axisymmetric Q4 elements.  A rigid
parabolic sphere is enforced by nodal penalty contact on the top boundary.
This is a compact axisymmetric verification model, not general 3-D contact.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class AxisymmetricHertzResult:
    displacement: torch.Tensor
    radii: torch.Tensor
    pressure: torch.Tensor
    load: torch.Tensor
    contact_radius: torch.Tensor
    peak_pressure: torch.Tensor
    active_extent: torch.Tensor
    raw_peak_pressure: torch.Tensor
    iterations: int
    converged: bool


def hertz_reference(indentation: float, radius: float, young: float, poisson: float):
    """Rigid sphere/elastic half-space small-contact Hertz solution."""
    if min(indentation,radius,young)<=0 or not (-1<poisson<.5):
        raise ValueError("invalid Hertz parameters")
    effective=young/(1-poisson**2)
    a=math.sqrt(radius*indentation)
    load=4*effective*a**3/(3*radius)
    peak=3*load/(2*math.pi*a*a)
    return load,a,peak


def _constitutive(E,nu,dtype,device):
    c=E/((1+nu)*(1-2*nu))
    return c*torch.tensor([[1-nu,nu,nu,0.],[nu,1-nu,nu,0.],
                           [nu,nu,1-nu,0.],[0.,0.,0.,(1-2*nu)/2]],dtype=dtype,device=device)


def axisymmetric_mesh(nr: int,nz: int,width: float,depth: float,*,grading: float=1.,dtype=torch.float64,device=None):
    """Structured `(r,z)` mesh, top at z=0 and bottom at -depth."""
    if nr<2 or nz<2 or width<=0 or depth<=0: raise ValueError("invalid mesh")
    if grading < 1: raise ValueError("grading must be at least one")
    unit=torch.linspace(0.,1.,nr+1,dtype=dtype,device=device)
    r=width*unit**grading
    unit_z=torch.linspace(0.,1.,nz+1,dtype=dtype,device=device)
    z=-depth*unit_z**grading
    nodes=torch.stack(torch.meshgrid(r,z,indexing="ij"),dim=-1).reshape(-1,2)
    elems=[]
    def node(i,j): return i*(nz+1)+j
    for i in range(nr):
        for j in range(nz):
            elems.append([node(i,j+1),node(i+1,j+1),node(i+1,j),node(i,j)])
    return nodes,torch.tensor(elems,dtype=torch.long,device=device)


def assemble_axisymmetric_stiffness(nodes,elements,young,poisson):
    """Dense reference assembly for axisymmetric Q4 elasticity."""
    D=_constitutive(young,poisson,nodes.dtype,nodes.device)
    K=nodes.new_zeros((2*len(nodes),2*len(nodes)))
    g=1/math.sqrt(3)
    for conn in elements:
        x=nodes[conn]; ke=nodes.new_zeros((8,8))
        for xi in (-g,g):
            for eta in (-g,g):
                N=nodes.new_tensor([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])/4
                dnat=nodes.new_tensor([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],
                    [(1+eta),(1+xi)],[-(1+eta),(1-xi)]])/4
                J=dnat.T@x; det=torch.linalg.det(J)
                if bool(det<=0): raise ValueError("non-positive element Jacobian")
                dN=dnat@torch.linalg.inv(J); radius=N@x[:,0]
                B=nodes.new_zeros((4,8))
                B[0,0::2]=dN[:,0]; B[1,1::2]=dN[:,1]
                B[2,0::2]=N/radius
                B[3,0::2]=dN[:,1]; B[3,1::2]=dN[:,0]
                ke += B.T@D@B*(2*math.pi*radius*det)
        dof=torch.stack((2*conn,2*conn+1),dim=1).flatten()
        K[dof[:,None],dof[None,:]] += ke
    return K


def _ring_areas(r):
    edges=torch.empty(len(r)+1,dtype=r.dtype,device=r.device)
    edges[0]=0.; edges[-1]=r[-1]
    edges[1:-1]=(r[:-1]+r[1:])/2
    return math.pi*(edges[1:]**2-edges[:-1]**2)


def solve_axisymmetric_hertz(*,nr=40,nz=40,width=12.,depth=12.,radius=10.,
                             indentation=.025,young=1e5,poisson=.3,
                             penalty_factor=28.,tolerance=1e-10,max_iterations=30,
                             grading=1.8,dtype=torch.float64,device=None):
    """Solve prescribed-indentation rigid-sphere contact by active-set Newton."""
    if min(radius,indentation,young,penalty_factor,tolerance)>0 and max_iterations>0 and -1<poisson<.5:
        pass
    else:
        raise ValueError("invalid material, contact, or iteration parameter")
    nodes,elements=axisymmetric_mesh(nr,nz,width,depth,grading=grading,dtype=dtype,device=device)
    K=assemble_axisymmetric_stiffness(nodes,elements,young,poisson)
    fixed=[]
    for i,(r,z) in enumerate(nodes):
        if bool(torch.abs(r)<1e-14): fixed.append(2*i)
        if bool(torch.abs(z+depth)<1e-14): fixed.extend((2*i,2*i+1))
    # Far radial boundary is a symmetry-free truncation: constrain radial DOF.
    for i,(r,z) in enumerate(nodes):
        if bool(torch.abs(r-width)<1e-14): fixed.append(2*i)
    fixed=torch.tensor(sorted(set(fixed)),dtype=torch.long,device=device)
    mask=torch.ones(2*len(nodes),dtype=torch.bool,device=device); mask[fixed]=False
    free=torch.nonzero(mask,as_tuple=False).flatten()
    top=torch.tensor([i*(nz+1) for i in range(nr+1)],dtype=torch.long,device=device)
    radii=nodes[top,0]; area=_ring_areas(radii)
    characteristic=math.sqrt(radius*indentation)
    penalty=penalty_factor*young/characteristic
    obstacle=radii*radii/(2*radius)-indentation
    u=nodes.new_zeros(2*len(nodes)); converged=False
    for iteration in range(1,max_iterations+1):
        uz=u[2*top+1]; penetration=torch.clamp(uz-obstacle,min=0.)
        pressure=penalty*penetration
        residual=-(K@u); residual[2*top+1] -= pressure*area
        tangent=K.clone(); active=penetration>0
        ids=2*top[active]+1
        tangent[ids,ids] += penalty*area[active]
        rf=residual[free]
        scale=max(float(torch.linalg.vector_norm(pressure*area)),1.)
        if float(torch.linalg.vector_norm(rf)) <= tolerance*scale:
            converged=True; break
        du=torch.linalg.solve(tangent[free][:,free],rf)
        u[free] += du
    uz=u[2*top+1]; penetration=torch.clamp(uz-obstacle,min=0.); pressure=penalty*penetration
    load=torch.sum(pressure*area); active=pressure>max(float(pressure.max())*1e-8,torch.finfo(dtype).eps)
    if bool(torch.any(active)):
        # Edge inferred by linear pressure extrapolation of the last active pair.
        ia=int(torch.nonzero(active,as_tuple=False)[-1]); a=radii[ia]
        if ia+1<len(radii) and ia>0:
            a=radii[ia]+(radii[ia]-radii[ia-1])*pressure[ia]/(pressure[ia-1]-pressure[ia])
    else: a=nodes.new_zeros(())
    # Hertz fields obey p^2 = p0^2 - (p0^2/a^2) r^2.  A least-squares
    # recovery avoids identifying the contact edge/peak from a single lumped
    # boundary node while still using only the computed pressure field.
    if int(active.sum()) >= 3:
        rr=radii[active]; pp=pressure[active]
        design=torch.stack((torch.ones_like(rr),rr*rr),dim=1)
        coeff=torch.linalg.lstsq(design,pp*pp).solution
        fitted_peak=torch.sqrt(torch.clamp(coeff[0],min=0.))
        fitted_radius=torch.sqrt(torch.clamp(-coeff[0]/coeff[1],min=0.))
    else:
        fitted_peak=pressure.max(); fitted_radius=a
    return AxisymmetricHertzResult(u,radii,pressure,load,fitted_radius,fitted_peak,
                                   a,pressure.max(),iteration,converged)
