"""Projected MITC-like Shell4 and the 18-degree-hole hemisphere benchmark."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .shell4_mitc import _membrane_sri
from .plate import q4_mindlin_stiffness


def projected_shell4_stiffness(xyz, young, poisson, thickness, *, drilling_factor=1e-6):
    """Flat-projected facet for mildly warped quads on a smooth surface."""
    if xyz.shape!=(4,3): raise ValueError("xyz must have shape (4,3)")
    dtype,device=xyz.dtype,xyz.device
    E=torch.as_tensor(young,dtype=dtype,device=device); nu=torch.as_tensor(poisson,dtype=dtype,device=device); t=torch.as_tensor(thickness,dtype=dtype,device=device)
    if bool((E<=0).item()) or bool((t<=0).item()) or drilling_factor<0:
        raise ValueError("young/thickness must be positive and drilling factor nonnegative")
    # Diagonal tangents give an unbiased centre normal for a warped quad.
    a=xyz[2]-xyz[0]; b=xyz[3]-xyz[1]
    normal=torch.linalg.cross(a,b,dim=0); norm=torch.linalg.vector_norm(normal)
    if bool((norm<=torch.finfo(dtype).eps).item()): raise ValueError("degenerate projected shell")
    e3=normal/norm; raw=xyz[1]-xyz[0]; e1=raw-torch.dot(raw,e3)*e3
    e1=e1/torch.linalg.vector_norm(e1); e2=torch.linalg.cross(e3,e1,dim=0); basis=torch.stack((e1,e2,e3))
    xy=(xyz-xyz.mean(0))@basis[:2].T
    km=_membrane_sri(xy,E,nu,t); kp=q4_mindlin_stiffness(xy,E,nu,t)
    kl=torch.zeros((24,24),dtype=dtype,device=device); ids=torch.arange(4,device=device)
    membrane=torch.stack((6*ids,6*ids+1),1).reshape(-1)
    plate=torch.stack((6*ids+2,6*ids+4,6*ids+3),1).reshape(-1); sign=xyz.new_tensor([1.,1.,-1.]).repeat(4)
    kl[membrane[:,None],membrane]+=km; kl[plate[:,None],plate]+=sign[:,None]*kp*sign[None,:]
    area=.5*abs(torch.linalg.det(torch.stack((xy[1]-xy[0],xy[3]-xy[0])))); rz=6*ids+5
    lap=xyz.new_tensor([[2.,-1.,0.,-1.],[-1.,2.,-1.,0.],[0.,-1.,2.,-1.],[-1.,0.,-1.,2.]])
    kl[rz[:,None],rz]+=drilling_factor*E*t*area*lap
    T=torch.zeros((24,24),dtype=dtype,device=device)
    for n in range(4): T[6*n:6*n+3,6*n:6*n+3]=basis; T[6*n+3:6*n+6,6*n+3:6*n+6]=basis
    return T.T@kl@T


@dataclass(frozen=True)
class HemisphereResult:
    meridional_elements: int
    circumferential_elements: int
    displacement: float
    solution: torch.Tensor
    reaction: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    reference: float=.0924
    @property
    def relative_error(self): return abs(abs(self.displacement)/self.reference-1.)


def hemisphere_with_hole(nphi: int, ntheta: int | None=None, *, drilling_factor=1e-6):
    """Solve a quarter of the MacNeal--Harder 18-degree-hole hemisphere."""
    if ntheta is None: ntheta=nphi
    if nphi<2 or ntheta<2: raise ValueError("mesh counts must be at least two")
    D=torch.float64; R=10.; E=6.825e7; nu=.3; t=.04
    phis=torch.linspace(math.radians(18),math.pi/2,nphi+1,dtype=D)
    thetas=torch.linspace(0,math.pi/2,ntheta+1,dtype=D)
    gp,gt=torch.meshgrid(phis,thetas,indexing="ij")
    nodes=R*torch.stack((torch.sin(gp)*torch.cos(gt),torch.sin(gp)*torch.sin(gt),torch.cos(gp)),-1).reshape(-1,3)
    conn=[]
    for i in range(nphi):
        for j in range(ntheta):
            a=i*(ntheta+1)+j;conn.append((a,a+ntheta+1,a+ntheta+2,a+1))
    elements=torch.tensor(conn,dtype=torch.long); ndof=6*len(nodes)
    K=torch.zeros((ndof,ndof),dtype=D); f=torch.zeros(ndof,dtype=D)
    for el in elements:
        ids=torch.stack(tuple(6*el+k for k in range(6)),1).reshape(-1)
        K[ids[:,None],ids]+=projected_shell4_stiffness(nodes[el],E,nu,t,drilling_factor=drilling_factor)
    # Classical quarter-model point loads P=1 at the two equator corners.
    a=nphi*(ntheta+1); b=a+ntheta
    f[6*a]+=1.; f[6*b+1]-=1.
    fixed=[]
    for i in range(nphi+1):
        a=i*(ntheta+1); fixed += [6*a+1,6*a+3,6*a+5]
        b=a+ntheta; fixed += [6*b+0,6*b+4,6*b+5]
    fixed.append(2)  # gauge the otherwise free vertical rigid translation
    fixed=torch.tensor(sorted(set(fixed)),dtype=torch.long); mask=torch.ones(ndof,dtype=torch.bool);mask[fixed]=False;free=torch.nonzero(mask).flatten()
    q=torch.zeros(ndof,dtype=D);q[free]=torch.linalg.solve(K[free[:,None],free],f[free])
    return HemisphereResult(nphi,ntheta,float(q[6*(nphi*(ntheta+1))]),q,K@q-f,nodes,elements)
