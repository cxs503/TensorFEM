"""Locking-controlled four-node flat shell.

The membrane shear term is evaluated at the element centre (selective reduced
integration), while extensional terms retain 2x2 integration.  This assumed
constant natural shear removes the dominant parasitic in-plane shear mode on
faceted cylindrical meshes. Plate transverse shear remains the SRI/MITC-like
field used by :mod:`tensorfem.plate`.
"""
from __future__ import annotations
import torch
from .shell4 import shell4_local_frame
from .plate import q4_mindlin_stiffness


def _membrane_sri(xy, E, nu, t):
    dtype, device = xy.dtype, xy.device
    D = E/(1-nu**2)*torch.tensor([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]],dtype=dtype,device=device)
    kn=torch.zeros((8,8),dtype=dtype,device=device)
    ks=torch.zeros_like(kn)
    g=3**-.5
    centre=None; centre_det=None
    for xi,eta in ((-g,-g),(g,-g),(g,g),(-g,g),(0.,0.)):
        nat=xy.new_tensor([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],[(1+eta),(1+xi)],[-(1+eta),(1-xi)]])/4
        J=xy.T@nat; det=torch.linalg.det(J); deriv=nat@torch.linalg.inv(J)
        B=torch.zeros((3,8),dtype=dtype,device=device)
        B[0,0::2]=deriv[:,0]; B[1,1::2]=deriv[:,1]
        B[2,0::2]=deriv[:,1]; B[2,1::2]=deriv[:,0]
        if xi==0: centre,centre_det=B,det
        else: kn += B[:2].T@D[:2,:2]@B[:2]*det*t
    # Weight four at centre for in-plane engineering shear.
    ks=centre[2:3].T*D[2,2]@centre[2:3] * centre_det*t*4
    return kn+ks


def shell4_mitc_stiffness(xyz, young, poisson, thickness, *, drilling_factor=1e-6):
    xy,basis=shell4_local_frame(xyz); dtype,device=xyz.dtype,xyz.device
    E=torch.as_tensor(young,dtype=dtype,device=device); nu=torch.as_tensor(poisson,dtype=dtype,device=device); t=torch.as_tensor(thickness,dtype=dtype,device=device)
    km=_membrane_sri(xy,E,nu,t); kp=q4_mindlin_stiffness(xy,E,nu,t)
    kl=torch.zeros((24,24),dtype=dtype,device=device); ids=torch.arange(4,device=device)
    membrane=torch.stack((6*ids,6*ids+1),1).reshape(-1)
    plate=torch.stack((6*ids+2,6*ids+4,6*ids+3),1).reshape(-1); sign=xyz.new_tensor([1.,1.,-1.]).repeat(4)
    kl[membrane[:,None],membrane]+=km; kl[plate[:,None],plate]+=sign[:,None]*kp*sign[None,:]
    area=.5*abs(torch.linalg.det(torch.stack((xy[1]-xy[0],xy[3]-xy[0])))); kd=drilling_factor*E*t*area
    rz=6*ids+5; lap=xyz.new_tensor([[2.,-1.,0.,-1.],[-1.,2.,-1.,0.],[0.,-1.,2.,-1.],[-1.,0.,-1.,2.]])
    kl[rz[:,None],rz]+=kd*lap
    T=torch.zeros((24,24),dtype=dtype,device=device)
    for n in range(4): T[6*n:6*n+3,6*n:6*n+3]=basis; T[6*n+3:6*n+6,6*n+3:6*n+6]=basis
    return T.T@kl@T
