"""Benchmarks for the metric-consistent cylindrical Shell4 element."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .cylindrical_shell4 import cylindrical_shell4_stiffness


@dataclass(frozen=True)
class ShellBenchmarkResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    probe_dof: int
    @property
    def probe_displacement(self) -> float:
        return float(self.displacement[self.probe_dof])


def scordelis_lo_cylindrical(nx: int, nt: int, *, dtype=torch.float64, device=None):
    if nx < 1 or nt < 1 or nx % 2 or nt % 2:
        raise ValueError("nx and nt must be positive even integers")
    dev=torch.device(device or "cpu")
    L,R,angle=50.,25.,math.radians(40.); E,nu,t,p=4.32e8,0.,.25,90.
    xs=torch.linspace(-L/2,L/2,nx+1,dtype=dtype,device=dev)
    ts=torch.linspace(-angle,angle,nt+1,dtype=dtype,device=dev)
    gx,gt=torch.meshgrid(xs,ts,indexing="ij")
    nodes=torch.stack((gx,R*torch.sin(gt),R*torch.cos(gt)),-1).reshape(-1,3)
    param=torch.stack((gx,gt),-1).reshape(-1,2)
    conn=[]
    for i in range(nx):
        for j in range(nt):
            a=i*(nt+1)+j; conn.append((a,a+nt+1,a+nt+2,a+1))
    elements=torch.tensor(conn,dtype=torch.long,device=dev)
    ndof=6*len(nodes); K=torch.zeros((ndof,ndof),dtype=dtype,device=dev); f=torch.zeros(ndof,dtype=dtype,device=dev)
    for el in elements:
        ke=cylindrical_shell4_stiffness(param[el],R,E,nu,t)
        dofs=torch.stack(tuple(6*el+k for k in range(6)),1).reshape(-1)
        K[dofs[:,None],dofs]+=ke
        # Exact cylindrical surface area and vertical resultant integrated by
        # consistent bilinear nodal loads; pressure is global vertical.
        area=(param[el,0].max()-param[el,0].min())*R*(param[el,1].max()-param[el,1].min())
        f[6*el+2]-=p*area/4
    constrained=[]
    for i in (0,nx):
        row=torch.arange(i*(nt+1),(i+1)*(nt+1),device=dev)
        constrained.extend((6*row+1).tolist()); constrained.extend((6*row+2).tolist())
    constrained.append(0)
    fixed=torch.tensor(sorted(set(constrained)),dtype=torch.long,device=dev)
    mask=torch.ones(ndof,dtype=torch.bool,device=dev); mask[fixed]=False; free=torch.nonzero(mask).flatten()
    u=torch.zeros(ndof,dtype=dtype,device=dev); u[free]=torch.linalg.solve(K[free[:,None],free],f[free])
    return ShellBenchmarkResult(u,K@u-f,nodes,elements,6*((nx//2)*(nt+1)+nt)+2)
