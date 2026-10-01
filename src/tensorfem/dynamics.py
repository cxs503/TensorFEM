"""Average-acceleration Newmark integration for linear dynamics."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TransientResult:
    time: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor

def newmark_linear(M: torch.Tensor, K: torch.Tensor, force: torch.Tensor,
                   time: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor,
                   C: torch.Tensor|None=None, beta: float=.25,
                   gamma: float=.5) -> TransientResult:
    if M.ndim != 2 or M.shape != K.shape or M.shape[0] != M.shape[1]:
        raise ValueError("M and K must be equal square matrices")
    if force.shape != (time.numel(), M.shape[0]) or time.numel() < 2:
        raise ValueError("incompatible time or force history")
    dtv = time[1:]-time[:-1]
    if torch.any(dtv <= 0) or not torch.allclose(dtv, dtv[0]):
        raise ValueError("time must be uniformly increasing")
    C = torch.zeros_like(M) if C is None else C
    if C.shape != M.shape or u0.shape != (M.shape[0],) or v0.shape != u0.shape:
        raise ValueError("incompatible damping or initial condition")
    u0 = u0.to(dtype=M.dtype, device=M.device)
    v0 = v0.to(dtype=M.dtype, device=M.device)
    dt=dtv[0]; n=time.numel(); ndof=M.shape[0]
    u=torch.zeros((n,ndof),dtype=M.dtype); v=torch.zeros_like(u); a=torch.zeros_like(u)
    u[0]=u0; v[0]=v0; a[0]=torch.linalg.solve(M,force[0]-C@v0-K@u0)
    c0=1/(beta*dt**2); c1=gamma/(beta*dt); c2=1/(beta*dt)
    c3=1/(2*beta)-1; c4=gamma/beta-1; c5=dt*(gamma/(2*beta)-1)
    Ke=K+c0*M+c1*C
    for i in range(n-1):
        rhs=force[i+1]+M@(c0*u[i]+c2*v[i]+c3*a[i])+C@(c1*u[i]+c4*v[i]+c5*a[i])
        u[i+1]=torch.linalg.solve(Ke,rhs)
        a[i+1]=c0*(u[i+1]-u[i])-c2*v[i]-c3*a[i]
        v[i+1]=v[i]+dt*((1-gamma)*a[i]+gamma*a[i+1])
    return TransientResult(time,u,v,a)

def mechanical_energy(M: torch.Tensor, K: torch.Tensor,
                      u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return .5*torch.einsum("ti,ij,tj->t",v,M,v)+.5*torch.einsum("ti,ij,tj->t",u,K,u)
