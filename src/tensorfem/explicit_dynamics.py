"""Conditionally stable central-difference dynamics with fail-closed checks."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ExplicitResult:
    time: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    kinetic_energy: torch.Tensor
    strain_energy: torch.Tensor
    critical_time_step: float


def critical_time_step(mass: torch.Tensor, stiffness: torch.Tensor) -> float:
    """Return ``2/omega_max`` for a diagonal lumped mass matrix."""
    if mass.ndim != 2 or mass.shape != stiffness.shape or mass.shape[0] != mass.shape[1]:
        raise ValueError("mass and stiffness must be equal square matrices")
    diagonal=torch.diagonal(mass)
    if torch.any(diagonal <= 0) or not torch.allclose(mass,torch.diag(diagonal)):
        raise ValueError("central difference requires a positive diagonal lumped mass")
    scaled=stiffness/torch.sqrt(diagonal[:,None]*diagonal[None,:])
    largest=torch.linalg.eigvalsh(scaled)[-1]
    if largest <= 0: raise ValueError("stiffness must have a positive vibration mode")
    return float(2/torch.sqrt(largest))


def central_difference(mass: torch.Tensor, stiffness: torch.Tensor,
                       force: torch.Tensor, time: torch.Tensor,
                       initial_displacement: torch.Tensor,
                       initial_velocity: torch.Tensor,
                       *, stability_safety: float=1.0) -> ExplicitResult:
    """Integrate undamped linear dynamics and reject unstable time steps."""
    nd=mass.shape[0]
    if force.shape != (time.numel(),nd) or initial_displacement.shape != (nd,) or initial_velocity.shape != (nd,):
        raise ValueError("incompatible force or initial-condition shape")
    dtv=time[1:]-time[:-1]
    if time.numel() < 2 or torch.any(dtv <= 0) or not torch.allclose(dtv,dtv[0]):
        raise ValueError("time must be uniformly increasing")
    if not 0 < stability_safety <= 1: raise ValueError("stability_safety must be in (0,1]")
    limit=critical_time_step(mass,stiffness)
    dt=float(dtv[0])
    if dt > stability_safety*limit*(1+1e-12):
        raise ValueError(f"time step {dt:g} exceeds stable limit {stability_safety*limit:g}")
    dtype=mass.dtype; device=mass.device
    u=torch.zeros((len(time),nd),dtype=dtype,device=device)
    v=torch.zeros_like(u); a=torch.zeros_like(u)
    u[0]=initial_displacement.to(dtype=dtype,device=device)
    v[0]=initial_velocity.to(dtype=dtype,device=device)
    invm=1/torch.diagonal(mass)
    a[0]=invm*(force[0]-stiffness@u[0])
    if len(time)>1: u[1]=u[0]+dt*v[0]+.5*dt**2*a[0]
    for i in range(1,len(time)-1):
        a[i]=invm*(force[i]-stiffness@u[i])
        u[i+1]=2*u[i]-u[i-1]+dt**2*a[i]
        v[i]=(u[i+1]-u[i-1])/(2*dt)
    if len(time)>1:
        a[-1]=invm*(force[-1]-stiffness@u[-1])
        v[-1]=(u[-1]-u[-2])/dt+.5*dt*a[-1]
    kinetic=.5*torch.einsum("ti,ij,tj->t",v,mass,v)
    strain=.5*torch.einsum("ti,ij,tj->t",u,stiffness,u)
    return ExplicitResult(time,u,v,a,kinetic,strain,limit)
