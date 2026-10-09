"""Locking-free two-node Timoshenko beam bending."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TimoshenkoResult:
    displacement: torch.Tensor
    reaction: torch.Tensor

def timoshenko_stiffness(E: float, I: float, G: float, A: float, kappa: float,
                         length: float, *, dtype=torch.float64) -> torch.Tensor:
    """Exact-prismatic stiffness; shear-flexibility avoids equal-order locking."""
    if min(E, I, G, A, kappa, length) <= 0:
        raise ValueError("beam properties must be positive")
    phi = 12*E*I/(kappa*G*A*length**2)
    return E*I/(length**3*(1+phi))*torch.tensor([
        [12, 6*length, -12, 6*length],
        [6*length, (4+phi)*length**2, -6*length, (2-phi)*length**2],
        [-12, -6*length, 12, -6*length],
        [6*length, (2-phi)*length**2, -6*length, (4+phi)*length**2],
    ], dtype=dtype)

def solve_timoshenko_cantilever(length: float, elements: int, E: float, I: float,
                                G: float, A: float, load: float,
                                kappa: float=5/6) -> TimoshenkoResult:
    if elements < 1:
        raise ValueError("elements must be positive")
    ndof = 2*(elements+1)
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    ke = timoshenko_stiffness(E, I, G, A, kappa, length/elements)
    for e in range(elements):
        dofs = torch.tensor([2*e, 2*e+1, 2*e+2, 2*e+3])
        K[dofs[:, None], dofs] += ke
    f = torch.zeros(ndof, dtype=torch.float64); f[-2] = load
    free = torch.arange(2, ndof)
    u = torch.zeros_like(f); u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    return TimoshenkoResult(u, K@u-f)

def exact_tip_deflection(length: float, E: float, I: float, G: float, A: float,
                         load: float, kappa: float=5/6) -> float:
    return load*(length**3/(3*E*I) + length/(kappa*G*A))
