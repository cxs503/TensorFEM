import torch
from tensorfem.finite_strain_plasticity import MultiplicativeJ2State, update_multiplicative_j2
from tensorfem.implicit_plastic_tangent import implicit_material_tangent

D = torch.float64
M = (1000.0, 0.3, 35.0, 60.0)
state = MultiplicativeJ2State(torch.eye(3, dtype=D), torch.zeros((), dtype=D))
F0 = torch.tensor([[1.04, 0.12, 0.01], [0.02, 0.98, 0.03], [0.0, 0.01, 1.0]], dtype=D)
_, state, _, _ = update_multiplicative_j2(F0, *M, state)
F = F0 + 0.1 * (
    torch.tensor([[1.08, 0.16, 0.02], [0.03, 0.95, 0.04], [0.01, 0.02, 1.01]], dtype=D) - F0
)
P, C, state, diss, seconds = implicit_material_tangent(F, *M, state)
print(
    {
        "seconds": seconds,
        "finite": bool(torch.isfinite(C).all()),
        "detFp": float(torch.linalg.det(state.plastic_gradient)),
        "dissipation": float(diss),
    }
)
