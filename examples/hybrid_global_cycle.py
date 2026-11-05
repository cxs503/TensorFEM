import torch
from tensorfem.hybrid_global_solver import solve_hybrid_global_path
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model

D = torch.float64
x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=D)
m = FiniteStrainTet4Model(
    x,
    torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    1000.0,
    0.3,
    torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 11]),
)
f = torch.zeros(12, dtype=D)
f[3] = 32.0
f[7] = 1.0
for r in solve_hybrid_global_path(m, f, (0.3, -0.1, 0.0), yield_stress=35.0, hardening=60.0):
    print(
        {
            "factor": r.load_factor,
            "paths": r.material_paths,
            "strategies": r.strategy_counts,
            "fallbacks": r.fallback_counts,
            "calls": r.assembly_calls,
        }
    )
