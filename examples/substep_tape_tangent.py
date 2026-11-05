import time
import torch
from tensorfem.nonproportional_plasticity import (
    virgin_adaptive_state,
    integrate_adaptive,
    richardson_algorithmic_tangent,
)
from tensorfem.substep_tape_tangent import build_tape, taped_material_tangent

D = torch.float64
M = (1000.0, 0.3, 35.0, 60.0)
s = virgin_adaptive_state()
F0 = torch.tensor([[1.05, 0.18, 0.02], [0.01, 0.97, 0.04], [0.0, 0.02, 1.01]], dtype=D)
s = integrate_adaptive(F0, s, *M, rtol=3e-5).state
F = torch.tensor([[1.14, 0.25, 0.04], [0.03, 0.91, 0.08], [0.01, 0.04, 1.03]], dtype=D)
tape = build_tape(F, s, *M, rtol=3e-5)
start = time.perf_counter()
result = taped_material_tangent(F, s, tape, *M)
taped = time.perf_counter() - start
start = time.perf_counter()
richardson_algorithmic_tangent(F, s, *M, rtol=3e-5)
oracle = time.perf_counter() - start
print(
    {
        "accepted_updates": len(tape.fractions),
        "taped_seconds": taped,
        "richardson_seconds": oracle,
        "speedup": oracle / taped,
        "detFp": float(torch.linalg.det(result[2].plastic.plastic_gradient)),
    }
)
