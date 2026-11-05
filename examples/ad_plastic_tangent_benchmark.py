"""Measured AD versus Richardson tangent comparison at a non-coaxial state."""

import torch
from tensorfem.ad_plastic_tangent import benchmark_tangents
from tensorfem.adaptive_global_plasticity import AdaptiveGlobalState
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonproportional_plasticity import integrate_adaptive

D = torch.float64
x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=D)
m = FiniteStrainTet4Model(x, torch.tensor([[0, 1, 2, 3]]), 1000.0, 0.3, torch.tensor([0, 1, 2]))
s = AdaptiveGlobalState.virgin(m)
F1 = torch.tensor([[1.05, 0.18, 0.02], [0.01, 0.97, 0.04], [0.0, 0.02, 1.01]], dtype=D)
p = integrate_adaptive(F1, s.points[0], 1000.0, 0.3, 35.0, 60.0, rtol=3e-5).state
s = AdaptiveGlobalState((p,), s.cumulative_dissipation)
F2 = torch.tensor([[1.14, 0.25, 0.04], [0.03, 0.91, 0.08], [0.01, 0.04, 1.03]], dtype=D)
u = ((F2 - torch.eye(3, dtype=D)) @ x.T).T.reshape(-1)
print(benchmark_tangents(m, u, s, yield_stress=35.0, hardening=60.0, material_rtol=3e-5))
