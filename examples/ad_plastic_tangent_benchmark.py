"""Measured AD versus Richardson tangent comparison at a non-coaxial state."""
import torch
from tensorfem.ad_plastic_tangent import benchmark_tangents
from tensorfem.adaptive_global_plasticity import AdaptiveGlobalState
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonproportional_plasticity import integrate_adaptive
D=torch.float64
x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
m=FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3]]),1000.,.3,torch.tensor([0,1,2]))
s=AdaptiveGlobalState.virgin(m)
F1=torch.tensor([[1.05,.18,.02],[.01,.97,.04],[0.,.02,1.01]],dtype=D)
p=integrate_adaptive(F1,s.points[0],1000.,.3,35.,60.,rtol=3e-5).state
s=AdaptiveGlobalState((p,),s.cumulative_dissipation)
F2=torch.tensor([[1.14,.25,.04],[.03,.91,.08],[.01,.04,1.03]],dtype=D)
u=((F2-torch.eye(3,dtype=D))@x.T).T.reshape(-1)
print(benchmark_tangents(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5))
