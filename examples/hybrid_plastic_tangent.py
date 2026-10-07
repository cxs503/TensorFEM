"""Show safe per-element tangent selection for small and large increments."""
import torch
from tensorfem.hybrid_plastic_tangent import assemble_hybrid
from tensorfem.adaptive_global_plasticity import AdaptiveGlobalState
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonproportional_plasticity import integrate_adaptive
D=torch.float64;x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
m=FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3]]),1000.,.3,torch.tensor([0,1,2]));s=AdaptiveGlobalState.virgin(m)
F0=torch.tensor([[1.04,.12,.01],[.02,.98,.03],[0.,.01,1.]],dtype=D);p=integrate_adaptive(F0,s.points[0],1000.,.3,35.,60.,rtol=3e-5).state;s=AdaptiveGlobalState((p,),s.cumulative_dissipation)
Ft=torch.tensor([[1.08,.16,.02],[.03,.95,.04],[.01,.02,1.01]],dtype=D)
for scale in (.05,1.):
 F=F0+scale*(Ft-F0);u=((F-torch.eye(3,dtype=D))@x.T).T.reshape(-1)
 *_,metrics=assemble_hybrid(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5)
 print(scale,metrics)
